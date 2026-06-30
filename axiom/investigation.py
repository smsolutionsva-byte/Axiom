from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .citation import citation_token, extract_citations
from .embeddings import tokenize
from .retrieval import SearchHit, search
from .workstation import find_files


NAME_RE = re.compile(r"\b[A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,3}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,32}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>)\"]+")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?)?\b")

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".ts", ".tsx", ".html", ".css"}
WEAK_WORDS = {"the", "and", "for", "with", "from", "this", "that", "guy", "person", "man", "woman", "find", "search"}
GENERIC_NAME_PARTS = {"The", "At", "Transcript", "Report", "Text", "Image", "Audio"}


@dataclass(frozen=True)
class FileLead:
    path: str
    name: str
    reason: str
    score: float


def investigate_subject(
    conn: sqlite3.Connection,
    subject: str,
    *,
    roots: list[str] | None = None,
    top_k: int = 8,
    max_file_results: int = 40,
) -> dict[str, object]:
    cleaned = subject.strip()
    if not cleaned:
        return empty_investigation("No subject supplied.")

    query = build_query(cleaned)
    query_id, hits = search(conn, query, top_k=top_k)
    exact_hits = exact_evidence_hits(conn, cleaned, limit=top_k)
    merged_hits = merge_hits(hits, exact_hits)
    file_leads = filesystem_leads(roots or [], cleaned, max_results=max_file_results)
    entities = extract_entity_pack(cleaned, merged_hits, file_leads)
    timeline = investigation_timeline(merged_hits)
    confidence = investigation_confidence(merged_hits, file_leads, entities)
    guard = passive_guard_for_hits(cleaned, merged_hits)

    return {
        "subject": cleaned,
        "query_id": query_id,
        "confidence": confidence,
        "summary": investigation_summary(cleaned, merged_hits, file_leads, confidence),
        "entities": entities,
        "evidence": [hit_payload(hit) for hit in merged_hits],
        "file_leads": [asdict(item) for item in file_leads],
        "timeline": timeline,
        "risk_flags": risk_flags(merged_hits),
        "next_actions": next_actions(cleaned, merged_hits, file_leads, roots or []),
        "hallucination_guard": guard,
    }


def empty_investigation(reason: str) -> dict[str, object]:
    return {
        "subject": "",
        "query_id": "",
        "confidence": 0.0,
        "summary": reason,
        "entities": {},
        "evidence": [],
        "file_leads": [],
        "timeline": [],
        "risk_flags": [],
        "next_actions": ["Provide a name, handle, email, phone, or identifier to investigate."],
        "hallucination_guard": {"status": "blocked", "reason": reason},
    }


def build_query(subject: str) -> str:
    tokens = [token for token in tokenize(subject) if token not in WEAK_WORDS]
    aliases = alias_candidates(subject)
    return " ".join(dict.fromkeys(tokens + aliases))


def alias_candidates(subject: str) -> list[str]:
    aliases = [subject.strip()]
    parts = [part for part in re.split(r"\s+", subject.strip()) if part]
    if len(parts) >= 2:
        aliases.append(parts[0])
        aliases.append(parts[-1])
        aliases.append(" ".join(part[0] for part in parts if part).upper())
    if subject.startswith("@"):
        aliases.append(subject[1:])
    return [alias.lower() for alias in aliases if alias]


def exact_evidence_hits(conn: sqlite3.Connection, subject: str, *, limit: int) -> list[SearchHit]:
    from .retrieval import make_hit

    patterns = [subject, *alias_candidates(subject)]
    seen: set[str] = set()
    hits: list[SearchHit] = []
    for pattern in patterns:
        if not pattern.strip():
            continue
        rows = conn.execute(
            """
            SELECT chunk_id
            FROM content_chunks
            WHERE chunk_kind = 'child' AND LOWER(text_content) LIKE ?
            LIMIT ?
            """,
            (f"%{pattern.lower()}%", limit),
        ).fetchall()
        for row in rows:
            if row["chunk_id"] in seen:
                continue
            hit = make_hit(conn, row["chunk_id"], 1.0, subject)
            if hit is not None:
                hits.append(hit)
                seen.add(row["chunk_id"])
    return hits[:limit]


def merge_hits(primary: list[SearchHit], secondary: list[SearchHit]) -> list[SearchHit]:
    merged: dict[str, SearchHit] = {}
    for hit in secondary + primary:
        merged[hit.chunk_id] = hit
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)


def filesystem_leads(roots: list[str], subject: str, *, max_results: int) -> list[FileLead]:
    leads: list[FileLead] = []
    if not roots:
        return leads
    aliases = alias_candidates(subject)
    for root in roots:
        for alias in aliases[:5]:
            try:
                matches = find_files(root, alias, content=True, max_results=max_results, extensions=TEXT_EXTENSIONS)
            except (OSError, FileNotFoundError):
                continue
            for item in matches:
                reason = "content or filename match"
                score = 0.7 if alias in item.name.lower() else 0.55
                leads.append(FileLead(item.path, item.name, reason, score))
                if len(leads) >= max_results:
                    return dedupe_file_leads(leads)
    return dedupe_file_leads(leads)


def dedupe_file_leads(leads: list[FileLead]) -> list[FileLead]:
    best: dict[str, FileLead] = {}
    for lead in leads:
        current = best.get(lead.path)
        if current is None or lead.score > current.score:
            best[lead.path] = lead
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def extract_entity_pack(subject: str, hits: list[SearchHit], file_leads: list[FileLead]) -> dict[str, object]:
    text = "\n".join(hit.text for hit in hits)
    normalized_text = " ".join(text.split())
    return {
        "aliases": alias_candidates(subject),
        "names": ranked(clean_name(name) for name in NAME_RE.findall(normalized_text)),
        "emails": ranked(EMAIL_RE.findall(text)),
        "phones": ranked(clean_phone(phone) for phone in PHONE_RE.findall(text)),
        "handles": ranked(HANDLE_RE.findall(text)),
        "urls": ranked(URL_RE.findall(text)),
        "dates": ranked(DATE_RE.findall(text)),
        "file_names": [lead.name for lead in file_leads[:12]],
        "keywords": ranked_keywords(text),
    }


def ranked(items) -> list[str]:
    return [item for item, _count in Counter(item for item in items if item).most_common(20)]


def clean_name(name: str) -> str:
    cleaned = " ".join(name.split())
    parts = cleaned.split()
    if any(part in GENERIC_NAME_PARTS for part in parts):
        return ""
    if len(parts) > 3:
        return ""
    return cleaned


def ranked_keywords(text: str) -> list[str]:
    tokens = [token for token in tokenize(text) if len(token) > 4 and token not in WEAK_WORDS]
    return [token for token, _count in Counter(tokens).most_common(18)]


def clean_phone(phone: str) -> str:
    return re.sub(r"\s+", " ", phone).strip()


def investigation_timeline(hits: list[SearchHit]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for hit in hits:
        for date in DATE_RE.findall(hit.text):
            items.append(
                {
                    "when": date,
                    "source": hit.file_name,
                    "location": hit.location,
                    "citation": citation_token(hit.chunk_id),
                    "summary": compact(hit.snippet, 220),
                }
            )
    return sorted(items, key=lambda item: (str(item["when"]), str(item["source"])))[:40]


def investigation_confidence(hits: list[SearchHit], file_leads: list[FileLead], entities: dict[str, object]) -> float:
    score = 0.1
    score += min(len(hits), 8) * 0.07
    score += min(len(set(hit.file_name for hit in hits)), 5) * 0.05
    score += min(len(file_leads), 8) * 0.025
    if entities.get("emails") or entities.get("phones") or entities.get("handles"):
        score += 0.12
    if entities.get("dates"):
        score += 0.06
    return round(max(0.0, min(score, 0.92)), 2)


def investigation_summary(subject: str, hits: list[SearchHit], file_leads: list[FileLead], confidence: float) -> str:
    if not hits and not file_leads:
        return f"No indexed or local file evidence was found for '{subject}'. This is a negative search result, not proof the subject does not exist."
    if confidence >= 0.65:
        return f"Strong local evidence cluster found for '{subject}'. Review cited chunks before drawing conclusions."
    if hits:
        return f"Partial evidence found for '{subject}'. The dossier needs corroboration from more sources."
    return f"Only file leads were found for '{subject}'. Ingest the matching files to build cited evidence."


def risk_flags(hits: list[SearchHit]) -> list[str]:
    text = "\n".join(hit.text.lower() for hit in hits)
    flags: list[str] = []
    for label, words in {
        "identity-link": ["email", "phone", "contact", "handle", "account"],
        "time-sensitive": ["urgent", "today", "deadline", "warning"],
        "operational": ["operation", "framework", "field", "plan"],
        "contradiction-check": ["however", "but", "disputed", "denied", "inconsistent"],
    }.items():
        if any(word in text for word in words):
            flags.append(label)
    return flags


def next_actions(subject: str, hits: list[SearchHit], file_leads: list[FileLead], roots: list[str]) -> list[str]:
    actions: list[str] = []
    if hits:
        actions.append(f"Inspect the strongest citation: python -m axiom sources inspect {hits[0].chunk_id}")
    if file_leads:
        actions.append("Ingest the highest-scoring file leads so they become citation-backed evidence.")
    if not roots:
        actions.append("Add one or more local folder roots for a wider offline sweep.")
    actions.append(f"Run a focused follow-up query for aliases: {', '.join(alias_candidates(subject)[:4])}")
    actions.append("Capture screenshots or browser-tab text if the subject is visible on screen but not indexed.")
    return actions[:6]


def passive_guard_for_hits(subject: str, hits: list[SearchHit]) -> dict[str, object]:
    evidence_text = "\n".join(hit.text for hit in hits).lower()
    aliases = alias_candidates(subject)
    alias_hits = [alias for alias in aliases if alias and alias in evidence_text]
    status = "supported" if alias_hits else "weak"
    if not hits:
        status = "no-evidence"
    return {
        "status": status,
        "allowed_citations": [hit.chunk_id for hit in hits],
        "matched_aliases": alias_hits,
        "rules": [
            "Do not assert identity facts without a cited chunk.",
            "Treat filesystem-only leads as leads, not verified evidence.",
            "If no alias appears in evidence text, answer as weak or no evidence.",
            "Use negative search language carefully: absence in local data is not real-world absence.",
        ],
    }


def validate_investigation_answer(answer: str, allowed_citations: list[str]) -> dict[str, object]:
    allowed = set(allowed_citations)
    sentences = split_sentences(answer)
    unsupported: list[str] = []
    unknown_citations: list[str] = []
    for sentence in sentences:
        citations = extract_citations(sentence)
        if not citations:
            unsupported.append(sentence)
            continue
        unknown = [citation for citation in citations if citation not in allowed]
        unknown_citations.extend(unknown)
    return {
        "safe": not unsupported and not unknown_citations,
        "unsupported_sentences": unsupported,
        "unknown_citations": sorted(set(unknown_citations)),
    }


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def hit_payload(hit: SearchHit) -> dict[str, object]:
    return {
        "citation": citation_token(hit.chunk_id),
        "chunk_id": hit.chunk_id,
        "file_name": hit.file_name,
        "file_path": hit.file_path,
        "modality": hit.modality,
        "location": hit.location,
        "score": hit.score,
        "snippet": compact(hit.snippet, 360),
    }


def compact(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
