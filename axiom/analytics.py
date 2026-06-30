from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .citation import citation_token
from .database import get_chunk
from .embeddings import tokenize
from .retrieval import search


DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?)?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?\b")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}(?:\s+[A-Z][A-Za-z0-9]{2,}){0,3}\b")

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "during",
    "report",
    "transcript",
    "text",
    "image",
    "audio",
    "page",
    "source",
}

GENERIC_ENTITIES = {
    "The",
    "At",
    "Axiom",
    "Text",
    "Image",
    "Audio",
    "Transcript",
}

SIGNAL_WORDS = {
    "risk": 3,
    "urgent": 3,
    "warning": 3,
    "threat": 4,
    "error": 2,
    "shift": 2,
    "changed": 2,
    "compare": 2,
    "cross": 2,
    "target": 2,
    "plan": 1,
    "screenshot": 2,
    "field": 1,
    "review": 1,
    "operational": 2,
}


@dataclass(frozen=True)
class AnalyticsRow:
    chunk_id: str
    file_id: str
    file_name: str
    file_type: str
    modality: str
    text_content: str
    page_number: int | None
    start_timestamp: str | None
    end_timestamp: str | None


def build_analytics(conn: sqlite3.Connection, *, query: str | None = None, limit: int = 80) -> dict[str, object]:
    rows = selected_rows(conn, query=query, limit=limit)
    graph = build_graph(conn, rows)
    timeline = build_timeline(rows)
    prediction = build_prediction(rows, timeline)
    return {
        "query": query or "",
        "metrics": build_metrics(conn, rows, timeline, graph),
        "graph": graph,
        "timeline": timeline,
        "prediction": prediction,
    }


def selected_rows(conn: sqlite3.Connection, *, query: str | None, limit: int) -> list[AnalyticsRow]:
    if query and query.strip():
        _query_id, hits = search(conn, query, top_k=min(max(limit, 5), 25))
        rows = [get_chunk(conn, hit.chunk_id) for hit in hits]
        return [row_to_analytics(row) for row in rows if row is not None]

    rows = conn.execute(
        """
        SELECT c.*, d.file_name, d.file_type
        FROM content_chunks c
        JOIN document_registry d ON d.file_id = c.file_id
        WHERE c.chunk_kind = 'child'
        ORDER BY d.ingested_at DESC, d.file_name ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [row_to_analytics(row) for row in rows]


def row_to_analytics(row: sqlite3.Row) -> AnalyticsRow:
    return AnalyticsRow(
        chunk_id=row["chunk_id"],
        file_id=row["file_id"],
        file_name=row["file_name"],
        file_type=row["file_type"],
        modality=row["modality"],
        text_content=row["text_content"],
        page_number=row["page_number"],
        start_timestamp=row["start_timestamp"],
        end_timestamp=row["end_timestamp"],
    )


def build_metrics(conn: sqlite3.Connection, rows: list[AnalyticsRow], timeline: list[dict[str, object]], graph: dict[str, object]) -> dict[str, object]:
    doc_count = conn.execute("SELECT COUNT(*) AS count FROM document_registry").fetchone()["count"]
    chunk_count = conn.execute("SELECT COUNT(*) AS count FROM content_chunks WHERE chunk_kind = 'child'").fetchone()["count"]
    modality_counts = Counter(row.modality for row in rows)
    return {
        "documents_total": doc_count,
        "chunks_total": chunk_count,
        "chunks_analyzed": len(rows),
        "modalities": dict(modality_counts),
        "timeline_items": len(timeline),
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
    }


def build_graph(conn: sqlite3.Connection, rows: list[AnalyticsRow]) -> dict[str, object]:
    nodes: dict[str, dict[str, object]] = {}
    edges: dict[tuple[str, str, str], dict[str, object]] = {}

    def node(node_id: str, label: str, kind: str, weight: int = 1, **extra: object) -> None:
        existing = nodes.get(node_id)
        if existing:
            existing["weight"] = int(existing.get("weight", 1)) + weight
            return
        nodes[node_id] = {"id": node_id, "label": label, "type": kind, "weight": weight, **extra}

    def edge(source: str, target: str, kind: str, weight: float = 1.0) -> None:
        key = (source, target, kind)
        existing = edges.get(key)
        if existing:
            existing["weight"] = float(existing.get("weight", 1.0)) + weight
            return
        edges[key] = {"source": source, "target": target, "type": kind, "weight": weight}

    for row in rows:
        doc_id = f"doc:{row.file_id}"
        chunk_id = f"chunk:{row.chunk_id}"
        modality_id = f"modality:{row.modality}"
        node(doc_id, row.file_name, "document", 2)
        node(chunk_id, row.chunk_id[:8], "chunk", 1, citation=citation_token(row.chunk_id))
        node(modality_id, row.modality, "modality", 1)
        edge(doc_id, chunk_id, "contains")
        edge(chunk_id, modality_id, "modality")

        for entity in top_entities(row.text_content, limit=8):
            entity_id = f"entity:{entity.lower()}"
            node(entity_id, entity, "entity", 1)
            edge(chunk_id, entity_id, "mentions")

        for date in sorted(set(DATE_RE.findall(row.text_content))):
            date_id = f"date:{date}"
            node(date_id, date, "date", 2)
            edge(chunk_id, date_id, "references")

    links = conn.execute(
        """
        SELECT source_chunk_id, target_chunk_id, confidence_score, link_type
        FROM cross_modal_links
        ORDER BY confidence_score DESC
        LIMIT 200
        """
    ).fetchall()
    chunk_ids = {row.chunk_id for row in rows}
    for link in links:
        if link["source_chunk_id"] in chunk_ids and link["target_chunk_id"] in chunk_ids:
            edge(f"chunk:{link['source_chunk_id']}", f"chunk:{link['target_chunk_id']}", f"cross:{link['link_type']}", link["confidence_score"])

    return {
        "nodes": sorted(nodes.values(), key=lambda item: (-int(item.get("weight", 1)), str(item["label"])))[:160],
        "edges": list(edges.values())[:260],
    }


def build_timeline(rows: list[AnalyticsRow]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for row in rows:
        dates = sorted(set(DATE_RE.findall(row.text_content)))
        times = sorted(set(TIME_RE.findall(row.text_content)))
        if row.start_timestamp or row.end_timestamp:
            items.append(timeline_item(row, row.start_timestamp or "start", "audio-timestamp", 0.9))
        for date in dates:
            items.append(timeline_item(row, date, "date-mention", 0.76))
        for time in times[:4]:
            items.append(timeline_item(row, time, "time-mention", 0.62))
    return sorted(items, key=lambda item: (str(item["when"]), str(item["source"])))[:100]


def timeline_item(row: AnalyticsRow, when: str, kind: str, confidence: float) -> dict[str, object]:
    return {
        "when": when,
        "kind": kind,
        "confidence": confidence,
        "source": row.file_name,
        "modality": row.modality,
        "citation": citation_token(row.chunk_id),
        "summary": compact(row.text_content, 240),
    }


def build_prediction(rows: list[AnalyticsRow], timeline: list[dict[str, object]]) -> dict[str, object]:
    all_text = "\n".join(row.text_content for row in rows).lower()
    signal_score = 0
    signals: list[dict[str, object]] = []
    for word, weight in SIGNAL_WORDS.items():
        count = len(re.findall(rf"\b{re.escape(word)}\b", all_text))
        if count:
            score = count * weight
            signal_score += score
            signals.append({"signal": word, "count": count, "score": score})

    modalities = Counter(row.modality for row in rows)
    years = Counter(match[:4] for match in DATE_RE.findall("\n".join(row.text_content for row in rows)))
    trend = trend_summary(years)
    gaps = evidence_gaps(modalities, rows)
    confidence = confidence_score(rows, timeline, signal_score, gaps)
    next_actions = recommended_actions(gaps, signals, timeline)
    forecast = forecast_summary(confidence, signal_score, modalities, years)
    return {
        "confidence": confidence,
        "forecast": forecast,
        "trend": trend,
        "signals": sorted(signals, key=lambda item: (-int(item["score"]), str(item["signal"])))[:12],
        "gaps": gaps,
        "next_actions": next_actions,
        "caveat": "Prediction is evidence-prioritization, not external forecasting. It uses only indexed local evidence.",
    }


def confidence_score(rows: list[AnalyticsRow], timeline: list[dict[str, object]], signal_score: int, gaps: list[str]) -> float:
    if not rows:
        return 0.0
    base = 0.35
    base += min(len(rows), 10) * 0.035
    base += min(len(set(row.file_id for row in rows)), 5) * 0.04
    base += min(len(timeline), 6) * 0.025
    base += min(signal_score, 20) * 0.008
    base -= min(len(gaps), 5) * 0.04
    return round(max(0.05, min(base, 0.92)), 2)


def evidence_gaps(modalities: Counter[str], rows: list[AnalyticsRow]) -> list[str]:
    gaps: list[str] = []
    if not rows:
        return ["No indexed evidence is available."]
    if len(set(row.file_id for row in rows)) < 2:
        gaps.append("Only one source file supports the current analysis.")
    if not any(name in modalities for name in ("ocr", "image_caption")):
        gaps.append("No screenshot/image OCR or caption evidence is present.")
    if "transcript" not in modalities and "audio" not in modalities:
        gaps.append("No audio transcript evidence is present.")
    if not any(DATE_RE.search(row.text_content) for row in rows):
        gaps.append("No explicit date or year mentions were found.")
    return gaps


def recommended_actions(gaps: list[str], signals: list[dict[str, object]], timeline: list[dict[str, object]]) -> list[str]:
    actions: list[str] = []
    if any("screenshot" in gap.lower() or "image" in gap.lower() for gap in gaps):
        actions.append("Capture or ingest screenshots, then run OCR/VLM analysis.")
    if any("audio" in gap.lower() for gap in gaps):
        actions.append("Transcribe related voice recordings and ingest the transcript sidecars.")
    if timeline:
        actions.append("Review the timeline and open the highest-confidence cited source.")
    if signals:
        top = signals[0]["signal"]
        actions.append(f"Run a focused query for the strongest signal: {top}.")
    if not actions:
        actions.append("Evidence coverage is broad enough for a first-pass report.")
    return actions[:5]


def forecast_summary(confidence: float, signal_score: int, modalities: Counter[str], years: Counter[str]) -> str:
    if confidence < 0.35:
        return "Low-confidence analysis: more indexed evidence is needed before drawing operational conclusions."
    if signal_score >= 16 and len(modalities) >= 2:
        return "High-priority cluster: repeated signals across multiple modalities suggest this topic deserves immediate analyst review."
    if years:
        latest = max(years)
        return f"Time-anchored cluster: evidence centers on {latest}; monitor for related records and screenshots tied to that period."
    return "Stable evidence cluster: available material supports summarization, but predictive strength is limited without dates or multiple modalities."


def trend_summary(years: Counter[str]) -> dict[str, object]:
    if not years:
        return {"direction": "unknown", "points": {}, "summary": "No year-based trend can be computed."}
    ordered = dict(sorted(years.items()))
    values = list(ordered.values())
    if len(values) == 1:
        direction = "single-period"
    elif values[-1] > values[0]:
        direction = "increasing"
    elif values[-1] < values[0]:
        direction = "decreasing"
    else:
        direction = "flat"
    return {
        "direction": direction,
        "points": ordered,
        "summary": f"Year mentions are {direction} across indexed evidence.",
    }


def top_entities(text: str, *, limit: int) -> list[str]:
    entities = []
    for match in ENTITY_RE.findall(text):
        cleaned = " ".join(match.split())
        if cleaned in GENERIC_ENTITIES or cleaned.lower() in STOPWORDS:
            continue
        if len(cleaned) < 4 or cleaned.isupper():
            continue
        entities.append(cleaned)
    token_counts = Counter(token for token in tokenize(text) if token not in STOPWORDS and len(token) > 4)
    keyword_entities = [token.title() for token, _count in token_counts.most_common(limit)]
    combined = list(dict.fromkeys(entities + keyword_entities))
    return combined[:limit]


def compact(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
