from __future__ import annotations

import sqlite3
from collections import Counter

from .dependencies import audit_dependencies


PROBLEM = {
    "organization": "National Technical Research Organisation (NTRO)",
    "ps_id": "SIH25231",
    "category": "Software",
    "theme": "Smart Automation",
    "statement": (
        "Design and build a multimodal Retrieval-Augmented Generation system leveraging a Large Language Model "
        "for offline mode that can ingest, index, and query DOC, PDF, images, and voice recordings within a "
        "unified semantic retrieval framework."
    ),
}


MODALITY_TARGETS = [
    {"key": "doc", "label": "DOC/DOCX", "file_types": {"doc", "docx"}, "modalities": {"text"}},
    {"key": "pdf", "label": "PDF", "file_types": {"pdf"}, "modalities": {"text"}},
    {"key": "image", "label": "Images", "file_types": {"image"}, "modalities": {"ocr", "image_caption"}},
    {"key": "audio", "label": "Voice", "file_types": {"audio"}, "modalities": {"transcript"}},
]


def build_mission_brief(conn: sqlite3.Connection) -> dict[str, object]:
    documents = [dict(row) for row in conn.execute("SELECT * FROM document_registry").fetchall()]
    chunks = [dict(row) for row in conn.execute("SELECT modality, file_id FROM content_chunks WHERE chunk_kind = 'child'").fetchall()]
    links = conn.execute("SELECT COUNT(*) AS count FROM cross_modal_links").fetchone()["count"]
    queries = conn.execute("SELECT COUNT(*) AS count FROM query_audit").fetchone()["count"]
    actions = conn.execute("SELECT COUNT(*) AS count FROM operator_audit").fetchone()["count"]
    file_counts = Counter(row["file_type"] for row in documents)
    status_counts = Counter(row["status"] for row in documents)
    modality_counts = Counter(row["modality"] for row in chunks)

    coverage = coverage_rows(documents, chunks)
    score = readiness_score(documents, chunks, links, coverage)
    gaps = readiness_gaps(coverage, status_counts, links)
    return {
        "problem": PROBLEM,
        "score": score,
        "verdict": verdict(score, gaps),
        "corpus": {
            "documents": len(documents),
            "chunks": len(chunks),
            "cross_modal_links": links,
            "queries": queries,
            "audited_actions": actions,
            "file_types": dict(sorted(file_counts.items())),
            "modalities": dict(sorted(modality_counts.items())),
            "statuses": dict(sorted(status_counts.items())),
        },
        "coverage": coverage,
        "offline": offline_status(),
        "differentiators": differentiators(links),
        "gaps": gaps,
        "demo_script": demo_script(coverage),
    }


def coverage_rows(documents: list[dict[str, object]], chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    file_types_by_id = {row["file_id"]: row["file_type"] for row in documents}
    modality_by_file: dict[str, set[str]] = {}
    for row in chunks:
        modality_by_file.setdefault(str(row["file_id"]), set()).add(str(row["modality"]))

    rows = []
    for target in MODALITY_TARGETS:
        if target["key"] in {"doc", "pdf"}:
            matching_files = [doc for doc in documents if str(doc["file_type"]) in target["file_types"]]
        else:
            matching_files = [
                doc
                for doc in documents
                if str(doc["file_type"]) in target["file_types"]
                or (str(doc["file_id"]) in modality_by_file and modality_by_file[str(doc["file_id"])] & target["modalities"])
            ]
        indexed = [doc for doc in matching_files if doc.get("status") == "indexed"]
        needs_adapter = [doc for doc in matching_files if doc.get("status") == "needs_adapter"]
        modality_hits = sum(
            1
            for file_id, file_type in file_types_by_id.items()
            if file_type in target["file_types"] and modality_by_file.get(str(file_id), set()) & target["modalities"]
        )
        rows.append(
            {
                "key": target["key"],
                "label": target["label"],
                "files": len(matching_files),
                "indexed": len(indexed),
                "needs_adapter": len(needs_adapter),
                "modality_hits": modality_hits,
                "ready": bool(indexed and (target["key"] in {"doc", "pdf"} or modality_hits)),
            }
        )
    return rows


def readiness_score(
    documents: list[dict[str, object]],
    chunks: list[dict[str, object]],
    links: int,
    coverage: list[dict[str, object]],
) -> int:
    score = 18
    if documents:
        score += 12
    if chunks:
        score += 16
    score += sum(10 for row in coverage if row["ready"])
    if links:
        score += 8
    if len({row["modality"] for row in chunks}) >= 2:
        score += 6
    if any(row.get("status") == "needs_adapter" for row in documents):
        score -= 8
    return max(0, min(100, score))


def readiness_gaps(coverage: list[dict[str, object]], statuses: Counter[str], links: int) -> list[str]:
    gaps = []
    for row in coverage:
        if not row["ready"]:
            gaps.append(f"Add or index {row['label']} evidence for the SIH demo corpus.")
        elif row["needs_adapter"]:
            gaps.append(f"{row['label']} has files waiting on local extraction adapters.")
    if not links:
        gaps.append("Index at least two modalities with overlapping topics to show cross-modal semantic links.")
    if statuses.get("needs_adapter", 0):
        gaps.append("Open Setup and install/stage missing local adapters before the final demo.")
    return gaps[:8]


def offline_status() -> dict[str, object]:
    audit = audit_dependencies()
    checks = audit.get("checks", [])
    important = [
        "pymupdf",
        "pypdf",
        "python-docx",
        "pillow",
        "mss",
        "pytesseract",
        "paddleocr",
        "tesseract",
        "ollama",
        "model:llama3.2-vision",
    ]
    by_key = {item["key"]: item for item in checks if isinstance(item, dict)}
    return {
        "ready": audit.get("ready", False),
        "required_installed": audit.get("required_installed", 0),
        "required_total": audit.get("required_total", 0),
        "adapters": [
            {
                "key": key,
                "label": by_key.get(key, {}).get("label", key),
                "installed": bool(by_key.get(key, {}).get("installed", False)),
                "note": by_key.get(key, {}).get("note", ""),
            }
            for key in important
        ],
    }


def differentiators(links: int) -> list[dict[str, str]]:
    return [
        {
            "title": "Air-gapped evidence ledger",
            "body": "SQLite stores file hashes, chunks, metadata, query context, and operator audit locally.",
        },
        {
            "title": "Citation allow-list guard",
            "body": "Answers can cite only chunks returned by retrieval; unknown citations are marked unverified.",
        },
        {
            "title": "Cross-modal evidence graph",
            "body": f"Semantic links connect evidence across modalities; current graph has {links} link(s).",
        },
        {
            "title": "Screenshot-to-evidence workflow",
            "body": "Snips and desktop screenshots become OCR/VLM sidecars and can be queried like documents.",
        },
        {
            "title": "Operator audit plane",
            "body": "Local file, folder, command, window, tab, and screenshot actions are explicit and auditable.",
        },
    ]


def demo_script(coverage: list[dict[str, object]]) -> list[str]:
    missing = [row["label"] for row in coverage if not row["ready"]]
    if missing:
        return [
            f"Use the + menu to add a folder containing: {', '.join(missing)}.",
            "Run Mission Brief again and show the coverage tiles turning ready.",
            "Ask a cross-modal question that mentions a year, screenshot, and audio clue.",
            "Open Analytics to show graph, timeline, gaps, and next actions.",
            "Export a cited report and show the query/operator audit trail.",
        ]
    return [
        "Open Mission Brief and show full SIH modality coverage.",
        "Ask a cross-modal query across document, image, and voice evidence.",
        "Open Analytics to show the cross-modal graph and timeline.",
        "Paste a fresh screenshot through the + menu and ingest it live.",
        "Export a cited report and show that citations map back to local files.",
    ]


def verdict(score: int, gaps: list[str]) -> str:
    if score >= 88 and not gaps:
        return "Demo-ready multimodal offline RAG workspace."
    if score >= 70:
        return "Strong SIH demo base; fill the remaining modality gaps for maximum judge impact."
    if score >= 45:
        return "Evidence core is working; add missing modalities and adapter setup before demo."
    return "Needs a richer local corpus before it can win the SIH problem statement."
