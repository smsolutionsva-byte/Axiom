from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .chunking import stable_id
from .database import (
    get_chunk,
    get_context_for_child,
    iter_links_for_chunks,
    iter_child_vectors,
    lexical_search,
    record_query_context,
    vector_from_row,
)
from .embeddings import cosine, embed_text, lexical_overlap, tokenize


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    context_id: str
    score: float
    text: str
    snippet: str
    file_name: str
    file_path: str
    sha256: str
    modality: str
    location: str


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for index, ranking in enumerate(rankings):
        weight = weights[index] if weights and index < len(weights) else 1.0
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def dense_search(conn: sqlite3.Connection, query: str, limit: int = 25) -> list[tuple[str, float]]:
    query_vector = embed_text(query)
    scored: list[tuple[str, float]] = []
    for row in iter_child_vectors(conn):
        score = cosine(query_vector, vector_from_row(row))
        if score > 0:
            scored.append((row["chunk_id"], score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def search(conn: sqlite3.Connection, query: str, *, top_k: int = 5) -> tuple[str, list[SearchHit]]:
    dense = dense_search(conn, query, limit=max(top_k * 8, 25))
    sparse = lexical_search(conn, query, limit=max(top_k * 8, 25))
    fused = reciprocal_rank_fusion([dense, sparse])
    reranked = rerank(conn, query, fused[: max(top_k * 4, 20)])
    selected = reranked[:top_k]

    query_id = stable_id("query", query, *[chunk_id for chunk_id, _ in selected], length=24)
    record_query_context(conn, query_id=query_id, query_text=query, ranked=selected)
    conn.commit()

    hits = [make_hit(conn, chunk_id, score, query) for chunk_id, score in selected]
    return query_id, [hit for hit in hits if hit is not None]


def avtr_search(conn: sqlite3.Connection, query: str, *, top_k: int = 5) -> tuple[str, list[SearchHit]]:
    seed_limit = max(top_k * 10, 40)
    dense = dense_search(conn, query, limit=seed_limit)
    sparse = lexical_search(conn, query, limit=seed_limit)
    weights = adaptive_branch_weights(query)
    fused = reciprocal_rank_fusion([dense, sparse], weights=weights)
    seeded = rerank(conn, query, fused[: max(top_k * 8, 40)])

    web_candidates = expand_spider_web(conn, seeded, max_depth=2, max_candidates=max(top_k * 16, 90))
    reranked = rerank(conn, query, web_candidates[: max(top_k * 20, 100)])
    selected = coverage_select(conn, query, reranked, top_k=top_k)

    if not selected:
        selected = rerank(conn, query, fused[: max(top_k * 4, 20)])[:top_k]

    query_id = stable_id("avtr-query", query, *[chunk_id for chunk_id, _ in selected], length=24)
    record_query_context(conn, query_id=query_id, query_text=query, ranked=selected)
    conn.commit()

    hits = [make_hit(conn, chunk_id, score, query) for chunk_id, score in selected]
    return query_id, [hit for hit in hits if hit is not None]


def adaptive_branch_weights(query: str) -> list[float]:
    tokens = tokenize(query)
    has_identifier = any(any(char.isdigit() for char in token) or "-" in token or "_" in token for token in tokens)
    is_long_question = len(tokens) >= 8
    dense_weight = 1.15 if is_long_question else 1.0
    lexical_weight = 1.25 if has_identifier else 1.0
    return [dense_weight, lexical_weight]


def expand_spider_web(
    conn: sqlite3.Connection,
    seeds: list[tuple[str, float]],
    *,
    max_depth: int = 2,
    max_candidates: int = 90,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    frontier: dict[str, float] = {}
    for chunk_id, score in seeds[:max_candidates]:
        scores[chunk_id] = max(scores.get(chunk_id, 0.0), score)
        if len(frontier) < 40:
            frontier[chunk_id] = max(frontier.get(chunk_id, 0.0), score)

    for depth in range(max_depth):
        if not frontier:
            break
        next_frontier: dict[str, float] = {}
        for link in iter_links_for_chunks(conn, frontier.keys()):
            source = link["source_chunk_id"]
            target = link["target_chunk_id"]
            confidence = float(link["confidence_score"] or 0.0)
            link_type = str(link["link_type"] or "")
            for from_id, to_id in ((source, target), (target, source)):
                if from_id not in frontier:
                    continue
                candidate = get_chunk(conn, to_id)
                if candidate is None or candidate["chunk_kind"] != "child":
                    continue
                decay = link_decay(link_type, depth)
                propagated = frontier[from_id] * max(confidence, 0.35) * decay
                if propagated <= scores.get(to_id, 0.0):
                    continue
                scores[to_id] = propagated
                next_frontier[to_id] = propagated
                if len(scores) >= max_candidates:
                    break
            if len(scores) >= max_candidates:
                break
        frontier = dict(sorted(next_frontier.items(), key=lambda item: item[1], reverse=True)[:40])

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def link_decay(link_type: str, depth: int) -> float:
    base = {
        "semantic": 0.72,
        "entity": 0.68,
        "sibling": 0.55,
    }.get(link_type, 0.58)
    return base * (0.72 ** depth)


def coverage_select(
    conn: sqlite3.Connection,
    query: str,
    candidates: list[tuple[str, float]],
    *,
    top_k: int,
    min_relevance: float = 0.0,
) -> list[tuple[str, float]]:
    query_tokens = tokenize(query)
    selected: list[tuple[str, float]] = []
    selected_chunks: list[sqlite3.Row] = []
    seen_files: set[str] = set()
    seen_pages: set[tuple[str, int | None]] = set()
    seen_modalities: set[str] = set()

    pool: list[tuple[str, float, sqlite3.Row, float, float]] = []
    for chunk_id, base_score in candidates:
        chunk = get_chunk(conn, chunk_id)
        if chunk is None or chunk["chunk_kind"] != "child":
            continue
        search_text = evidence_search_text(chunk)
        overlap = lexical_overlap(query_tokens, search_text)
        identifier_overlap = exact_identifier_overlap(query_tokens, search_text)
        novelty = 0.0
        if chunk["file_id"] not in seen_files:
            novelty += 0.018
        if (chunk["file_id"], chunk["page_number"]) not in seen_pages:
            novelty += 0.008
        if chunk["modality"] not in seen_modalities:
            novelty += 0.006
        risk = prompt_injection_risk(chunk["text_content"]) * 0.025
        redundancy = max((chunk_redundancy(chunk, existing) for existing in selected_chunks), default=0.0)
        pool.append(
            (
                chunk_id,
                base_score + overlap * 0.085 + identifier_overlap * 0.11 + novelty - redundancy * 0.025 - risk,
                chunk,
                overlap,
                identifier_overlap,
            )
        )

    for chunk_id, score, chunk, overlap, identifier_overlap in sorted(pool, key=lambda item: item[1], reverse=True):
        if any(chunk_id == selected_id for selected_id, _ in selected):
            continue
        if min_relevance and selected and overlap < min_relevance and identifier_overlap <= 0:
            continue
        redundancy = max((chunk_redundancy(chunk, existing) for existing in selected_chunks), default=0.0)
        if redundancy > 0.92 and len(selected) >= 1:
            continue
        selected.append((chunk_id, score))
        selected_chunks.append(chunk)
        seen_files.add(chunk["file_id"])
        seen_pages.add((chunk["file_id"], chunk["page_number"]))
        seen_modalities.add(chunk["modality"])
        if len(selected) >= top_k:
            break
    return selected


def exact_identifier_overlap(query_tokens: list[str], text: str) -> float:
    identifiers = {token for token in query_tokens if any(char.isdigit() for char in token)}
    if not identifiers:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(identifiers & text_tokens) / max(len(identifiers), 1)


def chunk_redundancy(left: sqlite3.Row, right: sqlite3.Row) -> float:
    left_terms = set(tokenize(left["text_content"]))
    right_terms = set(tokenize(right["text_content"]))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def prompt_injection_risk(text: str) -> float:
    lowered = text.lower()
    markers = (
        "ignore previous instructions",
        "system prompt",
        "developer message",
        "forget your instructions",
        "you are now",
    )
    return 1.0 if any(marker in lowered for marker in markers) else 0.0


def rerank(conn: sqlite3.Connection, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    query_tokens = tokenize(query)
    reranked: list[tuple[str, float]] = []
    for chunk_id, base_score in candidates:
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue
        overlap = lexical_overlap(query_tokens, evidence_search_text(chunk))
        reranked.append((chunk_id, base_score + overlap * 0.095))
    return sorted(reranked, key=lambda item: item[1], reverse=True)


def evidence_search_text(row: sqlite3.Row) -> str:
    file_name = str(row["file_name"] or "")
    file_type = str(row["file_type"] or "")
    modality = str(row["modality"] or "")
    labels = [file_name, file_type, modality, row["text_content"]]
    labels.append(evidence_kind_terms(file_name=file_name, file_type=file_type, modality=modality))
    return " ".join(str(part) for part in labels if part)


def evidence_kind_terms(*, file_name: str, file_type: str, modality: str) -> str:
    lowered_name = file_name.lower()
    lowered_type = file_type.lower()
    lowered_modality = modality.lower()
    terms: list[str] = []
    if lowered_modality in {"transcript", "audio"} or lowered_type in {"wav", "mp3", "m4a", "ogg"}:
        terms.append("voice transcript audio recording")
    if lowered_modality in {"ocr", "image"} or lowered_type in {"png", "jpg", "jpeg", "webp"}:
        terms.append("screenshot ocr image visual evidence")
    if lowered_type in {"doc", "docx"}:
        terms.append("document brief notes")
    if lowered_type == "pdf":
        terms.append("pdf annexure report")
    if "screen" in lowered_name or "screenshot" in lowered_name:
        terms.append("screenshot")
    if "voice" in lowered_name or "call" in lowered_name:
        terms.append("voice transcript")
    return " ".join(terms)


def make_hit(conn: sqlite3.Connection, child_id: str, score: float, query: str) -> SearchHit | None:
    child = get_chunk(conn, child_id)
    context = get_context_for_child(conn, child_id)
    if child is None or context is None or child["chunk_kind"] != "child":
        return None

    location = format_location(child)
    snippet = make_snippet(child["text_content"], query)
    return SearchHit(
        chunk_id=child_id,
        context_id=context["chunk_id"],
        score=score,
        text=context["text_content"],
        snippet=snippet,
        file_name=child["file_name"],
        file_path=child["file_path"],
        sha256=child["sha256"],
        modality=child["modality"],
        location=location,
    )


def format_location(row: sqlite3.Row) -> str:
    if row["page_number"] is not None:
        return f"Page {row['page_number']}"
    if row["start_timestamp"] or row["end_timestamp"]:
        start = row["start_timestamp"] or "start"
        end = row["end_timestamp"] or "end"
        return f"{start} to {end}"
    if row["char_start"] is not None and row["char_end"] is not None:
        return f"Chars {row['char_start']}-{row['char_end']}"
    return "File"


def make_snippet(text: str, query: str, radius: int = 180) -> str:
    lower_text = text.lower()
    tokens = tokenize(query)
    positions = [lower_text.find(token) for token in tokens if lower_text.find(token) != -1]
    if not positions:
        for token in tokens:
            if len(token) > 4 and token.endswith("s"):
                singular = token[:-1]
                index = lower_text.find(singular)
                if index != -1:
                    positions.append(index)
    if not positions:
        return text[: radius * 2].strip()
    center = min(positions)
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"
