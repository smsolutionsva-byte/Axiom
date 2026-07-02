from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import parent_child_windows, stable_id
from .biorag import refresh_biorag_index
from .database import child_rows_for_parents, insert_chunk, insert_cross_link, iter_child_vectors, upsert_document, vector_from_row
from .embeddings import cosine, embed_text, tokenize
from .extractors import MissingExtractor, extract_segments, file_type_for, is_sidecar


@dataclass
class IngestReport:
    indexed_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    error_files: dict[str, str] = field(default_factory=dict)
    chunks_created: int = 0
    links_created: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def ingest_path(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    build_links: bool = True,
    link_strategy: str = "incremental",
    build_biorag_index: bool = True,
    progress: bool = False,
    progress_every: int = 250,
) -> IngestReport:
    root = Path(path)
    if link_strategy not in {"incremental", "batch"}:
        raise ValueError(f"Unknown link strategy: {link_strategy}")
    report = IngestReport()
    indexed_child_ids: list[str] = []
    files = iter_input_files(root)

    if progress:
        print(f"Found {len(files)} input file(s) under {root}.", flush=True)

    for file_index, file_path in enumerate(files, start=1):
        if is_sidecar(file_path):
            continue
        if file_type_for(file_path) == "unsupported":
            report.skipped_files.append(str(file_path))
            continue
        new_child_ids = ingest_file(conn, file_path, report)
        indexed_child_ids.extend(new_child_ids)
        if build_links and link_strategy == "incremental" and new_child_ids:
            report.links_created += build_cross_modal_links(conn, new_child_ids)
            report.links_created += build_spider_links(conn, new_child_ids)
        if progress and (file_index == 1 or file_index % progress_every == 0 or file_index == len(files)):
            print(
                f"Indexed {file_index}/{len(files)} file(s): "
                f"{report.chunks_created} chunk(s), {report.links_created} link(s).",
                flush=True,
            )

    if build_links and indexed_child_ids:
        if link_strategy == "batch":
            if progress:
                print(f"Building retrieval links for {len(indexed_child_ids)} chunk(s)...", flush=True)
            report.links_created += build_cross_modal_links(conn, indexed_child_ids)
            report.links_created += build_spider_links(conn, indexed_child_ids)
            if progress:
                print(f"Built {report.links_created} retrieval link(s).", flush=True)
        if build_biorag_index:
            if progress:
                print("Building HiveRAG index layers...", flush=True)
            report.links_created += refresh_biorag_index(conn, indexed_child_ids)

    conn.commit()
    return report


def ingest_file(conn: sqlite3.Connection, path: Path, report: IngestReport) -> list[str]:
    resolved = path.resolve()
    digest = sha256_file(resolved)
    file_id = stable_id(digest, str(resolved), length=24)
    file_type = file_type_for(resolved)
    size_bytes = resolved.stat().st_size

    try:
        segments = extract_segments(resolved)
    except MissingExtractor as exc:
        upsert_document(
            conn,
            file_id=file_id,
            file_name=resolved.name,
            file_type=file_type,
            file_path=str(resolved),
            sha256=digest,
            size_bytes=size_bytes,
            status="needs_adapter",
            error_message=str(exc),
        )
        report.error_files[str(resolved)] = str(exc)
        return []

    upsert_document(
        conn,
        file_id=file_id,
        file_name=resolved.name,
        file_type=file_type,
        file_path=str(resolved),
        sha256=digest,
        size_bytes=size_bytes,
        status="indexed",
    )

    child_ids: list[str] = []
    for segment_index, segment in enumerate(segments):
        for parent_index, (parent, children) in enumerate(parent_child_windows(segment.text)):
            parent_id = stable_id(file_id, segment_index, parent_index, "parent", parent.char_start, parent.char_end)
            insert_chunk(
                conn,
                chunk_id=parent_id,
                parent_id=None,
                file_id=file_id,
                chunk_kind="parent",
                text_content=parent.text,
                modality=segment.modality,
                page_number=segment.page_number,
                start_timestamp=segment.start_timestamp,
                end_timestamp=segment.end_timestamp,
                char_start=parent.char_start,
                char_end=parent.char_end,
                token_count=parent.token_count,
            )
            report.chunks_created += 1

            for child_index, child in enumerate(children):
                child_id = stable_id(
                    file_id,
                    segment_index,
                    parent_index,
                    child_index,
                    "child",
                    child.char_start,
                    child.char_end,
                )
                insert_chunk(
                    conn,
                    chunk_id=child_id,
                    parent_id=parent_id,
                    file_id=file_id,
                    chunk_kind="child",
                    text_content=child.text,
                    modality=segment.modality,
                    page_number=segment.page_number,
                    start_timestamp=segment.start_timestamp,
                    end_timestamp=segment.end_timestamp,
                    char_start=child.char_start,
                    char_end=child.char_end,
                    token_count=child.token_count,
                    vector=embed_text(child.text),
                )
                child_ids.append(child_id)
                report.chunks_created += 1

    report.indexed_files.append(str(resolved))
    return child_ids


def build_cross_modal_links(
    conn: sqlite3.Connection,
    new_child_ids: list[str],
    *,
    threshold: float = 0.78,
    lexical_threshold: float = 0.28,
    max_links: int = 200,
) -> int:
    all_rows = list(iter_child_vectors(conn))
    if len({row["modality"] for row in all_rows}) < 2:
        return 0
    rows_by_id = {row["chunk_id"]: row for row in all_rows}
    new_rows = [rows_by_id[chunk_id] for chunk_id in new_child_ids if chunk_id in rows_by_id]
    links = 0

    for new_row in new_rows:
        new_vector = vector_from_row(new_row)
        for existing_row in all_rows:
            if existing_row["chunk_id"] == new_row["chunk_id"]:
                continue
            if existing_row["file_id"] == new_row["file_id"]:
                continue
            if existing_row["modality"] == new_row["modality"]:
                continue
            lexical_score = lexical_cross_modal_score(new_row["text_content"], existing_row["text_content"])
            score = max(cosine(new_vector, vector_from_row(existing_row)), lexical_score)
            if score >= threshold or lexical_score >= lexical_threshold:
                source, target = sorted([new_row["chunk_id"], existing_row["chunk_id"]])
                insert_cross_link(
                    conn,
                    source_chunk_id=source,
                    target_chunk_id=target,
                    confidence_score=score,
                    link_type="semantic",
                )
                links += 1
                if links >= max_links:
                    return links
    return links


STOP_TERMS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "between",
    "could",
    "from",
    "have",
    "into",
    "only",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "using",
    "with",
    "without",
    "would",
}


def build_spider_links(
    conn: sqlite3.Connection,
    new_child_ids: list[str],
    *,
    term_threshold: float = 0.2,
    semantic_threshold: float = 0.62,
    max_links: int = 600,
) -> int:
    """Build AVTR-style web edges for sibling, entity, and semantic hops."""
    links = build_sibling_links(conn, new_child_ids, max_links=max_links)
    if links >= max_links:
        return links

    all_rows = list(iter_child_vectors(conn))
    rows_by_id = {row["chunk_id"]: row for row in all_rows}
    new_rows = [rows_by_id[chunk_id] for chunk_id in new_child_ids if chunk_id in rows_by_id]
    if not new_rows:
        return links

    terms_by_id: dict[str, set[str]] = {}
    term_index: dict[str, list[str]] = defaultdict(list)
    for row in all_rows:
        terms = set(key_terms(row["text_content"]))
        terms_by_id[row["chunk_id"]] = terms
        for term in terms:
            if len(term_index[term]) < 160:
                term_index[term].append(row["chunk_id"])

    for new_row in new_rows:
        new_terms = terms_by_id.get(new_row["chunk_id"], set())
        if not new_terms:
            continue

        candidate_counts: Counter[str] = Counter()
        for term in new_terms:
            for candidate_id in term_index.get(term, ()):
                if candidate_id != new_row["chunk_id"]:
                    candidate_counts[candidate_id] += 1

        new_vector = vector_from_row(new_row)
        for candidate_id, shared_count in candidate_counts.most_common(36):
            if links >= max_links:
                return links
            existing_row = rows_by_id.get(candidate_id)
            if existing_row is None:
                continue
            if existing_row["file_id"] == new_row["file_id"] and existing_row["parent_id"] == new_row["parent_id"]:
                continue

            existing_terms = terms_by_id.get(candidate_id, set())
            denominator = max(1, min(len(new_terms), len(existing_terms)))
            term_score = shared_count / denominator
            semantic_score = cosine(new_vector, vector_from_row(existing_row))
            score = max(term_score, semantic_score)
            if term_score < term_threshold and semantic_score < semantic_threshold:
                continue

            source, target = sorted([new_row["chunk_id"], candidate_id])
            insert_cross_link(
                conn,
                source_chunk_id=source,
                target_chunk_id=target,
                confidence_score=min(score, 0.99),
                link_type="semantic" if semantic_score >= term_score else "entity",
            )
            links += 1
    return links


def build_sibling_links(conn: sqlite3.Connection, new_child_ids: list[str], *, max_links: int) -> int:
    parent_ids: list[str] = []
    for batch in chunked(new_child_ids):
        parent_ids.extend(
            row["parent_id"]
            for row in conn.execute(
                f"""
                SELECT parent_id
                FROM content_chunks
                WHERE chunk_id IN ({",".join("?" for _ in batch)}) AND parent_id IS NOT NULL
                """,
                batch,
            ).fetchall()
        )
    rows = child_rows_for_parents(conn, parent_ids)
    links = 0
    previous_by_parent: dict[str, sqlite3.Row] = {}
    for row in rows:
        parent_id = row["parent_id"]
        previous = previous_by_parent.get(parent_id)
        if previous is not None:
            source, target = sorted([previous["chunk_id"], row["chunk_id"]])
            insert_cross_link(
                conn,
                source_chunk_id=source,
                target_chunk_id=target,
                confidence_score=0.92,
                link_type="sibling",
            )
            links += 1
            if links >= max_links:
                return links
        previous_by_parent[parent_id] = row
    return links


def chunked(items: list[str], size: int = 500) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def key_terms(text: str, *, limit: int = 28) -> list[str]:
    counts = Counter(
        token
        for token in tokenize(text)
        if len(token) >= 4 and token not in STOP_TERMS
    )
    return [token for token, _count in counts.most_common(limit)]


def lexical_cross_modal_score(left: str, right: str) -> float:
    left_tokens = {token for token in tokenize(left) if len(token) >= 4}
    right_tokens = {token for token in tokenize(right) if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    if not intersection:
        return 0.0
    return len(intersection) / max(1, min(len(left_tokens), len(right_tokens)))
