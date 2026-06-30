from __future__ import annotations

import json
import math
import sqlite3
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .chunking import stable_id
from .database import (
    get_chunk,
    iter_child_vectors,
    iter_links_for_chunks,
    lexical_search,
    record_query_context,
    utc_now,
    vector_from_row,
)
from .embeddings import cosine, dump_vector, embed_text, lexical_overlap, load_vector, tokenize
from .retrieval import (
    SearchHit,
    coverage_select,
    dense_search,
    make_hit,
    reciprocal_rank_fusion,
    rerank,
)


BIORAG_INDEX_VERSION = "biorag-v2"
HEX_CELL_RADIUS = 0.24
HEX_DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
WEB_DAMPING = 0.64
HEBBIAN_LEARNING_RATE = 0.18
HEBBIAN_DECAY_HALF_LIFE_DAYS = 21.0
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


@dataclass(frozen=True)
class EnergyBudget:
    complexity: float
    seed_limit: int
    sphere_limit: int
    tree_limit: int
    hex_frontier: int
    graph_depth: int
    max_candidates: int
    max_subqueries: int
    mode: str


def refresh_biorag_index(conn: sqlite3.Connection, focus_child_ids: list[str] | None = None) -> int:
    build_tree_index(conn)
    build_sphere_index(conn)
    return build_hex_neighbors(conn, focus_child_ids)


def ensure_biorag_index(conn: sqlite3.Connection) -> None:
    child_count = conn.execute(
        "SELECT COUNT(*) AS count FROM content_chunks WHERE chunk_kind = 'child'"
    ).fetchone()["count"]
    if not child_count:
        return
    sphere_count = conn.execute("SELECT COUNT(*) AS count FROM sphere_summaries").fetchone()["count"]
    tree_count = conn.execute("SELECT COUNT(*) AS count FROM biorag_tree_nodes").fetchone()["count"]
    hex_count = conn.execute(
        "SELECT COUNT(*) AS count FROM hex_neighbors WHERE index_version = ?",
        (BIORAG_INDEX_VERSION,),
    ).fetchone()["count"]
    cell_count = conn.execute(
        "SELECT COUNT(*) AS count FROM biorag_chunk_cells WHERE index_version = ?",
        (BIORAG_INDEX_VERSION,),
    ).fetchone()["count"]
    if sphere_count and tree_count and hex_count and cell_count >= min(child_count, 160):
        return
    refresh_biorag_index(conn)
    conn.commit()


def biorag_search(conn: sqlite3.Connection, query: str, *, top_k: int = 5) -> tuple[str, list[SearchHit]]:
    ensure_biorag_index(conn)
    budget = energy_budget(query, top_k=top_k)
    paths: dict[str, set[str]] = defaultdict(set)

    dense = dense_search(conn, query, limit=budget.seed_limit)
    sparse = lexical_search(conn, query, limit=budget.seed_limit)
    mark_paths(paths, dense, "vector")
    mark_paths(paths, sparse, "lexical")

    sphere = sphere_route(conn, query, limit=budget.sphere_limit)
    tree = tree_route(conn, query, sphere, limit=budget.tree_limit)
    seed_fused = reciprocal_rank_fusion(
        [dense, sparse, sphere, tree],
        weights=[1.1, 1.15, 0.85, 0.9],
    )
    mark_paths(paths, sphere, "sphere")
    mark_paths(paths, tree, "tree")

    seeded = rerank(conn, query, seed_fused[: max(top_k * 10, 50)])
    hexed = hex_expand(conn, seeded, frontier=budget.hex_frontier, limit=max(top_k * 12, 50))
    mark_paths(paths, hexed, "hex")

    webbed = propagate_web_signal(
        conn,
        seeded + hexed,
        iterations=budget.graph_depth + 1,
        max_candidates=budget.max_candidates,
    )
    mark_paths(paths, webbed, "spider_signal")

    fused = reciprocal_rank_fusion(
        [seeded, hexed, webbed],
        weights=[1.25, 0.9, 0.95],
    )
    boosted = apply_adaptive_growth(conn, query, fused, paths)
    reranked = rerank(conn, query, boosted[: budget.max_candidates])
    selected = coverage_select(conn, query, reranked, top_k=top_k)

    if not selected:
        selected = rerank(conn, query, seed_fused[: max(top_k * 4, 20)])[:top_k]

    query_id = stable_id("biorag-query", query, *[chunk_id for chunk_id, _ in selected], length=24)
    record_query_context(conn, query_id=query_id, query_text=query, ranked=selected)
    record_retrieval_run(conn, query_id, query, budget, paths, selected)
    update_adaptive_paths(conn, query, paths, selected)
    update_hebbian_edges(conn, selected)
    conn.commit()

    hits = [make_hit(conn, chunk_id, score, query) for chunk_id, score in selected]
    return query_id, [hit for hit in hits if hit is not None]


def biorag_status(conn: sqlite3.Connection) -> dict[str, object]:
    tables = {
        "chunks": "SELECT COUNT(*) AS count FROM content_chunks WHERE chunk_kind = 'child'",
        "spheres": "SELECT COUNT(*) AS count FROM sphere_summaries",
        "tree_nodes": "SELECT COUNT(*) AS count FROM biorag_tree_nodes",
        "hex_neighbors": "SELECT COUNT(*) AS count FROM hex_neighbors",
        "hex_cells": "SELECT COUNT(*) AS count FROM biorag_hex_cells",
        "chunk_cells": "SELECT COUNT(*) AS count FROM biorag_chunk_cells",
        "web_links": "SELECT COUNT(*) AS count FROM cross_modal_links",
        "edge_weights": "SELECT COUNT(*) AS count FROM biorag_edge_weights",
        "adaptive_paths": "SELECT COUNT(*) AS count FROM adaptive_path_stats",
        "retrieval_runs": "SELECT COUNT(*) AS count FROM biorag_retrieval_runs",
    }
    counts = {name: conn.execute(sql).fetchone()["count"] for name, sql in tables.items()}
    latest = conn.execute(
        """
        SELECT query_id, query_text, complexity, budget_json, trace_json, created_at
        FROM biorag_retrieval_runs
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    return {
        "name": "HiveRAG",
        "ready": bool(
            counts["chunks"]
            and counts["spheres"]
            and counts["tree_nodes"]
            and counts["hex_neighbors"]
            and counts["hex_cells"]
            and counts["chunk_cells"]
        ),
        "index_version": BIORAG_INDEX_VERSION,
        "counts": counts,
        "latest_run": dict(latest) if latest else None,
    }


def energy_budget(query: str, *, top_k: int) -> EnergyBudget:
    tokens = tokenize(query)
    lowered = query.lower()
    has_comparison = any(term in lowered for term in ("compare", "versus", "vs", "difference", "between"))
    has_multi_hop = any(term in lowered for term in ("why", "relationship", "related", "connect", "trace", "prove"))
    has_table = any(term in lowered for term in ("table", "row", "cell", "amount", "number", "total", "deadline"))
    has_identifier = any(any(char.isdigit() for char in token) or "-" in token or "_" in token for token in tokens)
    requires_docs = any(term in lowered for term in ("documents", "files", "sources", "pdfs", "across"))
    complexity = (
        1.0
        + 0.08 * len(tokens)
        + 1.0 * has_comparison
        + 1.2 * has_multi_hop
        + 1.0 * has_table
        + 0.8 * has_identifier
        + 0.8 * requires_docs
    )
    if complexity < 2.0:
        mode = "cheap"
        graph_depth = 1
        max_candidates = max(top_k * 12, 60)
    elif complexity < 4.0:
        mode = "balanced"
        graph_depth = 2
        max_candidates = max(top_k * 18, 90)
    else:
        mode = "deep"
        graph_depth = 3
        max_candidates = max(top_k * 24, 140)
    return EnergyBudget(
        complexity=round(complexity, 3),
        seed_limit=max(top_k * 10, 40),
        sphere_limit=8 if mode == "cheap" else 14,
        tree_limit=16 if mode == "cheap" else 28,
        hex_frontier=10 if mode == "cheap" else 18,
        graph_depth=graph_depth,
        max_candidates=max_candidates,
        max_subqueries=1 if mode == "cheap" else 2,
        mode=mode,
    )


def build_tree_index(conn: sqlite3.Connection) -> None:
    now = utc_now()
    rows = list(iter_child_vectors(conn))
    if not rows:
        return
    support_ids = [row["chunk_id"] for row in rows]
    document_ids = sorted({row["file_id"] for row in rows})
    upsert_tree_node(
        conn,
        node_id=stable_id("biorag-tree", "corpus", length=24),
        parent_node_id=None,
        source_id="corpus",
        node_type="corpus",
        level=0,
        summary_text=summary_for_rows(rows, title="Corpus"),
        vector=centroid(rows),
        support_chunk_ids=support_ids[:240],
        document_ids=document_ids,
        now=now,
    )

    rows_by_file: dict[str, list[sqlite3.Row]] = defaultdict(list)
    rows_by_parent: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        rows_by_file[row["file_id"]].append(row)
        if row["parent_id"]:
            rows_by_parent[row["parent_id"]].append(row)

    corpus_node_id = stable_id("biorag-tree", "corpus", length=24)
    for file_id, file_rows in rows_by_file.items():
        doc_node_id = stable_id("biorag-tree", "document", file_id, length=24)
        upsert_tree_node(
            conn,
            node_id=doc_node_id,
            parent_node_id=corpus_node_id,
            source_id=file_id,
            node_type="document",
            level=1,
            summary_text=summary_for_rows(file_rows, title="Document"),
            vector=centroid(file_rows),
            support_chunk_ids=[row["chunk_id"] for row in file_rows[:180]],
            document_ids=[file_id],
            now=now,
        )

    for parent_id, parent_rows in rows_by_parent.items():
        file_id = parent_rows[0]["file_id"]
        upsert_tree_node(
            conn,
            node_id=stable_id("biorag-tree", "section", parent_id, length=24),
            parent_node_id=stable_id("biorag-tree", "document", file_id, length=24),
            source_id=parent_id,
            node_type="section",
            level=2,
            summary_text=summary_for_rows(parent_rows, title="Section"),
            vector=centroid(parent_rows),
            support_chunk_ids=[row["chunk_id"] for row in parent_rows[:80]],
            document_ids=[file_id],
            now=now,
        )


def build_sphere_index(conn: sqlite3.Connection, *, max_spheres: int = 32) -> None:
    rows = list(iter_child_vectors(conn))
    if not rows:
        return
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        terms = key_terms(row["text_content"], limit=6)
        topic = terms[0] if terms else row["modality"]
        grouped[topic].append(row)

    ranked_groups = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:max_spheres]
    active_ids: set[str] = set()
    now = utc_now()
    for topic, topic_rows in ranked_groups:
        sphere_id = stable_id("biorag-sphere", topic, length=24)
        active_ids.add(sphere_id)
        support_ids = [row["chunk_id"] for row in topic_rows[:220]]
        document_ids = sorted({row["file_id"] for row in topic_rows})
        topic_terms = key_terms(" ".join(row["text_content"] for row in topic_rows[:24]), limit=18)
        center = centroid(topic_rows)
        geometry = sphere_geometry(topic_rows, center)
        conn.execute(
            """
            INSERT INTO sphere_summaries (
                sphere_id, sphere_name, summary_text, centroid_vector_json, topic_terms_json,
                support_chunk_ids_json, document_ids_json, chunk_count, radius, shell_mean,
                shell_std, density, dimensionality, strength, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?)
            ON CONFLICT(sphere_id) DO UPDATE SET
                sphere_name = excluded.sphere_name,
                summary_text = excluded.summary_text,
                centroid_vector_json = excluded.centroid_vector_json,
                topic_terms_json = excluded.topic_terms_json,
                support_chunk_ids_json = excluded.support_chunk_ids_json,
                document_ids_json = excluded.document_ids_json,
                chunk_count = excluded.chunk_count,
                radius = excluded.radius,
                shell_mean = excluded.shell_mean,
                shell_std = excluded.shell_std,
                density = excluded.density,
                dimensionality = excluded.dimensionality,
                updated_at = excluded.updated_at
            """,
            (
                sphere_id,
                topic,
                summary_for_rows(topic_rows, title=f"Topic {topic}"),
                dump_vector(center),
                json.dumps(topic_terms),
                json.dumps(support_ids),
                json.dumps(document_ids),
                len(topic_rows),
                geometry["radius"],
                geometry["shell_mean"],
                geometry["shell_std"],
                geometry["density"],
                geometry["dimensionality"],
                now,
                now,
            ),
        )


def build_hex_neighbors(
    conn: sqlite3.Connection,
    focus_child_ids: list[str] | None = None,
    *,
    neighbor_count: int = 6,
    max_focus: int = 160,
) -> int:
    rows = list(iter_child_vectors(conn))
    if not rows:
        return 0
    rows_by_id = {row["chunk_id"]: row for row in rows}
    cell_assignments = build_hex_cells(conn, rows)
    focus_ids = focus_child_ids or [row["chunk_id"] for row in rows]
    focus_rows = [rows_by_id[chunk_id] for chunk_id in focus_ids[:max_focus] if chunk_id in rows_by_id]
    now = utc_now()
    link_count = 0

    for row in focus_rows:
        vector = vector_from_row(row)
        neighbors: list[tuple[str, float, str]] = []
        row_terms = set(key_terms(row["text_content"], limit=20))
        candidate_ids = hex_candidate_ids(conn, row["chunk_id"], cell_assignments)
        if len(candidate_ids) < neighbor_count:
            candidate_ids.update(other["chunk_id"] for other in rows)
        for candidate_id in candidate_ids:
            other = rows_by_id.get(candidate_id)
            if other is None:
                continue
            if other["chunk_id"] == row["chunk_id"]:
                continue
            semantic = cosine(vector, vector_from_row(other))
            other_terms = set(key_terms(other["text_content"], limit=20))
            shared = len(row_terms & other_terms) / max(1, min(len(row_terms), len(other_terms)))
            hex_bonus = hex_proximity_bonus(row["chunk_id"], other["chunk_id"], cell_assignments)
            score = max(semantic, shared * 0.9) + hex_bonus
            if score <= 0:
                continue
            hint = "hex_cell" if hex_bonus else "semantic" if semantic >= shared else "shared_terms"
            neighbors.append((other["chunk_id"], score, hint))

        for rank, (neighbor_id, similarity, hint) in enumerate(
            sorted(neighbors, key=lambda item: item[1], reverse=True)[:neighbor_count],
            start=1,
        ):
            conn.execute(
                """
                INSERT INTO hex_neighbors (
                    chunk_id, neighbor_id, rank, similarity, relation_hint, index_version, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, neighbor_id) DO UPDATE SET
                    rank = excluded.rank,
                    similarity = excluded.similarity,
                    relation_hint = excluded.relation_hint,
                    index_version = excluded.index_version,
                    updated_at = excluded.updated_at
                """,
                (row["chunk_id"], neighbor_id, rank, similarity, hint, BIORAG_INDEX_VERSION, now),
            )
            source, target = sorted([row["chunk_id"], neighbor_id])
            conn.execute(
                """
                INSERT OR IGNORE INTO cross_modal_links (
                    source_chunk_id, target_chunk_id, confidence_score, link_type, created_at
                )
                VALUES (?, ?, ?, 'hex', ?)
                """,
                (source, target, min(similarity, 0.99), now),
            )
            link_count += 1
    return link_count


def build_hex_cells(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[str, tuple[str, int, int]]:
    now = utc_now()
    grouped: dict[tuple[int, int], list[sqlite3.Row]] = defaultdict(list)
    projections: dict[str, tuple[float, float, int, int]] = {}
    for row in rows:
        x, y = project_vector_2d(vector_from_row(row))
        q, r = point_to_hex(x, y, HEX_CELL_RADIUS)
        grouped[(q, r)].append(row)
        projections[row["chunk_id"]] = (x, y, q, r)

    assignments: dict[str, tuple[str, int, int]] = {}
    for (q, r), cell_rows in grouped.items():
        cell_id = stable_id("biorag-hex-cell", BIORAG_INDEX_VERSION, q, r, length=24)
        center_x, center_y = hex_to_point(q, r, HEX_CELL_RADIUS)
        chunk_ids = [row["chunk_id"] for row in cell_rows]
        conn.execute(
            """
            INSERT INTO biorag_hex_cells (
                cell_id, q, r, center_x, center_y, radius, occupant_count,
                centroid_vector_json, chunk_ids_json, index_version, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(q, r, index_version) DO UPDATE SET
                cell_id = excluded.cell_id,
                center_x = excluded.center_x,
                center_y = excluded.center_y,
                radius = excluded.radius,
                occupant_count = excluded.occupant_count,
                centroid_vector_json = excluded.centroid_vector_json,
                chunk_ids_json = excluded.chunk_ids_json,
                updated_at = excluded.updated_at
            """,
            (
                cell_id,
                q,
                r,
                center_x,
                center_y,
                HEX_CELL_RADIUS,
                len(cell_rows),
                dump_vector(centroid(cell_rows)),
                json.dumps(chunk_ids),
                BIORAG_INDEX_VERSION,
                now,
            ),
        )
        for row in cell_rows:
            x, y, projected_q, projected_r = projections[row["chunk_id"]]
            ring = max(abs(projected_q), abs(projected_r), abs(-projected_q - projected_r))
            conn.execute(
                """
                INSERT INTO biorag_chunk_cells (
                    chunk_id, cell_id, q, r, projected_x, projected_y, ring, index_version, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    cell_id = excluded.cell_id,
                    q = excluded.q,
                    r = excluded.r,
                    projected_x = excluded.projected_x,
                    projected_y = excluded.projected_y,
                    ring = excluded.ring,
                    index_version = excluded.index_version,
                    updated_at = excluded.updated_at
                """,
                (row["chunk_id"], cell_id, projected_q, projected_r, x, y, ring, BIORAG_INDEX_VERSION, now),
            )
            assignments[row["chunk_id"]] = (cell_id, projected_q, projected_r)
    return assignments


def hex_candidate_ids(
    conn: sqlite3.Connection,
    chunk_id: str,
    assignments: dict[str, tuple[str, int, int]],
    *,
    radius: int = 1,
) -> set[str]:
    assignment = assignments.get(chunk_id)
    if assignment is None:
        return set()
    _cell_id, q, r = assignment
    coords = {(q, r)}
    frontier = {(q, r)}
    for _step in range(radius):
        next_frontier: set[tuple[int, int]] = set()
        for current_q, current_r in frontier:
            for dq, dr in HEX_DIRECTIONS:
                coord = (current_q + dq, current_r + dr)
                if coord not in coords:
                    coords.add(coord)
                    next_frontier.add(coord)
        frontier = next_frontier

    ids: set[str] = set()
    for cell_q, cell_r in coords:
        row = conn.execute(
            """
            SELECT chunk_ids_json
            FROM biorag_hex_cells
            WHERE q = ? AND r = ? AND index_version = ?
            """,
            (cell_q, cell_r, BIORAG_INDEX_VERSION),
        ).fetchone()
        if row:
            ids.update(json_list(row["chunk_ids_json"]))
    return ids


def hex_proximity_bonus(
    left_id: str,
    right_id: str,
    assignments: dict[str, tuple[str, int, int]],
) -> float:
    left = assignments.get(left_id)
    right = assignments.get(right_id)
    if left is None or right is None:
        return 0.0
    _left_cell, left_q, left_r = left
    _right_cell, right_q, right_r = right
    distance = hex_distance(left_q, left_r, right_q, right_r)
    if distance == 0:
        return 0.08
    if distance == 1:
        return 0.035
    return 0.0


def hex_distance(left_q: int, left_r: int, right_q: int, right_r: int) -> int:
    left_s = -left_q - left_r
    right_s = -right_q - right_r
    return max(abs(left_q - right_q), abs(left_r - right_r), abs(left_s - right_s))


def project_vector_2d(vector: dict[int, float]) -> tuple[float, float]:
    x = 0.0
    y = 0.0
    for index, value in vector.items():
        x += value * projection_weight(index, "x")
        y += value * projection_weight(index, "y")
    return x, y


def projection_weight(index: int, salt: str) -> float:
    digest = hashlib.blake2b(f"{salt}:{index}".encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "big") / float(2**64 - 1)
    return raw * 2.0 - 1.0


def point_to_hex(x: float, y: float, size: float) -> tuple[int, int]:
    q = (math.sqrt(3.0) / 3.0 * x - y / 3.0) / size
    r = (2.0 / 3.0 * y) / size
    return hex_round(q, r)


def hex_round(q: float, r: float) -> tuple[int, int]:
    s = -q - r
    rounded_q = round(q)
    rounded_r = round(r)
    rounded_s = round(s)
    q_diff = abs(rounded_q - q)
    r_diff = abs(rounded_r - r)
    s_diff = abs(rounded_s - s)
    if q_diff > r_diff and q_diff > s_diff:
        rounded_q = -rounded_r - rounded_s
    elif r_diff > s_diff:
        rounded_r = -rounded_q - rounded_s
    return int(rounded_q), int(rounded_r)


def hex_to_point(q: int, r: int, size: float) -> tuple[float, float]:
    x = size * math.sqrt(3.0) * (q + r / 2.0)
    y = size * 1.5 * r
    return x, y


def sphere_route(conn: sqlite3.Connection, query: str, *, limit: int) -> list[tuple[str, float]]:
    query_vector = embed_text(query)
    query_tokens = tokenize(query)
    candidates: list[tuple[str, float]] = []
    for row in conn.execute("SELECT * FROM sphere_summaries ORDER BY strength DESC, chunk_count DESC LIMIT 80"):
        terms = " ".join(json_list(row["topic_terms_json"]))
        center = load_vector(row["centroid_vector_json"])
        query_distance = vector_distance(query_vector, center)
        radius = float(row["radius"] or 0.0)
        shell_std = max(float(row["shell_std"] or 0.0), 0.04)
        shell_alignment = math.exp(-abs(query_distance - radius) / shell_std) if radius else 0.0
        score = (
            cosine(query_vector, center) * 0.64
            + lexical_overlap(query_tokens, f"{row['summary_text']} {terms}") * 0.22
            + shell_alignment * 0.08
            + min(float(row["density"] or 0.0), 1.0) * 0.025
            + min(float(row["strength"] or 1.0), 3.0) * 0.015
        )
        if score <= 0:
            continue
        for index, chunk_id in enumerate(json_list(row["support_chunk_ids_json"])[:10]):
            candidates.append((chunk_id, score / (index + 1)))
    return sorted(candidates, key=lambda item: item[1], reverse=True)[:limit]


def tree_route(
    conn: sqlite3.Connection,
    query: str,
    sphere_candidates: list[tuple[str, float]],
    *,
    limit: int,
) -> list[tuple[str, float]]:
    query_vector = embed_text(query)
    query_tokens = tokenize(query)
    sphere_support = {chunk_id for chunk_id, _score in sphere_candidates}
    candidates: list[tuple[str, float]] = []
    for row in conn.execute("SELECT * FROM biorag_tree_nodes ORDER BY level ASC LIMIT 240"):
        support = json_list(row["support_chunk_ids_json"])
        overlap_bonus = 0.05 if sphere_support & set(support) else 0.0
        score = (
            cosine(query_vector, load_vector(row["centroid_vector_json"])) * 0.62
            + lexical_overlap(query_tokens, row["summary_text"]) * 0.28
            + overlap_bonus
            - int(row["level"]) * 0.01
        )
        if score <= 0:
            continue
        for index, chunk_id in enumerate(support[:8]):
            candidates.append((chunk_id, score / (index + 1)))
    return sorted(candidates, key=lambda item: item[1], reverse=True)[:limit]


def hex_expand(
    conn: sqlite3.Connection,
    seeds: list[tuple[str, float]],
    *,
    frontier: int,
    limit: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for chunk_id, seed_score in seeds[:frontier]:
        scores[chunk_id] = max(scores.get(chunk_id, 0.0), seed_score)
        rows = conn.execute(
            """
            SELECT neighbor_id, similarity
            FROM hex_neighbors
            WHERE chunk_id = ? AND index_version = ?
            ORDER BY rank ASC
            LIMIT 12
            """,
            (chunk_id, BIORAG_INDEX_VERSION),
        ).fetchall()
        for row in rows:
            scores[row["neighbor_id"]] = max(
                scores.get(row["neighbor_id"], 0.0),
                seed_score * float(row["similarity"]) * 0.72,
            )
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def propagate_web_signal(
    conn: sqlite3.Connection,
    seeds: list[tuple[str, float]],
    *,
    iterations: int,
    max_candidates: int,
) -> list[tuple[str, float]]:
    signal: dict[str, float] = {}
    frontier: dict[str, float] = {}
    for chunk_id, score in seeds[:max_candidates]:
        signal[chunk_id] = max(signal.get(chunk_id, 0.0), score)
        if len(frontier) < 56:
            frontier[chunk_id] = max(frontier.get(chunk_id, 0.0), score)

    for _iteration in range(iterations):
        if not frontier:
            break
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for link in iter_links_for_chunks(conn, frontier.keys()):
            source = link["source_chunk_id"]
            target = link["target_chunk_id"]
            link_type = str(link["link_type"] or "web")
            confidence = max(float(link["confidence_score"] or 0.0), 0.08)
            adaptive = edge_weight(conn, source, target, link_type)
            weight = confidence * adaptive
            adjacency[source].append((target, weight))
            adjacency[target].append((source, weight))

        next_frontier: dict[str, float] = {}
        for source, neighbors in adjacency.items():
            source_signal = frontier.get(source, 0.0)
            if source_signal <= 0:
                continue
            total_weight = sum(weight for _target, weight in neighbors) or 1.0
            for target, weight in neighbors:
                chunk = get_chunk(conn, target)
                if chunk is None or chunk["chunk_kind"] != "child":
                    continue
                propagated = source_signal * WEB_DAMPING * (weight / total_weight)
                if propagated <= 0.0001:
                    continue
                if propagated > signal.get(target, 0.0):
                    signal[target] = propagated
                    next_frontier[target] = propagated

        frontier = dict(sorted(next_frontier.items(), key=lambda item: item[1], reverse=True)[:56])
        if len(signal) >= max_candidates:
            break
    return sorted(signal.items(), key=lambda item: item[1], reverse=True)[:max_candidates]


def edge_weight(conn: sqlite3.Connection, source_id: str, target_id: str, edge_type: str) -> float:
    edge_key = stable_edge_key(source_id, target_id, edge_type)
    row = conn.execute(
        "SELECT weight, pheromone, last_decayed_at FROM biorag_edge_weights WHERE edge_key = ?",
        (edge_key,),
    ).fetchone()
    if row is None:
        return 1.0
    return decayed_weight(float(row["weight"] or 1.0), str(row["last_decayed_at"] or utc_now()))


def update_hebbian_edges(conn: sqlite3.Connection, selected: list[tuple[str, float]]) -> None:
    selected_ids = [chunk_id for chunk_id, _score in selected]
    if len(selected_ids) < 2:
        decay_stale_edges(conn)
        return
    now = utc_now()
    decay_stale_edges(conn, now=now)
    for index, source_id in enumerate(selected_ids):
        for target_id in selected_ids[index + 1 :]:
            link_type = inferred_edge_type(conn, source_id, target_id)
            edge_key = stable_edge_key(source_id, target_id, link_type)
            existing = conn.execute(
                "SELECT weight, pheromone, coactivation_count FROM biorag_edge_weights WHERE edge_key = ?",
                (edge_key,),
            ).fetchone()
            old_weight = float(existing["weight"] or 1.0) if existing else 1.0
            old_pheromone = float(existing["pheromone"] or 1.0) if existing else 1.0
            coactivation = int(existing["coactivation_count"] or 0) if existing else 0
            next_weight = min(5.0, old_weight + HEBBIAN_LEARNING_RATE)
            next_pheromone = min(5.0, old_pheromone * 0.98 + HEBBIAN_LEARNING_RATE)
            conn.execute(
                """
                INSERT INTO biorag_edge_weights (
                    edge_key, source_id, target_id, edge_type, weight, pheromone,
                    coactivation_count, last_reinforced_at, last_decayed_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_key) DO UPDATE SET
                    weight = excluded.weight,
                    pheromone = excluded.pheromone,
                    coactivation_count = excluded.coactivation_count,
                    last_reinforced_at = excluded.last_reinforced_at,
                    last_decayed_at = excluded.last_decayed_at
                """,
                (
                    edge_key,
                    min(source_id, target_id),
                    max(source_id, target_id),
                    link_type,
                    next_weight,
                    next_pheromone,
                    coactivation + 1,
                    now,
                    now,
                    now,
                ),
            )


def decay_stale_edges(conn: sqlite3.Connection, *, now: str | None = None, limit: int = 400) -> None:
    current_time = now or utc_now()
    rows = conn.execute(
        """
        SELECT edge_key, weight, pheromone, last_decayed_at
        FROM biorag_edge_weights
        ORDER BY last_decayed_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        decayed = decayed_weight(float(row["weight"] or 1.0), str(row["last_decayed_at"] or current_time), now=current_time)
        pheromone = max(0.1, float(row["pheromone"] or 1.0) * 0.997)
        conn.execute(
            """
            UPDATE biorag_edge_weights
            SET weight = ?, pheromone = ?, last_decayed_at = ?
            WHERE edge_key = ?
            """,
            (decayed, pheromone, current_time, row["edge_key"]),
        )


def decayed_weight(weight: float, last_decayed_at: str, *, now: str | None = None) -> float:
    try:
        from datetime import datetime

        previous = datetime.fromisoformat(last_decayed_at)
        current = datetime.fromisoformat(now or utc_now())
        age_days = max((current - previous).total_seconds() / 86400.0, 0.0)
    except ValueError:
        age_days = 0.0
    decay = 0.5 ** (age_days / HEBBIAN_DECAY_HALF_LIFE_DAYS)
    return max(0.1, weight * decay)


def inferred_edge_type(conn: sqlite3.Connection, source_id: str, target_id: str) -> str:
    source, target = sorted([source_id, target_id])
    row = conn.execute(
        """
        SELECT link_type
        FROM cross_modal_links
        WHERE source_chunk_id = ? AND target_chunk_id = ?
        ORDER BY confidence_score DESC
        LIMIT 1
        """,
        (source, target),
    ).fetchone()
    return str(row["link_type"]) if row else "coactivated"


def stable_edge_key(source_id: str, target_id: str, edge_type: str) -> str:
    source, target = sorted([source_id, target_id])
    return stable_id("biorag-edge", edge_type, source, target, length=24)


def apply_adaptive_growth(
    conn: sqlite3.Connection,
    query: str,
    candidates: list[tuple[str, float]],
    paths: dict[str, set[str]],
) -> list[tuple[str, float]]:
    signature = query_signature(query)
    boosted: list[tuple[str, float]] = []
    for chunk_id, score in candidates:
        rows = conn.execute(
            """
            SELECT AVG(strength) AS strength
            FROM adaptive_path_stats
            WHERE target_id = ? AND query_signature = ?
            """,
            (chunk_id, signature),
        ).fetchone()
        strength = float(rows["strength"] or 1.0) if rows else 1.0
        layer_bonus = 0.004 * len(paths.get(chunk_id, ()))
        boosted.append((chunk_id, score + min(strength, 3.0) * 0.018 + layer_bonus))
    return sorted(boosted, key=lambda item: item[1], reverse=True)


def update_adaptive_paths(
    conn: sqlite3.Connection,
    query: str,
    paths: dict[str, set[str]],
    selected: list[tuple[str, float]],
) -> None:
    signature = query_signature(query)
    selected_ids = {chunk_id for chunk_id, _score in selected}
    now = utc_now()
    for chunk_id, layers in paths.items():
        for layer in layers:
            was_selected = chunk_id in selected_ids
            path_id = stable_id("biorag-path", signature, layer, chunk_id, length=24)
            current = conn.execute(
                "SELECT strength FROM adaptive_path_stats WHERE path_id = ?",
                (path_id,),
            ).fetchone()
            old_strength = float(current["strength"]) if current else 1.0
            next_strength = min(3.0, old_strength + 0.12) if was_selected else max(0.25, old_strength * 0.992)
            conn.execute(
                """
                INSERT INTO adaptive_path_stats (
                    path_id, layer, source_id, target_id, query_signature, impressions,
                    selected_count, verified_count, rejected_count, strength, last_used_at, created_at
                )
                VALUES (?, ?, NULL, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path_id) DO UPDATE SET
                    impressions = impressions + 1,
                    selected_count = selected_count + excluded.selected_count,
                    verified_count = verified_count + excluded.verified_count,
                    rejected_count = rejected_count + excluded.rejected_count,
                    strength = excluded.strength,
                    last_used_at = excluded.last_used_at
                """,
                (
                    path_id,
                    layer,
                    chunk_id,
                    signature,
                    1 if was_selected else 0,
                    1 if was_selected else 0,
                    0 if was_selected else 1,
                    next_strength,
                    now,
                    now,
                ),
            )


def record_retrieval_run(
    conn: sqlite3.Connection,
    query_id: str,
    query: str,
    budget: EnergyBudget,
    paths: dict[str, set[str]],
    selected: list[tuple[str, float]],
) -> None:
    layer_counts = Counter(layer for layers in paths.values() for layer in layers)
    trace = {
        "layers": dict(layer_counts),
        "selected": [
            {"chunk_id": chunk_id, "score": score, "layers": sorted(paths.get(chunk_id, ()))}
            for chunk_id, score in selected
        ],
    }
    conn.execute(
        """
        INSERT OR REPLACE INTO biorag_retrieval_runs (
            query_id, query_text, complexity, budget_json, trace_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            query,
            budget.complexity,
            json.dumps(asdict(budget), sort_keys=True),
            json.dumps(trace, sort_keys=True),
            utc_now(),
        ),
    )


def upsert_tree_node(
    conn: sqlite3.Connection,
    *,
    node_id: str,
    parent_node_id: str | None,
    source_id: str,
    node_type: str,
    level: int,
    summary_text: str,
    vector: dict[int, float],
    support_chunk_ids: list[str],
    document_ids: list[str],
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO biorag_tree_nodes (
            node_id, parent_node_id, source_id, node_type, level, summary_text,
            centroid_vector_json, support_chunk_ids_json, document_ids_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            parent_node_id = excluded.parent_node_id,
            source_id = excluded.source_id,
            node_type = excluded.node_type,
            level = excluded.level,
            summary_text = excluded.summary_text,
            centroid_vector_json = excluded.centroid_vector_json,
            support_chunk_ids_json = excluded.support_chunk_ids_json,
            document_ids_json = excluded.document_ids_json,
            updated_at = excluded.updated_at
        """,
        (
            node_id,
            parent_node_id,
            source_id,
            node_type,
            level,
            summary_text,
            dump_vector(vector),
            json.dumps(support_chunk_ids),
            json.dumps(document_ids),
            now,
            now,
        ),
    )


def summary_for_rows(rows: list[sqlite3.Row], *, title: str) -> str:
    terms = key_terms(" ".join(row["text_content"] for row in rows[:32]), limit=12)
    snippets = []
    for row in rows[:3]:
        text = " ".join(str(row["text_content"]).split())
        if text:
            snippets.append(text[:180])
    return f"{title}: terms {', '.join(terms)}. " + " ".join(snippets)


def centroid(rows: list[sqlite3.Row]) -> dict[int, float]:
    accum: dict[int, float] = {}
    for row in rows:
        for index, value in vector_from_row(row).items():
            accum[index] = accum.get(index, 0.0) + value
    if not accum:
        return {}
    norm = math.sqrt(sum(value * value for value in accum.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in accum.items()}


def sphere_geometry(rows: list[sqlite3.Row], center: dict[int, float]) -> dict[str, float | int]:
    distances = [vector_distance(vector_from_row(row), center) for row in rows]
    if not distances:
        return {"radius": 0.0, "shell_mean": 0.0, "shell_std": 0.0, "density": 0.0, "dimensionality": 0}
    radius = max(distances)
    shell_mean = sum(distances) / len(distances)
    variance = sum((distance - shell_mean) ** 2 for distance in distances) / max(len(distances), 1)
    shell_std = math.sqrt(variance)
    dimensionality = len(center)
    capped_dimension = max(1, min(dimensionality, 6))
    density = len(rows) / max(radius**capped_dimension, 0.001)
    normalized_density = min(density / 100.0, 1.0)
    return {
        "radius": round(radius, 6),
        "shell_mean": round(shell_mean, 6),
        "shell_std": round(shell_std, 6),
        "density": round(normalized_density, 6),
        "dimensionality": dimensionality,
    }


def vector_distance(left: dict[int, float], right: dict[int, float]) -> float:
    if not left and not right:
        return 0.0
    indexes = set(left) | set(right)
    return math.sqrt(sum((left.get(index, 0.0) - right.get(index, 0.0)) ** 2 for index in indexes))


def mark_paths(paths: dict[str, set[str]], candidates: list[tuple[str, float]], layer: str) -> None:
    for chunk_id, _score in candidates:
        paths[chunk_id].add(layer)


def key_terms(text: str, *, limit: int = 24) -> list[str]:
    counts = Counter(
        token
        for token in tokenize(text)
        if len(token) >= 4 and token not in STOP_TERMS
    )
    return [token for token, _count in counts.most_common(limit)]


def query_signature(query: str) -> str:
    terms = key_terms(query, limit=8)
    return "|".join(terms) or stable_id("query-signature", query, length=16)


def json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def linked_chunks(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for link in iter_links_for_chunks(conn, chunk_ids):
        for chunk_id in (link["source_chunk_id"], link["target_chunk_id"]):
            if chunk_id not in chunk_ids:
                scores[chunk_id] = max(scores.get(chunk_id, 0.0), float(link["confidence_score"] or 0.0))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def chunk_exists(conn: sqlite3.Connection, chunk_id: str) -> bool:
    return get_chunk(conn, chunk_id) is not None
