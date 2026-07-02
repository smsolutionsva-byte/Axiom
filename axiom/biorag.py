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
from .embeddings import cosine, dump_vector, embed_text, lexical_overlap, load_vector, normalize_token, tokenize
from .retrieval import (
    SearchHit,
    coverage_select,
    dense_search,
    evidence_roles_for_source,
    evidence_search_text,
    exact_identifier_overlap,
    make_hit,
    reciprocal_rank_fusion,
    requested_evidence_roles,
    rerank,
)


BIORAG_INDEX_VERSION = "hiverag-v0.5"
HEX_CELL_RADIUS = 0.24
HEX_DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
SPHERE_HONEYCOMB_RINGS = 2
HEX_EXPANSION_RINGS = 2
WEB_DAMPING = 0.64
HEBBIAN_LEARNING_RATE = 0.18
HEBBIAN_DECAY_HALF_LIFE_DAYS = 21.0
BEE_MAX_SUPPORT = 96
BEE_MAX_KEY_TERMS = 128
BEE_MAX_QUERY_FACETS = 4
BEE_MAX_PER_FACET = 3
BEE_ACTIVATION_THRESHOLD = 0.12
BEE_ROLE_ONLY_THRESHOLD = 0.22
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


@dataclass(frozen=True)
class BeeActivation:
    bee_id: str
    bee_name: str
    facet_index: int
    score: float
    matched_terms: tuple[str, ...]
    matched_roles: tuple[str, ...]


def refresh_biorag_index(conn: sqlite3.Connection, focus_child_ids: list[str] | None = None) -> int:
    build_tree_index(conn)
    build_sphere_index(conn)
    build_sphere_honeycomb(conn)
    hex_links = build_hex_neighbors(conn, focus_child_ids)
    build_bee_index(conn)
    return hex_links


def ensure_biorag_index(conn: sqlite3.Connection) -> None:
    child_count = conn.execute(
        "SELECT COUNT(*) AS count FROM content_chunks WHERE chunk_kind = 'child'"
    ).fetchone()["count"]
    if not child_count:
        return
    sphere_count = conn.execute("SELECT COUNT(*) AS count FROM sphere_summaries").fetchone()["count"]
    sphere_neighbor_count = conn.execute(
        "SELECT COUNT(*) AS count FROM sphere_neighbors WHERE index_version = ?",
        (BIORAG_INDEX_VERSION,),
    ).fetchone()["count"]
    tree_count = conn.execute("SELECT COUNT(*) AS count FROM biorag_tree_nodes").fetchone()["count"]
    hex_count = conn.execute(
        "SELECT COUNT(*) AS count FROM hex_neighbors WHERE index_version = ?",
        (BIORAG_INDEX_VERSION,),
    ).fetchone()["count"]
    cell_count = conn.execute(
        "SELECT COUNT(*) AS count FROM biorag_chunk_cells WHERE index_version = ?",
        (BIORAG_INDEX_VERSION,),
    ).fetchone()["count"]
    bee_count = conn.execute(
        "SELECT COUNT(*) AS count FROM biorag_bees WHERE index_version = ?",
        (BIORAG_INDEX_VERSION,),
    ).fetchone()["count"]
    sphere_neighbors_ready = sphere_count <= 1 or sphere_neighbor_count > 0
    if (
        sphere_count
        and sphere_neighbors_ready
        and tree_count
        and hex_count
        and bee_count
        and cell_count >= min(child_count, 160)
    ):
        return
    refresh_biorag_index(conn)
    conn.commit()


def biorag_search(conn: sqlite3.Connection, query: str, *, top_k: int = 5) -> tuple[str, list[SearchHit]]:
    ensure_biorag_index(conn)
    budget = energy_budget(query, top_k=top_k)
    paths: dict[str, set[str]] = defaultdict(set)

    bee_candidates, bee_trace = bee_route(conn, query, top_k=top_k, max_candidates=budget.max_candidates)
    bee_candidate_ids = [chunk_id for chunk_id, _score in bee_candidates]
    facet_count = len(bee_trace.get("facets", [])) if isinstance(bee_trace.get("facets"), list) else 0
    covered_facets = int(bee_trace.get("covered_facets", 0))
    bee_territory_ready = bool(bee_candidate_ids and covered_facets >= max(facet_count, 1))
    bee_source_fast_path = bee_territory_ready and is_source_record_query(query)
    mark_paths(paths, bee_candidates, "bee_router")

    if bee_candidate_ids and covered_facets > 0:
        dense = dense_search_candidates(conn, query, bee_candidate_ids, limit=budget.seed_limit)
        sparse = lexical_search_candidates(conn, query, bee_candidate_ids, limit=budget.seed_limit)
    else:
        dense = dense_search(conn, query, limit=budget.seed_limit)
        sparse = lexical_search(conn, query, limit=budget.seed_limit)
    mark_paths(paths, dense, "vector")
    mark_paths(paths, sparse, "lexical")

    if bee_territory_ready:
        sphere: list[tuple[str, float]] = []
        tree: list[tuple[str, float]] = []
    else:
        sphere = sphere_route(conn, query, limit=budget.sphere_limit)
        tree = tree_route(conn, query, sphere, limit=budget.tree_limit)
    seed_fused = reciprocal_rank_fusion(
        [bee_candidates, dense, sparse, sphere, tree],
        weights=[1.2, 1.1, 1.15, 0.85, 0.9],
    )
    mark_paths(paths, sphere, "sphere")
    mark_paths(paths, tree, "tree")

    seed_window = max(top_k * 5, 24) if bee_source_fast_path else max(top_k * 8, 32) if bee_territory_ready else max(top_k * 10, 50)
    hex_frontier = 0 if bee_source_fast_path else min(budget.hex_frontier, max(top_k * 2, 8)) if bee_territory_ready else budget.hex_frontier
    hex_limit = 0 if bee_source_fast_path else min(max(top_k * 12, 50), max(len(bee_candidate_ids) * 3, top_k * 12, 36)) if bee_territory_ready else max(top_k * 12, 50)
    web_iterations = 0 if bee_source_fast_path else max(1, min(budget.graph_depth, 2)) if bee_territory_ready else budget.graph_depth + 1
    web_max_candidates = (
        max(top_k * 8, 32)
        if bee_source_fast_path
        else min(budget.max_candidates, max(len(bee_candidate_ids) * 3, top_k * 18, 48))
        if bee_territory_ready
        else budget.max_candidates
    )

    seeded = rerank(conn, query, seed_fused[:seed_window])
    hexed = [] if bee_source_fast_path else hex_expand(conn, seeded, frontier=hex_frontier, limit=hex_limit)
    mark_paths(paths, hexed, "hex")

    webbed = [] if bee_source_fast_path else propagate_web_signal(
        conn,
        seeded + hexed,
        iterations=web_iterations,
        max_candidates=web_max_candidates,
    )
    mark_paths(paths, webbed, "spider_signal")

    fused = reciprocal_rank_fusion(
        [seeded, hexed, webbed],
        weights=[1.25, 0.9, 0.95],
    )
    boosted = apply_adaptive_growth(conn, query, fused, paths)
    reranked = precision_boost_candidates(conn, query, rerank(conn, query, boosted[:web_max_candidates]))
    territory_rescue_ids = (
        sorted({*bee_candidate_ids, *(chunk_id for chunk_id, _score in reranked[:web_max_candidates])})
        if bee_territory_ready
        else None
    )
    covered_candidates, rescued_ids = coverage_rescue(
        conn,
        query,
        reranked,
        max_candidates=web_max_candidates,
        allowed_chunk_ids=territory_rescue_ids,
    )
    covered_candidates = precision_boost_candidates(conn, query, covered_candidates)
    for chunk_id in rescued_ids:
        paths[chunk_id].add("coverage_rescue")
    selected = coverage_select(
        conn,
        query,
        covered_candidates,
        top_k=top_k,
        min_relevance=min_relevance_for_query(query),
    )

    if not selected:
        selected = rerank(conn, query, seed_fused[: max(top_k * 4, 20)])[:top_k]

    if is_source_record_query(query):
        selected = filter_source_record_context(conn, query, selected) or selected

    query_id = stable_id("biorag-query", query, *[chunk_id for chunk_id, _ in selected], length=24)
    record_query_context(conn, query_id=query_id, query_text=query, ranked=selected)
    record_retrieval_run(conn, query_id, query, budget, paths, selected, bee_trace=bee_trace)
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
        "sphere_neighbors": f"SELECT COUNT(*) AS count FROM sphere_neighbors WHERE index_version = '{BIORAG_INDEX_VERSION}'",
        "hex_neighbors": f"SELECT COUNT(*) AS count FROM hex_neighbors WHERE index_version = '{BIORAG_INDEX_VERSION}'",
        "hex_cells": f"SELECT COUNT(*) AS count FROM biorag_hex_cells WHERE index_version = '{BIORAG_INDEX_VERSION}'",
        "chunk_cells": f"SELECT COUNT(*) AS count FROM biorag_chunk_cells WHERE index_version = '{BIORAG_INDEX_VERSION}'",
        "bees": f"SELECT COUNT(*) AS count FROM biorag_bees WHERE index_version = '{BIORAG_INDEX_VERSION}'",
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
            and (counts["spheres"] <= 1 or counts["sphere_neighbors"])
            and counts["tree_nodes"]
            and counts["hex_neighbors"]
            and counts["hex_cells"]
            and counts["chunk_cells"]
            and counts["bees"]
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


def min_relevance_for_query(query: str) -> float:
    lowered = query.lower()
    if any(phrase in lowered for phrase in ("which document", "what document", "which source", "what source")):
        return 0.34
    return 0.0


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
    if active_ids:
        placeholders = ",".join("?" for _ in active_ids)
        conn.execute(f"DELETE FROM sphere_summaries WHERE sphere_id NOT IN ({placeholders})", tuple(sorted(active_ids)))


def build_sphere_honeycomb(
    conn: sqlite3.Connection,
    *,
    max_rings: int = SPHERE_HONEYCOMB_RINGS,
    min_similarity: float = 0.08,
) -> int:
    spheres = conn.execute(
        """
        SELECT *
        FROM sphere_summaries
        ORDER BY strength DESC, chunk_count DESC, sphere_name ASC
        """
    ).fetchall()
    conn.execute("DELETE FROM sphere_neighbors WHERE index_version = ?", (BIORAG_INDEX_VERSION,))
    if len(spheres) < 2:
        return 0

    now = utc_now()
    max_neighbors = 6 * sum(range(1, max_rings + 1))
    inserted = 0
    for sphere in spheres:
        scored: list[tuple[sqlite3.Row, float, str]] = []
        for candidate in spheres:
            if candidate["sphere_id"] == sphere["sphere_id"]:
                continue
            similarity, hint = sphere_similarity(sphere, candidate)
            if similarity < min_similarity:
                continue
            scored.append((candidate, similarity, hint))

        for rank, (candidate, similarity, hint) in enumerate(
            sorted(scored, key=lambda item: item[1], reverse=True)[:max_neighbors],
            start=1,
        ):
            ring = honeycomb_ring_for_rank(rank)
            chain_group = stable_id(
                "biorag-sphere-chain",
                BIORAG_INDEX_VERSION,
                sphere["sphere_id"],
                ring,
                length=24,
            )
            conn.execute(
                """
                INSERT INTO sphere_neighbors (
                    sphere_id, neighbor_sphere_id, rank, ring, similarity,
                    relation_hint, chain_group, index_version, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sphere_id, neighbor_sphere_id) DO UPDATE SET
                    rank = excluded.rank,
                    ring = excluded.ring,
                    similarity = excluded.similarity,
                    relation_hint = excluded.relation_hint,
                    chain_group = excluded.chain_group,
                    index_version = excluded.index_version,
                    updated_at = excluded.updated_at
                """,
                (
                    sphere["sphere_id"],
                    candidate["sphere_id"],
                    rank,
                    ring,
                    similarity,
                    hint,
                    chain_group,
                    BIORAG_INDEX_VERSION,
                    now,
                ),
            )
            insert_sphere_bridge_link(conn, sphere, candidate, similarity, now)
            inserted += 1
    return inserted


def sphere_similarity(left: sqlite3.Row, right: sqlite3.Row) -> tuple[float, str]:
    left_terms = set(json_list(left["topic_terms_json"]))
    right_terms = set(json_list(right["topic_terms_json"]))
    term_overlap = len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))
    left_docs = set(json_list(left["document_ids_json"]))
    right_docs = set(json_list(right["document_ids_json"]))
    document_overlap = len(left_docs & right_docs) / max(1, min(len(left_docs), len(right_docs)))
    semantic = max(cosine(load_vector(left["centroid_vector_json"]), load_vector(right["centroid_vector_json"])), 0.0)
    score = semantic * 0.62 + term_overlap * 0.3 + document_overlap * 0.08
    if term_overlap >= semantic and term_overlap >= document_overlap:
        hint = "keyword_chain"
    elif document_overlap >= semantic:
        hint = "document_bridge"
    else:
        hint = "semantic_chain"
    return round(score, 6), hint


def honeycomb_ring_for_rank(rank: int) -> int:
    capacity = 0
    ring = 1
    while True:
        capacity += 6 * ring
        if rank <= capacity:
            return ring
        ring += 1


def insert_sphere_bridge_link(
    conn: sqlite3.Connection,
    left: sqlite3.Row,
    right: sqlite3.Row,
    similarity: float,
    now: str,
) -> None:
    left_support = json_list(left["support_chunk_ids_json"])
    right_support = json_list(right["support_chunk_ids_json"])
    if not left_support or not right_support:
        return
    source, target = sorted([left_support[0], right_support[0]])
    if source == target:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO cross_modal_links (
            source_chunk_id, target_chunk_id, confidence_score, link_type, created_at
        )
        VALUES (?, ?, ?, 'sphere', ?)
        """,
        (source, target, min(max(similarity, 0.08), 0.99), now),
    )


def build_bee_index(conn: sqlite3.Connection) -> int:
    spheres = conn.execute(
        """
        SELECT *
        FROM sphere_summaries
        ORDER BY strength DESC, chunk_count DESC, sphere_name ASC
        """
    ).fetchall()
    conn.execute("DELETE FROM biorag_bees WHERE index_version = ?", (BIORAG_INDEX_VERSION,))
    if not spheres:
        return 0

    now = utc_now()
    sphere_to_bee = {
        row["sphere_id"]: stable_id("hiverag-bee", BIORAG_INDEX_VERSION, row["sphere_id"], length=24)
        for row in spheres
    }
    inserted = 0
    for sphere in spheres:
        support_ids = [chunk_id for chunk_id in json_list(sphere["support_chunk_ids_json"]) if chunk_exists(conn, chunk_id)]
        if not support_ids:
            continue
        support_chunks = [chunk for chunk_id in support_ids[:BEE_MAX_SUPPORT] if (chunk := get_chunk(conn, chunk_id)) is not None]
        source_roles = sorted({role for chunk in support_chunks for role in source_roles_for_chunk(chunk)})
        key_text = " ".join(
            [
                str(sphere["sphere_name"] or ""),
                str(sphere["summary_text"] or ""),
                " ".join(json_list(sphere["topic_terms_json"])),
                " ".join(source_roles),
                " ".join(str(chunk["file_name"] or "") for chunk in support_chunks),
                " ".join(str(chunk["text_content"] or "")[:220] for chunk in support_chunks[:24]),
            ]
        )
        support_identifiers = ordered_unique(
            [
                token
                for chunk in support_chunks
                for token in identifier_terms(f"{chunk['file_name']} {chunk['text_content']}")
            ]
        )
        bee_terms = ordered_unique(
            [
                *support_identifiers,
                *json_list(sphere["topic_terms_json"]),
                *key_terms(key_text, limit=48),
                *source_roles,
            ]
        )[:BEE_MAX_KEY_TERMS]
        neighbor_bees = [
            sphere_to_bee[row["neighbor_sphere_id"]]
            for row in conn.execute(
                """
                SELECT neighbor_sphere_id
                FROM sphere_neighbors
                WHERE sphere_id = ? AND index_version = ?
                ORDER BY ring ASC, rank ASC
                LIMIT 12
                """,
                (sphere["sphere_id"], BIORAG_INDEX_VERSION),
            ).fetchall()
            if row["neighbor_sphere_id"] in sphere_to_bee
        ]
        conn.execute(
            """
            INSERT INTO biorag_bees (
                bee_id, bee_name, sphere_id, key_terms_json, source_roles_json,
                support_chunk_ids_json, neighbor_bee_ids_json, index_version, strength, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bee_id) DO UPDATE SET
                bee_name = excluded.bee_name,
                sphere_id = excluded.sphere_id,
                key_terms_json = excluded.key_terms_json,
                source_roles_json = excluded.source_roles_json,
                support_chunk_ids_json = excluded.support_chunk_ids_json,
                neighbor_bee_ids_json = excluded.neighbor_bee_ids_json,
                index_version = excluded.index_version,
                strength = excluded.strength,
                updated_at = excluded.updated_at
            """,
            (
                sphere_to_bee[sphere["sphere_id"]],
                f"{sphere['sphere_name']}-bee",
                sphere["sphere_id"],
                json.dumps(bee_terms),
                json.dumps(source_roles),
                json.dumps(support_ids[:BEE_MAX_SUPPORT]),
                json.dumps(neighbor_bees),
                BIORAG_INDEX_VERSION,
                float(sphere["strength"] or 1.0),
                now,
            ),
        )
        inserted += 1
    return inserted


def bee_route(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
    max_candidates: int,
) -> tuple[list[tuple[str, float]], dict[str, object]]:
    bees = conn.execute(
        """
        SELECT *
        FROM biorag_bees
        WHERE index_version = ?
        ORDER BY strength DESC, bee_name ASC
        """,
        (BIORAG_INDEX_VERSION,),
    ).fetchall()
    facets = query_facets(query)
    trace: dict[str, object] = {
        "facets": [],
        "active_bees": [],
        "active_count": 0,
        "dormant_bees": len(bees),
        "covered_facets": 0,
        "candidate_count": 0,
    }
    if not bees or not facets:
        return [], trace

    active: dict[str, BeeActivation] = {}
    covered_facets = 0
    for facet_index, facet in enumerate(facets[:BEE_MAX_QUERY_FACETS]):
        scored: list[tuple[sqlite3.Row, float, set[str], set[str]]] = []
        for bee in bees:
            score, matched_terms, matched_roles = bee_activation_score(bee, facet)
            if score >= BEE_ACTIVATION_THRESHOLD or (matched_roles and score >= BEE_ROLE_ONLY_THRESHOLD):
                scored.append((bee, score, matched_terms, matched_roles))

        chosen: list[tuple[sqlite3.Row, float, set[str], set[str]]] = []
        covered_terms: set[str] = set()
        covered_roles: set[str] = set()
        for bee, score, matched_terms, matched_roles in sorted(scored, key=lambda item: item[1], reverse=True):
            adds_facet_value = bool(
                (matched_terms - covered_terms)
                or (matched_roles - covered_roles)
                or not chosen
            )
            if not adds_facet_value:
                continue
            chosen.append((bee, score, matched_terms, matched_roles))
            covered_terms.update(matched_terms)
            covered_roles.update(matched_roles)
            if len(chosen) >= BEE_MAX_PER_FACET:
                break
            if facet["terms"] <= covered_terms and facet["roles"] <= covered_roles:
                break

        if chosen:
            covered_facets += 1
        trace["facets"].append(
            {
                "terms": sorted(facet["terms"]),
                "roles": sorted(facet["roles"]),
                "active_bees": [
                    {
                        "bee_id": bee["bee_id"],
                        "bee_name": bee["bee_name"],
                        "score": round(score, 4),
                        "matched_terms": sorted(matched_terms),
                        "matched_roles": sorted(matched_roles),
                    }
                    for bee, score, matched_terms, matched_roles in chosen
                ],
            }
        )
        for bee, score, matched_terms, matched_roles in chosen:
            current = active.get(bee["bee_id"])
            if current is None or score > current.score:
                active[bee["bee_id"]] = BeeActivation(
                    bee_id=bee["bee_id"],
                    bee_name=bee["bee_name"],
                    facet_index=facet_index,
                    score=score,
                    matched_terms=tuple(sorted(matched_terms)),
                    matched_roles=tuple(sorted(matched_roles)),
                )

    if not active:
        return [], trace

    bees_by_id = {bee["bee_id"]: bee for bee in bees}
    overall_terms = set().union(*(facet["terms"] for facet in facets))
    overall_roles = set().union(*(facet["roles"] for facet in facets))
    query_tokens = tokenize(query)
    candidate_scores: dict[str, float] = {}
    for activation in active.values():
        bee = bees_by_id[activation.bee_id]
        for rank, chunk_id in enumerate(json_list(bee["support_chunk_ids_json"])):
            chunk = get_chunk(conn, chunk_id)
            if chunk is None or chunk["chunk_kind"] != "child":
                continue
            text = evidence_search_text(chunk)
            chunk_terms = bee_query_terms(text)
            term_overlap = len(overall_terms & chunk_terms) / max(len(overall_terms), 1) if overall_terms else 0.0
            chunk_roles = source_roles_for_chunk(chunk)
            role_overlap = len(overall_roles & chunk_roles) / max(len(overall_roles), 1) if overall_roles else 0.0
            identifier_overlap = exact_identifier_overlap(query_tokens, text)
            rank_decay = 1.0 / math.sqrt(rank + 1)
            score = (
                activation.score * 0.72
                + rank_decay * 0.055
                + term_overlap * 0.18
                + role_overlap * 0.16
                + identifier_overlap * 0.22
            )
            candidate_scores[chunk_id] = max(candidate_scores.get(chunk_id, 0.0), score)

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:max(max_candidates, top_k * 12)]
    active_trace = [
        {
            "bee_id": activation.bee_id,
            "bee_name": activation.bee_name,
            "facet_index": activation.facet_index,
            "score": round(activation.score, 4),
            "matched_terms": list(activation.matched_terms),
            "matched_roles": list(activation.matched_roles),
        }
        for activation in sorted(active.values(), key=lambda item: item.score, reverse=True)
    ]
    trace.update(
        {
            "active_bees": active_trace,
            "active_count": len(active),
            "dormant_bees": max(len(bees) - len(active), 0),
            "covered_facets": covered_facets,
            "candidate_count": len(ranked),
        }
    )
    return ranked, trace


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
    sphere_rows = conn.execute(
        """
        SELECT *
        FROM sphere_summaries
        ORDER BY strength DESC, chunk_count DESC
        LIMIT 96
        """
    ).fetchall()
    if not sphere_rows:
        return []

    rows_by_id = {row["sphere_id"]: row for row in sphere_rows}
    scored_spheres: list[tuple[sqlite3.Row, float]] = []
    for row in sphere_rows:
        score = sphere_query_score(row, query_vector, query_tokens)
        if score > 0:
            scored_spheres.append((row, score))

    candidates: dict[str, float] = {}
    direct_spheres = sorted(scored_spheres, key=lambda item: item[1], reverse=True)[: max(3, min(limit, 10))]
    for row, score in direct_spheres:
        add_sphere_support(candidates, row, score, support_limit=10)
        neighbors = conn.execute(
            """
            SELECT neighbor_sphere_id, ring, similarity
            FROM sphere_neighbors
            WHERE sphere_id = ? AND index_version = ?
            ORDER BY ring ASC, rank ASC
            LIMIT ?
            """,
            (
                row["sphere_id"],
                BIORAG_INDEX_VERSION,
                6 * sum(range(1, SPHERE_HONEYCOMB_RINGS + 1)),
            ),
        ).fetchall()
        for neighbor in neighbors:
            neighbor_row = rows_by_id.get(neighbor["neighbor_sphere_id"])
            if neighbor_row is None:
                continue
            ring = int(neighbor["ring"] or 1)
            chain_score = score * float(neighbor["similarity"] or 0.0) * (0.58 ** ring)
            if chain_score <= 0.012:
                continue
            add_sphere_support(
                candidates,
                neighbor_row,
                chain_score,
                support_limit=max(3, 8 - ring * 2),
            )
    return sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:limit]


def sphere_query_score(row: sqlite3.Row, query_vector: dict[int, float], query_tokens: list[str]) -> float:
    terms = " ".join(json_list(row["topic_terms_json"]))
    center = load_vector(row["centroid_vector_json"])
    query_distance = vector_distance(query_vector, center)
    radius = float(row["radius"] or 0.0)
    shell_std = max(float(row["shell_std"] or 0.0), 0.04)
    shell_alignment = math.exp(-abs(query_distance - radius) / shell_std) if radius else 0.0
    return (
        cosine(query_vector, center) * 0.64
        + lexical_overlap(query_tokens, f"{row['summary_text']} {terms}") * 0.22
        + shell_alignment * 0.08
        + min(float(row["density"] or 0.0), 1.0) * 0.025
        + min(float(row["strength"] or 1.0), 3.0) * 0.015
    )


def add_sphere_support(
    candidates: dict[str, float],
    row: sqlite3.Row,
    score: float,
    *,
    support_limit: int,
) -> None:
    for index, chunk_id in enumerate(json_list(row["support_chunk_ids_json"])[:support_limit]):
        candidates[chunk_id] = max(
            candidates.get(chunk_id, 0.0),
            score / (index + 1),
        )


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
        for candidate_id, ring in hex_cell_ring_candidates(conn, chunk_id, max_ring=HEX_EXPANSION_RINGS):
            if candidate_id == chunk_id:
                continue
            scores[candidate_id] = max(
                scores.get(candidate_id, 0.0),
                seed_score * hex_ring_weight(ring),
            )
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


def hex_cell_ring_candidates(
    conn: sqlite3.Connection,
    chunk_id: str,
    *,
    max_ring: int,
    max_candidates: int = 36,
) -> list[tuple[str, int]]:
    assignment = conn.execute(
        """
        SELECT q, r
        FROM biorag_chunk_cells
        WHERE chunk_id = ? AND index_version = ?
        """,
        (chunk_id, BIORAG_INDEX_VERSION),
    ).fetchone()
    if assignment is None:
        return []

    seen: set[str] = {chunk_id}
    candidates: list[tuple[str, int]] = []
    for cell_q, cell_r, ring in hex_coords_within_radius(int(assignment["q"]), int(assignment["r"]), max_ring):
        row = conn.execute(
            """
            SELECT chunk_ids_json
            FROM biorag_hex_cells
            WHERE q = ? AND r = ? AND index_version = ?
            """,
            (cell_q, cell_r, BIORAG_INDEX_VERSION),
        ).fetchone()
        if row is None:
            continue
        for candidate_id in json_list(row["chunk_ids_json"]):
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append((candidate_id, ring))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def hex_coords_within_radius(q: int, r: int, radius: int) -> list[tuple[int, int, int]]:
    coords: list[tuple[int, int, int]] = []
    for dq in range(-radius, radius + 1):
        min_dr = max(-radius, -dq - radius)
        max_dr = min(radius, -dq + radius)
        for dr in range(min_dr, max_dr + 1):
            cell_q = q + dq
            cell_r = r + dr
            coords.append((cell_q, cell_r, hex_distance(q, r, cell_q, cell_r)))
    return sorted(coords, key=lambda item: (item[2], item[0], item[1]))


def hex_ring_weight(ring: int) -> float:
    if ring <= 0:
        return 0.5
    if ring == 1:
        return 0.32
    return max(0.08, 0.2 * (0.64 ** (ring - 2)))


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
        links = iter_links_for_chunks(conn, frontier.keys())
        adaptive_weights = edge_weights_for_links(conn, links)
        for link in links:
            source = link["source_chunk_id"]
            target = link["target_chunk_id"]
            link_type = str(link["link_type"] or "web")
            confidence = max(float(link["confidence_score"] or 0.0), 0.08)
            adaptive = adaptive_weights.get(stable_edge_key(source, target, link_type), 1.0)
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


def edge_weights_for_links(conn: sqlite3.Connection, links: list[sqlite3.Row]) -> dict[str, float]:
    edge_keys = sorted(
        {
            stable_edge_key(link["source_chunk_id"], link["target_chunk_id"], str(link["link_type"] or "web"))
            for link in links
        }
    )
    if not edge_keys:
        return {}

    weights: dict[str, float] = {}
    for batch in chunked_items(edge_keys):
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            SELECT edge_key, weight, last_decayed_at
            FROM biorag_edge_weights
            WHERE edge_key IN ({placeholders})
            """,
            batch,
        ).fetchall()
        for row in rows:
            weights[row["edge_key"]] = decayed_weight(
                float(row["weight"] or 1.0),
                str(row["last_decayed_at"] or utc_now()),
            )
    return weights


def chunked_items(items: list[str], size: int = 500) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


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


def coverage_rescue(
    conn: sqlite3.Connection,
    query: str,
    candidates: list[tuple[str, float]],
    *,
    max_candidates: int,
    allowed_chunk_ids: list[str] | None = None,
) -> tuple[list[tuple[str, float]], set[str]]:
    query_terms = critical_query_terms(query)
    if not query_terms:
        return candidates[:max_candidates], set()

    candidate_scores = dict(candidates[:max_candidates])
    covered = covered_query_terms(conn, query_terms, candidates[: min(len(candidates), 12)])
    missing = query_terms - covered
    if not missing and len(covered) / max(len(query_terms), 1) >= 0.75:
        return candidates[:max_candidates], set()

    rescued: set[str] = set()
    rescue_rows = child_vectors_for_ids(conn, allowed_chunk_ids) if allowed_chunk_ids is not None else list(iter_child_vectors(conn))
    for row in rescue_rows:
        chunk = get_chunk(conn, row["chunk_id"])
        if chunk is None:
            continue
        search_text = evidence_search_text(chunk)
        text_terms = critical_query_terms(search_text)
        missing_coverage = len(missing & text_terms) / max(len(missing), 1)
        query_coverage = len(query_terms & text_terms) / max(len(query_terms), 1)
        lexical_score = lexical_overlap(query_terms, search_text)
        if missing_coverage <= 0 and query_coverage < 0.38 and lexical_score < 0.22:
            continue
        rescue_score = 0.038 + missing_coverage * 0.18 + query_coverage * 0.075 + lexical_score * 0.06
        old_score = candidate_scores.get(row["chunk_id"], 0.0)
        candidate_scores[row["chunk_id"]] = max(old_score, old_score + rescue_score if old_score else rescue_score)
        rescued.add(row["chunk_id"])

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:max_candidates], rescued


def covered_query_terms(
    conn: sqlite3.Connection,
    query_terms: set[str],
    candidates: list[tuple[str, float]],
) -> set[str]:
    covered: set[str] = set()
    for chunk_id, _score in candidates:
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue
        covered.update(query_terms & critical_query_terms(evidence_search_text(chunk)))
    return covered


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
    bee_trace: dict[str, object] | None = None,
) -> None:
    layer_counts = Counter(layer for layers in paths.values() for layer in layers)
    trace = {
        "layers": dict(layer_counts),
        "selected": [
            {"chunk_id": chunk_id, "score": score, "layers": sorted(paths.get(chunk_id, ()))}
            for chunk_id, score in selected
        ],
    }
    if bee_trace is not None:
        trace["bee_router"] = bee_trace
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


def critical_query_terms(text: str) -> set[str]:
    return {
        normalize_token(token)
        for token in tokenize(text)
        if len(token) >= 4 and token not in STOP_TERMS and token not in QUERY_STOP_TERMS
    }


QUERY_STOP_TERMS = {
    "answer",
    "based",
    "confirm",
    "data",
    "discuss",
    "discusses",
    "evidence",
    "find",
    "found",
    "give",
    "how",
    "impact",
    "mentions",
    "need",
    "needs",
    "question",
    "show",
    "source",
    "sources",
    "support",
    "supports",
    "tell",
    "which",
}


BEE_QUERY_STOP_TERMS = QUERY_STOP_TERMS | {
    "affected",
    "also",
    "and",
    "enter",
    "for",
    "record",
    "records",
    "recorded",
    "stuff",
    "value",
    "values",
}


def dense_search_candidates(
    conn: sqlite3.Connection,
    query: str,
    chunk_ids: list[str],
    *,
    limit: int,
) -> list[tuple[str, float]]:
    query_vector = embed_text(query)
    if not query_vector:
        return []
    scored: list[tuple[str, float]] = []
    for row in child_vectors_for_ids(conn, chunk_ids):
        score = cosine(query_vector, vector_from_row(row))
        if score > 0:
            scored.append((row["chunk_id"], score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def lexical_search_candidates(
    conn: sqlite3.Connection,
    query: str,
    chunk_ids: list[str],
    *,
    limit: int,
) -> list[tuple[str, float]]:
    query_tokens = tokenize(query)
    requested_roles = requested_evidence_roles(query_tokens)
    scored: list[tuple[str, float]] = []
    for chunk_id in ordered_unique(chunk_ids):
        chunk = get_chunk(conn, chunk_id)
        if chunk is None or chunk["chunk_kind"] != "child":
            continue
        text = evidence_search_text(chunk)
        roles = source_roles_for_chunk(chunk)
        role_overlap = len(requested_roles & roles) / max(len(requested_roles), 1) if requested_roles else 0.0
        score = (
            lexical_overlap(query_tokens, text)
            + exact_identifier_overlap(query_tokens, text) * 0.8
            + role_overlap * 0.35
        )
        if score > 0:
            scored.append((chunk_id, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def precision_boost_candidates(
    conn: sqlite3.Connection,
    query: str,
    candidates: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    if not candidates:
        return []
    query_tokens = tokenize(query)
    requested_roles = requested_evidence_roles(query_tokens)
    source_record = is_source_record_query(query)
    boosted: list[tuple[str, float]] = []
    for chunk_id, score in candidates:
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue
        text = evidence_search_text(chunk)
        chunk_roles = source_roles_for_chunk(chunk)
        role_overlap = len(requested_roles & chunk_roles) / max(len(requested_roles), 1) if requested_roles else 0.0
        identifier_overlap = exact_identifier_overlap(query_tokens, text)
        exact_anchor_bonus = 0.22 if source_record and query_anchor_terms(query_tokens) & set(tokenize(text)) else 0.0
        boosted.append(
            (
                chunk_id,
                score
                + identifier_overlap * (0.72 if source_record else 0.26)
                + role_overlap * 0.36
                + exact_anchor_bonus,
            )
        )
    return sorted(boosted, key=lambda item: item[1], reverse=True)


def filter_source_record_context(
    conn: sqlite3.Connection,
    query: str,
    selected: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    query_terms = bee_query_terms(query)
    anchors = query_anchor_terms(tokenize(query))
    if not anchors:
        return selected
    filtered: list[tuple[str, float]] = []
    exact: list[tuple[str, float]] = []
    for chunk_id, score in selected:
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue
        text_terms = bee_query_terms(evidence_search_text(chunk))
        anchor_match = bool(anchors & text_terms)
        topic_overlap = len((query_terms - anchors) & text_terms) / max(len(query_terms - anchors), 1)
        if anchor_match:
            exact.append((chunk_id, score + 0.12))
            filtered.append((chunk_id, score + 0.12))
        elif topic_overlap >= 0.34:
            filtered.append((chunk_id, score))
    if not exact:
        return selected
    return sorted(ordered_ranked_unique(filtered), key=lambda item: item[1], reverse=True)


def is_source_record_query(query: str) -> bool:
    lowered = query.lower()
    return (
        ("which source records" in lowered or "what source records" in lowered)
        and bool(query_anchor_terms(tokenize(query)))
    )


def query_anchor_terms(query_tokens: list[str]) -> set[str]:
    return {
        token
        for token in query_tokens
        if any(char.isdigit() for char in token) and ("-" in token or "_" in token)
    }


def identifier_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if any(char.isdigit() for char in token) and ("-" in token or "_" in token)
    ]


def child_vectors_for_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[sqlite3.Row]:
    unique_ids = ordered_unique(chunk_ids)
    if not unique_ids:
        return []
    rows: list[sqlite3.Row] = []
    for batch in chunked_items(unique_ids):
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            conn.execute(
                f"""
                SELECT c.chunk_id, c.file_id, c.parent_id, c.text_content, c.modality, v.vector_json
                FROM content_chunks c
                JOIN chunk_vectors v ON v.chunk_id = c.chunk_id
                WHERE c.chunk_kind = 'child' AND c.chunk_id IN ({placeholders})
                """,
                batch,
            ).fetchall()
        )
    return rows


def query_facets(query: str) -> list[dict[str, set[str]]]:
    normalized = f" {query.lower()} "
    for connector in (
        " but also ",
        " and also ",
        " as well as ",
        " along with ",
        " together with ",
        " plus ",
        " and ",
        ";",
        ",",
    ):
        normalized = normalized.replace(connector, "|")
    facets: list[dict[str, set[str]]] = []
    for part in normalized.split("|"):
        terms = bee_query_terms(part)
        roles = requested_evidence_roles(tokenize(part))
        if terms or roles:
            facets.append({"terms": terms, "roles": roles})
    if not facets:
        facets.append({"terms": bee_query_terms(query), "roles": requested_evidence_roles(tokenize(query))})
    return facets[:BEE_MAX_QUERY_FACETS]


def bee_activation_score(
    bee: sqlite3.Row,
    facet: dict[str, set[str]],
) -> tuple[float, set[str], set[str]]:
    bee_terms = {normalize_token(term) for term in json_list(bee["key_terms_json"])}
    bee_roles = set(json_list(bee["source_roles_json"]))
    facet_terms = facet["terms"]
    facet_roles = facet["roles"]
    matched_terms = facet_terms & bee_terms
    matched_roles = facet_roles & bee_roles

    term_score = len(matched_terms) / max(min(len(facet_terms), 6), 1) if facet_terms else 0.0
    role_score = len(matched_roles) / max(len(facet_roles), 1) if facet_roles else 0.0
    identifier_bonus = 0.08 if any(any(char.isdigit() for char in term) for term in matched_terms) else 0.0
    bonded_bonus = 0.045 if matched_terms and (matched_roles or not facet_roles) else 0.0
    strength_bonus = min(float(bee["strength"] or 1.0), 3.0) * 0.012
    score = term_score * 0.72 + role_score * 0.26 + identifier_bonus + bonded_bonus + strength_bonus
    return score, matched_terms, matched_roles


def bee_query_terms(text: str) -> set[str]:
    return {
        normalize_token(token)
        for token in tokenize(text)
        if len(token) >= 3 and token not in STOP_TERMS and token not in BEE_QUERY_STOP_TERMS
    }


def source_roles_for_chunk(row: sqlite3.Row) -> set[str]:
    return evidence_roles_for_source(
        file_name=str(row["file_name"] or ""),
        file_type=str(row["file_type"] or ""),
        modality=str(row["modality"] or ""),
    )


def ordered_unique(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        value = str(item)
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def ordered_ranked_unique(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    order: list[str] = []
    for chunk_id, score in items:
        if chunk_id not in scores:
            order.append(chunk_id)
            scores[chunk_id] = score
        else:
            scores[chunk_id] = max(scores[chunk_id], score)
    return [(chunk_id, scores[chunk_id]) for chunk_id in order]


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
