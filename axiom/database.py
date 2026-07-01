from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .embeddings import EMBEDDING_MODEL, dump_vector, load_vector, tokenize


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS document_registry (
            file_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            status TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS content_chunks (
            chunk_id TEXT PRIMARY KEY,
            parent_id TEXT,
            file_id TEXT NOT NULL,
            chunk_kind TEXT NOT NULL CHECK(chunk_kind IN ('parent', 'child')),
            text_content TEXT NOT NULL,
            modality TEXT NOT NULL,
            page_number INTEGER,
            start_timestamp TEXT,
            end_timestamp TEXT,
            char_start INTEGER,
            char_end INTEGER,
            token_count INTEGER NOT NULL,
            FOREIGN KEY(file_id) REFERENCES document_registry(file_id) ON DELETE CASCADE,
            FOREIGN KEY(parent_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chunk_vectors (
            chunk_id TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS cross_modal_links (
            link_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_chunk_id TEXT NOT NULL,
            target_chunk_id TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            link_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(source_chunk_id, target_chunk_id, link_type),
            FOREIGN KEY(source_chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE,
            FOREIGN KEY(target_chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS query_audit (
            query_id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_context (
            query_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY(query_id, chunk_id),
            FOREIGN KEY(query_id) REFERENCES query_audit(query_id) ON DELETE CASCADE,
            FOREIGN KEY(chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS sphere_summaries (
            sphere_id TEXT PRIMARY KEY,
            sphere_name TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            centroid_vector_json TEXT NOT NULL,
            topic_terms_json TEXT NOT NULL,
            support_chunk_ids_json TEXT NOT NULL,
            document_ids_json TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            radius REAL NOT NULL DEFAULT 0.0,
            shell_mean REAL NOT NULL DEFAULT 0.0,
            shell_std REAL NOT NULL DEFAULT 0.0,
            density REAL NOT NULL DEFAULT 0.0,
            dimensionality INTEGER NOT NULL DEFAULT 0,
            strength REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sphere_neighbors (
            sphere_id TEXT NOT NULL,
            neighbor_sphere_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            ring INTEGER NOT NULL,
            similarity REAL NOT NULL,
            relation_hint TEXT NOT NULL,
            chain_group TEXT NOT NULL,
            index_version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(sphere_id, neighbor_sphere_id),
            FOREIGN KEY(sphere_id) REFERENCES sphere_summaries(sphere_id) ON DELETE CASCADE,
            FOREIGN KEY(neighbor_sphere_id) REFERENCES sphere_summaries(sphere_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS biorag_tree_nodes (
            node_id TEXT PRIMARY KEY,
            parent_node_id TEXT,
            source_id TEXT,
            node_type TEXT NOT NULL,
            level INTEGER NOT NULL,
            summary_text TEXT NOT NULL,
            centroid_vector_json TEXT NOT NULL,
            support_chunk_ids_json TEXT NOT NULL,
            document_ids_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(parent_node_id) REFERENCES biorag_tree_nodes(node_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS hex_neighbors (
            chunk_id TEXT NOT NULL,
            neighbor_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            similarity REAL NOT NULL,
            relation_hint TEXT NOT NULL,
            index_version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(chunk_id, neighbor_id),
            FOREIGN KEY(chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE,
            FOREIGN KEY(neighbor_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS biorag_hex_cells (
            cell_id TEXT PRIMARY KEY,
            q INTEGER NOT NULL,
            r INTEGER NOT NULL,
            center_x REAL NOT NULL,
            center_y REAL NOT NULL,
            radius REAL NOT NULL,
            occupant_count INTEGER NOT NULL,
            centroid_vector_json TEXT NOT NULL,
            chunk_ids_json TEXT NOT NULL,
            index_version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(q, r, index_version)
        );

        CREATE TABLE IF NOT EXISTS biorag_chunk_cells (
            chunk_id TEXT PRIMARY KEY,
            cell_id TEXT NOT NULL,
            q INTEGER NOT NULL,
            r INTEGER NOT NULL,
            projected_x REAL NOT NULL,
            projected_y REAL NOT NULL,
            ring INTEGER NOT NULL,
            index_version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE,
            FOREIGN KEY(cell_id) REFERENCES biorag_hex_cells(cell_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS adaptive_path_stats (
            path_id TEXT PRIMARY KEY,
            layer TEXT NOT NULL,
            source_id TEXT,
            target_id TEXT NOT NULL,
            query_signature TEXT NOT NULL,
            impressions INTEGER NOT NULL DEFAULT 0,
            selected_count INTEGER NOT NULL DEFAULT 0,
            verified_count INTEGER NOT NULL DEFAULT 0,
            rejected_count INTEGER NOT NULL DEFAULT 0,
            strength REAL NOT NULL DEFAULT 1.0,
            last_used_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS biorag_edge_weights (
            edge_key TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            pheromone REAL NOT NULL DEFAULT 1.0,
            coactivation_count INTEGER NOT NULL DEFAULT 0,
            last_reinforced_at TEXT,
            last_decayed_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS biorag_retrieval_runs (
            query_id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            complexity REAL NOT NULL,
            budget_json TEXT NOT NULL,
            trace_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_id) REFERENCES query_audit(query_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS operator_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type TEXT NOT NULL,
            target TEXT,
            parameters_json TEXT NOT NULL,
            executed INTEGER NOT NULL,
            success INTEGER NOT NULL,
            return_code INTEGER,
            stdout_preview TEXT,
            stderr_preview TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_file ON content_chunks(file_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent ON content_chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_modality ON content_chunks(modality);
        CREATE INDEX IF NOT EXISTS idx_chunks_kind ON content_chunks(chunk_kind);
        CREATE INDEX IF NOT EXISTS idx_cross_links_source ON cross_modal_links(source_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_cross_links_target ON cross_modal_links(target_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_cross_links_type ON cross_modal_links(link_type);
        CREATE INDEX IF NOT EXISTS idx_spheres_strength ON sphere_summaries(strength);
        CREATE INDEX IF NOT EXISTS idx_sphere_neighbors_source ON sphere_neighbors(sphere_id, index_version);
        CREATE INDEX IF NOT EXISTS idx_sphere_neighbors_target ON sphere_neighbors(neighbor_sphere_id, index_version);
        CREATE INDEX IF NOT EXISTS idx_sphere_neighbors_group ON sphere_neighbors(chain_group, ring);
        CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent ON biorag_tree_nodes(parent_node_id);
        CREATE INDEX IF NOT EXISTS idx_tree_nodes_type ON biorag_tree_nodes(node_type);
        CREATE INDEX IF NOT EXISTS idx_hex_neighbors_chunk ON hex_neighbors(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_hex_neighbors_neighbor ON hex_neighbors(neighbor_id);
        CREATE INDEX IF NOT EXISTS idx_hex_cells_qr ON biorag_hex_cells(q, r);
        CREATE INDEX IF NOT EXISTS idx_chunk_cells_cell ON biorag_chunk_cells(cell_id);
        CREATE INDEX IF NOT EXISTS idx_edge_weights_source ON biorag_edge_weights(source_id);
        CREATE INDEX IF NOT EXISTS idx_edge_weights_target ON biorag_edge_weights(target_id);
        CREATE INDEX IF NOT EXISTS idx_edge_weights_type ON biorag_edge_weights(edge_type);
        CREATE INDEX IF NOT EXISTS idx_adaptive_target ON adaptive_path_stats(target_id);
        CREATE INDEX IF NOT EXISTS idx_adaptive_signature ON adaptive_path_stats(query_signature);
        CREATE INDEX IF NOT EXISTS idx_operator_audit_created ON operator_audit(created_at);
        """
    )
    ensure_column(conn, "sphere_summaries", "radius", "REAL NOT NULL DEFAULT 0.0")
    ensure_column(conn, "sphere_summaries", "shell_mean", "REAL NOT NULL DEFAULT 0.0")
    ensure_column(conn, "sphere_summaries", "shell_std", "REAL NOT NULL DEFAULT 0.0")
    ensure_column(conn, "sphere_summaries", "density", "REAL NOT NULL DEFAULT 0.0")
    ensure_column(conn, "sphere_summaries", "dimensionality", "INTEGER NOT NULL DEFAULT 0")
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS content_chunks_fts
            USING fts5(chunk_id UNINDEXED, file_name UNINDEXED, modality UNINDEXED, text_content)
            """
        )
    except sqlite3.OperationalError:
        pass
    conn.commit()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def fts_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'content_chunks_fts'"
    ).fetchone()
    return row is not None


def upsert_document(
    conn: sqlite3.Connection,
    *,
    file_id: str,
    file_name: str,
    file_type: str,
    file_path: str,
    sha256: str,
    size_bytes: int,
    status: str = "indexed",
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO document_registry (
            file_id, file_name, file_type, file_path, sha256, size_bytes, status, ingested_at, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            file_name = excluded.file_name,
            file_type = excluded.file_type,
            file_path = excluded.file_path,
            sha256 = excluded.sha256,
            size_bytes = excluded.size_bytes,
            status = excluded.status,
            ingested_at = excluded.ingested_at,
            error_message = excluded.error_message
        """,
        (file_id, file_name, file_type, file_path, sha256, size_bytes, status, utc_now(), error_message),
    )


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    parent_id: str | None,
    file_id: str,
    chunk_kind: str,
    text_content: str,
    modality: str,
    page_number: int | None,
    start_timestamp: str | None,
    end_timestamp: str | None,
    char_start: int | None,
    char_end: int | None,
    token_count: int,
    vector: dict[int, float] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO content_chunks (
            chunk_id, parent_id, file_id, chunk_kind, text_content, modality, page_number,
            start_timestamp, end_timestamp, char_start, char_end, token_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            parent_id = excluded.parent_id,
            file_id = excluded.file_id,
            chunk_kind = excluded.chunk_kind,
            text_content = excluded.text_content,
            modality = excluded.modality,
            page_number = excluded.page_number,
            start_timestamp = excluded.start_timestamp,
            end_timestamp = excluded.end_timestamp,
            char_start = excluded.char_start,
            char_end = excluded.char_end,
            token_count = excluded.token_count
        """,
        (
            chunk_id,
            parent_id,
            file_id,
            chunk_kind,
            text_content,
            modality,
            page_number,
            start_timestamp,
            end_timestamp,
            char_start,
            char_end,
            token_count,
        ),
    )
    if vector is not None:
        conn.execute(
            """
            INSERT INTO chunk_vectors (chunk_id, vector_json, embedding_model)
            VALUES (?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                vector_json = excluded.vector_json,
                embedding_model = excluded.embedding_model
            """,
            (chunk_id, dump_vector(vector), EMBEDDING_MODEL),
        )
    if chunk_kind == "child" and fts_available(conn):
        file_name = conn.execute(
            "SELECT file_name FROM document_registry WHERE file_id = ?", (file_id,)
        ).fetchone()["file_name"]
        conn.execute("DELETE FROM content_chunks_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            """
            INSERT INTO content_chunks_fts (chunk_id, file_name, modality, text_content)
            VALUES (?, ?, ?, ?)
            """,
            (chunk_id, file_name, modality, text_content),
        )


def iter_child_vectors(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    return conn.execute(
        """
        SELECT c.chunk_id, c.file_id, c.parent_id, c.text_content, c.modality, v.vector_json
        FROM content_chunks c
        JOIN chunk_vectors v ON v.chunk_id = c.chunk_id
        WHERE c.chunk_kind = 'child'
        """
    )


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT c.*, d.file_name, d.file_path, d.file_type, d.sha256
        FROM content_chunks c
        JOIN document_registry d ON d.file_id = c.file_id
        WHERE c.chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()


def get_chunk_by_prefix(conn: sqlite3.Connection, chunk_prefix: str) -> sqlite3.Row | None:
    prefix = chunk_prefix.strip()
    if prefix.startswith("[Axiom:") and prefix.endswith("]"):
        prefix = prefix[len("[Axiom:") : -1]
    return conn.execute(
        """
        SELECT c.*, d.file_name, d.file_path, d.file_type, d.sha256
        FROM content_chunks c
        JOIN document_registry d ON d.file_id = c.file_id
        WHERE c.chunk_id LIKE ?
        ORDER BY LENGTH(c.chunk_id)
        LIMIT 1
        """,
        (prefix + "%",),
    ).fetchone()


def get_context_for_child(conn: sqlite3.Connection, child_id: str) -> sqlite3.Row | None:
    child = get_chunk(conn, child_id)
    if child is None:
        return None
    if child["parent_id"]:
        parent = get_chunk(conn, child["parent_id"])
        if parent is not None:
            return parent
    return child


def lexical_search(conn: sqlite3.Connection, query: str, limit: int = 25) -> list[tuple[str, float]]:
    tokens = tokenize(query)
    if not tokens:
        return []
    if fts_available(conn):
        match_query = " OR ".join(tokens[:24])
        try:
            rows = conn.execute(
                """
                SELECT chunk_id, bm25(content_chunks_fts) AS rank
                FROM content_chunks_fts
                WHERE content_chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
            return [(row["chunk_id"], 1.0 / (index + 1)) for index, row in enumerate(rows)]
        except sqlite3.OperationalError:
            pass

    like_terms = [f"%{token}%" for token in tokens[:8]]
    scores: dict[str, float] = {}
    for term in like_terms:
        rows = conn.execute(
            """
            SELECT chunk_id
            FROM content_chunks
            WHERE chunk_kind = 'child' AND LOWER(text_content) LIKE ?
            LIMIT ?
            """,
            (term, limit),
        ).fetchall()
        for index, row in enumerate(rows):
            scores[row["chunk_id"]] = scores.get(row["chunk_id"], 0.0) + 1.0 / (index + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def record_query_context(
    conn: sqlite3.Connection,
    *,
    query_id: str,
    query_text: str,
    ranked: list[tuple[str, float]],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO query_audit (query_id, query_text, created_at) VALUES (?, ?, ?)",
        (query_id, query_text, utc_now()),
    )
    conn.execute("DELETE FROM query_context WHERE query_id = ?", (query_id,))
    conn.executemany(
        "INSERT INTO query_context (query_id, rank, chunk_id, score) VALUES (?, ?, ?, ?)",
        [(query_id, rank + 1, chunk_id, score) for rank, (chunk_id, score) in enumerate(ranked)],
    )


def insert_cross_link(
    conn: sqlite3.Connection,
    *,
    source_chunk_id: str,
    target_chunk_id: str,
    confidence_score: float,
    link_type: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO cross_modal_links (
            source_chunk_id, target_chunk_id, confidence_score, link_type, created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (source_chunk_id, target_chunk_id, confidence_score, link_type, utc_now()),
    )


def iter_links_for_chunks(conn: sqlite3.Connection, chunk_ids: Iterable[str]) -> list[sqlite3.Row]:
    ids = sorted({chunk_id for chunk_id in chunk_ids if chunk_id})
    if not ids:
        return []
    rows: list[sqlite3.Row] = []
    for batch in batched(ids):
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            conn.execute(
                f"""
                SELECT source_chunk_id, target_chunk_id, confidence_score, link_type
                FROM cross_modal_links
                WHERE source_chunk_id IN ({placeholders}) OR target_chunk_id IN ({placeholders})
                ORDER BY confidence_score DESC
                """,
                (*batch, *batch),
            ).fetchall()
        )
    return sorted(rows, key=lambda row: row["confidence_score"], reverse=True)


def child_rows_for_parents(conn: sqlite3.Connection, parent_ids: Iterable[str]) -> list[sqlite3.Row]:
    ids = sorted({parent_id for parent_id in parent_ids if parent_id})
    if not ids:
        return []
    rows: list[sqlite3.Row] = []
    for batch in batched(ids):
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            conn.execute(
                f"""
                SELECT c.*, d.file_name, d.file_path, d.file_type, d.sha256
                FROM content_chunks c
                JOIN document_registry d ON d.file_id = c.file_id
                WHERE c.chunk_kind = 'child' AND c.parent_id IN ({placeholders})
                ORDER BY c.parent_id, c.char_start
                """,
                batch,
            ).fetchall()
        )
    return sorted(rows, key=lambda row: (row["parent_id"], row["char_start"] or 0))


def batched(items: list[str], size: int = 500) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            d.*,
            COUNT(c.chunk_id) AS chunk_count
        FROM document_registry d
        LEFT JOIN content_chunks c ON c.file_id = d.file_id AND c.chunk_kind = 'child'
        GROUP BY d.file_id
        ORDER BY d.ingested_at DESC, d.file_name ASC
        """
    ).fetchall()


def record_operator_audit(
    conn: sqlite3.Connection,
    *,
    action_type: str,
    target: str | None,
    parameters_json: str,
    executed: bool,
    success: bool,
    return_code: int | None = None,
    stdout_preview: str | None = None,
    stderr_preview: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO operator_audit (
            action_type, target, parameters_json, executed, success, return_code,
            stdout_preview, stderr_preview, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            action_type,
            target,
            parameters_json,
            int(executed),
            int(success),
            return_code,
            (stdout_preview or "")[:2000],
            (stderr_preview or "")[:2000],
            utc_now(),
        ),
    )
    conn.commit()


def list_operator_audit(conn: sqlite3.Connection, *, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM operator_audit
        ORDER BY audit_id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def vector_from_row(row: sqlite3.Row) -> dict[int, float]:
    return load_vector(row["vector_json"])
