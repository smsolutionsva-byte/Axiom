from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from axiom.biorag import BIORAG_INDEX_VERSION, biorag_search, biorag_status, sphere_route
from axiom.database import connect, get_chunk
from axiom.ingestion import ingest_path


class BioragHoneycombTests(unittest.TestCase):
    def test_exact_identifier_ocr_source_survives_relevance_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "briefing.txt").write_text(
                "Rural Clinic briefing AXM-01-01. Program AAROGYA-27 tracks the cold-chain vaccine route. "
                "The briefing records thermal breach and storage temperature evidence for the clinic team.",
                encoding="utf-8",
            )
            (corpus / "dashboard_ocr.txt").write_text(
                "Dashboard OCR AXM-01-04. Visible panel: AAROGYA-27. Alert card: thermal breach. "
                "Metric tile: storage temperature 8.7 celsius.",
                encoding="utf-8",
            )

            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)

                _query_id, hits = biorag_search(
                    conn,
                    "Which source records AXM-01-04 for AAROGYA-27 and cold-chain vaccine route?",
                    top_k=2,
                )
                self.assertIn("dashboard_ocr.txt", {hit.file_name for hit in hits})
            finally:
                conn.close()

    def test_sphere_cache_keywords_and_honeycomb_rings_are_built(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            anchors = [
                "alphaanchor",
                "bravoanchor",
                "charlieanchor",
                "deltaanchor",
                "echoanchor",
                "foxtrotanchor",
                "golfanchor",
                "hotelanchor",
            ]
            for index, anchor in enumerate(anchors, start=1):
                (corpus / f"{index:02d}_{anchor}.txt").write_text(
                    f"{anchor} {anchor} {anchor} {anchor}. "
                    "Omega refund honeycomb cluster deadline evidence supplier memo shared context. "
                    f"Operational note {index} keeps this topic distinct but relevant.",
                    encoding="utf-8",
                )

            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)

                sphere_rows = conn.execute(
                    "SELECT summary_text, topic_terms_json FROM sphere_summaries"
                ).fetchall()
                ring_two_count = conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM sphere_neighbors
                    WHERE index_version = ? AND ring = 2
                    """,
                    (BIORAG_INDEX_VERSION,),
                ).fetchone()["count"]
                status = biorag_status(conn)

                self.assertGreaterEqual(len(sphere_rows), 7)
                self.assertGreater(ring_two_count, 0)
                self.assertGreater(status["counts"]["sphere_neighbors"], 0)
                self.assertTrue(all(row["summary_text"] for row in sphere_rows))
                all_terms = {
                    term
                    for row in sphere_rows
                    for term in json.loads(row["topic_terms_json"])
                }
                self.assertIn("omega", all_terms)
                self.assertIn("honeycomb", all_terms)

                routed = sphere_route(conn, "omega refund honeycomb deadline evidence", limit=12)
                routed_files = {
                    get_chunk(conn, chunk_id)["file_name"]
                    for chunk_id, _score in routed
                    if get_chunk(conn, chunk_id) is not None
                }
                self.assertGreaterEqual(len(routed_files), 6)

                _query_id, hits = biorag_search(conn, "omega refund honeycomb deadline evidence", top_k=4)
                self.assertTrue(hits)
                self.assertTrue(any("anchor" in hit.file_name for hit in hits))
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
