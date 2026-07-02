from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from axiom.biorag import bee_route, biorag_search, biorag_status
from axiom.database import connect
from axiom.ingestion import ingest_path


class BioragBeeRouterTests(unittest.TestCase):
    def test_bees_wake_only_for_matching_query_facets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            write_bee_corpus(corpus)

            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)

                _candidates, trace = bee_route(
                    conn,
                    "give me monsoon OCR but also climate change impact",
                    top_k=3,
                    max_candidates=80,
                )
                active_names = {
                    item["bee_name"]
                    for item in trace["active_bees"]
                    if isinstance(item, dict)
                }

                self.assertGreaterEqual(biorag_status(conn)["counts"]["bees"], 3)
                self.assertEqual(trace["covered_facets"], 2)
                self.assertGreater(trace["dormant_bees"], 0)
                self.assertTrue(any("dashboard" in name or "monsoon" in name for name in active_names))
                self.assertTrue(any("climate" in name for name in active_names))
                self.assertFalse(any("rail" in name for name in active_names))
            finally:
                conn.close()

    def test_multifacet_query_retrieves_bonded_territory_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            write_bee_corpus(corpus)

            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)

                query_id, hits = biorag_search(
                    conn,
                    "give me data on monsoon OCR but also tell me how it affected climate change",
                    top_k=2,
                )
                hit_names = {hit.file_name for hit in hits}
                trace_row = conn.execute(
                    "SELECT trace_json FROM biorag_retrieval_runs WHERE query_id = ?",
                    (query_id,),
                ).fetchone()
                trace = json.loads(trace_row["trace_json"])

                self.assertIn("monsoon_dashboard_ocr.txt", hit_names)
                self.assertIn("climate_change.md", hit_names)
                self.assertEqual(trace["bee_router"]["covered_facets"], 2)
                self.assertEqual(trace["layers"]["bee_router"], 2)
            finally:
                conn.close()


def write_bee_corpus(corpus: Path) -> None:
    (corpus / "monsoon_briefing.txt").write_text(
        "Monsoon field briefing KAVERI-14. Rainfall surge changed soil moisture and crop risk.",
        encoding="utf-8",
    )
    (corpus / "monsoon_dashboard_ocr.txt").write_text(
        "Dashboard OCR KAVERI-14. Visible panel: monsoon rainfall. "
        "Metric tile: soil moisture 31 percent.",
        encoding="utf-8",
    )
    (corpus / "climate_change.md").write_text(
        "Climate change impact note. Warmer air increased monsoon variability and rainfall intensity.",
        encoding="utf-8",
    )
    (corpus / "rail_audit.txt").write_text(
        "Rail audit TRACK-18. Axle sensor maintenance has no rainfall relationship.",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
