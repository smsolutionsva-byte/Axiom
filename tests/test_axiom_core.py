from __future__ import annotations

import sqlite3
import tempfile
import unittest
import base64
import os
import json
import math
from pathlib import Path

from axiom.answering import answer_query
from axiom.analytics import build_analytics
from axiom.biorag import biorag_search, biorag_status, energy_budget
from axiom.citation import validate_citations
from axiom.database import connect
from axiom.dependencies import audit_dependencies, dependency_by_key
from axiom.evaluation import (
    BenchmarkCase,
    CaseResult,
    evaluator_framework_summary,
    official_metric_or_fallback,
    run_benchmark,
    summarize_results,
    term_supported,
)
from axiom.extractors import extract_docx_xml, extract_segments, file_type_for
from axiom.image_generation import ImageGenerationRequest, generate_image, image_generation_status
from axiom.ingestion import ingest_path
from axiom.investigation import investigate_subject, validate_investigation_answer
from axiom.mission import build_mission_brief
from axiom.reports import generate_case_report
from axiom.retrieval import avtr_search
from axiom.uploads import save_uploaded_files
from axiom.vision import analyze_image, ingest_visual_analysis, run_ocr, save_pasted_image
from axiom.workstation import find_files, plan_operator_task, run_command, scan_folder
from tools.build_large_benchmark import build_dataset


class AxiomCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._previous_model_disable = os.environ.get("AXIOM_DISABLE_MODEL_ANSWERS")
        os.environ["AXIOM_DISABLE_MODEL_ANSWERS"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_model_disable is None:
            os.environ.pop("AXIOM_DISABLE_MODEL_ANSWERS", None)
        else:
            os.environ["AXIOM_DISABLE_MODEL_ANSWERS"] = cls._previous_model_disable

    def test_ingest_and_answer_with_valid_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "annexure.txt").write_text(
                "The international development targets for 2024 require cross-referencing with field screenshots.",
                encoding="utf-8",
            )
            (corpus / "call.txt").write_text(
                "The speaker references a screenshot taken during field review and compares it with Annexure 4B.",
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                report = ingest_path(conn, corpus)

                self.assertEqual(len(report.indexed_files), 2)
                self.assertGreaterEqual(report.chunks_created, 4)

                result = answer_query(conn, "international development 2024 screenshot", top_k=3)
                self.assertIn("[Axiom:", result.answer)
                self.assertNotIn("[Unverified Claim]", result.answer)
                self.assertTrue(result.sources)
            finally:
                conn.close()

    def test_avtr_spider_web_retrieval_expands_to_linked_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "clause.txt").write_text(
                "The arachne-marker zeta clause asks the operator to prepare the prototype refund memo.",
                encoding="utf-8",
            )
            (corpus / "schedule.txt").write_text(
                "Arachne-marker zeta is also used in the supplier schedule, which lists deadline evidence.",
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                report = ingest_path(conn, corpus)
                self.assertGreater(report.links_created, 0)

                _query_id, hits = avtr_search(conn, "prototype refund memo", top_k=2)
                names = {hit.file_name for hit in hits}

                self.assertIn("clause.txt", names)
                self.assertIn("schedule.txt", names)
            finally:
                conn.close()

    def test_biorag_builds_layers_and_records_adaptive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "sphere.txt").write_text(
                "Bio sphere memory stores compressed refund intelligence for the omega program.",
                encoding="utf-8",
            )
            (corpus / "hex.txt").write_text(
                "The omega program refund memo is semantically close to deadline evidence and supplier notes.",
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)

                sphere_count = conn.execute("SELECT COUNT(*) AS count FROM sphere_summaries").fetchone()["count"]
                tree_count = conn.execute("SELECT COUNT(*) AS count FROM biorag_tree_nodes").fetchone()["count"]
                hex_count = conn.execute("SELECT COUNT(*) AS count FROM hex_neighbors").fetchone()["count"]
                hex_cell_count = conn.execute("SELECT COUNT(*) AS count FROM biorag_hex_cells").fetchone()["count"]
                chunk_cell_count = conn.execute("SELECT COUNT(*) AS count FROM biorag_chunk_cells").fetchone()["count"]
                self.assertGreater(sphere_count, 0)
                self.assertGreater(tree_count, 0)
                self.assertGreater(hex_count, 0)
                self.assertGreater(hex_cell_count, 0)
                self.assertGreaterEqual(chunk_cell_count, 2)

                query_id, hits = biorag_search(conn, "omega refund intelligence deadline evidence", top_k=2)
                self.assertTrue(hits)
                run = conn.execute(
                    "SELECT * FROM biorag_retrieval_runs WHERE query_id = ?",
                    (query_id,),
                ).fetchone()
                status = biorag_status(conn)
                adaptive_count = conn.execute("SELECT COUNT(*) AS count FROM adaptive_path_stats").fetchone()["count"]
                edge_count = conn.execute("SELECT COUNT(*) AS count FROM biorag_edge_weights").fetchone()["count"]
                max_edge = conn.execute("SELECT MAX(weight) AS weight FROM biorag_edge_weights").fetchone()["weight"]
                self.assertIsNotNone(run)
                self.assertTrue(status["ready"])
                self.assertGreater(adaptive_count, 0)
                self.assertGreater(edge_count, 0)
                self.assertGreater(float(max_edge), 1.0)
            finally:
                conn.close()

    def test_biorag_energy_budget_scales_with_complexity(self) -> None:
        cheap = energy_budget("refund memo", top_k=3)
        deep = energy_budget(
            "compare deadline table values across multiple pdfs and trace related contradictions",
            top_k=3,
        )
        self.assertLess(cheap.complexity, deep.complexity)
        self.assertLessEqual(cheap.graph_depth, deep.graph_depth)

    def test_benchmark_harness_compares_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "bench.txt").write_text(
                "Benchmark evidence says BioRAG should recover omega deadline proof.",
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)
                report = run_benchmark(
                    conn,
                    [
                        BenchmarkCase(
                            case_id="bench-1",
                            question="omega deadline proof",
                            expected_sources=["bench.txt"],
                            expected_terms=["omega", "deadline"],
                        )
                    ],
                    modes=["vector", "biorag"],
                    top_k=2,
                )
                self.assertIn("biorag", report["summary"])
                self.assertGreaterEqual(report["summary"]["biorag"]["hit_at_k"], 1.0)
            finally:
                conn.close()

    def test_benchmark_term_support_handles_wording_variants(self) -> None:
        haystack = "The voice recording transcript says the screenshot was reviewed in the final report."

        self.assertTrue(term_supported("voice recordings", haystack))
        self.assertTrue(term_supported("screenshot review", haystack))
        self.assertTrue(term_supported("field report", "Field dashboard evidence appears in the final report."))

    def test_evaluator_sanitizes_nan_official_scores(self) -> None:
        value, used_fallback = official_metric_or_fallback({"faithfulness": math.nan}, "faithfulness", 0.875)
        self.assertEqual(value, 0.875)
        self.assertTrue(used_fallback)

        rows = [
            CaseResult(
                case_id="nan-case",
                mode="hiverag",
                latency_ms=10.0,
                hit_at_k=1.0,
                mrr=1.0,
                source_recall=1.0,
                term_recall=1.0,
                evidence_count=1,
                matched_sources=["a.txt"],
                matched_terms=["alpha"],
                returned_sources=["a.txt"],
                returned_contexts=["alpha"],
                context_precision_proxy=math.nan,
                context_recall_proxy=1.0,
                faithfulness_proxy=0.875,
                answer_relevancy_proxy=0.75,
            )
        ]

        summary = summarize_results(rows)
        framework = evaluator_framework_summary(rows)
        self.assertEqual(summary["hiverag"]["ragas_context_precision_proxy"], 0.0)
        self.assertEqual(summary["hiverag"]["ragas_faithfulness_proxy"], 0.875)
        self.assertFalse(math.isnan(framework["ragas"]["hiverag"]["overall_proxy"]))

    def test_large_benchmark_builder_exports_axiom_and_ragas_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_dataset(
                corpus_dir=root / "corpus",
                axiom_path=root / "stress_eval.jsonl",
                ragas_path=root / "stress_ragas.jsonl",
                topics=2,
                files_per_topic=4,
                clean=True,
            )

            self.assertEqual(summary["files"], 8)
            self.assertGreaterEqual(summary["cases"], 10)
            self.assertEqual(len(list((root / "corpus").glob("*.txt"))), 8)
            cases = (root / "stress_eval.jsonl").read_text(encoding="utf-8").splitlines()
            ragas_rows = (root / "stress_ragas.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(cases), summary["cases"])
            self.assertEqual(len(ragas_rows), summary["ragas_samples"])
            first_ragas = json.loads(ragas_rows[0])
            self.assertIn("user_input", first_ragas)
            self.assertIn("reference_contexts", first_ragas)

    def test_analytics_builds_graph_timeline_and_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "annexure.txt").write_text(
                "Annexure 4B says international development targets for 2024 require screenshot review.",
                encoding="utf-8",
            )
            (corpus / "call.txt").write_text(
                "At 00:14:32 the operational plan shifted and the field review compared the screenshot.",
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)
                result = build_analytics(conn, query="development 2024 screenshot", limit=10)
                self.assertGreater(result["metrics"]["graph_nodes"], 0)
                self.assertGreater(result["metrics"]["timeline_items"], 0)
                self.assertIn("forecast", result["prediction"])
                self.assertTrue(result["prediction"]["next_actions"])
            finally:
                conn.close()

    def test_unknown_citation_is_flagged(self) -> None:
        answer = "This claim cites a fake source. [Axiom:deadbeef]"
        cleaned = validate_citations(answer, {"cafebabe"})
        self.assertIn("[Unverified Claim]", cleaned)

    def test_workstation_file_tools_and_guarded_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "field_report.txt").write_text("offline screenshot evidence", encoding="utf-8")

            scanned = scan_folder(root, max_depth=1)
            self.assertTrue(any(item.name == "field_report.txt" for item in scanned))

            matches = find_files(root, "screenshot", content=True)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].name, "field_report.txt")

            dry = run_command(["python", "--version"], execute=False)
            self.assertTrue(dry.allowed)
            self.assertFalse(dry.executed)

            blocked = run_command(["powershell", "Remove-Item", "x"], execute=True)
            self.assertFalse(blocked.allowed)
            self.assertFalse(blocked.executed)

            plan = plan_operator_task(f"find screenshot in {root}")
            self.assertEqual(plan["action"], "find")

    def test_dependency_audit_reports_required_setup(self) -> None:
        audit = audit_dependencies()
        self.assertIn("checks", audit)
        self.assertGreaterEqual(audit["required_total"], 1)
        pillow = dependency_by_key("pillow")
        self.assertIsNotNone(pillow)
        self.assertEqual(pillow.category, "python")

    def test_image_generation_status_and_missing_model_failure(self) -> None:
        status = image_generation_status()
        self.assertIn("automatic1111", status)
        self.assertIn("diffusers", status)
        result = generate_image(
            ImageGenerationRequest(
                prompt="offline dashboard",
                backend="diffusers",
                model_path="C:/definitely/missing/model",
            )
        )
        self.assertFalse(result.success)
        self.assertIn("local model", result.error.lower())

    def test_investigation_builds_dossier_and_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "person.txt").write_text(
                "Jordan Vale can be reached at jordan@example.test. In 2024 Jordan Vale reviewed the field screenshot.",
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)
                result = investigate_subject(conn, "Jordan Vale", roots=[str(corpus)])
                self.assertGreater(result["confidence"], 0)
                self.assertTrue(result["evidence"])
                self.assertIn("jordan@example.test", result["entities"]["emails"])
                self.assertIn("supported", result["hallucination_guard"]["status"])
                validation = validate_investigation_answer("Jordan Vale is verified.", result["hallucination_guard"]["allowed_citations"])
                self.assertFalse(validation["safe"])
            finally:
                conn.close()

    def test_case_report_exports_markdown_html_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "case.txt").write_text(
                "Jordan Vale appears in a 2024 field review and references a screenshot.",
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)
                result = generate_case_report(conn, "Jordan Vale", roots=[str(corpus)], output_dir=root / "reports")
                self.assertTrue(Path(result.markdown_path).exists())
                self.assertTrue(Path(result.html_path).exists())
                self.assertTrue(Path(result.json_path).exists())
                self.assertIn("Axiom Case Report", Path(result.markdown_path).read_text(encoding="utf-8"))
                self.assertIn("<html", Path(result.html_path).read_text(encoding="utf-8"))
            finally:
                conn.close()

    def test_audio_transcript_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "call.wav"
            audio.write_bytes(b"RIFF")
            (root / "call.wav.transcript.json").write_text(
                '{"segments":[{"start":872,"end":902,"text":"The 2024 plan references Annexure 4B."}]}',
                encoding="utf-8",
            )
            conn = connect(root / "axiom.sqlite")
            try:
                report = ingest_path(conn, audio)

                self.assertEqual(len(report.indexed_files), 1)
                row = conn.execute(
                    "SELECT start_timestamp FROM content_chunks WHERE modality = 'transcript' AND chunk_kind = 'child'"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["start_timestamp"], "00:14:32.000")
            finally:
                conn.close()

    def test_docx_xml_fallback_and_doc_file_type(self) -> None:
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docx = root / "brief.docx"
            xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>Offline DOCX evidence for SIH25231.</w:t></w:r></w:p></w:body>
</w:document>"""
            with zipfile.ZipFile(docx, "w") as package:
                package.writestr("word/document.xml", xml)

            segments = extract_docx_xml(docx)
            self.assertEqual(file_type_for(Path("legacy.doc")), "doc")
            self.assertIn("SIH25231", segments[0].text)

    def test_pdf_and_doc_sidecar_fallbacks_work_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "annexure.pdf"
            pdf.write_text("%PDF placeholder for staged offline text extraction", encoding="utf-8")
            Path(str(pdf) + ".txt").write_text("PDF sidecar text mentions NTRO multimodal RAG.", encoding="utf-8")
            doc = root / "legacy.doc"
            doc.write_text("legacy binary placeholder", encoding="utf-8")
            Path(str(doc) + ".txt").write_text("DOC sidecar text mentions voice recordings.", encoding="utf-8")

            self.assertIn("multimodal RAG", extract_segments(pdf)[0].text)
            self.assertIn("voice recordings", extract_segments(doc)[0].text)

    def test_mission_brief_scores_multimodal_sih_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "field.txt").write_text("NTRO SIH25231 field report mentions 2024 screenshot evidence.", encoding="utf-8")
            image = corpus / "screen.png"
            image.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
                )
            )
            (corpus / "screen.png.ocr.txt").write_text("Screenshot OCR mentions the same SIH25231 field report.", encoding="utf-8")
            audio = corpus / "call.wav"
            audio.write_bytes(b"RIFF")
            (corpus / "call.wav.transcript.txt").write_text("Voice transcript confirms 2024 screenshot review.", encoding="utf-8")

            conn = connect(root / "axiom.sqlite")
            try:
                ingest_path(conn, corpus)
                brief = build_mission_brief(conn)
                coverage = {item["key"]: item for item in brief["coverage"]}

                self.assertEqual(brief["problem"]["ps_id"], "SIH25231")
                self.assertFalse(coverage["doc"]["ready"])
                self.assertFalse(coverage["pdf"]["ready"])
                self.assertTrue(coverage["image"]["ready"])
                self.assertTrue(coverage["audio"]["ready"])
                self.assertGreaterEqual(brief["corpus"]["documents"], 3)
                self.assertGreater(brief["corpus"]["cross_modal_links"], 0)
                self.assertTrue(brief["differentiators"])
                self.assertGreaterEqual(brief["score"], 60)
            finally:
                conn.close()

    def test_vision_analysis_uses_ocr_sidecar_and_ingests_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "screen.png"
            image.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
                )
            )
            (root / "screen.png.ocr.txt").write_text(
                "Axiom dashboard shows international development 2024 screenshot evidence.",
                encoding="utf-8",
            )

            ocr = run_ocr(image, engine="auto")
            self.assertEqual(ocr.engine, "sidecar")
            self.assertIn("international development", ocr.text)

            analysis = analyze_image(image, use_vlm=False)
            self.assertIn("OCR-only visual analysis", analysis.visual_summary)
            self.assertTrue(Path(analysis.sidecars["caption"]).exists())

            conn = connect(root / "axiom.sqlite")
            try:
                report = ingest_visual_analysis(conn, analysis)
                self.assertEqual(len(report.indexed_files), 1)
                result = answer_query(conn, "development screenshot evidence", top_k=2)
                self.assertIn("[Axiom:", result.answer)
            finally:
                conn.close()

    def test_clipboard_data_url_is_saved_as_screenshot_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_url = (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
            )
            saved = save_pasted_image(data_url, root, file_name="../snip.png")

            self.assertEqual(saved.capture_method, "clipboard-paste")
            self.assertTrue(Path(saved.image_path).exists())
            self.assertEqual(Path(saved.image_path).parent, root.resolve())
            self.assertEqual(Path(saved.image_path).name, "snip.png")

    def test_uploaded_files_preserve_safe_folder_structure_and_can_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = base64.b64encode(b"Folder evidence references Annexure 4B and the 2024 screenshot.").decode("ascii")
            batch = save_uploaded_files(
                [
                    {
                        "name": "note.txt",
                        "relative_path": "../case pack/note.txt",
                        "data_url": f"data:text/plain;base64,{text}",
                    }
                ],
                root,
                batch_name="picked-folder",
            )

            saved = Path(batch.files[0].saved_path)
            self.assertTrue(saved.exists())
            self.assertEqual(batch.files[0].relative_path, "case pack\\note.txt")
            self.assertEqual(batch.root_path, str((root / "picked-folder").resolve()))

            conn = connect(root / "axiom.sqlite")
            try:
                report = ingest_path(conn, batch.root_path)
                self.assertEqual(len(report.indexed_files), 1)
                answer = answer_query(conn, "2024 screenshot", top_k=1)
                self.assertIn("[Axiom:", answer.answer)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
