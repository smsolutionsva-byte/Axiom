from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from axiom.evaluation import (
    BenchmarkCase,
    CaseResult,
    load_benchmark_cases,
    markdown_report,
    ollama_base_url,
    preflight_ollama,
    ragas_sample_payload,
    score_case,
)
from axiom.retrieval import SearchHit
from tools.build_beir_benchmark import safe_doc_filename


def hit(file_name: str, text: str) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-{file_name}",
        context_id=f"context-{file_name}",
        score=1.0,
        text=text,
        snippet=text,
        file_name=file_name,
        file_path=f"C:/tmp/{file_name}",
        sha256="abc",
        modality="text",
        location="File",
    )


class EvaluationMetricTests(unittest.TestCase):
    def test_hit_and_mrr_require_expected_source_when_provided(self) -> None:
        case = BenchmarkCase(
            case_id="source-strict",
            question="Which source records AXM-01-04 for KAVERI-14 and drip irrigation?",
            expected_sources=["dashboard_ocr.txt"],
            expected_terms=["axm-01-04", "kaveri-14", "drip irrigation"],
        )
        result = score_case(
            case,
            "vector",
            [
                hit("briefing.txt", "AXM-01-04 KAVERI-14 drip irrigation appears here."),
                hit("annex.txt", "KAVERI-14 drip irrigation appears here too."),
            ],
            latency_ms=1.0,
        )

        self.assertEqual(result.hit_at_k, 0.0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.source_recall, 0.0)
        self.assertEqual(result.term_recall, 1.0)

    def test_ragas_payload_response_omits_unsupported_query_terms(self) -> None:
        case = BenchmarkCase(
            case_id="grounded-answer",
            question="Which source records AXM-01-04 for KAVERI-14 and drip irrigation?",
            expected_sources=["dashboard_ocr.txt"],
            expected_terms=["axm-01-04", "kaveri-14", "drip irrigation"],
        )
        result = CaseResult(
            case_id="grounded-answer",
            mode="hiverag",
            latency_ms=1.0,
            hit_at_k=1.0,
            mrr=1.0,
            source_recall=1.0,
            term_recall=0.6667,
            evidence_count=1,
            matched_sources=["dashboard_ocr.txt"],
            matched_terms=["axm-01-04", "kaveri-14"],
            returned_sources=["dashboard_ocr.txt"],
            returned_contexts=[
                "Source: dashboard_ocr.txt\nText: Dashboard OCR AXM-01-04 shows KAVERI-14 and storage alert."
            ],
            context_precision_proxy=1.0,
            context_recall_proxy=0.6667,
            faithfulness_proxy=0.6667,
            answer_relevancy_proxy=0.8,
        )

        response = str(ragas_sample_payload(case, result)["response"]).lower()

        self.assertIn("axm-01-04", response)
        self.assertIn("kaveri-14", response)
        self.assertNotIn("drip irrigation", response)

    def test_loader_preserves_beir_reference_for_ragas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "beir_eval.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "beir-1",
                        "question": "What evidence supports the trial result?",
                        "expected_sources": ["doc__abc.txt"],
                        "reference": "Gold BEIR document text with the trial result.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            case = load_benchmark_cases(path)[0]

        self.assertEqual(case.reference, "Gold BEIR document text with the trial result.")
        payload = ragas_sample_payload(
            case,
            CaseResult(
                case_id=case.case_id,
                mode="hiverag",
                latency_ms=1.0,
                hit_at_k=1.0,
                mrr=1.0,
                source_recall=1.0,
                term_recall=0.0,
                evidence_count=1,
                matched_sources=["doc__abc.txt"],
                matched_terms=[],
                returned_sources=["doc__abc.txt"],
                returned_contexts=["Source: doc__abc.txt\nText: Retrieved trial result."],
                context_precision_proxy=1.0,
                context_recall_proxy=1.0,
                faithfulness_proxy=1.0,
                answer_relevancy_proxy=1.0,
            ),
        )
        self.assertEqual(payload["reference"], case.reference)
        self.assertEqual(payload["reference_contexts"], [case.reference])

    def test_beir_doc_ids_become_safe_unique_filenames(self) -> None:
        url_name = safe_doc_filename("https://example.org/papers/a/b?x=1")
        similar_name = safe_doc_filename("https_example.org_papers_a_b_x_1")

        self.assertTrue(url_name.endswith(".txt"))
        self.assertNotIn("/", url_name)
        self.assertNotIn("\\", url_name)
        self.assertNotIn(":", url_name)
        self.assertNotEqual(url_name, similar_name)

    def test_markdown_labels_actual_ragas_judge_separately_from_crosswalks(self) -> None:
        report = {
            "summary": {},
            "results": [],
            "evaluator_availability": {},
            "evaluator_status": {
                "requested": "ollama",
                "model": "llama3",
                "ragas_llm_judge_used": True,
                "fallback_metric_cells": 0,
            },
            "framework_summary": {"ragas": {}, "trulens": {}, "deepeval": {}},
        }

        markdown = markdown_report(report)

        self.assertIn("RAGAS LLM Judge Summary", markdown)
        self.assertIn("TruLens LLM Judge Crosswalk Summary", markdown)
        self.assertIn("DeepEval LLM Judge Crosswalk Summary", markdown)

    def test_ollama_base_url_normalizes_api_paths(self) -> None:
        with patch.dict(os.environ, {"AXIOM_OLLAMA_BASE_URL": "http://127.0.0.1:11434/api/chat"}, clear=False):
            self.assertEqual(ollama_base_url(), "http://127.0.0.1:11434")

    def test_ollama_preflight_reports_unreachable_endpoint(self) -> None:
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no server")):
            error = preflight_ollama(
                "http://127.0.0.1:11434",
                model="llama3",
                embedding_model="nomic-embed-text",
            )

        self.assertIsNotNone(error)
        self.assertIn("Ollama is not reachable", str(error))


if __name__ == "__main__":
    unittest.main()
