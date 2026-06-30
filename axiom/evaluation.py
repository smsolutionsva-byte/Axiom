from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from .biorag import (
    biorag_search,
    coverage_select,
    ensure_biorag_index,
    hex_expand,
    propagate_web_signal,
    sphere_route,
    tree_route,
)
from .database import lexical_search
from .retrieval import SearchHit, avtr_search, dense_search, make_hit, reciprocal_rank_fusion, rerank, search


DEFAULT_MODES = ("vector", "hybrid", "tree", "graph", "avtr", "hiverag")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_terms: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    mode: str
    latency_ms: float
    hit_at_k: float
    mrr: float
    source_recall: float
    term_recall: float
    evidence_count: int
    matched_sources: list[str]
    matched_terms: list[str]
    returned_sources: list[str]
    returned_contexts: list[str]
    context_precision_proxy: float
    context_recall_proxy: float
    faithfulness_proxy: float
    answer_relevancy_proxy: float


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data = json.loads(stripped)
            cases.append(
                BenchmarkCase(
                    case_id=str(data.get("id") or f"case-{index}"),
                    question=str(data["question"]),
                    expected_sources=[str(item).lower() for item in data.get("expected_sources", [])],
                    expected_terms=[str(item).lower() for item in data.get("expected_terms", [])],
                    notes=str(data.get("notes", "")),
                )
            )
    return cases


def run_benchmark(
    conn: sqlite3.Connection,
    cases: list[BenchmarkCase],
    *,
    modes: list[str] | tuple[str, ...] = DEFAULT_MODES,
    top_k: int = 5,
    evaluator: str | None = None,
    evaluator_model: str | None = None,
) -> dict[str, object]:
    results: list[CaseResult] = []
    for case in cases:
        for mode in modes:
            start = time.perf_counter()
            hits = retrieve_for_mode(conn, case.question, mode=mode, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0
            results.append(score_case(case, mode, hits, latency_ms=latency_ms))
            
    if evaluator in ("ollama", "openai"):
        results = apply_official_ragas(cases, results, evaluator, evaluator_model)
        
    return {
        "cases": [case.__dict__ for case in cases],
        "results": [result.__dict__ for result in results],
        "summary": summarize_results(results),
        "evaluator_availability": evaluator_availability(),
        "framework_summary": evaluator_framework_summary(results),
    }


def retrieve_for_mode(conn: sqlite3.Connection, query: str, *, mode: str, top_k: int) -> list[SearchHit]:
    normalized = mode.strip().lower()
    if normalized == "vector":
        ranked = dense_search(conn, query, limit=max(top_k * 4, 20))
        selected = rerank(conn, query, ranked)[:top_k]
        return hits_from_ranked(conn, query, selected)
    if normalized == "lexical":
        ranked = lexical_search(conn, query, limit=max(top_k * 4, 20))
        selected = rerank(conn, query, ranked)[:top_k]
        return hits_from_ranked(conn, query, selected)
    if normalized in {"hybrid", "classic"}:
        _query_id, hits = search(conn, query, top_k=top_k)
        return hits
    if normalized == "avtr":
        _query_id, hits = avtr_search(conn, query, top_k=top_k)
        return hits
    if normalized == "tree":
        ensure_biorag_index(conn)
        sphere = sphere_route(conn, query, limit=max(top_k * 3, 12))
        tree = tree_route(conn, query, sphere, limit=max(top_k * 4, 20))
        selected = coverage_select(conn, query, rerank(conn, query, tree), top_k=top_k)
        return hits_from_ranked(conn, query, selected)
    if normalized in {"graph", "graphrag-lite"}:
        ensure_biorag_index(conn)
        dense = dense_search(conn, query, limit=max(top_k * 6, 30))
        sparse = lexical_search(conn, query, limit=max(top_k * 6, 30))
        seeded = rerank(conn, query, reciprocal_rank_fusion([dense, sparse])[: max(top_k * 8, 40)])
        graph = propagate_web_signal(conn, seeded, iterations=3, max_candidates=max(top_k * 16, 80))
        selected = coverage_select(conn, query, rerank(conn, query, graph), top_k=top_k)
        return hits_from_ranked(conn, query, selected)
    if normalized == "hex":
        ensure_biorag_index(conn)
        seeded = rerank(conn, query, dense_search(conn, query, limit=max(top_k * 6, 30)))
        selected = coverage_select(conn, query, rerank(conn, query, hex_expand(conn, seeded, frontier=18, limit=80)), top_k=top_k)
        return hits_from_ranked(conn, query, selected)
    if normalized in {"biorag", "hiverag"}:
        _query_id, hits = biorag_search(conn, query, top_k=top_k)
        return hits
    raise ValueError(f"Unknown benchmark mode: {mode}")


def hits_from_ranked(conn: sqlite3.Connection, query: str, ranked: list[tuple[str, float]]) -> list[SearchHit]:
    hits = [make_hit(conn, chunk_id, score, query) for chunk_id, score in ranked]
    return [hit for hit in hits if hit is not None]


def score_case(case: BenchmarkCase, mode: str, hits: list[SearchHit], *, latency_ms: float) -> CaseResult:
    returned_sources = [hit.file_name for hit in hits]
    returned_contexts = [f"{hit.text}\n{hit.snippet}" for hit in hits]
    combined = "\n".join(returned_contexts).lower()
    matched_sources = sorted(
        {
            expected
            for expected in case.expected_sources
            if any(expected in hit.file_name.lower() or expected in hit.file_path.lower() for hit in hits)
        }
    )
    matched_terms = sorted({term for term in case.expected_terms if term in combined})
    source_recall = len(matched_sources) / len(case.expected_sources) if case.expected_sources else 0.0
    term_recall = len(matched_terms) / len(case.expected_terms) if case.expected_terms else 0.0
    context_precision = context_precision_proxy(case, hits)
    faithfulness = faithfulness_proxy(case, hits)
    answer_relevancy = answer_relevancy_proxy(case, hits)
    first_relevant = first_relevant_rank(case, hits)
    mrr = 1.0 / first_relevant if first_relevant else 0.0
    hit_at_k = 1.0 if first_relevant else 0.0
    return CaseResult(
        case_id=case.case_id,
        mode=mode,
        latency_ms=round(latency_ms, 3),
        hit_at_k=hit_at_k,
        mrr=round(mrr, 4),
        source_recall=round(source_recall, 4),
        term_recall=round(term_recall, 4),
        evidence_count=len(hits),
        matched_sources=matched_sources,
        matched_terms=matched_terms,
        returned_sources=returned_sources,
        returned_contexts=returned_contexts,
        context_precision_proxy=round(context_precision, 4),
        context_recall_proxy=round(term_recall, 4),
        faithfulness_proxy=round(faithfulness, 4),
        answer_relevancy_proxy=round(answer_relevancy, 4),
    )


def first_relevant_rank(case: BenchmarkCase, hits: list[SearchHit]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        haystack = f"{hit.file_name}\n{hit.file_path}\n{hit.text}\n{hit.snippet}".lower()
        source_match = any(expected in hit.file_name.lower() or expected in hit.file_path.lower() for expected in case.expected_sources)
        term_match = any(term in haystack for term in case.expected_terms)
        if source_match or term_match:
            return rank
    return None


def summarize_results(results: list[CaseResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)
    summary: dict[str, dict[str, float]] = {}
    for mode, rows in grouped.items():
        count = max(len(rows), 1)
        summary[mode] = {
            "hit_at_k": round(sum(row.hit_at_k for row in rows) / count, 4),
            "mrr": round(sum(row.mrr for row in rows) / count, 4),
            "source_recall": round(sum(row.source_recall for row in rows) / count, 4),
            "term_recall": round(sum(row.term_recall for row in rows) / count, 4),
            "ragas_context_precision_proxy": round(sum(row.context_precision_proxy for row in rows) / count, 4),
            "ragas_context_recall_proxy": round(sum(row.context_recall_proxy for row in rows) / count, 4),
            "ragas_faithfulness_proxy": round(sum(row.faithfulness_proxy for row in rows) / count, 4),
            "ragas_answer_relevancy_proxy": round(sum(row.answer_relevancy_proxy for row in rows) / count, 4),
            "avg_latency_ms": round(sum(row.latency_ms for row in rows) / count, 3),
        }
    return summary


def evaluator_framework_summary(results: list[CaseResult]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)
    framework_summary: dict[str, dict[str, dict[str, float]]] = {
        "ragas": {},
        "trulens": {},
        "deepeval": {},
    }
    for mode, rows in grouped.items():
        count = max(len(rows), 1)
        context_precision = _avg(row.context_precision_proxy for row in rows)
        context_recall = _avg(row.context_recall_proxy for row in rows)
        faithfulness = _avg(row.faithfulness_proxy for row in rows)
        answer_relevancy = _avg(row.answer_relevancy_proxy for row in rows)
        context_relevance = round((context_precision + answer_relevancy) / 2.0, 4)
        contextual_relevancy = round((context_precision + context_recall + answer_relevancy) / 3.0, 4)
        framework_summary["ragas"][mode] = {
            "context_precision_proxy": context_precision,
            "context_recall_proxy": context_recall,
            "faithfulness_proxy": faithfulness,
            "answer_relevancy_proxy": answer_relevancy,
            "overall_proxy": _avg([context_precision, context_recall, faithfulness, answer_relevancy]),
        }
        framework_summary["trulens"][mode] = {
            "context_relevance_proxy": context_relevance,
            "groundedness_proxy": faithfulness,
            "answer_relevance_proxy": answer_relevancy,
            "rag_triad_proxy": _avg([context_relevance, faithfulness, answer_relevancy]),
        }
        framework_summary["deepeval"][mode] = {
            "contextual_precision_proxy": context_precision,
            "contextual_recall_proxy": context_recall,
            "contextual_relevancy_proxy": contextual_relevancy,
            "faithfulness_proxy": faithfulness,
            "answer_relevancy_proxy": answer_relevancy,
            "overall_proxy": _avg([context_precision, context_recall, contextual_relevancy, faithfulness, answer_relevancy]),
        }
        assert count >= 1
    return framework_summary


def _avg(values) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return round(sum(rows) / len(rows), 4)


def apply_official_ragas(cases: list[BenchmarkCase], results: list[CaseResult], evaluator: str, evaluator_model: str | None) -> list[CaseResult]:
    try:
        import sys, types
        m = types.ModuleType('langchain_community.chat_models.vertexai')
        m.ChatVertexAI = type('ChatVertexAI', (), {})
        sys.modules['langchain_community.chat_models.vertexai'] = m
        import langchain_community.llms
        if not hasattr(langchain_community.llms, 'VertexAI'):
            langchain_community.llms.VertexAI = type('VertexAI', (), {})
            
        from ragas import evaluate
        from datasets import Dataset
        from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
        
        if evaluator == "ollama":
            from langchain_ollama import ChatOllama
            from langchain_ollama import OllamaEmbeddings
            llm = ChatOllama(model=evaluator_model or "qwen2.5:7b")
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
        else:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            llm = ChatOpenAI(model=evaluator_model or "gpt-4o-mini")
            embeddings = OpenAIEmbeddings()
            
    except ImportError as e:
        print(f"Warning: Could not import required eval libraries. Install with pip install -e .[eval] langchain langchain-community. Error: {e}")
        return results

    grouped = {}
    for res in results:
        grouped.setdefault(res.mode, []).append(res)
        
    case_map = {c.case_id: c for c in cases}
    updated_results = []
    
    from dataclasses import replace
    import pandas as pd
    
    for mode, mode_results in grouped.items():
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        for res in mode_results:
            case = case_map[res.case_id]
            data["question"].append(case.question)
            data["answer"].append(" ".join(res.matched_terms) if res.matched_terms else "I don't know based on the context.")
            data["contexts"].append(res.returned_contexts)
            data["ground_truth"].append(" ".join(case.expected_terms) if case.expected_terms else "None")
            
        dataset = Dataset.from_dict(data)
        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        
        print(f"Running actual Ragas evaluation for {mode} mode with {evaluator} (model={evaluator_model or 'default'})...")
        try:
            from ragas.run_config import RunConfig
            run_config = RunConfig(timeout=300, max_retries=5, max_workers=2)
            eval_result = evaluate(
                dataset, 
                metrics=metrics, 
                llm=llm, 
                embeddings=embeddings, 
                run_config=run_config, 
                raise_exceptions=False
            )
            df = eval_result.to_pandas()
            
            for i, res in enumerate(mode_results):
                row = df.iloc[i]
                new_res = replace(
                    res,
                    context_precision_proxy=row.get("context_precision", res.context_precision_proxy),
                    context_recall_proxy=row.get("context_recall", res.context_recall_proxy),
                    faithfulness_proxy=row.get("faithfulness", res.faithfulness_proxy),
                    answer_relevancy_proxy=row.get("answer_relevancy", res.answer_relevancy_proxy),
                )
                updated_results.append(new_res)
        except Exception as e:
            print(f"Ragas evaluation failed: {e}")
            updated_results.extend(mode_results)
            
    return updated_results


def context_precision_proxy(case: BenchmarkCase, hits: list[SearchHit]) -> float:
    if not hits:
        return 0.0
    relevant = 0
    for hit in hits:
        haystack = f"{hit.file_name}\n{hit.file_path}\n{hit.text}\n{hit.snippet}".lower()
        source_match = any(expected in hit.file_name.lower() or expected in hit.file_path.lower() for expected in case.expected_sources)
        term_match = any(term in haystack for term in case.expected_terms)
        if source_match or term_match:
            relevant += 1
    return relevant / len(hits)


def faithfulness_proxy(case: BenchmarkCase, hits: list[SearchHit]) -> float:
    if not case.expected_terms:
        return 0.0
    combined = "\n".join(f"{hit.text}\n{hit.snippet}" for hit in hits).lower()
    supported = sum(1 for term in case.expected_terms if term in combined)
    return supported / len(case.expected_terms)


def answer_relevancy_proxy(case: BenchmarkCase, hits: list[SearchHit]) -> float:
    question_terms = {token for token in case.question.lower().replace("?", " ").split() if len(token) >= 4}
    if not question_terms:
        return 0.0
    combined = "\n".join(f"{hit.text}\n{hit.snippet}" for hit in hits).lower()
    matched = sum(1 for term in question_terms if term in combined)
    return matched / len(question_terms)


def official_ragas_available() -> bool:
    return find_spec("ragas") is not None and find_spec("datasets") is not None


def official_trulens_available() -> bool:
    return find_spec("trulens") is not None


def official_deepeval_available() -> bool:
    return find_spec("deepeval") is not None


def evaluator_availability() -> dict[str, dict[str, object]]:
    return {
        "ragas": {
            "official_available": official_ragas_available(),
            "required_packages": ["ragas", "datasets"],
            "local_track": "offline proxy",
            "metrics": ["context_precision", "context_recall", "faithfulness", "answer_relevancy"],
        },
        "trulens": {
            "official_available": official_trulens_available(),
            "required_packages": ["trulens"],
            "local_track": "offline proxy",
            "metrics": ["context_relevance", "groundedness", "answer_relevance"],
        },
        "deepeval": {
            "official_available": official_deepeval_available(),
            "required_packages": ["deepeval"],
            "local_track": "offline proxy",
            "metrics": ["contextual_precision", "contextual_recall", "contextual_relevancy", "faithfulness", "answer_relevancy"],
        },
    }


def write_benchmark_report(report: dict[str, object], output_dir: str | Path, *, label: str = "biorag") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{label}_benchmark_results.json"
    md_path = out / f"{label}_benchmark_results.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def markdown_report(report: dict[str, object]) -> str:
    summary = report.get("summary", {})
    results = report.get("results", [])
    availability = report.get("evaluator_availability", {})
    framework_summary = report.get("framework_summary", {})
    lines = [
        "# HiveRAG Benchmark Results",
        "",
        "These numbers are generated by the local Axiom benchmark harness. Treat small local runs as smoke-test evidence, not publication-grade final results.",
        "",
        "Official RAGAS, TruLens, and DeepEval package runs require their optional dependencies and an evaluator LLM. When those packages are unavailable, this report records deterministic offline proxy metrics mapped to the same metric families.",
        "",
        "## Summary",
        "",
        "| Mode | Hit@k | MRR | Source Recall | Term Recall | Context Precision Proxy | Faithfulness Proxy | Avg Latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(summary, dict):
        for mode, metrics in summary.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                f"| {mode} | {metrics.get('hit_at_k', 0)} | {metrics.get('mrr', 0)} | "
                f"{metrics.get('source_recall', 0)} | {metrics.get('term_recall', 0)} | "
                f"{metrics.get('ragas_context_precision_proxy', 0)} | "
                f"{metrics.get('ragas_faithfulness_proxy', 0)} | {metrics.get('avg_latency_ms', 0)} |"
            )
    lines.extend(["", "## Evaluator Framework Availability", "", "| Framework | Official Package Available | Local Track | Metrics |", "| --- | --- | --- | --- |"])
    if isinstance(availability, dict):
        for framework, info in availability.items():
            if not isinstance(info, dict):
                continue
            metrics = ", ".join(str(item) for item in info.get("metrics", []))
            lines.append(
                f"| {framework} | {info.get('official_available', False)} | {info.get('local_track', 'offline proxy')} | {metrics} |"
            )
    if isinstance(framework_summary, dict):
        ragas = framework_summary.get("ragas", {})
        lines.extend(
            [
                "",
                "## RAGAS Proxy Summary",
                "",
                "| Mode | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Overall |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        _append_metric_rows(
            lines,
            ragas,
            ["context_precision_proxy", "context_recall_proxy", "faithfulness_proxy", "answer_relevancy_proxy", "overall_proxy"],
        )
        trulens = framework_summary.get("trulens", {})
        lines.extend(
            [
                "",
                "## TruLens Proxy Summary",
                "",
                "| Mode | Context Relevance | Groundedness | Answer Relevance | RAG Triad |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        _append_metric_rows(
            lines,
            trulens,
            ["context_relevance_proxy", "groundedness_proxy", "answer_relevance_proxy", "rag_triad_proxy"],
        )
        deepeval = framework_summary.get("deepeval", {})
        lines.extend(
            [
                "",
                "## DeepEval Proxy Summary",
                "",
                "| Mode | Contextual Precision | Contextual Recall | Contextual Relevancy | Faithfulness | Answer Relevancy | Overall |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        _append_metric_rows(
            lines,
            deepeval,
            [
                "contextual_precision_proxy",
                "contextual_recall_proxy",
                "contextual_relevancy_proxy",
                "faithfulness_proxy",
                "answer_relevancy_proxy",
                "overall_proxy",
            ],
        )
    lines.extend(["", "## Per-Case Results", "", "| Case | Mode | Hit@k | MRR | Sources | Terms | Returned Sources |", "| --- | --- | ---: | ---: | --- | --- | --- |"])
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {item.get('case_id')} | {item.get('mode')} | {item.get('hit_at_k')} | {item.get('mrr')} | "
                f"{', '.join(item.get('matched_sources', []))} | {', '.join(item.get('matched_terms', []))} | "
                f"{', '.join(item.get('returned_sources', []))} |"
            )
    lines.append("")
    return "\n".join(lines)


def _append_metric_rows(lines: list[str], metrics_by_mode, keys: list[str]) -> None:
    if not isinstance(metrics_by_mode, dict):
        return
    for mode, metrics in metrics_by_mode.items():
        if not isinstance(metrics, dict):
            continue
        cells = " | ".join(str(metrics.get(key, 0)) for key in keys)
        lines.append(f"| {mode} | {cells} |")
