from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
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
from .embeddings import normalize_token, tokenize
from .retrieval import SearchHit, avtr_search, dense_search, make_hit, reciprocal_rank_fusion, rerank, search


DEFAULT_MODES = ("vector", "hybrid", "tree", "graph", "avtr", "hiverag")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_terms: list[str] = field(default_factory=list)
    notes: str = ""
    reference: str = ""


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
                    reference=str(data.get("reference", "")),
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
    progress: bool = False,
) -> dict[str, object]:
    results: list[CaseResult] = []
    total_runs = len(cases) * len(modes)
    if progress:
        print(f"Evaluating {len(cases)} case(s) across {len(modes)} mode(s): {total_runs} retrieval run(s).", flush=True)
    completed = 0
    for case_index, case in enumerate(cases, start=1):
        for mode in modes:
            completed += 1
            if progress:
                print(
                    f"Evaluating {completed}/{total_runs}: case={case.case_id} "
                    f"({case_index}/{len(cases)}), mode={mode}",
                    flush=True,
                )
            start = time.perf_counter()
            hits = retrieve_for_mode(conn, case.question, mode=mode, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0
            results.append(score_case(case, mode, hits, latency_ms=latency_ms))
            
    evaluator_status = {
        "requested": evaluator,
        "model": evaluator_model,
        "ragas_llm_judge_used": False,
        "fallback_metric_cells": 0,
    }
    if evaluator in ("ollama", "openai"):
        results, evaluator_status = apply_official_ragas(cases, results, evaluator, evaluator_model)
        
    return {
        "cases": [case.__dict__ for case in cases],
        "results": [result.__dict__ for result in results],
        "summary": summarize_results(results),
        "evaluator_availability": evaluator_availability(),
        "evaluator_status": evaluator_status,
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
    returned_contexts = [evaluation_context(hit) for hit in hits]
    combined = "\n".join(returned_contexts).lower()
    matched_sources = sorted(
        {
            expected
            for expected in case.expected_sources
            if any(expected in hit.file_name.lower() or expected in hit.file_path.lower() for hit in hits)
        }
    )
    matched_terms = sorted({term for term in case.expected_terms if term_supported(term, combined)})
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
        haystack = f"{hit.file_name}\n{hit.file_path}\n{evaluation_context(hit)}".lower()
        source_match = any(expected in hit.file_name.lower() or expected in hit.file_path.lower() for expected in case.expected_sources)
        term_match = any(term_supported(term, haystack) for term in case.expected_terms)
        if case.expected_sources:
            if source_match:
                return rank
            continue
        if term_match:
            return rank
    return None


def summarize_results(results: list[CaseResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)
    summary: dict[str, dict[str, float]] = {}
    for mode, rows in grouped.items():
        summary[mode] = {
            "hit_at_k": _avg(row.hit_at_k for row in rows),
            "mrr": _avg(row.mrr for row in rows),
            "source_recall": _avg(row.source_recall for row in rows),
            "term_recall": _avg(row.term_recall for row in rows),
            "ragas_context_precision_proxy": _avg(row.context_precision_proxy for row in rows),
            "ragas_context_recall_proxy": _avg(row.context_recall_proxy for row in rows),
            "ragas_faithfulness_proxy": _avg(row.faithfulness_proxy for row in rows),
            "ragas_answer_relevancy_proxy": _avg(row.answer_relevancy_proxy for row in rows),
            "avg_latency_ms": _avg((row.latency_ms for row in rows), digits=3),
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


def _avg(values, *, digits: int = 4) -> float:
    rows = [float(value) for value in values if is_finite_number(value)]
    if not rows:
        return 0.0
    return round(sum(rows) / len(rows), digits)


def is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def ollama_base_url() -> str:
    configured = os.environ.get("AXIOM_OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
    if not configured:
        configured = os.environ.get("AXIOM_OLLAMA_URL", "")
    if not configured:
        return "http://127.0.0.1:11434"
    normalized = configured.rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api/embeddings", "/api/embed", "/api/tags"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized.rstrip("/")


def preflight_ollama(base_url: str, *, model: str, embedding_model: str) -> str | None:
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    timeout = float(os.environ.get("AXIOM_RAGAS_PREFLIGHT_TIMEOUT", "5"))
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return (
            f"Ollama is not reachable at {base_url}. Start the Ollama server in this runtime "
            "or set AXIOM_OLLAMA_BASE_URL to the reachable evaluator endpoint. "
            f"Original error: {exc}"
        )

    available = {
        str(item.get("name") or item.get("model") or "")
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }
    missing = [
        candidate
        for candidate in (model, embedding_model)
        if available and not ollama_model_available(candidate, available)
    ]
    if missing:
        return (
            f"Ollama is reachable at {base_url}, but missing model(s): {', '.join(missing)}. "
            f"Run `ollama pull {missing[0]}` or set --evaluator-model / AXIOM_RAGAS_EMBEDDING_MODEL."
        )
    return None


def ollama_model_available(model: str, available: set[str]) -> bool:
    if model in available:
        return True
    if ":" not in model and f"{model}:latest" in available:
        return True
    return False


def apply_official_ragas(
    cases: list[BenchmarkCase],
    results: list[CaseResult],
    evaluator: str,
    evaluator_model: str | None,
) -> tuple[list[CaseResult], dict[str, object]]:
    status: dict[str, object] = {
        "requested": evaluator,
        "model": evaluator_model,
        "ragas_llm_judge_used": False,
        "fallback_metric_cells": 0,
    }
    try:
        import sys, types
        m = types.ModuleType('langchain_community.chat_models.vertexai')
        m.ChatVertexAI = type('ChatVertexAI', (), {})
        sys.modules['langchain_community.chat_models.vertexai'] = m
        import langchain_community.llms
        if not hasattr(langchain_community.llms, 'VertexAI'):
            langchain_community.llms.VertexAI = type('VertexAI', (), {})
            
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
        
        if evaluator == "ollama":
            from langchain_ollama import ChatOllama
            from langchain_ollama import OllamaEmbeddings
            model = evaluator_model or "qwen2.5:7b"
            base_url = ollama_base_url()
            embedding_model = os.environ.get("AXIOM_RAGAS_EMBEDDING_MODEL", "nomic-embed-text")
            preflight_error = preflight_ollama(base_url, model=model, embedding_model=embedding_model)
            status["model"] = model
            status["base_url"] = base_url
            status["embedding_model"] = embedding_model
            if preflight_error:
                status["error"] = preflight_error
                print(f"Ragas evaluator skipped: {preflight_error}")
                return results, status
            llm = ChatOllama(model=model, base_url=base_url)
            embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        else:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            model = evaluator_model or "gpt-4o-mini"
            base_url = os.environ.get("AXIOM_OPENAI_BASE_URL")
            status["model"] = model
            status["base_url"] = base_url or "openai-default"
            llm_kwargs = {"model": model}
            embedding_kwargs = {}
            if base_url:
                llm_kwargs["base_url"] = base_url
                embedding_kwargs["base_url"] = base_url
            llm = ChatOpenAI(**llm_kwargs)
            embeddings = OpenAIEmbeddings(**embedding_kwargs)
            
    except ImportError as e:
        print(f"Warning: Could not import required eval libraries. Install with pip install -e .[eval] langchain langchain-community. Error: {e}")
        status["error"] = str(e)
        return results, status

    grouped = {}
    for res in results:
        grouped.setdefault(res.mode, []).append(res)
        
    case_map = {c.case_id: c for c in cases}
    updated_results = []
    
    from dataclasses import replace
    
    for mode, mode_results in grouped.items():
        samples = []
        for res in mode_results:
            case = case_map[res.case_id]
            samples.append(SingleTurnSample(**ragas_sample_payload(case, res)))
            
        dataset = EvaluationDataset(samples=samples, name=f"axiom-{mode}")
        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        
        print(
            f"Running actual Ragas evaluation for {mode} mode with {evaluator} "
            f"(model={evaluator_model or 'default'})...",
            flush=True,
        )
        try:
            from ragas.run_config import RunConfig
            run_config = RunConfig(
                timeout=float(os.environ.get("AXIOM_RAGAS_TIMEOUT", "300")),
                max_retries=int(os.environ.get("AXIOM_RAGAS_MAX_RETRIES", "1")),
                max_workers=int(os.environ.get("AXIOM_RAGAS_MAX_WORKERS", "2")),
            )
            eval_result = evaluate(
                dataset, 
                metrics=metrics, 
                llm=llm, 
                embeddings=embeddings, 
                run_config=run_config, 
                raise_exceptions=False
            )
            df = eval_result.to_pandas()
            fallback_count = 0
            
            for i, res in enumerate(mode_results):
                row = df.iloc[i]
                context_precision_value, used_context_precision_fallback = official_metric_or_fallback(
                    row, "context_precision", res.context_precision_proxy
                )
                context_recall_value, used_context_recall_fallback = official_metric_or_fallback(
                    row, "context_recall", res.context_recall_proxy
                )
                faithfulness_value, used_faithfulness_fallback = official_metric_or_fallback(
                    row, "faithfulness", res.faithfulness_proxy
                )
                answer_relevancy_value, used_answer_relevancy_fallback = official_metric_or_fallback(
                    row, "answer_relevancy", res.answer_relevancy_proxy
                )
                fallback_count += sum(
                    [
                        used_context_precision_fallback,
                        used_context_recall_fallback,
                        used_faithfulness_fallback,
                        used_answer_relevancy_fallback,
                    ]
                )
                new_res = replace(
                    res,
                    context_precision_proxy=context_precision_value,
                    context_recall_proxy=context_recall_value,
                    faithfulness_proxy=faithfulness_value,
                    answer_relevancy_proxy=answer_relevancy_value,
                )
                updated_results.append(new_res)
            status["ragas_llm_judge_used"] = True
            status["fallback_metric_cells"] = int(status["fallback_metric_cells"]) + fallback_count
            if fallback_count:
                print(
                    f"Ragas returned {fallback_count} invalid metric value(s) for {mode}; "
                    "kept deterministic proxy scores for those cells."
                )
        except Exception as e:
            print(f"Ragas evaluation failed: {e}")
            status["error"] = str(e)
            updated_results.extend(mode_results)
            
    return updated_results, status


def official_metric_or_fallback(row: object, metric: str, fallback: float) -> tuple[float, bool]:
    value = row.get(metric, fallback) if hasattr(row, "get") else fallback
    if is_finite_number(value):
        return round(float(value), 4), False
    if is_finite_number(fallback):
        return round(float(fallback), 4), True
    return 0.0, True


def ragas_sample_payload(case: BenchmarkCase, result: CaseResult) -> dict[str, object]:
    reference = ground_truth_for_case(case)
    return {
        "user_input": case.question,
        "retrieved_contexts": result.returned_contexts,
        "reference_contexts": [reference],
        "response": evaluation_answer(case.question, result.returned_contexts),
        "reference": reference,
        "rubrics": {
            "expected_sources": ", ".join(case.expected_sources),
            "expected_terms": ", ".join(case.expected_terms),
            "case_id": case.case_id,
            "mode": result.mode,
        },
    }


def context_precision_proxy(case: BenchmarkCase, hits: list[SearchHit]) -> float:
    if not hits:
        return 0.0
    relevant = 0
    for hit in hits:
        haystack = f"{hit.file_name}\n{hit.file_path}\n{evaluation_context(hit)}".lower()
        source_match = any(expected in hit.file_name.lower() or expected in hit.file_path.lower() for expected in case.expected_sources)
        term_match = any(term_supported(term, haystack) for term in case.expected_terms)
        if source_match or term_match:
            relevant += 1
    return relevant / len(hits)


def faithfulness_proxy(case: BenchmarkCase, hits: list[SearchHit]) -> float:
    if not case.expected_terms:
        return 0.0
    combined = "\n".join(evaluation_context(hit) for hit in hits).lower()
    supported = sum(1 for term in case.expected_terms if term_supported(term, combined))
    return supported / len(case.expected_terms)


def answer_relevancy_proxy(case: BenchmarkCase, hits: list[SearchHit]) -> float:
    question_terms = {normalize_token(token) for token in tokenize(case.question) if len(token) >= 4}
    if not question_terms:
        return 0.0
    combined = "\n".join(evaluation_context(hit) for hit in hits).lower()
    combined_terms = {normalize_token(token) for token in tokenize(combined)}
    matched = sum(1 for term in question_terms if term in combined_terms)
    return matched / len(question_terms)


def evaluation_context(hit: SearchHit) -> str:
    return "\n".join(
        [
            f"Source: {hit.file_name}",
            f"Evidence kind: {evaluation_kind(hit)}",
            f"Modality: {hit.modality}",
            f"Location: {hit.location}",
            f"Text: {hit.text}",
            f"Snippet: {hit.snippet}",
        ]
    )


def evaluation_kind(hit: SearchHit) -> str:
    name = hit.file_name.lower()
    modality = hit.modality.lower()
    if modality == "transcript" or name.endswith((".wav", ".mp3", ".m4a", ".ogg")):
        return "voice transcript"
    if modality == "ocr" or name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "screenshot OCR"
    if name.endswith((".doc", ".docx")):
        return "document"
    if name.endswith(".pdf"):
        return "PDF report"
    return hit.modality


def term_supported(term: str, haystack: str) -> bool:
    lowered_term = term.lower()
    lowered_haystack = haystack.lower()
    if lowered_term in lowered_haystack:
        return True
    term_tokens = {normalize_token(token) for token in tokenize(lowered_term)}
    if not term_tokens:
        return False
    haystack_tokens = {normalize_token(token) for token in tokenize(lowered_haystack)}
    return term_tokens <= haystack_tokens


def evaluation_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "I do not know based on the retrieved context."
    combined = "\n".join(contexts)
    sources = context_sources(contexts)
    supported_terms = [
        term
        for term in evaluation_query_terms(question)
        if term_supported(term, combined)
    ]
    excerpts = relevant_context_excerpts(contexts, supported_terms, limit=2)
    parts: list[str] = []
    if sources:
        parts.append(f"Retrieved source evidence includes {', '.join(sources[:3])}.")
    if supported_terms:
        parts.append(f"Supported details present in the retrieved evidence: {', '.join(supported_terms[:10])}.")
    if excerpts:
        parts.append("Evidence excerpts: " + " | ".join(excerpts))
    if not parts:
        return compact_context(contexts[0], limit=520)
    return " ".join(parts)


def context_sources(contexts: list[str]) -> list[str]:
    sources: list[str] = []
    for context in contexts:
        for line in context.splitlines():
            if not line.startswith("Source:"):
                continue
            source = line.split(":", 1)[1].strip()
            if source and source not in sources:
                sources.append(source)
            break
    return sources


def evaluation_query_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize(question):
        normalized = normalize_token(token)
        if len(normalized) < 4 or normalized in EVALUATION_ANSWER_STOP_TERMS:
            continue
        if normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


def relevant_context_excerpts(contexts: list[str], supported_terms: list[str], *, limit: int) -> list[str]:
    scored: list[tuple[int, str]] = []
    for context in contexts:
        compact = compact_context(context, limit=420)
        score = sum(1 for term in supported_terms if term_supported(term, compact))
        scored.append((score, compact))
    return [
        excerpt
        for score, excerpt in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
        if score > 0 or not supported_terms
    ]


EVALUATION_ANSWER_STOP_TERMS = {
    "about",
    "answer",
    "based",
    "connect",
    "discuss",
    "evidence",
    "record",
    "records",
    "retrieved",
    "source",
    "sources",
    "which",
}


def ground_truth_for_case(case: BenchmarkCase) -> str:
    if case.reference.strip():
        return case.reference.strip()
    parts = []
    if case.expected_sources:
        parts.append(f"Expected source evidence: {', '.join(case.expected_sources)}.")
    if case.expected_terms:
        parts.append(f"Expected supported terms: {', '.join(case.expected_terms)}.")
    return " ".join(parts) or "No ground truth supplied."


def compact_context(text: str, *, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


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
    evaluator_status = report.get("evaluator_status", {})
    framework_summary = report.get("framework_summary", {})
    ragas_judge_used = isinstance(evaluator_status, dict) and bool(evaluator_status.get("ragas_llm_judge_used"))
    ragas_label = "RAGAS LLM Judge Summary" if ragas_judge_used else "RAGAS Proxy Summary"
    ragas_metric_label = "Context Precision" if ragas_judge_used else "Context Precision Proxy"
    faithfulness_label = "Faithfulness" if ragas_judge_used else "Faithfulness Proxy"
    lines = [
        "# HiveRAG Benchmark Results",
        "",
        "These numbers are generated by the local Axiom benchmark harness. Treat small local runs as smoke-test evidence, not publication-grade final results.",
        "",
        "Official RAGAS package runs require optional dependencies and an evaluator LLM. When no RAGAS evaluator is requested or available, this report records deterministic offline proxy metrics mapped to the same metric families. TruLens and DeepEval sections are proxy crosswalks unless those frameworks are run directly.",
        "",
        "## Summary",
        "",
        f"| Mode | Hit@k | MRR | Source Recall | Term Recall | {ragas_metric_label} | {faithfulness_label} | Avg Latency ms |",
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
    if isinstance(evaluator_status, dict) and evaluator_status.get("requested"):
        lines.extend(
            [
                "",
                "## Evaluator Run",
                "",
                f"- Requested evaluator: {evaluator_status.get('requested')}",
                f"- Model: {evaluator_status.get('model') or 'default'}",
                f"- RAGAS LLM judge used: {bool(evaluator_status.get('ragas_llm_judge_used'))}",
                f"- Fallback metric cells: {evaluator_status.get('fallback_metric_cells', 0)}",
            ]
        )
        if evaluator_status.get("error"):
            lines.append(f"- Evaluator error: {evaluator_status.get('error')}")
    if isinstance(framework_summary, dict):
        ragas = framework_summary.get("ragas", {})
        lines.extend(
            [
                "",
                f"## {ragas_label}",
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
                f"## TruLens {'LLM Judge' if ragas_judge_used else 'Proxy'} Crosswalk Summary",
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
                f"## DeepEval {'LLM Judge' if ragas_judge_used else 'Proxy'} Crosswalk Summary",
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
