from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from .biorag import biorag_search
from .citation import citation_token, extract_citations, validate_citations
from .embeddings import lexical_overlap, tokenize
from .retrieval import SearchHit, avtr_search, search


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    answer: str
    sources: list[dict[str, object]]


@dataclass(frozen=True)
class ChatResult:
    response: str
    model: str | None
    used_model: bool
    error: str | None = None


def answer_query(conn: sqlite3.Connection, query: str, *, top_k: int = 5) -> QueryResult:
    retrieval_mode = os.environ.get("AXIOM_RETRIEVAL_MODE", "biorag").strip().lower()
    if retrieval_mode in {"classic", "hybrid"}:
        retriever = search
    elif retrieval_mode == "avtr":
        retriever = avtr_search
    else:
        retriever = biorag_search
    query_id, hits = retriever(conn, query, top_k=top_k)
    allowed = {hit.chunk_id for hit in hits}
    if not hits:
        return QueryResult(
            query_id=query_id,
            answer="No matching local evidence was found. [Unverified Claim]",
            sources=[],
        )

    model_answer = try_ollama_answer(query, hits)
    validated = validate_citations(model_answer, allowed) if model_answer else ""
    if not validated or "[Unverified Claim]" in validated or not cited_claims_are_grounded(validated, hits):
        validated = validate_citations(extractive_answer(query, hits), allowed)
    return QueryResult(
        query_id=query_id,
        answer=validated,
        sources=[source_payload(hit) for hit in hits],
    )


def extractive_answer(query: str, hits: list[SearchHit]) -> str:
    primary = hits[0]
    lines = [
        (
            f"Answer: Axiom found cited local evidence for '{query}'. The strongest match is "
            f"{primary.file_name} at {primary.location}. {citation_token(primary.chunk_id)}"
        )
    ]
    for index, hit in enumerate(hits, start=1):
        snippet = clean_excerpt(hit.snippet)
        lines.append(
            f"Evidence {index}: {hit.file_name} ({evidence_kind_label(hit)}, {hit.location}) says: "
            f"\"{snippet}\" {citation_token(hit.chunk_id)}"
        )
    if len(hits) >= 2:
        pair = hits[:2]
        lines.append(
            f"Cross-check: review {pair[0].file_name} with {pair[1].file_name} because HiveRAG retrieved both for the same question. "
            f"{citation_token(pair[0].chunk_id)} {citation_token(pair[1].chunk_id)}"
        )
    return "\n".join(lines)


def clean_excerpt(text: str, limit: int = 420) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def try_ollama_answer(query: str, hits: list[SearchHit]) -> str | None:
    candidate_models = evidence_answer_models()
    if not candidate_models:
        return None

    context_blocks = "\n\n".join(
        evidence_context_block(hit, index=index)
        for index, hit in enumerate(hits, start=1)
    )
    prompt = f"""You are an offline secure evidence assistant.
Use only the context blocks below.
Every factual sentence must include one of the provided citation tokens exactly.
Keep the answer concise and copy important wording from the evidence.
Do not use one citation to support a claim that appears only in another context.
If the evidence is insufficient, say what is missing and cite the closest context.

{context_blocks}

User query: {query}
"""
    for model in candidate_models:
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            os.environ.get("AXIOM_OLLAMA_URL", "http://127.0.0.1:11434/api/generate"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=float(os.environ.get("AXIOM_OLLAMA_TIMEOUT", "45")),
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue
        answer = str(data.get("response", "")).strip()
        if answer:
            return answer
    return None


def evidence_context_block(hit: SearchHit, *, index: int) -> str:
    return "\n".join(
        [
            f"Context {index}",
            f"Citation: {citation_token(hit.chunk_id)}",
            f"Source: {hit.file_name}",
            f"Evidence kind: {evidence_kind_label(hit)}",
            f"Location: {hit.location}",
            f"Text: {hit.text}",
        ]
    )


def evidence_kind_label(hit: SearchHit) -> str:
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


def cited_claims_are_grounded(answer: str, hits: list[SearchHit]) -> bool:
    if not answer.strip():
        return False
    contexts = {
        hit.chunk_id: f"{hit.file_name} {hit.modality} {evidence_kind_label(hit)} {hit.location} {hit.text} {hit.snippet}"
        for hit in hits
    }
    checked = 0
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        citations = extract_citations(stripped)
        if not citations:
            return False
        claim = remove_citations(stripped)
        claim_tokens = [
            token
            for token in tokenize(claim)
            if len(token) >= 4 and token not in CLAIM_STOP_TERMS
        ]
        if not claim_tokens:
            continue
        support = " ".join(contexts.get(chunk_id, "") for chunk_id in citations)
        if lexical_overlap(claim_tokens, support) < 0.34:
            return False
        checked += 1
    return checked > 0


def remove_citations(text: str) -> str:
    for citation in extract_citations(text):
        text = text.replace(citation_token(citation), "")
    return text


CLAIM_STOP_TERMS = {
    "answer",
    "axiom",
    "because",
    "cited",
    "claim",
    "context",
    "evidence",
    "found",
    "local",
    "match",
    "matching",
    "question",
    "retrieved",
    "review",
    "source",
    "strongest",
    "this",
    "with",
}


def evidence_answer_models() -> list[str]:
    if os.environ.get("AXIOM_DISABLE_MODEL_ANSWERS", "").strip().lower() in {"1", "true", "yes"}:
        return []
    for key in ("AXIOM_EVIDENCE_MODEL", "AXIOM_OLLAMA_MODEL", "AXIOM_CHAT_MODEL"):
        value = os.environ.get(key)
        if value:
            return [value]
    return default_chat_models()


def chat_with_ollama(message: str, *, history: list[dict[str, str]] | None = None, model: str | None = None) -> ChatResult:
    prompt = message.strip()
    if not prompt:
        return ChatResult("Ask me something and I will help.", None, False, "No message supplied.")

    candidate_models = [model] if model else default_chat_models()
    candidate_models = [item for item in candidate_models if item]
    if not candidate_models:
        return ChatResult(
            "Local chat model is not ready yet. Install or pull an Ollama model, then refresh Setup.",
            None,
            False,
            "No Ollama chat model found.",
        )

    last_error = ""
    for selected_model in candidate_models:
        result = try_ollama_chat_model(prompt, selected_model, history=history)
        if result.used_model:
            return result
        last_error = result.error or last_error
    return ChatResult(
        "Local chat is not responding yet. Check Ollama in Setup, then try again.",
        candidate_models[0],
        False,
        last_error or "No candidate chat model responded.",
    )


def try_ollama_chat_model(message: str, model: str, *, history: list[dict[str, str]] | None = None) -> ChatResult:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Axiom, a concise offline intelligence workspace assistant. "
                "Be natural and helpful. For evidence claims, tell the user to ask a cited evidence question or use /sources. "
                "Do not invent local evidence or citations."
            ),
        }
    ]
    for item in (history or [])[-10:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:2000]})
    messages.append({"role": "user", "content": message})

    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        os.environ.get("AXIOM_OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat"),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("AXIOM_CHAT_TIMEOUT", "60"))) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return ChatResult("", model, False, body or str(exc))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return ChatResult("", model, False, str(exc))
    response_text = str((data.get("message") or {}).get("content") or data.get("response") or "").strip()
    if not response_text:
        return ChatResult("", model, False, "Empty Ollama response.")
    return ChatResult(response_text, model, True)


def default_chat_model() -> str | None:
    models = default_chat_models()
    return models[0] if models else None


def default_chat_models() -> list[str]:
    for key in ("AXIOM_CHAT_MODEL", "AXIOM_OLLAMA_MODEL", "AXIOM_VISION_MODEL"):
        value = os.environ.get(key)
        if value:
            return [value]
    models = installed_ollama_models()
    ordered: list[str] = []
    for preferred in ("qwen2.5:7b", "llama3.2-vision:latest", "llama3.2-vision"):
        if preferred in models:
            ordered.append(preferred)
    ordered.extend(item for item in models if item not in ordered)
    return ordered


def installed_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    models = data.get("models", []) if isinstance(data, dict) else []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def source_payload(hit: SearchHit) -> dict[str, object]:
    payload = asdict(hit)
    payload["citation"] = citation_token(hit.chunk_id)
    return payload
