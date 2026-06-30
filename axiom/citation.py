from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[Axiom:([a-f0-9]{8,64})\]")


def citation_token(chunk_id: str) -> str:
    return f"[Axiom:{chunk_id}]"


def extract_citations(text: str) -> list[str]:
    return CITATION_RE.findall(text)


def validate_citations(answer: str, allowed_chunk_ids: set[str]) -> str:
    cleaned_lines: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue

        citations = extract_citations(stripped)
        if not citations:
            cleaned_lines.append(f"{line} [Unverified Claim]")
            continue

        unknown = [chunk_id for chunk_id in citations if chunk_id not in allowed_chunk_ids]
        if unknown:
            cleaned_lines.append(f"{line} [Unverified Claim]")
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)
