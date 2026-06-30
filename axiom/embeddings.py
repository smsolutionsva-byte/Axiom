from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}")
DEFAULT_DIMENSIONS = 1024
EMBEDDING_MODEL = "axiom-hashed-bow-v1"


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def normalize_token(token: str) -> str:
    lower = token.lower()
    if len(lower) > 5 and lower.endswith("ies"):
        return lower[:-3] + "y"
    if len(lower) > 5 and lower.endswith("ings"):
        return lower[:-1]
    if len(lower) > 4 and lower.endswith("es") and not lower.endswith(("ses", "xes")):
        return lower[:-2]
    if len(lower) > 4 and lower.endswith("s") and not lower.endswith(("ss", "us")):
        return lower[:-1]
    if len(lower) > 4 and lower.endswith("ed"):
        return lower[:-2]
    return lower


def expand_tokens(tokens: Iterable[str]) -> set[str]:
    expanded: set[str] = set()
    for token in tokens:
        normalized = normalize_token(token)
        expanded.add(token.lower())
        expanded.add(normalized)
    return expanded


def expanded_token_set(text: str) -> set[str]:
    return expand_tokens(tokenize(text))


def embed_text(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> dict[int, float]:
    counts: Counter[str] = Counter()
    for token in tokenize(text):
        counts[token] += 1.0
        normalized = normalize_token(token)
        if normalized != token:
            counts[normalized] += 0.65
    if not counts:
        return {}

    vector: dict[int, float] = {}
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] = vector.get(bucket, 0.0) + sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in vector.items()}


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def dump_vector(vector: dict[int, float]) -> str:
    return json.dumps({str(index): round(value, 8) for index, value in vector.items()}, separators=(",", ":"))


def load_vector(raw: str | None) -> dict[int, float]:
    if not raw:
        return {}
    data = json.loads(raw)
    return {int(index): float(value) for index, value in data.items()}


def lexical_overlap(query_tokens: Iterable[str], text: str) -> float:
    original_query = {token.lower() for token in query_tokens}
    normalized_query = {normalize_token(token) for token in original_query}
    if not original_query and not normalized_query:
        return 0.0
    original_text = set(tokenize(text))
    normalized_text = {normalize_token(token) for token in original_text}
    exact = len(original_query & original_text) / max(len(original_query), 1)
    normalized = len(normalized_query & normalized_text) / max(len(normalized_query), 1)
    return max(exact, normalized)
