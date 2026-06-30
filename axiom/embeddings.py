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


def embed_text(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> dict[int, float]:
    counts = Counter(tokenize(text))
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
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    text_set = set(tokenize(text))
    return len(query_set & text_set) / len(query_set)
