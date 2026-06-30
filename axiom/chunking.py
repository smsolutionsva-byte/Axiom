from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .embeddings import tokenize


@dataclass(frozen=True)
class TextWindow:
    text: str
    char_start: int
    char_end: int
    token_count: int


def stable_id(*parts: object, length: int = 24) -> str:
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def _word_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokenize(text):
        index = text.lower().find(token, cursor)
        if index == -1:
            continue
        end = index + len(token)
        spans.append((index, end))
        cursor = end
    return spans


def make_windows(text: str, window_tokens: int, overlap_tokens: int) -> list[TextWindow]:
    clean = text.strip()
    if not clean:
        return []

    spans = _word_spans(clean)
    if not spans:
        return [TextWindow(text=clean, char_start=0, char_end=len(clean), token_count=0)]

    windows: list[TextWindow] = []
    step = max(1, window_tokens - overlap_tokens)
    start_word = 0
    while start_word < len(spans):
        end_word = min(len(spans), start_word + window_tokens)
        char_start = spans[start_word][0]
        char_end = spans[end_word - 1][1]
        windows.append(
            TextWindow(
                text=clean[char_start:char_end].strip(),
                char_start=char_start,
                char_end=char_end,
                token_count=end_word - start_word,
            )
        )
        if end_word == len(spans):
            break
        start_word += step
    return windows


def parent_child_windows(
    text: str,
    parent_tokens: int = 650,
    parent_overlap: int = 100,
    child_tokens: int = 180,
    child_overlap: int = 40,
) -> list[tuple[TextWindow, list[TextWindow]]]:
    parents = make_windows(text, parent_tokens, parent_overlap)
    result: list[tuple[TextWindow, list[TextWindow]]] = []
    for parent in parents:
        children = make_windows(parent.text, child_tokens, child_overlap)
        adjusted_children = [
            TextWindow(
                text=child.text,
                char_start=parent.char_start + child.char_start,
                char_end=parent.char_start + child.char_end,
                token_count=child.token_count,
            )
            for child in children
        ]
        result.append((parent, adjusted_children))
    return result
