"""Character-span -> token-span mapping.

Training code supervises token spans, the interface returns character spans.
`RegexTokenizer` is a dependency-free stand-in with the same offset API as a
HuggingFace fast tokenizer; `HFTokenizer` wraps the real one when the policy
model's tokenizer is available.
"""

from __future__ import annotations

import re
from typing import Protocol


class OffsetTokenizer(Protocol):
    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class RegexTokenizer:
    _pat = re.compile(r"\w+|[^\w\s]")

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in self._pat.finditer(text)]


class HFTokenizer:
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer  # optional dependency

        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(x) for x in enc["offset_mapping"]]


def char_to_token_span(offsets: list[tuple[int, int]], span: tuple[int, int]) -> tuple[int, int]:
    """Smallest [u, w) token range overlapping the character span."""
    s, e = span
    idx = [i for i, (a, b) in enumerate(offsets) if b > s and a < e]
    if not idx:
        return (0, 0)
    return (idx[0], idx[-1] + 1)


def make_tokenizer(spec: str | None) -> OffsetTokenizer:
    if not spec or spec == "regex":
        return RegexTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer spec '{spec}' (use 'regex' or 'hf:<model>')")
