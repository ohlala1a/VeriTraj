"""Action interface: generated text -> grounded operator + character span."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pddl.model import Domain, GroundAction
from .store import FactStore

UNKNOWN_ARG = "<unknown>"

_OBJ = r"([a-z][a-z_\-]*(?:\s+[a-z][a-z_\-]*)*?(?:[\s_]+\d+)?)"
_ALF_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("goto", re.compile(rf"^go\s+to\s+(?:the\s+)?{_OBJ}$")),
    ("open", re.compile(rf"^open\s+(?:the\s+)?{_OBJ}$")),
    ("close", re.compile(rf"^close\s+(?:the\s+)?{_OBJ}$")),
    ("take", re.compile(rf"^(?:take|pick\s+up)\s+(?:the\s+)?{_OBJ}\s+from\s+(?:the\s+)?{_OBJ}$")),
    ("put", re.compile(rf"^(?:put|place|move)\s+(?:the\s+)?{_OBJ}\s+(?:in/on|in|on|into|onto)\s+(?:the\s+)?{_OBJ}$")),
    ("heat", re.compile(rf"^heat\s+(?:the\s+)?{_OBJ}\s+(?:with|using)\s+(?:the\s+)?{_OBJ}$")),
    ("cool", re.compile(rf"^cool\s+(?:the\s+)?{_OBJ}\s+(?:with|using)\s+(?:the\s+)?{_OBJ}$")),
    ("clean", re.compile(rf"^clean\s+(?:the\s+)?{_OBJ}\s+(?:with|using)\s+(?:the\s+)?{_OBJ}$")),
    ("toggle", re.compile(rf"^(?:use|toggle|turn\s+on)\s+(?:the\s+)?{_OBJ}$")),
    ("examine", re.compile(rf"^(?:examine|look\s+at)\s+(?:the\s+)?{_OBJ}\s+(?:with|under)\s+(?:the\s+)?{_OBJ}$")),
    ("look", re.compile(r"^look$")),
    ("inventory", re.compile(r"^(?:inventory|inv)$")),
]
_FUNC = re.compile(r"([A-Za-z][A-Za-z_\-]*)\s*\(([^()]*)\)")
_ACTION_TAG = re.compile(r"action\s*[:：]", re.IGNORECASE)
_END_MARKERS = re.compile(r"(?:(?:<\|(?:eot_id|im_end|endoftext|end_of_text)\|>|</s>)\s*)+$")


@dataclass(slots=True)
class ParsedAction:
    ok: bool
    name: str | None = None
    args: tuple[str, ...] = ()
    char_span: tuple[int, int] = (0, 0)
    error: str | None = None

    def canonical(self) -> str:
        return f"{self.name}({', '.join(self.args)})" if self.ok else "<parse-failure>"


@dataclass(slots=True)
class InterfaceConfig:
    synonyms: dict[str, str] = field(default_factory=dict)
    implicit: dict[str, dict[int, tuple[str, ...]]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "InterfaceConfig":
        d = d or {}
        implicit = {op: {int(k): tuple(v) for k, v in spec.items()}
                    for op, spec in (d.get("implicit") or {}).items()}
        return cls(synonyms=dict(d.get("synonyms") or {}), implicit=implicit,
                   aliases={k.lower(): v for k, v in (d.get("aliases") or {}).items()})


def canonical_object(mention: str, aliases: dict[str, str] | None = None) -> str:
    """'Desk Lamp 1' -> 'desklamp_1', 'mug_2' -> 'mug_2'."""
    s = re.sub(r"\s+", " ", mention.strip().lower())
    m = re.match(r"^(.*?)[\s_]*(\d+)$", s)
    base, idx = (m.group(1), m.group(2)) if m else (s, None)
    base = base.strip()
    if aliases and base in aliases:
        base = aliases[base]
    base = base.replace(" ", "")
    return f"{base}_{idx}" if idx is not None else base


class ActionInterface:
    def __init__(self, domain: Domain, config: InterfaceConfig | None = None):
        self.domain = domain
        self.config = config or InterfaceConfig()

    def _region(self, text: str) -> tuple[int, str]:
        tags = list(_ACTION_TAG.finditer(text))
        if tags:
            start = tags[-1].end()
            end = text.find("\n", start)
            end = len(text) if end < 0 else end
            return start, _END_MARKERS.sub("", text[start:end])
        lines = [(m.start(), m.group(0)) for m in re.finditer(r"[^\n]+", text) if m.group(0).strip()]
        return (lines[-1][0], _END_MARKERS.sub("", lines[-1][1])) if lines else (0, "")

    def parse(self, text: str, store: FactStore | None = None) -> ParsedAction:
        offset, region = self._region(text)
        found = self._parse_functional(region, offset) or self._parse_alfworld(region, offset)
        if found is None:
            return ParsedAction(False, char_span=(0, len(text)), error="no command found")
        name, args, span = found
        name = self.config.synonyms.get(name, name)
        schema = self.domain.actions.get(name)
        if schema is None:
            return ParsedAction(False, char_span=(0, len(text)), error=f"unknown operator '{name}'")
        args = self._fill_implicit(name, args, schema.arity, store)
        if len(args) != schema.arity:
            return ParsedAction(False, char_span=(0, len(text)),
                                error=f"'{name}' expects {schema.arity} args, got {len(args)}")
        return ParsedAction(True, name, tuple(args), span)

    def _parse_functional(self, region: str, offset: int):
        matches = list(_FUNC.finditer(region))
        if not matches:
            return None
        m = matches[-1]
        name = m.group(1).lower().replace("-", "_")
        name = self.config.synonyms.get(name, name)
        if name not in self.domain.actions:
            name = name.replace("_", "-")
        raw = [a for a in (x.strip() for x in m.group(2).split(",")) if a]
        args = [canonical_object(a, self.config.aliases) for a in raw]
        return name, args, (offset + m.start(), offset + m.end())

    def _parse_alfworld(self, region: str, offset: int):
        stripped = region.strip().rstrip(".").strip()
        if not stripped:
            return None
        lead = len(region) - len(region.lstrip())
        low = stripped.lower()
        for name, pat in _ALF_PATTERNS:
            m = pat.match(low)
            if m:
                args = [canonical_object(g, self.config.aliases) for g in m.groups()]
                start = offset + lead
                return name, args, (start, start + len(stripped))
        return None

    def _fill_implicit(self, name: str, args: list[str], arity: int, store: FactStore | None) -> list[str]:
        spec = self.config.implicit.get(name)
        if not spec or len(args) != arity - len(spec):
            return args
        out, it = [], iter(args)
        for i in range(arity):
            if i in spec:
                pred, *key = spec[i]
                atom = store.lookup(pred, *key) if store is not None else None
                out.append(atom.args[-1] if atom is not None else UNKNOWN_ARG)
            else:
                out.append(next(it))
        return out

    @staticmethod
    def mention(obj: str) -> str:
        m = re.match(r"^(.*)_(\d+)$", obj)
        return f"{m.group(1)} {m.group(2)}" if m else obj

    def render(self, action: GroundAction | tuple[str, tuple[str, ...]], style: str = "alfworld") -> str:
        name, args = (action.name, action.args) if isinstance(action, GroundAction) else action
        if style == "functional":
            return f"{name.upper()}({', '.join(args)})"
        m = [self.mention(a) for a in args]
        return {
            "goto": lambda: f"go to {m[1]}",
            "open": lambda: f"open {m[0]}",
            "close": lambda: f"close {m[0]}",
            "take": lambda: f"take {m[0]} from {m[1]}",
            "put": lambda: f"put {m[0]} in/on {m[1]}",
            "heat": lambda: f"heat {m[0]} with {m[1]}",
            "cool": lambda: f"cool {m[0]} with {m[1]}",
            "clean": lambda: f"clean {m[0]} with {m[1]}",
            "toggle": lambda: f"use {m[0]}",
            "examine": lambda: f"examine {m[0]} with {m[1]}",
            "look": lambda: "look",
            "inventory": lambda: "inventory",
        }.get(name, lambda: f"{name} {' '.join(m)}")()
