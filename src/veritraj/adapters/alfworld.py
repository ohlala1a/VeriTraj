"""Observation adapter P_obs for ALFWorld-style text."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable

from ..interface import canonical_object
from ..pddl.model import Atom
from ..store import Delta, FactStore

AGENT = "agent"
ROOM = "room_1"


@dataclass(slots=True)
class Lexicon:
    receptacles: dict[str, dict] = field(default_factory=dict)
    lamps: set[str] = field(default_factory=set)
    items: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)
    paraphrases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Lexicon":
        return cls(
            receptacles={k: dict(v or {}) for k, v in (d.get("receptacles") or {}).items()},
            lamps=set(d.get("lamps") or []),
            items=set(d.get("items") or []),
            aliases={k.lower(): v for k, v in (d.get("aliases") or {}).items()},
            paraphrases=dict(d.get("paraphrases") or {}),
        )

    def classes(self) -> list[str]:
        return sorted(set(self.receptacles) | self.lamps | self.items)

    def kind(self, cls: str) -> str | None:
        if cls in self.receptacles:
            return "receptacle"
        if cls in self.lamps:
            return "lamp"
        if cls in self.items:
            return "item"
        return None


@dataclass(slots=True)
class GroundingStats:
    mentions: int = 0
    resolved: int = 0
    sentences: int = 0
    matched_sentences: int = 0

    def add(self, other: "GroundingStats") -> None:
        self.mentions += other.mentions
        self.resolved += other.resolved
        self.sentences += other.sentences
        self.matched_sentences += other.matched_sentences


_SENT = re.compile(r"[^.!?\n]+[.!?]?")
_LIST_SPLIT = re.compile(r",\s*(?:and\s+)?|\s+and\s+")
_ARTICLE = re.compile(r"^(?:a|an|the|some)\s+")

_T_START = re.compile(r"you are in the middle of a room")
_T_LOOK_AROUND = re.compile(r"looking quickly around you, you see (.+)")
_T_ARRIVE = re.compile(r"you arrive at (?:the\s+)?(.+)")
_T_ON = re.compile(r"(?:on|in) the (.+?), you see (.+)")
_T_STATE = re.compile(r"the (.+?) is (open|closed)")
_T_IN_IT = re.compile(r"in it, you see (.+)")
_T_PICK = re.compile(r"you pick up the (.+?) from the (.+)")
_T_PUT = re.compile(r"you (?:put|move) the (.+?) (?:in/on|in|on|into|onto) the (.+)")
_T_OPENED = re.compile(r"you open the (.+)")
_T_CLOSED = re.compile(r"you close the (.+)")
_T_HEAT = re.compile(r"you heat the (.+?) using the (.+)")
_T_COOL = re.compile(r"you cool the (.+?) using the (.+)")
_T_CLEAN = re.compile(r"you clean the (.+?) using the (.+)")
_T_TURN_ON = re.compile(r"you turn on the (.+)")
_T_CARRY = re.compile(r"you are carrying: (.+)|you carry: (.+)")
_T_EMPTY = re.compile(r"you are not carrying anything")
_T_NOTHING = re.compile(r"nothing happens")
_T_TASK = re.compile(r"your task is to")


class AlfworldAdapter:
    def __init__(self, lexicon: Lexicon, policy: str = "conservative",
                 resolver: Callable[[str, list[str]], str | None] | None = None,
                 initial_facts: list[str] | None = None, fuzzy_cutoff: float = 0.84):
        if policy not in {"conservative", "rule_norm", "llm_impute"}:
            raise ValueError(f"unknown grounding policy '{policy}'")
        if policy == "llm_impute" and resolver is None:
            raise ValueError("llm_impute needs a resolver (see veritraj.llm.make_class_resolver)")
        self.lex = lexicon
        self.policy = policy
        self.resolver = resolver
        self.initial_facts = initial_facts or ["handempty"]
        self.fuzzy_cutoff = fuzzy_cutoff
        self._cache: dict[str, str | None] = {}
        self.stats = GroundingStats()

    def resolve(self, mention: str) -> tuple[str | None, str | None]:
        """Return (object_name, class) or (None, None) if unresolved."""
        self.stats.mentions += 1
        name = canonical_object(_ARTICLE.sub("", mention.strip().lower()), self.lex.aliases)
        m = re.match(r"^(.*)_(\d+)$", name)
        if not m:
            return None, None
        cls, idx = m.group(1), m.group(2)
        resolved = cls if self.lex.kind(cls) else self._recover(cls)
        if resolved is None:
            return None, None
        self.stats.resolved += 1
        return f"{resolved}_{idx}", resolved

    def _recover(self, cls: str) -> str | None:
        if self.policy == "conservative":
            return None
        if cls in self._cache:
            return self._cache[cls]
        out = None
        cands = [cls.rstrip("s"), cls[:-2] if cls.endswith("es") else cls]
        for c in cands:
            if self.lex.kind(c):
                out = c
                break
        if out is None:
            close = difflib.get_close_matches(cls, self.lex.classes(), n=1, cutoff=self.fuzzy_cutoff)
            out = close[0] if close else None
        if out is None and self.policy == "llm_impute":
            guess = self.resolver(cls, self.lex.classes())  # type: ignore[misc]
            out = guess if guess and self.lex.kind(guess) else None
        self._cache[cls] = out
        return out

    def initial(self) -> Delta:
        d = Delta(types={ROOM: "room"})
        d.add.append(Atom("at", (AGENT, ROOM)))
        for f in self.initial_facts:
            d.add.append(Atom(f, ()))
        return d

    def _typed(self, d: Delta, mention: str) -> tuple[str | None, str | None]:
        obj, cls = self.resolve(mention)
        if obj is None:
            return None, None
        kind = self.lex.kind(cls)
        d.types[obj] = kind  # type: ignore[assignment]
        if kind == "receptacle":
            props = self.lex.receptacles[cls]
            d.add.append(Atom("visible", (obj,)))
            if props.get("openable"):
                d.add.append(Atom("openable", (obj,)))
            else:
                d.add.append(Atom("open", (obj,)))
            for cap in ("heat", "cool", "clean"):
                if props.get(cap):
                    d.add.append(Atom(f"can-{cap}", (obj,)))
        return obj, kind

    def _listing(self, d: Delta, store: FactStore | None, rec: str | None, text: str) -> None:
        if rec is None:
            return
        listed: set[str] = set()
        if text.strip() not in {"nothing", "nothing."}:
            for part in _LIST_SPLIT.split(text.rstrip(". ")):
                part = part.strip()
                if not part:
                    continue
                obj, kind = self._typed(d, part)
                if obj is None:
                    continue
                if kind == "item":
                    d.add.append(Atom("in", (obj, rec)))
                    listed.add(obj)
                elif kind == "lamp":
                    d.add.append(Atom("lamp-at", (obj, rec)))
        if store is not None:
            for atom in store.true:
                if atom.pred == "in" and atom.args[1] == rec and atom.args[0] not in listed:
                    d.delete.append(atom)

    def _normalize_sentence(self, s: str) -> str:
        s = s.strip().lower().rstrip(".!? ")
        if self.policy != "conservative":
            for src, dst in self.lex.paraphrases.items():
                s = s.replace(src, dst)
        return s

    @staticmethod
    def _transform(s: str):
        for pat, pred in ((_T_HEAT, "heated"), (_T_COOL, "cooled"), (_T_CLEAN, "cleaned")):
            m = pat.match(s)
            if m:
                return pred, m
        return None

    def ground(self, observation: str, store: FactStore | None = None, first: bool = False) -> Delta:
        d = self.initial() if first else Delta()
        here: str | None = None
        st = GroundingStats()
        for raw in _SENT.findall(observation):
            s = self._normalize_sentence(raw)
            if not s or s.startswith(("-=", "=-")) or not re.search(r"[a-z]", s):
                continue
            st.sentences += 1
            matched = True
            if _T_TASK.match(s):
                st.sentences -= 1
                continue
            if _T_START.match(s):
                pass
            elif m := _T_LOOK_AROUND.search(s):
                for part in _LIST_SPLIT.split(m.group(1)):
                    if part.strip():
                        self._typed(d, part)
            elif m := _T_ARRIVE.match(s):
                target = m.group(1)
                if not re.match(r"^loc \d+$", target):
                    obj, kind = self._typed(d, target)
                    if obj is not None:
                        here = obj
                        d.add.append(Atom("at", (AGENT, obj)))
            elif m := _T_ON.match(s):
                obj, _ = self._typed(d, m.group(1))
                if obj is not None and here is None:
                    here = obj
                    d.add.append(Atom("at", (AGENT, obj)))
                self._listing(d, store, obj, m.group(2))
            elif m := _T_STATE.match(s):
                obj, kind = self._typed(d, m.group(1))
                if obj is not None and kind == "receptacle":
                    d.add.append(Atom(m.group(2), (obj,)))
                    here = here or obj
            elif m := _T_IN_IT.match(s):
                self._listing(d, store, here, m.group(1))
            elif m := _T_PICK.match(s):
                o, _ = self._typed(d, m.group(1))
                r, _ = self._typed(d, m.group(2))
                if o and r:
                    d.add.append(Atom("in-hand", (o,)))
                    d.delete += [Atom("in", (o, r)), Atom("handempty", ())]
            elif m := _T_PUT.match(s):
                o, _ = self._typed(d, m.group(1))
                r, _ = self._typed(d, m.group(2))
                if o and r:
                    d.add += [Atom("in", (o, r)), Atom("handempty", ())]
                    d.delete.append(Atom("in-hand", (o,)))
            elif m := _T_OPENED.match(s):
                r, _ = self._typed(d, m.group(1))
                if r:
                    here = r
                    d.add.append(Atom("open", (r,)))
            elif m := _T_CLOSED.match(s):
                r, _ = self._typed(d, m.group(1))
                if r:
                    d.add.append(Atom("closed", (r,)))
            elif (hit := self._transform(s)) is not None:
                pred, m = hit
                o, _ = self._typed(d, m.group(1))
                if o:
                    d.add.append(Atom(pred, (o,)))
            elif m := _T_TURN_ON.match(s):
                lamp, kind = self._typed(d, m.group(1))
                if lamp and kind == "lamp":
                    d.add.append(Atom("lit", (lamp,)))
            elif m := _T_CARRY.match(s):
                for part in _LIST_SPLIT.split(m.group(1) or m.group(2)):
                    o, kind = self._typed(d, part) if part.strip() else (None, None)
                    if o and kind == "item":
                        d.add.append(Atom("in-hand", (o,)))
                        d.delete.append(Atom("handempty", ()))
            elif _T_EMPTY.match(s):
                d.add.append(Atom("handempty", ()))
                if store is not None:
                    d.delete += [a for a in store.true if a.pred == "in-hand"]
            elif _T_NOTHING.match(s):
                pass
            else:
                matched = False
            st.matched_sentences += int(matched)
        self.stats.sentences += st.sentences
        self.stats.matched_sentences += st.matched_sentences
        return d


def build_adapter(cfg: dict, policy: str | None = None, resolver=None) -> AlfworldAdapter:
    lex = Lexicon.from_dict(cfg["lexicon"])
    pol = policy or cfg.get("policy", "conservative")
    return AlfworldAdapter(lex, pol, resolver=resolver, initial_facts=cfg.get("initial_facts"))
