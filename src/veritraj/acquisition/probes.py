"""Positive, state-perturbation and type-perturbation probes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from ..llm import ChatClient, extract_json
from ..pddl.model import Atom
from ..pddl.parser import parse_atom_text
from .environment import Environment


@dataclass(frozen=True)
class Probe:
    state: frozenset[Atom]
    name: str
    args: tuple[str, ...]
    objects_key: str
    expected: bool
    kind: str
    note: str = ""
    removed: Atom | None = None
    param: int | None = None

    def action_str(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


@dataclass
class DevTransition:
    state: frozenset[Atom]
    name: str
    args: tuple[str, ...]
    next_state: frozenset[Atom]
    objects_key: str


class DeterministicChallenger:
    def __init__(self, env: Environment, objects: dict[str, dict[str, str]], max_per_transition: int = 12,
                 seed: int = 0):
        self.env = env
        self.objects = objects
        self.max_per_transition = max_per_transition
        self.rng = random.Random(seed)

    def probes(self, transitions: list[DevTransition]) -> list[Probe]:
        out: list[Probe] = []
        seen: set[tuple] = set()
        for tr in transitions:
            objs = self.objects[tr.objects_key]
            cands: list[tuple] = [(tr.state, tr.args, "positive", "", None, None)]
            relevant = sorted((f for f in tr.state if not f.args or set(f.args) & set(tr.args)), key=str)
            self.rng.shuffle(relevant)
            for f in relevant[: self.max_per_transition]:
                cands.append((tr.state - {f}, tr.args, "state", f"removed {f}", f, None))
            for i, arg in enumerate(tr.args):
                typ = objs.get(arg)
                others = sorted(o for o, t in objs.items() if t != typ and o not in tr.args)
                if not others:
                    continue
                y = self.rng.choice(others)
                ren = frozenset(Atom(f.pred, tuple(y if a == arg else a for a in f.args)) for f in tr.state
                                if y not in f.args)
                args = tr.args[:i] + (y,) + tr.args[i + 1:]
                cands.append((ren, args, "type", f"{arg} -> {y} ({objs[y]})", None, i))
            for state, args, kind, note, removed, param in cands:
                key = (state, tr.name, args)
                if key in seen:
                    continue
                seen.add(key)
                expected = self.env.applicable(state, tr.name, args, objs)
                out.append(Probe(state, tr.name, args, tr.objects_key, expected, kind, note, removed, param))
        return out


PROBE_PROMPT = """You are testing a PDDL domain for a household text environment against the real environment.
Given a state and an action that the environment executed, propose up to {k} minimally changed
variants that the candidate domain might judge incorrectly (missing preconditions, overly broad
parameter types, incomplete effects, unintended shortcuts).

Return JSON: [{{"action": "name(arg1, arg2)", "remove": ["fact(...)"], "add": ["fact(...)"]}}]

Candidate domain:
```pddl
{domain}
```
State: {state}
Executed action: {action}
"""


class LLMChallenger:
    def __init__(self, client: ChatClient, env: Environment, objects: dict[str, dict[str, str]], k: int = 4):
        self.client = client
        self.env = env
        self.objects = objects
        self.k = k
        self.rejected = 0

    def probes(self, transitions: list[DevTransition], domain_text: str, limit: int = 20) -> list[Probe]:
        out = []
        for tr in transitions[:limit]:
            prompt = PROBE_PROMPT.format(k=self.k, domain=domain_text,
                                         state=", ".join(sorted(map(str, tr.state))),
                                         action=f"{tr.name}({', '.join(tr.args)})")
            try:
                items = extract_json(self.client.chat([{"role": "user", "content": prompt}], 0.7, 800))
            except (ValueError, json.JSONDecodeError):
                self.rejected += 1
                continue
            for it in items if isinstance(items, list) else []:
                try:
                    act = parse_atom_text(it["action"])
                    state = (tr.state - {parse_atom_text(x) for x in it.get("remove", [])}) | \
                            {parse_atom_text(x) for x in it.get("add", [])}
                except (KeyError, ValueError, TypeError):
                    self.rejected += 1
                    continue
                objs = self.objects[tr.objects_key]
                if any(a not in objs and a != "agent" for a in act.args):
                    self.rejected += 1
                    continue
                expected = self.env.applicable(frozenset(state), act.pred, act.args, objs)
                out.append(Probe(frozenset(state), act.pred, act.args, tr.objects_key, expected, "llm"))
        return out
