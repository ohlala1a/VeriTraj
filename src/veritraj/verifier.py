"""Verify action preconditions and types against known facts, returning unsupported witnesses."""

from __future__ import annotations

from dataclasses import dataclass

from .interface import ParsedAction
from .pddl.ground import ground_action
from .pddl.model import Domain, GroundAction
from .store import FactStore


@dataclass(frozen=True, slots=True)
class Verdict:
    valid: bool
    witness: tuple[str, ...]
    parse_failed: bool
    action: GroundAction | None

    @property
    def label(self) -> int:
        return int(self.valid and not self.parse_failed)


class Verifier:
    def __init__(self, domain: Domain, check_types: bool = True):
        self.domain = domain
        self.check_types = check_types

    def check(self, store: FactStore, parsed: ParsedAction) -> Verdict:
        if not parsed.ok:
            return Verdict(False, (), True, None)
        ga, issues = ground_action(self.domain, parsed.name, parsed.args, store.types)
        if ga is None:
            return Verdict(False, (), True, None)
        missing: list[str] = []
        if self.check_types:
            missing.extend(str(i.as_atom()) for i in issues)
        for lit in ga.pre:
            ok = store.holds(lit.atom) if lit.positive else store.known_false(lit.atom)
            if not ok:
                missing.append(str(lit))
        return Verdict(not missing, tuple(missing), False, ga)

    def check_action(self, store: FactStore, action: GroundAction) -> Verdict:
        missing = [str(l) for l in action.pre
                   if not (store.holds(l.atom) if l.positive else store.known_false(l.atom))]
        return Verdict(not missing, tuple(missing), False, action)
