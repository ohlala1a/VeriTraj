"""Adjudication protocol and reference PDDL execution."""

from __future__ import annotations

from typing import Protocol

from ..pddl.ground import applicable, ground_action, progress
from ..pddl.model import Atom, Domain


class Environment(Protocol):
    def applicable(self, state: frozenset[Atom], name: str, args: tuple[str, ...],
                   objects: dict[str, str]) -> bool: ...

    def step(self, state: frozenset[Atom], name: str, args: tuple[str, ...],
             objects: dict[str, str]) -> frozenset[Atom] | None: ...


class ReferenceEnvironment:
    def __init__(self, domain: Domain):
        self.domain = domain
        self.calls = 0

    def _ground(self, name: str, args: tuple[str, ...], objects: dict[str, str]):
        ga, issues = ground_action(self.domain, name, args, {**self.domain.constants, **objects})
        return None if ga is None or issues else ga

    def applicable(self, state, name, args, objects) -> bool:
        self.calls += 1
        ga = self._ground(name, args, objects)
        return ga is not None and applicable(ga, set(state))

    def step(self, state, name, args, objects):
        self.calls += 1
        ga = self._ground(name, args, objects)
        if ga is None or not applicable(ga, set(state)):
            return None
        return frozenset(progress(ga, set(state)))
