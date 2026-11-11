"""Core PDDL data model (STRIPS + typing + negative preconditions).

Atoms are immutable and hashable, so they can live directly in fact stores
and caches. Action schemas keep variables as strings starting with '?'.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Atom:
    pred: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.pred}({', '.join(self.args)})" if self.args else self.pred

    def to_pddl(self) -> str:
        return f"({' '.join((self.pred, *self.args))})"

    def substitute(self, binding: dict[str, str]) -> "Atom":
        return Atom(self.pred, tuple(binding.get(a, a) for a in self.args))


@dataclass(frozen=True, slots=True)
class Literal:
    atom: Atom
    positive: bool = True

    def __str__(self) -> str:
        return str(self.atom) if self.positive else f"not {self.atom}"

    def to_pddl(self) -> str:
        return self.atom.to_pddl() if self.positive else f"(not {self.atom.to_pddl()})"

    def substitute(self, binding: dict[str, str]) -> "Literal":
        return Literal(self.atom.substitute(binding), self.positive)


@dataclass(slots=True)
class ActionSchema:
    name: str
    params: list[tuple[str, str]]
    pre: list[Literal]
    add: list[Atom]
    delete: list[Atom]

    @property
    def arity(self) -> int:
        return len(self.params)


@dataclass(frozen=True, slots=True)
class GroundAction:
    name: str
    args: tuple[str, ...]
    pre: tuple[Literal, ...]
    add: tuple[Atom, ...]
    delete: tuple[Atom, ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


NOOP = GroundAction("noop", (), (), (), ())


@dataclass(slots=True)
class Domain:
    name: str
    requirements: list[str] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)
    constants: dict[str, str] = field(default_factory=dict)
    predicates: dict[str, list[str]] = field(default_factory=dict)
    actions: dict[str, ActionSchema] = field(default_factory=dict)

    def supertypes(self, t: str) -> list[str]:
        chain, seen = [t], {t}
        while t in self.types and self.types[t] not in seen:
            t = self.types[t]
            chain.append(t)
            seen.add(t)
        if "object" not in seen:
            chain.append("object")
        return chain

    def is_subtype(self, t: str, sup: str) -> bool:
        return sup in self.supertypes(t)

    def all_types(self) -> set[str]:
        return set(self.types) | set(self.types.values()) | {"object"}


@dataclass(slots=True)
class Problem:
    name: str
    domain_name: str
    objects: dict[str, str] = field(default_factory=dict)
    init: set[Atom] = field(default_factory=set)
    goal: list[Literal] = field(default_factory=list)
    comment: str = ""

    def goal_atoms(self) -> list[Atom]:
        return [lit.atom for lit in self.goal if lit.positive]
