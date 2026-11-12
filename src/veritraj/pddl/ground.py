"""Grounding of action schemas."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .model import ActionSchema, Atom, Domain, GroundAction, Literal


@dataclass(frozen=True, slots=True)
class TypeIssue:
    param: str
    obj: str
    expected: str
    actual: str | None

    def as_atom(self) -> Atom:
        return Atom(self.expected, (self.obj,))


def bind(schema: ActionSchema, args: tuple[str, ...]) -> dict[str, str]:
    return {var: arg for (var, _), arg in zip(schema.params, args)}


def instantiate(schema: ActionSchema, args: tuple[str, ...]) -> GroundAction:
    b = bind(schema, args)
    return GroundAction(
        schema.name,
        tuple(args),
        tuple(lit.substitute(b) for lit in schema.pre),
        tuple(a.substitute(b) for a in schema.add),
        tuple(a.substitute(b) for a in schema.delete),
    )


def type_issues(domain: Domain, schema: ActionSchema, args: tuple[str, ...],
                object_types: dict[str, str]) -> list[TypeIssue]:
    issues = []
    for (var, expected), obj in zip(schema.params, args):
        actual = object_types.get(obj) or domain.constants.get(obj)
        if actual is None or not domain.is_subtype(actual, expected):
            issues.append(TypeIssue(var, obj, expected, actual))
    return issues


def ground_action(domain: Domain, name: str, args: tuple[str, ...],
                  object_types: dict[str, str]) -> tuple[GroundAction | None, list[TypeIssue]]:
    schema = domain.actions.get(name)
    if schema is None or len(args) != schema.arity:
        return None, []
    return instantiate(schema, args), type_issues(domain, schema, args, object_types)


def objects_by_type(domain: Domain, object_types: dict[str, str]) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    merged = {**domain.constants, **object_types}
    for obj, typ in sorted(merged.items()):
        for sup in domain.supertypes(typ):
            table.setdefault(sup, []).append(obj)
    return table


def ground_all(domain: Domain, object_types: dict[str, str],
               static_true: set[Atom] | None = None,
               static_preds: Iterable[str] = ()) -> list[GroundAction]:
    """Enumerate type-correct ground actions."""
    by_type = objects_by_type(domain, object_types)
    static_preds = set(static_preds)
    out = []
    for schema in domain.actions.values():
        pools = [by_type.get(t, []) for _, t in schema.params]
        for args in product(*pools):
            ga = instantiate(schema, tuple(args))
            if static_true is not None and static_preds:
                if any(l.positive and l.atom.pred in static_preds and l.atom not in static_true
                       for l in ga.pre):
                    continue
            out.append(ga)
    return out


def static_predicates(domain: Domain) -> set[str]:
    changed = set()
    for a in domain.actions.values():
        changed.update(x.pred for x in a.add)
        changed.update(x.pred for x in a.delete)
    return set(domain.predicates) - changed


def applicable(action: GroundAction, true_facts: set[Atom], false_facts: set[Atom] | None = None,
               closed_world: bool = True) -> bool:
    for lit in action.pre:
        if lit.positive:
            if lit.atom not in true_facts:
                return False
        elif closed_world:
            if lit.atom in true_facts:
                return False
        elif false_facts is None or lit.atom not in false_facts:
            return False
    return True


def progress(action: GroundAction, true_facts: set[Atom]) -> set[Atom]:
    out = set(true_facts)
    out.difference_update(action.delete)
    out.update(action.add)
    return out


def goal_satisfied(goal: list[Literal], true_facts: set[Atom]) -> bool:
    return all((l.atom in true_facts) == l.positive for l in goal)
