"""S-expression reader and PDDL domain/problem parser.

Supported fragment: :strips, :typing, :negative-preconditions, :equality is
rejected. Preconditions and goals are conjunctions of (possibly negated)
atoms; effects are conjunctions of add/delete atoms. Anything else raises
PDDLError with the offending form, which the acquisition loop reports back to
the Builder as a parse failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .model import ActionSchema, Atom, Domain, Literal, Problem

SExpr = Union[str, list["SExpr"]]

SUPPORTED_REQUIREMENTS = {":strips", ":typing", ":negative-preconditions"}


class PDDLError(ValueError):
    pass


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.split(";", 1)[0]
        out.extend(line.replace("(", " ( ").replace(")", " ) ").split())
    return [tok.lower() for tok in out]


def read_sexpr(text: str) -> SExpr:
    tokens = tokenize(text)
    if not tokens:
        raise PDDLError("empty input")
    pos = 0

    def read() -> SExpr:
        nonlocal pos
        if pos >= len(tokens):
            raise PDDLError("unexpected end of input (unbalanced parentheses)")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items: list[SExpr] = []
            while True:
                if pos >= len(tokens):
                    raise PDDLError("unbalanced parentheses")
                if tokens[pos] == ")":
                    pos += 1
                    return items
                items.append(read())
        if tok == ")":
            raise PDDLError(f"unexpected ')' at token {pos}")
        return tok

    expr = read()
    if pos != len(tokens):
        raise PDDLError(f"trailing tokens after top-level form: {tokens[pos:pos + 5]}")
    return expr


def leading_comment(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(";"):
            lines.append(s.lstrip(";").strip())
        elif s:
            break
    return " ".join(lines)


def parse_typed_list(items: list[SExpr], default: str = "object") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pending: list[str] = []
    i = 0
    while i < len(items):
        tok = items[i]
        if not isinstance(tok, str):
            raise PDDLError(f"expected name in typed list, got {tok}")
        if tok == "-":
            if i + 1 >= len(items) or not isinstance(items[i + 1], str):
                raise PDDLError("dangling '-' in typed list")
            typ = items[i + 1]
            out.extend((name, typ) for name in pending)
            pending = []
            i += 2
            continue
        pending.append(tok)
        i += 1
    out.extend((name, default) for name in pending)
    return out


def _atom(expr: SExpr) -> Atom:
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        raise PDDLError(f"malformed atom: {expr}")
    if expr[0] in {"and", "or", "not", "forall", "exists", "when", "imply", "="}:
        raise PDDLError(f"expected atom, got '{expr[0]}' form")
    if not all(isinstance(a, str) for a in expr[1:]):
        raise PDDLError(f"nested term in atom: {expr}")
    return Atom(expr[0], tuple(expr[1:]))  # type: ignore[arg-type]


def _conjunction(expr: SExpr, what: str) -> list[SExpr]:
    if expr == [] or expr == ["and"]:
        return []
    if isinstance(expr, list) and expr and expr[0] == "and":
        return list(expr[1:])
    if isinstance(expr, list) and expr and expr[0] in {"or", "imply", "exists", "forall", "when"}:
        raise PDDLError(f"unsupported '{expr[0]}' in {what}")
    return [expr]


def parse_literals(expr: SExpr, what: str) -> list[Literal]:
    lits = []
    for part in _conjunction(expr, what):
        if isinstance(part, list) and part and part[0] == "not":
            if len(part) != 2:
                raise PDDLError(f"malformed negation in {what}: {part}")
            lits.append(Literal(_atom(part[1]), False))
        else:
            lits.append(Literal(_atom(part), True))
    return lits


def parse_effects(expr: SExpr) -> tuple[list[Atom], list[Atom]]:
    add, delete = [], []
    for lit in parse_literals(expr, "effect"):
        (add if lit.positive else delete).append(lit.atom)
    return add, delete


def _sections(expr: list[SExpr]) -> dict[str, list[SExpr]]:
    out: dict[str, list[SExpr]] = {}
    for part in expr[2:]:
        if not isinstance(part, list) or not part or not isinstance(part[0], str):
            raise PDDLError(f"malformed section: {part}")
        if part[0] == ":action":
            out.setdefault(":action", []).append(part)
        else:
            out[part[0]] = part
    return out


def parse_action(form: list[SExpr]) -> ActionSchema:
    if len(form) < 2 or not isinstance(form[1], str):
        raise PDDLError("action without name")
    name = form[1]
    fields: dict[str, SExpr] = {}
    i = 2
    while i < len(form):
        key = form[i]
        if not isinstance(key, str) or not key.startswith(":") or i + 1 >= len(form):
            raise PDDLError(f"malformed action body for '{name}'")
        fields[key] = form[i + 1]
        i += 2
    unknown = set(fields) - {":parameters", ":precondition", ":effect"}
    if unknown:
        raise PDDLError(f"unsupported action fields in '{name}': {sorted(unknown)}")
    params_raw = fields.get(":parameters", [])
    if not isinstance(params_raw, list):
        raise PDDLError(f"bad parameter list in '{name}'")
    params = parse_typed_list(params_raw)
    pre = parse_literals(fields.get(":precondition", []), f"precondition of {name}")
    add, delete = parse_effects(fields.get(":effect", []))
    return ActionSchema(name, params, pre, add, delete)


def parse_domain(text: str) -> Domain:
    expr = read_sexpr(text)
    if not (isinstance(expr, list) and len(expr) >= 2 and expr[0] == "define"):
        raise PDDLError("domain must start with (define ...)")
    head = expr[1]
    if not (isinstance(head, list) and len(head) == 2 and head[0] == "domain"):
        raise PDDLError("missing (domain <name>)")
    dom = Domain(name=str(head[1]))
    sec = _sections(expr)
    if ":requirements" in sec:
        dom.requirements = [str(r) for r in sec[":requirements"][1:]]
        unsupported = set(dom.requirements) - SUPPORTED_REQUIREMENTS
        if unsupported:
            raise PDDLError(f"unsupported requirements: {sorted(unsupported)}")
    if ":types" in sec:
        for name, parent in parse_typed_list(sec[":types"][1:]):
            if name != "object":
                dom.types[name] = parent
    if ":constants" in sec:
        dom.constants = dict(parse_typed_list(sec[":constants"][1:]))
    if ":predicates" in sec:
        for p in sec[":predicates"][1:]:
            if not isinstance(p, list) or not p or not isinstance(p[0], str):
                raise PDDLError(f"malformed predicate declaration: {p}")
            dom.predicates[p[0]] = [t for _, t in parse_typed_list(p[1:])]
    for form in sec.get(":action", []):
        act = parse_action(form)  # type: ignore[arg-type]
        if act.name in dom.actions:
            raise PDDLError(f"duplicate action '{act.name}'")
        dom.actions[act.name] = act
    return dom


def parse_problem(text: str) -> Problem:
    expr = read_sexpr(text)
    if not (isinstance(expr, list) and len(expr) >= 2 and expr[0] == "define"):
        raise PDDLError("problem must start with (define ...)")
    head = expr[1]
    if not (isinstance(head, list) and len(head) == 2 and head[0] == "problem"):
        raise PDDLError("missing (problem <name>)")
    sec = _sections(expr)
    if ":domain" not in sec:
        raise PDDLError("missing (:domain ...)")
    prob = Problem(name=str(head[1]), domain_name=str(sec[":domain"][1]), comment=leading_comment(text))
    if ":objects" in sec:
        prob.objects = dict(parse_typed_list(sec[":objects"][1:]))
    if ":init" in sec:
        for a in sec[":init"][1:]:
            prob.init.add(_atom(a))
    if ":goal" in sec:
        prob.goal = parse_literals(sec[":goal"][1], "goal")
    return prob


def load_domain(path: str | Path) -> Domain:
    return parse_domain(Path(path).read_text(encoding="utf-8"))


def load_problem(path: str | Path) -> Problem:
    return parse_problem(Path(path).read_text(encoding="utf-8"))


def parse_atom_text(text: str) -> Atom:
    """Parse 'open(microwave_1)' or '(open microwave_1)'."""
    s = text.strip()
    if s.startswith("("):
        return _atom(read_sexpr(s))
    if "(" in s and s.endswith(")"):
        pred, rest = s.split("(", 1)
        args = [a.strip() for a in rest[:-1].split(",") if a.strip()]
        return Atom(pred.strip().lower(), tuple(a.lower() for a in args))
    return Atom(s.lower(), ())
