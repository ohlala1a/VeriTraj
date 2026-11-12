"""Static checks for domains and problems (the 'Parse' stage of acquisition)."""

from __future__ import annotations

from .model import Atom, Domain, Problem


def _check_atom(domain: Domain, atom: Atom, where: str, scope: dict[str, str] | None) -> list[str]:
    issues = []
    if atom.pred not in domain.predicates:
        return [f"{where}: undeclared predicate '{atom.pred}'"]
    sig = domain.predicates[atom.pred]
    if len(sig) != len(atom.args):
        return [f"{where}: '{atom.pred}' expects {len(sig)} args, got {len(atom.args)}"]
    for arg, expected in zip(atom.args, sig):
        if arg.startswith("?"):
            if scope is None or arg not in scope:
                issues.append(f"{where}: unbound variable {arg}")
                continue
            actual = scope[arg]
        else:
            actual = (scope or {}).get(arg) or domain.constants.get(arg)
            if actual is None:
                issues.append(f"{where}: unknown constant '{arg}'")
                continue
        if not (domain.is_subtype(actual, expected) or domain.is_subtype(expected, actual)):
            issues.append(f"{where}: '{arg}' of type {actual} incompatible with {expected} in {atom.pred}")
    return issues


def check_domain(domain: Domain) -> list[str]:
    issues = []
    known = domain.all_types()
    for t, parent in domain.types.items():
        if parent not in known:
            issues.append(f"type '{t}' has undeclared parent '{parent}'")
    for c, t in domain.constants.items():
        if t not in known:
            issues.append(f"constant '{c}' has undeclared type '{t}'")
    for p, sig in domain.predicates.items():
        for t in sig:
            if t not in known:
                issues.append(f"predicate '{p}' uses undeclared type '{t}'")
    for a in domain.actions.values():
        scope = {}
        for var, t in a.params:
            if not var.startswith("?"):
                issues.append(f"action '{a.name}': parameter '{var}' must start with '?'")
            if var in scope:
                issues.append(f"action '{a.name}': duplicate parameter '{var}'")
            if t not in known:
                issues.append(f"action '{a.name}': parameter {var} has undeclared type '{t}'")
            scope[var] = t
        for lit in a.pre:
            issues += _check_atom(domain, lit.atom, f"{a.name}.pre", scope)
            if not lit.positive and ":negative-preconditions" not in domain.requirements:
                issues.append(f"{a.name}.pre: negative precondition without :negative-preconditions")
        for atom in a.add:
            issues += _check_atom(domain, atom, f"{a.name}.add", scope)
        for atom in a.delete:
            issues += _check_atom(domain, atom, f"{a.name}.del", scope)
        if not a.add and not a.delete and a.name not in {"look", "inventory"}:
            issues.append(f"action '{a.name}' has no effects")
    return issues


def check_problem(domain: Domain, problem: Problem) -> list[str]:
    issues = []
    if problem.domain_name != domain.name:
        issues.append(f"problem refers to domain '{problem.domain_name}', loaded '{domain.name}'")
    known = domain.all_types()
    for o, t in problem.objects.items():
        if t not in known:
            issues.append(f"object '{o}' has undeclared type '{t}'")
    scope = {**domain.constants, **problem.objects}
    for atom in sorted(problem.init, key=str):
        issues += _check_atom(domain, atom, "init", scope)
    for lit in problem.goal:
        issues += _check_atom(domain, lit.atom, "goal", scope)
    return issues
