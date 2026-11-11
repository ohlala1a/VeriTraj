"""Adjudicator: deterministic, tool-based tests of a candidate domain.

Stages (each reported separately, cf. Parse / Trace / Plan / Execute):
  parse     the domain parses and passes static checks; every interface
            operator exists with the right arity
  trace     every development transition replays: the candidate accepts the
            executed action and predicts exactly the observed successor
  probes    candidate verdicts agree with the environment on challenger probes
  plan      the candidate yields a plan for each development problem
  execute   that plan runs in the environment and reaches the goal

Only reproducible mismatches are returned to the Builder, each with the
evidence needed to reproduce it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..pddl import PDDLError, check_domain, parse_domain
from ..pddl.ground import applicable, goal_satisfied, ground_action, progress
from ..pddl.model import Domain, Problem
from ..planner import gbfs_plan
from .environment import Environment
from .probes import DevTransition, Probe


@dataclass
class Mismatch:
    kind: str
    operator: str
    detail: str
    evidence: dict = field(default_factory=dict)
    key: str = ""
    count: int = 1


@dataclass
class Report:
    parse_ok: bool = False
    static_issues: list[str] = field(default_factory=list)
    trace_total: int = 0
    trace_ok: int = 0
    probe_total: int = 0
    probe_ok: int = 0
    plan_total: int = 0
    plan_ok: int = 0
    exec_ok: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return (self.parse_ok and not self.static_issues and not self.mismatches
                and self.trace_ok == self.trace_total and self.exec_ok == self.plan_total)

    def rates(self) -> dict:
        def pct(a, b):
            return round(100 * a / b, 1) if b else None
        return {"parse": self.parse_ok and not self.static_issues, "trace": pct(self.trace_ok, self.trace_total),
                "probes": pct(self.probe_ok, self.probe_total), "plan": pct(self.plan_ok, self.plan_total),
                "execute": pct(self.exec_ok, self.plan_total), "mismatches": len(self.mismatches)}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accepted"] = self.accepted
        d["rates"] = self.rates()
        return d


class Adjudicator:
    def __init__(self, env: Environment, interface_ops: dict[str, int], objects: dict[str, dict[str, str]],
                 transitions: list[DevTransition], problems: list[Problem], max_feedback: int = 12,
                 plan_expansions: int = 8000):
        self.env = env
        self.interface_ops = interface_ops
        self.objects = objects
        self.transitions = transitions
        self.problems = problems
        self.max_feedback = max_feedback
        self.plan_expansions = plan_expansions

    def evaluate(self, domain_text: str, probes: list[Probe] = ()) -> tuple[Report, Domain | None]:
        rep = Report()
        try:
            dom = parse_domain(domain_text)
        except PDDLError as e:
            rep.mismatches.append(Mismatch("parse_error", "-", str(e)))
            return rep, None
        rep.parse_ok = True
        rep.static_issues = check_domain(dom)
        for op, arity in self.interface_ops.items():
            if op not in dom.actions:
                rep.static_issues.append(f"interface operator '{op}' missing")
            elif dom.actions[op].arity != arity:
                rep.static_issues.append(f"operator '{op}' has arity {dom.actions[op].arity}, interface uses {arity}")
        if rep.static_issues:
            return rep, dom
        mm: list[Mismatch] = []
        self._trace(dom, rep, mm)
        self._probes(dom, probes, rep, mm)
        self._plans(dom, rep, mm)
        rep.mismatches = _dedup(mm)[: self.max_feedback]
        return rep, dom

    # -- stages ------------------------------------------------------------
    def _trace(self, dom: Domain, rep: Report, mm: list[Mismatch]) -> None:
        for tr in self.transitions:
            rep.trace_total += 1
            objs = {**dom.constants, **self.objects[tr.objects_key]}
            ga, issues = ground_action(dom, tr.name, tr.args, objs)
            act = f"{tr.name}({', '.join(tr.args)})"
            if ga is None or issues:
                mm.append(Mismatch("overly_narrow_type", tr.name, f"{act} rejected on types",
                                   {"issues": [str(i.as_atom()) for i in issues]}))
                continue
            if not applicable(ga, set(tr.state)):
                missing = [str(l) for l in ga.pre if (l.atom in tr.state) != l.positive]
                mm.append(Mismatch("overly_strict_precondition", tr.name,
                                   f"{act} was executed by the environment but the candidate rejects it",
                                   {"unsatisfied": missing}))
                continue
            pred = frozenset(progress(ga, set(tr.state)))
            if pred != tr.next_state:
                missing, extra = tr.next_state - pred, pred - tr.next_state
                sig = ",".join(sorted({f"+{a.pred}" for a in missing} | {f"-{a.pred}" for a in extra}))
                mm.append(Mismatch("incomplete_effect", tr.name, f"successor of {act} differs from the environment",
                                   {"missing": sorted(map(str, missing)), "extra": sorted(map(str, extra))},
                                   key=f"incomplete_effect:{tr.name}:{sig}"))
                continue
            rep.trace_ok += 1

    def _probes(self, dom: Domain, probes, rep: Report, mm: list[Mismatch]) -> None:
        for pr in probes:
            rep.probe_total += 1
            objs = {**dom.constants, **self.objects[pr.objects_key]}
            ga, issues = ground_action(dom, pr.name, pr.args, objs)
            got = ga is not None and not issues and applicable(ga, set(pr.state))
            if got == pr.expected:
                rep.probe_ok += 1
                continue
            if got and not pr.expected:
                kind = "overly_broad_type" if pr.kind == "type" else "missing_precondition"
                detail = f"candidate accepts {pr.action_str()} but the environment rejects it ({pr.note})"
            else:
                kind = "overly_strict_precondition"
                detail = f"candidate rejects {pr.action_str()} but the environment accepts it ({pr.note})"
            what = pr.removed.pred if pr.removed is not None else (f"param{pr.param}" if pr.param is not None else pr.kind)
            mm.append(Mismatch(kind, pr.name, detail, {"probe": pr.kind, "note": pr.note},
                               key=f"{kind}:{pr.name}:{what}"))

    def _plans(self, dom: Domain, rep: Report, mm: list[Mismatch]) -> None:
        for prob in self.problems:
            rep.plan_total += 1
            cand = Problem(prob.name, dom.name, prob.objects, prob.init, prob.goal)
            plan = gbfs_plan(dom, cand, max_expansions=self.plan_expansions)
            if plan is None:
                mm.append(Mismatch("unsolvable", "-", f"no plan for {prob.name} under the candidate"))
                continue
            rep.plan_ok += 1
            state = frozenset(prob.init)
            ok = True
            for i, step in enumerate(plan):
                nxt = self.env.step(state, step.name, step.args, prob.objects)
                if nxt is None:
                    mm.append(Mismatch("unintended_shortcut", step.name,
                                       f"step {i} {step} of the candidate plan for {prob.name} fails in the environment",
                                       {"plan": [str(s) for s in plan]}, key=f"unintended_shortcut:{step.name}"))
                    ok = False
                    break
                state = nxt
            if ok and not goal_satisfied(prob.goal, set(state)):
                mm.append(Mismatch("unintended_shortcut", "-", f"candidate plan for {prob.name} does not reach the goal"))
                ok = False
            rep.exec_ok += int(ok)


def _dedup(mm: list[Mismatch]) -> list[Mismatch]:
    """One representative per schematic defect (kind, operator, predicate), with a count."""
    groups: dict[str, Mismatch] = {}
    for m in mm:
        key = m.key or f"{m.kind}:{m.operator}:{m.detail}"
        if key in groups:
            groups[key].count += 1
        else:
            groups[key] = m
    return list(groups.values())
