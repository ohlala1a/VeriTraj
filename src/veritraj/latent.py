"""Replay a logged trajectory against its PDDL problem (closed-world latent state)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .interface import ActionInterface
from .pddl.ground import applicable, goal_satisfied, ground_action, progress
from .pddl.model import Atom, Domain, GroundAction, Problem
from .pipeline import parse_executed
from .records import Trajectory


class LatentView:
    """Read-only store facade so the interface can resolve implicit arguments."""

    def __init__(self, state: set[Atom]):
        self.state = state

    def lookup(self, pred: str, *key: str):
        for a in self.state:
            if a.pred == pred and a.args[: len(key)] == key:
                return a
        return None


@dataclass
class Transition:
    state: frozenset[Atom]
    action: GroundAction
    next_state: frozenset[Atom]
    task_id: str = ""


@dataclass
class ReplayResult:
    applicable: list[bool | None] = field(default_factory=list)
    states: list[frozenset[Atom]] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    goal_reached: bool = False


def proposal_applicable(domain: Domain, problem: Problem, interface: ActionInterface,
                        state: set[Atom], generation: str) -> bool | None:
    parsed = interface.parse(generation, LatentView(state))  # type: ignore[arg-type]
    if not parsed.ok:
        return None
    ga, issues = ground_action(domain, parsed.name, parsed.args, {**domain.constants, **problem.objects})
    return ga is not None and not issues and applicable(ga, state)


def replay(domain: Domain, problem: Problem, interface: ActionInterface, traj: Trajectory) -> ReplayResult:
    res = ReplayResult()
    state = set(problem.init)
    for rec in traj.records:
        res.states.append(frozenset(state))
        app = proposal_applicable(domain, problem, interface, state, rec.generation)
        res.applicable.append(app)
        if rec.latent_applicable is not None and app is not None and rec.latent_applicable != app:
            res.issues.append(f"{traj.traj_id} t={rec.t}: latent_applicable={rec.latent_applicable}, replay says {app}")
        try:
            ex = parse_executed(domain, rec.executed)
        except ValueError as e:
            res.issues.append(f"{traj.traj_id} t={rec.t}: {e}")
            ex = None
        if ex is not None:
            if not applicable(ex, state):
                res.issues.append(f"{traj.traj_id} t={rec.t}: executed {ex} is not applicable")
                continue
            if not traj.gate and app is False:
                res.issues.append(f"{traj.traj_id} t={rec.t}: proposal inapplicable but {ex} executed")
            nxt = progress(ex, state)
            res.transitions.append(Transition(frozenset(state), ex, frozenset(nxt), traj.task_id))
            state = nxt
        elif not traj.gate and app:
            res.issues.append(f"{traj.traj_id} t={rec.t}: applicable proposal was not executed (gate off)")
    res.states.append(frozenset(state))
    res.goal_reached = goal_satisfied(problem.goal, state)
    if res.goal_reached != traj.success:
        res.issues.append(f"{traj.traj_id}: success={traj.success} but goal reached={res.goal_reached}")
    return res
