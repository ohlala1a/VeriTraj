"""Planners used by task generation and the acquisition tests.

`gbfs_plan` is a small greedy best-first search with h_FF on closed-world
states; it is enough for the household tasks shipped here. `fast_downward_plan`
calls an external Fast Downward checkout (set FAST_DOWNWARD to the path of
fast-downward.py) and is what we use for larger domains.
"""

from __future__ import annotations

import heapq
import itertools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .pddl.ground import applicable, goal_satisfied, ground_all, progress, static_predicates
from .pddl.model import Domain, GroundAction, Problem
from .progress import INF, RelaxedTask


def ground_problem(domain: Domain, problem: Problem) -> list[GroundAction]:
    statics = static_predicates(domain)
    return ground_all(domain, problem.objects, problem.init, statics)


def gbfs_plan(domain: Domain, problem: Problem, max_expansions: int = 20000,
              actions: list[GroundAction] | None = None) -> list[GroundAction] | None:
    actions = actions if actions is not None else ground_problem(domain, problem)
    actions = [a for a in actions if a.add or a.delete]
    rt = RelaxedTask(actions, problem.goal_atoms())
    start = frozenset(problem.init)
    h0 = rt.evaluate(set(start))
    if h0 == INF:
        return None
    counter = itertools.count()
    frontier = [(h0, next(counter), start)]
    parent: dict[frozenset, tuple[frozenset, GroundAction] | None] = {start: None}
    expansions = 0
    while frontier and expansions < max_expansions:
        _, _, state = heapq.heappop(frontier)
        s = set(state)
        if goal_satisfied(problem.goal, s):
            plan = []
            cur = state
            while parent[cur] is not None:
                prev, act = parent[cur]  # type: ignore[misc]
                plan.append(act)
                cur = prev
            return plan[::-1]
        expansions += 1
        for a in actions:
            if not applicable(a, s):
                continue
            nxt = frozenset(progress(a, s))
            if nxt in parent:
                continue
            parent[nxt] = (state, a)
            h = rt.evaluate(set(nxt))
            if h < INF:
                heapq.heappush(frontier, (h, next(counter), nxt))
    return None


def fast_downward_plan(domain_text: str, problem_text: str, alias: str = "lama-first",
                       timeout: int = 120) -> list[tuple[str, tuple[str, ...]]] | None:
    fd = os.environ.get("FAST_DOWNWARD") or shutil.which("fast-downward.py") or shutil.which("fast-downward")
    if not fd:
        raise FileNotFoundError("Fast Downward not found; set FAST_DOWNWARD=/path/to/fast-downward.py")
    with tempfile.TemporaryDirectory(prefix="vt_fd_") as tmp:
        d, p = Path(tmp, "domain.pddl"), Path(tmp, "problem.pddl")
        d.write_text(domain_text)
        p.write_text(problem_text)
        cmd = (["python3", fd] if fd.endswith(".py") else [fd]) + \
              ["--alias", alias, "--plan-file", str(Path(tmp, "sas_plan")), str(d), str(p)]
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        plan_file = Path(tmp, "sas_plan")
        if proc.returncode != 0 or not plan_file.exists():
            return None
        steps = []
        for line in plan_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.strip("()").split()
            steps.append((parts[0], tuple(parts[1:])))
        return steps
