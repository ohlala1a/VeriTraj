"""Corpus-level operations behind the CLI."""

from __future__ import annotations

import math
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field

from .acquisition import Adjudicator, DeterministicChallenger, DevTransition, ReferenceEnvironment
from .config import Components, build_components
from .latent import replay
from .metrics import label_quality, invalid_proposal_rate, violation_breakdown
from .pddl.ground import ground_all, static_predicates
from .progress import ProgressAnnotator
from .records import LoggedRecord, Trajectory, VerifiedRecord
from .store import Delta, FactStore
from .tasks import FAMILIES, generate_task, scaled_problem
from .planner import gbfs_plan
from .verifier import Verifier


_WORKER_COMPONENTS: Components | None = None


def _init_pool(cfg_path, policy, heuristic, progress):
    global _WORKER_COMPONENTS
    _WORKER_COMPONENTS = build_components(cfg_path, policy=policy, heuristic=heuristic, progress=progress)


def _verify_one(traj_dict: dict) -> list[dict]:
    comps = _WORKER_COMPONENTS
    traj = Trajectory.from_dict(traj_dict)
    return [r.to_dict() for r in comps.worker().process(traj, comps.goal(traj))]  # type: ignore[union-attr]


@dataclass
class VerifyStats:
    trajectories: int = 0
    records: int = 0
    flagged: int = 0
    parse_failures: int = 0
    seconds: float = 0.0
    workers: int = 1
    cache_hit_rate: float | None = None

    def as_dict(self) -> dict:
        return {"trajectories": self.trajectories, "records": self.records,
                "flagged_pct": round(100 * self.flagged / max(self.records, 1), 2),
                "parse_failure_pct": round(100 * self.parse_failures / max(self.records, 1), 2),
                "ms_per_record": round(1000 * self.seconds / max(self.records, 1), 3),
                "workers": self.workers, "heuristic_cache_hit_rate": self.cache_hit_rate}


def verify_corpus(trajs: list[Trajectory], cfg_path=None, *, policy=None, heuristic=None, progress=True,
                  workers: int = 1, recompute: bool = False) -> tuple[dict[str, list[VerifiedRecord]], VerifyStats]:
    stats = VerifyStats(trajectories=len(trajs), workers=workers)
    out: dict[str, list[VerifiedRecord]] = {}
    t0 = time.perf_counter()
    if workers > 1 and not recompute:
        with mp.get_context("spawn").Pool(workers, _init_pool, (cfg_path, policy, heuristic, progress)) as pool:
            for traj, rows in zip(trajs, pool.map(_verify_one, [t.to_dict() for t in trajs], chunksize=4)):
                out[traj.traj_id] = [VerifiedRecord.from_dict(r) for r in rows]
    else:
        comps = build_components(cfg_path, policy=policy, heuristic=heuristic, progress=progress and not recompute)
        hits = misses = 0
        for traj in trajs:
            if recompute:
                out[traj.traj_id] = comps.recompute_worker().process(traj)
            else:
                w = comps.worker()
                out[traj.traj_id] = w.process(traj, comps.goal(traj))
                if w.annotator is not None:
                    hits += w.annotator.stats.hits
                    misses += w.annotator.stats.misses
        if hits + misses:
            stats.cache_hit_rate = round(100 * hits / (hits + misses), 1)
    stats.seconds = time.perf_counter() - t0
    for rows in out.values():
        stats.records += len(rows)
        stats.flagged += sum(1 for r in rows if r.v == 0)
        stats.parse_failures += sum(1 for r in rows if r.p)
    return out, stats


def evaluate_labels(trajs: list[Trajectory], verified: dict[str, list[VerifiedRecord]],
                    comps: Components, recomputed: dict[str, list[VerifiedRecord]] | None = None) -> dict:
    all_labels: list[VerifiedRecord] = []
    all_app: list[bool | None] = []
    issues: list[str] = []
    for traj in trajs:
        rep = replay(comps.domain, comps.problem(traj), comps.interface, traj)
        issues += rep.issues
        all_labels += verified[traj.traj_id]
        all_app += rep.applicable
    conf = label_quality(all_labels, all_app)
    out = {"label_quality": conf.as_dict(), "ipr_pct": round(100 * invalid_proposal_rate(all_app), 2),
           "violations": violation_breakdown(all_labels), "replay_issues": issues}
    if recomputed is not None:
        diff = sum(1 for tid, rows in verified.items()
                   for a, b in zip(rows, recomputed[tid]) if (a.v, a.p, a.witness) != (b.v, b.p, b.witness))
        out["incremental_vs_recompute_disagreements"] = diff
    return out


@dataclass
class AcquisitionSetup:
    env: ReferenceEnvironment
    objects: dict[str, dict[str, str]]
    transitions: list[DevTransition]
    dev_problems: list = field(default_factory=list)
    acceptance_problems: list = field(default_factory=list)
    interface_ops: dict[str, int] = field(default_factory=dict)


def acquisition_setup(comps: Components, trajs: list[Trajectory], acceptance_seed: int = 7000,
                      acceptance_problems: list | None = None) -> AcquisitionSetup:
    if acceptance_problems is None and comps.domain.name != "veritraj-alfworld":
        raise ValueError("This domain requires explicit acceptance_problems")
    env = ReferenceEnvironment(comps.domain)
    objects: dict[str, dict[str, str]] = {}
    transitions: list[DevTransition] = []
    problems = {}
    for traj in trajs:
        prob = comps.problem(traj)
        problems[traj.task_id] = prob
        objects[traj.task_id] = dict(prob.objects)
        for tr in replay(comps.domain, prob, comps.interface, traj).transitions:
            transitions.append(DevTransition(tr.state, tr.action.name, tr.action.args, tr.next_state, traj.task_id))
    acceptance = list(acceptance_problems or [])
    if acceptance_problems is None:
        for i, fam in enumerate(FAMILIES):
            seed = acceptance_seed + 10 * i
            for attempt in range(100):
                spec = generate_task(fam, seed + attempt)
                if gbfs_plan(comps.domain, spec.problem, max_expansions=5000) is not None:
                    acceptance.append(spec.problem)
                    break
            else:
                raise ValueError(f"No solvable acceptance task for {fam} in 100 attempts")
    if not acceptance:
        raise ValueError("Acceptance problems must not be empty")
    dev_names = {problem.name for problem in problems.values()}
    for problem in acceptance:
        if problem.domain_name != comps.domain.name or problem.name in dev_names:
            raise ValueError("Acceptance problems require the same domain and distinct development-task names")
        objects[problem.name] = dict(problem.objects)
    ops = {name: a.arity for name, a in comps.domain.actions.items()}
    return AcquisitionSetup(env, objects, transitions, list(problems.values()), acceptance, ops)


def make_adjudicators(setup: AcquisitionSetup) -> tuple[Adjudicator, Adjudicator, DeterministicChallenger]:
    dev = Adjudicator(setup.env, setup.interface_ops, setup.objects, setup.transitions, setup.dev_problems)
    acc = Adjudicator(setup.env, setup.interface_ops, setup.objects, setup.transitions, setup.acceptance_problems)
    return dev, acc, DeterministicChallenger(setup.env, setup.objects)


def tile(traj: Trajectory, length: int) -> Trajectory:
    """Timing harness: repeat the records of `traj` up to `length` records."""
    recs = []
    i = 0
    while len(recs) < length:
        r = traj.records[i % len(traj.records)]
        recs.append(LoggedRecord(len(recs), r.observation if recs else traj.records[0].observation,
                                 r.generation, r.executed))
        i += 1
    return Trajectory(f"{traj.traj_id}@{length}", traj.task_id, recs, traj.final_observation,
                      problem=traj.problem)


def bench_length(traj: Trajectory, lengths: list[int], cfg_path=None, recompute_max: int = 2000) -> list[dict]:
    comps = build_components(cfg_path, progress=False)
    rows = []
    for n in lengths:
        tt = tile(traj, n)
        t0 = time.perf_counter()
        comps.worker().process(tt)
        inc = 1000 * (time.perf_counter() - t0) / n
        rec = None
        if n <= recompute_max:
            t0 = time.perf_counter()
            comps.recompute_worker().process(tt)
            rec = 1000 * (time.perf_counter() - t0) / n
        rows.append({"length": n, "incremental_ms": round(inc, 3),
                     "recompute_ms": round(rec, 3) if rec is not None else None,
                     "speedup": round(rec / inc, 1) if rec else None})
    return rows


def bench_state(sizes: list[int], cfg_path=None, probes: int = 2000, seed: int = 0) -> list[dict]:
    comps = build_components(cfg_path, progress=False)
    verifier = Verifier(comps.domain)
    rows = []
    for n in sizes:
        prob = scaled_problem(n, items_per_receptacle=1, seed=seed)
        store = FactStore(comps.store_config)
        store.apply(Delta(add=sorted(prob.init, key=str), types=dict(prob.objects)))
        rng = random.Random(seed)
        acts = ground_all(comps.domain, prob.objects, prob.init, static_predicates(comps.domain))
        sample = [rng.choice(acts) for _ in range(probes)]
        t0 = time.perf_counter()
        for a in sample:
            verifier.check_action(store, a)
        ver_ms = 1000 * (time.perf_counter() - t0) / probes
        ann = ProgressAnnotator(comps.domain, prob.goal, "hff", mutex=comps.store_config.mutex)
        t0 = time.perf_counter()
        h_cold = ann.h(store)
        cold_ms = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(100):
            ann.h(store)
        warm_ms = 1000 * (time.perf_counter() - t0) / 100
        rows.append({"receptacles": n, "objects": len(prob.objects), "facts_in_store": len(store),
                     "ground_actions": len(acts), "verify_ms": round(ver_ms, 4),
                     "hff_first_call_ms": round(cold_ms, 2), "hff_cached_ms": round(warm_ms, 4),
                     "h": None if math.isinf(h_cold) else h_cold})
    return rows
