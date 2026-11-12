"""Relaxed-planning heuristics, shaping and caching."""

import math

from veritraj.pddl import Atom, load_problem
from veritraj.planner import gbfs_plan
from veritraj.progress import ProgressAnnotator, RelaxedTask
from veritraj.planner import ground_problem
from veritraj.store import Delta, FactStore
from conftest import ROOT

HEAT = ROOT / "domains/alfworld/problems/heat_mug_01.pddl"


def full_store(comps, prob):
    s = FactStore(comps.store_config)
    s.apply(Delta(add=sorted(prob.init, key=str), types=dict(prob.objects)))
    return s


def test_heuristics_on_initial_state(comps):
    prob = load_problem(HEAT)
    rt = RelaxedTask(ground_problem(comps.domain, prob), prob.goal_atoms())
    hff, hadd, hmax = (rt.evaluate(set(prob.init), k) for k in ("hff", "hadd", "hmax"))
    plan = gbfs_plan(comps.domain, prob)
    assert 0 < hmax <= hff <= len(plan)
    assert hadd >= hmax
    assert rt.evaluate(set(prob.init), "goalcount") == 2


def test_unknown_goal_object_is_unreachable(comps):
    prob = load_problem(HEAT)
    ann = ProgressAnnotator(comps.domain, prob.goal)
    assert math.isinf(ann.h(FactStore(comps.store_config)))


def test_shaping_rules(comps):
    ann = ProgressAnnotator(comps.domain, load_problem(HEAT).goal, gamma=0.99)
    assert ann.shaping(5, 4, changed=False) == 0.0
    assert ann.shaping(math.inf, 4, changed=True) == 0.0
    assert math.isclose(ann.shaping(5, 4, changed=True), 0.99 * -4 + 5)


def test_cache_by_store_hash(comps):
    prob = load_problem(HEAT)
    ann = ProgressAnnotator(comps.domain, prob.goal)
    s = full_store(comps, prob)
    h1 = ann.h(s)
    h2 = ann.h(s.copy())
    assert h1 == h2 and ann.stats.hits == 1 and ann.stats.misses == 1


def test_mutex_relaxation_only_for_unknown(comps):
    prob = load_problem(HEAT)
    ann = ProgressAnnotator(comps.domain, prob.goal, mutex=comps.store_config.mutex)
    s = FactStore(comps.store_config)
    s.apply(Delta(types={"cabinet_1": "receptacle"}))
    facts = ann._relaxed_facts(s)
    assert {Atom("open", ("cabinet_1",)), Atom("closed", ("cabinet_1",))} <= facts
    s.apply(Delta(add=[Atom("closed", ("cabinet_1",))]))
    facts = ann._relaxed_facts(s)
    assert Atom("closed", ("cabinet_1",)) in facts and Atom("open", ("cabinet_1",)) not in facts
