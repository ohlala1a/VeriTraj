"""Curation strategies and span-level advantages."""

import math

import pytest

from veritraj.advantage import (AdvantageConfig, clipped_objective, group_advantages, group_normalize,
                                record_advantage, token_advantages)
from veritraj.curation import curate


@pytest.fixture(scope="module")
def verified(comps, trajs):
    return {tid: comps.worker().process(t, comps.goal(t)) for tid, t in trajs.items()}


@pytest.mark.parametrize("strategy, kept, targets", [
    ("none", 6, 55), ("outcome", 5, 45), ("trajectory", 1, 7), ("outcome_trajectory", 1, 7),
    ("record", 6, 45), ("record_weighted", 6, 45),
])
def test_curation_counts(trajs, verified, strategy, kept, targets):
    res = curate(list(trajs.values()), verified, strategy)
    assert (res.traj_kept, res.records_target) == (kept, targets)


def test_record_level_never_targets_flagged_spans(trajs, verified):
    res = curate(list(trajs.values()), verified, "record")
    flagged = {f"{tid}#{r.t}" for tid, rows in verified.items() for r in rows if r.v == 0}
    assert not flagged & {e["id"] for e in res.examples}


def test_outcome_weight(trajs, verified):
    res = curate(list(trajs.values()), verified, "record_weighted", omega=0.5)
    w = {e["id"].split("#")[0]: e["weight"] for e in res.examples}
    assert w["heat-mug-01/c"] == 0.5 and w["heat-mug-01/a"] == 1.0


def test_group_normalize():
    a = group_normalize([1.0, 0.0, 0.5, 0.5])
    assert abs(sum(a)) < 1e-9


def test_record_advantage():
    cfg = AdvantageConfig(lambda_hard=2.0, lambda_pot=0.2)
    assert record_advantage(0, 0.7, 1.0, cfg) == -2.0
    assert math.isclose(record_advantage(1, 0.7, 1.0, cfg), 0.9)


def test_violation_penalty_does_not_touch_other_records(trajs, verified):
    group = [trajs[t] for t in ("heat-mug-01/a", "heat-mug-01/b", "heat-mug-01/c")]
    adv = group_advantages(group, verified, AdvantageConfig())
    b = adv["heat-mug-01/b"]
    assert b[6] == -2.0
    assert all(x > 0 for i, x in enumerate(b) if i != 6)


def test_token_broadcast_and_objective():
    adv, mask = token_advantages(10, [(1, 3), (6, 8)], [0.5, -2.0])
    assert mask == [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    lp = [-1.0] * 10
    val = clipped_objective(lp, lp, lp, adv, mask, beta=0.0)
    assert math.isclose(val, (0.5 * 2 - 2.0 * 2) / 4)
