"""End-to-end labels on the example trajectories, and incremental == recompute."""

import pytest

from veritraj.latent import replay

EXPECTED_FLAGS = {
    "heat-mug-01/a": {},
    "heat-mug-01/b": {6: ["open(microwave_1)"]},
    "heat-mug-01/c": {6: ["open(cabinet_1)"],
                      7: ["item(mug_1)", "at(agent, countertop_1)", "in(mug_1, countertop_1)", "handempty"],
                      8: None, 9: None},
    "examine-book-01/a": {6: ["lit(desklamp_1)"]},
    "clean-fork-01/a": {1: ["item(fork_1)", "in(fork_1, diningtable_1)"],
                        2: ["item(fork_2)", "in(fork_2, diningtable_1)", "handempty"], 3: "parse"},
    "pick-two-cellphone-01/a": {4: ["item(cellphone_2)", "at(agent, coffeetable_1)", "in(cellphone_2, coffeetable_1)"]},
}


@pytest.mark.parametrize("tid", sorted(EXPECTED_FLAGS))
def test_labels_and_witnesses(comps, trajs, tid):
    traj = trajs[tid]
    rows = comps.worker().process(traj, comps.goal(traj))
    flagged = {r.t: r for r in rows if r.v == 0}
    assert set(flagged) == set(EXPECTED_FLAGS[tid])
    for t, wit in EXPECTED_FLAGS[tid].items():
        if wit == "parse":
            assert flagged[t].p == 1
        elif wit is not None:
            assert flagged[t].witness == wit


@pytest.mark.parametrize("tid", sorted(EXPECTED_FLAGS))
def test_incremental_equals_recompute(comps, trajs, tid):
    traj = trajs[tid]
    inc = comps.worker().process(traj)
    rec = comps.recompute_worker().process(traj)
    assert [(r.v, r.p, r.witness, r.span, r.store_facts) for r in inc] == \
           [(r.v, r.p, r.witness, r.span, r.store_facts) for r in rec]


@pytest.mark.parametrize("tid", sorted(EXPECTED_FLAGS))
def test_examples_replay_consistently(comps, trajs, tid):
    traj = trajs[tid]
    assert replay(comps.domain, comps.problem(traj), comps.interface, traj).issues == []


def test_flagged_span_is_the_command(comps, trajs):
    traj = trajs["heat-mug-01/b"]
    r = comps.worker().process(traj)[6]
    g = traj.records[6].generation
    assert g[r.char_span[0]:r.char_span[1]] == "put mug 1 in/on microwave 1"
