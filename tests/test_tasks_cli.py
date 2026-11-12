"""Task generator and CLI smoke tests."""

import pytest

from veritraj.cli import main
from veritraj.pddl import check_problem, parse_problem
from veritraj.planner import gbfs_plan
from veritraj.tasks import FAMILIES, generate_task, problem_to_pddl
from conftest import EXAMPLES, ROOT


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_tasks_roundtrip(comps, family):
    for seed in range(3):
        spec = generate_task(family, seed)
        prob = parse_problem(problem_to_pddl(spec))
        assert check_problem(comps.domain, prob) == []
        if gbfs_plan(comps.domain, prob, max_expansions=5000):
            return
    pytest.fail(f"no solvable {family} task in 3 seeds")


def test_cli_commands(tmp_path, capsys):
    assert main(["check-domain", str(ROOT / "domains/alfworld/domain.pddl")]) == 0
    assert main(["check-trajectories", str(EXAMPLES)]) == 0
    out = tmp_path / "verified.jsonl"
    assert main(["verify", str(EXAMPLES), "--out", str(out)]) == 0
    assert main(["curate", str(EXAMPLES), "--verified", str(out), "--out", str(tmp_path / "sft")]) == 0
    assert main(["advantages", str(EXAMPLES), "--verified", str(out)]) == 0
    assert (tmp_path / "sft" / "sft_record.jsonl").exists()
    assert "record_weighted" in capsys.readouterr().out
