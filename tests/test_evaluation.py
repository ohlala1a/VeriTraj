"""Corpus truth coverage, aggregation and entry-point checks."""

import copy
import json

import pytest

from veritraj.evaluation.aggregation import aggregate_runs, latex_table
from veritraj.evaluation.benchmark import benchmark_prefixes
from veritraj.evaluation.corpus import compare_workers, evaluate_corpus
from veritraj.evaluation.observations import drop_sentences
from veritraj.records import LoggedRecord, Trajectory, VerifiedRecord
from veritraj.runtime import init_run


def label(t=0, parse=0):
    return VerifiedRecord("one", t, (0, 1), (0, 4), "look()", 1, p=parse)


def test_coverage_separates_missing_truth_and_parse_failures():
    trajectory = Trajectory("one", "task", [LoggedRecord(0, "", "look()", latent_applicable=False),
                            LoggedRecord(1, "", "look()"), LoggedRecord(2, "", "invalid", latent_applicable=False)])
    metrics = evaluate_corpus([trajectory], {"one": [label(0), label(1), label(2, parse=1)]})
    assert metrics["parsed_proposals"] == 2
    assert metrics["applicability_labeled"] == 1
    assert metrics["ipr_pct"] == 100
    assert metrics["truth_coverage_pct"] == 50
    trajectory.records[0].latent_applicable = None
    assert evaluate_corpus([trajectory], {"one": [label(0), label(1), label(2, 1)]})["ipr_pct"] is None
    trajectory.success_observed = False
    assert evaluate_corpus([trajectory], {"one": [label(0), label(1), label(2, 1)]})["sr_pct"] is None


def test_incremental_equivalence_and_raw_prefix_measurement(comps, trajs):
    trajectories = list(trajs.values())
    assert compare_workers(trajectories, comps)["disagreements"] == 0
    measured = benchmark_prefixes(trajectories, comps, [1, 10000], repeats=1)
    assert len(measured) == 1 and measured[0]["length"] == 1
    assert measured[0]["incremental_ms_per_record"] >= 0


def test_seed_table_and_provenance(tmp_path):
    paths = []
    for seed, score in enumerate((80.0, 90.0)):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps({"environment": "alfworld", "model": "checkpoint", "method": "veritraj",
                        "gate": False, "seed": seed, "split": "test", "protocol": "fixed",
                        "task_ids": ["one"], "metrics": {"sr_pct": score, "ipr_pct": 10, "mean_steps": 20}}))
        paths.append(path)
    rows = aggregate_runs(paths)
    assert rows[0]["sr_pct"]["mean"] == 85
    assert rows[0]["sr_pct"]["std"] == pytest.approx(50 ** .5)
    assert len(rows[0]["runs"]) == 2
    assert r"\pm" in latex_table(rows)
    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate_runs([paths[0], paths[0]])


def test_run_output_cannot_replace_previous_run(tmp_path):
    init_run(tmp_path, {"seed": 3}, 3)
    with pytest.raises(FileExistsError):
        init_run(tmp_path, {}, 3)


def test_observation_change_preserves_logged_proposals(trajs):
    import random

    original = next(iter(trajs.values()))
    changed = drop_sentences(original, 1.0, random.Random(0))
    assert all(not record.observation for record in changed.records)
    assert [r.generation for r in changed.records] == [r.generation for r in original.records]
    assert original.records[0].observation


def test_transfer_rejects_policy_overrides_before_loading(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    import yaml

    path = Path(__file__).resolve().parents[1] / 'scripts/eval_transfer.py'
    spec = importlib.util.spec_from_file_location('transfer_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / 'base.yaml'
    manifest = tmp_path / 'manifest.yaml'
    base.write_text(yaml.safe_dump({'model': 'local-model', 'generation': {'max_new_tokens': 10}}))
    manifest.write_text(yaml.safe_dump({'evaluations': [{'name': 'changed', 'generation': {'max_new_tokens': 20}}]}))
    monkeypatch.setattr(module, 'load_generator', lambda _: pytest.fail('Policy loading should follow validation'))
    monkeypatch.setattr(sys, 'argv', ['eval_transfer', '--config', str(base), '--manifest', str(manifest),
                                    '--out', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='policy setting'):
        module.main()
