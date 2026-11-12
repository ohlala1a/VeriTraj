"""Evaluate label quality and incremental maintenance on logged corpora."""

import argparse

from veritraj.config import build_components
from veritraj.curation import STRATEGIES, curate
from veritraj.evaluation.corpus import compare_workers, evaluate_corpus
from veritraj.records import read_trajectories
from veritraj.runtime import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="+")
    parser.add_argument("--config", default="configs/alfworld.yaml")
    parser.add_argument("--policies", nargs="+", default=["conservative"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reports = []
    for path in args.trajectories:
        trajectories = list(read_trajectories(path))
        for policy in args.policies:
            components = build_components(args.config, policy=policy)
            verified = {t.traj_id: components.worker().process(t, components.goal(t)) for t in trajectories}
            reports.append({"source": path, "grounding_policy": policy,
                            "metrics": evaluate_corpus(trajectories, verified),
                            "equivalence": compare_workers(trajectories, components),
                            "curation": [curate(trajectories, verified, strategy).summary() for strategy in STRATEGIES]})
    write_json(args.out, reports)


if __name__ == "__main__":
    main()
