"""Check verifier labels under sentence deletion in household observations."""

import argparse

from veritraj.config import build_components
from veritraj.evaluation.observations import evaluate_observation_changes
from veritraj.records import read_trajectories
from veritraj.runtime import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories")
    parser.add_argument("--config", default="configs/alfworld.yaml")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--policies", nargs="+", default=["conservative", "rule_norm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    trajectories = list(read_trajectories(args.trajectories))
    reports = evaluate_observation_changes(trajectories,
              lambda policy: build_components(args.config, policy=policy), args.rates, args.policies, args.seed)
    write_json(args.out, {"source": args.trajectories, "seed": args.seed, "results": reports})


if __name__ == "__main__":
    main()
