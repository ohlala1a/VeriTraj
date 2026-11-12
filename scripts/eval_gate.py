"""Paired gate-on and gate-off evaluation with common task seeds."""

import argparse
from pathlib import Path

from veritraj.evaluation.online import run_evaluation
from veritraj.policy.generate import load_generator
from veritraj.runtime import load_yaml, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    policy = load_generator(config)
    reports = []
    for seed in args.seeds:
        for gate in (True, False):
            path = Path(args.out) / f"seed_{seed}" / ("gate_on" if gate else "gate_off")
            reports.append(run_evaluation({**config, "seed": seed, "gate": gate, "output_dir": str(path)},
                                           policy=policy))
    write_json(Path(args.out) / "paired.json", reports)


if __name__ == "__main__":
    main()
