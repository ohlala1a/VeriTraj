"""Evaluate a checkpoint on the configured held-out tasks."""

import argparse
import json

from veritraj.evaluation.online import run_evaluation
from veritraj.runtime import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", choices=["on", "off"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.gate:
        config["gate"] = args.gate == "on"
    if args.seed is not None:
        config["seed"] = args.seed
    if args.out:
        config["output_dir"] = args.out
    print(json.dumps(run_evaluation(config), indent=2))


if __name__ == "__main__":
    main()
