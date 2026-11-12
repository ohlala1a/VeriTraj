"""Run named task/domain evaluations from a transfer manifest."""

import argparse
from pathlib import Path

from veritraj.evaluation.online import run_evaluation
from veritraj.policy.generate import load_generator
from veritraj.runtime import load_yaml, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = load_yaml(args.config)
    evaluations = load_yaml(args.manifest)["evaluations"]
    names = [item["name"] for item in evaluations]
    if not names or len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("Transfer evaluation names must be unique directory names")
    configurations = []
    for entry in evaluations:
        name = entry["name"]
        overrides = {key: value for key, value in entry.items() if key != "name"}
        if "model" in overrides and overrides["model"] != base["model"]:
            raise ValueError("Transfer evaluations must share the configured policy checkpoint")
        for key in ("generation", "backend", "vllm", "dtype", "device", "lora", "gradient_checkpointing"):
            if key in overrides and overrides[key] != base.get(key):
                raise ValueError(f"Transfer entries must share policy setting {key!r}")
        config = {**base, **overrides, "output_dir": str(Path(args.out) / name)}
        configurations.append(config)
    policy = load_generator(base)
    reports = [run_evaluation(config, policy=policy) for config in configurations]
    write_json(Path(args.out) / "transfer.json", reports)


if __name__ == "__main__":
    main()
