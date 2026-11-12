"""Verify collected trajectory shards and export curation datasets."""

import argparse
import re
from pathlib import Path

from veritraj.config import build_components
from veritraj.curation import STRATEGIES, curate
from veritraj.records import read_trajectories, write_jsonl
from veritraj.runtime import write_json


def prepare_data(input_path, config_path, output_path):
    source = Path(input_path)
    paths = ([source] if source.is_file() else sorted(path for path in source.iterdir()
             if re.fullmatch(r"task-\d+\.jsonl\.gz", path.name)))
    if not paths:
        raise ValueError("No trajectory input files")
    trajectories = [trajectory for path in paths for trajectory in read_trajectories(path)]
    if len({trajectory.traj_id for trajectory in trajectories}) != len(trajectories):
        raise ValueError("Trajectory IDs must be unique across input shards")
    components = build_components(config_path)
    verified = {trajectory.traj_id: components.worker().process(trajectory, components.goal(trajectory))
                for trajectory in trajectories}
    output = Path(output_path)
    write_jsonl(output / "trajectories.jsonl.gz", (trajectory.to_dict() for trajectory in trajectories))
    write_jsonl(output / "verified.jsonl.gz", (row.to_dict() for rows in verified.values() for row in rows))
    summaries = []
    for strategy in STRATEGIES:
        result = curate(trajectories, verified, strategy,
                        omega=float(components.cfg.get("curation", {}).get("omega", .5)),
                        window=int(components.cfg.get("curation", {}).get("history_window", 6)))
        write_jsonl(output / f"{strategy}.jsonl", result.examples)
        summaries.append(result.summary())
    write_json(output / "summary.json", {"sources": [str(path) for path in paths], "curation": summaries})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prepare_data(args.input, args.config, args.out)


if __name__ == "__main__":
    main()
