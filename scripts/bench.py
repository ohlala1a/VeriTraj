"""Measure incremental maintenance, recomputation and per-state cost."""

import argparse
import platform

from veritraj.config import build_components
from veritraj.evaluation.benchmark import benchmark_prefixes, benchmark_states
from veritraj.records import read_trajectories
from veritraj.runtime import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["length", "state"])
    parser.add_argument("trajectories")
    parser.add_argument("--config", default="configs/alfworld.yaml")
    parser.add_argument("--lengths", nargs="+", type=int, default=[10, 50, 100, 500])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    trajectories = list(read_trajectories(args.trajectories))
    components = build_components(args.config, progress=args.kind == "state")
    rows = (benchmark_prefixes(trajectories, components, args.lengths, args.repeats) if args.kind == "length"
            else benchmark_states(trajectories, components, args.repeats))
    write_json(args.out, {"source": args.trajectories, "kind": args.kind, "platform": platform.platform(),
                          "python": platform.python_version(), "repeats": args.repeats, "measurements": rows})


if __name__ == "__main__":
    main()
