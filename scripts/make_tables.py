"""Aggregate seed-level evaluation files and export a LaTeX table."""

import argparse
from pathlib import Path

from veritraj.evaluation.aggregation import aggregate_runs, latex_table
from veritraj.runtime import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = []
    for pattern in args.metrics:
        candidate = Path(pattern)
        paths.extend(sorted(candidate.rglob("metrics.json")) if candidate.is_dir() else [candidate])
    rows = aggregate_runs(paths)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex_table(rows), encoding="utf-8")
    write_json(output.with_suffix(".sources.json"), rows)


if __name__ == "__main__":
    main()
