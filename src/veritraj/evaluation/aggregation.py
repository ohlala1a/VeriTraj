"""Seed-level summaries and traceable LaTeX tables."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = ("environment", "model", "method", "gate", "split", "protocol")


def aggregate_runs(paths: list[str | Path]) -> list[dict]:
    groups = defaultdict(list)
    for path in paths:
        source = Path(path).resolve()
        row = json.loads(source.read_text(encoding="utf-8"))
        missing = [key for key in (*GROUP_FIELDS, "seed", "task_ids", "metrics") if key not in row]
        if missing:
            raise ValueError(f"{source}: missing {', '.join(missing)}")
        key = tuple(row[field] for field in GROUP_FIELDS)
        groups[key].append((source, row))
    output = []
    for key, runs in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = [row["seed"] for _, row in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seed in comparison group {key}")
        task_sets = {tuple(sorted(row["task_ids"])) for _, row in runs}
        if len(task_sets) != 1:
            raise ValueError(f"Task sets differ within comparison group {key}")
        summary = {**dict(zip(GROUP_FIELDS, key)), "seeds": seeds,
                   "runs": [str(path) for path, _ in runs], "task_ids": list(next(iter(task_sets)))}
        for metric in ("sr_pct", "ipr_pct", "mean_steps"):
            values = [row["metrics"].get(metric) for _, row in runs]
            if any(value is None for value in values):
                summary[metric] = {"mean": None, "std": None, "n": 0}
            else:
                summary[metric] = {"mean": statistics.mean(values),
                                   "std": statistics.stdev(values) if len(values) > 1 else None,
                                   "n": len(values)}
        output.append(summary)
    return output


def latex_table(rows: list[dict]) -> str:
    def escape(value) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    def cell(value: dict) -> str:
        if value["mean"] is None:
            return "--"
        if value["std"] is None:
            return f"{value['mean']:.1f}"
        return f"${value['mean']:.1f} \\pm {value['std']:.1f}$"

    lines = [r"\begin{tabular}{llllrrr}", r"\hline",
             r"Environment & Model & Method & Gate & SR (\%) & IPR (\%) & Steps \\", r"\hline"]
    for row in rows:
        fields = [escape(row[field]) for field in ("environment", "model", "method", "gate")]
        fields += [cell(row[metric]) for metric in ("sr_pct", "ipr_pct", "mean_steps")]
        lines.append(" & ".join(fields) + r" \\")
    return "\n".join(lines + [r"\hline", r"\end{tabular}", ""])
