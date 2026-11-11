"""Offline curation of a logged corpus into a supervised fine-tuning set.

Strategies (all read the same verified table):
  none                 every trajectory, every action span is a target
  outcome              successful trajectories only
  trajectory           trajectories without any flagged record
  outcome_trajectory   both of the above
  record               every trajectory; supported spans are targets,
                       flagged / unparsed spans are masked out
  record_weighted      as `record`, weight 1 for successful and omega for
                       failed trajectories

Reasoning text outside the action span stays in the completion but is not a
target. Prompts are rebuilt from the logged observations with the same
template the policy used (`build_prompt`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .records import Trajectory, VerifiedRecord

STRATEGIES = ("none", "outcome", "trajectory", "outcome_trajectory", "record", "record_weighted")

SYSTEM = (
    "You are a household agent in a text environment. At every step, think briefly and then give "
    "exactly one command on a line starting with 'Action:'."
)
_TASK = re.compile(r"Your task is to:\s*(.+)", re.IGNORECASE)


def task_of(traj: Trajectory) -> str:
    if traj.goal_text:
        return traj.goal_text
    m = _TASK.search(traj.records[0].observation) if traj.records else None
    return m.group(1).strip() if m else ""


def build_prompt(task: str, history: list[tuple[str, str]], observation: str, window: int = 6) -> str:
    parts = [SYSTEM, f"Task: {task}"]
    for obs, act in history[-window:] if window > 0 else []:
        parts.append(f"Observation: {obs}\n{act}")
    parts.append(f"Observation: {observation}")
    return "\n\n".join(parts) + "\n\n"


@dataclass
class CurationResult:
    strategy: str
    examples: list[dict] = field(default_factory=list)
    traj_total: int = 0
    traj_kept: int = 0
    records_total: int = 0
    records_target: int = 0

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "trajectories_kept_pct": round(100 * self.traj_kept / max(self.traj_total, 1), 1),
            "records_as_targets_pct": round(100 * self.records_target / max(self.records_total, 1), 1),
            "examples": len(self.examples),
        }


def _keep(traj: Trajectory, labels: list[VerifiedRecord], strategy: str) -> bool:
    clean = all(r.v == 1 for r in labels)
    return {
        "none": True, "record": True, "record_weighted": True,
        "outcome": traj.success, "trajectory": clean, "outcome_trajectory": traj.success and clean,
    }[strategy]


def curate(trajs: list[Trajectory], verified: dict[str, list[VerifiedRecord]], strategy: str,
           omega: float = 0.5, window: int = 6) -> CurationResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    res = CurationResult(strategy)
    record_level = strategy in {"record", "record_weighted"}
    for traj in trajs:
        labels = verified[traj.traj_id]
        if len(labels) != len(traj.records):
            raise ValueError(f"{traj.traj_id}: {len(labels)} labels for {len(traj.records)} records")
        res.traj_total += 1
        res.records_total += len(traj.records)
        if not _keep(traj, labels, strategy):
            continue
        res.traj_kept += 1
        task = task_of(traj)
        weight = 1.0 if (strategy != "record_weighted" or traj.success) else omega
        history: list[tuple[str, str]] = []
        for rec, lab in zip(traj.records, labels):
            target = (lab.v == 1) if record_level else True
            if target:
                res.records_target += 1
                res.examples.append({
                    "id": f"{traj.traj_id}#{rec.t}",
                    "prompt": rec.prompt or build_prompt(task, history, rec.observation, window),
                    "completion": rec.generation,
                    "target_char_spans": [list(lab.char_span)],
                    "weight": weight,
                })
            history.append((rec.observation, rec.generation))
    return res
