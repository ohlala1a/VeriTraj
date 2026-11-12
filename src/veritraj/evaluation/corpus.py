"""Metrics over logged proposals, with explicit truth coverage."""

from __future__ import annotations

from collections import Counter

from ..metrics import label_quality, violation_breakdown
from ..records import Trajectory, VerifiedRecord, discounted_return


def evaluate_corpus(trajectories: list[Trajectory], verified: dict[str, list[VerifiedRecord]],
                    gamma: float = 0.99) -> dict:
    ids = [trajectory.traj_id for trajectory in trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("Trajectory IDs must be unique")
    labels, truth = [], []
    total_records = known = invalid = parsed = blocked = 0
    families = Counter()
    for trajectory in trajectories:
        rows = verified[trajectory.traj_id]
        if len(rows) != len(trajectory.records):
            raise ValueError(f"{trajectory.traj_id}: label count differs from record count")
        families[trajectory.family] += 1
        for record, label in zip(trajectory.records, rows):
            if record.t != label.t:
                raise ValueError(f"{trajectory.traj_id}: record index differs from label index")
            total_records += 1
            parsed += int(not label.p)
            blocked += int(trajectory.gate and not label.v)
            if not label.p and record.latent_applicable is not None:
                known += 1
                invalid += int(not record.latent_applicable)
            labels.append(label)
            truth.append(record.latent_applicable)
    count = len(trajectories)
    observed = [trajectory for trajectory in trajectories if trajectory.success_observed]
    successful = [trajectory for trajectory in observed if trajectory.success]
    return {
        "trajectories": count,
        "records": total_records,
        "successful_trajectories": len(successful),
        "sr_pct": 100.0 * len(successful) / len(observed) if observed else None,
        "success_observed": len(observed),
        "success_coverage_pct": 100.0 * len(observed) / count if count else None,
        "ipr_pct": 100.0 * invalid / known if known else None,
        "parsed_proposals": parsed,
        "applicability_labeled": known,
        "invalid_proposals": invalid,
        "truth_coverage_pct": 100.0 * known / parsed if parsed else None,
        "parse_failures": sum(label.p for label in labels),
        "gate_blocked": blocked,
        "mean_steps": total_records / count if count else None,
        "successful_mean_steps": (sum(len(trajectory.records) for trajectory in successful) / len(successful)
                                  if successful else None),
        "mean_discounted_return": (sum(discounted_return(trajectory, gamma) for trajectory in trajectories) / count
                                   if count else None),
        "label_quality": label_quality(labels, truth).as_dict() if known else None,
        "violations": violation_breakdown(labels),
        "families": dict(sorted(families.items())),
    }


def compare_workers(trajectories: list[Trajectory], components) -> dict:
    differences = []
    for trajectory in trajectories:
        incremental = components.worker().process(trajectory, components.goal(trajectory))
        recomputed = components.recompute_worker().process(trajectory)
        if len(incremental) != len(recomputed):
            raise ValueError(f"{trajectory.traj_id}: worker output length differs")
        for left, right in zip(incremental, recomputed):
            if (left.v, left.p, left.witness, left.char_span) != (right.v, right.p, right.witness, right.char_span):
                differences.append({"traj_id": trajectory.traj_id, "t": left.t,
                                    "incremental": left.to_dict(), "recompute": right.to_dict()})
    return {"disagreements": len(differences), "details": differences}
