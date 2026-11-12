"""Observation-level checks with fixed logged action proposals."""

from __future__ import annotations

import random
import re
from dataclasses import replace

from ..records import Trajectory
from .corpus import evaluate_corpus


def drop_sentences(trajectory: Trajectory, rate: float, rng: random.Random) -> Trajectory:
    if not 0 <= rate <= 1:
        raise ValueError("Observation deletion rate must be in [0, 1]")

    def alter(text: str) -> str:
        sentences = re.findall(r"[^.!?\n]+[.!?]?", text)
        return " ".join(sentence for sentence in sentences if rng.random() >= rate)

    return replace(trajectory, records=[replace(record, observation=alter(record.observation))
                                        for record in trajectory.records],
                   final_observation=alter(trajectory.final_observation))


def evaluate_observation_changes(trajectories, components_factory, rates, policies, seed=0):
    output = []
    for rate in rates:
        rng = random.Random(seed)
        changed = [drop_sentences(trajectory, rate, rng) for trajectory in trajectories]
        for policy in policies:
            components = components_factory(policy)
            verified = {trajectory.traj_id: components.worker().process(trajectory, components.goal(trajectory))
                        for trajectory in changed}
            metrics = evaluate_corpus(changed, verified)
            output.append({"deletion_rate": rate, "grounding_policy": policy,
                           "label_quality": metrics["label_quality"], "violations": metrics["violations"],
                           "parsed_proposals": metrics["parsed_proposals"],
                           "applicability_labeled": metrics["applicability_labeled"],
                           "truth_coverage_pct": metrics["truth_coverage_pct"]})
    return output
