"""Checkpoint evaluation over explicit task sets and runtime-gate settings."""

from __future__ import annotations

from ..config import build_components
from ..envs.factory import make_environment
from ..policy.generate import load_generator
from ..records import write_jsonl
from ..runtime import init_run, write_json
from ..train.rollout import collect_group
from .corpus import evaluate_corpus


def run_evaluation(config: dict, *, policy=None, environment=None) -> dict:
    tasks = config["tasks"]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Evaluation requires a nonempty list of unique task IDs")
    seed = int(config.get("seed", 0))
    output = init_run(config["output_dir"], config, seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    components = build_components(config["components"])
    policy = policy if policy is not None else load_generator(config)
    env = environment if environment is not None else make_environment(config["environment"], components)
    trajectories, verified = [], {}
    try:
        for index, task_id in enumerate(tasks):
            group = collect_group(env, policy, components, task_id, group_size=int(config.get("group_size", 1)),
                                  seed=seed + index, gate=bool(config.get("gate", False)),
                                  max_steps=int(config.get("max_steps", 50)), group_id=f"eval-{index}",
                                  window=int(config.get("history_window", 6)),
                                  policy_name=config.get("method", "veritraj"), checkpoint=config["model"],
                                  environment=config.get("environment_name", config["environment"]["kind"]),
                                  split=config.get("split", "test"))
            trajectories.extend(group.trajectories)
            verified.update(group.verified)
    finally:
        env.close()
    write_jsonl(output / "trajectories.jsonl.gz", (trajectory.to_dict() for trajectory in trajectories))
    write_jsonl(output / "verified.jsonl.gz", (row.to_dict() for rows in verified.values() for row in rows))
    report = {"environment": config.get("environment_name", config["environment"]["kind"]),
              "model": config.get("model_name", config["model"]), "checkpoint": config["model"],
              "method": config.get("method", "veritraj"), "gate": bool(config.get("gate", False)),
              "seed": seed, "split": config.get("split", "test"), "protocol": config.get("protocol", "default"),
              "task_ids": tasks, "metrics": evaluate_corpus(trajectories, verified)}
    write_json(output / "metrics.json", report)
    return report
