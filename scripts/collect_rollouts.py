"""Collect simulator rollouts and verified records with an HF policy."""
import argparse
from veritraj.runtime import load_yaml, init_run
from veritraj.config import build_components
from veritraj.envs.factory import make_environment
from veritraj.policy.generate import load_generator
from veritraj.train.rollout import collect_group
from veritraj.records import write_jsonl

def main(config):
    import torch
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    output = init_run(config["output_dir"], config, seed)
    components = build_components(config["components"])
    policy = load_generator(config)
    env = make_environment(config["environment"], components)
    try:
        for index, task in enumerate(config["tasks"]):
            group = collect_group(env, policy, components, task, seed=seed+index,
                                  group_id=f"task-{index}", group_size=config.get("group_size", 8),
                                  max_steps=config.get("max_steps", 50), gate=config.get("gate", True),
                                  policy_name=config.get("backend", "hf"), checkpoint=config["model"],
                                  environment=config["environment"].get("kind", "pddl"),
                                  split=config.get("split", "train"))
            write_jsonl(output/f"task-{index}.jsonl.gz", (t.to_dict() for t in group.trajectories))
            write_jsonl(output/f"task-{index}-verified.jsonl.gz",
                        (r.to_dict() for rows in group.verified.values() for r in rows))
    finally:
        env.close()
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    main(load_yaml(parser.parse_args().config))
