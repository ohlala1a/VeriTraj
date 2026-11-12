"""Controls share the collector and optimizer."""
from ..advantage import group_normalize
from ..records import discounted_return

ALGORITHMS = ("veritraj", "grpo", "grpo_progress", "shielded", "no_progress", "trajectory_penalty", "trajectory_filter")

def advantages(group, cfg, algorithm="veritraj"):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algorithm!r}. PPO requires a trained value head.")
    returns = [discounted_return(t, cfg.gamma) for t in group.trajectories]
    if algorithm == "trajectory_penalty":
        returns = [r-cfg.lambda_hard*sum(1-l.v for l in group.verified[t.traj_id])
                   for r, t in zip(returns, group.trajectories)]
    normalized = group_normalize(returns, cfg.eps_norm)
    out = {}
    for traj, task_adv in zip(group.trajectories, normalized):
        labels = group.verified[traj.traj_id]
        if algorithm == "trajectory_filter" and any(l.v != 1 for l in labels):
            out[traj.traj_id] = None
            continue
        values = []
        for label in labels:
            if algorithm in {"grpo", "trajectory_penalty", "trajectory_filter"}:
                value = task_adv
            elif algorithm in {"grpo_progress", "shielded"}:
                value = task_adv + label.v*cfg.lambda_pot*label.r_prog
            else:
                potential = cfg.lambda_pot*label.r_prog if algorithm == "veritraj" else 0
                value = label.v*(task_adv+potential)-(1-label.v)*cfg.lambda_hard
            values.append(value)
        out[traj.traj_id] = values
    return out
