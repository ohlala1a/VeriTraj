"""Group-normalized returns and action-span advantages."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .records import Trajectory, VerifiedRecord, discounted_return


@dataclass(slots=True)
class AdvantageConfig:
    lambda_hard: float = 2.0
    lambda_pot: float = 0.2
    eps_norm: float = 1e-6
    gamma: float = 0.99

    @classmethod
    def from_dict(cls, d: dict | None) -> "AdvantageConfig":
        d = d or {}
        return cls(**{k: float(d[k]) for k in ("lambda_hard", "lambda_pot", "eps_norm", "gamma") if k in d})


def group_normalize(returns: list[float], eps: float = 1e-6) -> list[float]:
    n = len(returns)
    if n == 0:
        return []
    mean = sum(returns) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / n)
    return [(r - mean) / (std + eps) for r in returns]


def record_advantage(v: int, a_bar: float, r_prog: float, cfg: AdvantageConfig) -> float:
    return v * a_bar - (1 - v) * cfg.lambda_hard + v * cfg.lambda_pot * r_prog


def group_advantages(group: list[Trajectory], verified: dict[str, list[VerifiedRecord]],
                     cfg: AdvantageConfig) -> dict[str, list[float]]:
    returns = [discounted_return(t, cfg.gamma) for t in group]
    a_bar = group_normalize(returns, cfg.eps_norm)
    out = {}
    for traj, ab in zip(group, a_bar):
        out[traj.traj_id] = [record_advantage(r.v, ab, r.r_prog, cfg) for r in verified[traj.traj_id]]
    return out


def token_advantages(n_tokens: int, spans: list[tuple[int, int]], advs: list[float]) -> tuple[list[float], list[int]]:
    """Broadcast per-record advantages to token positions of one concatenated sequence."""
    adv = [0.0] * n_tokens
    mask = [0] * n_tokens
    for (u, w), a in zip(spans, advs):
        for j in range(max(u, 0), min(w, n_tokens)):
            adv[j] = a
            mask[j] = 1
    return adv, mask


def clipped_objective(logp_new: list[float], logp_old: list[float], logp_ref: list[float],
                      adv: list[float], mask: list[int], eps_clip: float = 0.2, beta: float = 0.01) -> float:

    n = sum(mask)
    if n == 0:
        return 0.0
    total = 0.0
    for lp, lo, lr, a, m in zip(logp_new, logp_old, logp_ref, adv, mask):
        if not m:
            continue
        ratio = math.exp(lp - lo)
        clipped = min(max(ratio, 1 - eps_clip), 1 + eps_clip)
        surrogate = min(ratio * a, clipped * a)
        d = lr - lp
        kl = math.exp(d) - d - 1
        total += surrogate - beta * kl
    return total / n
