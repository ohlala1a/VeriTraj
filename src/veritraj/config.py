"""Config loading and component wiring.

`build_components(cfg_path)` returns everything a worker needs (domain,
adapter, interface, store config, tokenizer, progress factory). Paths inside
the YAML are resolved against the repository root (the parent of configs/).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from .adapters.alfworld import AlfworldAdapter, build_adapter
from .interface import ActionInterface, InterfaceConfig
from .pddl import Domain, Literal, Problem, load_domain, load_problem
from .pipeline import RecomputeWorker, TrajectoryWorker
from .progress import ProgressAnnotator
from .records import Trajectory
from .store import StoreConfig
from .tokenization import OffsetTokenizer, make_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "alfworld.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(p.resolve().parent.parent)
    return cfg


def resolve(cfg: dict, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


@dataclass
class Components:
    cfg: dict
    domain: Domain
    adapter: AlfworldAdapter
    interface: ActionInterface
    store_config: StoreConfig
    tokenizer: OffsetTokenizer
    progress_factory: Callable[[list[Literal]], ProgressAnnotator] | None

    def worker(self) -> TrajectoryWorker:
        return TrajectoryWorker(self.domain, self.adapter, self.interface, self.store_config,
                                self.tokenizer, self.progress_factory)

    def recompute_worker(self) -> RecomputeWorker:
        return RecomputeWorker(self.domain, self.adapter, self.interface, self.store_config, self.tokenizer)

    def problem(self, traj: Trajectory) -> Problem:
        return _load_problem_cached(str(resolve(self.cfg, traj.problem)))

    def goal(self, traj: Trajectory) -> list[Literal]:
        return self.problem(traj).goal if traj.problem else []


@lru_cache(maxsize=4096)
def _load_problem_cached(path: str) -> Problem:
    return load_problem(path)


def build_components(cfg_path: str | Path | None = None, *, policy: str | None = None,
                     resolver=None, heuristic: str | None = None, progress: bool = True,
                     tokenizer: str | None = None) -> Components:
    cfg = load_config(cfg_path)
    domain = load_domain(resolve(cfg, cfg["domain"]))
    adapter = build_adapter(cfg["adapter"], policy=policy, resolver=resolver)
    interface = ActionInterface(domain, InterfaceConfig.from_dict(cfg.get("interface")))
    store_cfg = StoreConfig.from_dict(cfg.get("store"))
    tok = make_tokenizer(tokenizer or cfg.get("tokenizer"))
    pcfg = cfg.get("progress") or {}
    factory: Callable[[list[Literal]], ProgressAnnotator] | None = None
    if progress:
        kind = heuristic or pcfg.get("heuristic", "hff")

        def _factory(goal: list[Literal]) -> ProgressAnnotator:
            return ProgressAnnotator(domain, goal, kind, float(pcfg.get("gamma", 0.99)),
                                     int(pcfg.get("cache_size", 65536)), mutex=store_cfg.mutex)

        factory = _factory
    return Components(cfg, domain, adapter, interface, store_cfg, tok, factory)
