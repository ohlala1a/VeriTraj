"""Trajectory and verification schemas. Latent applicability is used only for evaluation."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(slots=True)
class LoggedRecord:
    t: int
    observation: str
    generation: str
    executed: str | None = None
    feedback: str = ""
    env_reward: float = 0.0
    latent_applicable: bool | None = None
    prompt: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LoggedRecord":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})  # type: ignore[arg-type]


@dataclass(slots=True)
class Trajectory:
    traj_id: str
    task_id: str
    records: list[LoggedRecord]
    final_observation: str = ""
    family: str = ""
    problem: str = ""
    goal_text: str = ""
    policy: str = ""
    checkpoint: str = ""
    gate: bool = False
    success: bool = False
    env_return: float = 0.0
    group_id: str = ""
    seed: int = 0
    split: str = ""
    environment: str = ""
    success_observed: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        recs = [LoggedRecord.from_dict(r) for r in d.get("records", [])]
        meta = {k: d[k] for k in cls.__dataclass_fields__ if k in d and k != "records"}
        return cls(records=recs, **meta)

    def to_dict(self) -> dict:
        return asdict(self)

    def __len__(self) -> int:
        return len(self.records)


@dataclass(slots=True)
class VerifiedRecord:
    traj_id: str
    t: int
    span: tuple[int, int]
    char_span: tuple[int, int]
    action: str
    v: int
    witness: list[str] = field(default_factory=list)
    p: int = 0
    r_prog: float = 0.0
    h: float | None = None
    executed: str | None = None
    store_facts: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"], d["char_span"] = list(self.span), list(self.char_span)
        if d["h"] is not None and d["h"] == float("inf"):
            d["h"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VerifiedRecord":
        d = dict(d)
        d["span"], d["char_span"] = tuple(d["span"]), tuple(d["char_span"])
        return cls(**d)


def _open(path: str | Path, mode: str):
    p = str(path)
    return gzip.open(p, mode + "t", encoding="utf-8") if p.endswith(".gz") else open(p, mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with _open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_trajectories(path: str | Path) -> Iterator[Trajectory]:
    """JSONL (one trajectory per line, optionally .gz) or a YAML list of trajectories."""
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        with open(path, encoding="utf-8") as f:
            rows = yaml.safe_load(f) or []
    else:
        rows = read_jsonl(path)
    for d in rows:
        yield Trajectory.from_dict(d)


def discounted_return(traj: Trajectory, gamma: float = 0.99) -> float:
    return sum((gamma ** r.t) * (r.env_reward or 0.0) for r in traj.records)


def read_verified(path: str | Path) -> list[VerifiedRecord]:
    return [VerifiedRecord.from_dict(d) for d in read_jsonl(path)]
