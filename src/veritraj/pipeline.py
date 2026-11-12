"""Incremental belief maintenance and action verification with a full-prefix reference worker."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .adapters.alfworld import AlfworldAdapter
from .interface import ActionInterface, ParsedAction
from .pddl.ground import instantiate
from .pddl.model import Domain, GroundAction, Literal
from .progress import ProgressAnnotator
from .records import LoggedRecord, Trajectory, VerifiedRecord
from .store import Delta, FactStore, StoreConfig
from .tokenization import OffsetTokenizer, RegexTokenizer, char_to_token_span
from .verifier import Verdict, Verifier

_EXEC = re.compile(r"^\s*([a-z][a-z_\-]*)\s*\((.*)\)\s*$", re.IGNORECASE)


def parse_executed(domain: Domain, text: str | None) -> GroundAction | None:
    """'put(mug_1, microwave_1)' -> GroundAction; None/'noop' -> None."""
    if not text or text.strip().lower() in {"noop", "none", "null"}:
        return None
    m = _EXEC.match(text)
    if not m:
        raise ValueError(f"cannot parse executed operator '{text}'")
    name = m.group(1).lower()
    args = tuple(a.strip().lower() for a in m.group(2).split(",") if a.strip())
    schema = domain.actions.get(name)
    if schema is None or schema.arity != len(args):
        raise ValueError(f"executed operator '{text}' does not match the domain")
    return instantiate(schema, args)


def effect_delta(action: GroundAction | None) -> Delta:
    if action is None:
        return Delta()
    return Delta(add=list(action.add), delete=list(action.delete))


@dataclass(slots=True)
class StepResult:
    parsed: ParsedAction
    verdict: Verdict
    record: VerifiedRecord


class TrajectoryWorker:
    def __init__(self, domain: Domain, adapter: AlfworldAdapter, interface: ActionInterface,
                 store_config: StoreConfig | None = None, tokenizer: OffsetTokenizer | None = None,
                 progress_factory: Callable[[list[Literal]], ProgressAnnotator] | None = None):
        self.domain = domain
        self.adapter = adapter
        self.interface = interface
        self.verifier = Verifier(domain)
        self.store_config = store_config or StoreConfig()
        self.tokenizer = tokenizer or RegexTokenizer()
        self.progress_factory = progress_factory
        self.reset("", [])

    def reset(self, traj_id: str, goal: list[Literal], first_observation: str | None = None) -> None:
        self.traj_id = traj_id
        self.store = FactStore(self.store_config)
        self.annotator = self.progress_factory(goal) if (self.progress_factory and goal) else None
        self._pending: VerifiedRecord | None = None
        self._h_pending: float | None = None
        self._hash_pending: int | None = None
        self.t = 0
        if first_observation is not None:
            self.store.apply(self.adapter.ground(first_observation, self.store, first=True))

    def maintain(self, executed: GroundAction | None, observation: str) -> bool:
        """b_t <- T_sym(b_{t-1}, a~_{t-1}, P_obs(o_t)); finalizes the previous record."""
        changed = self.store.apply(effect_delta(executed))
        changed |= self.store.apply(self.adapter.ground(observation, self.store))
        self._finalize_pending()
        return changed

    def verify(self, generation: str, executed_text: str | None = None) -> StepResult:
        parsed = self.interface.parse(generation, self.store)
        verdict = self.verifier.check(self.store, parsed)
        offsets = self.tokenizer.offsets(generation)
        span = char_to_token_span(offsets, parsed.char_span)
        rec = VerifiedRecord(
            traj_id=self.traj_id, t=self.t, span=span, char_span=parsed.char_span,
            action=parsed.canonical(), v=verdict.label, witness=list(verdict.witness),
            p=int(verdict.parse_failed), executed=executed_text, store_facts=len(self.store),
        )
        if self.annotator is not None:
            self._h_pending = self.annotator.h(self.store)
            rec.h = self._h_pending
            self._hash_pending = self.store.hash
        self._pending = rec
        self.t += 1
        return StepResult(parsed, verdict, rec)

    def _finalize_pending(self) -> None:
        rec = self._pending
        if rec is None or self.annotator is None:
            return
        h_after = self.annotator.h(self.store)
        rec.r_prog = self.annotator.shaping(self._h_pending, h_after, self.store.hash != self._hash_pending)  # type: ignore[arg-type]

    def process(self, traj: Trajectory, goal: list[Literal] | None = None) -> list[VerifiedRecord]:
        out: list[VerifiedRecord] = []
        if not traj.records:
            return out
        self.reset(traj.traj_id, goal or [], traj.records[0].observation)
        prev: GroundAction | None = None
        for i, rec in enumerate(traj.records):
            if i > 0:
                self.maintain(prev, rec.observation)
            out.append(self.verify(rec.generation, rec.executed).record)
            prev = parse_executed(self.domain, rec.executed)
        self.maintain(prev, traj.final_observation or "")
        return out


class RecomputeWorker:
    """Reference: re-derive b_t from r_{1:t-1} at every record."""

    def __init__(self, domain: Domain, adapter: AlfworldAdapter, interface: ActionInterface,
                 store_config: StoreConfig | None = None, tokenizer: OffsetTokenizer | None = None):
        self.domain = domain
        self.adapter = adapter
        self.interface = interface
        self.verifier = Verifier(domain)
        self.store_config = store_config or StoreConfig()
        self.tokenizer = tokenizer or RegexTokenizer()

    def _store_at(self, records: list[LoggedRecord], executed: list[GroundAction | None], t: int) -> FactStore:
        store = FactStore(self.store_config)
        store.apply(self.adapter.ground(records[0].observation, store, first=True))
        for k in range(1, t + 1):
            store.apply(effect_delta(executed[k - 1]))
            store.apply(self.adapter.ground(records[k].observation, store))
        return store

    def process(self, traj: Trajectory, goal: list[Literal] | None = None) -> list[VerifiedRecord]:
        executed = [parse_executed(self.domain, r.executed) for r in traj.records]
        out = []
        for t, rec in enumerate(traj.records):
            store = self._store_at(traj.records, executed, t)
            parsed = self.interface.parse(rec.generation, store)
            verdict = self.verifier.check(store, parsed)
            span = char_to_token_span(self.tokenizer.offsets(rec.generation), parsed.char_span)
            out.append(VerifiedRecord(
                traj_id=traj.traj_id, t=t, span=span, char_span=parsed.char_span,
                action=parsed.canonical(), v=verdict.label, witness=list(verdict.witness),
                p=int(verdict.parse_failed), executed=rec.executed, store_facts=len(store),
            ))
        return out


def resolve_problem_path(traj: Trajectory, root: str | Path) -> Path:
    p = Path(traj.problem)
    return p if p.is_absolute() else Path(root) / p
