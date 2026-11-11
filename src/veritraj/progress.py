"""Relaxed-planning heuristics and potential-based progress annotation.

    Phi(b) = -h(b)
    r_prog = gamma * Phi(b') - Phi(b)    if b' != b and both h finite
           = 0                           otherwise

h is computed on the *true* facts of the store, over actions grounded on the
objects the store has typed so far. Unknown facts are not reachable for free,
with one exception: for a mutex pair such as open/closed whose value is unknown
for an object, both values are assumed in the relaxation (the attribute is
binary, so one of them holds). The verifier never makes this assumption. Values are cached by the store's incremental hash, so a symbolic
state that recurs is evaluated once.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass

from .pddl.ground import ground_all, static_predicates
from .pddl.model import Atom, Domain, GroundAction, Literal
from .store import FactStore

INF = math.inf
HEURISTICS = ("hff", "hadd", "hmax", "goalcount")


class RelaxedTask:
    """Delete relaxation over a fixed set of ground actions."""

    def __init__(self, actions: list[GroundAction], goal: list[Atom]):
        self.actions = actions
        self.goal = list(goal)
        self.fid: dict[Atom, int] = {}
        self.pre: list[list[int]] = []
        self.add: list[list[int]] = []
        self.by_pre: dict[int, list[int]] = {}
        self.no_pre: list[int] = []
        for ai, a in enumerate(actions):
            pre = [self._id(l.atom) for l in a.pre if l.positive]
            self.pre.append(pre)
            self.add.append([self._id(x) for x in a.add])
            if not pre:
                self.no_pre.append(ai)
            for f in pre:
                self.by_pre.setdefault(f, []).append(ai)
        self.goal_ids = [self.fid.get(g, -1) for g in self.goal]

    def _id(self, atom: Atom) -> int:
        if atom not in self.fid:
            self.fid[atom] = len(self.fid)
        return self.fid[atom]

    def _explore(self, facts: set[Atom], combine) -> tuple[dict[int, float], dict[int, int]]:
        cost: dict[int, float] = {}
        supporter: dict[int, int] = {}
        heap: list[tuple[float, int]] = []
        for atom in facts:
            fid = self.fid.get(atom)
            if fid is not None and fid not in cost:
                cost[fid] = 0.0
                heapq.heappush(heap, (0.0, fid))
        waiting = [len(p) for p in self.pre]
        acc = [0.0] * len(self.actions)

        def fire(ai: int, base: float) -> None:
            c = base + 1.0
            for f in self.add[ai]:
                if c < cost.get(f, INF):
                    cost[f] = c
                    supporter[f] = ai
                    heapq.heappush(heap, (c, f))

        for ai in self.no_pre:
            fire(ai, 0.0)
        done: set[int] = set()
        while heap:
            c, f = heapq.heappop(heap)
            if f in done or c > cost.get(f, INF):
                continue
            done.add(f)
            for ai in self.by_pre.get(f, ()):
                acc[ai] = combine(acc[ai], c)
                waiting[ai] -= 1
                if waiting[ai] == 0:
                    fire(ai, acc[ai])
        return cost, supporter

    def evaluate(self, facts: set[Atom], kind: str = "hff") -> float:
        if any(g < 0 for g in self.goal_ids):
            missing = [g for g, i in zip(self.goal, self.goal_ids) if i < 0 and g not in facts]
            if missing:
                return INF
        if kind == "goalcount":
            return float(sum(1 for g in self.goal if g not in facts))
        combine = max if kind == "hmax" else (lambda a, b: a + b)
        cost, supporter = self._explore(facts, combine)
        gids = [i for i in self.goal_ids if i >= 0]
        if any(i not in cost for i in gids):
            return INF
        if kind == "hmax":
            return max((cost[i] for i in gids), default=0.0)
        if kind == "hadd":
            return float(sum(cost[i] for i in gids))
        # hFF: extract a relaxed plan from h_add best supporters.
        plan: set[int] = set()
        stack = [i for i in gids if cost[i] > 0]
        seen: set[int] = set()
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            ai = supporter.get(f)
            if ai is None or ai in plan:
                continue
            plan.add(ai)
            stack.extend(p for p in self.pre[ai] if cost.get(p, INF) > 0)
        return float(len(plan))


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


class ProgressAnnotator:
    def __init__(self, domain: Domain, goal: list[Literal], heuristic: str = "hff",
                 gamma: float = 0.99, cache_size: int = 65536, prune_static: bool = True,
                 mutex: list[tuple[str, str]] | None = None):
        if heuristic not in HEURISTICS:
            raise ValueError(f"heuristic must be one of {HEURISTICS}")
        self.domain = domain
        self.goal = [l.atom for l in goal if l.positive]
        self.heuristic = heuristic
        self.gamma = gamma
        self.prune_static = prune_static
        self._static = static_predicates(domain)
        self._task: RelaxedTask | None = None
        self._task_key: tuple | None = None
        self._cache: OrderedDict[int, float] = OrderedDict()
        self._cache_size = cache_size
        self.stats = CacheStats()
        self.mutex = [tuple(p) for p in (mutex or [])]

    def _relaxed_facts(self, store: FactStore) -> set[Atom]:
        if not self.mutex:
            return store.true
        facts = set(store.true)
        for a, b in self.mutex:
            sig = self.domain.predicates.get(a)
            if not sig or len(sig) != 1:
                continue
            for obj, typ in store.types.items():
                if not self.domain.is_subtype(typ, sig[0]):
                    continue
                fa, fb = Atom(a, (obj,)), Atom(b, (obj,))
                if store.value(fa) is None and store.value(fb) is None:
                    facts.add(fa)
                    facts.add(fb)
        return facts

    def _relaxed_task(self, store: FactStore) -> RelaxedTask:
        static_true = {a for a in store.true if a.pred in self._static}
        key = (frozenset(store.types.items()), frozenset(static_true) if self.prune_static else None)
        if key != self._task_key:
            actions = ground_all(self.domain, store.types,
                                 static_true if self.prune_static else None, self._static)
            self._task = RelaxedTask(actions, self.goal)
            self._task_key = key
            self._cache.clear()
        return self._task  # type: ignore[return-value]

    def h(self, store: FactStore) -> float:
        key = store.hash
        if key in self._cache:
            self.stats.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.stats.misses += 1
        val = self._relaxed_task(store).evaluate(self._relaxed_facts(store), self.heuristic)
        if key not in self._cache:  # task rebuild clears the cache
            self._cache[key] = val
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return val

    def shaping(self, h_before: float, h_after: float, changed: bool) -> float:
        if not changed or math.isinf(h_before) or math.isinf(h_after):
            return 0.0
        return self.gamma * (-h_after) - (-h_before)
