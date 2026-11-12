"""Incremental fact store under open-world semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

from .pddl.model import Atom


def _fact_key(atom: Atom) -> int:
    digest = hashlib.blake2b(str(atom).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


@dataclass(slots=True)
class StoreConfig:
    functional: dict[str, tuple[int, ...]] = field(default_factory=dict)
    mutex: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "StoreConfig":
        d = d or {}
        return cls(
            functional={k: tuple(v) for k, v in (d.get("functional") or {}).items()},
            mutex=[tuple(p) for p in (d.get("mutex") or [])],  # type: ignore[misc]
        )


@dataclass(slots=True)
class Delta:
    add: list[Atom] = field(default_factory=list)
    delete: list[Atom] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.add) + len(self.delete) + len(self.types)

    def extend(self, other: "Delta") -> "Delta":
        self.add.extend(other.add)
        self.delete.extend(other.delete)
        self.types.update(other.types)
        return self


class FactStore:
    __slots__ = ("config", "true", "false", "types", "_hash", "_index", "_mutex", "version")

    def __init__(self, config: StoreConfig | None = None):
        self.config = config or StoreConfig()
        self.true: set[Atom] = set()
        self.false: set[Atom] = set()
        self.types: dict[str, str] = {}
        self._hash = 0
        self._index: dict[tuple, Atom] = {}
        self._mutex: dict[str, str] = {}
        for a, b in self.config.mutex:
            self._mutex[a] = b
            self._mutex[b] = a
        self.version = 0

    def holds(self, atom: Atom) -> bool:
        return atom in self.true

    def known_false(self, atom: Atom) -> bool:
        return atom in self.false

    def value(self, atom: Atom) -> bool | None:
        if atom in self.true:
            return True
        if atom in self.false:
            return False
        return None

    def lookup(self, pred: str, *key: str) -> Atom | None:
        """Current value of a functional predicate, e.g. lookup('at', 'agent')."""
        return self._index.get((pred, *key))

    @property
    def hash(self) -> int:
        return self._hash

    def __len__(self) -> int:
        return len(self.true)

    def _set_true(self, atom: Atom) -> bool:
        if atom in self.true:
            return False
        fkey = self._functional_key(atom)
        if fkey is not None:
            old = self._index.get(fkey)
            if old is not None and old != atom:
                self._set_false(old)
            self._index[fkey] = atom
        partner = self._mutex.get(atom.pred)
        if partner is not None:
            self._set_false(Atom(partner, atom.args))
        self.false.discard(atom)
        self.true.add(atom)
        self._hash ^= _fact_key(atom)
        return True

    def _set_false(self, atom: Atom) -> bool:
        changed = False
        if atom in self.true:
            self.true.remove(atom)
            self._hash ^= _fact_key(atom)
            fkey = self._functional_key(atom)
            if fkey is not None and self._index.get(fkey) == atom:
                del self._index[fkey]
            changed = True
        if atom not in self.false:
            self.false.add(atom)
            changed = True
        return changed

    def _functional_key(self, atom: Atom) -> tuple | None:
        idx = self.config.functional.get(atom.pred)
        if idx is None:
            return None
        return (atom.pred, *(atom.args[i] for i in idx))

    def apply(self, delta: Delta) -> bool:
        """Apply deletes, then adds (PDDL semantics). Returns True if anything changed."""
        changed = False
        for obj, typ in delta.types.items():
            if self.types.get(obj) != typ:
                self.types[obj] = typ
                changed = True
        for atom in delta.delete:
            changed |= self._set_false(atom)
        for atom in delta.add:
            changed |= self._set_true(atom)
        if changed:
            self.version += 1
        return changed

    def assert_facts(self, true: Iterable[Atom] = (), false: Iterable[Atom] = ()) -> bool:
        return self.apply(Delta(add=list(true), delete=list(false)))

    def copy(self) -> "FactStore":
        other = FactStore(self.config)
        other.true = set(self.true)
        other.false = set(self.false)
        other.types = dict(self.types)
        other._hash = self._hash
        other._index = dict(self._index)
        other.version = self.version
        return other

    def snapshot(self) -> dict:
        return {
            "true": sorted(map(str, self.true)),
            "false": sorted(map(str, self.false)),
            "types": dict(sorted(self.types.items())),
        }
