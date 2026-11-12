"""Evaluation metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .pddl.parser import parse_atom_text
from .records import VerifiedRecord

CATEGORY_BY_PRED = {
    "at": "reachability", "visible": "reachability",
    "openable": "affordance", "can-heat": "affordance", "can-cool": "affordance",
    "can-clean": "affordance", "lamp-at": "affordance",
    "item": "affordance", "receptacle": "affordance", "lamp": "affordance", "place": "affordance",
    "open": "state", "closed": "state", "in": "state", "in-hand": "state", "handempty": "state",
    "lit": "sequence", "heated": "sequence", "cooled": "sequence", "cleaned": "sequence",
}


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted_invalid: bool, actually_invalid: bool) -> None:
        if predicted_invalid and actually_invalid:
            self.tp += 1
        elif predicted_invalid:
            self.fp += 1
        elif actually_invalid:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision": round(100 * self.precision, 1), "recall": round(100 * self.recall, 1),
                "f1": round(100 * self.f1, 1)}


def label_quality(labels: list[VerifiedRecord], applicable: list[bool | None]) -> Confusion:
    conf = Confusion()
    for lab, app in zip(labels, applicable):
        if lab.p or app is None:
            continue
        conf.add(lab.v == 0, not app)
    return conf


def invalid_proposal_rate(applicable: list[bool | None]) -> float:
    parsed = [a for a in applicable if a is not None]
    return sum(1 for a in parsed if not a) / len(parsed) if parsed else 0.0


CATEGORY_PRIORITY = ("reachability", "sequence", "state", "affordance", "other")


def category(witness: list[str]) -> str:
    """Category of a flagged record: the highest-priority category among its missing literals."""
    if not witness:
        return "none"
    cats = {CATEGORY_BY_PRED.get(parse_atom_text(w.removeprefix("not ")).pred, "other") for w in witness}
    return next(c for c in CATEGORY_PRIORITY if c in cats)


def violation_breakdown(labels: list[VerifiedRecord]) -> dict[str, int]:
    c = Counter(category(l.witness) for l in labels if l.v == 0 and not l.p)
    c["parse_failure"] = sum(1 for l in labels if l.p)
    return dict(c)
