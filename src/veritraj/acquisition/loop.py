"""Iterative domain construction, challenge probes and acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .adjudicator import Adjudicator, Report
from .builder import Builder
from .probes import DeterministicChallenger, DevTransition, LLMChallenger


@dataclass
class AcquisitionResult:
    accepted: bool
    domain_text: str | None
    rounds: list[dict] = field(default_factory=list)
    final_report: dict | None = None


def acquire(builder: Builder, challenger: DeterministicChallenger, adjudicator: Adjudicator,
            transitions: list[DevTransition], acceptance: Adjudicator | None = None, rounds: int = 3,
            llm_challenger: LLMChallenger | None = None, round_callback=None) -> AcquisitionResult:
    if rounds < 1:
        raise ValueError("Acquisition rounds must be positive")
    base_probes = challenger.probes(transitions)
    history: list[dict] = []
    text: str | None = None
    report: Report | None = None
    for k in range(rounds):
        text = builder.build(k, text, report.mismatches if report else [])
        probes = list(base_probes)
        if llm_challenger is not None:
            probes += llm_challenger.probes(transitions, text)
        report, _ = adjudicator.evaluate(text, probes)
        if round_callback is not None:
            round_callback(k, text, report, probes)
        history.append({"round": k, **report.rates(),
                        "mismatch_kinds": sorted({m.kind for m in report.mismatches}),
                        "feedback": [f"[{m.kind}] {m.operator}: {m.detail}" + (f" [x{m.count}]" if m.count > 1 else "")
                                     for m in report.mismatches]})
        if report.accepted:
            break
    if report is None or not report.accepted or text is None:
        return AcquisitionResult(False, None, history)
    final = (acceptance or adjudicator).evaluate(text, base_probes)[0]
    return AcquisitionResult(final.accepted, text if final.accepted else None, history, final.to_dict())
