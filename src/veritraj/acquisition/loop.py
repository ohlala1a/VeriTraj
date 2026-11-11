"""Builder-Challenger loop (Algorithm 2).

  D_0 <- Build(d, U, E_dev)
  for k = 0..K-1:
      q_k <- Challenge(D_k)            probes (deterministic + optional LLM)
      f_k <- Adjudicate(D_k, q_k)      reproducible mismatches only
      if D_k passes and f_k is empty: break
      D_{k+1} <- Build(..., f_k)
  return D_k only if the final acceptance test (held-out problems) passes
"""

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
            llm_challenger: LLMChallenger | None = None) -> AcquisitionResult:
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
