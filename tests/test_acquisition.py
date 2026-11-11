"""Builder-Challenger loop on the shipped drafts, plus an LLM builder with a fixed-reply client."""

import pytest

from veritraj.acquisition import FileBuilder, LLMBuilder, acquire
from veritraj.ops import acquisition_setup, make_adjudicators
from conftest import ROOT


@pytest.fixture(scope="module")
def setup(comps, trajs):
    return acquisition_setup(comps, list(trajs.values()))


def test_drafts_converge(setup):
    dev, acc, ch = make_adjudicators(setup)
    res = acquire(FileBuilder(ROOT / "examples/acquisition/drafts"), ch, dev, setup.transitions, acc, rounds=3)
    assert res.accepted and len(res.rounds) == 3
    assert {"incomplete_effect", "missing_precondition", "unintended_shortcut"} <= set(res.rounds[0]["mismatch_kinds"])
    assert res.rounds[1]["mismatch_kinds"] == ["missing_precondition", "unintended_shortcut"]


def test_reference_domain_passes(setup):
    dev, _, ch = make_adjudicators(setup)
    rep, _ = dev.evaluate((ROOT / "domains/alfworld/domain.pddl").read_text(), ch.probes(setup.transitions))
    assert rep.accepted, rep.mismatches


def test_type_probe_detects_broad_parameter(setup):
    dev, _, ch = make_adjudicators(setup)
    text = (ROOT / "domains/alfworld/domain.pddl").read_text().replace(
        ":parameters (?o - item ?r - receptacle)\n    :precondition (and (at agent ?r) (open ?r) (in ?o ?r)",
        ":parameters (?o - object ?r - receptacle)\n    :precondition (and (at agent ?r) (open ?r) (in ?o ?r)")
    rep, _ = dev.evaluate(text, ch.probes(setup.transitions))
    assert "overly_broad_type" in {m.kind for m in rep.mismatches} or rep.static_issues


def test_parse_error_is_reported(setup):
    dev, _, _ = make_adjudicators(setup)
    rep, dom = dev.evaluate("(define (domain broken")
    assert dom is None and rep.mismatches[0].kind == "parse_error"


class FixedReplyClient:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def chat(self, messages, temperature=0.0, max_tokens=512):
        self.prompts.append(messages[-1]["content"])
        return f"Here is the domain.\n```pddl\n{self.answer}\n```"


def test_llm_builder_fixed_reply(comps, setup):
    dev, acc, ch = make_adjudicators(setup)
    client = FixedReplyClient((ROOT / "domains/alfworld/domain.pddl").read_text())
    builder = LLMBuilder(client, "docs", "interface", setup.transitions)
    res = acquire(builder, ch, dev, setup.transitions, acc, rounds=3)
    assert res.accepted and len(res.rounds) == 1
    assert "Action interface" in client.prompts[0]
