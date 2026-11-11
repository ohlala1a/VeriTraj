"""Builder: produces candidate domains D_k.

FileBuilder replays drafts stored on disk (round0.pddl, round1.pddl, ...),
which makes the loop reproducible without an API. LLMBuilder drafts and
repairs the domain with an OpenAI-compatible model from the documentation,
the action interface, sample development transitions and the adjudicator's
feedback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..llm import ChatClient, LLMError, extract_block
from .adjudicator import Mismatch
from .probes import DevTransition


class Builder(Protocol):
    def build(self, round_idx: int, previous: str | None, feedback: list[Mismatch]) -> str: ...


class FileBuilder:
    def __init__(self, drafts_dir: str | Path):
        self.paths = sorted(Path(drafts_dir).glob("round*.pddl"))
        if not self.paths:
            raise FileNotFoundError(f"no round*.pddl drafts in {drafts_dir}")
        self.log: list[dict] = []

    def build(self, round_idx: int, previous: str | None, feedback: list[Mismatch]) -> str:
        path = self.paths[min(round_idx, len(self.paths) - 1)]
        self.log.append({"round": round_idx, "draft": path.name, "feedback_items": len(feedback)})
        return path.read_text()


BUILD_PROMPT = """Write a typed STRIPS PDDL domain (requirements :strips :typing) for the environment below.
Use exactly the operator names and parameter orders of the action interface.

# Documentation
{docs}

# Action interface
{interface}

# Example transitions observed in the environment (state facts relevant to the action)
{transitions}

Return only the domain in a ```pddl block."""

REPAIR_PROMPT = """The candidate domain below was tested against the environment. Fix every reproducible
mismatch listed, keep everything else unchanged, and return the full domain in a ```pddl block.

```pddl
{domain}
```

# Mismatches
{feedback}"""


def describe_transitions(transitions: list[DevTransition], k: int = 12) -> str:
    lines = []
    for tr in transitions[:k]:
        rel = sorted(str(f) for f in tr.state if not f.args or set(f.args) & set(tr.args))
        add = sorted(map(str, tr.next_state - tr.state))
        rem = sorted(map(str, tr.state - tr.next_state))
        lines.append(f"- {tr.name}({', '.join(tr.args)}) | before: {', '.join(rel)} | added: {', '.join(add)} "
                     f"| removed: {', '.join(rem)}")
    return "\n".join(lines)


class LLMBuilder:
    def __init__(self, client: ChatClient, docs: str, interface: str, transitions: list[DevTransition],
                 temperature: float = 0.2):
        self.client = client
        self.docs = docs
        self.interface = interface
        self.transitions = transitions
        self.temperature = temperature
        self.log: list[dict] = []

    def build(self, round_idx: int, previous: str | None, feedback: list[Mismatch]) -> str:
        if previous is None:
            prompt = BUILD_PROMPT.format(docs=self.docs, interface=self.interface,
                                         transitions=describe_transitions(self.transitions))
        else:
            fb = "\n".join(f"- [{m.kind}] {m.operator}: {m.detail} {json.dumps(m.evidence)} (seen {m.count}x)"
                           for m in feedback)
            prompt = REPAIR_PROMPT.format(domain=previous, feedback=fb or "- none")
        out = self.client.chat([{"role": "user", "content": prompt}], self.temperature, 2500)
        text = extract_block(out, "pddl") or extract_block(out, "lisp") or extract_block(out)
        if text is None:
            raise LLMError("builder response contains no code block")
        self.log.append({"round": round_idx, "prompt_chars": len(prompt), "response_chars": len(out)})
        return text
