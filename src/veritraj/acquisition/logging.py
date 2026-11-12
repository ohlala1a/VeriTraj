"""Per-round domains, probes, tool reports and model transcripts."""

from __future__ import annotations

from pathlib import Path

from ..runtime import write_json


class AcquisitionLogger:
    def __init__(self, output_dir, builder):
        self.output = Path(output_dir)
        self.builder = builder

    def __call__(self, round_index, domain_text, report, probes):
        output = self.output / f"round_{round_index}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "domain.pddl").write_text(domain_text, encoding="utf-8")
        write_json(output / "adjudicator.json", report.to_dict())
        write_json(output / "probes.json", [
            {"state": sorted(map(str, probe.state)), "name": probe.name, "args": list(probe.args),
             "objects_key": probe.objects_key, "expected": probe.expected, "kind": probe.kind,
             "note": probe.note} for probe in probes])
        if getattr(self.builder, "log", None):
            entry = self.builder.log[-1]
            write_json(output / "builder.json", entry)


class TranscriptClient:
    def __init__(self, client, output_dir):
        self.client = client
        self.output = Path(output_dir)
        self.calls = 0

    def chat(self, messages, temperature=0.0, max_tokens=512):
        self.calls += 1
        path = self.output / f"call_{self.calls:05d}.json"
        entry = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        try:
            response = self.client.chat(messages, temperature, max_tokens)
        except Exception as error:
            entry["error_type"] = type(error).__name__
            write_json(path, entry)
            raise
        entry["response"] = response
        write_json(path, entry)
        return response
