"""Build and check domains with configured Builder and Challenger endpoints."""

import argparse
import os
from pathlib import Path

from veritraj.acquisition import LLMBuilder, LLMChallenger, acquire
from veritraj.acquisition.logging import AcquisitionLogger, TranscriptClient
from veritraj.config import build_components, resolve
from veritraj.pddl import load_problem
from veritraj.llm import OpenAICompatibleClient
from veritraj.ops import acquisition_setup, make_adjudicators
from veritraj.records import read_trajectories
from veritraj.runtime import init_run, load_yaml, write_json


def client_for(spec):
    prefix = spec["prefix"]
    base = os.environ.get(f"{prefix}_BASE_URL", spec.get("base_url", ""))
    key = os.environ.get(f"{prefix}_API_KEY", "")
    model = os.environ.get(f"{prefix}_MODEL", spec.get("model", ""))
    if not base or not key or not model:
        raise ValueError(f"Configure the endpoint, API key and model for {prefix}")
    return OpenAICompatibleClient(base, key, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    components = build_components(config["components"], progress=False)
    trajectories = list(read_trajectories(args.trajectories))
    if any(trajectory.split not in ("", "train", "dev", "valid_seen") for trajectory in trajectories):
        raise ValueError("Acquisition transitions must use the configured development split")
    acceptance = config.get("acceptance_problems")
    if acceptance is not None:
        acceptance = [load_problem(resolve(components.cfg, path)) for path in acceptance]
    setup = acquisition_setup(components, trajectories, acceptance_problems=acceptance)
    development, acceptance, challenger = make_adjudicators(setup)
    output = init_run(args.out, config, int(config.get("seed", 0)))
    builder_client = TranscriptClient(client_for(config["builder"]), output / "builder_calls")
    docs = Path(config["documentation"]).read_text(encoding="utf-8")
    interface = "\n".join(f"{name}({', '.join(t for _, t in schema.params)})"
                          for name, schema in components.domain.actions.items())
    builder = LLMBuilder(builder_client, docs, interface, setup.transitions,
                         temperature=float(config["builder"].get("temperature", .2)))
    llm_challenger = None
    if config.get("challenger"):
        challenger_client = TranscriptClient(client_for(config["challenger"]), output / "challenger_calls")
        llm_challenger = LLMChallenger(challenger_client, setup.env, setup.objects,
                                     k=int(config["challenger"].get("probes_per_transition", 4)))
    result = acquire(builder, challenger, development, setup.transitions, acceptance,
                     rounds=int(config.get("rounds", 3)), llm_challenger=llm_challenger,
                     round_callback=AcquisitionLogger(output, builder))
    write_json(output / "report.json", result.__dict__)
    if result.accepted:
        (output / "domain.pddl").write_text(result.domain_text, encoding="utf-8")
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
