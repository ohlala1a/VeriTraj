# VeriTraj

VeriTraj provides record-level verification and curation for LLM-agent trajectories using
state-dependent PDDL constraints. It maintains a symbolic belief state as observations and
executed actions arrive, then checks each proposed action against the current state.

The resulting records contain validity labels, unsupported preconditions, action-token spans,
and planning-progress annotations. These records support both online policy updates and
offline supervised fine-tuning.

Due to privacy restrictions, we provide example trajectory data that illustrates the core
data-engineering workflow, including state maintenance, constraint verification, and record-level curation.

## What is included

- Incremental belief-state maintenance, action parsing, and verification with violation witnesses.
- PDDL domains and problems for household tasks and abstract CyberBattle actions.
- Builder–Challenger constraint acquisition with executable checks and per-round logs.
- Rollout collection, action-span policy optimization, and six offline curation strategies.
- Checkpoint evaluation, gate-on/gate-off comparisons, transfer evaluation, and timing tools.

## Repository structure

| Directory | Contents |
| --- | --- |
| `src/veritraj/` | Verification, state maintenance, policy training, curation, and evaluation |
| `configs/` | Environment, collection, training, SFT, and evaluation settings |
| `domains/` | PDDL action schemas and task problems |
| `examples/` | Trajectory data and constraint-acquisition inputs |
| `scripts/` | Collection, training, evaluation, and table-export entry points |
| `tests/` | Component tests and end-to-end checks |

## Quick start

Run the following commands from the repository root to verify trajectories, export curated
training records, and evaluate the verifier labels.

```bash
veritraj verify examples/trajectories/household_examples.yaml --out runs/verified.jsonl
veritraj curate examples/trajectories/household_examples.yaml --verified runs/verified.jsonl --out runs/sft
veritraj eval-labels examples/trajectories/household_examples.yaml
```

The verified JSONL file stores one result per action record. Curation exports prompt/completion
pairs with target character spans and example weights. Observations and earlier actions provide
context within the configured history window, while selected action spans determine the supervised targets.

