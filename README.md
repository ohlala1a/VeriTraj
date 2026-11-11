# VeriTraj

Reference implementation of **VeriTraj**, a framework for record-level verification and curation
of LLM agent trajectory data. Each trajectory is processed as a stream. VeriTraj incrementally
maintains a symbolic fact store under open-world semantics. It checks every action record against
executable, state-dependent PDDL constraints and writes a verified table. That table holds a label,
a violation witness, the token span of the action and a progress annotation for every record. The
same table drives online span-level advantages and offline record-level curation. Constraints are
acquired with a Builder–Challenger loop that accepts a domain only after deterministic, executable
tests.

## Scope of this release

This repository contains the core verification and curation code, together with a household
(ALFWorld-style) domain, a few hand-written example trajectories, and tests. The examples show the
data format and the cases the verifier has to handle. **They are not the data used in the paper, and
running the code on them does not reproduce the paper's tables.**

The following are not released, as stated in the paper's Ethics and Artifacts section:
- the RL training loop and policy checkpoints;
- the logged trajectory corpora;
- the CyberBattleSim adapter and constraint domain.

What you can check with this code are properties that hold independently of the data:
- incremental maintenance yields the same labels as recomputation from the full prefix;
- the per-record verification cost does not grow with trajectory length;
- witnesses name the unsupported preconditions;
- record-level curation keeps more supervision than trajectory-level filters;
- the acquisition loop turns tool feedback into domain repairs.

| Paper | Code |
|---|---|
| Data model, record schema (Sec. II-A) | `records.py`, `docs/data_format.md` |
| Ground / Maintain: adapter, fact store (Sec. III-A) | `adapters/alfworld.py`, `store.py`, `pipeline.py` |
| Verify: labels and witnesses (Sec. III-B) | `interface.py`, `verifier.py`, `tokenization.py` |
| Online use: span-level advantages, Eqs. (3)–(5) (Sec. III-C) | `advantage.py` |
| Progress annotation, h_FF with cache (Sec. III-D) | `progress.py` |
| Offline record-level curation (Sec. III-E) | `curation.py` |
| Builder–Challenger acquisition, Algorithm 2 (Sec. IV) | `acquisition/` |
| Recomputation baseline, label quality, cost benchmarks (Sec. V-D) | `pipeline.RecomputeWorker`, `metrics.py`, `ops.py` |

## Installation

Python 3.10 or newer. The only runtime dependency is PyYAML.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 76 tests, a few seconds
```

Optional extras:
- `pip install -e ".[hf]"`: use the policy model's tokenizer for token spans (`tokenizer: hf:<model>`).
- Fast Downward: set `FAST_DOWNWARD=/path/to/fast-downward.py` to plan with `planner.fast_downward_plan`.
- An OpenAI-compatible endpoint, for the LLM builder, the LLM challenger and the `llm_impute` grounding
  policy: set `VERITRAJ_LLM_BASE_URL`, `VERITRAJ_LLM_API_KEY` and `VERITRAJ_LLM_MODEL`.

## Quick start

`make demo` runs all of the steps below. Each step can also be run on its own:

```bash
E=examples/trajectories/household_examples.yaml

# Static checks of the domain and problems, plus a solvability check
veritraj check-domain domains/alfworld/domain.pddl domains/alfworld/problems/*.pddl --plan

# Replay the logged trajectories against their PDDL problems (consistency of `executed` fields)
veritraj check-trajectories $E

# Build the verified trajectory table
veritraj verify $E --out runs/demo/verified.jsonl

# Show labels, spans, witnesses and progress for one trajectory
veritraj show $E --traj heat-mug-01/b
```

```
  t=5   ok    goto(countertop_1, microwave_1)            h=  4 r_prog=+1.03  span='go to microwave 1'
  t=6   FLAG  put(mug_1, microwave_1)                    h=  3 r_prog=+0.00  span='put mug 1 in/on microwave 1'
        witness: {open(microwave_1)}
  t=7   ok    heat(mug_1, microwave_1)                   h=  3 r_prog=+1.02  span='heat mug 1 with microwave 1'
```

The history still shows the microwave open at t=1. The `close` at t=2, however, replaced that fact
in the store. The put at t=6 is therefore flagged, and its witness names the missing fact.

```bash
# Curation strategies -> SFT files (prompt, completion, target character spans, weight)
veritraj curate $E --verified runs/demo/verified.jsonl --out runs/demo/sft

# Span-level advantages for a rollout group (Eq. 4)
veritraj advantages $E --verified runs/demo/verified.jsonl

# Label quality vs. latent applicability; incremental-vs-recompute agreement
veritraj eval-labels $E

# Builder–Challenger loop with the shipped drafts (no API needed)
veritraj acquire $E --drafts examples/acquisition/drafts --out runs/demo/acquisition

# Cost benchmarks
veritraj bench length       # per-record cost vs. trajectory length, incremental vs. recompute
veritraj bench state        # verification and heuristic cost vs. state size
```

The acquisition demo uses three drafts. The first has three seeded defects: `take` and `put` lack
`(open ?r)`, `close` does not delete `open`, and `goto` lacks `(at agent ?from)`. The adjudicator
reports each defect once, as one aggregated mismatch, and the loop accepts the third draft:

```
round 0: parse=True trace=97.8% probes=92.8% plan=100.0% execute=0.0% mismatches=5
     [incomplete_effect] close: successor of close(microwave_1) differs from the environment
     [missing_precondition] take: ... (removed open(countertop_1)) [x6]
     ...
round 2: parse=True trace=100.0% probes=100.0% plan=100.0% execute=100.0% mismatches=0
accepted
```

To run the loop with an LLM, use `veritraj acquire $E --llm [--llm-challenger]`. The challenger
reads its own endpoint from `VERITRAJ_CHALLENGER_*`, so the Builder and the Challenger can use
different models.

## Repository layout

```
configs/alfworld.yaml          store rules, interface, adapter lexicon, heuristic, advantage, curation
domains/alfworld/domain.pddl   household constraint domain (typed STRIPS)
domains/alfworld/problems/     PDDL problems for the example trajectories
examples/trajectories/         hand-written example trajectories (YAML; JSONL is also accepted)
examples/acquisition/          task documentation and Builder drafts for the acquisition demo
src/veritraj/
  pddl/                        parser, model, grounding, static checks
  store.py                     incremental fact store (open world, functional/mutex rules, hash)
  adapters/alfworld.py         observation grounding with conservative / rule_norm / llm_impute
  interface.py                 generated text -> grounded operator + character span
  verifier.py                  V(b, a) with violation witness
  pipeline.py                  TrajectoryWorker (incremental) and RecomputeWorker (reference)
  progress.py                  h_FF / h_add / h_max / goal-count, cached potential shaping
  advantage.py                 Eqs. (3)-(5)
  curation.py                  six curation strategies, SFT export
  acquisition/                 environment, probes, adjudicator, builders, loop
  latent.py                    replay against the PDDL problem (consistency, evaluation)
  metrics.py, ops.py, cli.py   evaluation, benchmarks, command line
tests/                         unit and end-to-end tests
```

## Using your own data

1. **Format.** Write logged trajectories in the format of `docs/data_format.md`. One JSON object per
   line works, and so does `.jsonl.gz`.
2. **New environment.** If your observations come from another environment:
   - implement an adapter with the `ground(observation, store, first)` interface;
   - write a PDDL domain;
   - list the store rules (functional predicates, mutex pairs) in a config file.
3. **Acquisition against a simulator.** Implement `acquisition.environment.Environment` on top of the
   simulator's reset-to-state and step API. The shipped `ReferenceEnvironment` answers from a
   reference PDDL model.
4. **Parallel runs.** `veritraj verify --workers N` partitions the corpus by `traj_id` over a process
   pool. Stores are private to a trajectory, so no coordination is needed.

## License

MIT, see `LICENSE`.
