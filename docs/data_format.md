# Data formats

## Logged trajectories (input)

One trajectory per line (JSONL, optionally gzipped), or a YAML list with the same fields.

| field | type | meaning |
|---|---|---|
| `traj_id` | str | unique id; the unit of partitioning |
| `task_id` | str | task instance |
| `group_id` | str | rollout group (same task, same policy snapshot); used for group-relative advantages |
| `family` | str | task family (pick, examine, clean, heat, cool, pick_two) |
| `problem` | str | path of the PDDL problem, relative to the repository root; used for goals (progress) and for evaluation only |
| `policy`, `checkpoint` | str | which policy produced the trajectory |
| `gate` | bool | whether a runtime checker was active while logging |
| `success` | bool | task outcome |
| `records` | list | action records, see below |
| `final_observation` | str | observation after the last action |

Each record:

| field | type | meaning |
|---|---|---|
| `t` | int | position in the trajectory |
| `observation` | str | o_t, seen before generating |
| `prompt` | str, optional | the exact model input h_t; rebuilt from the history if absent |
| `generation` | str | y_t, the full model output (reasoning + command) |
| `executed` | str or null | operator the environment ran, e.g. `put(mug_1, cabinet_1)`; null if blocked, unparsable or rejected by the environment |
| `env_reward` | float | environment reward |
| `latent_applicable` | bool or null | evaluation only: whether the proposal was applicable in the simulator state |

The store is updated from `executed`, never from the proposal. Proposals are what the labels refer to.

## Verified table (output of `veritraj verify`)

One row per record:

| field | meaning |
|---|---|
| `traj_id`, `t` | record key |
| `span` | [u, w) token span of the command in `generation` |
| `char_span` | the same span in characters |
| `action` | grounded proposal, or `<parse-failure>` |
| `v` | 1 if supported, 0 if flagged or unparsed |
| `witness` | unsupported preconditions, e.g. `["open(microwave_1)"]`; type evidence appears as `item(fork_2)` |
| `p` | 1 if no command could be parsed (span then covers the whole generation) |
| `r_prog` | potential-based progress annotation (0 if the store did not change or h is infinite) |
| `h` | heuristic value of the store the record was checked against (null = unreachable) |
| `executed` | copied from the input |
| `store_facts` | number of true facts in the store |

## SFT export (output of `veritraj curate`)

| field | meaning |
|---|---|
| `id` | `traj_id#t` |
| `prompt` | model input |
| `completion` | generation |
| `target_char_spans` | character spans inside `completion` that receive loss |
| `weight` | example weight (outcome weighting) |
