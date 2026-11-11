PY ?= python
E := examples/trajectories/household_examples.yaml
OUT := runs/demo

.PHONY: install test demo bench clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

demo:
	veritraj check-domain domains/alfworld/domain.pddl domains/alfworld/problems/*.pddl --plan
	veritraj check-trajectories $(E)
	veritraj verify $(E) --out $(OUT)/verified.jsonl
	veritraj show $(E) --traj heat-mug-01/b
	veritraj curate $(E) --verified $(OUT)/verified.jsonl --out $(OUT)/sft
	veritraj advantages $(E) --verified $(OUT)/verified.jsonl
	veritraj eval-labels $(E)
	veritraj acquire $(E) --drafts examples/acquisition/drafts --out $(OUT)/acquisition

bench:
	veritraj bench length
	veritraj bench state

clean:
	rm -rf runs .pytest_cache
