from types import SimpleNamespace
from veritraj.config import build_components
from veritraj.envs.base import EnvStep, TaskSpec
from veritraj.policy.generate import Generation
from veritraj.train.rollout import collect_group

def test_blocked_and_parse_failed_proposals_never_execute():
    class Env:
        task = TaskSpec("test")
        calls = 0
        seeds = []
        def reset(self, task_id, seed=0):
            self.seeds.append(seed)
            return "You are in the middle of a room."
        def applicable(self, command):
            return True
        def step(self, command):
            self.calls += 1
            return EnvStep("", 1., True)
    class Policy:
        def generate(self, prompt):
            return Generation("Action: take(mug_1, desk_1)", [1], [2])
    env = Env()
    group = collect_group(env, Policy(), build_components(), "test", group_size=2,
                          max_steps=2, seed=13)
    assert env.calls == 0
    assert env.seeds == [13, 13]
    assert all(record.latent_applicable is True for t in group.trajectories for record in t.records)
    assert all(label.v == 0 for rows in group.verified.values() for label in rows)
    class ParsePolicy:
        def generate(self, prompt):
            return Generation("No command here", [1], [2])
    group = collect_group(env, ParsePolicy(), build_components(), "test", group_size=1, max_steps=1)
    assert group.verified["group-0"][0].p == 1
    assert env.calls == 0
