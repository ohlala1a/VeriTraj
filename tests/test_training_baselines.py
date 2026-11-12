from veritraj.advantage import AdvantageConfig
from veritraj.records import Trajectory, LoggedRecord, VerifiedRecord
from veritraj.train.rollout import RolloutGroup
from veritraj.train.baselines import advantages

def test_shared_progress_signal_and_shielded_unsupported_task_advantage():
    group = RolloutGroup()
    group.trajectories = [Trajectory("one", "task", [LoggedRecord(0, "", "", env_reward=1)]),
                          Trajectory("two", "task", [LoggedRecord(0, "", "", env_reward=0)])]
    group.verified = {key: [VerifiedRecord(key, 0, (0, 1), (0, 1), "action", 1, r_prog=3),
                            VerifiedRecord(key, 1, (0, 1), (0, 1), "action", 0, r_prog=99)]
                      for key in ("one", "two")}
    cfg = AdvantageConfig(lambda_pot=.2)
    task = advantages(group, cfg, "grpo")
    for algorithm in ("grpo_progress", "shielded"):
        actual = advantages(group, cfg, algorithm)
        assert abs(actual["one"][0] - (task["one"][0]+.6)) < 1e-10
        assert actual["one"][1] == task["one"][1]
