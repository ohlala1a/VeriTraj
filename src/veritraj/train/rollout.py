"""Verify before execution, finalize progress after confirmed transitions."""
from dataclasses import dataclass, field
from ..records import Trajectory, LoggedRecord
from ..pipeline import parse_executed
from ..policy.prompts import policy_prompt

@dataclass
class RolloutGroup:
    trajectories: list = field(default_factory=list)
    verified: dict = field(default_factory=dict)
    generations: dict = field(default_factory=dict)

def collect_group(env, policy, components, task_id, group_size=8, seed=0, gate=True,
                  max_steps=50, group_id="group", window=6, policy_name="", checkpoint="",
                  environment="alfworld", split="train"):
    if group_size < 1 or max_steps < 1:
        raise ValueError("group_size and max_steps must be positive")
    out = RolloutGroup()
    for member in range(group_size):
        obs = env.reset(task_id, seed=seed)
        task = env.task
        traj = Trajectory(traj_id=f"{group_id}-{member}", task_id=task_id, records=[],
                          goal_text=task.goal_text, problem=task.problem, family=task.family,
                          group_id=group_id, gate=gate, policy=policy_name, checkpoint=checkpoint,
                          seed=seed, split=split, environment=environment)
        traj.success_observed = getattr(env, "success_available", True)
        worker = components.worker()
        worker.reset(traj.traj_id, components.goal(traj), obs)
        history, labels, generations = [], [], []
        for t in range(max_steps):
            prompt = policy_prompt(task.goal_text, history, obs, window, environment)
            generation = policy.generate(prompt)
            result = worker.verify(generation.text)
            command = result.parsed.canonical() if result.parsed.ok else None
            truth = env.applicable(command) if command else None
            executed, reward, feedback, done = None, 0.0, "", False
            next_obs = obs
            if command and (not gate or result.record.v == 1):
                transition = env.step(command)
                next_obs, reward, done = transition.observation, transition.reward, transition.done
                executed = transition.executed
                feedback = str(transition.info.get("feedback", ""))
                traj.success_observed = bool(transition.info.get("success_available", True))
                if traj.success_observed:
                    traj.success = bool(transition.info.get("success", transition.info.get("won", False)))
            else:
                feedback = "Proposal blocked by verifier" if command else "Cannot parse action"
            result.record.executed = executed
            worker.maintain(parse_executed(components.domain, executed), next_obs)
            traj.records.append(LoggedRecord(t, obs, generation.text, executed, feedback,
                                             float(reward), truth, prompt))
            labels.append(result.record)
            generations.append(generation)
            history.append((obs, generation.text))
            traj.env_return += float(reward)
            obs = next_obs
            if done:
                break
        traj.final_observation = obs
        out.trajectories.append(traj)
        out.verified[traj.traj_id] = labels
        out.generations[traj.traj_id] = generations
    return out
