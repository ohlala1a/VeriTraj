"""Shared online environment contract."""
from dataclasses import dataclass, field
from typing import Protocol
from ..pddl.model import Atom

@dataclass
class TaskSpec:
    task_id: str
    goal_text: str = ''
    problem: str = ''
    family: str = ''

@dataclass
class EnvStep:
    observation: str
    reward: float
    done: bool
    executed: str | None = None
    info: dict = field(default_factory=dict)

class AgentEnvironment(Protocol):
    task: TaskSpec
    def reset(self, task_id: str, seed: int = 0) -> str: ...
    def step(self, command: str) -> EnvStep: ...
    def applicable(self, command: str) -> bool | None: ...
    def latent_facts(self) -> set[Atom] | None: ...
    def close(self) -> None: ...
