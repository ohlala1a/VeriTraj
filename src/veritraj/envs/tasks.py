"""Explicit task selection for online environments."""
from .base import TaskSpec

class TaskEnvironment:
    def __init__(self, tasks):
        self.tasks = {t.task_id: t for t in (TaskSpec(**x) if isinstance(x, dict) else x for x in tasks)}
        self.task = TaskSpec('')

    def select_task(self, task_id):
        if task_id not in self.tasks:
            raise ValueError(f'Unknown task {task_id!r}')
        self.task = self.tasks[task_id]
