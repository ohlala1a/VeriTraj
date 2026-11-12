"""Executable PDDL environment for deterministic integration runs."""
import json
from .base import EnvStep
from .tasks import TaskEnvironment
from .actions import parse_command
from ..config import resolve
from ..pddl import load_problem
from ..pddl.ground import ground_action, applicable, progress

class PddlEnvironment(TaskEnvironment):
    def __init__(self, components, tasks):
        super().__init__(tasks)
        self.components = components
        self.state = set()
        self.done = False

    def reset(self, task_id, seed=0):
        self.select_task(task_id)
        self.problem = load_problem(resolve(self.components.cfg, self.task.problem))
        self.state = set(self.problem.init)
        self.done = self._won()
        return self._observe()

    def _action(self, command):
        name, args = parse_command(command)
        action, issues = ground_action(self.components.domain, name, args,
                                       {**self.components.domain.constants, **self.problem.objects})
        return None if issues else action

    def applicable(self, command):
        action = self._action(command)
        return action is not None and applicable(action, self.state)

    def _won(self):
        return all((literal.atom in self.state) == literal.positive for literal in self.problem.goal)

    def step(self, command):
        if self.done:
            return EnvStep(self._observe(), 0.0, True)
        action = self._action(command)
        valid = action is not None and applicable(action, self.state)
        was_done = self.done
        if valid:
            self.state = progress(action, self.state)
        self.done = self._won()
        return EnvStep(self._observe(), float(self.done and not was_done), self.done,
                       command if valid else None, {'won': self.done})

    def _observe(self):
        if self.components.domain.name == 'veritraj-cyberbattle':
            return json.dumps({'facts': [{'pred': f.pred, 'args': f.args} for f in sorted(self.state, key=str)],
                               'types': self.problem.objects})
        location = next(f.args[1] for f in self.state if f.pred == 'at')
        places = sorted(f.args[0].replace('_', ' ') for f in self.state if f.pred == 'visible')
        sentences = ['You are in the middle of a room.', 'Looking quickly around you, you see ' + ', '.join(places) + '.']
        if location != 'room_1':
            native = location.replace('_', ' ')
            sentences.append(f'You arrive at the {native}.')
            for predicate in ('open', 'closed'):
                if any(f.pred == predicate and f.args == (location,) for f in self.state):
                    sentences.append(f'The {native} is {predicate}.')
            if any(f.pred == 'open' and f.args == (location,) for f in self.state):
                items = sorted(f.args[0].replace('_', ' ') for f in self.state if f.pred == 'in' and f.args[1] == location)
                sentences.append(f'On the {native}, you see ' + (', '.join(items) or 'nothing') + '.')
        return ' '.join(sentences)

    def latent_facts(self):
        return set(self.state)

    def close(self):
        pass
