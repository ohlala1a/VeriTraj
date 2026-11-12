"""Replay-prefix adjudication for a resettable simulator."""

class CyberEnvironment:
    def __init__(self, environment, task_id, seed=0, state_restorer=None):
        self.environment = environment
        self.task_id = task_id
        self.seed = seed
        self.state_restorer = state_restorer
        self.prefixes = {}
        environment.reset(task_id, seed)
        facts = environment.latent_facts()
        if facts is None:
            raise ValueError('Adjudication requires an explicit trusted latent-fact provider')
        self.prefixes[frozenset(facts)] = []

    def _restore(self, state, objects):
        if self.state_restorer is not None:
            self.state_restorer(self.environment, state, objects)
            if frozenset(self.environment.latent_facts()) != state:
                raise RuntimeError('State restoration did not match the requested facts')
            return
        if state not in self.prefixes:
            raise ValueError('State has no recorded replay prefix')
        self.environment.reset(self.task_id, self.seed)
        for command in self.prefixes[state]:
            self.environment.step(command)
        if frozenset(self.environment.latent_facts()) != state:
            raise RuntimeError('Replay failed to restore the adjudication state')

    def applicable(self, state, name, args, objects):
        self._restore(state, objects)
        result = self.environment.applicable(f'{name}({",".join(args)})')
        if result is None:
            raise ValueError('Simulator did not expose applicability')
        return result

    def step(self, state, name, args, objects):
        self._restore(state, objects)
        command = f'{name}({",".join(args)})'
        if self.environment.applicable(command) is not True:
            return None
        result = self.environment.step(command)
        if result.executed is None:
            return None
        successor = frozenset(self.environment.latent_facts())
        if self.state_restorer is None:
            self.prefixes[successor] = [*self.prefixes[state], command]
        return successor
