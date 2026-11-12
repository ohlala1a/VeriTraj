"""Official ALFWorld TextWorld batch-of-one integration."""
from pathlib import Path
from .base import EnvStep
from .tasks import TaskEnvironment
from .actions import parse_command

class AlfworldEnvironment(TaskEnvironment):
    def __init__(self, tasks, env=None, max_episode_steps=200, game_files=None):
        super().__init__(tasks)
        self.env = env
        self.injected = env is not None
        self.max_episode_steps = max_episode_steps
        self.game_files = dict(game_files or {})
        self.info = {}
        self.batched = False

    def _create_native(self, task_id):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
        game_file = self.game_files.get(task_id, task_id)
        if not Path(game_file).is_file():
            raise FileNotFoundError(f'ALFWorld task requires a game.tw-pddl file: {game_file}')
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True,
                                           extras=['gamefile'])
        env_id = textworld.gym.register_games(
            [game_file], request_infos, batch_size=1, asynchronous=False,
            max_episode_steps=self.max_episode_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.env = textworld.gym.make(env_id)

    def _receive(self, observation, info):
        self.batched = isinstance(observation, (tuple, list))
        if self.batched:
            if len(observation) != 1:
                raise ValueError('ALFWorld wrapper requires batch_size=1')
            self.info = {key: value[0] if isinstance(value, (tuple, list)) else value
                         for key, value in info.items()}
            return str(observation[0])
        self.info = info
        return str(observation)

    def reset(self, task_id, seed=0):
        self.select_task(task_id)
        if not self.injected:
            self.close()
            self._create_native(task_id)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
            return self._receive(observation, info)
        self.info = result if isinstance(result, dict) else vars(result)
        self.batched = False
        return str(self.info.get('feedback', result))

    def native_command(self, command):
        name, args = parse_command(command)
        a = [x.replace('_', ' ') for x in args]
        if name == 'goto' and len(a) == 2:
            return f'go to {a[1]}'
        if name in ('open', 'close') and len(a) == 1:
            return f'{name} {a[0]}'
        if name == 'put' and len(a) == 2:
            return f'move {a[0]} to {a[1]}'
        if name in ('take', 'heat', 'cool', 'clean') and len(a) == 2:
            link = 'from' if name == 'take' else 'with'
            return f'{name} {a[0]} {link} {a[1]}'
        if name == 'toggle' and len(a) == 2:
            return f'use {a[0]}'
        if name == 'examine' and len(a) == 3:
            return f'use {a[1]}'
        if name in ('look', 'inventory') and not a:
            return name
        raise ValueError(f'Unsupported ALFWorld action {command!r}')

    def applicable(self, command):
        commands = self.info.get('admissible_commands')
        native = self.native_command(command)
        if commands is None:
            return None
        if native not in commands:
            return False
        if parse_command(command)[0] == 'examine':
            return None
        return True

    def step(self, command):
        valid = self.applicable(command)
        native = self.native_command(command)
        result = self.env.step([native] if self.batched else native)
        if len(result) == 4:
            observation, reward, done, info = result
            obs = self._receive(observation, info)
            reward = reward[0] if isinstance(reward, (tuple, list)) else reward
            done = done[0] if isinstance(done, (tuple, list)) else done
        else:
            state, reward, done = result
            self.info = state if isinstance(state, dict) else vars(state)
            obs = str(self.info.get('feedback', state))
        won = bool(self.info.get('won', False))
        confirmed = valid is True and 'nothing happens' not in obs.lower()
        if parse_command(command)[0] == 'examine':
            confirmed = valid is not False and self.task.family == 'examine' and won
        return EnvStep(obs, float(reward), bool(done), command if confirmed else None,
                       {'won': won})

    def latent_facts(self):
        return None

    def close(self):
        if self.env is not None:
            self.env.close()
