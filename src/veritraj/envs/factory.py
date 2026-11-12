"""Environment construction from explicit task specifications."""
def make_environment(spec, components):
    kind = spec.get('kind', 'pddl')
    tasks = spec.get('tasks', [])
    if kind == 'pddl':
        from .pddl_env import PddlEnvironment
        return PddlEnvironment(components, tasks)
    if kind == 'alfworld':
        from .alfworld_env import AlfworldEnvironment
        return AlfworldEnvironment(tasks, env=spec.get('env'),
                                   max_episode_steps=spec.get('max_episode_steps', 200),
                                   game_files=spec.get('game_files'))
    if kind == 'cyberbattle':
        from .cyberbattle_env import CyberBattleEnvironment
        return CyberBattleEnvironment(tasks, env=spec.get('env'), env_id=spec.get('env_id'),
                                      applicability_provider=spec.get('applicability_provider'),
                                      success_provider=spec.get('success_provider'))
    raise ValueError(f'Unknown environment kind {kind!r}')
