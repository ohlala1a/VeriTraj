"""Environment contracts and public-observation integration."""
import json
import pytest
from veritraj.config import build_components, REPO_ROOT
from veritraj.envs import make_environment
from veritraj.envs.alfworld_env import AlfworldEnvironment
from veritraj.adapters.cyberbattle import CyberBattleAdapter
from veritraj.store import FactStore
from veritraj.pddl import Atom, check_domain, check_problem, load_problem
from veritraj.planner import gbfs_plan
from veritraj.records import LoggedRecord, Trajectory


def cyber_components():
    return build_components(REPO_ROOT / 'configs/cyberbattle.yaml')


def test_cyber_domain_solvable_and_replay_equivalence():
    components = cyber_components()
    path = 'domains/cyberbattle/problems/abstract-connect.pddl'
    problem = load_problem(REPO_ROOT / path)
    assert not check_domain(components.domain)
    assert not check_problem(components.domain, problem)
    plan = gbfs_plan(components.domain, problem)
    assert plan and len(plan) == 1
    env = make_environment({'kind': 'pddl', 'tasks': [{'task_id': 'one', 'problem': path}]}, components)
    obs = env.reset('one')
    command = 'connect(node_0,node_1,service_0,credential_0)'
    assert env.applicable(command)
    result = env.step(command)
    assert result.done and result.reward == 1 and result.executed == command
    traj = Trajectory('one', 'one', [LoggedRecord(0, obs, command, command)], result.observation)
    inc = components.worker().process(traj)
    rec = components.recompute_worker().process(traj)
    assert [row.to_dict() for row in inc] == [row.to_dict() for row in rec]
    assert inc[0].v == 1
    assert not env.step(command).executed


def test_adapter_ignores_hidden_fields_and_explicit_deletes():
    store = FactStore()
    adapter = CyberBattleAdapter()
    hidden = {'latent_facts': [{'pred': 'owned', 'args': ['secret']}],
              'facts': [{'pred': 'discovered', 'args': ['node_0']}], 'types': {'node_0': 'node'}}
    store.apply(adapter.ground(json.dumps(hidden), store, True))
    assert Atom('owned', ('secret',)) not in store.true
    store.apply(adapter.ground(json.dumps({'delete': hidden['facts']}), store))
    assert Atom('discovered', ('node_0',)) not in store.true


def test_native_cyber_bad_arguments_are_rejected_without_stepping():
    pytest.importorskip('numpy')
    from veritraj.envs.cyberbattle_env import CyberBattleEnvironment

    class Native:
        def reset(self, seed=0):
            return {}, {}
        def step(self, action):
            raise AssertionError('Invalid proposals must not reach the native environment')

    env = CyberBattleEnvironment([{'task_id': 'one'}], env=Native())
    env.reset('one')
    assert env.applicable('local(server,capability_0)') is False
    transition = env.step('local(server,capability_0)')
    assert transition.executed is None and transition.reward == 0


def test_cyber_acquisition_needs_domain_specific_acceptance():
    from veritraj.ops import acquisition_setup

    with pytest.raises(ValueError, match='acceptance_problems'):
        acquisition_setup(cyber_components(), [])


def test_counterfactual_state_restoration_does_not_require_a_replay_prefix():
    from types import SimpleNamespace
    from veritraj.acquisition.cyber_env import CyberEnvironment

    class Native:
        def reset(self, task, seed):
            self.state = {Atom('owned', ('node_0',))}
        def latent_facts(self):
            return self.state
        def applicable(self, command):
            return True
        def step(self, command):
            self.state.add(Atom('owned', ('node_1',)))
            return SimpleNamespace(executed=command)

    native = Native()
    def restore(environment, state, objects):
        environment.state = set(state)
    adjudication = CyberEnvironment(native, 'one', state_restorer=restore)
    counterfactual = frozenset({Atom('owned', ('node_2',))})
    successor = adjudication.step(counterfactual, 'local', ('node_2', 'capability_0'), {})
    assert Atom('owned', ('node_1',)) in successor


class NativeHousehold:
    def __init__(self):
        self.commands = []
    def seed(self, seed):
        self.seed_value = seed
    def reset(self):
        return {'feedback': 'room', 'admissible_commands': ['go to cabinet 1']}
    def step(self, command):
        self.commands.append(command)
        return {'feedback': 'You arrive at cabinet 1.', 'admissible_commands': []}, 0, False
    def close(self):
        pass


def test_injected_household_mapping_and_reset():
    native = NativeHousehold()
    env = AlfworldEnvironment([{'task_id': 'one'}], native)
    assert env.reset('one', 7) == 'room'
    assert native.seed_value == 7
    assert env.applicable('goto(room_1,cabinet_1)')
    assert env.step('goto(room_1,cabinet_1)').executed == 'goto(room_1,cabinet_1)'
    assert native.commands == ['go to cabinet 1']
    assert env.latent_facts() is None
    with pytest.raises(ValueError):
        env.reset('missing')

class NativeCyber:
    def reset(self, seed=0):
        import numpy as np
        self.seed_value = seed
        return {'discovered_node_count': 2, 'nodes_privilegelevel': [1, 0],
                'action_mask': {'remote_vulnerability': np.ones((2, 2, 1), dtype=int)},
                'private_network': 'never serialize'}, {}
    def step(self, command):
        self.command = command
        obs, _ = self.reset()
        obs['nodes_privilegelevel'] = [1, 1]
        return obs, 1.0, True, False, {'credential_cache': ['never serialize']}
    def close(self):
        pass


def test_native_cyber_public_observation_mapping_and_receipt():
    pytest.importorskip('numpy')
    from veritraj.envs.cyberbattle_env import CyberBattleEnvironment
    native = NativeCyber()
    env = CyberBattleEnvironment([{'task_id': 'one'}], env=native)
    observation = env.reset('one', 9)
    assert native.seed_value == 9
    assert 'never serialize' not in observation
    command = 'remote(node_0,node_1,capability_0)'
    assert env.applicable(command) is None
    result = env.step(command)
    assert native.command['remote_vulnerability'].tolist() == [0, 1, 0]
    assert result.executed == command
    assert result.done
    assert 'never serialize' not in result.observation
    assert 'credential_cache' not in result.info
    env.reset('one')
    assert env.latent_facts() is None
    with pytest.raises(ValueError):
        env.native_command('remote(credential_0,node_1,capability_0)')

class NativeBatchHousehold:
    def seed(self, seed):
        self.seed_value = seed
    def reset(self):
        return ['room'], {'admissible_commands': [['go to cabinet 1', 'use desklamp 1']],
                          'won': [False], 'facts': [['hidden fact']]}
    def step(self, commands):
        self.commands = commands
        return ['You arrive at cabinet 1.'], [0.0], [False], {
            'admissible_commands': [['move mug 1 to cabinet 1']], 'won': [False]}
    def close(self):
        pass


def test_official_batched_alfworld_contract_and_mapping():
    native = NativeBatchHousehold()
    env = AlfworldEnvironment([{'task_id': 'one', 'family': 'examine'}], native)
    assert env.reset('one', 23) == 'room'
    assert native.seed_value == 23
    assert env.applicable('examine(book_1,desklamp_1,cabinet_1)') is None
    result = env.step('goto(room_1,cabinet_1)')
    assert native.commands == ['go to cabinet 1']
    assert result.executed == 'goto(room_1,cabinet_1)'
    assert not result.done and result.reward == 0.0
    assert 'facts' not in result.info
    assert env.native_command('put(mug_1,cabinet_1)') == 'move mug 1 to cabinet 1'
    assert env.applicable('put(mug_1,cabinet_1)') is True
    env.info = {}
    assert env.applicable('put(mug_1,cabinet_1)') is None
    env.info = {'admissible_commands': []}
    assert env.applicable('put(mug_1,cabinet_1)') is False


def test_cyber_eligibility_truth_and_success_receipts_without_optional_runtime():
    from veritraj.envs.cyberbattle_env import CyberBattleEnvironment
    class InstrumentedCyber(CyberBattleEnvironment):
        def native_command(self, command):
            return {'local_vulnerability': [0, 0]}
        def _eligible(self, command):
            return self.eligibility
        def public_observation(self):
            return '{}'
    class Simulator:
        def step(self, action):
            return {}, 0, True, False, self.receipt
    simulator = Simulator()
    simulator.receipt = {}
    env = InstrumentedCyber([{'task_id': 'one'}], env=simulator)
    env.eligibility = True
    assert env.applicable('local(node_0,capability_0)') is None
    result = env.step('local(node_0,capability_0)')
    assert result.done
    assert result.info['success_available'] is False
    assert 'success' not in result.info
    env.applicability_provider = lambda native: True
    assert env.applicable('local(node_0,capability_0)') is True
    env.applicability_provider = lambda native: False
    assert env.applicable('local(node_0,capability_0)') is False
    env.eligibility = False
    env.applicability_provider = lambda native: True
    assert env.applicable('local(node_0,capability_0)') is False
    env.eligibility = True
    simulator.receipt = {'success': True}
    assert env.step('local(node_0,capability_0)').info['success'] is True
    simulator.receipt = {'won': False}
    assert env.step('local(node_0,capability_0)').info['success'] is False
    simulator.receipt = {}
    env.success_provider = lambda native: True
    assert env.step('local(node_0,capability_0)').info['success'] is True
