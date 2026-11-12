"""Topology-to-problem conversion for abstract local/remote/connect tasks."""
from .pddl.model import Atom, Literal, Problem

def problem_from_topology(name, topology):
    objects = dict(topology['types'])
    facts = {Atom(f['pred'], tuple(f.get('args', []))) for f in topology.get('facts', [])}
    goal = [Literal(Atom('owned', (node,))) for node in topology['goal_nodes']]
    return Problem(name, 'veritraj-cyberbattle', objects, facts, goal)

def problem_to_pddl(problem):
    objects = '\n'.join(f'    {name} - {kind}' for name, kind in sorted(problem.objects.items()))
    facts = '\n'.join(f'    {fact.to_pddl()}' for fact in sorted(problem.init, key=str))
    goal = ' '.join(lit.to_pddl() for lit in problem.goal)
    return f'(define (problem {problem.name})\n (:domain {problem.domain_name})\n (:objects\n{objects})\n (:init\n{facts})\n (:goal (and {goal})))\n'
