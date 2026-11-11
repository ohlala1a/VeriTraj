"""PDDL parsing, validation and grounding."""

import pytest

from veritraj.pddl import (Atom, PDDLError, check_domain, check_problem, ground_action, load_problem,
                           parse_atom_text, parse_domain)
from conftest import ROOT


def test_domain_is_clean(comps):
    assert check_domain(comps.domain) == []
    assert set(comps.domain.actions) >= {"goto", "open", "close", "take", "put", "heat", "examine"}
    assert comps.domain.is_subtype("receptacle", "place")


def test_problems_are_clean(comps):
    for p in sorted((ROOT / "domains/alfworld/problems").glob("*.pddl")):
        assert check_problem(comps.domain, load_problem(p)) == [], p.name


@pytest.mark.parametrize("bad, msg", [
    ("(define (domain d) (:requirements :strips) (:predicates (p)) (:action a :parameters () "
     ":precondition (or (p) (p)) :effect (p)))", "unsupported 'or'"),
    ("(define (domain d) (:predicates (p))", "unbalanced"),
    ("(define (domain d) (:requirements :fluents))", "unsupported requirements"),
])
def test_parse_errors(bad, msg):
    with pytest.raises(PDDLError, match=msg):
        parse_domain(bad)


def test_static_check_reports_undeclared_predicate():
    d = parse_domain("(define (domain d) (:requirements :strips :typing) (:types x) (:predicates (p ?a - x))"
                     "(:action a :parameters (?a - x) :precondition (and (q ?a)) :effect (and (p ?a))))")
    assert any("undeclared predicate 'q'" in i for i in check_domain(d))


def test_ground_action_reports_type_issues(comps):
    types = {"apple_1": "item", "cabinet_1": "receptacle"}
    ga, issues = ground_action(comps.domain, "put", ("cabinet_1", "apple_1"), types)
    assert ga is not None
    assert [str(i.as_atom()) for i in issues] == ["item(cabinet_1)", "receptacle(apple_1)"]


def test_parse_atom_text():
    assert parse_atom_text("open(microwave_1)") == Atom("open", ("microwave_1",))
    assert parse_atom_text("(at agent cabinet_1)") == Atom("at", ("agent", "cabinet_1"))
    assert parse_atom_text("handempty") == Atom("handempty", ())
