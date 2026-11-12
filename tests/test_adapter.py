"""Observation adapter: templates, listings, recovery policies."""

from veritraj.adapters import build_adapter
from veritraj.pddl import Atom
from veritraj.store import FactStore

INIT = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, "
        "a microwave 1, and a sinkbasin 1.\n\nYour task is to: put a hot mug in cabinet 1.")


def grounded(comps, policy, *observations):
    ad = build_adapter(comps.cfg["adapter"], policy=policy)
    s = FactStore(comps.store_config)
    for i, o in enumerate(observations):
        s.apply(ad.ground(o, s, first=(i == 0)))
    return s, ad


def test_initial_observation_types_receptacles(comps):
    s, _ = grounded(comps, "conservative", INIT)
    assert s.types["microwave_1"] == "receptacle"
    assert s.holds(Atom("openable", ("microwave_1",)))
    assert s.holds(Atom("can-heat", ("microwave_1",)))
    assert s.holds(Atom("open", ("countertop_1",)))
    assert s.value(Atom("open", ("microwave_1",))) is None
    assert s.holds(Atom("at", ("agent", "room_1"))) and s.holds(Atom("handempty", ()))


def test_arrival_and_listing(comps):
    s, _ = grounded(comps, "conservative", INIT, "You arrive at countertop 1. On the countertop 1, you see a mug 1, and a desk lamp 1.")
    assert s.holds(Atom("at", ("agent", "countertop_1")))
    assert s.holds(Atom("in", ("mug_1", "countertop_1")))
    assert s.holds(Atom("lamp-at", ("desklamp_1", "countertop_1")))


def test_listing_is_complete_for_receptacle(comps):
    s, _ = grounded(comps, "conservative", INIT,
                    "You arrive at countertop 1. On the countertop 1, you see a mug 1.",
                    "You arrive at countertop 1. On the countertop 1, you see nothing.")
    assert s.known_false(Atom("in", ("mug_1", "countertop_1")))


def test_closed_state(comps):
    s, _ = grounded(comps, "conservative", INIT, "You arrive at microwave 1. The microwave 1 is closed.")
    assert s.holds(Atom("closed", ("microwave_1",)))
    assert s.known_false(Atom("open", ("microwave_1",)))


def test_conservative_leaves_unresolved_unknown(comps):
    obs = "You arrive at countertop 1. On the countertop 1, you see a mugs 1."
    s, ad = grounded(comps, "conservative", INIT, obs)
    assert "mugs_1" not in s.types and "mug_1" not in s.types
    s2, _ = grounded(comps, "rule_norm", INIT, obs)
    assert s2.holds(Atom("in", ("mug_1", "countertop_1")))


def test_paraphrase_only_with_rule_norm(comps):
    obs = "You reach microwave 1. The microwave 1 is shut."
    s, _ = grounded(comps, "conservative", INIT, obs)
    assert s.value(Atom("closed", ("microwave_1",))) is None
    s2, _ = grounded(comps, "rule_norm", INIT, obs)
    assert s2.holds(Atom("closed", ("microwave_1",)))


def test_llm_impute_uses_resolver(comps):
    ad = build_adapter(comps.cfg["adapter"], policy="llm_impute", resolver=lambda m, cls: "fridge" if m == "icebox" else None)
    s = FactStore(comps.store_config)
    s.apply(ad.ground("You arrive at icebox 1. The icebox 1 is closed.", s))
    assert s.holds(Atom("closed", ("fridge_1",)))
