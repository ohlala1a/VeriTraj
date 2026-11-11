"""Fact store: open-world semantics, functional and mutex rules, incremental hash."""

from veritraj.pddl import Atom
from veritraj.store import Delta, FactStore, StoreConfig

CFG = StoreConfig(functional={"at": (0,)}, mutex=[("open", "closed")])


def A(p, *a):
    return Atom(p, a)


def test_unknown_is_neither_true_nor_false():
    s = FactStore(CFG)
    assert s.value(A("open", "fridge_1")) is None
    s.apply(Delta(add=[A("closed", "fridge_1")]))
    assert s.value(A("closed", "fridge_1")) is True
    assert s.value(A("open", "fridge_1")) is False  # mutex partner becomes known false


def test_functional_predicate_replaces_old_value():
    s = FactStore(CFG)
    s.apply(Delta(add=[A("at", "agent", "room_1")]))
    s.apply(Delta(add=[A("at", "agent", "fridge_1")]))
    assert s.holds(A("at", "agent", "fridge_1"))
    assert s.known_false(A("at", "agent", "room_1"))
    assert s.lookup("at", "agent") == A("at", "agent", "fridge_1")


def test_delete_then_add():
    s = FactStore(CFG)
    s.apply(Delta(add=[A("p")]))
    s.apply(Delta(add=[A("p")], delete=[A("p")]))
    assert s.holds(A("p"))


def test_hash_is_order_independent_and_incremental():
    facts = [A("in", f"o_{i}", "r_1") for i in range(20)]
    s1, s2 = FactStore(CFG), FactStore(CFG)
    s1.apply(Delta(add=facts))
    for f in reversed(facts):
        s2.apply(Delta(add=[f]))
    assert s1.hash == s2.hash
    s2.apply(Delta(delete=[facts[3]]))
    s2.apply(Delta(add=[facts[3]]))
    assert s1.hash == s2.hash


def test_apply_reports_change():
    s = FactStore(CFG)
    assert s.apply(Delta(add=[A("p")])) is True
    assert s.apply(Delta(add=[A("p")])) is False
