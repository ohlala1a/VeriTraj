"""Action interface: command location, implicit arguments, spans."""

from veritraj.interface import UNKNOWN_ARG, canonical_object
from veritraj.pddl import Atom
from veritraj.store import Delta, FactStore
from veritraj.tokenization import RegexTokenizer, char_to_token_span


def store_at(comps, place):
    s = FactStore(comps.store_config)
    s.apply(Delta(add=[Atom("at", ("agent", place))]))
    return s


def test_alfworld_command(comps):
    text = "Think: put it away.\nAction: put mug 1 in/on cabinet 1"
    p = comps.interface.parse(text)
    assert p.ok and p.name == "put" and p.args == ("mug_1", "cabinet_1")
    assert text[p.char_span[0]:p.char_span[1]] == "put mug 1 in/on cabinet 1"


def test_functional_command_and_synonym(comps):
    p = comps.interface.parse("Action: PUT_IN(mug_1, microwave_1)")
    assert p.ok and p.name == "put" and p.args == ("mug_1", "microwave_1")


def test_implicit_location_from_store(comps):
    p = comps.interface.parse("Action: go to fridge 1", store_at(comps, "countertop_1"))
    assert p.args == ("countertop_1", "fridge_1")
    p = comps.interface.parse("Action: use desklamp 1", store_at(comps, "sidetable_1"))
    assert p.name == "toggle" and p.args == ("desklamp_1", "sidetable_1")


def test_implicit_location_unknown(comps):
    p = comps.interface.parse("Action: go to fridge 1", FactStore(comps.store_config))
    assert p.args == (UNKNOWN_ARG, "fridge_1")


def test_parse_failure_covers_whole_generation(comps):
    text = "Think: hmm.\nAction: I will wash it now."
    p = comps.interface.parse(text)
    assert not p.ok and p.char_span == (0, len(text))


def test_last_action_tag_wins(comps):
    p = comps.interface.parse("Action: open fridge 1\nWait.\nAction: open microwave 1")
    assert p.args == ("microwave_1",)


def test_canonical_object_aliases():
    aliases = {"desk lamp": "desklamp", "refrigerator": "fridge"}
    assert canonical_object("Desk Lamp 1", aliases) == "desklamp_1"
    assert canonical_object("refrigerator 2", aliases) == "fridge_2"
    assert canonical_object("mug_3") == "mug_3"


def test_token_span():
    text = "Think: ok.\nAction: open fridge 1"
    offs = RegexTokenizer().offsets(text)
    s = text.index("open")
    u, w = char_to_token_span(offs, (s, len(text)))
    assert [text[a:b] for a, b in offs[u:w]] == ["open", "fridge", "1"]
