"""Command spans exclude trailing model termination markers."""

import pytest


@pytest.mark.parametrize('marker', ['<|im_end|>', '<|eot_id|>', '</s>'])
def test_alfworld_command_followed_by_termination_marker(comps, marker):
    text = 'Action: open cabinet 1' + marker
    parsed = comps.interface.parse(text)
    assert parsed.ok and parsed.name == 'open' and parsed.args == ('cabinet_1',)
    assert text[slice(*parsed.char_span)] == 'open cabinet 1'
