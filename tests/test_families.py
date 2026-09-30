import random
import re

import pytest

from vishalam.data.agent.families import ALL, HELD_OUT, BY_ID, make_instance
from vishalam.data.agent.families.base import compile_function


@pytest.mark.parametrize("fam", ALL, ids=lambda f: f.id)
def test_family_instances(fam):
    rng = random.Random(hash(fam.id) & 0xFFFF)
    for split in ("train", "test"):
        for _ in range(6):
            t = make_instance(fam, rng, desc_split=split)
            assert t.fname.isidentifier(), t.fname
            assert f"`{t.fname}" in t.request, t.request
            # no unfilled placeholders like {p0} or {parity}
            assert not re.search(r"\{[a-z_0-9]+\}", t.request), t.request
            assert len(t.agent_cases) >= 2, (fam.id, t.agent_cases)
            assert len(t.hidden_cases) >= 3, (fam.id, len(t.hidden_cases))
            fn = compile_function(t.ref_source, t.fname)
            for c in t.agent_cases + t.hidden_cases:
                assert fn(*c.args) == c.expected
            for p in t.params:
                assert p.isidentifier()


@pytest.mark.parametrize("fam", ALL, ids=lambda f: f.id)
def test_every_param_choice_has_a_safe_option(fam):
    from vishalam.data.agent.families.base import forbidden_names

    bad = forbidden_names(fam)
    for choices in fam.params:
        assert [c for c in choices if c not in bad], (fam.id, choices, bad)


def test_held_out_ids_exist():
    assert HELD_OUT <= set(BY_ID)
    assert 80 <= len(ALL)
