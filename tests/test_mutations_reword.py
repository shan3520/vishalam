import random

from vishalam.data.agent.families import ALL, make_instance
from vishalam.data.agent.mutations import fix_edit, mutate
from vishalam.data.agent.reword_qwen import valid


def test_mutations_valid_and_fixable():
    rng = random.Random(0)
    made = 0
    for fam in ALL:
        t = make_instance(fam, rng)
        m = mutate(t.ref_source, rng)
        if m is None:
            continue
        made += 1
        compile(m, "m", "exec")
        assert m != t.ref_source
        old, new = fix_edit(m, t.ref_source)
        assert m.count(old) == 1
        assert m.replace(old, new, 1) == t.ref_source
    assert made > 0.9 * len(ALL)


def test_reword_validation():
    orig = "Write a function `count_even(nums)` that counts the even numbers in `nums`. Return 0 if there are none."
    assert valid(orig, "I need `count_even(nums)`: it should tell me how many values in `nums` are even, or 0 if there are none.")
    assert not valid(orig, "Write `count_evens(nums)` counting even numbers in `nums`, 0 if none.")  # renamed function
    assert not valid(orig, "Count the even numbers in `nums` with `count_even(nums)`, returning -1 if none.")  # number changed
    assert not valid(orig, "```python\ndef count_even(nums): ...```")
