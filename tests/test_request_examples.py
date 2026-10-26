"""Requests carry worked examples, and the agent's tests check exactly those (never invented values);
requests without examples get smoke tests only."""
import ast
import random
import re

from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.data.agent.add_examples import convert
from vishalam.data.agent.families import BY_ID, make_instance
from vishalam.data.agent.trajectories import eval_task, synthesize

SB = SubprocessSandbox(timeout=10)


def _episode(fam_id, seed, kind="create", **kw):
    rng = random.Random(seed)
    while True:
        task = make_instance(BY_ID[fam_id], rng, "train")
        ep = synthesize(task, rng, kind, SB, "train", **kw)
        if ep is not None:
            return task, ep


def _test_file(ep):
    return next(s["call"]["args"]["content"] for s in ep["steps"]
                if s["call"]["name"] == "write_file" and s["call"]["args"]["path"].startswith("test_"))


def _asserted_values(test_src):
    return [ast.unparse(n.test.comparators[0]) for n in ast.walk(ast.parse(test_src))
            if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare)]


def test_examples_in_request_match_the_agents_tests():
    task, ep = _episode("group_by_length", 1, examples=True)
    assert ep["has_examples"]
    for c in task.agent_cases:
        assert repr(c.expected) in ep["request"]
    for value in _asserted_values(_test_file(ep)):
        assert value.replace(" ", "") in ep["request"].replace(" ", "") or value.startswith("pytest.approx")


def test_without_examples_the_agent_only_smoke_tests():
    task, ep = _episode("group_by_length", 2, examples=False)
    assert not ep["has_examples"]
    tests = _test_file(ep)
    assert "isinstance(" in tests and "==" not in tests
    for c in task.agent_cases:
        assert f"`{c.expected!r}`" not in ep["request"]
    assert ep["steps"][-1]["call"]["name"] == "done"


def test_fix_requests_always_have_examples():
    for seed in range(3):
        _, ep = _episode("count_pairs_sum", 10 + seed, kind="fix")  # group_by_length has no mutations
        assert ep["has_examples"]


def test_eval_tasks_get_held_out_example_phrasing():
    rng = random.Random(0)
    shown = 0
    for _ in range(40):
        t = eval_task(make_instance(BY_ID["group_by_length"], rng, "test"), rng, "create", "test_seen")
        shown += t["has_examples"]
        if t["has_examples"]:
            assert "Here is how it should work:" in t["request"]
    assert 25 <= shown < 40


def test_convert_keeps_steps_when_adding_examples_and_resynthesizes_otherwise():
    task, ep = _episode("group_by_length", 3, examples=True)
    old = {**ep, "request": task.request}  # an episode from the first data: no examples in the request
    old.pop("has_examples")
    kinds = set()
    for seed in range(30):
        new = convert(dict(old), random.Random(seed), SB)
        kinds.add(new["has_examples"])
        if new["has_examples"]:
            assert new["steps"] == old["steps"] and new["request"].startswith(task.request)
            assert re.search(r"(should return|->|>>>|==)", new["request"])
        else:
            assert new["request"] == task.request and "isinstance(" in _test_file(new)
            assert new["steps"][-1]["call"]["name"] == "done"
        if kinds == {True, False}:
            break
    assert kinds == {True, False}
