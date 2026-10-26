"""MBPP records become well-formed training episodes (no network: a hand-made record)."""
import random

from vishalam.agent.format import Step
from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.data.agent.mbpp_episodes import SPLITS, episode_from_record

REC = {"task_id": 900, "text": "Write a function to find the maximum of two numbers.",
       "code": "def max_of_two(a, b):\r\n    return a if a > b else b",
       "test_list": ["assert max_of_two(1, 2) == 2", "assert max_of_two(5, 3) == 5"], "test_setup_code": ""}


def test_record_becomes_a_passing_episode():
    ep = episode_from_record(REC, SubprocessSandbox(timeout=20), random.Random(0))
    assert ep is not None
    names = [s["call"]["name"] for s in ep["steps"]]
    assert names[-2:] == ["run_tests", "done"] and "write_file" in names
    assert ("The function must be called `max_of_two`.\nYour code should pass these tests:\n"
            "assert max_of_two(1, 2) == 2") in ep["request"]
    assert " passed" in ep["steps"][-2]["result"]
    [Step.from_dict(s) for s in ep["steps"]]  # loads as SFT steps


def test_broken_reference_is_skipped_and_test_split_never_used():
    bad = dict(REC, code="def max_of_two(a, b):\n    return a")
    assert episode_from_record(bad, SubprocessSandbox(timeout=20), random.Random(0)) is None
    assert "test" not in SPLITS


def test_sft_row_split_mixes_evenly_and_uses_every_window_once():
    from vishalam.train.sft import split_row

    G, n = 32, 6
    rows = [split_row(g, G, n) for g in range(3 * G)]
    for s in range(3):
        step = rows[s * G:(s + 1) * G]
        assert sum(k == "mix" for k, _ in step) == n
        assert [i for k, i in step if k == "mix"] == list(range(s * n, (s + 1) * n))
    sft = [i for k, i in rows if k == "sft"]
    assert sft == list(range(3 * (G - n)))  # every SFT window once, in order
    assert [split_row(g, G, 0) for g in range(G)] == [("sft", g) for g in range(G)]  # no mix = as before
