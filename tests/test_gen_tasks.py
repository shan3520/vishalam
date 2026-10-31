"""Generated tasks: only cross-checked, decontaminated, safe tasks become episodes."""
import json
import random
import subprocess
import sys

from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.data.agent.gen_episodes import HELD_OUT_KEYWORDS, rename_function, split_cases, verify
from vishalam.data.agent.gen_tasks import parse_task
from vishalam.data.agent.trajectories import synthesize

SB = SubprocessSandbox(timeout=10)
GOOD = {"fname": "count_long", "request": "Write `count_long(words, n)` that counts the words longer than `n`.",
        "solution": "def count_long(words, n):\n    return sum(len(w) > n for w in words)\n",
        "solution2": "def count_long(words, n):\n    c = 0\n    for w in words:\n        if len(w) > n:\n            c += 1\n    return c\n",
        "inputs_src": "[(['a', 'bb', 'ccc'], 1), (['hello'], 2), ([], 3), (['x', 'yy'], 0), "
                      "(['abcd', 'ab', 'abc', 'a'], 2), (['zz', 'zzz', 'zzzz'], 3), (['q', 'q'], 0)]"}


def test_parse_task_reads_the_reply_format():
    reply = ("FUNCTION: count_long\nREQUEST: Write `count_long(words, n)` that counts\nlong words.\nSOLUTION:\n"
             "```python\n" + GOOD["solution"] + "```\nINPUTS:\n```python\n" + GOOD["inputs_src"] + "\n```")
    t = parse_task(reply)
    assert t["fname"] == "count_long" and t["request"].endswith("counts long words.")
    assert t["solution"] == GOOD["solution"] and t["inputs_src"] == GOOD["inputs_src"]


def test_agreeing_solutions_make_a_task():
    task, cases = verify(GOOD, SB)
    assert task.fname == "count_long" and task.params == ["words", "n"] and len(cases) == 7
    agent, hidden = split_cases(cases, random.Random(0))
    assert len(agent) >= 3 and len(hidden) >= 3 and not set(map(repr, agent)) & set(map(repr, hidden))


def test_disagreement_unsafe_code_and_hangs_are_rejected():
    off_by_one = GOOD["solution2"].replace("> n", ">= n")
    assert verify({**GOOD, "solution2": off_by_one}, SB) is None
    assert verify({**GOOD, "solution": "import os\n" + GOOD["solution"]}, SB) is None
    hang = "def count_long(words, n):\n    while True:\n        pass\n"
    assert verify({**GOOD, "solution": hang, "solution2": hang}, SB) is None
    constant = "def count_long(words, n):\n    return 1\n"
    assert verify({**GOOD, "solution": constant, "solution2": constant}, SB) is None


def test_held_out_concepts_are_recognised():
    for text in ["convert to roman numerals", "find the median score", "compare version strings", "peaks"]:
        assert HELD_OUT_KEYWORDS.search(text)
    for text in ["unit conversion", "leaderboard positions", "speaker volume", "database rows"]:
        assert not HELD_OUT_KEYWORDS.search(text)


def test_wrong_approach_episode_rewrites_the_file():
    task, cases = verify(GOOD, SB)
    task.agent_cases, task.hidden_cases = split_cases(cases, random.Random(1))
    wrong = rename_function("def other(xs):\n    return len(xs)\n", "other", "count_long")
    ep = synthesize(task, random.Random(2), "create", SB, "train", examples=True, wrong_first=wrong)
    writes = [s["call"]["args"]["content"] for s in ep["steps"]
              if s["call"]["name"] == "write_file" and s["call"]["args"]["path"] == "count_long.py"]
    assert writes == [wrong, task.ref_source]
    assert ep["steps"][-1]["call"]["name"] == "done"


def test_pipeline_end_to_end(tmp_path):
    cands = tmp_path / "c.jsonl"
    rows = [GOOD, {**GOOD, "fname": "roman_value", "request": "Convert a roman numeral."},
            {**GOOD, "solution2": GOOD["solution2"].replace("> n", ">= n")}]
    cands.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fams = tmp_path / "f.jsonl"
    fams.write_text("".join(json.dumps({"family_id": f"f{i % 2}", "i": i}) + "\n" for i in range(6)))
    out = tmp_path / "out.jsonl"
    subprocess.run([sys.executable, "-m", "vishalam.data.agent.gen_episodes", "--cands", str(cands),
                    "--families", str(fams), "--per-family", "2", "--out", str(out), "--workers", "1",
                    "--no-benchmarks"], check=True)
    eps = [json.loads(l) for l in open(out)]
    stats = json.loads((tmp_path / "gen_stats.json").read_text())
    assert stats["tasks"] == 1 and stats["family episodes"] == 4
    gen = [e for e in eps if e.get("request_source") == "ai-generated"]
    assert gen and all(e["fname"] == "count_long" for e in gen)
