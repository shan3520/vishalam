"""Plain HumanEval completion: stop rules, scoring through the sandbox, and generation with a tiny model."""
import torch

from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.eval.base_completion import complete, score, summarize, truncate
from vishalam.model.config import ModelConfig
from vishalam.model.model import Vishalam
from vishalam.tokenizer.bpe import Tokenizer

TASK = {"id": "t/0", "fname": "add", "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
        "hidden_code": "from {module} import *\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"}
SB = SubprocessSandbox(timeout=15)


def test_truncate_stops_at_the_first_top_level_marker():
    assert truncate("    return 1\n\ndef other():\n    pass") == ("    return 1\n", True)
    assert truncate("    return 1\n# note") == ("    return 1", True)
    assert truncate("    if x:\n        return 1\n") == ("    if x:\n        return 1\n", False)  # indented: kept


def test_score_passes_only_correct_completions():
    assert score(TASK, "    return a + b\n", SB)[0]
    assert not score(TASK, "    return a - b\n", SB)[0]
    assert not score(TASK, "    return (\n", SB)[0]  # syntax error


def test_complete_runs_with_a_small_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=384, d_model=32, n_layers=1, n_heads=2, ffn_dim=64, max_seq_len=64)
    model = Vishalam(cfg).eval()
    tok = Tokenizer([])
    text = complete(model, cfg, "cpu", tok, "def f():\n", max_new=8, chunk=3)
    assert isinstance(text, str) and not any(s in text for s in ("\ndef", "\nclass"))


def test_summary_reports_clean_tasks_separately():
    rows = [{"first_pass": 1, "family_overlap": True}, {"first_pass": None, "family_overlap": False},
            {"first_pass": 1, "family_overlap": False}, {"first_pass": 3, "family_overlap": False}]
    t = summarize(rows, (1, 5))
    assert t["n"] == 4 and t["pass_at_1"] == 0.5 and t["n_clean"] == 3 and abs(t["pass_at_1_clean"] - 1 / 3) < 1e-9
    assert t["pass_at_5"] == 0.75 and abs(t["pass_at_5_clean"] - 2 / 3) < 1e-9


def test_sampled_attempts_are_reproducible_and_differ_from_each_other():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=384, d_model=32, n_layers=1, n_heads=2, ffn_dim=64, max_seq_len=64)
    model = Vishalam(cfg).eval()
    tok = Tokenizer([])

    def run(seed):
        return complete(model, cfg, "cpu", tok, "def f():", max_new=12, chunk=4, temperature=1.0,
                        generator=torch.Generator().manual_seed(seed))

    assert run(1) == run(1) and len({run(i) for i in range(5)}) > 1
