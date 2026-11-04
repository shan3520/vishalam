import pytest

from vishalam.agent.format import ParseError, Step, ToolCall, encode_episode, parse_call, training_example
from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.agent.tools import Workspace
from vishalam.tokenizer.bpe import NUM_SPECIAL_SLOTS, Tokenizer
from vishalam.tokenizer.bpe_train import count_chunks, train_bpe


@pytest.fixture(scope="module")
def tok():
    texts = ["def f(x):\n    return x + 1\n", "write a function that returns the sum"] * 20
    return Tokenizer(train_bpe(count_chunks(texts), 256 + NUM_SPECIAL_SLOTS + 100, 1, verbose=False))


EPISODE = [
    Step(ToolCall("write_file", {"path": "test_f.py", "content": "from f import f\n\ndef test_f():\n    assert f(1) == 2\n"}), "wrote test_f.py (4 lines)"),
    Step(ToolCall("run_tests", {}), "1 passed"),
    Step(ToolCall("edit_file", {"path": "f.py", "old": "x - 1", "new": "x + 1"}), "edited f.py"),
    Step(ToolCall("done", {"message": "Done. <|end|> is just text here"}), None),
]


def test_episode_roundtrip_and_mask(tok):
    ids, is_model = encode_episode(tok, "Write f", EPISODE)
    assert len(ids) == len(is_model)
    assert ids[-1] == tok.eot_id and is_model[-1] == 1
    # request tokens are not model tokens; tool-call tokens are
    assert is_model[0] == 0
    assert is_model[ids.index(tok.sid("<|write_file|>"))] == 1
    assert is_model[ids.index(tok.sid("<|result|>"))] == 0
    x, y, m = training_example(tok, "Write f", EPISODE)
    assert len(x) == len(y) == len(m) == len(ids) - 1


def test_parse_each_call(tok):
    ids, is_model = encode_episode(tok, "Write f", EPISODE)
    # split model spans and parse each
    spans, cur = [], []
    for i, m in zip(ids, is_model):
        if m and i != tok.eot_id:
            cur.append(i)
            if i == tok.sid("<|end|>"):
                spans.append(cur)
                cur = []
    parsed = [parse_call(tok, s) for s in spans]
    assert [p.name for p in parsed] == [s.call.name for s in EPISODE]
    assert parsed[0].args == EPISODE[0].call.args
    assert parsed[3].args["message"] == "Done. <|end|> is just text here"  # forged tag stayed text


def test_parse_errors(tok):
    with pytest.raises(ParseError):
        parse_call(tok, [tok.sid("<|write_file|>"), tok.sid("<|content|>"), tok.sid("<|end|>")])
    with pytest.raises(ParseError):
        parse_call(tok, tok.encode("hello"))
    with pytest.raises(ParseError):
        parse_call(tok, [tok.sid("<|read_file|>"), tok.sid("<|path|>")] + tok.encode("a.py"))  # no <|end|>


def test_workspace_tools(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.list_files() == "(workspace is empty)"
    assert "wrote" in ws.write_file("m.py", "def f(x):\n    return x - 1\n")
    ws.write_file("test_m.py", "from m import f\n\n\ndef test_f():\n    assert f(1) == 2\n")
    out = ws.run_tests()
    assert "1 failed" in out and str(tmp_path) not in out and " in 0." not in out
    assert ws.execute(ToolCall("edit_file", {"path": "m.py", "old": "x - 1", "new": "x + 1"})) == "edited m.py"
    assert ws.run_tests().endswith("1 passed")
    assert "not found" in ws.execute(ToolCall("edit_file", {"path": "m.py", "old": "zzz", "new": "y"}))
    ws.write_file("d.py", "a = 1\na = 1\n")
    assert "appears 2 times" in ws.execute(ToolCall("edit_file", {"path": "d.py", "old": "a = 1", "new": "b"}))
    assert "outside the workspace" in ws.execute(ToolCall("read_file", {"path": "../x.py"}))
    assert "m.py:2:" in ws.search(r"return")
    assert ws.run_python("print(2 + 3)") == "5"


def test_sandbox_timeout(tmp_path):
    sb = SubprocessSandbox(timeout=2)
    r = sb.run_python(tmp_path, "while True: pass")
    assert r.timed_out and "Timed out" in r.output


def test_fastest_setup_never_picks_int8(tmp_path):
    # speed is all it measures, and int8 broke the 350M agent
    import torch

    from vishalam.infer.generate import fastest_setup
    from vishalam.model.config import ModelConfig
    from vishalam.model.model import Vishalam

    cfg = ModelConfig(vocab_size=384, d_model=32, n_layers=1, n_heads=2, ffn_dim=64, max_seq_len=128)
    path = tmp_path / "m.pt"
    torch.save({"model": Vishalam(cfg).state_dict(), "config": cfg.__dict__}, path)
    device, quantize = fastest_setup(path)
    assert quantize is False and device in ("cpu", "cuda")


def test_default_sandbox_needs_a_running_docker(monkeypatch):
    # installed but not running (or image not built): every test run would fail to connect
    import subprocess
    from types import SimpleNamespace

    from vishalam.agent import sandbox as sb

    monkeypatch.delenv("VISHALAM_SANDBOX", raising=False)
    monkeypatch.setattr(sb.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert isinstance(sb.default_sandbox(), sb.SubprocessSandbox)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert isinstance(sb.default_sandbox(), sb.DockerSandbox)
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    assert isinstance(sb.default_sandbox(), sb.SubprocessSandbox)
