"""vishalam.eval.run keeps finished sets when a session runs out of time, and a later run can skip them."""
import json
import sys

import vishalam.eval.run as run


def _fake(monkeypatch, calls):
    monkeypatch.setattr(run, "load_model", lambda w, d: (None, None, "cpu"))
    monkeypatch.setattr(run, "load_tokenizer", lambda p: None)
    monkeypatch.setattr(run, "Agent", lambda *a, **k: None)
    monkeypatch.setattr(run, "SubprocessSandbox", lambda: None)
    tasks = lambda split: (lambda: [{"split": split, "i": i} for i in range(3)])
    monkeypatch.setattr(run, "load_humaneval", tasks("humaneval"))
    monkeypatch.setattr(run, "load_mbpp", tasks("mbpp"))

    def run_task(agent, task, sandbox):
        calls.append(task["split"])
        return {"split": task["split"], "kind": "create", "passed": True, "first_try": True, "finished": True,
                "n_steps": 1, "parse_errors": 0}
    monkeypatch.setattr(run, "run_task", run_task)


def _main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run", "--weights", "w", "--tokenizer", "t", "--sandbox", "subprocess", *args])
    run.main()


def test_out_of_time_keeps_finished_sets_and_resume_skips_them(monkeypatch, tmp_path):
    calls = []
    _fake(monkeypatch, calls)
    out = tmp_path / "results.json"
    clock = iter([0, 0, 1, 2, 3, 10_000, 10_000, 10_000])  # mbpp's first task is past the limit
    monkeypatch.setattr(run.time, "time", lambda: next(clock, 10_000))
    _main(monkeypatch, "--sets", "humaneval,mbpp", "--max-hours", "1", "--out", str(out))
    first = json.loads(out.read_text())
    assert first["complete_sets"] == ["humaneval"] and calls == ["humaneval"] * 3

    calls.clear()
    monkeypatch.setattr(run.time, "time", lambda: 0)
    out2 = tmp_path / "results2.json"
    _main(monkeypatch, "--sets", "humaneval,mbpp", "--skip-done", str(out), "--out", str(out2))
    second = json.loads(out2.read_text())
    assert calls == ["mbpp"] * 3
    assert second["complete_sets"] == ["humaneval", "mbpp"]
    assert second["table"]["humaneval"]["n"] == 3 and second["table"]["mbpp"]["n"] == 3
