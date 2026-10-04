"""The PyTorch/XLA (TPU) path of the pretraining loop, run on XLA's CPU backend.

Skipped unless torch_xla is installed (Linux only). On Kaggle CPU:
    PJRT_DEVICE=CPU CPU_NUM_DEVICES=1 python -m pytest tests/test_pretrain_xla.py
The TPU path is SPMD (one process, the batch split over the chips), so several simulated CPU
chips exercise the real data-parallel setup.
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch_xla")

from test_pretrain_smoke import ROOT, losses, setup  # noqa: E402,F401  (setup is a fixture)


def run_xla(cfg, shards, out, *extra, xla=True):
    cmd = [sys.executable, "-m", "vishalam.train.pretrain", "--config", str(cfg), "--data-dir", str(shards),
           "--out", str(out), "--max-steps", "40", *(["--xla"] if xla else []), *extra]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PJRT_DEVICE": "CPU"}
    env.setdefault("CPU_NUM_DEVICES", "1")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode != 0:  # full output on disk: pytest shortens long assertion messages
        (out.parent / "last_failure.txt").write_text(r.stdout + "\n---- stderr ----\n" + r.stderr)
    assert r.returncode == 0, f"failed, see {out.parent / 'last_failure.txt'}"
    return r.stdout


def test_xla_trains_and_resumes_exactly(setup):
    tmp, shards, cfg = setup
    run_xla(cfg, shards, tmp / "x_straight")
    straight = losses(tmp / "x_straight")
    assert straight[40] < straight[1] - 1.0, straight
    assert (tmp / "x_straight" / "model.pt").exists()

    run_xla(cfg, shards, tmp / "x_resumed", "--stop-at-step", "20")
    out = run_xla(cfg, shards, tmp / "x_resumed")
    assert "resumed from" in out
    resumed = losses(tmp / "x_resumed")
    for s in (30, 40):
        assert abs(resumed[s] - straight[s]) < 1e-3, (s, resumed[s], straight[s])


def test_checkpoints_move_between_cpu_and_xla(setup):
    tmp, shards, cfg = setup
    run_xla(cfg, shards, tmp / "c2x", "--stop-at-step", "20", xla=False)  # GPU/CPU session ...
    run_xla(cfg, shards, tmp / "c2x")                                      # ... continued on "TPU"
    run_xla(cfg, shards, tmp / "x2c", "--stop-at-step", "20")             # and the other way round
    run_xla(cfg, shards, tmp / "x2c", xla=False)
    for d in ("c2x", "x2c"):
        ls = losses(tmp / d)
        assert ls[40] < ls[1] - 1.0, (d, ls)


def test_xla_muon(setup):
    import yaml

    tmp, shards, cfg = setup
    raw = yaml.safe_load(cfg.read_text())
    raw["train"].update(optimizer="muon", muon_lr=0.02, muon_momentum=0.95)
    mcfg = tmp / "muon_xla.yaml"
    mcfg.write_text(yaml.safe_dump(raw))
    out = run_xla(mcfg, shards, tmp / "x_muon")
    assert "Muon(" in out
    ls = losses(tmp / "x_muon")
    assert ls[40] < ls[1] - 1.0, ls


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_xla_does_not_recompile_every_step(setup, optimizer):
    """The LR changes every step (warmup + cosine). If XLA baked it into the graph, every step would
    compile again: hours of compile time and growing host memory on the TPU."""
    import re

    tmp, shards, cfg = setup
    out = run_xla(cfg, shards, tmp / f"x_compiles_{optimizer}", "--max-steps", "30", "--set", f"optimizer={optimizer}",
                  "--set", "muon_lr=0.02")
    counts = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"step (\d+)/\d+ .*compiles (\d+)", out)}
    assert counts, out
    assert counts[30] == counts[10], counts  # nothing new to compile after the first steps


def test_xla_four_chips_match_one(setup):
    """Data parallel over 4 simulated chips computes the same training as one chip (same global
    batch, same data order); only the batch layout differs."""
    tmp, shards, cfg = setup
    run_xla(cfg, shards, tmp / "x_1chip", "--max-steps", "20")
    os.environ["CPU_NUM_DEVICES"] = "4"
    try:
        out = run_xla(cfg, shards, tmp / "x_4chips", "--max-steps", "20")
    finally:
        os.environ.pop("CPU_NUM_DEVICES")
    assert "x4," in out, out  # "device=xla:0x4, ..."
    one, four = losses(tmp / "x_1chip"), losses(tmp / "x_4chips")
    for s in (1, 10, 20):
        assert abs(one[s] - four[s]) < 2e-2, (s, one[s], four[s])
