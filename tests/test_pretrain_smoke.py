"""End-to-end: tokenizer -> shards -> pretrain loop -> checkpoint -> exact resume."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from vishalam.tokenizer.bpe import NUM_SPECIAL_SLOTS, Tokenizer
from vishalam.tokenizer.bpe_train import count_chunks, train_bpe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pre")
    import glob, os

    lib = os.path.join(sys.base_prefix, "Lib")
    texts = []
    for f in sorted(glob.glob(lib + "/json/*.py") + glob.glob(lib + "/email/*.py"))[:30]:
        texts.append(Path(f).read_text(encoding="utf-8", errors="replace"))
    vocab = 256 + NUM_SPECIAL_SLOTS + 512
    tok = Tokenizer(train_bpe(count_chunks(texts), vocab, verbose=False))
    shard_dir = tmp / "shards"
    shard_dir.mkdir()
    ids = []
    for t in texts:
        ids += tok.encode(t) + [tok.eot_id]
    arr = np.asarray(ids, dtype=np.uint16)
    cut = len(arr) // 10
    arr[cut:].tofile(shard_dir / "train_00000.bin")
    arr[:cut].tofile(shard_dir / "val_00000.bin")
    (shard_dir / "manifest.json").write_text(json.dumps({
        "train": [{"name": "train_00000.bin", "tokens": len(arr) - cut}],
        "val": [{"name": "val_00000.bin", "tokens": cut}],
    }))
    cfg = {
        "name": "smoke",
        "model": {"vocab_size": vocab, "d_model": 64, "n_layers": 2, "n_heads": 4, "ffn_dim": 128, "max_seq_len": 128},
        "train": {"seq_len": 128, "micro_batch": 4, "global_batch_seqs": 8, "total_tokens": 10**9, "lr": 3e-3,
                  "min_lr_ratio": 0.1, "warmup_steps": 5, "weight_decay": 0.1, "grad_clip": 1.0, "z_loss": 1e-4,
                  "grad_checkpointing": False, "eval_every": 10, "eval_batches": 2, "save_every_minutes": 999},
    }
    cfg_path = tmp / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return tmp, shard_dir, cfg_path


def run(cfg, shards, out, *extra):
    data = ["--data-dirs", shards] if isinstance(shards, str) else ["--data-dir", str(shards)]
    cmd = [sys.executable, "-m", "vishalam.train.pretrain", "--config", str(cfg), *data,
           "--out", str(out), "--max-steps", "40", *extra]
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"}  # deterministic CPU run
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def losses(out):
    with open(out / "log.csv") as f:
        return {int(r["step"]): float(r["loss"]) for r in csv.DictReader(f)}


def test_loss_decreases_and_resume_is_exact(setup):
    tmp, shards, cfg = setup
    run(cfg, shards, tmp / "straight")
    straight = losses(tmp / "straight")
    assert straight[40] < straight[1] - 1.0, straight
    assert (tmp / "straight" / "model.pt").exists()

    run(cfg, shards, tmp / "resumed", "--stop-at-step", "20")
    assert not (tmp / "resumed" / "model.pt").exists()  # paused, not finished
    out = run(cfg, shards, tmp / "resumed")
    assert "resumed from" in out
    resumed = losses(tmp / "resumed")
    for s in (30, 40):
        assert abs(resumed[s] - straight[s]) < 1e-3, (s, resumed[s], straight[s])


def test_kaggle_ping_pong_with_split_shard_dirs(setup):
    """Shards spread over two read-only 'kernel output' dirs; session A pauses, session B (new
    output dir) resumes from A's dir. Must match one uninterrupted run exactly."""
    tmp, shards, cfg = setup
    arr = np.fromfile(shards / "train_00000.bin", dtype=np.uint16)
    d1, d2 = tmp / "data-1", tmp / "data-2"
    for d, part, has_val in [(d1, arr[: len(arr) // 2], True), (d2, arr[len(arr) // 2 :], False)]:
        d.mkdir()
        part.tofile(d / "train_00000.bin")  # same file name in both dirs: paths must stay distinct
        m = {"train": [{"name": "train_00000.bin", "tokens": len(part)}], "val": []}
        if has_val:
            shutil.copy(shards / "val_00000.bin", d / "val_00000.bin")
            m["val"] = [{"name": "val_00000.bin", "tokens": int(np.fromfile(d / "val_00000.bin", dtype=np.uint16).size)}]
        (d / "manifest.json").write_text(json.dumps(m))
    dirs = f"{d1},{d2}"
    run(cfg, dirs, tmp / "pp_straight")
    straight = losses(tmp / "pp_straight")
    run(cfg, dirs, tmp / "pp_a", "--stop-at-step", "20")
    out = run(cfg, dirs, tmp / "pp_b", "--resume-from", str(tmp / "pp_a"))
    assert "resumed from" in out
    b = losses(tmp / "pp_b")
    assert min(b) == 1  # log.csv carried over from session A: one continuous curve
    for s in (30, 40):
        assert abs(b[s] - straight[s]) < 1e-3, (s, b[s], straight[s])


def test_muon_training_and_exact_resume(setup):
    tmp, shards, cfg = setup
    c = yaml.safe_load(cfg.read_text())
    c["train"].update(optimizer="muon", muon_lr=0.02, muon_momentum=0.95)
    mcfg = tmp / "muon.yaml"
    mcfg.write_text(yaml.safe_dump(c))
    out = run(mcfg, shards, tmp / "m_straight")
    assert "Muon(" in out and "AdamW(" in out
    straight = losses(tmp / "m_straight")
    assert straight[40] < straight[1] - 1.0, straight
    run(mcfg, shards, tmp / "m_resumed", "--stop-at-step", "20")
    out = run(mcfg, shards, tmp / "m_resumed")
    assert "resumed from" in out
    resumed = losses(tmp / "m_resumed")
    for s in (30, 40):
        assert abs(resumed[s] - straight[s]) < 1e-3, (s, resumed[s], straight[s])
