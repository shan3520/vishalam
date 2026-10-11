"""SFT with pretraining-text mixing runs end to end (tiny model, CPU)."""
import json
import os
import random
import subprocess
import sys

import numpy as np
import yaml

from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.data.agent.mbpp_episodes import episode_from_record
from vishalam.tokenizer.bpe import NUM_SPECIAL_SLOTS, Tokenizer
from vishalam.tokenizer.bpe_train import count_chunks, train_bpe

from test_mbpp_episodes import REC
from test_pretrain_smoke import ROOT


def test_sft_with_mix_trains(tmp_path):
    ep = episode_from_record(REC, SubprocessSandbox(timeout=20), random.Random(0))
    eps = tmp_path / "eps.jsonl"
    eps.write_text("".join(json.dumps(ep) + "\n" for _ in range(40)))
    text = [ep["request"], ep["ref_source"], json.dumps(ep["steps"])] * 10
    tok = Tokenizer(train_bpe(count_chunks(text), 256 + NUM_SPECIAL_SLOTS + 128, verbose=False))
    tok.save(tmp_path / "tok.json")
    shards = tmp_path / "shards"
    shards.mkdir()
    ids = np.asarray([t for s in text for t in tok.encode(s) + [tok.eot_id]] * 4, dtype=np.uint16)
    ids.tofile(shards / "train_00000.bin")
    (shards / "manifest.json").write_text(json.dumps({"train": [{"name": "train_00000.bin", "tokens": len(ids)}]}))
    cfg = {"name": "sft-mix", "model": {"vocab_size": tok.vocab_size, "d_model": 32, "n_layers": 1, "n_heads": 4,
                                        "ffn_dim": 64, "max_seq_len": 512},
           "train": {"seq_len": 512, "micro_batch": 2, "global_batch_seqs": 4, "epochs": 1, "lr": 3e-3,
                     "min_lr_ratio": 0.1, "warmup_steps": 2, "weight_decay": 0.0, "grad_clip": 1.0,
                     "mix_fraction": 0.25, "eval_every": 1000}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))

    def sft(out, *extra):
        r = subprocess.run([sys.executable, "-m", "vishalam.train.sft", "--config", str(tmp_path / "cfg.yaml"),
                            "--episodes", str(eps), "--tokenizer", str(tmp_path / "tok.json"),
                            "--mix-data-dirs", str(shards), "--out", str(out), "--max-steps", "20", *extra],
                           cwd=ROOT, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    out = sft(tmp_path / "a")
    assert "1/4 rows per step from pretraining text" in out and "finished" in out
    rows = [r.split(",") for r in (tmp_path / "a" / "log.csv").read_text().splitlines()[1:]]
    assert [int(r[0]) for r in rows] == [1, 10, 20]
    assert float(rows[-1][1]) < float(rows[0][1]) - 0.5  # loss went down
