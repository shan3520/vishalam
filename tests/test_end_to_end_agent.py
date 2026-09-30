"""Integration: episode -> SFT (overfit a tiny model) -> agent run in a fresh workspace -> hidden tests pass.

Proves the data format, packing + loss masking, generation, call parsing, tool execution and
scoring all agree with each other. Takes ~1-2 minutes on CPU.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from vishalam.agent.format import Step, encode_episode
from vishalam.agent.runtime import Agent
from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.data.agent.families import BY_ID, make_instance
from vishalam.data.agent.phrasing import make_wrap
from vishalam.data.agent.trajectories import synthesize
from vishalam.eval.harness import run_task
from vishalam.infer.generate import load_model
from vishalam.tokenizer.bpe import NUM_SPECIAL_SLOTS, Tokenizer
from vishalam.tokenizer.bpe_train import count_chunks, train_bpe

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_overfit_agent_solves_its_task(tmp_path):
    rng = random.Random(5)
    sb = SubprocessSandbox(timeout=10)
    ep = None
    while ep is None:  # a create-episode with a bug + fix, so every tool path is exercised
        task = make_instance(BY_ID["second_largest"], rng, "train", wrap=make_wrap("train"))
        ep = synthesize(task, rng, "create", sb, p_bug=1.0)
    ep_path = tmp_path / "eps.jsonl"
    ep_path.write_text((json.dumps(ep) + "\n") * 40)

    steps = [Step.from_dict(s) for s in ep["steps"]]
    corpus = [ep["request"]] + [json.dumps(s) for s in ep["steps"]]
    tok = Tokenizer(train_bpe(count_chunks(corpus * 5), 256 + NUM_SPECIAL_SLOTS + 400, 1, verbose=False))
    tok.save(tmp_path / "tok.json")
    n_tokens = len(encode_episode(tok, ep["request"], steps)[0])
    assert n_tokens < 1024, n_tokens

    cfg = {
        "name": "e2e",
        "model": {"vocab_size": tok.vocab_size, "d_model": 128, "n_layers": 3, "n_heads": 4,
                  "ffn_dim": 384, "max_seq_len": 1024},
        "train": {"seq_len": 1024, "micro_batch": 4, "global_batch_seqs": 8, "epochs": 1, "lr": 3e-3,
                  "min_lr_ratio": 0.1, "warmup_steps": 10, "weight_decay": 0.0, "grad_clip": 1.0,
                  "eval_every": 1000, "save_every_minutes": 999},
    }
    (tmp_path / "sft.yaml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, "-m", "vishalam.train.sft", "--config", str(tmp_path / "sft.yaml"),
                        "--episodes", str(ep_path), "--tokenizer", str(tmp_path / "tok.json"),
                        "--out", str(tmp_path / "run"), "--max-steps", "250"],
                       cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]

    model, mcfg, device = load_model(tmp_path / "run" / "model.pt", "cpu", quantize_cpu=False)
    agent = Agent(model, mcfg, device, tok, max_steps=12)
    task_rec = {"split": "e2e", "request": ep["request"], "fname": ep["fname"], "initial_files": {},
                "hidden_cases": ep["hidden_cases"]}
    rec = run_task(agent, task_rec, sb)
    names = [s["call"]["name"] for s in rec["steps"]]
    assert rec["parse_errors"] == 0, rec
    assert names == [s.call.name for s in steps], names
    assert rec["finished"] and rec["passed"], rec["hidden_output"]
