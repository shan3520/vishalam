"""Local (Kaggle-output) shard build with file-range splitting, using fake sources (no network)."""
import argparse
import json

import numpy as np

import vishalam.data.pretrain.shard as shard
from vishalam.data.pretrain.decontam import Decontaminator
from vishalam.tokenizer.bpe import NUM_SPECIAL_SLOTS, Tokenizer
from vishalam.tokenizer.bpe_train import count_chunks, train_bpe
from vishalam.train.data import merge_local_manifests

PY = [f"data/python/p{i}.parquet" for i in range(4)]
FW = [f"sample/10BT/f{i}.parquet" for i in range(2)]


def fake_rows(repo, filename, columns, cache_dir, skip_rows=0, batch_size=2000):
    if repo == shard.STACK_REPO:
        i = PY.index(filename)
        for j in range(30):
            code = f"def f_{i}_{j}(x):\n    # helper number {j}\n    return x * {j} + {i}\n" * 3
            yield {"content": code, "max_stars_repo_licenses": ["MIT"] if j % 5 else ["GPL-3.0"],
                   "max_line_length": 40, "avg_line_length": 20.0, "alphanum_fraction": 0.6}
    else:
        for j in range(skip_rows, 400):
            yield {"text": f"English document {j}."}


def build(tmp_path, monkeypatch, out, py, fw, make_val):
    monkeypatch.setattr(shard, "list_parquet", lambda repo, prefix: PY if repo == shard.STACK_REPO else FW)
    monkeypatch.setattr(shard, "iter_parquet_rows", fake_rows)
    monkeypatch.setattr(Decontaminator, "from_benchmarks", classmethod(lambda cls: cls(["unrelated benchmark text"])))
    monkeypatch.setattr(shard, "VAL_TOKENS", 200)
    tok_path = tmp_path / "tok.json"
    if not tok_path.exists():
        corpus = [r["content"] for r in fake_rows(shard.STACK_REPO, PY[0], [], None)]
        Tokenizer(train_bpe(count_chunks(corpus), 256 + NUM_SPECIAL_SLOTS + 200, 1, verbose=False)).save(tok_path)
    args = argparse.Namespace(tokenizer=str(tok_path), repo=None, local_out=str(out), work_dir=str(tmp_path / "w"),
                              raw_dir=str(tmp_path / "raw"), py_files=py, fw_files=fw, make_val=make_val,
                              decontam=None, target_tokens=1e12, max_hours=1.0, workers=1)
    shard.cmd_build(args)
    return json.loads((out / "progress.json").read_text())


def test_two_kernels_split_files_and_merge(tmp_path, monkeypatch):
    p1 = build(tmp_path, monkeypatch, tmp_path / "data-1", "0:0.5", "0:0.5", True)
    p2 = build(tmp_path, monkeypatch, tmp_path / "data-2", "0.5:1.0", "0.5:1.0", False)
    assert p1["completed_py_files"] == [0, 1] and p2["completed_py_files"] == [2, 3]
    assert p1["val"] and not p2["val"]
    assert p1["stats"]["py_rej_license"] == 12  # GPL files dropped (6 per source file)
    for p, d in [(p1, "data-1"), (p2, "data-2")]:
        for s in p["train"]:
            assert (tmp_path / d / s["name"]).stat().st_size == 2 * s["tokens"]
        eng = p["tokens"]["english"] / (p["tokens"]["python"] + p["tokens"]["english"])
        assert 0.1 < eng < 0.3, eng
    m = merge_local_manifests([tmp_path / "data-1", tmp_path / "data-2"])
    assert len(m["train"]) == 4 and len(m["val"]) == 1
    assert len({s["name"] for s in m["train"]}) == 4  # same file names in two dirs stay distinct
    tok = Tokenizer.load(tmp_path / "tok.json")
    ids = np.fromfile(m["train"][0]["name"], dtype=np.uint16)
    assert tok.eot_id in ids and "def f_" in tok.decode(ids[:200].tolist())
