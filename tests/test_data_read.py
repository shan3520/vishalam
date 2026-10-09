"""Shard reads survive a flaky (network-mounted) input: errors are retried, not fatal."""
import json

import numpy as np

import vishalam.train.data as data
from vishalam.train.data import GlobalSampler, ShardStore


def make_store(tmp_path, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(tmp_path / "train_00000.bin")
    (tmp_path / "manifest.json").write_text(json.dumps({"train": [{"name": "train_00000.bin", "tokens": len(tokens)}]}))
    store = ShardStore(tmp_path)
    return store, store.manifest()["train"]


def test_read_returns_the_right_tokens(tmp_path):
    store, shards = make_store(tmp_path, range(100))
    samp = GlobalSampler(store, shards, seq_len=8)
    assert samp.read(0, 10, 5).tolist() == [10, 11, 12, 13, 14]
    assert samp.read(0, 10, 5).dtype == np.int64


def test_read_retries_a_failing_input(tmp_path, monkeypatch):
    store, shards = make_store(tmp_path, range(100))
    samp = GlobalSampler(store, shards, seq_len=8)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    real_open, failures = open, [2]

    class Flaky:
        def __init__(self, f):
            self.f = f

        def seek(self, n):
            self.f.seek(n)

        def read(self, n):
            if failures[0]:
                failures[0] -= 1
                raise OSError("Transport endpoint is not connected")
            return self.f.read(n)

        def close(self):
            self.f.close()

    monkeypatch.setattr(data, "open", lambda *a, **k: Flaky(real_open(*a, **k)), raising=False)
    assert samp.read(0, 20, 3).tolist() == [20, 21, 22]
    assert failures[0] == 0  # both failures happened and were retried
