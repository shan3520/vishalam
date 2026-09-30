"""Build the pretraining corpus.

1) Tokenizer sample (text file used to train the BPE):
    python -m vishalam.data.pretrain.shard sample --out data/tok_sample.txt --python-mb 850 --fineweb-mb 150

2) Token shards (resumable; run repeatedly in Kaggle CPU sessions until the target is reached):
    python -m vishalam.data.pretrain.shard build --tokenizer tokenizer.json \
        --repo shantanu3520/vishalam-data --work-dir /kaggle/working/shards --target-tokens 10.5e9 --max-hours 11

Each "unit" = one the-stack Python parquet file (license + quality filtered, decontaminated)
mixed at the document level with FineWeb-Edu text so that English is ~15% of tokens. Every
unit becomes one shard (raw uint16 tokens). Progress, stats and the manifest are stored in the
HF dataset repo, so a new session continues exactly where the last one stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from vishalam.data.pretrain.decontam import Decontaminator
from vishalam.data.pretrain.filters import license_ok, python_quality_reason
from vishalam.data.pretrain.sources import (
    FINEWEB_COLUMNS, FINEWEB_PREFIX, FINEWEB_REPO, STACK_COLUMNS, STACK_PREFIX, STACK_REPO,
    iter_parquet_rows, list_parquet,
)

ENGLISH_FRACTION = 0.15
VAL_TOKENS = 10_000_000

# ---------------- worker side ----------------
_TOK = None
_DEC = None


def _init_worker(tok_path: str, dec_path: str | None):
    global _TOK, _DEC
    from vishalam.tokenizer.bpe import Tokenizer

    _TOK = Tokenizer.load(tok_path)
    _DEC = Decontaminator.load(dec_path) if dec_path else None


def _encode_doc(text: str):
    if _DEC is not None and _DEC.contaminated(text):
        return None
    ids = _TOK.encode_ordinary(text)
    ids.append(_TOK.eot_id)
    return np.asarray(ids, dtype=np.uint16)


# ---------------- helpers ----------------
def filter_python_rows(rows, stats: dict):
    for r in rows:
        stats["py_seen"] = stats.get("py_seen", 0) + 1
        if not license_ok(r.get("max_stars_repo_licenses")):
            stats["py_rej_license"] = stats.get("py_rej_license", 0) + 1
            continue
        reason = python_quality_reason(r["content"], r)
        if reason:
            stats[f"py_rej_{reason}"] = stats.get(f"py_rej_{reason}", 0) + 1
            continue
        yield r["content"]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------- sample mode ----------------
def cmd_sample(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "_raw"
    stats: dict = {}
    with open(out, "w", encoding="utf-8") as f:
        budget = args.python_mb * 1e6
        # take from the END of the file list so the sample spans different repos than unit 0
        for fn in reversed(list_parquet(STACK_REPO, STACK_PREFIX)):
            for text in filter_python_rows(iter_parquet_rows(STACK_REPO, fn, STACK_COLUMNS, cache), stats):
                f.write(text + "\n")
                budget -= len(text)
                if budget <= 0:
                    break
            if budget <= 0:
                break
        budget = args.fineweb_mb * 1e6
        for fn in list_parquet(FINEWEB_REPO, FINEWEB_PREFIX):
            for r in iter_parquet_rows(FINEWEB_REPO, fn, FINEWEB_COLUMNS, cache):
                f.write(r["text"] + "\n")
                budget -= len(r["text"])
                if budget <= 0:
                    break
            if budget <= 0:
                break
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB); python filter stats: {stats}")


# ---------------- build mode ----------------
class FinewebCursor:
    """Resumable position in the FineWeb-Edu file list."""

    def __init__(self, files: list[str], file_idx: int, row: int, cache: Path):
        self.files, self.file_idx, self.row, self.cache = files, file_idx, row, cache
        self._it = None

    def next_docs(self, n: int) -> list[str]:
        docs = []
        while len(docs) < n:
            if self._it is None:
                if self.file_idx >= len(self.files):
                    self.file_idx, self.row = 0, 0  # wrap (English is only 15%, 10BT sample is plenty)
                self._it = iter_parquet_rows(FINEWEB_REPO, self.files[self.file_idx], FINEWEB_COLUMNS,
                                             self.cache, skip_rows=self.row)
            try:
                docs.append(next(self._it)["text"])
                self.row += 1
            except StopIteration:
                self._it = None
                self.file_idx += 1
                self.row = 0
        return docs


class LocalSink:
    """Writes outputs straight into a directory (a Kaggle kernel's /kaggle/working output)."""

    keeps_files = True

    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)

    def pull(self, name: str, _work: Path):
        p = self.out / name
        return p if p.exists() else None

    def push(self, local: Path, name: str, _msg: str = ""):
        local, dst = Path(local), self.out / name
        if local.resolve() != dst.resolve():
            shutil.copy(local, dst)


class HubSink:
    """Uploads outputs to a private HF dataset repo (original design; needs a write HF_TOKEN)."""

    keeps_files = False

    def __init__(self, repo: str):
        from vishalam.train.hub import HubSync

        self.hub = HubSync(repo, repo_type="dataset")

    def pull(self, name: str, work: Path):
        return self.hub.pull(name, work)

    def push(self, local: Path, name: str, msg: str = "update"):
        self.hub.push(local, name, msg)


def parse_range(spec: str | None, n: int) -> tuple[int, int]:
    """File range 'a:b' as indices, or as fractions of n when either bound contains a dot
    (e.g. '0:0.5' and '0.5:1.0')."""
    if not spec:
        return 0, n
    lo, hi = spec.split(":")
    if "." in lo or "." in hi:
        a, b = round(float(lo) * n), round(float(hi) * n)
    else:
        a, b = int(lo), int(hi)
    return max(0, a), min(n, b)


def cmd_build(args):
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else work / "_raw"
    if args.local_out:
        sink = LocalSink(Path(args.local_out))
        work = sink.out  # shards are written in place, never copied
    else:
        sink = HubSink(args.repo)

    tok_sha = sha256(args.tokenizer)
    py_all = list_parquet(STACK_REPO, STACK_PREFIX)
    fw_all = list_parquet(FINEWEB_REPO, FINEWEB_PREFIX)
    py_lo, py_hi = parse_range(args.py_files, len(py_all))
    fw_lo, fw_hi = parse_range(args.fw_files, len(fw_all))
    print(f"python files {py_lo}:{py_hi} of {len(py_all)}, fineweb files {fw_lo}:{fw_hi} of {len(fw_all)}",
          flush=True)

    prog_path = sink.pull("progress.json", work)
    progress = json.loads(prog_path.read_text()) if prog_path else {
        "tokenizer_sha": tok_sha, "python_file_idx": py_lo, "fineweb_file_idx": 0,
        "fineweb_row": 0, "train": [], "val": [], "stats": {}, "tokens": {"python": 0, "english": 0},
        "py_range": [py_lo, py_hi], "fw_range": [fw_lo, fw_hi], "completed_py_files": [],
    }
    assert progress["tokenizer_sha"] == tok_sha, "tokenizer changed since sharding began"
    sink.push(args.tokenizer, "tokenizer.json", "tokenizer")

    dec_path = work / "decontam.json"
    if not dec_path.exists():
        reuse = Path(args.decontam) if args.decontam else None
        if reuse is not None and reuse.exists():
            shutil.copy(reuse, dec_path)
        else:
            Decontaminator.from_benchmarks().save(dec_path)
        sink.push(dec_path, "decontam.json", "decontamination n-grams")

    fw = FinewebCursor(fw_all[fw_lo:fw_hi], progress["fineweb_file_idx"], progress["fineweb_row"], raw_dir)
    stats = progress["stats"]
    t0 = time.time()

    def total_tokens():
        return progress["tokens"]["python"] + progress["tokens"]["english"]

    def save_progress(msg):
        (work / "progress.json").write_text(json.dumps(progress, indent=1))
        manifest = {"tokenizer_sha": progress["tokenizer_sha"], "train": progress["train"], "val": progress["val"]}
        (work / "manifest.json").write_text(json.dumps(manifest, indent=1))
        sink.push(work / "progress.json", "progress.json", msg)
        sink.push(work / "manifest.json", "manifest.json", msg)

    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer, str(dec_path))) as pool:
        while progress["python_file_idx"] < py_hi and total_tokens() < args.target_tokens:
            if (time.time() - t0) / 3600 > args.max_hours:
                print("time budget reached; progress saved - continue with a follow-up run")
                break
            ui = progress["python_file_idx"]
            fn = py_all[ui]
            print(f"unit {ui}: {fn}", flush=True)
            docs, py_tok, en_tok = [], 0, 0
            batch = []

            def flush_python(batch):
                nonlocal py_tok, en_tok
                for arr in pool.imap(_encode_doc, batch, chunksize=32):
                    if arr is None:
                        stats["contaminated"] = stats.get("contaminated", 0) + 1
                        continue
                    docs.append(arr)
                    py_tok += len(arr)
                # top up English to keep the target ratio
                want = py_tok * ENGLISH_FRACTION / (1 - ENGLISH_FRACTION)
                while en_tok < want:
                    n_docs = 256 if want - en_tok > 500_000 else 16  # don't overshoot small deficits
                    for arr in pool.imap(_encode_doc, fw.next_docs(n_docs), chunksize=16):
                        if arr is None:
                            stats["contaminated"] = stats.get("contaminated", 0) + 1
                            continue
                        docs.append(arr)
                        en_tok += len(arr)

            for text in filter_python_rows(iter_parquet_rows(STACK_REPO, fn, STACK_COLUMNS, raw_dir), stats):
                batch.append(text)
                if len(batch) >= 4096:
                    flush_python(batch)
                    batch = []
            if batch:
                flush_python(batch)

            random.Random(ui).shuffle(docs)
            if args.make_val and not progress["val"]:
                val_docs, n = [], 0
                while docs and n < VAL_TOKENS:
                    d = docs.pop()
                    val_docs.append(d)
                    n += len(d)
                vpath = work / "val_00000.bin"
                np.concatenate(val_docs).tofile(vpath)
                sink.push(vpath, vpath.name, "val shard")
                progress["val"] = [{"name": vpath.name, "tokens": int(n)}]
                if not sink.keeps_files:
                    vpath.unlink()
            arr = np.concatenate(docs) if docs else np.zeros(0, dtype=np.uint16)
            spath = work / f"train_{ui:05d}.bin"
            arr.tofile(spath)
            sink.push(spath, spath.name, f"shard {ui}")
            if not sink.keeps_files:
                spath.unlink()

            progress["train"].append({"name": spath.name, "tokens": int(len(arr))})
            progress["tokens"]["python"] += int(py_tok)
            progress["tokens"]["english"] += int(en_tok)
            progress["python_file_idx"] = ui + 1
            progress.setdefault("completed_py_files", []).append(ui)
            progress["fineweb_file_idx"], progress["fineweb_row"] = fw.file_idx, fw.row
            save_progress(f"progress unit {ui}")
            print(f"  +{py_tok / 1e6:.1f}M python, +{en_tok / 1e6:.1f}M english tokens; "
                  f"total {total_tokens() / 1e9:.2f}B ({(time.time() - t0) / 60:.0f} min)", flush=True)
    save_progress("final")
    print(json.dumps({"tokens": progress["tokens"], "next_python_file": progress["python_file_idx"],
                      "py_range": [py_lo, py_hi], "stats": stats}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--python-mb", type=float, default=850)
    s.add_argument("--fineweb-mb", type=float, default=150)
    b = sub.add_parser("build")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--repo", default=None, help="HF dataset repo (upload mode, needs a write token)")
    b.add_argument("--local-out", default=None, help="write shards here instead (Kaggle kernel output)")
    b.add_argument("--work-dir", default="/tmp/vishalam_work")
    b.add_argument("--raw-dir", default=None, help="download dir for raw parquet files (deleted after use)")
    b.add_argument("--py-files", default=None, help="the-stack Python file range: 'a:b' or fractions '0:0.5'")
    b.add_argument("--fw-files", default=None, help="FineWeb-Edu file range, same format")
    b.add_argument("--make-val", action="store_true", help="carve the validation shard from the first unit")
    b.add_argument("--decontam", default=None, help="existing decontam.json to reuse")
    b.add_argument("--target-tokens", type=float, default=10.5e9)
    b.add_argument("--max-hours", type=float, default=11.0)
    b.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"sample": cmd_sample, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
