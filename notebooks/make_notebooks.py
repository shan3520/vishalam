"""Generates the Kaggle notebooks (thin wrappers around the repo's scripts).

    python notebooks/make_notebooks.py

Design (no GitHub token, no Hugging Face uploads):
  * the code arrives as the private Kaggle dataset <user>/vishalam-code (a tarball of the repo,
    uploaded by tools/kaggle_driver.py push-code);
  * every intermediate artefact (tokenizer, token shards, checkpoints, agent data) is a kernel's
    /kaggle/working output, attached to later kernels as an input ("kernel_sources");
  * only the two data kernels need a secret (HF_TOKEN, read-only) because bigcode/the-stack-dedup
    is gated. Kaggle hands secrets only to runs started from the web editor.
tools/kaggle_driver.py fills in dataset/kernel sources and pushes/monitors the kernels.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def _kaggle_user() -> str:
    """Kaggle username from ~/.kaggle/kaggle.json (only the username field is read)."""
    try:
        return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    except (OSError, KeyError, ValueError):
        return "KAGGLE_USERNAME"


KAGGLE_USER = _kaggle_user()
CODE_DATASET = f"{KAGGLE_USER}/vishalam-code"

SETUP = '''\
# --- Vishalam setup: find the code dataset among the inputs, unpack, install (no secrets) ---
import glob, os, shutil, subprocess, sys, tarfile, time
T0 = time.time()
def hours_left(limit=11.2):  # Kaggle stops sessions at 12 h; keep a margin for saving outputs
    return max(0.25, limit - (time.time() - T0) / 3600)
WORK, INPUT, CODE = "/kaggle/working", "/kaggle/input", "/tmp/vishalam"

def inputs_with(marker, hint=""):
    """Input directories that contain `marker` (a relative path) and whose path contains `hint`."""
    hits = glob.glob(f"{INPUT}/**/{marker}", recursive=True)
    dirs = [h[: -len(marker)].rstrip("/") for h in hits]
    return sorted(d for d in set(dirs) if hint in d)

if not os.path.isdir(CODE):
    tars = glob.glob(f"{INPUT}/**/vishalam_code.tar.gz", recursive=True)
    if tars:
        os.makedirs(CODE)
        with tarfile.open(tars[0]) as t:
            t.extractall(CODE, filter="data")
    else:  # Kaggle sometimes unpacks archives itself
        roots = [d for d in inputs_with("vishalam/tokenizer/bpe.py") if os.path.exists(f"{d}/requirements.txt")]
        if not roots:
            raise SystemExit("code dataset not attached - run tools/kaggle_driver.py push-code and re-push this kernel")
        shutil.copytree(roots[0], CODE)
os.chdir(CODE)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("ready:", open("VERSION").read().strip() if os.path.exists("VERSION") else "code unpacked")
'''

HF_SECRET = '''\
# --- HF_TOKEN (read-only) is needed only to download the gated bigcode/the-stack-dedup ---
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN").strip()
except Exception as e:
    raise SystemExit("HF_TOKEN secret not available. Open this notebook in the Kaggle editor, "
                     "Add-ons -> Secrets -> tick HF_TOKEN, then Save Version -> Save & Run All. "
                     f"({type(e).__name__})")
print("HF_TOKEN present")
'''

GPU_GUARD = '''\
# --- GPU guard: stop before spending quota if Kaggle gave us the wrong accelerator ---
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPUs:", names)
if not names or not all("T4" in n for n in names):
    raise SystemExit(f"Expected T4 GPUs, got {names}. Set Accelerator -> 'GPU T4 x2' and run again.")
'''

DATA1_CELLS = [
    SETUP, HF_SECRET,
    '''\
# --- Tokenizer: ~1 GB sample (MIT/Apache/BSD Python + FineWeb-Edu) -> our own 50k BPE ---
if not os.path.exists(f"{WORK}/tokenizer.json"):
    get_ipython().system("python -m vishalam.data.pretrain.shard sample --out /tmp/tok_sample.txt --python-mb 850 --fineweb-mb 150")
    get_ipython().system(f"python -m vishalam.tokenizer.bpe_train --input /tmp/tok_sample.txt --vocab-size 50304 --workers 4 --out {WORK}/tokenizer.json")
    os.remove("/tmp/tok_sample.txt")
print("tokenizer:", os.path.getsize(f"{WORK}/tokenizer.json"), "bytes")
''',
    '''\
# --- Token shards, first half of the source files (~5.3B tokens, ~10.6 GB) ---
get_ipython().system(f"python -m vishalam.data.pretrain.shard build --tokenizer {WORK}/tokenizer.json "
                     f"--local-out {WORK} --work-dir /tmp/w --raw-dir /tmp/raw --py-files 0:0.5 --fw-files 0:0.5 "
                     f"--make-val --target-tokens 5.3e9 --max-hours {hours_left():.2f} --workers 4")
''',
]

DATA2_CELLS = [
    SETUP, HF_SECRET,
    '''\
# --- Token shards, second half of the source files, same tokenizer as vishalam-data-1 ---
d1 = inputs_with("tokenizer.json", "vishalam-data-1")
if not d1:
    raise SystemExit("vishalam-data-1 output not attached")
shutil.copy(f"{d1[0]}/tokenizer.json", f"{WORK}/tokenizer.json")
get_ipython().system(f"python -m vishalam.data.pretrain.shard build --tokenizer {WORK}/tokenizer.json "
                     f"--local-out {WORK} --work-dir /tmp/w --raw-dir /tmp/raw --py-files 0.5:1.0 --fw-files 0.5:1.0 "
                     f"--decontam {d1[0]}/decontam.json --target-tokens 5.3e9 --max-hours {hours_left():.2f} --workers 4")
''',
]

DATA_INPUTS = '''\
# --- locate the tokenizer and all token-shard dirs among the attached kernel outputs ---
DATA_DIRS = inputs_with("manifest.json", "vishalam-data-")
TOKENIZER = inputs_with("tokenizer.json", "vishalam-data-1")[0] + "/tokenizer.json"
print("shard dirs:", DATA_DIRS)
'''


def pretrain_cells(size: str) -> list[str]:
    return [
        SETUP, GPU_GUARD, DATA_INPUTS,
        f'''\
# --- Pretraining session: resume from the other ping-pong kernel's output, stop cleanly before 12 h ---
prev = inputs_with("run/ckpt.pt", "vishalam-pretrain-{size}-")
resume = f"--resume-from {{prev[0]}}/run" if prev else ""
print("resuming from:", prev or "scratch")
n = torch.cuda.device_count()
launcher = f"torchrun --standalone --nproc_per_node={{n}}" if n > 1 else "python"
get_ipython().system(f"{{launcher}} -m vishalam.train.pretrain --config configs/vishalam-{size}.yaml "
                     f"--data-dirs {{','.join(DATA_DIRS)}} --tokenizer {{TOKENIZER}} {{resume}} "
                     f"--out {{WORK}}/run --max-hours {{hours_left():.2f}}")
''',
        '''\
# --- Loss curve so far ---
import pandas as pd
df = pd.read_csv(f"{WORK}/run/log.csv")
ax = df.plot(x="tokens", y="loss", logy=True, figsize=(9, 4))
v = df.dropna(subset=["val_loss"])
if len(v): v.plot(x="tokens", y="val_loss", ax=ax, style="o-")
df.tail()
''',
    ]


TPU_GUARD = '''\
# --- TPU guard: CLI pushes get a plain CPU machine; the real run is started from the editor with
#     Accelerator = TPU v5e-8. Stop at once if there is no TPU (checked in a subprocess so this
#     notebook never holds the chips that the training processes need).
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
probe = subprocess.run([sys.executable, "-c", "import torch_xla, torch_xla.core.xla_model as xm; "
                        "print(xm.xla_device_hw(torch_xla.device()))"], capture_output=True, text=True)
print("accelerator:", probe.stdout.strip() or probe.stderr.strip()[-300:])
if "TPU" not in probe.stdout:
    raise SystemExit("No TPU. Open this notebook in the Kaggle editor, Accelerator -> TPU v5e-8, "
                     "then Save Version -> Save & Run All.")
'''


def pretrain_tpu_cells(size: str) -> list[str]:
    return [
        SETUP, TPU_GUARD, DATA_INPUTS,
        f'''\
# --- Pretraining session on TPU v5e-8 (8 chips, bf16): resume from the latest {size} session
#     (GPU or TPU, checkpoints are interchangeable), stop cleanly before Kaggle's 9 h TPU limit ---
prev = inputs_with("run/ckpt.pt", "vishalam-pretrain-{size}-")
resume = f"--resume-from {{prev[0]}}/run" if prev else ""
print("resuming from:", prev or "scratch")
get_ipython().system(f"python -m vishalam.train.pretrain --xla --config configs/vishalam-{size}.yaml "
                     f"--data-dirs {{','.join(DATA_DIRS)}} --tokenizer {{TOKENIZER}} {{resume}} "
                     f"--out {{WORK}}/run --max-hours {{hours_left(8.3):.2f}} --set micro_batch=4")
''',
    ]


BENCH_350M_CELLS = [
    SETUP, GPU_GUARD, DATA_INPUTS,
    '''\
# --- 350M speed test on 2xT4: real data + real training code, a few settings, ~20 steps each ---
import re, subprocess
SETTINGS = [  # micro batch per GPU, activation checkpointing
    (1, False), (2, True), (2, False), (4, True),
]
summary = []
for micro, ckpt in SETTINGS:
    tag = f"micro={micro} ckpt={ckpt}"
    cmd = (f"torchrun --standalone --nproc_per_node={torch.cuda.device_count()} -m vishalam.train.pretrain "
           f"--config configs/vishalam-350m.yaml --data-dirs {','.join(DATA_DIRS)} --tokenizer {TOKENIZER} "
           f"--out /tmp/bench_{micro}_{int(ckpt)} --stop-at-step 20 --set global_batch_seqs=32 "
           f"--set micro_batch={micro} --set grad_checkpointing={str(ckpt).lower()} --set eval_every=100000")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    steps = [l for l in out.splitlines() if l.startswith("step ")]
    print(f"==== {tag}\\n" + ("\\n".join(steps) or out[-1500:]), flush=True)
    m = re.search(r"([\\d,]+) tok/s mem ([\\d.]+)GB", steps[-1]) if steps else None
    summary.append(f"{tag}: {m.group(1)} tok/s, {m.group(2)} GB" if m else f"{tag}: FAILED (out of memory?)")
print("\\n==== SUMMARY (350M, 2xT4, Muon) ====\\n" + "\\n".join(summary))
''',
]

XLA_CPUTEST_CELLS = [
    SETUP,
    '''\
# --- Test the TPU (PyTorch/XLA) training path on XLA's CPU backend: no TPU queue, no quota ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "--index-url",
                "https://download.pytorch.org/whl/cpu"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_xla==2.8.0"], check=True)
for n in ("1", "2"):  # one simulated chip first, then two (collectives)
    env = {**os.environ, "PJRT_DEVICE": "CPU", "CPU_NUM_DEVICES": n}
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-v", "-p", "no:cacheprovider",
                        "--basetemp", f"/tmp/xla{n}", "tests/test_pretrain_xla.py"],
                       capture_output=True, text=True, env=env)
    print(f"==== CPU_NUM_DEVICES={n}
", r.stdout[-4000:])
    for f in glob.glob(f"/tmp/xla{n}/**/last_failure.txt", recursive=True):
        text = open(f).read()
        print(f"---- {f}
{text[:3000]}
[...]
{text[-6000:]}")
    print(f"XLA TESTS ({n} device) " + ("PASSED" if r.returncode == 0 else "FAILED"))
''',
]

AGENTDATA_CELLS = [
    SETUP,
    '''\
# --- Build the M1 agent dataset (CPU; ~6-8 h for 100k episodes on 4 cores) ---
get_ipython().system(f"python -m vishalam.data.agent.build_m1 --out {WORK}/m1 --n-train 100000 --workers 4")
''',
]

REWORD_CELLS = [
    SETUP, GPU_GUARD,
    '''\
# --- Reword ~40% of requests with Qwen2.5-7B-Instruct-AWQ (Apache-2.0); output tagged ai-reworded ---
get_ipython().system("pip install -q vllm")
m1 = inputs_with("m1/train.jsonl", "vishalam-agent-data")[0] + "/m1"
os.makedirs(f"{WORK}/m1", exist_ok=True)
for f in ["test_seen.jsonl", "test_unseen.jsonl", "stats.json"]:
    shutil.copy(f"{m1}/{f}", f"{WORK}/m1/{f}")
get_ipython().system(f"python -m vishalam.data.agent.reword_qwen --inp {m1}/train.jsonl "
                     f"--out {WORK}/m1/train_mixed.jsonl --fraction 0.4")
''',
]


def sft_cells(size: str) -> list[str]:
    return [
        SETUP, GPU_GUARD,
        f'''\
# --- Agent fine-tuning session (ping-pong resume like pretraining) ---
TOKENIZER = inputs_with("tokenizer.json", "vishalam-data-1")[0] + "/tokenizer.json"
m1 = inputs_with("m1/train_mixed.jsonl", "vishalam-reword")[0] + "/m1"
init = inputs_with("run/model.pt", "vishalam-pretrain-{size}-")[0] + "/run/model.pt"
extra = " ".join(f"--episodes {{d}}/self.jsonl" for d in inputs_with("self.jsonl", "vishalam-eval-m1-{size}"))
prev = inputs_with("sft/ckpt.pt", "vishalam-sft-m1-{size}-")
resume = f"--resume-from {{prev[0]}}/sft" if prev else ""
n = torch.cuda.device_count()
launcher = f"torchrun --standalone --nproc_per_node={{n}}" if n > 1 else "python"
get_ipython().system(f"{{launcher}} -m vishalam.train.sft --config configs/sft-m1-{size}.yaml --init {{init}} "
                     f"--episodes {{m1}}/train_mixed.jsonl {{extra}} --tokenizer {{TOKENIZER}} {{resume}} "
                     f"--out {{WORK}}/sft --max-hours {{hours_left():.2f}}")
''',
    ]


def eval_cells(size: str) -> list[str]:
    return [
        SETUP, GPU_GUARD,
        f'''\
# --- Evaluate the agent (generated seen/unseen, hand-written, HumanEval, MBPP) ---
TOKENIZER = inputs_with("tokenizer.json", "vishalam-data-1")[0] + "/tokenizer.json"
m1 = inputs_with("m1/test_seen.jsonl", "vishalam-reword")[0] + "/m1"
MODEL = inputs_with("sft/model.pt", "vishalam-sft-m1-{size}-")[0] + "/sft/model.pt"
get_ipython().system(f"python -m vishalam.eval.run --weights {{MODEL}} --tokenizer {{TOKENIZER}} "
                     f"--sets test_seen,test_unseen,handwritten,humaneval,mbpp --data {{m1}} "
                     f"--sandbox subprocess --out {{WORK}}/results.json")
''',
        '''\
# --- One expert-iteration round: self-generated, hidden-test-verified episodes for the next SFT ---
get_ipython().system(f"python -m vishalam.train.expert_iter --weights {MODEL} --tokenizer {TOKENIZER} "
                     f"--n-tasks 3000 --samples 4 --out {WORK}/self.jsonl --max-hours {hours_left():.2f}")
''',
    ]


def notebook(cells: list[str]) -> dict:
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def kaggle_kernel(folder: str, title: str, cells: list[str], gpu: bool, tpu: bool = False) -> None:
    d = HERE / folder
    d.mkdir(exist_ok=True)
    (d / "notebook.ipynb").write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    meta = {
        "id": f"{KAGGLE_USER}/{title.lower().replace(' ', '-')}",  # Kaggle wants slug == slugified title
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": tpu,
        "enable_internet": True,
        "dataset_sources": [CODE_DATASET],
        "competition_sources": [],
        "kernel_sources": [],  # filled per push by tools/kaggle_driver.py
        "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"  # T4 x2
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main() -> None:
    import shutil

    for old in HERE.iterdir():  # regenerate from scratch (removes kernels of the old HF design)
        if old.is_dir():
            shutil.rmtree(old)
    for stale in HERE.glob("*.ipynb"):
        stale.unlink()
    kaggle_kernel("vishalam_data_1", "vishalam data 1", DATA1_CELLS, gpu=False)
    kaggle_kernel("vishalam_data_2", "vishalam data 2", DATA2_CELLS, gpu=False)
    for size in ["30m", "125m", "350m"]:
        for side in "ab":
            kaggle_kernel(f"vishalam_pretrain_{size}_{side}", f"vishalam pretrain {size} {side}",
                          pretrain_cells(size), gpu=True)
    for side in "ab":  # 350M on Kaggle's TPU v5e-8 (started from the editor; see TPU_GUARD)
        kaggle_kernel(f"vishalam_pretrain_350m_tpu_{side}", f"vishalam pretrain 350m tpu {side}",
                      pretrain_tpu_cells("350m"), gpu=False, tpu=True)
    # A/B pilot: same data/tokens/schedule as vishalam-pretrain-30m-a, but Muon for hidden matrices
    kaggle_kernel("vishalam_pretrain_30m_muon", "vishalam pretrain 30m muon", pretrain_cells("30m-muon"), gpu=True)
    kaggle_kernel("vishalam_bench_350m", "vishalam bench 350m", BENCH_350M_CELLS, gpu=True)
    kaggle_kernel("vishalam_xla_cputest", "vishalam xla cputest", XLA_CPUTEST_CELLS, gpu=False)
    kaggle_kernel("vishalam_agent_data", "vishalam agent data", AGENTDATA_CELLS, gpu=False)
    kaggle_kernel("vishalam_reword", "vishalam reword", REWORD_CELLS, gpu=True)
    for size in ["125m", "350m"]:
        for side in "ab":
            kaggle_kernel(f"vishalam_sft_m1_{size}_{side}", f"vishalam sft m1 {size} {side}", sft_cells(size), gpu=True)
        kaggle_kernel(f"vishalam_eval_m1_{size}", f"vishalam eval m1 {size}", eval_cells(size), gpu=True)
    print("notebooks written")


if __name__ == "__main__":
    main()
