"""Generates the Kaggle/Colab notebooks (thin wrappers around the repo's scripts).

    python notebooks/make_notebooks.py

Each Kaggle notebook lives in its own folder with kernel-metadata.json so it can be launched with
`kaggle kernels push -p notebooks/<folder>`.
"""
from __future__ import annotations

import json
from pathlib import Path

GITHUB_REPO = "shan3520/vishalam"
HF_USER = "shantanu3520"
KAGGLE_USER = "KAGGLE_USERNAME"  # replaced by setup (see README) once kaggle.json exists

HERE = Path(__file__).parent

SETUP = f'''\
# --- Vishalam setup: secrets -> clone private repo -> install ---
import os, subprocess, sys
GITHUB_REPO = "{GITHUB_REPO}"
DATA_REPO = "{HF_USER}/vishalam-data"
CKPT_REPO = "{HF_USER}/vishalam"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except ImportError:
        from google.colab import userdata
        return userdata.get(name)

os.environ["HF_TOKEN"] = secret("HF_TOKEN")
os.chdir(WORK)
if not os.path.isdir("vishalam"):
    r = subprocess.run(["git", "clone", "--depth", "1",
                        f"https://x-access-token:{{secret('GITHUB_TOKEN')}}@github.com/{{GITHUB_REPO}}.git", "vishalam"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("git clone failed (check the GITHUB_TOKEN secret and repo name)")
    subprocess.run(["git", "-C", "vishalam", "remote", "set-url", "origin", f"https://github.com/{{GITHUB_REPO}}.git"], check=True)
else:
    subprocess.run(["git", "-C", "vishalam", "pull", "--ff-only",
                    f"https://x-access-token:{{secret('GITHUB_TOKEN')}}@github.com/{{GITHUB_REPO}}.git"],
                   capture_output=True, text=True)
os.chdir(f"{{WORK}}/vishalam")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("ready:", subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
'''

DATA_CELLS = [
    SETUP,
    '''\
# --- Step 1: tokenizer (only runs once; later sessions download it) ---
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
if api.file_exists(DATA_REPO, "tokenizer.json", repo_type="dataset"):
    hf_hub_download(DATA_REPO, "tokenizer.json", repo_type="dataset", local_dir=WORK)
    print("tokenizer already trained - downloaded")
else:
    get_ipython().system(f"python -m vishalam.data.pretrain.shard sample --out {WORK}/tok_sample.txt --python-mb 850 --fineweb-mb 150")
    get_ipython().system(f"python -m vishalam.tokenizer.bpe_train --input {WORK}/tok_sample.txt --vocab-size 50304 --workers 4 --out {WORK}/tokenizer.json")
    api.upload_file(path_or_fileobj=f"{WORK}/tokenizer.json", path_in_repo="tokenizer.json", repo_id=DATA_REPO, repo_type="dataset")
    os.remove(f"{WORK}/tok_sample.txt")
''',
    '''\
# --- Step 2: token shards (resumable: re-run this notebook until it prints the target is reached) ---
get_ipython().system(f"python -m vishalam.data.pretrain.shard build --tokenizer {WORK}/tokenizer.json --repo {DATA_REPO} --work-dir {WORK}/shards --target-tokens 10.5e9 --max-hours 11 --workers 4")
''',
]


def pretrain_cells(config: str, max_hours: float) -> list[str]:
    return [
        SETUP,
        f'''\
# --- Pretraining (auto-resumes from the HF checkpoint repo; stops cleanly before the session limit) ---
# Never run the same model on Kaggle and Colab at the same time: the heartbeat check will refuse.
import torch
CONFIG = "{config}"
MAX_HOURS = {max_hours}
n = torch.cuda.device_count()
print("GPUs:", n, [torch.cuda.get_device_name(i) for i in range(n)])
launcher = f"torchrun --standalone --nproc_per_node={{n}}" if n > 1 else "python"
get_ipython().system(f"{{launcher}} -m vishalam.train.pretrain --config {{CONFIG}} --data-repo {{DATA_REPO}} "
                     f"--data-dir {{WORK}}/shards --ckpt-repo {{CKPT_REPO}} --out {{WORK}}/run --max-hours {{MAX_HOURS}}")
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


FETCH = '''\
# --- fetch shared files from the private HF data repo ---
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
def fetch(name):
    return hf_hub_download(DATA_REPO, name, repo_type="dataset", local_dir=WORK)
def upload(local, name):
    api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=DATA_REPO, repo_type="dataset")
TOKENIZER = fetch("tokenizer.json")
'''

AGENTDATA_CELLS = [
    SETUP, FETCH,
    '''\
# --- Build the M1 agent dataset (CPU; ~6-8 h for 100k episodes on 4 cores) ---
get_ipython().system(f"python -m vishalam.data.agent.build_m1 --out {WORK}/m1 --n-train 100000 --workers 4")
for name in ["train.jsonl", "test_seen.jsonl", "test_unseen.jsonl", "stats.json"]:
    upload(f"{WORK}/m1/{name}", f"m1/{name}")
print("uploaded m1/*")
''',
]

REWORD_CELLS = [
    SETUP, FETCH,
    '''\
# --- Reword ~40% of requests with Qwen2.5-7B-Instruct-AWQ (Apache-2.0) on one T4 ---
get_ipython().system("pip install -q vllm")
fetch("m1/train.jsonl")
# your hand-written "seed" requests (handwritten/requests.jsonl in the repo) are used as style examples
get_ipython().system(f"python -m vishalam.data.agent.reword_qwen --inp {WORK}/m1/train.jsonl "
                     f"--out {WORK}/m1/train_mixed.jsonl --fraction 0.4")
upload(f"{WORK}/m1/train_mixed.jsonl", "m1/train_mixed.jsonl")
''',
]


def sft_cells(config: str) -> list[str]:
    return [
        SETUP, FETCH,
        f'''\
# --- Agent fine-tuning (auto-resumes from the HF checkpoint repo) ---
import torch
fetch("m1/train_mixed.jsonl")
extra = ""
for name in ["m1/self_round1.jsonl", "m1/self_round2.jsonl"]:  # expert-iteration rounds, if any
    try:
        extra += f" --episodes {{fetch(name)}}"
    except Exception:
        pass
n = torch.cuda.device_count()
launcher = f"torchrun --standalone --nproc_per_node={{n}}" if n > 1 else "python"
get_ipython().system(f"{{launcher}} -m vishalam.train.sft --config {config} --episodes {{WORK}}/m1/train_mixed.jsonl{{extra}} "
                     f"--tokenizer {{TOKENIZER}} --ckpt-repo {{CKPT_REPO}} --out {{WORK}}/sft --max-hours 11.3")
''',
    ]


def eval_cells(model_name: str) -> list[str]:
    return [
        SETUP, FETCH,
        f'''\
# --- Evaluate the agent (generated seen/unseen, hand-written, HumanEval, MBPP) ---
from huggingface_hub import hf_hub_download
MODEL = hf_hub_download(CKPT_REPO, "{model_name}/final/model.pt", local_dir=WORK)
for name in ["m1/test_seen.jsonl", "m1/test_unseen.jsonl"]:
    fetch(name)
get_ipython().system(f"python -m vishalam.eval.run --weights {{MODEL}} --tokenizer {{TOKENIZER}} "
                     f"--sets test_seen,test_unseen,handwritten,humaneval,mbpp --data {{WORK}}/m1 "
                     f"--sandbox subprocess --out {{WORK}}/results_{model_name}.json")
upload(f"{{WORK}}/results_{model_name}.json", "results/{model_name}.json")
''',
        f'''\
# --- Optional: one expert-iteration round (self-improvement data for the next SFT) ---
ROUND = 1
get_ipython().system(f"python -m vishalam.train.expert_iter --weights {{MODEL}} --tokenizer {{TOKENIZER}} "
                     f"--n-tasks 3000 --samples 4 --out {{WORK}}/self_round{{ROUND}}.jsonl --max-hours 10")
upload(f"{{WORK}}/self_round{{ROUND}}.jsonl", f"m1/self_round{{ROUND}}.jsonl")
''',
    ]


def notebook(cells: list[str]) -> dict:
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }


def kaggle_kernel(folder: str, title: str, cells: list[str], gpu: bool) -> None:
    d = HERE / folder
    d.mkdir(exist_ok=True)
    (d / "notebook.ipynb").write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    meta = {
        "id": f"{KAGGLE_USER}/{folder.replace('_', '-')}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"  # T4 x2
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main() -> None:
    kaggle_kernel("kaggle_data", "vishalam data", DATA_CELLS, gpu=False)
    for size in ["30m", "125m", "350m"]:
        kaggle_kernel(f"kaggle_pretrain_{size}", f"vishalam pretrain {size}",
                      pretrain_cells(f"configs/vishalam-{size}.yaml", 11.3), gpu=True)
    kaggle_kernel("kaggle_agentdata", "vishalam agent data m1", AGENTDATA_CELLS, gpu=False)
    kaggle_kernel("kaggle_reword", "vishalam reword m1", REWORD_CELLS, gpu=True)
    for size in ["125m", "350m"]:
        kaggle_kernel(f"kaggle_sft_{size}", f"vishalam sft m1 {size}", sft_cells(f"configs/sft-m1-{size}.yaml"), gpu=True)
        kaggle_kernel(f"kaggle_eval_{size}", f"vishalam eval m1 {size}", eval_cells(f"vishalam-{size}-agent-m1"), gpu=True)
    colab = HERE / "colab_pretrain.ipynb"
    cells = pretrain_cells("configs/vishalam-350m.yaml", 11.0)
    cells[1] = cells[1].replace('CONFIG = "configs/vishalam-350m.yaml"',
                                'CONFIG = "configs/vishalam-350m.yaml"  # change to 30m / 125m as needed')
    colab.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print("notebooks written")


if __name__ == "__main__":
    main()
