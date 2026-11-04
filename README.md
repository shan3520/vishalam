# Vishalam

**Vishalam is a small coding-agent language model built from scratch**: my own tokenizer, my own
transformer, my own training code, my own agent dataset, and a local web app where the agent writes
tests, writes code, runs the tests in a sandbox and tries to fix its own mistakes.

> **Status: paused after Milestone 1, which is only partly met.** The 350M model is fully trained
> (10B tokens, free Kaggle hardware only). The agent passes **88%** of new tasks from the task families
> it trained on, but only **3.7%** from families it never saw, so it is **not** a general coding
> assistant. The hand-written-request target was never measured. M2 and M3 were not started.
>
> **This source is public so the work can be read and reviewed. It is not open source, and the trained
> weights are not included.** See [License](#license).

## What it does

You type a request such as *"Write a function `count_vowels(text)` that returns how many vowels are in the
string `text`, ignoring case. For example, `count_vowels('Hello')` should return `2`."* The agent then:

1. writes tests based on your examples (`write_file test_count_vowels.py`)
2. writes the implementation
3. runs the tests in a sandbox (`run_tests`) and reads the real pytest output
4. if something fails, reads the error and edits the code (`edit_file`, exact find-and-replace)
5. reruns the tests and finishes with a short summary (`done`)

Tools: `list_files`, `read_file`, `write_file`, `edit_file`, `search`, `run_tests`, `run_python`, `done`.
Tool calls use constrained decoding, so they follow the protocol's grammar.

## Results

All numbers come from `results/summary.json`. "Pass" means the code passes hidden tests within up to
3 fixes. Generated tasks are new instances with request wording held out from training.

| Agent | Seen families (500) | Unseen families (300) | HumanEval (164) | MBPP (500) |
|---|---|---|---|---|
| 125M, requests without examples | 79.2% | 4.0% | 1.2% | 2.2% |
| 350M, requests without examples | 47.8% | 3.7% | 4.9% | 3.6% |
| **350M, requests with examples (best)** | **88.0%** | **3.7%** | **10.4%** | **8.4%** |
| 350M, 7,024 AI-generated tasks | 53.8% | 7.0% | 7.3% | 9.2% |

Rows are not all like for like. The test tasks changed when examples were added to requests, and the
first two rows ran MBPP without its test lines in the prompt (the standard format, used in the lower two).
Benchmark scores on tasks whose function name matches no training family ("clean") are in the summary
file: for the best agent, HumanEval 8.1% and MBPP 7.3%.

**Milestone 1 targets:** at least 70% on seen families (**met**, 88.0%) and at least 40% on hand-written
requests (**not measured**: the requests were not written, so M1 is not complete).

### What the experiments showed

- **Worked examples fixed self-sabotage.** Without examples the agent invented wrong expected values
  for its tests, then "fixed" correct code to match them. On seen families the best agent passes 92.9%
  of requests with examples, against 68.6% without.
- **New kinds of tasks remain the failure.** On unseen families it mostly writes the solution of the
  closest family it memorised, and its one-token repairs rarely help.
- **7,024 generated tasks did not make a better model.** Against the best agent, on the same 1,464 tasks:
  seen families 9 gained / 180 lost, unseen 14 / 4 (weak evidence: 5 of its 6 new "write" passes are one
  family), HumanEval 2 / 7, MBPP 27 / 23. Several things changed at once and about 3 of 14 sampled tasks
  had wrong solutions, so this does not prove that diverse data cannot help.
- **The base model is the limit.** The pretrained 350M alone, with one greedy attempt and no agent,
  passes **9.8%** of HumanEval (16/164). So the fine-tuning does not lose ability, but adds little.
- **Retrying does not rescue it.** Up to 10 sampled attempts (temperature 0.8) reach at most **14.0%**
  (11.4% clean), even when the hidden tests choose the winner. Lower temperatures were not tried.

## Running it

The trained weights are not included, so the steps below run the interface with a scripted agent and no
model. Running them needs Python 3.11 and permission from the author (see [License](#license)).

```bash
git clone https://github.com/shan3520/vishalam.git
cd vishalam
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m app.server --demo
```

Then open http://127.0.0.1:8765. Tests: `.venv\Scripts\python -m pytest` (fast) and `-m slow`
(end-to-end agent).

With the weights in `weights/` (`vishalam-350m-agent.pt` and `tokenizer.json`), `python -m app.server` runs the
real agent. Loading took 1-2 minutes on a laptop, and small tasks then took 20-50 seconds.

Requests work best as a single Python function with one or more worked examples.

Do not use int8 quantization: it makes the 350M much faster but broke the agent when tried
(`quantize_cpu=True` in `vishalam/infer/generate.py` is off by default for that reason).

The app uses a Docker sandbox when Docker is running and the image is built
(`docker build -t vishalam-sandbox sandbox/`). Otherwise it falls back to a plain subprocess with a
timeout, which does **not** isolate the code; the header shows which one is in use. Use Docker for anything
you would not run yourself.

## Built from scratch vs. libraries

| From scratch (this repo) | Libraries used |
|---|---|
| Byte-level BPE tokenizer: trainer + encoder, 50k vocab (`vishalam/tokenizer`) | PyTorch (tensors, autograd, SDPA attention kernel), PyTorch/XLA |
| Llama-style transformer: RMSNorm, RoPE, SwiGLU, KV cache (`vishalam/model`) | Hugging Face `datasets` / `huggingface_hub` (download/upload only) |
| Pretraining + SFT loops: Muon + AdamW, fp16/bf16, DDP, TPU SPMD, resumable sessions (`vishalam/train`) | FastAPI / uvicorn (web server) |
| Agent protocol, constrained decoding, tools, sandbox, runtime (`vishalam/agent`) | pytest (runs the tests) |
| 250 hand-written task families, bug injector, episode synthesizer (`vishalam/data/agent`) | vLLM + Qwen2.5-7B (data creation only; disclosed) |
| Evaluation harness, web UI | |

## Architecture

```
 the-stack-dedup (Python, MIT/Apache/BSD only) ─┐
 FineWeb-Edu (English, ODC-BY) ────────────────┼─> filter + decontaminate ─> own BPE ─> token shards
                                               │
                                               └─> pretraining  30M pilot -> 125M -> 350M   (Kaggle 2xT4 + TPU)
                                                                                   │
 211 task families ─> tasks + injected bugs ─> episodes with REAL tool output ─────┤
 Qwen2.5-7B rewording (about 34% of requests, tagged) ─┘                           │
 worked examples in requests + MBPP train splits ─────────────────────> agent SFT ◄┘
                                                                                   │
                                  eval: generated seen/unseen, HumanEval, MBPP ◄───┤
                                                                                   │
                           local app: FastAPI + live step stream + sandbox ◄───────┘
```

Model sizes (all 4,096-token context, 50,304-token vocabulary):

| name | d_model | layers | heads | FFN | params | tokens |
|---|---|---|---|---|---|---|
| vishalam-30m (pilot) | 384 | 6 | 6 | 1024 | ~30M | 0.5B |
| vishalam-125m | 768 | 12 | 12 | 2048 | ~124M | 2.5B |
| vishalam-350m | 1024 | 24 | 16 | 2816 | ~360M | 10B |

## Data provenance

- **Python code:** `bigcode/the-stack-dedup` (Python subset), keeping only files whose repositories
  are licensed MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause, then quality-filtered.
- **English:** `HuggingFaceFW/fineweb-edu` (sample-10BT), about 15% of pretraining tokens.
- **Decontamination:** every pretraining document sharing a 13-word sequence with HumanEval or MBPP is
  removed.
- **Agent data (best agent):** 100,000 episodes from 211 rule-based task families, plus episodes from
  MBPP's train, validation and prompt splits. Every tool result is the *real* output of running that
  step, and every episode ends with code that passes hidden tests. 39 more families are never trained
  on, so they measure generalization. 87.7% of requests carry worked examples.
- **AI-reworded share:** 34.3% of those requests were rewritten by Qwen2.5-7B-Instruct (Apache-2.0) for
  more natural English, validated and tagged `ai-reworded`.
- **AI-generated tasks (experiment only, not in the best agent):** 7,024 tasks written by Qwen2.5-7B.
  Each was kept only if two independently written solutions agreed on every test input in a sandbox,
  and it passed filters against HumanEval, MBPP-test and the held-out families. Agreement does not catch
  a mistake both solutions share. Tagged `ai-generated` (`vishalam/data/agent/gen_*.py`).
- **Not done:** the planned 300+ hand-written requests were not written, and the self-improvement
  loop (`vishalam/train/expert_iter.py`) is not part of the best agent.

## Training pipeline (Kaggle)

Everything ran on free Kaggle sessions, and no GitHub or Hugging Face upload tokens are involved:
- **Code** reaches Kaggle as the private dataset `vishalam-code`, a tarball of the repo at HEAD,
  uploaded with `python tools/kaggle_driver.py push-code`. Run it after every commit: kernels check it.
- **Every intermediate result** (tokenizer, token shards, checkpoints, agent data) is a kernel's
  `/kaggle/working` output. Later kernels attach it as an input.
- **Long training** alternates between two kernels (`...-a` / `...-b`). Each session resumes from the
  other kernel's output and stops cleanly before Kaggle's time limit. `tools/chain350.sh` prefers the TPU
  and falls back to GPU after a long queue.

| step | kernel(s) | hardware |
|---|---|---|
| 1. tokenizer + token shards | `vishalam-data-{1,2}` | CPU (Kaggle editor, needs the `HF_TOKEN` secret: the-stack is gated) |
| 2. pretraining | `vishalam-pretrain-{30m,125m,350m}-{a,b}`, `vishalam-pretrain-350m-tpu-{a,b}` | 2×T4, TPU v5e-8 |
| 3. agent data | `vishalam-agent-data-p{0..5}` | 6 CPU sessions |
| 4. rewording | `vishalam-reword` | T4 |
| 5. worked examples + test sets | `vishalam-m1-examples` | CPU |
| 6. agent SFT | `vishalam-sft-m1-{125m,350m}-{a,b}` | 2×T4 |
| 7. evaluation | `vishalam-eval-m1-{size}-cpu{0..4}` | 5 CPU sessions (under 3 hours, no GPU quota) |
| experiment: generated tasks | `vishalam-gen-tasks`, `vishalam-gen-tasks-2`, `vishalam-gen-episodes` | 2×T4, CPU |
| experiment: base model alone | `vishalam-eval-base-350m`, `...-k10` | CPU |

**Actual cost of 350M on 10B tokens:** about three weeks of calendar time within Kaggle's weekly quotas
(30 GPU hours, 20 TPU hours). Measured speed: about 42k tokens/s on TPU v5e-8, 8k on 2×T4. TPU sessions
often queued for 8–25 hours.

To reproduce the best agent, run steps 1–7 with `configs/sft-m1-350m.yaml`. Kaggle's CLI only returns a
kernel's latest output, so older versions must be fetched from its website.

## Limitations

- Narrow: single Python functions only, strongest on the 211 families it trained on. On other requests it
  usually fails, and it can finish with wrong code when its own tests are weak.
- Requests without worked examples are weaker, because the agent can then only check that its code runs.
- Even with examples it can add a test of its own with a wrong expected value, then break correct code
  to match it. Check the code it writes rather than relying on "tests pass" or "failed".
- In a final check through the app, 1 of 3 simple requests succeeded: `count_vowels` passed,
  `second_largest` stayed wrong and reported failure, and `count_evens` broke its own correct code.
  Three requests are only an anecdote, but expect failures even on simple tasks.
- Validation losses are not comparable across hardware: GPU sessions report about 0.2 higher than TPU
  sessions (1.70 at step 19,073 vs 1.49 at step 18,500 of the 350M), probably because each scores a
  different sample.
- The subprocess sandbox (no Docker) only enforces a timeout.

## License

Proprietary, all rights reserved. See `LICENSE`. This repository is public for viewing: GitHub's terms let
other users view and fork a public repository on GitHub, and no other permission is granted. To use, copy or
build on this work, ask the author through GitHub. Third-party data and models used in building Vishalam are
credited in `THIRD_PARTY_NOTICES.md`.
