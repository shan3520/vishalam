# Third-party notices

Vishalam's code, weights and generated datasets are proprietary (see `LICENSE`). Building them used
the following third-party materials, which remain under their own licenses and terms.

## Training data

**The Stack (deduplicated), `bigcode/the-stack-dedup`**: Python subset, used for pretraining.
- Used under the BigCode terms of use for The Stack, which require respecting the license of each
  source file.
- Only files from repositories licensed **MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause** were used.
  Those licenses permit this use and require attribution, which this notice provides.
- Per-file repository names and licenses are preserved in the source dataset. Vishalam does not
  redistribute the dataset or its files.
- If a repository owner opts out of The Stack, future versions of Vishalam will be trained on the
  updated dataset, as the terms require.

**FineWeb-Edu, `HuggingFaceFW/fineweb-edu`** (sample-10BT): English web text used for about 15% of
pretraining tokens.
- License: Open Data Commons Attribution License (ODC-BY) v1.0. Also subject to the Common Crawl
  terms of use.

**MBPP, `google-research-datasets/mbpp`** (Austin et al., 2021), train, validation and prompt splits
(474 tasks): turned into agent fine-tuning episodes (`vishalam/data/agent/mbpp_episodes.py`). The
500-task test split is not trained on; it is an evaluation benchmark (below).
- License: Creative Commons Attribution 4.0 (CC-BY-4.0). Google Research.

## Models used during data creation

**Qwen2.5-7B-Instruct (AWQ build), `Qwen/Qwen2.5-7B-Instruct-AWQ`**: used during data creation to
(1) reword a disclosed share of training requests and (2) write candidate programming tasks with
reference solutions (`vishalam/data/agent/gen_tasks.py`); a candidate is kept only if an independent
second solution agrees with the reference on every test input when run in a sandbox. Generated
episodes are tagged `ai-generated`. Qwen is not part of the Vishalam model and no Qwen weights are
used or distilled; only verified task text and code enter the fine-tuning data. The published weights
(`weights/vishalam-350m-agent.pt`) used only (1); the generated tasks of (2) were an experiment whose
model is not published.
- License: Apache License 2.0. Copyright Alibaba Cloud / Qwen team.

## Evaluation benchmarks (never trained on; overlap removed from training data)

- **HumanEval, `openai/openai_humaneval`**: MIT License. Copyright (c) 2021 OpenAI.
- **MBPP test split, `google-research-datasets/mbpp`**: Creative Commons Attribution 4.0
  (CC-BY-4.0). Google Research. (Its other splits are training data, above.)

## Software

- PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), regex (Apache-2.0), PyYAML (MIT), tqdm (MIT/MPL-2.0)
- Hugging Face `datasets` and `huggingface_hub` (Apache-2.0), pyarrow (Apache-2.0)
- FastAPI (MIT), uvicorn (BSD-3-Clause), pytest (MIT), vLLM (Apache-2.0)
