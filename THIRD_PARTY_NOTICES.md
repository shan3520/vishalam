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

## Models used during data creation

**Qwen2.5-7B-Instruct (AWQ build), `Qwen/Qwen2.5-7B-Instruct-AWQ`**: used only to reword a disclosed
share of training requests. It is not part of the Vishalam model.
- License: Apache License 2.0. Copyright Alibaba Cloud / Qwen team.

## Evaluation benchmarks (never trained on; overlap removed from training data)

- **HumanEval, `openai/openai_humaneval`**: MIT License. Copyright (c) 2021 OpenAI.
- **MBPP, `google-research-datasets/mbpp`**: Creative Commons Attribution 4.0 (CC-BY-4.0). Google
  Research.

## Software

- PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), regex (Apache-2.0), PyYAML (MIT), tqdm (MIT/MPL-2.0)
- Hugging Face `datasets` and `huggingface_hub` (Apache-2.0), pyarrow (Apache-2.0)
- FastAPI (MIT), uvicorn (BSD-3-Clause), pytest (MIT), vLLM (Apache-2.0)
