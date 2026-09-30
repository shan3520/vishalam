"""Raw pretraining sources, read file-by-file from Hugging Face (resumable, low disk use).

- Python: bigcode/the-stack-dedup, data/python (gated: accept the terms on HF first).
  Each file carries its repo's licenses; we keep only MIT / Apache-2.0 / BSD-2 / BSD-3.
- English: HuggingFaceFW/fineweb-edu, sample/10BT (ODC-BY).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

STACK_REPO = "bigcode/the-stack-dedup"
STACK_PREFIX = "data/python/"
STACK_COLUMNS = ["content", "max_stars_repo_licenses", "max_line_length", "avg_line_length",
                 "alphanum_fraction", "max_stars_repo_name", "max_stars_repo_path"]

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_PREFIX = "sample/10BT/"
FINEWEB_COLUMNS = ["text", "url"]


def list_parquet(repo: str, prefix: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def iter_parquet_rows(repo: str, filename: str, columns: list[str], cache_dir: str | Path,
                      skip_rows: int = 0, batch_size: int = 2000) -> Iterator[dict]:
    """Download one parquet file, stream its rows, then delete it."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=cache_dir)
    try:
        pf = pq.ParquetFile(path)
        cols = [c for c in columns if c in pf.schema_arrow.names]
        seen = 0
        for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
            if seen + batch.num_rows <= skip_rows:
                seen += batch.num_rows
                continue
            rows = batch.to_pylist()
            start = max(0, skip_rows - seen)
            seen += batch.num_rows
            yield from rows[start:]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
