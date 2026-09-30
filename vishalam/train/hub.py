"""Checkpoint sync with a private Hugging Face repo, so runs move freely between Kaggle and Colab.

The token is read from the HF_TOKEN environment variable (set it from Kaggle/Colab Secrets);
it is never written to disk or logs by this code.
"""
from __future__ import annotations

import threading
from pathlib import Path


class HubSync:
    def __init__(self, repo_id: str, repo_type: str = "model"):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi()
        self.api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    def push(self, local: str | Path, path_in_repo: str, message: str = "update") -> None:
        self.api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=message,
        )

    def squash(self) -> None:
        """Drop git history so overwritten checkpoints stop counting toward the ~100 GB private
        storage of a free HF account. The current files (latest + final weights) are kept."""
        self.api.super_squash_history(self.repo_id, repo_type=self.repo_type,
                                      commit_message="squash: keep only current checkpoint files")

    def push_async(self, files: list[tuple[Path, str]], message: str, squash: bool = False) -> None:
        """Upload in a background thread; waits for any previous upload first."""
        self.wait()

        def run():
            try:
                for local, remote in files:
                    self.push(local, remote, message)
                if squash:
                    self.squash()
            except Exception as e:  # keep training alive; report on next wait()
                self.last_error = e

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self.last_error is not None:
            print(f"[hub] WARNING: previous upload failed: {self.last_error!r}", flush=True)
            self.last_error = None

    def pull(self, path_in_repo: str, local_dir: str | Path) -> Path | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        try:
            return Path(
                hf_hub_download(self.repo_id, path_in_repo, repo_type=self.repo_type, local_dir=local_dir)
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
