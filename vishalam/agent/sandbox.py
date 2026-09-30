"""Where agent code runs.

- DockerSandbox (local app, default): network off, memory/CPU/process limits, read-only root and a
  read-only mount of the workspace. Build the image once: docker build -t vishalam-sandbox sandbox/
- SubprocessSandbox: plain subprocess with a timeout (+ rlimits on Linux). Used on disposable
  Kaggle/Colab VMs for data generation and self-improvement runs, and for tests.

Outputs are normalized (no absolute paths, no timings) so identical code gives identical text,
which keeps training data consistent.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PYTEST_ARGS = ["-q", "--tb=short", "-p", "no:cacheprovider", "--no-header", "--color=no"]
_TIMING = re.compile(r" in \d+\.\d+s")


@dataclass
class RunResult:
    output: str
    returncode: int
    timed_out: bool = False


_BANNER = re.compile(r"^[=_]{3,} (.+?) [=_]{3,}$", re.M)
_PROGRESS = re.compile(r"^([.FEsx]+)\s+\[\s*\d+%\]$", re.M)


def normalize(text: str, workspace: Path) -> str:
    ws = str(workspace)
    for variant in {ws, ws.replace("\\", "/"), "/work"}:
        text = text.replace(variant + os.sep, "").replace(variant + "/", "").replace(variant, ".")
    text = _TIMING.sub("", text)
    # compact pytest banners ("===== FAILURES =====" -> "== FAILURES ==") to save context tokens
    text = _BANNER.sub(r"== \1 ==", text)
    text = _PROGRESS.sub(r"\1", text)
    return text.strip()


class SubprocessSandbox:
    def __init__(self, timeout: float = 10.0, python: str | None = None, memory_mb: int = 512):
        self.timeout = timeout
        self.python = python or sys.executable
        self.memory_mb = memory_mb

    def _preexec(self):
        if sys.platform.startswith("linux"):
            import resource

            lim = self.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        return None

    def _run(self, argv: list[str], workspace: Path) -> RunResult:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # needed by Python on Windows
        }
        try:
            p = subprocess.run(
                argv, cwd=workspace, env=env, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self.timeout,
                preexec_fn=self._preexec if sys.platform.startswith("linux") else None,
            )
            return RunResult(normalize(p.stdout + p.stderr, workspace), p.returncode)
        except subprocess.TimeoutExpired:
            return RunResult(f"Timed out after {self.timeout:.0f} seconds.", -1, timed_out=True)

    def run_tests(self, workspace: Path, test_path: str | None = None) -> RunResult:
        argv = [self.python, "-m", "pytest", *PYTEST_ARGS]
        if test_path:
            argv.append(test_path)
        return self._run(argv, workspace)

    def run_python(self, workspace: Path, code: str) -> RunResult:
        return self._run([self.python, "-c", code], workspace)


class DockerSandbox:
    IMAGE = "vishalam-sandbox"

    def __init__(self, timeout: float = 15.0, memory_mb: int = 512):
        if shutil.which("docker") is None:
            raise RuntimeError("Docker not found. Install Docker Desktop, then: docker build -t vishalam-sandbox sandbox/")
        self.timeout = timeout
        self.memory_mb = memory_mb

    def _run(self, inner: list[str], workspace: Path) -> RunResult:
        argv = [
            "docker", "run", "--rm", "--network", "none", "--memory", f"{self.memory_mb}m",
            "--cpus", "1", "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp:size=64m",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--user", "65534:65534",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0",
            "-v", f"{workspace.resolve()}:/work:ro", "-w", "/work", self.IMAGE, *inner,
        ]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=self.timeout + 10)
            return RunResult(normalize(p.stdout + p.stderr, workspace), p.returncode)
        except subprocess.TimeoutExpired:
            return RunResult(f"Timed out after {self.timeout:.0f} seconds.", -1, timed_out=True)

    def run_tests(self, workspace: Path, test_path: str | None = None) -> RunResult:
        inner = ["timeout", str(int(self.timeout)), "python", "-m", "pytest", *PYTEST_ARGS]
        if test_path:
            inner.append(test_path)
        return self._run(inner, workspace)

    def run_python(self, workspace: Path, code: str) -> RunResult:
        return self._run(["timeout", str(int(self.timeout)), "python", "-c", code], workspace)


def default_sandbox():
    """Docker when available (local app), else subprocess (Kaggle/Colab/tests)."""
    if os.environ.get("VISHALAM_SANDBOX", "").lower() == "subprocess" or shutil.which("docker") is None:
        return SubprocessSandbox()
    return DockerSandbox()
