"""Agent tools operating on a workspace directory. Every tool returns plain text for the model."""
from __future__ import annotations

import re
from pathlib import Path

from vishalam.agent.format import ToolCall
from vishalam.agent.sandbox import SubprocessSandbox

MAX_FILE_CHARS = 12_000
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}


class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root: str | Path, sandbox=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox or SubprocessSandbox()

    # ---------- helpers ----------
    def _resolve(self, path: str) -> Path:
        path = path.strip()
        if not path:
            raise ToolError("path is empty")
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path {path!r} is outside the workspace")
        return p

    def files(self) -> list[str]:
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(self.root).parts):
                out.append(p.relative_to(self.root).as_posix())
        return out

    def snapshot(self) -> dict[str, str]:
        return {f: (self.root / f).read_text(encoding="utf-8", errors="replace") for f in self.files()}

    # ---------- tools ----------
    def list_files(self) -> str:
        fs = self.files()
        if not fs:
            return "(workspace is empty)"
        lines = []
        for f in fs:
            n = len((self.root / f).read_text(encoding="utf-8", errors="replace").splitlines())
            lines.append(f"{f} ({n} lines)")
        return "\n".join(lines)

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"file not found: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n... [file truncated]"
        return text

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if p.suffix not in {".py", ".txt", ".md", ".json", ".cfg", ".toml", ".ini", ".csv"}:
            raise ToolError(f"refusing to write {path}: unsupported file type")
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        if content and not content.endswith("\n"):
            content += "\n"
        p.write_text(content, encoding="utf-8")
        n = len(content.splitlines())
        return f"{'overwrote' if existed else 'wrote'} {path} ({n} lines)"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"file not found: {path}")
        text = p.read_text(encoding="utf-8")
        if not old:
            raise ToolError("old text is empty")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"old text not found in {path}")
        if count > 1:
            raise ToolError(f"old text appears {count} times in {path}; include more surrounding lines")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {path}"

    def search(self, pattern: str) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"bad regex: {e}")
        hits = []
        for f in self.files():
            for i, line in enumerate((self.root / f).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f}:{i}: {line.strip()}")
                    if len(hits) >= 50:
                        return "\n".join(hits) + "\n... (more matches)"
        return "\n".join(hits) if hits else "no matches"

    def run_tests(self) -> str:
        if not any(Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py") for f in self.files()):
            return "no test files found (name them test_*.py)"
        return self.sandbox.run_tests(self.root).output or "(no output)"

    def run_python(self, code: str) -> str:
        return self.sandbox.run_python(self.root, code).output or "(no output)"

    # ---------- dispatch ----------
    def execute(self, call: ToolCall) -> str:
        a = call.args
        try:
            match call.name:
                case "list_files":
                    return self.list_files()
                case "read_file":
                    return self.read_file(a["path"])
                case "write_file":
                    return self.write_file(a["path"], a["content"])
                case "edit_file":
                    return self.edit_file(a["path"], a["old"], a["new"])
                case "search":
                    return self.search(a["pattern"])
                case "run_tests":
                    return self.run_tests()
                case "run_python":
                    return self.run_python(a["code"])
                case _:
                    raise ToolError(f"unknown tool {call.name}")
        except ToolError as e:
            return f"error: {e}"
