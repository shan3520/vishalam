"""Drive Vishalam's Kaggle pipeline from this laptop (uses the Kaggle CLI's existing login).

    python tools/kaggle_driver.py push-code                    # repo HEAD -> private dataset vishalam-code
    python tools/kaggle_driver.py push vishalam_data_1          # push a kernel (starts a run)
    python tools/kaggle_driver.py push vishalam_data_2 --sources vishalam-data-1
    python tools/kaggle_driver.py status vishalam-data-1
    python tools/kaggle_driver.py wait vishalam-data-1
    python tools/kaggle_driver.py logs vishalam-data-1          # key log lines, secrets redacted
    python tools/kaggle_driver.py chain 30m --sessions 1        # ping-pong pretraining sessions

Notes
  * Runs started here never receive Kaggle Secrets. The two data kernels need HF_TOKEN, so after
    pushing them the user starts them once from the Kaggle editor (Save Version -> Save & Run All).
  * All other kernels need no secrets and are started and chained from here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
BUILD = ROOT / "runs" / "kaggle_build"
SECRET_RX = re.compile(r"(hf_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}|gh[pous]_[A-Za-z0-9]{10,})")
KEY_LINES = ("ready:", "HF_TOKEN", "GPUs:", "shard dirs:", "resuming", "resumed", "unique chunks", "saved ",
             "python files", "unit ", "time budget", "step ", "finished", "paused", "Error", "error:",
             "SystemExit", "Traceback", "vocab", "tokens", "passed", "pass so far", "M1 bar", "kept")


def kaggle_exe() -> str:
    local = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    return str(local) if local.exists() else (shutil.which("kaggle") or "kaggle")


def user() -> str:
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def kaggle(*args: str, check: bool = True) -> str:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # Windows 'charmap' crash on logs
    r = subprocess.run([kaggle_exe(), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env)
    out = SECRET_RX.sub("[REDACTED]", r.stdout + r.stderr)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def slug_of(name: str) -> str:
    """Folder name (vishalam_data_1) or slug (vishalam-data-1) -> slug."""
    return name.replace("_", "-")


# ------------------------------------------------------------------ code dataset
def push_code(message: str | None = None) -> None:
    d = BUILD / "code"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("WARNING: uncommitted changes are NOT included (the dataset is built from HEAD).")
    tar = d / "vishalam_code.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(tar), "HEAD"], cwd=ROOT, check=True)
    # add a VERSION file so every Kaggle log says which commit it ran
    import tarfile, io, gzip
    raw = gzip.decompress(tar.read_bytes())
    buf = io.BytesIO(raw)
    with tarfile.open(fileobj=buf, mode="a") as t:
        data = f"{sha}\n".encode()
        info = tarfile.TarInfo("VERSION")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    tar.write_bytes(gzip.compress(buf.getvalue()))
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "vishalam-code", "id": f"{user()}/vishalam-code", "licenses": [{"name": "other"}]}))
    listing = kaggle("datasets", "list", "--mine", "--search", "vishalam-code", check=False)
    if f"{user()}/vishalam-code" in listing:
        print(kaggle("datasets", "version", "-p", str(d), "-m", message or f"code {sha}"))
    else:
        print(kaggle("datasets", "create", "-p", str(d)))  # private by default
    for _ in range(60):
        st = kaggle("datasets", "status", f"{user()}/vishalam-code", check=False).strip()
        if "ready" in st.lower():
            print(f"vishalam-code ready ({sha})")
            return
        time.sleep(10)
    print("dataset still processing; kernels pushed now may see the previous version")


# ------------------------------------------------------------------ kernels
def push(folder: str, sources: list[str]) -> str:
    src = NOTEBOOKS / folder.replace("-", "_")
    if not src.exists():
        raise SystemExit(f"no notebook folder {src} (run python notebooks/make_notebooks.py)")
    d = BUILD / "kernels" / src.name
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d)
    meta = json.loads((d / "kernel-metadata.json").read_text())
    meta["kernel_sources"] = [s if "/" in s else f"{user()}/{slug_of(s)}" for s in sources]
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(kaggle("kernels", "push", "-p", str(d)).strip())
    return meta["id"]


def status(slug: str) -> str:
    out = kaggle("kernels", "status", f"{user()}/{slug_of(slug)}", check=False)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)', out)
    return m.group(1).upper() if m else out.strip()


def wait(slug: str, poll: int = 120, timeout_h: float = 12.5) -> str:
    t0 = time.time()
    time.sleep(20)
    while True:
        st = status(slug)
        if st not in ("RUNNING", "QUEUED", "NEW"):
            return st
        if (time.time() - t0) / 3600 > timeout_h:
            return "TIMEOUT"
        time.sleep(poll)


def log_text(slug: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("kernels", "output", f"{user()}/{slug_of(slug)}", "-p", tmp, "--file-pattern", r".*\.log$", check=False)
        logs = list(Path(tmp).glob("*.log"))
        if not logs:
            return ""
        raw = logs[0].read_text(encoding="utf-8", errors="replace")
    try:
        text = "".join(item.get("data", "") for item in json.loads(raw))
    except (ValueError, AttributeError):
        text = raw
    return SECRET_RX.sub("[REDACTED]", text)


def logs(slug: str, tail: int = 60) -> str:
    lines = [l for l in log_text(slug).splitlines() if any(k in l for k in KEY_LINES) and "Warning" not in l]
    return "\n".join(lines[-tail:])


# ------------------------------------------------------------------ ping-pong chains
DATA_SOURCES = ["vishalam-data-1", "vishalam-data-2"]


def chain(size: str, sessions: int) -> None:
    """Alternate vishalam-pretrain-<size>-a / -b; each session resumes from the other's output."""
    state_path = BUILD / f"chain_{size}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"last": None}
    for _ in range(sessions):
        side = "b" if state["last"] == "a" else "a"
        me, other = f"vishalam-pretrain-{size}-{side}", state["last"] and f"vishalam-pretrain-{size}-{state['last']}"
        sources = DATA_SOURCES + ([other] if other else [])
        print(f"== session on {me} (resume from {other or 'scratch'})")
        push(f"vishalam_pretrain_{size}_{side}", sources)
        st = wait(me)
        print(f"{me}: {st}\n{logs(me, 25)}")
        if st != "COMPLETE":
            print("session did not complete cleanly - stopping the chain (check the log above)")
            return
        state["last"] = side
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state))
        if "finished" in logs(me, 5):
            print(f"{size} pretraining finished")
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push-code").add_argument("-m", "--message", default=None)
    p = sub.add_parser("push")
    p.add_argument("folder")
    p.add_argument("--sources", default="", help="comma list of kernel slugs whose outputs to attach")
    for name in ("status", "wait", "logs"):
        sub.add_parser(name).add_argument("slug")
    c = sub.add_parser("chain")
    c.add_argument("size", choices=["30m", "125m", "350m"])
    c.add_argument("--sessions", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "push-code":
        push_code(a.message)
    elif a.cmd == "push":
        push(a.folder, [s for s in a.sources.split(",") if s])
    elif a.cmd == "status":
        print(status(a.slug))
    elif a.cmd == "wait":
        print(wait(a.slug))
    elif a.cmd == "logs":
        print(logs(a.slug))
    elif a.cmd == "chain":
        chain(a.size, a.sessions)


if __name__ == "__main__":
    main()
