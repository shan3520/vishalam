"""Splitting the agent-data build over machines gives exactly the single-machine dataset."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["--n-train", "12", "--chunks", "4", "--workers", "2", "--n-test-seen", "2", "--n-test-unseen", "2"]


def build(*args):
    r = subprocess.run([sys.executable, "-m", "vishalam.data.agent.build_m1", *args], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_parts_merge_to_the_single_machine_build(tmp_path):
    build("--out", str(tmp_path / "one"), *SMALL)
    build("--out", str(tmp_path / "p0"), "--part", "0/2", *SMALL)
    out = build("--out", str(tmp_path / "p1"), "--part", "1/2", *SMALL)
    assert "part 1/2: 2 chunks, 0 already done" in out
    assert not (tmp_path / "p1" / "test_seen.jsonl").exists()  # test sets come from part 0 only
    build("--out", str(tmp_path / "merged"), "--merge", f"{tmp_path / 'p0'},{tmp_path / 'p1'}")
    for name in ("train.jsonl", "test_seen.jsonl", "test_unseen.jsonl", "stats.json"):
        assert (tmp_path / "merged" / name).read_text() == (tmp_path / "one" / name).read_text(), name
    assert len((tmp_path / "one" / "train.jsonl").read_text().splitlines()) == 12


def test_interrupted_part_resumes(tmp_path):
    build("--out", str(tmp_path / "full"), "--part", "1/2", *SMALL)
    first = tmp_path / "full" / "train_parts" / "chunk_00001.jsonl"
    # an interrupted session: only one of its two chunks was finished
    (tmp_path / "cut" / "train_parts").mkdir(parents=True)
    (tmp_path / "cut" / "train_parts" / first.name).write_bytes(first.read_bytes())
    out = build("--out", str(tmp_path / "resumed"), "--part", "1/2", "--resume-from", str(tmp_path / "cut"), *SMALL)
    assert "2 chunks, 1 already done" in out
    for f in (tmp_path / "full" / "train_parts").glob("chunk_*.jsonl"):
        assert (tmp_path / "resumed" / "train_parts" / f.name).read_text() == f.read_text()
