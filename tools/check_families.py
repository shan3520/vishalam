"""Check that task families really produce training episodes (sandbox runs, bugs caught and fixed).

    python tools/check_families.py                 # every family
    python tools/check_families.py dict_ matrix_   # families whose module name starts with these

For each family: try a few "create" and "fix" episodes and report how many came out clean. A family
whose create episodes never succeed is broken; a low fix rate only means its code mutates poorly.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vishalam.agent.sandbox import SubprocessSandbox  # noqa: E402
from vishalam.data.agent.families import ALL, make_instance  # noqa: E402
from vishalam.data.agent.phrasing import make_wrap  # noqa: E402
from vishalam.data.agent.trajectories import synthesize  # noqa: E402


def main() -> None:
    prefixes = sys.argv[1:]
    fams = [f for f in ALL if not prefixes or any(type(f.gen).__module__ and f.gen.__module__.split(".")[-1].startswith(p)
                                                    for p in prefixes)]
    sb, wrap, bad = SubprocessSandbox(timeout=10), make_wrap("train"), []
    for fam in fams:
        rng = random.Random(7)
        ok = {"create": 0, "fix": 0}
        tries = {"create": 4, "fix": 4}
        for kind, n in tries.items():
            for _ in range(n):
                ep = synthesize(make_instance(fam, rng, "train", wrap=wrap), rng, kind, sb, "train")
                ok[kind] += ep is not None
        flag = "" if ok["create"] else "   <-- BROKEN"
        if not ok["create"]:
            bad.append(fam.id)
        print(f"{fam.id:28s} create {ok['create']}/{tries['create']}  fix {ok['fix']}/{tries['fix']}{flag}", flush=True)
    print(f"\n{len(fams)} families checked, {len(bad)} broken: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
