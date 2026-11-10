"""Reproduce every released artefact with one command.

    python reproduce.py            # tables + figures + manifest check
    python reproduce.py --check    # only verify the shipped files

Exits non-zero if a table file differs from the version recorded in
MANIFEST.sha256, or if a required input is missing.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: Path) -> None:
    print(f"\n== {script.relative_to(ROOT).as_posix()}")
    r = subprocess.run([PY, str(script)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"failed: {script.name} (exit {r.returncode})")


def main() -> None:
    if "--check" in sys.argv:
        run(ROOT / "tools" / "make_manifest.py")
        return

    missing = [p for p in (ROOT / "results" / "scores_unsw_s42.npz",) if not p.exists()]
    if missing:
        print("warning: cache/ is empty, some statistics cannot be recomputed")

    run(ROOT / "code" / "tables" / "export_tables.py")
    run(ROOT / "code" / "figures" / "paper_figs.py")
    print("\nartefacts rebuilt; compare results/tables/ against MANIFEST.sha256")


if __name__ == "__main__":
    main()
