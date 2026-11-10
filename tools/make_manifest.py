"""Write or check a SHA-256 manifest for this repository.

Usage
    python tools/make_manifest.py                  # write MANIFEST.sha256
    python tools/make_manifest.py --check data/    # verify against data/SHA256SUMS

The manifest records, per file: SHA-256 (first 16 hex digits), size in bytes,
modification time and relative path.
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
SKIP_NAMES = {"MANIFEST.sha256"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(base: Path, skip_names=SKIP_NAMES):
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in skip_names:
            continue
        if p.suffix == ".png" and p.parent.name == "figures":
            continue
        yield p


def write_manifest() -> None:
    lines = []
    for p in walk(ROOT):
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(
            f"{digest(p)[:16]}  {st.st_size:>10}  {mtime}  {p.relative_to(ROOT).as_posix()}"
        )
    out = ROOT / "MANIFEST.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.name}: {len(lines)} files")


def check(subdir: str) -> None:
    base = ROOT / subdir
    sums = base / "SHA256SUMS"
    if not sums.exists():
        print(f"{sums} not found")
        return
    bad = missing = ok = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        expected, rel = parts[0], parts[-1]
        target = base / rel
        if not target.exists():
            print(f"MISSING  {rel}")
            missing += 1
        elif digest(target)[: len(expected)] != expected.lower():
            print(f"MISMATCH {rel}")
            bad += 1
        else:
            ok += 1
    print(f"{subdir}: {ok} ok, {bad} mismatched, {missing} missing")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        check(sys.argv[2])
    else:
        write_manifest()
