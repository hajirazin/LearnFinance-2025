#!/usr/bin/env python3
"""QuantBMAD repo-green gate for qb-implement.

Runs (in order) and fails fast on first non-zero exit:
  1. ruff check + ruff format --check on brain_api, temporal, _bmad/quantbmad
  2. brain_api pytest (uv run)
  3. temporal pytest (uv run)

Usage:
  python3 _bmad/quantbmad/scripts/check_repo_green.py
  python3 _bmad/quantbmad/scripts/check_repo_green.py --phase pre   # before implement starts
  python3 _bmad/quantbmad/scripts/check_repo_green.py --phase post  # before marking implement done

Exit 0 = all green. Exit 1 = failure. Exit 2 = usage/env error.

Agents MUST NOT skip failures as \"unrelated\". Fix them, then re-run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_QUANTBMAD_DIR = _SCRIPTS_DIR.parent
_BMAD_DIR = _QUANTBMAD_DIR.parent
_ROOT = _BMAD_DIR.parent


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"+ ({cwd.name}) {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return int(proc.returncode)


def _ruff_bin(root: Path) -> str | None:
    venv_ruff = root / ".venv" / "bin" / "ruff"
    if venv_ruff.is_file():
        return str(venv_ruff)
    return shutil.which("ruff")


def run_ruff(root: Path) -> int:
    ruff = _ruff_bin(root)
    if not ruff:
        print("ERROR: ruff not found (.venv/bin/ruff or PATH)", file=sys.stderr)
        return 2
    targets = ["brain_api", "temporal", "_bmad/quantbmad"]
    rc = _run([ruff, "check", *targets], root)
    if rc != 0:
        return rc
    return _run([ruff, "format", "--check", *targets], root)


def run_brain_tests(root: Path) -> int:
    brain = root / "brain_api"
    uv = shutil.which("uv")
    if not uv:
        print("ERROR: uv not found on PATH", file=sys.stderr)
        return 2
    return _run([uv, "run", "pytest", "-q", "--tb=line"], brain)


def run_temporal_tests(root: Path) -> int:
    temporal = root / "temporal"
    uv = shutil.which("uv")
    if not uv:
        print("ERROR: uv not found on PATH", file=sys.stderr)
        return 2
    return _run([uv, "run", "pytest", "-q", "--tb=line"], temporal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Require ruff + brain_api + temporal tests all green."
    )
    parser.add_argument(
        "--phase",
        choices=("pre", "post", "any"),
        default="any",
        help="Label only (same checks); pre=before start, post=before done",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    root = (args.project_root or _ROOT).resolve()

    print(
        f"QuantBMAD repo-green gate ({args.phase}): "
        "ruff → brain_api pytest → temporal pytest",
        flush=True,
    )
    print(
        "Policy: failing checks cannot be skipped as unrelated — fix them.",
        flush=True,
    )

    for name, fn in (
        ("ruff", run_ruff),
        ("brain_api pytest", run_brain_tests),
        ("temporal pytest", run_temporal_tests),
    ):
        rc = fn(root)
        if rc != 0:
            print(
                f"FAILED: {name} (exit {rc}). "
                f"qb-implement must NOT {'start' if args.phase == 'pre' else 'finish'} "
                "until this is green. Fix related AND unrelated failures.",
                file=sys.stderr,
            )
            return 1 if rc != 2 else 2
        print(f"OK: {name}", flush=True)

    print("OK: repo-green (ruff + brain_api + temporal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
