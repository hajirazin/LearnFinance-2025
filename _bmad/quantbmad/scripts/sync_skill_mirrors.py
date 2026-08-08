#!/usr/bin/env python3
"""Literal-copy QuantBMAD SKILL.md files into IDE discovery roots.

No symlinks. Source of truth: _bmad/quantbmad/skills/<name>/SKILL.md
Targets: .agents/skills, .cursor/skills, .gemini/skills

Usage:
  python3 _bmad/quantbmad/scripts/sync_skill_mirrors.py
  python3 _bmad/quantbmad/scripts/sync_skill_mirrors.py --check  # exit 1 if drift
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# Allow importing sibling package when run as a script.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_QUANTBMAD_DIR = _SCRIPTS_DIR.parent
if str(_QUANTBMAD_DIR) not in sys.path:
    sys.path.insert(0, str(_QUANTBMAD_DIR.parent))

from quantbmad.research_globs import (  # noqa: E402
    IDE_SKILL_ROOTS,
    SKILL_NAMES,
    find_project_root,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_skill_path(project_root: Path, skill_name: str) -> Path:
    return project_root / "_bmad" / "quantbmad" / "skills" / skill_name / "SKILL.md"


def mirror_skill_path(project_root: Path, ide_root: str, skill_name: str) -> Path:
    return project_root / ide_root / skill_name / "SKILL.md"


def sync_mirrors(project_root: Path) -> list[Path]:
    """Copy canonical skills to all IDE roots. Returns written paths."""
    written: list[Path] = []
    for skill_name in SKILL_NAMES:
        src = canonical_skill_path(project_root, skill_name)
        if not src.is_file():
            raise FileNotFoundError(f"Missing canonical skill: {src}")
        for ide_root in IDE_SKILL_ROOTS:
            dest = mirror_skill_path(project_root, ide_root, skill_name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                dest.unlink()
            shutil.copy2(src, dest)
            written.append(dest)
    return written


def check_mirrors(project_root: Path) -> list[str]:
    """Return human-readable mismatch messages (empty if OK)."""
    errors: list[str] = []
    for skill_name in SKILL_NAMES:
        src = canonical_skill_path(project_root, skill_name)
        if not src.is_file():
            errors.append(f"missing canonical: {src}")
            continue
        expected = _sha256(src)
        for ide_root in IDE_SKILL_ROOTS:
            dest = mirror_skill_path(project_root, ide_root, skill_name)
            if dest.is_symlink():
                errors.append(f"symlink not allowed: {dest}")
                continue
            if not dest.is_file():
                errors.append(f"missing mirror: {dest}")
                continue
            actual = _sha256(dest)
            if actual != expected:
                errors.append(
                    f"checksum mismatch: {dest} != {src} "
                    f"(canonical={expected[:12]} mirror={actual[:12]})"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync or verify QuantBMAD skill mirrors (literal copies)."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify mirrors match canonical; do not write",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root (default: auto-detect)",
    )
    args = parser.parse_args(argv)
    root = (args.project_root or find_project_root()).resolve()

    if args.check:
        errors = check_mirrors(root)
        if errors:
            print("QuantBMAD skill mirror check FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print("OK: all QuantBMAD skill mirrors match canonical (no symlinks)")
        return 0

    written = sync_mirrors(root)
    for path in written:
        print(f"synced {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
