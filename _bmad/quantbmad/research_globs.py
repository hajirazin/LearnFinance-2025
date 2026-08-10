#!/usr/bin/env python3
"""Single source of truth for QuantBMAD Research-track path matching.

Skills and CI/pre-commit import this module. Editing this file itself is a
Go-live/Compliance change (see QUANTBMAD_RESEARCH_GLOBS self-entry).

CLI:
  python3 _bmad/quantbmad/research_globs.py --check <path> [<path> ...]
  Exit 0 = no Research match; exit 1 = at least one path is Research-tracked;
  exit 2 = usage error.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

QUANTBMAD_RESEARCH_GLOBS: list[str] = [
    "brain_api/brain_api/core/**",
    "brain_api/brain_api/universe/**",
    "brain_api/brain_api/storage/ibkr_orders.py",
    "brain_api/brain_api/routes/orders.py",
    "**/sticky*.py",
    "**/strategy_partitions.py",
    "**/promotion*.py",
    "**/hrp.py",
    "**/broker_costs.py",
    "**/rewards.py",
    "temporal/workflows/**",
    "_bmad/quantbmad/research_globs.py",
]

SKILL_NAMES: tuple[str, ...] = (
    "qb-agent-researcher",
    "qb-agent-pm",
    "qb-agent-ml",
    "qb-agent-risk",
    "qb-agent-skeptic",
    "qb-agent-validation",
    "qb-agent-dev",
    "qb-agent-architect",
)

PARTY_MEMBER_CODES: tuple[str, ...] = (
    "q-researcher",
    "q-pm",
    "q-ml",
    "q-risk",
    "q-skeptic",
    "q-validation",
    "q-dev",
    "q-architect",
)
IDE_SKILL_ROOTS: tuple[str, ...] = (
    ".agents/skills",
    ".cursor/skills",
    ".gemini/skills",
)


def _normalize(path: str | Path) -> str:
    """Normalize to forward-slash relative-style path for glob matching."""
    text = str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def is_research_path(path: str | Path) -> bool:
    """Return True if path matches any QUANTBMAD_RESEARCH_GLOBS entry."""
    normalized = _normalize(path)
    for pattern in QUANTBMAD_RESEARCH_GLOBS:
        # fnmatch does not treat ** as recursive; expand common cases.
        if _glob_match(normalized, pattern):
            return True
    return False


def _glob_match(path: str, pattern: str) -> bool:
    """Match path against a glob that may contain **."""
    pattern = pattern.replace("\\", "/")
    if "**" not in pattern:
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            Path(path).name, pattern
        )

    # Split on /** or **/ forms into prefix/suffix style matching.
    # Examples:
    #   brain_api/brain_api/core/**  -> path startswith prefix or equals prefix
    #   **/sticky*.py                -> any path ending matching sticky*.py
    #   temporal/workflows/**        -> under that directory
    if pattern.startswith("**/"):
        suffix = pattern[3:]
        parts = path.split("/")
        for i in range(len(parts)):
            candidate = "/".join(parts[i:])
            if fnmatch.fnmatch(candidate, suffix) or fnmatch.fnmatch(parts[-1], suffix):
                return True
        return fnmatch.fnmatch(Path(path).name, suffix)

    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return path == prefix or path.startswith(prefix + "/")

    # mid-pattern ** (rare); fall back to recursive segment walk
    head, tail = pattern.split("/**/", 1)
    if not (path == head or path.startswith(head + "/")):
        return False
    rest = path[len(head) + 1 :] if path.startswith(head + "/") else ""
    parts = rest.split("/") if rest else []
    for i in range(len(parts) + 1):
        candidate = "/".join(parts[i:])
        if fnmatch.fnmatch(candidate, tail):
            return True
    return False


def matching_paths(paths: list[str | Path]) -> list[str]:
    """Return normalized paths that are Research-tracked."""
    return [_normalize(p) for p in paths if is_research_path(p)]


def find_project_root(start: Path | None = None) -> Path:
    """Walk parents until _bmad/ or .git/ is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "_bmad").is_dir() or (candidate / ".git").is_dir():
            return candidate
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether paths are QuantBMAD Research-tracked."
    )
    parser.add_argument(
        "--check",
        nargs="+",
        metavar="PATH",
        help="Paths to check (exit 1 if any match Research globs)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print QUANTBMAD_RESEARCH_GLOBS and exit 0",
    )
    args = parser.parse_args(argv)

    if args.list:
        for glob in QUANTBMAD_RESEARCH_GLOBS:
            print(glob)
        return 0

    if not args.check:
        parser.print_help()
        return 2

    hits = matching_paths(args.check)
    if hits:
        for hit in hits:
            print(f"RESEARCH: {hit}")
        return 1
    print("OK: no Research-tracked paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
