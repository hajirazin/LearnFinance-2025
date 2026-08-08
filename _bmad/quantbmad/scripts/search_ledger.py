#!/usr/bin/env python3
"""Search docs/research-ledger.md for prior QuantBMAD hypotheses.

Usage:
  python3 _bmad/quantbmad/scripts/search_ledger.py --query "<topic>"
  python3 _bmad/quantbmad/scripts/search_ledger.py --query "<topic>" --ledger PATH

Prints matching rows or 'no prior entry'. Exit 0 always on successful search;
exit 2 on usage error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "_bmad").is_dir() or (candidate / ".git").is_dir():
            return candidate
    return current


def default_ledger_path(project_root: Path) -> Path:
    return project_root / "docs" / "research-ledger.md"


def search_ledger(ledger_path: Path, query: str) -> list[str]:
    """Return lines/rows that match all query tokens (case-insensitive)."""
    if not ledger_path.is_file():
        return []

    tokens = [t for t in re.split(r"\s+", query.strip().lower()) if t]
    if not tokens:
        return []

    text = ledger_path.read_text(encoding="utf-8")
    matches: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Skip header / separator rows
        if stripped.startswith("|---") or stripped.lower().startswith("| date"):
            continue
        lower = stripped.lower()
        if all(token in lower for token in tokens):
            matches.append(stripped)
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search the QuantBMAD research ledger."
    )
    parser.add_argument("--query", required=True, help="Search keywords")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Override ledger path (default: docs/research-ledger.md)",
    )
    args = parser.parse_args(argv)

    root = find_project_root()
    ledger = args.ledger or default_ledger_path(root)
    matches = search_ledger(ledger, args.query)

    if not matches:
        print("no prior entry")
        return 0

    print(f"Found {len(matches)} ledger hit(s) in {ledger}:")
    for row in matches:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
