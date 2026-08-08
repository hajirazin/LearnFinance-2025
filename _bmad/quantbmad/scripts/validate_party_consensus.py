#!/usr/bin/env python3
"""Validate QuantBMAD party-consensus artifacts.

Exit 0 if valid Approve or valid escalate-to-razin with razin_decision.
Exit 1 if invalid. Exit 2 on usage error.

Usage:
  python3 _bmad/quantbmad/scripts/validate_party_consensus.py PATH
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BMAD_DIR = _SCRIPTS_DIR.parent.parent
if str(_BMAD_DIR) not in sys.path:
    sys.path.insert(0, str(_BMAD_DIR))

from quantbmad.research_globs import PARTY_MEMBER_CODES  # noqa: E402

VOTE_RE = re.compile(
    r"(?im)^\s*(q-[\w-]+)\s*:\s*(agree|disagree)\b",
)


def _field(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)^{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    if not match:
        return None
    return match.group(1).strip()


def parse_votes(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2).lower() for m in VOTE_RE.finditer(text)}


def validate_party_consensus(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing consensus file: {path}"]

    text = path.read_text(encoding="utf-8")
    kind = _field(text, "kind")
    if kind not in {"plan", "implement"}:
        errors.append(f"{path}: kind must be plan|implement (got {kind!r})")

    round_raw = _field(text, "round")
    try:
        round_n = int(round_raw) if round_raw else -1
    except ValueError:
        round_n = -1
    if round_n < 1 or round_n > 3:
        errors.append(f"{path}: round must be 1..3 (got {round_raw!r})")

    votes = parse_votes(text)
    missing = [c for c in PARTY_MEMBER_CODES if c not in votes]
    if missing:
        errors.append(f"{path}: missing votes for {missing}")

    unanimous_raw = (_field(text, "unanimous") or "").lower()
    outcome = (_field(text, "outcome") or "").lower()
    razin = _field(text, "razin_decision")

    all_agree = (
        bool(votes) and not missing and all(v == "agree" for v in votes.values())
    )
    computed_unanimous = all_agree

    if unanimous_raw not in {"true", "false"}:
        errors.append(f"{path}: unanimous must be true|false")
    elif unanimous_raw == "true" and not computed_unanimous:
        errors.append(f"{path}: unanimous: true but not all members voted agree")
    elif unanimous_raw == "false" and computed_unanimous:
        errors.append(f"{path}: unanimous: false but all votes are agree")

    if outcome == "approve":
        if not computed_unanimous:
            errors.append(f"{path}: outcome approve requires unanimous agree")
    elif outcome == "escalate-to-razin":
        if round_n != 3:
            errors.append(
                f"{path}: escalate-to-razin requires round: 3 (got {round_n})"
            )
        if not razin or razin.startswith("<"):
            errors.append(
                f"{path}: escalate-to-razin requires non-empty razin_decision"
            )
    else:
        errors.append(
            f"{path}: outcome must be approve|escalate-to-razin (got {outcome!r})"
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate party-consensus.md")
    parser.add_argument("path", type=Path, help="Path to party-consensus-*.md")
    args = parser.parse_args(argv)
    errors = validate_party_consensus(args.path)
    if errors:
        print("INVALID party consensus:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"OK: {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
