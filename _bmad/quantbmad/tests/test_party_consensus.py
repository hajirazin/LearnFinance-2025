"""Tests for party consensus validation."""

from __future__ import annotations

import sys
from pathlib import Path

_BMAD = Path(__file__).resolve().parents[2]
if str(_BMAD) not in sys.path:
    sys.path.insert(0, str(_BMAD))

from quantbmad.research_globs import PARTY_MEMBER_CODES, SKILL_NAMES  # noqa: E402
from quantbmad.scripts.validate_party_consensus import (  # noqa: E402
    validate_party_consensus,
)


def _votes_block(agree: bool = True) -> str:
    verb = "agree" if agree else "disagree"
    lines = [f"  {code}: {verb} — reason" for code in PARTY_MEMBER_CODES]
    return "\n".join(lines)


def test_skill_names_are_agents_only() -> None:
    assert "qb-skeptic" not in SKILL_NAMES
    assert "qb-plan" not in SKILL_NAMES
    assert "qb-implement" not in SKILL_NAMES
    assert "qb-agent-skeptic" in SKILL_NAMES
    assert "qb-agent-researcher" in SKILL_NAMES
    assert all(name.startswith("qb-agent-") for name in SKILL_NAMES)


def test_unanimous_approve_ok(tmp_path: Path) -> None:
    path = tmp_path / "party-consensus-plan.md"
    path.write_text(
        f"""# Party consensus
kind: plan
plan_or_impl: docs/plans/x/plan.md
round: 1
members: {", ".join(PARTY_MEMBER_CODES)}
votes:
{_votes_block(True)}
unanimous: true
outcome: approve
razin_decision:
""",
        encoding="utf-8",
    )
    assert validate_party_consensus(path) == []


def test_implement_approve_requires_git_diff(tmp_path: Path) -> None:
    path = tmp_path / "party-consensus-implement.md"
    body = f"""# Party consensus
kind: implement
plan_or_impl: docs/plans/x/plan.md
round: 1
members: {", ".join(PARTY_MEMBER_CODES)}
votes:
{_votes_block(True)}
unanimous: true
outcome: approve
razin_decision:
"""
    path.write_text(body, encoding="utf-8")
    errs = validate_party_consensus(path)
    assert any("git_diff" in e for e in errs)

    path.write_text(
        body.replace(
            "plan_or_impl: docs/plans/x/plan.md\n",
            "plan_or_impl: docs/plans/x/plan.md\n"
            "git_diff: docs/plans/x/implement.diff\n",
        ),
        encoding="utf-8",
    )
    assert validate_party_consensus(path) == []


def test_approve_without_unanimity_fails(tmp_path: Path) -> None:
    path = tmp_path / "c.md"
    path.write_text(
        f"""# Party consensus
kind: implement
plan_or_impl: x
git_diff: unstaged working tree vs HEAD
round: 1
votes:
{_votes_block(False)}
unanimous: false
outcome: approve
razin_decision:
""",
        encoding="utf-8",
    )
    errs = validate_party_consensus(path)
    assert any("unanimous agree" in e for e in errs)


def test_escalate_requires_round_3_and_razin(tmp_path: Path) -> None:
    path = tmp_path / "c.md"
    path.write_text(
        f"""# Party consensus
kind: plan
plan_or_impl: x
round: 2
votes:
{_votes_block(False)}
unanimous: false
outcome: escalate-to-razin
razin_decision:
""",
        encoding="utf-8",
    )
    errs = validate_party_consensus(path)
    assert any("round: 3" in e for e in errs)
    assert any("razin_decision" in e for e in errs)

    path.write_text(
        f"""# Party consensus
kind: plan
plan_or_impl: x
round: 3
votes:
{_votes_block(False)}
unanimous: false
outcome: escalate-to-razin
razin_decision: Ship the probe as written; defer entropy work.
""",
        encoding="utf-8",
    )
    assert validate_party_consensus(path) == []


def test_missing_member_vote_fails(tmp_path: Path) -> None:
    path = tmp_path / "c.md"
    incomplete = "\n".join(f"  {code}: agree — ok" for code in PARTY_MEMBER_CODES[:-1])
    path.write_text(
        f"""# Party consensus
kind: plan
plan_or_impl: x
round: 1
votes:
{incomplete}
unanimous: true
outcome: approve
razin_decision:
""",
        encoding="utf-8",
    )
    errs = validate_party_consensus(path)
    assert any("missing votes" in e for e in errs)
