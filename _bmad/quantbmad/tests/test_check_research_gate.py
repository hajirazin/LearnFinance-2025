"""Unit tests for QuantBMAD research gate and skill mirrors."""

from __future__ import annotations

import sys
from pathlib import Path

_BMAD = Path(__file__).resolve().parents[2]
_ROOT = _BMAD.parent
if str(_BMAD) not in sys.path:
    sys.path.insert(0, str(_BMAD))

from quantbmad.research_globs import PARTY_MEMBER_CODES, SKILL_NAMES  # noqa: E402
from quantbmad.scripts import check_research_gate as gate  # noqa: E402
from quantbmad.scripts.sync_skill_mirrors import (  # noqa: E402
    check_mirrors,
    sync_mirrors,
)


def _write_approved_plan(plan_dir: Path) -> Path:
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_dir / "plan.md"
    plan.write_text(
        "# Plan: test\nDate: 2026-08-08\nTrack: Research\n"
        "Requested by: Razin\nStatus: Approved\n\n## Ledger search\nno prior entry\n",
        encoding="utf-8",
    )
    return plan


def _write_pass_cert(plan_dir: Path, plan_rel: str) -> Path:
    cert = plan_dir / "evidence-cert.md"
    cert.write_text(
        f"""# Evidence cert
plan_path: {plan_rel}
hypothesis: test hypothesis
experiment_path: scratch/exp.py
command: python scratch/exp.py
exit_code: 0
key_metrics: ok
falsification_criteria_checked: yes
author_model: test-author
skeptic_model: test-skeptic
skeptic_task_id_or_fresh_chat: test-task
verdict: PASS
written_by: qb-agent-skeptic
""",
        encoding="utf-8",
    )
    return cert


def _write_consensus(plan_dir: Path, kind: str) -> Path:
    votes = "\n".join(f"  {c}: agree — ok" for c in PARTY_MEMBER_CODES)
    path = plan_dir / f"party-consensus-{kind}.md"
    path.write_text(
        f"""# Party consensus
kind: {kind}
plan_or_impl: docs/plans/2026-08-08-test/plan.md
round: 1
votes:
{votes}
unanimous: true
outcome: approve
razin_decision:
""",
        encoding="utf-8",
    )
    return path


def test_plan_is_approved_detects_status(tmp_path: Path) -> None:
    draft = tmp_path / "plan.md"
    draft.write_text("Status: Draft\n", encoding="utf-8")
    assert gate.plan_is_approved(draft) is False

    approved = tmp_path / "ok.md"
    approved.write_text("Status: Approved\n", encoding="utf-8")
    assert gate.plan_is_approved(approved) is True


def test_validate_cert_requires_pass_and_skeptic(tmp_path: Path) -> None:
    plan_dir = tmp_path / "docs" / "plans" / "2026-08-08-test"
    _write_approved_plan(plan_dir)
    cert = _write_pass_cert(plan_dir, "docs/plans/2026-08-08-test/plan.md")

    errors = gate.validate_evidence_cert(cert, tmp_path)
    assert errors == []

    bad = plan_dir / "bad-cert.md"
    bad.write_text(
        cert.read_text(encoding="utf-8").replace("verdict: PASS", "verdict: FAIL"),
        encoding="utf-8",
    )
    errs = gate.validate_evidence_cert(bad, tmp_path)
    assert any("PASS" in e for e in errs)


def test_validate_cert_rejects_non_skeptic_author(tmp_path: Path) -> None:
    plan_dir = tmp_path / "docs" / "plans" / "2026-08-08-test"
    _write_approved_plan(plan_dir)
    cert = _write_pass_cert(plan_dir, "docs/plans/2026-08-08-test/plan.md")
    text = cert.read_text(encoding="utf-8").replace(
        "written_by: qb-agent-skeptic", "written_by: parent-agent"
    )
    cert.write_text(text, encoding="utf-8")
    errs = gate.validate_evidence_cert(cert, tmp_path)
    assert any("written_by" in e for e in errs)


def test_draft_plan_fails_cert_link(tmp_path: Path) -> None:
    plan_dir = tmp_path / "docs" / "plans" / "2026-08-08-test"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.md").write_text("Status: Draft\n", encoding="utf-8")
    cert = _write_pass_cert(plan_dir, "docs/plans/2026-08-08-test/plan.md")
    errs = gate.validate_evidence_cert(cert, tmp_path)
    assert any("Approved" in e for e in errs)


def test_mirrors_check_on_real_repo() -> None:
    errors = check_mirrors(_ROOT)
    assert errors == [], errors


def test_gate_mirrors_only_ok() -> None:
    assert gate.main(["--mirrors-only", "--project-root", str(_ROOT)]) == 0


def test_gate_research_without_artifacts_fails(tmp_path: Path) -> None:
    plan, cert, consensus = gate.discover_gate_artifacts(tmp_path)
    assert plan is None and cert is None and consensus is None


def test_discover_prefers_approved_with_cert_and_consensus(tmp_path: Path) -> None:
    plan_dir = tmp_path / "docs" / "plans" / "2026-08-08-test"
    plan = _write_approved_plan(plan_dir)
    cert = _write_pass_cert(plan_dir, "docs/plans/2026-08-08-test/plan.md")
    consensus = _write_consensus(plan_dir, "implement")
    found_plan, found_cert, found_c = gate.discover_gate_artifacts(tmp_path)
    assert found_plan == plan
    assert found_cert == cert
    assert found_c == consensus


def test_sync_writes_all_skills() -> None:
    written = sync_mirrors(_ROOT)
    assert len(written) == len(SKILL_NAMES) * 3
    assert check_mirrors(_ROOT) == []


def test_real_repo_gate_with_research_path() -> None:
    assert (
        gate.main(
            [
                "--project-root",
                str(_ROOT),
                "brain_api/brain_api/core/hrp.py",
            ]
        )
        == 0
    )
