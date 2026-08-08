#!/usr/bin/env python3
"""Pre-commit / CI gate for QuantBMAD Research-track changes.

Always verifies skill mirrors match canonical (literal copies, no symlinks).

When any passed filename matches research_globs:
  - Require QUANTBMAD_PLAN_PATH env OR discover latest Approved plan under
    docs/plans/ referenced by an evidence-cert in the same tree / env.
  - Simpler contract used by pre-commit: if Research paths are in the diff,
    require env vars or sibling artifact discovery:

Discovery rules when Research files are staged:
  1. QUANTBMAD_PLAN_PATH and QUANTBMAD_EVIDENCE_CERT_PATH if set, else
  2. Scan docs/plans/*/ for plan.md with Status: Approved and a sibling
     evidence-cert.md with verdict PASS and written_by skeptic-*.

Usage:
  python3 _bmad/quantbmad/scripts/check_research_gate.py [files...]
  python3 _bmad/quantbmad/scripts/check_research_gate.py --mirrors-only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_QUANTBMAD_DIR = _SCRIPTS_DIR.parent
_BMAD_DIR = _QUANTBMAD_DIR.parent
if str(_BMAD_DIR) not in sys.path:
    sys.path.insert(0, str(_BMAD_DIR))

from quantbmad.research_globs import (  # noqa: E402
    find_project_root,
    matching_paths,
)
from quantbmad.scripts.sync_skill_mirrors import check_mirrors  # noqa: E402
from quantbmad.scripts.validate_party_consensus import (  # noqa: E402
    validate_party_consensus,
)

ALLOWED_WRITTEN_BY = frozenset(
    {
        "qb-agent-skeptic",
        "skeptic-task",  # legacy
        "qb-skeptic-fresh-chat",  # legacy bootstrap
    }
)
REQUIRED_CERT_FIELDS = (
    "plan_path:",
    "hypothesis:",
    "experiment_path:",
    "command:",
    "exit_code:",
    "verdict:",
    "written_by:",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def plan_is_approved(plan_path: Path) -> bool:
    if not plan_path.is_file():
        return False
    text = _read(plan_path)
    return bool(re.search(r"(?m)^Status:\s*Approved\s*$", text))


def parse_cert_field(text: str, field: str) -> str | None:
    """Parse `field: value` from evidence-cert markdown (first match)."""
    # field already includes trailing colon in REQUIRED list style
    key = field if field.endswith(":") else f"{field}:"
    match = re.search(
        rf"(?im)^{re.escape(key)}\s*(.+?)\s*$",
        text,
    )
    if not match:
        return None
    return match.group(1).strip()


def validate_evidence_cert(cert_path: Path, project_root: Path) -> list[str]:
    errors: list[str] = []
    if not cert_path.is_file():
        return [f"missing evidence-cert: {cert_path}"]

    text = _read(cert_path)
    for field in REQUIRED_CERT_FIELDS:
        if parse_cert_field(text, field) is None:
            errors.append(f"{cert_path}: missing field {field}")

    verdict = parse_cert_field(text, "verdict:")
    if verdict and verdict.upper() != "PASS":
        errors.append(f"{cert_path}: verdict must be PASS (got {verdict!r})")

    written_by = parse_cert_field(text, "written_by:")
    if written_by and written_by not in ALLOWED_WRITTEN_BY:
        errors.append(
            f"{cert_path}: written_by must be one of {sorted(ALLOWED_WRITTEN_BY)} "
            f"(got {written_by!r})"
        )

    plan_rel = parse_cert_field(text, "plan_path:")
    if plan_rel:
        plan_path = (project_root / plan_rel).resolve()
        try:
            plan_path.relative_to(project_root.resolve())
        except ValueError:
            errors.append(f"{cert_path}: plan_path escapes project root")
        else:
            if not plan_is_approved(plan_path):
                errors.append(
                    f"{cert_path}: linked plan is not Status: Approved ({plan_rel})"
                )

    return errors


def discover_gate_artifacts(
    project_root: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    """Find Approved plan + PASS cert + implement consensus under docs/plans/*/."""
    plans_root = project_root / "docs" / "plans"
    if not plans_root.is_dir():
        return None, None, None

    env_plan = os.environ.get("QUANTBMAD_PLAN_PATH")
    env_cert = os.environ.get("QUANTBMAD_EVIDENCE_CERT_PATH")
    env_consensus = os.environ.get("QUANTBMAD_CONSENSUS_PATH")
    if env_plan and env_cert:
        consensus = (
            project_root / env_consensus
            if env_consensus
            else (project_root / env_plan).parent / "party-consensus-implement.md"
        )
        return project_root / env_plan, project_root / env_cert, consensus

    candidates: list[tuple[float, Path, Path, Path]] = []
    for plan_dir in plans_root.iterdir():
        if not plan_dir.is_dir():
            continue
        plan = plan_dir / "plan.md"
        cert = plan_dir / "evidence-cert.md"
        consensus = plan_dir / "party-consensus-implement.md"
        if (
            plan.is_file()
            and cert.is_file()
            and consensus.is_file()
            and plan_is_approved(plan)
        ):
            candidates.append((plan_dir.stat().st_mtime, plan, cert, consensus))

    if not candidates:
        return None, None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, plan, cert, consensus = candidates[0]
    return plan, cert, consensus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="QuantBMAD research gate (mirrors + Research artifacts)."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Paths from pre-commit (Research check if any match globs)",
    )
    parser.add_argument(
        "--mirrors-only",
        action="store_true",
        help="Only verify skill mirror checksums",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    root = (args.project_root or find_project_root()).resolve()

    errors: list[str] = []
    mirror_errors = check_mirrors(root)
    errors.extend(mirror_errors)

    if args.mirrors_only:
        if errors:
            print("QuantBMAD gate FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print("OK: QuantBMAD mirrors")
        return 0

    research_files = matching_paths(args.files) if args.files else []
    artifact_files = [
        p
        for p in research_files
        if (not p.startswith("_bmad/quantbmad/") or p.endswith("research_globs.py"))
    ]
    if artifact_files or os.environ.get("QUANTBMAD_FORCE_ARTIFACTS") == "1":
        plan, cert, consensus = discover_gate_artifacts(root)
        if plan is None or cert is None or consensus is None:
            errors.append(
                "Research-tracked changes require docs/plans/<slug>/plan.md "
                "(Status: Approved), evidence-cert.md (verdict PASS, "
                "written_by qb-agent-skeptic), and party-consensus-implement.md "
                "(approve or escalate-to-razin + razin_decision). "
                f"Research paths: {artifact_files or '(forced)'}"
            )
        else:
            if not plan_is_approved(plan):
                errors.append(f"plan not Approved: {plan}")
            errors.extend(validate_evidence_cert(cert, root))
            errors.extend(validate_party_consensus(consensus))
            plan_consensus = plan.parent / "party-consensus-plan.md"
            if plan_consensus.is_file():
                errors.extend(validate_party_consensus(plan_consensus))
            else:
                errors.append(f"missing plan consensus: {plan_consensus}")

    if errors:
        print("QuantBMAD research gate FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("OK: QuantBMAD research gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
