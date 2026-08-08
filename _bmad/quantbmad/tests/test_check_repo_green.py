"""Tests for check_repo_green helpers (unit-level; does not run full suites)."""

from __future__ import annotations

import sys
from pathlib import Path

_BMAD = Path(__file__).resolve().parents[2]
if str(_BMAD) not in sys.path:
    sys.path.insert(0, str(_BMAD))

from quantbmad.scripts import check_repo_green as green  # noqa: E402


def test_ruff_bin_prefers_venv(tmp_path: Path, monkeypatch) -> None:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    ruff = venv_bin / "ruff"
    ruff.write_text("#!/bin/sh\n", encoding="utf-8")
    ruff.chmod(0o755)
    assert green._ruff_bin(tmp_path) == str(ruff)


def test_main_rejects_unknown_phase() -> None:
    try:
        green.main(["--phase", "nope"])
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 2
