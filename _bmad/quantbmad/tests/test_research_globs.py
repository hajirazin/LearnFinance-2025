"""Unit tests for QuantBMAD research_globs matching."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BMAD = Path(__file__).resolve().parents[2]
if str(_BMAD) not in sys.path:
    sys.path.insert(0, str(_BMAD))

from quantbmad.research_globs import (  # noqa: E402
    is_research_path,
    main,
    matching_paths,
)


@pytest.mark.parametrize(
    "path",
    [
        "brain_api/brain_api/core/hrp.py",
        "brain_api/brain_api/core/sticky_selection.py",
        "brain_api/brain_api/core/sac/promotion.py",
        "brain_api/brain_api/storage/ibkr_orders.py",
        "brain_api/brain_api/routes/orders.py",
        "brain_api/brain_api/universe/halal_filtered.py",
        "brain_api/brain_api/core/portfolio_rl/broker_costs.py",
        "temporal/workflows/us_weekly_allocation.py",
        "_bmad/quantbmad/research_globs.py",
        "some/deep/path/strategy_partitions.py",
    ],
)
def test_research_paths_match(path: str) -> None:
    assert is_research_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "brain_api/brain_api/templates/email.html.j2",
        "docs/research-ledger.md",
        "AGENTS.md",
        "temporal/activities/client.py",
        "brain_api/brain_api/routes/universe.py",
    ],
)
def test_non_research_paths_do_not_match(path: str) -> None:
    assert is_research_path(path) is False


def test_matching_paths_filters() -> None:
    hits = matching_paths(
        [
            "docs/foo.md",
            "brain_api/brain_api/core/hrp.py",
            "README.md",
        ]
    )
    assert hits == ["brain_api/brain_api/core/hrp.py"]


def test_cli_check_exit_codes() -> None:
    assert main(["--check", "docs/foo.md"]) == 0
    assert main(["--check", "brain_api/brain_api/core/hrp.py"]) == 1
    assert main([]) == 2
