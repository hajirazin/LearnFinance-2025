"""Tests for the two-stage rank-band orchestration helper.

These exercise ``select_rank_band_with_persistence`` -- the in-process
helper underneath the ``/allocation/rank-band-top-n`` endpoint -- to
ensure refactoring out of the route did not move math or change the
side-effects on ``stage1_weight_history``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_api.core.rank_band_orchestration import (
    select_rank_band_with_persistence,
)
from brain_api.storage.sticky_history import (
    StickyHistoryRepository,
    WeightRow,
)

UNIVERSE = "halal_new_alpha"


@pytest.fixture
def repo(tmp_path: Path) -> StickyHistoryRepository:
    return StickyHistoryRepository(db_path=tmp_path / "sticky.db")


def _scores(*pairs: tuple[str, float]) -> dict[str, float]:
    return dict(pairs)


class TestSelectRankBandWithPersistence:
    def test_cold_start_picks_top_k_in(self, repo: StickyHistoryRepository):
        result, prev = select_rank_band_with_persistence(
            repo=repo,
            universe=UNIVERSE,
            year_week="202615",
            as_of_date="2026-04-10",
            run_id="cold",
            current_scores=_scores(("A", 5.0), ("B", 4.0), ("C", 3.0)),
            top_n=2,
            hold_threshold=3,
        )
        assert prev is None
        assert result.selected == ["A", "B"]

    def test_warm_start_uses_two_stage_carry_set(self, repo: StickyHistoryRepository):
        repo.persist_stage1(
            [
                WeightRow(
                    universe=UNIVERSE,
                    year_week="202610",
                    as_of_date="2026-03-09",
                    stock="A",
                    stage1_rank=1,
                    initial_allocation_pct=None,
                    signal_score=9.0,
                    final_allocation_pct=None,
                    selected_in_final=True,
                    selection_reason="top_rank",
                    run_id="seed",
                ),
                WeightRow(
                    universe=UNIVERSE,
                    year_week="202610",
                    as_of_date="2026-03-09",
                    stock="B",
                    stage1_rank=2,
                    initial_allocation_pct=None,
                    signal_score=8.0,
                    final_allocation_pct=None,
                    selected_in_final=False,
                    selection_reason=None,
                    run_id="seed",
                ),
            ]
        )
        repo.update_final_weights(
            universe=UNIVERSE, year_week="202610", final={"A": 100.0}
        )

        result, prev = select_rank_band_with_persistence(
            repo=repo,
            universe=UNIVERSE,
            year_week="202615",
            as_of_date="2026-04-10",
            run_id="warm",
            current_scores=_scores(("X", 10.0), ("Y", 9.0), ("A", 4.0), ("B", 3.0)),
            top_n=2,
            hold_threshold=3,
        )
        assert prev == "202610"
        assert "A" in result.selected
        assert result.reasons["A"] == "sticky"

    def test_persists_full_universe_with_signal_score_column(
        self, repo: StickyHistoryRepository
    ):
        scores = _scores(("A", 3.0), ("B", 2.0), ("C", 1.0))
        select_rank_band_with_persistence(
            repo=repo,
            universe=UNIVERSE,
            year_week="202615",
            as_of_date="2026-04-10",
            run_id="r1",
            current_scores=scores,
            top_n=2,
            hold_threshold=2,
        )
        rows = repo.read_week(universe=UNIVERSE, year_week="202615")
        assert {r.stock for r in rows} == {"A", "B", "C"}
        assert all(r.initial_allocation_pct is None for r in rows)
        assert all(r.signal_score is not None for r in rows)
        flags = {r.stock: r.selected_in_final for r in rows}
        assert flags["A"] is True
        assert flags["B"] is True
        assert flags["C"] is False

    def test_non_finite_raises(self, repo: StickyHistoryRepository):
        with pytest.raises(ValueError):
            select_rank_band_with_persistence(
                repo=repo,
                universe=UNIVERSE,
                year_week="202615",
                as_of_date="2026-04-10",
                run_id="bad",
                current_scores={"A": float("inf"), "B": 1.0},
                top_n=1,
                hold_threshold=1,
            )

    def test_empty_raises(self, repo: StickyHistoryRepository):
        with pytest.raises(ValueError):
            select_rank_band_with_persistence(
                repo=repo,
                universe=UNIVERSE,
                year_week="202615",
                as_of_date="2026-04-10",
                run_id="bad",
                current_scores={},
                top_n=1,
                hold_threshold=1,
            )
