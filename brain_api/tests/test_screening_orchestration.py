"""Tests for the single-stage screening orchestration helper.

Verifies that ``select_rank_band_for_screening`` correctly:

- cold-starts when no previous round exists,
- warm-starts off ``screening_history`` (carry-set defined by
  ``selected = 1`` last round, NOT by allocation columns),
- evicts via the rank-band selector and reports the expected reasons,
- propagates non-finite-score errors from the selector (no silent
  fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_api.core.screening_orchestration import (
    persist_screening_rows,
    select_rank_band_for_screening,
)
from brain_api.storage.screening_history import ScreeningHistoryRepository

PARTITION = "halal_filtered_alpha"


@pytest.fixture
def repo(tmp_path: Path) -> ScreeningHistoryRepository:
    return ScreeningHistoryRepository(db_path=tmp_path / "screen.db")


def _scores(*pairs: tuple[str, float]) -> dict[str, float]:
    return dict(pairs)


class TestSelectRankBandForScreening:
    def test_cold_start_picks_top_k_in(self, repo: ScreeningHistoryRepository):
        scores = _scores(
            ("A", 5.0),
            ("B", 4.0),
            ("C", 3.0),
            ("D", 2.0),
            ("E", 1.0),
        )
        result, prev = select_rank_band_for_screening(
            repo=repo,
            partition=PARTITION,
            period_key="202615",
            as_of_date="2026-04-06",
            run_id="cold",
            current_scores=scores,
            top_n=3,
            hold_threshold=4,
        )
        assert prev is None
        assert result.selected == ["A", "B", "C"]
        assert result.kept_count == 0
        assert result.fillers_count == 3
        assert result.evicted_from_previous == {}

    def test_warm_start_keeps_held_in_band(self, repo: ScreeningHistoryRepository):
        seed_scores = _scores(("A", 9.0), ("B", 8.0), ("C", 7.0))
        persist_screening_rows(
            repo=repo,
            partition=PARTITION,
            period_key="202610",
            as_of_date="2026-03-09",
            run_id="seed",
            scores=seed_scores,
            selected_set={"A"},
            selection_reasons={"A": "top_rank"},
        )

        scores = _scores(
            ("X", 10.0),
            ("Y", 8.0),
            ("Z", 6.0),
            ("A", 5.0),
            ("B", 4.0),
        )
        result, prev = select_rank_band_for_screening(
            repo=repo,
            partition=PARTITION,
            period_key="202615",
            as_of_date="2026-04-06",
            run_id="warm",
            current_scores=scores,
            top_n=2,
            hold_threshold=5,
        )
        assert prev == "202610"
        assert "A" in result.selected
        assert result.reasons["A"] == "sticky"
        assert result.kept_count == 1

    def test_eviction_dropped_from_universe(self, repo: ScreeningHistoryRepository):
        persist_screening_rows(
            repo=repo,
            partition=PARTITION,
            period_key="202610",
            as_of_date="2026-03-09",
            run_id="seed",
            scores=_scores(("DELISTED", 9.0), ("A", 1.0)),
            selected_set={"DELISTED"},
            selection_reasons={"DELISTED": "top_rank"},
        )
        scores = _scores(("A", 5.0), ("B", 4.0), ("C", 3.0))
        result, _ = select_rank_band_for_screening(
            repo=repo,
            partition=PARTITION,
            period_key="202615",
            as_of_date="2026-04-06",
            run_id="warm",
            current_scores=scores,
            top_n=2,
            hold_threshold=3,
        )
        assert "DELISTED" not in result.selected
        assert result.evicted_from_previous.get("DELISTED") == "dropped_from_universe"

    def test_eviction_rank_out_of_hold(self, repo: ScreeningHistoryRepository):
        persist_screening_rows(
            repo=repo,
            partition=PARTITION,
            period_key="202610",
            as_of_date="2026-03-09",
            run_id="seed",
            scores=_scores(("HELD", 9.0)),
            selected_set={"HELD"},
            selection_reasons={"HELD": "top_rank"},
        )
        scores = _scores(
            ("A", 10.0),
            ("B", 9.0),
            ("C", 8.0),
            ("D", 7.0),
            ("E", 6.0),
            ("HELD", 1.0),
        )
        result, _ = select_rank_band_for_screening(
            repo=repo,
            partition=PARTITION,
            period_key="202615",
            as_of_date="2026-04-06",
            run_id="warm",
            current_scores=scores,
            top_n=2,
            hold_threshold=3,
        )
        assert "HELD" not in result.selected
        assert "HELD" in result.evicted_from_previous

    def test_non_finite_raises_no_fallback(self, repo: ScreeningHistoryRepository):
        scores = {"A": float("nan"), "B": 1.0}
        with pytest.raises(ValueError):
            select_rank_band_for_screening(
                repo=repo,
                partition=PARTITION,
                period_key="202615",
                as_of_date="2026-04-06",
                run_id="bad",
                current_scores=scores,
                top_n=1,
                hold_threshold=1,
            )

    def test_empty_scores_raises(self, repo: ScreeningHistoryRepository):
        with pytest.raises(ValueError):
            select_rank_band_for_screening(
                repo=repo,
                partition=PARTITION,
                period_key="202615",
                as_of_date="2026-04-06",
                run_id="bad",
                current_scores={},
                top_n=1,
                hold_threshold=1,
            )

    def test_persists_full_candidate_round(self, repo: ScreeningHistoryRepository):
        scores = _scores(("A", 3.0), ("B", 2.0), ("C", 1.0))
        select_rank_band_for_screening(
            repo=repo,
            partition=PARTITION,
            period_key="202615",
            as_of_date="2026-04-06",
            run_id="r1",
            current_scores=scores,
            top_n=2,
            hold_threshold=2,
        )
        rows = repo.read_round(PARTITION, "202615")
        assert {r.stock for r in rows} == {"A", "B", "C"}
        selected_flags = {r.stock: r.selected for r in rows}
        assert selected_flags["A"] is True
        assert selected_flags["B"] is True
        assert selected_flags["C"] is False
