"""Tests for the ScreeningHistoryRepository (single-stage SQLite persistence).

These tests target the sibling ``screening_history`` table that holds
single-stage rank-band screening rounds (e.g. ``halal_filtered_alpha``).
They are kept independent of ``test_sticky_history.py`` because the two
tables have distinct schemas and bounded contexts -- shared tests would
mask invariant divergence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_api.storage.screening_history import (
    PreviousScreeningRound,
    ScreeningHistoryRepository,
    ScreeningRow,
)


@pytest.fixture
def repo(tmp_path: Path) -> ScreeningHistoryRepository:
    return ScreeningHistoryRepository(db_path=tmp_path / "screen.db")


def _row(
    *,
    partition: str = "halal_filtered_alpha",
    period_key: str = "202615",
    stock: str,
    rank: int,
    selected: bool = False,
    selection_reason: str | None = None,
    signal_score: float = 1.0,
    run_id: str = "test-run",
    as_of_date: str = "2026-04-06",
) -> ScreeningRow:
    return ScreeningRow(
        partition=partition,
        period_key=period_key,
        as_of_date=as_of_date,
        stock=stock,
        rank=rank,
        signal_score=signal_score,
        selected=selected,
        selection_reason=selection_reason,
        run_id=run_id,
    )


class TestSchemaInit:
    def test_schema_created_on_init(self, tmp_path: Path):
        db = tmp_path / "sub" / "screen.db"
        ScreeningHistoryRepository(db_path=db)
        assert db.exists()
        ScreeningHistoryRepository(db_path=db)

    def test_directory_auto_created(self, tmp_path: Path):
        db = tmp_path / "deep" / "nested" / "screen.db"
        ScreeningHistoryRepository(db_path=db)
        assert db.exists()


class TestPersistAndRead:
    def test_persist_and_read_roundtrip(self, repo: ScreeningHistoryRepository):
        rows = [
            _row(
                stock="AAPL",
                rank=1,
                signal_score=5.5,
                selected=True,
                selection_reason="top_rank",
            ),
            _row(
                stock="MSFT",
                rank=2,
                signal_score=4.5,
                selected=True,
                selection_reason="top_rank",
            ),
            _row(stock="NVDA", rank=3, signal_score=3.5, selected=False),
        ]
        n = repo.persist_screening_round(rows)
        assert n == 3

        got = repo.read_round("halal_filtered_alpha", "202615")
        assert [r.stock for r in got] == ["AAPL", "MSFT", "NVDA"]
        assert got[0].selected is True
        assert got[2].selected is False
        assert got[0].signal_score == 5.5

    def test_persist_empty_returns_zero(self, repo: ScreeningHistoryRepository):
        assert repo.persist_screening_round([]) == 0

    def test_rerun_overwrites_in_place(self, repo: ScreeningHistoryRepository):
        first = [_row(stock=f"S{i}", rank=i + 1, selected=(i < 3)) for i in range(5)]
        repo.persist_screening_round(first)

        rerun = [_row(stock=f"S{i}", rank=i + 1, selected=(i < 2)) for i in range(2)]
        repo.persist_screening_round(rerun)

        got = repo.read_round("halal_filtered_alpha", "202615")
        assert {r.stock for r in got} == {"S0", "S1"}


class TestReadPreviousSelectedSet:
    def test_no_previous_returns_none(self, repo: ScreeningHistoryRepository):
        assert repo.read_previous_selected_set("halal_filtered_alpha", "202615") is None

    def test_returns_most_recent_prior_period(self, repo: ScreeningHistoryRepository):
        repo.persist_screening_round(
            [_row(period_key="202610", stock="A", rank=1, selected=True)]
        )
        repo.persist_screening_round(
            [_row(period_key="202614", stock="B", rank=1, selected=True)]
        )
        prev = repo.read_previous_selected_set("halal_filtered_alpha", "202615")
        assert prev == PreviousScreeningRound(period_key="202614", selected_set={"B"})

    def test_excludes_current_period(self, repo: ScreeningHistoryRepository):
        repo.persist_screening_round(
            [_row(period_key="202615", stock="X", rank=1, selected=True)]
        )
        assert repo.read_previous_selected_set("halal_filtered_alpha", "202615") is None

    def test_only_selected_symbols_in_carry_set(self, repo: ScreeningHistoryRepository):
        repo.persist_screening_round(
            [
                _row(period_key="202610", stock="A", rank=1, selected=True),
                _row(period_key="202610", stock="B", rank=2, selected=False),
            ]
        )
        prev = repo.read_previous_selected_set("halal_filtered_alpha", "202615")
        assert prev is not None
        assert prev.selected_set == {"A"}

    def test_partition_isolation(self, repo: ScreeningHistoryRepository):
        repo.persist_screening_round(
            [
                _row(
                    partition="halal_filtered_alpha",
                    period_key="202610",
                    stock="A",
                    rank=1,
                    selected=True,
                )
            ]
        )
        repo.persist_screening_round(
            [
                _row(
                    partition="other_alpha",
                    period_key="202610",
                    stock="B",
                    rank=1,
                    selected=True,
                )
            ]
        )
        prev = repo.read_previous_selected_set("halal_filtered_alpha", "202615")
        assert prev is not None
        assert prev.selected_set == {"A"}

        prev_other = repo.read_previous_selected_set("other_alpha", "202615")
        assert prev_other is not None
        assert prev_other.selected_set == {"B"}


class TestCrossTableIsolation:
    def test_screening_repo_does_not_see_stage1_rows(self, tmp_path: Path):
        """Both tables share a DB file but each repo only sees its own table."""
        from brain_api.storage.sticky_history import (
            StickyHistoryRepository,
            WeightRow,
        )

        db = tmp_path / "shared.db"

        sticky = StickyHistoryRepository(db_path=db)
        sticky.persist_stage1(
            [
                WeightRow(
                    universe="halal_new_alpha",
                    year_week="202615",
                    as_of_date="2026-04-06",
                    stock="X",
                    stage1_rank=1,
                    initial_allocation_pct=None,
                    signal_score=2.0,
                    final_allocation_pct=None,
                    selected_in_final=True,
                    selection_reason="top_rank",
                    run_id="run",
                )
            ]
        )

        screening = ScreeningHistoryRepository(db_path=db)
        assert screening.read_previous_selected_set("halal_new_alpha", "999999") is None
        assert (
            screening.read_previous_selected_set("halal_filtered_alpha", "999999")
            is None
        )

        screening.persist_screening_round(
            [
                _row(
                    partition="halal_filtered_alpha",
                    period_key="202615",
                    stock="Y",
                    rank=1,
                    selected=True,
                )
            ]
        )

        prev_sticky = sticky.read_previous_final_set(
            universe="halal_filtered_alpha",
            current_year_week="999999",
        )
        assert prev_sticky is None
