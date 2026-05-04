"""Tests for the StickyHistoryRepository (SQLite persistence)."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_api.storage.sticky_history import (
    PreviousWeekSnapshot,
    StickyHistoryRepository,
    WeightRow,
)


def _row(
    *,
    universe: str = "halal_new",
    year_week: str = "202609",
    as_of_date: str = "2026-02-23",
    stock: str = "AAPL",
    stage1_rank: int = 1,
    initial_allocation_pct: float = 5.0,
    final_allocation_pct: float | None = None,
    selected_in_final: bool = False,
    selection_reason: str | None = None,
    run_id: str = "paper:2026-02-23",
) -> WeightRow:
    return WeightRow(
        universe=universe,
        year_week=year_week,
        as_of_date=as_of_date,
        stock=stock,
        stage1_rank=stage1_rank,
        initial_allocation_pct=initial_allocation_pct,
        final_allocation_pct=final_allocation_pct,
        selected_in_final=selected_in_final,
        selection_reason=selection_reason,
        run_id=run_id,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> StickyHistoryRepository:
    """Fresh repository with isolated SQLite file per test."""
    return StickyHistoryRepository(db_path=tmp_path / "sticky.db")


# ----------------------------------------------------------------------------
# Schema bootstrap
# ----------------------------------------------------------------------------


class TestSchemaBootstrap:
    def test_schema_created_on_init(self, tmp_path: Path):
        db = tmp_path / "sub" / "sticky.db"
        StickyHistoryRepository(db_path=db)
        # Re-opening should not fail
        StickyHistoryRepository(db_path=db)
        assert db.exists()

    def test_directory_auto_created(self, tmp_path: Path):
        db = tmp_path / "deep" / "nested" / "sticky.db"
        StickyHistoryRepository(db_path=db)
        assert db.parent.is_dir()


# ----------------------------------------------------------------------------
# persist_stage1 + read_week roundtrip
# ----------------------------------------------------------------------------


class TestPersistAndRead:
    def test_persist_and_read_roundtrip(self, repo: StickyHistoryRepository):
        rows = [
            _row(
                stock="AAPL",
                stage1_rank=1,
                initial_allocation_pct=5.0,
                selected_in_final=True,
                selection_reason="top_rank",
            ),
            _row(
                stock="MSFT",
                stage1_rank=2,
                initial_allocation_pct=4.0,
                selected_in_final=True,
                selection_reason="top_rank",
            ),
            _row(
                stock="GOOG",
                stage1_rank=3,
                initial_allocation_pct=3.0,
                selected_in_final=False,
            ),
        ]
        n = repo.persist_stage1(rows)
        assert n == 3

        read_back = repo.read_week("halal_new", "202609")
        assert len(read_back) == 3
        assert [r.stock for r in read_back] == ["AAPL", "MSFT", "GOOG"]
        assert read_back[0].selected_in_final is True
        assert read_back[2].selected_in_final is False
        assert read_back[2].selection_reason is None

    def test_persist_empty_returns_zero(self, repo: StickyHistoryRepository):
        assert repo.persist_stage1([]) == 0

    def test_read_unknown_week_returns_empty(self, repo: StickyHistoryRepository):
        assert repo.read_week("halal_new", "202609") == []

    def test_rows_ordered_by_stage1_rank(self, repo: StickyHistoryRepository):
        rows = [
            _row(stock="C", stage1_rank=3),
            _row(stock="A", stage1_rank=1),
            _row(stock="B", stage1_rank=2),
        ]
        repo.persist_stage1(rows)
        read_back = repo.read_week("halal_new", "202609")
        assert [r.stage1_rank for r in read_back] == [1, 2, 3]


# ----------------------------------------------------------------------------
# INSERT OR REPLACE semantics on rerun
# ----------------------------------------------------------------------------


class TestRerunOverwrite:
    def test_rerun_overwrites_in_place(self, repo: StickyHistoryRepository):
        # First run
        repo.persist_stage1(
            [
                _row(
                    stock="AAPL",
                    stage1_rank=1,
                    initial_allocation_pct=5.0,
                    selected_in_final=True,
                    selection_reason="top_rank",
                    run_id="paper:2026-02-23",
                ),
            ]
        )

        # Rerun with different data for the same (universe, year_week, stock)
        repo.persist_stage1(
            [
                _row(
                    stock="AAPL",
                    stage1_rank=2,
                    initial_allocation_pct=4.5,
                    selected_in_final=False,
                    selection_reason=None,
                    run_id="paper:2026-02-23-redo",
                ),
            ]
        )

        rows = repo.read_week("halal_new", "202609")
        assert len(rows) == 1
        assert rows[0].stage1_rank == 2
        assert rows[0].initial_allocation_pct == 4.5
        assert rows[0].selected_in_final is False
        assert rows[0].selection_reason is None
        assert rows[0].run_id == "paper:2026-02-23-redo"

    def test_rerun_drops_stocks_absent_from_new_batch(
        self, repo: StickyHistoryRepository
    ):
        # First run had {A, B}; rerun has {A, C} -> B is dropped, not retained.
        repo.persist_stage1(
            [
                _row(stock="A", stage1_rank=1, initial_allocation_pct=5.0),
                _row(stock="B", stage1_rank=2, initial_allocation_pct=4.0),
            ]
        )
        repo.persist_stage1(
            [
                _row(stock="A", stage1_rank=1, initial_allocation_pct=5.5),
                _row(stock="C", stage1_rank=2, initial_allocation_pct=4.0),
            ]
        )

        rows = repo.read_week("halal_new", "202609")
        assert {r.stock for r in rows} == {"A", "C"}

    def test_rerun_does_not_touch_other_weeks(self, repo: StickyHistoryRepository):
        # Persist two weeks, then rerun only one -> the other survives.
        repo.persist_stage1(
            [
                _row(year_week="202608", stock="X", initial_allocation_pct=2.0),
            ]
        )
        repo.persist_stage1(
            [
                _row(year_week="202609", stock="A", initial_allocation_pct=5.0),
                _row(year_week="202609", stock="B", initial_allocation_pct=4.0),
            ]
        )
        # Rerun 202609 only.
        repo.persist_stage1(
            [
                _row(year_week="202609", stock="A", initial_allocation_pct=6.0),
            ]
        )

        assert {r.stock for r in repo.read_week("halal_new", "202608")} == {"X"}
        assert {r.stock for r in repo.read_week("halal_new", "202609")} == {"A"}


# ----------------------------------------------------------------------------
# update_final_weights
# ----------------------------------------------------------------------------


class TestUpdateFinalWeights:
    def test_updates_only_supplied_stocks(self, repo: StickyHistoryRepository):
        repo.persist_stage1(
            [
                _row(stock="AAPL", stage1_rank=1, initial_allocation_pct=5.0),
                _row(stock="MSFT", stage1_rank=2, initial_allocation_pct=4.0),
                _row(stock="GOOG", stage1_rank=3, initial_allocation_pct=3.0),
            ]
        )

        n = repo.update_final_weights(
            universe="halal_new",
            year_week="202609",
            final={"AAPL": 40.0, "MSFT": 60.0},
        )
        assert n == 2

        rows = {r.stock: r for r in repo.read_week("halal_new", "202609")}
        assert rows["AAPL"].final_allocation_pct == 40.0
        assert rows["AAPL"].selected_in_final is True
        assert rows["MSFT"].final_allocation_pct == 60.0
        assert rows["MSFT"].selected_in_final is True
        # Untouched
        assert rows["GOOG"].final_allocation_pct is None
        assert rows["GOOG"].selected_in_final is False

    def test_empty_final_returns_zero(self, repo: StickyHistoryRepository):
        repo.persist_stage1([_row()])
        assert repo.update_final_weights("halal_new", "202609", {}) == 0

    def test_unknown_stock_silently_ignored(self, repo: StickyHistoryRepository):
        repo.persist_stage1([_row(stock="AAPL")])
        # ZZZZ doesn't exist for this week -> rowcount=0 for that one,
        # but the call should not raise.
        n = repo.update_final_weights(
            "halal_new", "202609", {"AAPL": 100.0, "ZZZZ": 0.0}
        )
        # SQLite executemany rowcount is the LAST statement's rowcount, but
        # what matters is the visible state.
        rows = repo.read_week("halal_new", "202609")
        assert len(rows) == 1
        assert rows[0].final_allocation_pct == 100.0
        # n is informational; just confirm it's non-negative
        assert n >= 0


# ----------------------------------------------------------------------------
# read_previous_final_set
# ----------------------------------------------------------------------------


class TestReadPreviousFinalSet:
    def test_no_previous_returns_none(self, repo: StickyHistoryRepository):
        assert repo.read_previous_final_set("halal_new", "202609") is None

    def test_returns_most_recent_prior_week(self, repo: StickyHistoryRepository):
        # Three weeks of data; query for 202609 should return 202608.
        # Post-M1 ``final_set`` reflects Stage 2 reality, so we must
        # populate ``final_allocation_pct`` via ``update_final_weights``.
        repo.persist_stage1(
            [
                _row(year_week="202607", stock="A", selected_in_final=True),
                _row(year_week="202607", stock="B", selected_in_final=False),
            ]
        )
        repo.update_final_weights("halal_new", "202607", {"A": 100.0})
        repo.persist_stage1(
            [
                _row(
                    year_week="202608",
                    stock="A",
                    initial_allocation_pct=5.5,
                    selected_in_final=True,
                ),
                _row(
                    year_week="202608",
                    stock="C",
                    initial_allocation_pct=3.0,
                    selected_in_final=True,
                ),
            ]
        )
        repo.update_final_weights("halal_new", "202608", {"A": 60.0, "C": 40.0})

        prev = repo.read_previous_final_set("halal_new", "202609")
        assert prev is not None
        assert prev.year_week == "202608"
        assert prev.final_set == {"A", "C"}
        assert prev.initial_allocation_by_stock == {"A": 5.5, "C": 3.0}

    def test_year_boundary_lex_ordering(self, repo: StickyHistoryRepository):
        # 202552 < 202601: querying for 202601 returns 202552.
        repo.persist_stage1(
            [
                _row(year_week="202552", stock="X", selected_in_final=True),
            ]
        )
        prev = repo.read_previous_final_set("halal_new", "202601")
        assert prev is not None
        assert prev.year_week == "202552"

    def test_excludes_current_week(self, repo: StickyHistoryRepository):
        # current year_week == stored year_week -> not "previous"
        repo.persist_stage1([_row(year_week="202609", selected_in_final=True)])
        assert repo.read_previous_final_set("halal_new", "202609") is None

    def test_ignores_future_weeks(self, repo: StickyHistoryRepository):
        repo.persist_stage1(
            [
                _row(year_week="202610", stock="A", selected_in_final=True),
                _row(year_week="202607", stock="B", selected_in_final=True),
            ]
        )
        prev = repo.read_previous_final_set("halal_new", "202609")
        assert prev is not None
        assert prev.year_week == "202607"

    def test_returns_snapshot_type(self, repo: StickyHistoryRepository):
        repo.persist_stage1([_row(year_week="202608", selected_in_final=True)])
        prev = repo.read_previous_final_set("halal_new", "202609")
        assert isinstance(prev, PreviousWeekSnapshot)


# ----------------------------------------------------------------------------
# Multi-universe isolation
# ----------------------------------------------------------------------------


class TestMultiUniverseIsolation:
    def test_universes_dont_cross_pollute(self, repo: StickyHistoryRepository):
        repo.persist_stage1(
            [
                _row(universe="halal_new", stock="AAPL", selected_in_final=True),
            ]
        )
        repo.persist_stage1(
            [
                _row(
                    universe="nifty_shariah_500",
                    stock="INFY.NS",
                    selected_in_final=True,
                ),
            ]
        )

        us_rows = repo.read_week("halal_new", "202609")
        in_rows = repo.read_week("nifty_shariah_500", "202609")

        assert {r.stock for r in us_rows} == {"AAPL"}
        assert {r.stock for r in in_rows} == {"INFY.NS"}

    def test_previous_week_lookup_is_universe_scoped(
        self, repo: StickyHistoryRepository
    ):
        # India has a 202608 row but US is queried -> US gets None.
        repo.persist_stage1(
            [
                _row(
                    universe="nifty_shariah_500",
                    year_week="202608",
                    stock="INFY.NS",
                    selected_in_final=True,
                ),
            ]
        )
        assert repo.read_previous_final_set("halal_new", "202609") is None
        assert repo.read_previous_final_set("nifty_shariah_500", "202609") is not None

    def test_update_final_weights_universe_scoped(self, repo: StickyHistoryRepository):
        # Two universes share the same year_week + stock symbol; updating
        # one universe must not touch the other's row.
        repo.persist_stage1(
            [
                _row(universe="halal_new", stock="ABC", initial_allocation_pct=5.0),
                _row(
                    universe="nifty_shariah_500",
                    stock="ABC",
                    initial_allocation_pct=2.0,
                ),
            ]
        )
        repo.update_final_weights("halal_new", "202609", {"ABC": 50.0})

        us = repo.read_week("halal_new", "202609")[0]
        ind = repo.read_week("nifty_shariah_500", "202609")[0]
        assert us.final_allocation_pct == 50.0
        assert ind.final_allocation_pct is None


# ----------------------------------------------------------------------------
# halal_india_double_hrp partition (per-strategy isolation)
# ----------------------------------------------------------------------------


class TestHalalIndiaDoubleHRPPartition:
    """Pin behaviour for the ``halal_india_double_hrp`` partition.

    These tests exist because the partition is strategy-named (not
    universe-named) on purpose: the underlying tradable universe is
    ``nifty_shariah_500``, but the carry-set must NOT mix with
    ``halal_india_alpha`` (rank-band, different math primitive). Both
    cohabit ``stage1_weight_history`` and target the same NSE symbols,
    so cross-partition isolation is the only thing preventing
    cross-contamination of "previously held" semantics.
    """

    def test_halal_india_double_hrp_does_not_see_halal_india_alpha(
        self, repo: StickyHistoryRepository
    ):
        repo.persist_stage1(
            [
                _row(
                    universe="halal_india_alpha",
                    year_week="202608",
                    stock="INFY.NS",
                    final_allocation_pct=10.0,
                    selected_in_final=True,
                ),
            ]
        )
        # Even though both strategies screen the SAME underlying NSE
        # universe, the double-HRP partition starts cold because nothing
        # has been written to it yet.
        assert repo.read_previous_final_set("halal_india_double_hrp", "202609") is None
        # And alpha-HRP still sees its own prior week unaffected.
        alpha_prev = repo.read_previous_final_set("halal_india_alpha", "202609")
        assert alpha_prev is not None
        assert alpha_prev.final_set == {"INFY.NS"}

    def test_halal_india_double_hrp_does_not_see_halal_new(
        self, repo: StickyHistoryRepository
    ):
        # US Double HRP and India Double HRP both use weight-band sticky
        # but they MUST stay in their own partitions -- they trade on
        # disjoint symbol universes (and would in any case be wrong to
        # blend even on shared tickers).
        repo.persist_stage1(
            [
                _row(
                    universe="halal_new",
                    year_week="202608",
                    stock="AAPL",
                    final_allocation_pct=8.0,
                    selected_in_final=True,
                ),
            ]
        )
        assert repo.read_previous_final_set("halal_india_double_hrp", "202609") is None

    def test_halal_india_double_hrp_cold_start_then_steady_state(
        self, repo: StickyHistoryRepository
    ):
        # Week 1: nothing in the partition -> cold start.
        assert repo.read_previous_final_set("halal_india_double_hrp", "202609") is None

        # Persist Stage 1 + record final weights for week 1.
        rows = [
            _row(
                universe="halal_india_double_hrp",
                year_week="202609",
                stock="INFY.NS",
                stage1_rank=1,
                initial_allocation_pct=8.0,
                selected_in_final=True,
                selection_reason="top_rank",
            ),
            _row(
                universe="halal_india_double_hrp",
                year_week="202609",
                stock="TCS.NS",
                stage1_rank=2,
                initial_allocation_pct=7.0,
                selected_in_final=True,
                selection_reason="top_rank",
            ),
        ]
        repo.persist_stage1(rows)
        repo.update_final_weights(
            "halal_india_double_hrp",
            "202609",
            {"INFY.NS": 60.0, "TCS.NS": 40.0},
        )

        # Week 2: the partition can now read week 1 as the "previous"
        # snapshot, and only stocks with non-null final_allocation_pct
        # appear in final_set (so a future Stage 1 row that was screened
        # but not picked won't pollute the carry-set).
        snap = repo.read_previous_final_set("halal_india_double_hrp", "202610")
        assert snap is not None
        assert isinstance(snap, PreviousWeekSnapshot)
        assert snap.year_week == "202609"
        assert snap.final_set == {"INFY.NS", "TCS.NS"}

    def test_halal_india_double_hrp_rerun_same_week_is_delete_then_insert(
        self, repo: StickyHistoryRepository
    ):
        # First write to the partition.
        repo.persist_stage1(
            [
                _row(
                    universe="halal_india_double_hrp",
                    year_week="202609",
                    stock="INFY.NS",
                    stage1_rank=1,
                    initial_allocation_pct=5.0,
                ),
                _row(
                    universe="halal_india_double_hrp",
                    year_week="202609",
                    stock="TCS.NS",
                    stage1_rank=2,
                    initial_allocation_pct=4.0,
                ),
            ]
        )
        # Rerun in the same week with a different ranking. The repo must
        # delete the prior week's rows and insert the new ones rather
        # than UPSERT-merging, because rank ordering is part of the
        # math-aware payload (see persist_stage1 docstring).
        repo.persist_stage1(
            [
                _row(
                    universe="halal_india_double_hrp",
                    year_week="202609",
                    stock="HDFCBANK.NS",
                    stage1_rank=1,
                    initial_allocation_pct=6.0,
                ),
            ]
        )
        rows = repo.read_week("halal_india_double_hrp", "202609")
        assert {r.stock for r in rows} == {"HDFCBANK.NS"}
