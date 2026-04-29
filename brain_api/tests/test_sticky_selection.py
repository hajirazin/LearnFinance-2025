"""Pure-function tests for the sticky-selection (rebalance-band) primitive.

Covers:
- Cold start (no previous week) returns plain top-N
- Fewer-than-N sticky candidates: kept first, fillers complete the list
- Exactly-N sticky candidates: no fillers
- More-than-N candidates trimmed by current weight desc (top-N tie-break)
- Sticky stock that disappeared from this week's universe is evicted as
  ``dropped_from_universe``
- Threshold strictly less-than: 1.0pp boundary stocks are evicted
- Parameterized threshold (0.5 vs 2.0) reproduces expected behavior
- Determinism: tie on weight uses symbol asc as secondary key
- Validation: top_n < 1, threshold < 0, empty current raise ValueError
- ISO year-week helper handles week-1 boundary correctly
"""

from __future__ import annotations

from datetime import date

import pytest

from brain_api.core.sticky_selection import (
    EVICTION_DROPPED_FROM_UNIVERSE,
    EVICTION_WEIGHT_DIFF,
    REASON_STICKY,
    REASON_TOP_RANK,
    SelectionResult,
    iso_year_week,
    select_with_stickiness,
)


def _weights(*pairs: tuple[str, float]) -> dict[str, float]:
    return dict(pairs)


# ----------------------------------------------------------------------------
# iso_year_week
# ----------------------------------------------------------------------------


class TestIsoYearWeek:
    def test_basic_week(self):
        # 2026-02-23 is a Monday in ISO week 9 of 2026
        assert iso_year_week(date(2026, 2, 23)) == "202609"

    def test_week_one_belongs_to_next_iso_year(self):
        # 2025-12-29 is Monday of ISO week 1 of 2026
        assert iso_year_week(date(2025, 12, 29)) == "202601"

    def test_lex_ordering_across_year_boundary(self):
        # "202552" < "202601" must hold for SQLite lex ordering
        assert iso_year_week(date(2025, 12, 22)) < iso_year_week(date(2025, 12, 29))

    def test_first_monday_of_2026(self):
        # 2026-01-05 is Monday of ISO week 2 of 2026
        assert iso_year_week(date(2026, 1, 5)) == "202602"


# ----------------------------------------------------------------------------
# Cold start
# ----------------------------------------------------------------------------


class TestColdStart:
    def test_cold_start_no_previous_data(self):
        current = _weights(("AAPL", 5.0), ("MSFT", 3.0), ("GOOG", 2.0))
        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=None,
            previous_final_set=None,
            top_n=2,
        )
        assert result.selected == ["AAPL", "MSFT"]
        assert result.reasons == {"AAPL": REASON_TOP_RANK, "MSFT": REASON_TOP_RANK}
        assert result.kept_count == 0
        assert result.fillers_count == 2
        assert result.evicted_from_previous == {}

    def test_cold_start_when_only_previous_weights_present(self):
        # If previous_stage1 is provided but previous_final_set is None,
        # treat as cold start to avoid silent half-truth behavior.
        current = _weights(("A", 4.0), ("B", 1.0))
        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1={"A": 4.5},
            previous_final_set=None,
            top_n=1,
        )
        assert result.selected == ["A"]
        assert result.kept_count == 0


# ----------------------------------------------------------------------------
# Sticky retention with various candidate counts
# ----------------------------------------------------------------------------


class TestStickyRetention:
    def test_fewer_than_top_n_sticky_candidates(self):
        # AAPL stable, MSFT moved too much (evicted from sticky). MSFT can
        # still re-enter as top_rank filler because eviction only blocks
        # sticky retention, not eligibility — the cost saving from staying
        # in MSFT vs. exiting+re-entering is negligible at that delta.
        current = _weights(
            ("AAPL", 5.5),
            ("GOOG", 4.0),
            ("MSFT", 3.0),  # was 5.0 -> 2.0pp diff -> evicted from sticky
            ("NVDA", 2.0),
        )
        previous = _weights(("AAPL", 5.0), ("MSFT", 5.0), ("GOOG", 1.0))
        prev_final = {"AAPL", "MSFT"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=3,
        )

        assert result.selected[0] == "AAPL"  # sticky comes first
        # MSFT comes back as top_rank filler (weight 3.0 > NVDA 2.0)
        assert set(result.selected) == {"AAPL", "GOOG", "MSFT"}
        assert result.reasons["AAPL"] == REASON_STICKY
        assert result.reasons["GOOG"] == REASON_TOP_RANK
        assert result.reasons["MSFT"] == REASON_TOP_RANK
        assert result.kept_count == 1
        assert result.fillers_count == 2
        assert result.evicted_from_previous == {"MSFT": EVICTION_WEIGHT_DIFF}

    def test_evicted_then_displaced_by_lower_rank_filler(self):
        # When the evicted stock's current weight is below the cutoff, it
        # is displaced by lower-ranked stocks.
        current = _weights(
            ("AAPL", 5.5),
            ("GOOG", 4.0),
            ("NVDA", 3.5),
            ("MSFT", 1.0),  # was 5.0 -> 4.0pp diff -> evicted; weight too low
        )
        previous = _weights(("AAPL", 5.0), ("MSFT", 5.0))
        prev_final = {"AAPL", "MSFT"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=3,
        )

        assert set(result.selected) == {"AAPL", "GOOG", "NVDA"}
        assert "MSFT" not in result.selected
        assert result.evicted_from_previous == {"MSFT": EVICTION_WEIGHT_DIFF}

    def test_exactly_top_n_sticky_candidates(self):
        # All previous final stocks remain stable -> no fillers needed.
        current = _weights(("A", 5.0), ("B", 4.0), ("C", 3.0), ("D", 2.0))
        previous = _weights(("A", 5.0), ("B", 4.5), ("C", 2.5))
        prev_final = {"A", "B", "C"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=3,
        )

        assert set(result.selected) == {"A", "B", "C"}
        # Sticky stocks ordered by current weight desc
        assert result.selected == ["A", "B", "C"]
        assert all(result.reasons[s] == REASON_STICKY for s in ["A", "B", "C"])
        assert result.kept_count == 3
        assert result.fillers_count == 0
        assert result.evicted_from_previous == {}

    def test_more_than_top_n_sticky_candidates_trimmed_by_weight(self):
        # 4 sticky-eligible but top_n=3 -> trim by current weight desc.
        # D has the lowest current weight, so D is dropped from sticky list.
        current = _weights(("A", 5.0), ("B", 4.0), ("C", 3.0), ("D", 2.0))
        previous = _weights(("A", 5.0), ("B", 4.0), ("C", 3.0), ("D", 2.0))
        prev_final = {"A", "B", "C", "D"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=3,
        )

        assert result.selected == ["A", "B", "C"]
        assert result.kept_count == 3
        assert result.fillers_count == 0
        # D was sticky-eligible but trimmed -> NOT marked as evicted
        # (eviction is only for weight_diff or dropped_from_universe).
        # The contract: evicted_from_previous tracks reasons we did not retain.
        # D was simply not selected; no eviction reason recorded.
        assert "D" not in result.evicted_from_previous


# ----------------------------------------------------------------------------
# Universe drift / dropped stocks
# ----------------------------------------------------------------------------


class TestDroppedFromUniverse:
    def test_sticky_stock_missing_this_week(self):
        # MSFT was held last week but is not in current universe (delisting,
        # ETF rebalance, etc.). Must be evicted with reason=dropped_from_universe.
        current = _weights(("AAPL", 6.0), ("GOOG", 4.0), ("NVDA", 3.0))
        previous = _weights(("AAPL", 5.5), ("MSFT", 4.5))
        prev_final = {"AAPL", "MSFT"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=2,
        )

        assert "MSFT" not in result.selected
        assert result.evicted_from_previous == {"MSFT": EVICTION_DROPPED_FROM_UNIVERSE}
        # AAPL remains sticky (diff 0.5pp < 1.0pp threshold)
        assert result.reasons["AAPL"] == REASON_STICKY


# ----------------------------------------------------------------------------
# Threshold semantics
# ----------------------------------------------------------------------------


class TestThresholdBoundary:
    def test_exact_threshold_evicts(self):
        # Strict less-than: diff == threshold -> evicted.
        current = _weights(("A", 6.0), ("B", 1.0))
        previous = _weights(("A", 5.0))
        prev_final = {"A"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=1,
            threshold_pp=1.0,
        )

        # A's diff is exactly 1.0pp -> evicted
        assert result.evicted_from_previous == {"A": EVICTION_WEIGHT_DIFF}
        assert result.kept_count == 0
        assert result.selected == ["A"]  # still highest current weight, but as top_rank
        assert result.reasons == {"A": REASON_TOP_RANK}

    def test_just_under_threshold_keeps(self):
        # diff < threshold by epsilon -> kept.
        current = _weights(("A", 5.999), ("B", 1.0))
        previous = _weights(("A", 5.0))
        prev_final = {"A"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=1,
            threshold_pp=1.0,
        )

        assert result.kept_count == 1
        assert result.reasons == {"A": REASON_STICKY}
        assert result.evicted_from_previous == {}

    def test_just_over_threshold_evicts(self):
        # diff > threshold by epsilon -> evicted.
        current = _weights(("A", 6.001), ("B", 1.0))
        previous = _weights(("A", 5.0))
        prev_final = {"A"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=1,
            threshold_pp=1.0,
        )

        assert result.kept_count == 0
        assert result.evicted_from_previous == {"A": EVICTION_WEIGHT_DIFF}


class TestThresholdParameterized:
    def test_tighter_threshold_evicts_smaller_moves(self):
        # 0.5pp threshold; A moved 0.6pp -> evicted, B moved 0.3pp -> kept.
        current = _weights(("A", 5.6), ("B", 4.3), ("C", 3.0))
        previous = _weights(("A", 5.0), ("B", 4.0))
        prev_final = {"A", "B"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=2,
            threshold_pp=0.5,
        )

        assert result.reasons["B"] == REASON_STICKY
        assert "A" in result.evicted_from_previous
        assert result.evicted_from_previous["A"] == EVICTION_WEIGHT_DIFF

    def test_looser_threshold_keeps_larger_moves(self):
        # 2.0pp threshold; A moved 1.5pp -> kept.
        current = _weights(("A", 6.5), ("B", 4.0), ("C", 3.0))
        previous = _weights(("A", 5.0), ("B", 4.0))
        prev_final = {"A", "B"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=2,
            threshold_pp=2.0,
        )

        assert result.reasons["A"] == REASON_STICKY
        assert result.reasons["B"] == REASON_STICKY
        assert result.kept_count == 2

    def test_zero_threshold_evicts_any_change(self):
        # threshold_pp=0.0 means strict inequality 0 < 0 is false -> always evict.
        current = _weights(("A", 5.000001), ("B", 4.0))
        previous = _weights(("A", 5.0))
        prev_final = {"A"}

        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=previous,
            previous_final_set=prev_final,
            top_n=1,
            threshold_pp=0.0,
        )

        assert result.kept_count == 0


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------


class TestDeterminism:
    def test_tied_weights_break_by_symbol_asc(self):
        # B and A both have weight 5.0; tie-break is symbol ascending -> A first.
        current = _weights(("B", 5.0), ("A", 5.0), ("C", 3.0))
        result = select_with_stickiness(
            current_stage1=current,
            previous_stage1=None,
            previous_final_set=None,
            top_n=2,
        )
        assert result.selected == ["A", "B"]


# ----------------------------------------------------------------------------
# Edge cases / validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_top_n_zero_raises(self):
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            select_with_stickiness(
                current_stage1={"A": 1.0},
                previous_stage1=None,
                previous_final_set=None,
                top_n=0,
            )

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold_pp must be >= 0"):
            select_with_stickiness(
                current_stage1={"A": 1.0},
                previous_stage1=None,
                previous_final_set=None,
                top_n=1,
                threshold_pp=-0.1,
            )

    def test_empty_current_raises(self):
        with pytest.raises(ValueError, match="current_stage1 must not be empty"):
            select_with_stickiness(
                current_stage1={},
                previous_stage1=None,
                previous_final_set=None,
                top_n=1,
            )


# ----------------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------------


class TestResultType:
    def test_returns_selection_result(self):
        result = select_with_stickiness(
            current_stage1={"A": 1.0},
            previous_stage1=None,
            previous_final_set=None,
            top_n=1,
        )
        assert isinstance(result, SelectionResult)
