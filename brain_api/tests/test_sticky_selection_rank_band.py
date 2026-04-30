"""Pure-function tests for the rank-band sticky-selection primitive.

Covers the asymmetric rank-band variant ``select_with_rank_band``:

- Cold start returns plain top-N by score desc
- Held name with rank inside ``hold_threshold`` is kept (``sticky``)
- Held name with rank outside ``hold_threshold`` is evicted with reason
  ``rank_out_of_hold``
- Held name absent from current scores is evicted with reason
  ``dropped_from_universe``
- More than ``top_n`` sticky candidates are trimmed to ``top_n`` by best
  (lowest) current rank
- Tie-break determinism: equal scores -> symbol ascending
- Result preserves order: kept (rank asc) then fillers (rank asc)
- Validation: ``top_n=0``, ``hold_threshold < top_n``, empty dict raise
"""

from __future__ import annotations

import pytest

from brain_api.core.sticky_selection import (
    EVICTION_DROPPED_FROM_UNIVERSE,
    EVICTION_RANK_OUT_OF_HOLD,
    REASON_STICKY,
    REASON_TOP_RANK,
    rank_by_score,
    select_with_rank_band,
)


def _scores(*pairs: tuple[str, float]) -> dict[str, float]:
    return dict(pairs)


# ----------------------------------------------------------------------------
# Cold start
# ----------------------------------------------------------------------------


class TestColdStart:
    def test_returns_top_n_by_score_desc(self):
        scores = _scores(("AAA", 0.5), ("BBB", 1.5), ("CCC", 1.0), ("DDD", 0.1))
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set=None,
            top_n=2,
            hold_threshold=4,
        )
        assert result.selected == ["BBB", "CCC"]
        assert result.kept_count == 0
        assert result.fillers_count == 2
        assert result.evicted_from_previous == {}
        assert all(r == REASON_TOP_RANK for r in result.reasons.values())


# ----------------------------------------------------------------------------
# Sticky kept by rank-inside-hold
# ----------------------------------------------------------------------------


class TestStickyKept:
    def test_held_name_inside_hold_threshold_is_sticky(self):
        # Ranks (desc score):
        #   BBB=1.5 -> 1, CCC=1.0 -> 2, AAA=0.5 -> 3, DDD=0.1 -> 4
        # Last week held AAA. With hold_threshold=4, AAA's rank=3 is
        # inside hold => kept as sticky.
        scores = _scores(("AAA", 0.5), ("BBB", 1.5), ("CCC", 1.0), ("DDD", 0.1))
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set={"AAA"},
            top_n=2,
            hold_threshold=4,
        )
        # Sticky stocks are listed first (rank asc), then fillers (rank asc).
        # AAA kept; remaining slot filled with the best non-kept symbol BBB.
        assert result.selected == ["AAA", "BBB"]
        assert result.reasons["AAA"] == REASON_STICKY
        assert result.reasons["BBB"] == REASON_TOP_RANK
        assert result.kept_count == 1
        assert result.fillers_count == 1
        assert result.evicted_from_previous == {}


# ----------------------------------------------------------------------------
# Eviction by rank-out-of-hold
# ----------------------------------------------------------------------------


class TestRankOutOfHoldEviction:
    def test_held_name_outside_hold_is_evicted(self):
        # Ranks:
        #   BBB=1.5 -> 1, CCC=1.0 -> 2, AAA=0.5 -> 3, DDD=0.1 -> 4
        # hold_threshold=2 means only ranks 1-2 retain sticky status.
        # Last week held AAA (now rank=3) -> evicted.
        scores = _scores(("AAA", 0.5), ("BBB", 1.5), ("CCC", 1.0), ("DDD", 0.1))
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set={"AAA"},
            top_n=2,
            hold_threshold=2,
        )
        assert result.selected == ["BBB", "CCC"]
        assert result.kept_count == 0
        assert result.fillers_count == 2
        assert result.evicted_from_previous == {"AAA": EVICTION_RANK_OUT_OF_HOLD}
        assert all(r == REASON_TOP_RANK for r in result.reasons.values())


# ----------------------------------------------------------------------------
# Eviction by dropped-from-universe
# ----------------------------------------------------------------------------


class TestDroppedFromUniverseEviction:
    def test_held_name_absent_from_current_is_evicted(self):
        scores = _scores(("BBB", 1.5), ("CCC", 1.0), ("DDD", 0.1))
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set={"AAA", "BBB"},
            top_n=2,
            hold_threshold=3,
        )
        # AAA dropped from universe; BBB held within hold_threshold.
        assert result.evicted_from_previous == {
            "AAA": EVICTION_DROPPED_FROM_UNIVERSE,
        }
        assert "BBB" in result.selected
        assert result.reasons["BBB"] == REASON_STICKY


# ----------------------------------------------------------------------------
# Sticky overflow trim
# ----------------------------------------------------------------------------


class TestStickyOverflowTrim:
    def test_trim_to_top_n_by_best_rank(self):
        # Ranks:
        #   BBB=1.5 -> 1, CCC=1.0 -> 2, AAA=0.5 -> 3, DDD=0.1 -> 4
        # hold_threshold=4 keeps all 4 names eligible for sticky, but
        # top_n=2 forces a trim to the two BEST current ranks (BBB, CCC).
        scores = _scores(("AAA", 0.5), ("BBB", 1.5), ("CCC", 1.0), ("DDD", 0.1))
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set={"AAA", "BBB", "CCC", "DDD"},
            top_n=2,
            hold_threshold=4,
        )
        # All four were inside hold; only two slots; pick best ranks.
        assert result.selected == ["BBB", "CCC"]
        assert result.kept_count == 2
        assert result.fillers_count == 0
        # AAA and DDD did not survive the trim, but they were not
        # "evicted" — they are still inside the hold band; they simply
        # lost the trim. The trim is a slot-budget decision, not an
        # eviction; matches the existing weight-band primitive's
        # convention of only recording eviction reasons.
        assert result.evicted_from_previous == {}
        assert result.reasons == {"BBB": REASON_STICKY, "CCC": REASON_STICKY}


# ----------------------------------------------------------------------------
# Tie-break determinism
# ----------------------------------------------------------------------------


class TestTieBreakDeterminism:
    def test_equal_scores_use_symbol_asc(self):
        # All equal scores -> ranks determined by symbol asc:
        # AAA=1, BBB=2, CCC=3, DDD=4.
        scores = _scores(("DDD", 1.0), ("BBB", 1.0), ("AAA", 1.0), ("CCC", 1.0))
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set=None,
            top_n=2,
            hold_threshold=2,
        )
        assert result.selected == ["AAA", "BBB"]


# ----------------------------------------------------------------------------
# Order preservation: kept (rank asc) then fillers (rank asc)
# ----------------------------------------------------------------------------


class TestSelectedOrder:
    def test_kept_listed_before_fillers_each_in_rank_order(self):
        # Ranks:
        #   AAA=10 -> 1, BBB=8 -> 2, CCC=6 -> 3, DDD=4 -> 4, EEE=2 -> 5
        # top_n=3, hold_threshold=4. Held names {DDD, BBB}: both inside
        # hold. Fillers fill the remaining slot from highest non-kept
        # rank (AAA=rank 1).
        scores = _scores(
            ("AAA", 10.0), ("BBB", 8.0), ("CCC", 6.0), ("DDD", 4.0), ("EEE", 2.0)
        )
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set={"DDD", "BBB"},
            top_n=3,
            hold_threshold=4,
        )
        # Kept first (rank asc): BBB (rank 2), DDD (rank 4); then filler
        # AAA (rank 1). The kept block precedes fillers regardless of the
        # filler having a better rank — this matches the existing
        # weight-band primitive's contract.
        assert result.selected == ["BBB", "DDD", "AAA"]
        assert result.reasons["BBB"] == REASON_STICKY
        assert result.reasons["DDD"] == REASON_STICKY
        assert result.reasons["AAA"] == REASON_TOP_RANK


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_top_n_zero_raises(self):
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            select_with_rank_band(
                current_scores={"AAA": 1.0},
                previous_selected_set=None,
                top_n=0,
                hold_threshold=1,
            )

    def test_hold_threshold_below_top_n_raises(self):
        with pytest.raises(ValueError, match="hold_threshold"):
            select_with_rank_band(
                current_scores={"AAA": 1.0, "BBB": 0.5},
                previous_selected_set=None,
                top_n=2,
                hold_threshold=1,
            )

    def test_empty_current_scores_raises(self):
        with pytest.raises(ValueError, match="current_scores must not be empty"):
            select_with_rank_band(
                current_scores={},
                previous_selected_set=None,
                top_n=1,
                hold_threshold=1,
            )

    def test_nan_score_raises(self):
        """M3: NaN breaks strict-weak ordering -> reject before sort."""
        with pytest.raises(ValueError, match="finite"):
            select_with_rank_band(
                current_scores={"AAA": 1.0, "BBB": float("nan")},
                previous_selected_set=None,
                top_n=1,
                hold_threshold=2,
            )

    def test_pos_inf_score_raises(self):
        with pytest.raises(ValueError, match="finite"):
            select_with_rank_band(
                current_scores={"AAA": 1.0, "BBB": float("inf")},
                previous_selected_set=None,
                top_n=1,
                hold_threshold=2,
            )

    def test_neg_inf_score_raises(self):
        with pytest.raises(ValueError, match="finite"):
            select_with_rank_band(
                current_scores={"AAA": 1.0, "BBB": float("-inf")},
                previous_selected_set=None,
                top_n=1,
                hold_threshold=2,
            )


# ----------------------------------------------------------------------------
# Edge cases (M6)
# ----------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_symbol_universe(self):
        """top_n=1 on a one-symbol universe selects that symbol."""
        result = select_with_rank_band(
            current_scores={"AAA": 1.0},
            previous_selected_set=None,
            top_n=1,
            hold_threshold=1,
        )
        assert result.selected == ["AAA"]
        assert result.kept_count == 0
        assert result.fillers_count == 1

    def test_empty_set_previous_treated_as_no_holdings(self):
        """``previous_selected_set == set()`` is NOT cold start.

        Cold start is signalled by ``None``. An empty set means we did
        run last week but ended up holding nothing -- treat as no
        sticky candidates, fall back to plain top-N filler. The
        downstream consumer (sticky_history) emits an empty set when
        Stage 2 weighted nothing, so this matches reality.
        """
        result = select_with_rank_band(
            current_scores={"AAA": 5.0, "BBB": 4.0, "CCC": 3.0},
            previous_selected_set=set(),
            top_n=2,
            hold_threshold=3,
        )
        # Equivalent to cold start in selected output.
        assert result.selected == ["AAA", "BBB"]
        assert result.kept_count == 0
        assert result.fillers_count == 2
        assert result.evicted_from_previous == {}

    def test_filler_ranks_always_within_top_n(self):
        """Asymmetry invariant: every filler must have rank <= top_n.

        Property test: with hold_threshold > top_n, even after kept
        names are placed, fillers never come from ranks beyond top_n.
        Together with kept being a subset of size <= top_n, this
        means new entries can only happen via the entry zone K_in.
        """
        # 10 symbols, ranks 1..10.
        scores = {f"S{i:02d}": 100 - i for i in range(10)}
        # Last week's holdings include some that have slipped.
        previous = {"S03", "S07"}
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set=previous,
            top_n=4,
            hold_threshold=8,
        )
        ranks = {sym: rank for sym, rank, _ in rank_by_score(scores)}
        # Identify fillers.
        filler_ranks = [
            ranks[sym]
            for sym, reason in result.reasons.items()
            if reason == REASON_TOP_RANK
        ]
        # Every filler must satisfy rank <= top_n (entry threshold).
        assert all(r <= 4 for r in filler_ranks), filler_ranks


# ----------------------------------------------------------------------------
# rank_by_score (D4)
# ----------------------------------------------------------------------------


class TestRankByScore:
    def test_returns_rank_ascending_with_score_desc_tie_break(self):
        scores = {"BBB": 1.0, "AAA": 1.0, "CCC": 0.5}
        ranked = rank_by_score(scores)
        # Tie at score=1.0; symbol asc -> AAA before BBB. CCC last.
        assert ranked == [
            ("AAA", 1, 1.0),
            ("BBB", 2, 1.0),
            ("CCC", 3, 0.5),
        ]

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            rank_by_score({})

    def test_non_finite_score_raises(self):
        with pytest.raises(ValueError, match="finite"):
            rank_by_score({"AAA": 1.0, "BBB": float("nan")})

    def test_matches_selector_rank_assignment(self):
        """Ranks from rank_by_score must match those select_with_rank_band saw.

        This protects the invariant that the ``stage1_rank`` written to
        sticky_history.db (via the rank-band endpoint, which calls
        ``rank_by_score``) matches the rank the selector actually used.
        """
        scores = {"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 2.0}
        ranks = {sym: rank for sym, rank, _ in rank_by_score(scores)}
        # Cold start: top_n=2 picks ranks 1..2.
        result = select_with_rank_band(
            current_scores=scores,
            previous_selected_set=None,
            top_n=2,
            hold_threshold=4,
        )
        for sym in result.selected:
            assert ranks[sym] <= 2
