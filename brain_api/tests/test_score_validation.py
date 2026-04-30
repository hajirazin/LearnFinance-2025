"""Pure-function tests for the rank-band score validation policy.

The math invariants (non-finite rejection + ``min_predictions`` floor)
live in :mod:`brain_api.core.patchtst.score_validation` and are
exercised end-to-end via the ``/inference/patchtst/score-batch``
endpoint. These unit tests target the function directly so a regression
shows up before the API tests fan out into the per-market matrix.
"""

from __future__ import annotations

import math

import pytest

from brain_api.core.patchtst.inference import SymbolPrediction
from brain_api.core.patchtst.score_validation import (
    validate_and_collect_finite_scores,
)


def _pred(symbol: str, score: float | None) -> SymbolPrediction:
    """Minimal :class:`SymbolPrediction` with only the fields the validator reads."""
    return SymbolPrediction(
        symbol=symbol,
        predicted_weekly_return_pct=score,
        direction="UP" if (score is not None and score > 0) else "FLAT",
        has_enough_history=True,
        history_days_used=400,
        data_end_date="2026-04-25",
        target_week_start="2026-04-27",
        target_week_end="2026-05-01",
    )


class TestValidateAndCollectFiniteScores:
    """Math invariants of the rank-band score gate."""

    def test_all_finite_passes(self) -> None:
        """Happy path: every prediction is finite, none excluded."""
        preds = [_pred("AAA", 1.5), _pred("BBB", -0.5), _pred("CCC", 2.0)]
        scores, excluded = validate_and_collect_finite_scores(
            preds, requested_count=3, min_predictions=2
        )
        assert scores == {"AAA": 1.5, "BBB": -0.5, "CCC": 2.0}
        assert excluded == []

    def test_none_predictions_are_excluded_not_raised(self) -> None:
        """``None`` predictions (insufficient history) are an expected outcome."""
        preds = [_pred("AAA", 1.5), _pred("BBB", None), _pred("CCC", 2.0)]
        scores, excluded = validate_and_collect_finite_scores(
            preds, requested_count=3, min_predictions=2
        )
        assert scores == {"AAA": 1.5, "CCC": 2.0}
        assert excluded == ["BBB"]

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_non_finite_raises(self, bad_value: float) -> None:
        """NaN/+inf/-inf MUST raise -- they break rank-band ordering."""
        preds = [_pred("AAA", 1.0), _pred("BAD", bad_value), _pred("CCC", 2.0)]
        with pytest.raises(RuntimeError, match="non-finite scores"):
            validate_and_collect_finite_scores(
                preds, requested_count=3, min_predictions=1
            )

    def test_non_finite_error_lists_offending_symbols(self) -> None:
        """The error message names the offenders for ops debugging."""
        preds = [
            _pred("AAA", 1.0),
            _pred("BAD1", math.nan),
            _pred("BAD2", math.inf),
        ]
        with pytest.raises(RuntimeError) as exc:
            validate_and_collect_finite_scores(
                preds, requested_count=3, min_predictions=1
            )
        assert "BAD1" in str(exc.value)
        assert "BAD2" in str(exc.value)

    def test_below_min_predictions_raises(self) -> None:
        """Floor check fires when too few finite scores survive."""
        preds = [_pred("AAA", 1.0), _pred("BBB", None), _pred("CCC", None)]
        with pytest.raises(RuntimeError, match="below"):
            validate_and_collect_finite_scores(
                preds, requested_count=3, min_predictions=2
            )

    def test_below_floor_message_includes_counts(self) -> None:
        """Operator can read "got X / Y" + "Excluded=N" from the error text."""
        preds = [_pred("AAA", 1.0), _pred("BBB", None)]
        with pytest.raises(RuntimeError) as exc:
            validate_and_collect_finite_scores(
                preds, requested_count=2, min_predictions=2
            )
        msg = str(exc.value)
        assert "1 valid predictions" in msg
        assert "2 requested" in msg
        assert "min_predictions=2" in msg
        assert "Excluded=1" in msg

    def test_exactly_at_min_predictions_passes(self) -> None:
        """Boundary: ``len(scores) == min_predictions`` is allowed."""
        preds = [_pred("AAA", 1.0), _pred("BBB", None), _pred("CCC", 0.5)]
        scores, excluded = validate_and_collect_finite_scores(
            preds, requested_count=3, min_predictions=2
        )
        assert len(scores) == 2
        assert excluded == ["BBB"]

    def test_non_finite_takes_priority_over_floor(self) -> None:
        """If predictions contain a NaN AND would also be below the floor,
        the non-finite error fires first -- it is the more specific
        failure (model corruption) and the operator should fix it
        before retrying."""
        preds = [_pred("AAA", math.nan), _pred("BBB", None)]
        with pytest.raises(RuntimeError, match="non-finite"):
            validate_and_collect_finite_scores(
                preds, requested_count=2, min_predictions=5
            )

    def test_empty_predictions_below_floor(self) -> None:
        """Empty input is below any positive floor; this raises cleanly."""
        with pytest.raises(RuntimeError, match="0 valid"):
            validate_and_collect_finite_scores([], requested_count=0, min_predictions=1)
