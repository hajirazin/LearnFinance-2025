"""Tests for brain_api.universe.stock_filter pure functions."""

import pytest

from brain_api.universe.stock_filter import (
    _percentile_rank,
    apply_junk_filter,
    compute_factor_scores,
)

# ============================================================================
# _percentile_rank tests
# ============================================================================


class TestPercentileRank:
    """Tests for _percentile_rank helper."""

    def test_empty_list(self) -> None:
        assert _percentile_rank([]) == []

    def test_all_none(self) -> None:
        assert _percentile_rank([None, None, None]) == [None, None, None]

    def test_single_value(self) -> None:
        result = _percentile_rank([5.0])
        assert result == [0.0]

    def test_two_values_ascending(self) -> None:
        result = _percentile_rank([1.0, 2.0])
        assert result == [0.0, 1.0]

    def test_two_values_descending(self) -> None:
        result = _percentile_rank([2.0, 1.0])
        assert result == [1.0, 0.0]

    def test_three_values(self) -> None:
        result = _percentile_rank([10.0, 30.0, 20.0])
        assert result == [0.0, 1.0, 0.5]

    def test_with_none_mixed_in(self) -> None:
        result = _percentile_rank([10.0, None, 30.0, None, 20.0])
        assert result[0] == 0.0
        assert result[1] is None
        assert result[2] == 1.0
        assert result[3] is None
        assert result[4] == 0.5

    def test_all_same_values(self) -> None:
        result = _percentile_rank([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]

    def test_result_bounded_zero_to_one(self) -> None:
        result = _percentile_rank([1.0, 2.0, 3.0, 4.0, 5.0])
        for r in result:
            assert r is not None
            assert 0.0 <= r <= 1.0


# ============================================================================
# apply_junk_filter tests
# ============================================================================


def _make_holding(symbol: str, **kwargs: object) -> dict:
    """Helper to create a minimal holding dict."""
    return {"symbol": symbol, "name": f"{symbol} Corp", "max_weight": 1.0, **kwargs}


def _good_metrics() -> dict:
    """Metrics that pass all junk filter criteria."""
    return {
        "roe": 0.15,
        "price": 150.0,
        "sma200": 140.0,
        "beta": 1.2,
        "gross_margin": 0.4,
        "roic": 0.12,
        "earnings_yield": 0.05,
        "six_month_return": 0.08,
    }


class TestApplyJunkFilter:
    """Tests for apply_junk_filter."""

    def test_stock_passing_all_criteria(self) -> None:
        holdings = [_make_holding("AAPL")]
        metrics = {"AAPL": _good_metrics()}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 1
        assert len(failed) == 0
        assert passed[0]["symbol"] == "AAPL"
        assert "metrics" in passed[0]

    def test_stock_fails_roe_zero(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["roe"] = 0
        metrics = {"BAD": m}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0
        assert len(failed) == 1
        assert "fail_reasons" in failed[0]

    def test_stock_fails_roe_negative(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["roe"] = -0.05
        metrics = {"BAD": m}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0
        assert len(failed) == 1

    def test_stock_fails_roe_none(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["roe"] = None
        metrics = {"BAD": m}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0
        assert len(failed) == 1

    def test_stock_fails_price_below_sma200(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["price"] = 130.0
        m["sma200"] = 140.0
        metrics = {"BAD": m}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0
        assert len(failed) == 1

    def test_stock_fails_price_equal_sma200(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["price"] = 140.0
        m["sma200"] = 140.0
        metrics = {"BAD": m}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0
        assert len(failed) == 1

    def test_stock_fails_price_none(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["price"] = None
        metrics = {"BAD": m}
        passed, _failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0

    def test_stock_fails_sma200_none(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["sma200"] = None
        metrics = {"BAD": m}
        passed, _failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0

    def test_stock_fails_beta_too_high(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["beta"] = 2.0
        metrics = {"BAD": m}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0
        assert len(failed) == 1

    def test_stock_fails_beta_none(self) -> None:
        holdings = [_make_holding("BAD")]
        m = _good_metrics()
        m["beta"] = None
        metrics = {"BAD": m}
        passed, _failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0

    def test_stock_with_missing_metrics_entry(self) -> None:
        """Stock not in metrics dict at all should fail."""
        holdings = [_make_holding("UNKNOWN")]
        metrics: dict[str, dict] = {}
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 0
        assert len(failed) == 1

    def test_empty_holdings(self) -> None:
        passed, failed = apply_junk_filter([], {})
        assert passed == []
        assert failed == []

    def test_multiple_stocks_mixed(self) -> None:
        holdings = [_make_holding("GOOD"), _make_holding("BAD")]
        metrics = {
            "GOOD": _good_metrics(),
            "BAD": {**_good_metrics(), "roe": -0.1},
        }
        passed, failed = apply_junk_filter(holdings, metrics)
        assert len(passed) == 1
        assert len(failed) == 1
        assert passed[0]["symbol"] == "GOOD"
        assert failed[0]["symbol"] == "BAD"

    def test_multiple_fail_reasons_accumulated(self) -> None:
        holdings = [_make_holding("BAD")]
        metrics = {"BAD": {"roe": -0.1, "price": None, "sma200": None, "beta": 3.0}}
        _, failed = apply_junk_filter(holdings, metrics)
        assert len(failed[0]["fail_reasons"]) == 3


# ============================================================================
# compute_factor_scores tests
# ============================================================================


def _make_scored_holding(
    symbol: str,
    six_month_return: float | None = 0.1,
    roic: float | None = 0.15,
    gross_margin: float | None = 0.4,
    earnings_yield: float | None = 0.05,
) -> dict:
    """Helper to create a holding with metrics for scoring."""
    return {
        "symbol": symbol,
        "name": f"{symbol} Corp",
        "metrics": {
            "six_month_return": six_month_return,
            "roic": roic,
            "gross_margin": gross_margin,
            "earnings_yield": earnings_yield,
        },
    }


class TestComputeFactorScores:
    """Tests for compute_factor_scores."""

    def test_empty_holdings(self) -> None:
        assert compute_factor_scores([]) == []

    def test_single_stock(self) -> None:
        holdings = [_make_scored_holding("AAPL")]
        result = compute_factor_scores(holdings)
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["factor_score"] is not None
        assert "factor_components" in result[0]

    def test_sorted_by_score_descending(self) -> None:
        holdings = [
            _make_scored_holding(
                "LOW", six_month_return=0.01, roic=0.05, earnings_yield=0.01
            ),
            _make_scored_holding(
                "HIGH", six_month_return=0.20, roic=0.25, earnings_yield=0.10
            ),
            _make_scored_holding(
                "MID", six_month_return=0.10, roic=0.15, earnings_yield=0.05
            ),
        ]
        result = compute_factor_scores(holdings)
        assert result[0]["symbol"] == "HIGH"
        assert result[-1]["symbol"] == "LOW"
        scores = [r["factor_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_none_components_ranked_last(self) -> None:
        holdings = [
            _make_scored_holding(
                "GOOD", six_month_return=0.1, roic=0.15, earnings_yield=0.05
            ),
            _make_scored_holding(
                "BAD", six_month_return=None, roic=0.15, earnings_yield=0.05
            ),
        ]
        result = compute_factor_scores(holdings)
        assert result[0]["symbol"] == "GOOD"
        assert result[0]["factor_score"] is not None
        assert result[1]["symbol"] == "BAD"
        assert result[1]["factor_score"] is None

    def test_roic_preferred_over_gross_margin(self) -> None:
        """When ROIC is available, it should be used instead of gross_margin."""
        holdings = [
            _make_scored_holding("A", roic=0.30, gross_margin=0.10),
        ]
        result = compute_factor_scores(holdings)
        fc = result[0]["factor_components"]
        assert fc["quality_raw"] == 0.30

    def test_gross_margin_fallback_when_roic_none(self) -> None:
        """When ROIC is None, gross_margin should be used."""
        holdings = [
            _make_scored_holding("A", roic=None, gross_margin=0.45),
        ]
        result = compute_factor_scores(holdings)
        fc = result[0]["factor_components"]
        assert fc["quality_raw"] == 0.45

    def test_factor_components_present(self) -> None:
        holdings = [_make_scored_holding("AAPL")]
        result = compute_factor_scores(holdings)
        fc = result[0]["factor_components"]
        expected_keys = {
            "momentum_6m_return",
            "momentum_pct",
            "quality_raw",
            "quality_pct",
            "value_earnings_yield",
            "value_pct",
        }
        assert set(fc.keys()) == expected_keys

    def test_score_formula_correctness(self) -> None:
        """With 3 stocks, verify the score = 0.4*M + 0.3*Q + 0.3*V."""
        holdings = [
            _make_scored_holding(
                "A", six_month_return=0.10, roic=0.10, earnings_yield=0.04
            ),
            _make_scored_holding(
                "B", six_month_return=0.20, roic=0.20, earnings_yield=0.08
            ),
            _make_scored_holding(
                "C", six_month_return=0.30, roic=0.30, earnings_yield=0.12
            ),
        ]
        result = compute_factor_scores(holdings)
        # C has rank 1.0 in all three factors -> score = 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
        assert result[0]["symbol"] == "C"
        assert result[0]["factor_score"] == pytest.approx(1.0, abs=1e-6)
        # A has rank 0.0 in all three -> score = 0.0
        assert result[-1]["symbol"] == "A"
        assert result[-1]["factor_score"] == pytest.approx(0.0, abs=1e-6)
        # B has rank 0.5 in all three -> score = 0.4*0.5 + 0.3*0.5 + 0.3*0.5 = 0.5
        assert result[1]["symbol"] == "B"
        assert result[1]["factor_score"] == pytest.approx(0.5, abs=1e-6)

    def test_preserves_original_holding_fields(self) -> None:
        holdings = [
            {
                "symbol": "AAPL",
                "name": "Apple",
                "max_weight": 5.0,
                "sources": ["SPUS"],
                "metrics": _good_metrics(),
            }
        ]
        result = compute_factor_scores(holdings)
        assert result[0]["name"] == "Apple"
        assert result[0]["max_weight"] == 5.0
        assert result[0]["sources"] == ["SPUS"]
