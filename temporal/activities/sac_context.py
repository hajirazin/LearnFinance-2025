"""Strict construction of the raw evidence bundle used for SAC decisions.

Momentum math is duplicated (not imported) from
``brain_api.core.sac.momentum_signals`` because Temporal and brain_api
are separate deployable services with independent dependency
environments (see ``temporal/pyproject.toml`` -- no brain_api
dependency). The formulas are simple, single-line arithmetic; keep the
bar counts (5/20/21/252) and encodings identical to brain_api's train
path if either changes.
"""

import math
from datetime import date
from itertools import pairwise

from models import (
    AdjustedClosesResponse,
    MarketHistoryResponse,
    NewsSignalResponse,
    PatchTSTInferenceResponse,
)

# momentum_1w = P_t/P_t-5 - 1 (5 trading bars)
MOM_1W_BARS = 5
# momentum_4w = P_t/P_t-20 - 1 (20 trading bars)
MOM_4W_BARS = 20
# momentum_12_1 = P_t-21/P_t-252 - 1 (skip 21 bars, then 252-bar lookback)
MOM_12_1_SKIP_BARS = 21
MOM_12_1_LOOKBACK_BARS = 252
REALIZED_VOL_RETURN_BARS = 20


def _exact_per_symbol(items: list, symbols: list[str], *, field: str) -> dict:
    """Index a response by symbol and reject missing, extra, or duplicate rows."""
    indexed = {}
    duplicates = []
    for item in items:
        if item.symbol in indexed:
            duplicates.append(item.symbol)
        indexed[item.symbol] = item
    expected = set(symbols)
    actual = set(indexed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if duplicates or missing or extra:
        raise ValueError(
            f"{field} symbol mismatch: missing={missing}, extra={extra}, "
            f"duplicates={sorted(set(duplicates))}"
        )
    return indexed


def _required_finite(value: float | None, *, field: str) -> float:
    """Return a finite float or fail the canonical SAC feature snapshot."""
    if value is None or not math.isfinite(value):
        raise ValueError(f"{field} is required and must be finite")
    return float(value)


def _normalize_news_items(news) -> list:
    """Accept either a ``NewsSignalResponse`` wrapper or a raw list of items."""
    return news.per_symbol if hasattr(news, "per_symbol") else news


def _normalize_patchtst_items(forecasts) -> list:
    """Accept either a ``PatchTSTInferenceResponse`` wrapper or a raw list."""
    return forecasts.predictions if hasattr(forecasts, "predictions") else forecasts


def _price_at(closes: list[float], index: int, *, field: str) -> float:
    if index < 0 or index >= len(closes):
        raise ValueError(
            f"{field} requires more price history (index {index}, have {len(closes)})"
        )
    value = closes[index]
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{field} price at index {index} must be finite and positive, got {value!r}"
        )
    return float(value)


def _compute_momentum_1w(closes: list[float], *, as_of_index: int) -> float:
    """Simple 1-week return: P[t] / P[t-5] - 1 (5 trading bars)."""
    p_t = _price_at(closes, as_of_index, field="momentum_1w.P_t")
    p_lag = _price_at(closes, as_of_index - MOM_1W_BARS, field="momentum_1w.P_t-5")
    return p_t / p_lag - 1.0


def _compute_momentum_4w(closes: list[float], *, as_of_index: int) -> float:
    """Simple 4-week return: P[t] / P[t-20] - 1 (20 trading bars)."""
    p_t = _price_at(closes, as_of_index, field="momentum_4w.P_t")
    p_lag = _price_at(closes, as_of_index - MOM_4W_BARS, field="momentum_4w.P_t-20")
    return p_t / p_lag - 1.0


def _compute_momentum_12_1(closes: list[float], *, as_of_index: int) -> float:
    """Classic 12-1 momentum: P[t-21] / P[t-252] - 1 (skip ~1mo, ~12mo lookback)."""
    p_skip = _price_at(
        closes, as_of_index - MOM_12_1_SKIP_BARS, field="momentum_12_1.P_t-21"
    )
    p_lookback = _price_at(
        closes,
        as_of_index - MOM_12_1_LOOKBACK_BARS,
        field="momentum_12_1.P_t-252",
    )
    return p_skip / p_lookback - 1.0


def _compute_realized_vol_20d(closes: list[float], *, as_of_index: int) -> float:
    """Annualized sample std of the last 20 adjusted-close log returns."""
    start = as_of_index - REALIZED_VOL_RETURN_BARS
    prices = [
        _price_at(closes, index, field="realized_vol_20d.price")
        for index in range(start, as_of_index + 1)
    ]
    returns = [math.log(current / previous) for previous, current in pairwise(prices)]
    mean = sum(returns) / len(returns)
    sample_variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(sample_variance) * math.sqrt(252.0)


def build_sac_feature_bundle(
    *,
    symbols: list[str],
    as_of_date: str | None = None,
    decision_date: str | None = None,
    news: NewsSignalResponse | list,
    patchtst: PatchTSTInferenceResponse | list | None = None,
    patchtst_forecasts: PatchTSTInferenceResponse | list | None = None,
    prices: AdjustedClosesResponse,
    market: MarketHistoryResponse,
) -> dict:
    """Build the point-in-time raw evidence bundle consumed by Brain.

    ``as_of_date``/``decision_date`` are aliases (same meaning); callers
    should use ``as_of_date``. ``patchtst``/``patchtst_forecasts`` are
    likewise aliases. Temporal deliberately does not send derived momentum,
    volatility, ranks, eligibility, or HMM probabilities. Brain owns that
    business math and validates the raw evidence. The duplicated momentum
    helpers above remain only for train/live formula-parity tests.
    """
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("SAC symbols must be non-empty and unique")

    resolved_date = as_of_date if as_of_date is not None else decision_date
    if resolved_date is None:
        raise ValueError("build_sac_feature_bundle requires as_of_date/decision_date")
    forecasts_source = patchtst if patchtst is not None else patchtst_forecasts
    if forecasts_source is None:
        raise ValueError(
            "build_sac_feature_bundle requires patchtst/patchtst_forecasts"
        )
    news_by_symbol = _exact_per_symbol(
        _normalize_news_items(news), symbols, field="news"
    )
    patchtst_by_symbol = _exact_per_symbol(
        _normalize_patchtst_items(forecasts_source), symbols, field="patchtst_forecasts"
    )
    if prices.as_of_date != resolved_date:
        raise ValueError(
            f"adjusted closes as_of_date {prices.as_of_date!r} does not match "
            f"decision date {resolved_date!r}"
        )
    if date.fromisoformat(market.as_of_date) >= date.fromisoformat(resolved_date):
        raise ValueError(
            f"market history as_of_date {market.as_of_date!r} must be before "
            f"decision date {resolved_date!r}"
        )

    news_sentiment: dict[str, float] = {}
    news_article_counts: dict[str, int] = {}
    for symbol in symbols:
        news_observation = news_by_symbol[symbol]
        if news_observation.article_count_used < 0:
            raise ValueError(f"news[{symbol}].article_count_used cannot be negative")
        news_sentiment[symbol] = _required_finite(
            news_observation.sentiment_score,
            field=f"news[{symbol}].sentiment_score",
        )
        news_article_counts[symbol] = news_observation.article_count_used

    news_provenance = {
        "as_of_date": getattr(news, "as_of_date", resolved_date),
        "run_id": getattr(news, "run_id", None),
        "attempt": getattr(news, "attempt", None),
        "from_cache": getattr(news, "from_cache", None),
    }
    patchtst_provenance = {
        "as_of_date": getattr(forecasts_source, "as_of_date", resolved_date),
        "model_version": getattr(forecasts_source, "model_version", None),
    }

    return {
        "symbols": symbols,
        "adjusted_closes": {
            symbol: prices.adjusted_closes.get(symbol, []) for symbol in symbols
        },
        "news_sentiment": news_sentiment,
        "news_article_counts": news_article_counts,
        "patchtst_forecasts": {
            symbol: _required_finite(
                patchtst_by_symbol[symbol].predicted_weekly_return_pct,
                field=f"patchtst_forecasts[{symbol}]",
            )
            / 100.0
            for symbol in symbols
            if patchtst_by_symbol[symbol].predicted_weekly_return_pct is not None
        },
        "execution_prices": prices.execution_prices,
        "market_history": [row.model_dump() for row in market.rows],
        "provenance": {
            "as_of_date": resolved_date,
            "adjusted_closes": prices.provenance,
            "news": news_provenance,
            "patchtst": patchtst_provenance,
            "market_history": market.provenance,
        },
    }
