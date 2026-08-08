"""Strict construction of the feature bundle used for SAC decisions.

Momentum + earnings-yield math is duplicated (not imported) from
``brain_api.core.sac.momentum_signals`` because Temporal and brain_api
are separate deployable services with independent dependency
environments (see ``temporal/pyproject.toml`` -- no brain_api
dependency). The formulas are simple, single-line arithmetic; keep the
bar counts (5/20/21/252) and encodings identical to brain_api's train
path if either changes.
"""

import math
from datetime import date

from models import (
    FundamentalsResponse,
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


def _normalize_fundamentals(fundamentals) -> dict:
    """Return ``{symbol: ratios}``, unwrapping a ``FundamentalsResponse`` if given.

    Error/missing-ratio per-symbol rows raise immediately (fail-loud);
    a raw ``{symbol: ratios}`` dict (already ratios, no error wrapper)
    passes through unchanged.
    """
    if not hasattr(fundamentals, "per_symbol"):
        return dict(fundamentals)
    resolved: dict = {}
    for item in fundamentals.per_symbol:
        if item.error or item.ratios is None:
            detail = item.error or "missing ratios"
            raise ValueError(f"fundamentals[{item.symbol}] unavailable: {detail}")
        resolved[item.symbol] = item.ratios
    return resolved


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


def _compute_earnings_yield(*, eps_diluted: float | None, as_of_close: float) -> float:
    """Earnings yield: eps_diluted / as_of_close.

    ``eps_diluted`` must be SEC point-in-time diluted EPS (Basic
    fallback) from ``/signals/fundamentals`` -- never raw P/E, and a
    distinct metric from the universe screening module's
    operating-income-over-enterprise-value ratio (different purpose;
    see AGENTS.md universe pipeline invariants).
    """
    if eps_diluted is None or not math.isfinite(eps_diluted):
        raise ValueError(
            f"earnings_yield requires a finite eps_diluted, got {eps_diluted!r}"
        )
    if as_of_close is None or not math.isfinite(as_of_close) or as_of_close <= 0:
        raise ValueError(
            f"earnings_yield requires a positive finite close, got {as_of_close!r}"
        )
    return eps_diluted / as_of_close


def build_sac_feature_bundle(
    *,
    symbols: list[str],
    as_of_date: str | None = None,
    decision_date: str | None = None,
    news: NewsSignalResponse | list,
    fundamentals: FundamentalsResponse | dict,
    patchtst: PatchTSTInferenceResponse | list | None = None,
    patchtst_forecasts: PatchTSTInferenceResponse | list | None = None,
    closes: dict[str, list[float]] | None = None,
) -> dict:
    """Build the strict feature bundle Brain uses for the SAC actor state.

    ``as_of_date``/``decision_date`` are aliases (same meaning); callers
    should use ``as_of_date``. ``patchtst``/``patchtst_forecasts`` are
    likewise aliases. ``closes`` is required: ``{symbol: [daily closes,
    oldest first, most recent last]}`` with at least
    ``MOM_12_1_SKIP_BARS + MOM_12_1_LOOKBACK_BARS`` (273) bars, used for
    momentum_1w/4w/12_1. Fail-loud everywhere: no silent zero-fill for
    insufficient bars, missing/non-finite EPS, or non-finite prices.
    """
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("SAC symbols must be non-empty and unique")

    resolved_date = as_of_date if as_of_date is not None else decision_date
    if resolved_date is None:
        raise ValueError("build_sac_feature_bundle requires as_of_date/decision_date")
    decision_date_value = date.fromisoformat(resolved_date)

    forecasts_source = patchtst if patchtst is not None else patchtst_forecasts
    if forecasts_source is None:
        raise ValueError(
            "build_sac_feature_bundle requires patchtst/patchtst_forecasts"
        )
    if closes is None:
        raise ValueError("build_sac_feature_bundle requires closes for momentum")

    news_by_symbol = _exact_per_symbol(
        _normalize_news_items(news), symbols, field="news"
    )
    fundamentals_by_symbol = _normalize_fundamentals(fundamentals)
    missing_fundamentals = sorted(set(symbols) - set(fundamentals_by_symbol))
    extra_fundamentals = sorted(set(fundamentals_by_symbol) - set(symbols))
    if missing_fundamentals or extra_fundamentals:
        raise ValueError(
            f"fundamentals symbol mismatch: missing={missing_fundamentals}, "
            f"extra={extra_fundamentals}"
        )
    patchtst_by_symbol = _exact_per_symbol(
        _normalize_patchtst_items(forecasts_source), symbols, field="patchtst_forecasts"
    )
    missing_closes = sorted(set(symbols) - set(closes))
    if missing_closes:
        raise ValueError(f"closes missing for symbols: {missing_closes}")

    signals: dict[str, dict[str, float]] = {}
    filing_provenance: dict[str, dict[str, str | None]] = {}
    for symbol in symbols:
        news_observation = news_by_symbol[symbol]
        ratios = fundamentals_by_symbol[symbol]
        if ratios.filing_available_date is None:
            raise ValueError(
                f"fundamentals[{symbol}].filing_available_date is required"
            )
        filing_date = date.fromisoformat(ratios.filing_available_date)
        fundamental_age = (decision_date_value - filing_date).days
        if fundamental_age < 0:
            raise ValueError(
                f"fundamentals[{symbol}] filing date is after decision date"
            )
        if news_observation.article_count_used < 0:
            raise ValueError(f"news[{symbol}].article_count_used cannot be negative")

        symbol_closes = closes[symbol]
        if len(symbol_closes) < MOM_12_1_LOOKBACK_BARS + 1:
            raise ValueError(
                f"closes[{symbol}] has {len(symbol_closes)} bars; need >= "
                f"{MOM_12_1_LOOKBACK_BARS + 1} for momentum_12_1"
            )
        as_of_index = len(symbol_closes) - 1
        as_of_close = _price_at(symbol_closes, as_of_index, field=f"closes[{symbol}]")

        signals[symbol] = {
            "news_sentiment": _required_finite(
                news_observation.sentiment_score,
                field=f"news[{symbol}].sentiment_score",
            ),
            "news_coverage": min(news_observation.article_count_used / 3.0, 1.0),
            "gross_margin": _required_finite(
                ratios.gross_margin, field=f"fundamentals[{symbol}].gross_margin"
            ),
            "debt_to_equity": _required_finite(
                ratios.debt_to_equity,
                field=f"fundamentals[{symbol}].debt_to_equity",
            ),
            "fundamental_age": float(fundamental_age),
            "momentum_1w": _compute_momentum_1w(symbol_closes, as_of_index=as_of_index),
            "momentum_4w": _compute_momentum_4w(symbol_closes, as_of_index=as_of_index),
            "momentum_12_1": _compute_momentum_12_1(
                symbol_closes, as_of_index=as_of_index
            ),
            "earnings_yield": _compute_earnings_yield(
                eps_diluted=getattr(ratios, "eps_diluted", None),
                as_of_close=as_of_close,
            ),
        }
        filing_provenance[symbol] = {
            "fiscal_period_end": ratios.fiscal_period_end,
            "filing_available_date": ratios.filing_available_date,
            "filing_accession_number": ratios.filing_accession_number,
            "filing_form": ratios.filing_form,
            "filing_source": ratios.filing_source,
        }

    news_provenance = {
        "as_of_date": getattr(news, "as_of_date", resolved_date),
        "run_id": getattr(news, "run_id", None),
        "attempt": getattr(news, "attempt", None),
        "from_cache": getattr(news, "from_cache", None),
    }
    fundamentals_provenance = {
        "as_of_date": getattr(fundamentals, "as_of_date", resolved_date),
        "filings": filing_provenance,
    }
    patchtst_provenance = {
        "as_of_date": getattr(forecasts_source, "as_of_date", resolved_date),
        "model_version": getattr(forecasts_source, "model_version", None),
    }

    return {
        "symbols": symbols,
        "signals": signals,
        "patchtst_forecasts": {
            symbol: _required_finite(
                patchtst_by_symbol[symbol].predicted_weekly_return_pct,
                field=f"patchtst_forecasts[{symbol}]",
            )
            / 100.0
            for symbol in symbols
        },
        "provenance": {
            "as_of_date": resolved_date,
            "news": news_provenance,
            "fundamentals": fundamentals_provenance,
            "patchtst": patchtst_provenance,
        },
    }
