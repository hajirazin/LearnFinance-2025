"""Strict construction of the feature bundle used for SAC decisions."""

import math
from datetime import date

from models import (
    FundamentalsResponse,
    NewsSignalResponse,
    PatchTSTInferenceResponse,
)


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


def build_sac_feature_bundle(
    *,
    symbols: list[str],
    as_of_date: str,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    patchtst: PatchTSTInferenceResponse,
) -> dict:
    """Build the strict feature bundle Brain uses for the SAC actor state."""
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("SAC symbols must be non-empty and unique")
    decision_date = date.fromisoformat(as_of_date)
    news_by_symbol = _exact_per_symbol(news.per_symbol, symbols, field="news")
    fundamentals_by_symbol = _exact_per_symbol(
        fundamentals.per_symbol, symbols, field="fundamentals"
    )
    patchtst_by_symbol = _exact_per_symbol(
        patchtst.predictions, symbols, field="patchtst_forecasts"
    )

    signals: dict[str, dict[str, float]] = {}
    filing_provenance: dict[str, dict[str, str | None]] = {}
    for symbol in symbols:
        news_observation = news_by_symbol[symbol]
        fundamental_observation = fundamentals_by_symbol[symbol]
        if fundamental_observation.error or fundamental_observation.ratios is None:
            detail = fundamental_observation.error or "missing ratios"
            raise ValueError(f"fundamentals[{symbol}] unavailable: {detail}")
        ratios = fundamental_observation.ratios
        if ratios.filing_available_date is None:
            raise ValueError(
                f"fundamentals[{symbol}].filing_available_date is required"
            )
        filing_date = date.fromisoformat(ratios.filing_available_date)
        fundamental_age = (decision_date - filing_date).days
        if fundamental_age < 0:
            raise ValueError(
                f"fundamentals[{symbol}] filing date is after decision date"
            )
        if news_observation.article_count_used < 0:
            raise ValueError(f"news[{symbol}].article_count_used cannot be negative")
        signals[symbol] = {
            "news_sentiment": _required_finite(
                news_observation.sentiment_score,
                field=f"news[{symbol}].sentiment_score",
            ),
            "news_coverage": min(news_observation.article_count_used / 3.0, 1.0),
            "gross_margin": _required_finite(
                ratios.gross_margin, field=f"fundamentals[{symbol}].gross_margin"
            ),
            "operating_margin": _required_finite(
                ratios.operating_margin,
                field=f"fundamentals[{symbol}].operating_margin",
            ),
            "current_ratio": _required_finite(
                ratios.current_ratio, field=f"fundamentals[{symbol}].current_ratio"
            ),
            "debt_to_equity": _required_finite(
                ratios.debt_to_equity,
                field=f"fundamentals[{symbol}].debt_to_equity",
            ),
            "fundamental_age": float(fundamental_age),
        }
        filing_provenance[symbol] = {
            "fiscal_period_end": ratios.fiscal_period_end,
            "filing_available_date": ratios.filing_available_date,
            "filing_accession_number": ratios.filing_accession_number,
            "filing_form": ratios.filing_form,
            "filing_source": ratios.filing_source,
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
            "as_of_date": as_of_date,
            "news": {
                "as_of_date": news.as_of_date,
                "run_id": news.run_id,
                "attempt": news.attempt,
                "from_cache": news.from_cache,
            },
            "fundamentals": {
                "as_of_date": fundamentals.as_of_date,
                "filings": filing_provenance,
            },
            "patchtst": {
                "as_of_date": patchtst.as_of_date,
                "model_version": patchtst.model_version,
            },
        },
    }
