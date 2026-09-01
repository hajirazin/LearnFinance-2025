"""PPO training VIX repair boundary and manifest provenance."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from brain_api.core.ppo_discovery.pipeline import (
    price_manifest_with_vix_audit,
    repair_ppo_training_vix,
)
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    weekly_trade_clock,
)
from brain_api.core.sac.market_sessions import xnys_session_dates
from brain_api.core.vix_fallback import VixFallbackAudit


def _frame(dates: list[date], value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": value,
            "high": value,
            "low": value,
            "close": value,
            "volume": 0.0,
        },
        index=pd.DatetimeIndex(dates),
    )


def test_august_window_uses_last_hmm_week_and_ignores_trailing_holes(
    monkeypatch,
) -> None:
    price_start = date(2026, 7, 1)
    clock = weekly_trade_clock(price_start, date(2026, 8, 28))
    transition_cutoffs = actor_cutoff_datetimes(clock)[:-1]
    assert transition_cutoffs[-1].date() == date(2026, 8, 14)
    hmm_weeks = [cutoff.date() for cutoff in transition_cutoffs]
    dates = xnys_session_dates(price_start, date(2026, 8, 28))
    prices = {"SPY": _frame(dates, 630.0), "^VIX": _frame(dates, 16.0)}
    prices["^VIX"].loc[["2026-08-21", "2026-08-28"], "close"] = float("nan")
    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: pytest.fail("post-HMM VIX holes must not call Cboe"),
    )

    result = repair_ppo_training_vix(
        prices,
        price_start=price_start,
        hmm_weeks=hmm_weeks,
    )

    assert result.audit.fallback_dates == ()


def test_ppo_training_repairs_only_through_max_hmm_week(monkeypatch) -> None:
    price_start = date(2026, 8, 3)
    hmm_weeks = [date(2026, 8, 7), date(2026, 8, 14)]
    required = xnys_session_dates(price_start, max(hmm_weeks))
    prices = {"SPY": _frame(required, 630.0), "^VIX": _frame(required[:-1], 16.0)}
    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: _frame([required[-1]], 15.5),
    )

    result = repair_ppo_training_vix(
        prices,
        price_start=price_start,
        hmm_weeks=hmm_weeks,
    )

    assert result.audit.fallback_dates == ("2026-08-14",)
    assert result.prices["^VIX"].loc["2026-08-14", "close"] == 15.5


def test_ppo_price_manifest_persists_vix_audit() -> None:
    audit = VixFallbackAudit(
        fallback_provider="cboe",
        fallback_dates=("2026-08-14",),
        source_url="https://example.test/vix.csv",
        retrieved_at="2026-08-17T13:00:00+00:00",
    )

    manifest = price_manifest_with_vix_audit(
        {"complete": True, "source": "yfinance"}, audit
    )

    assert manifest["source"] == "yfinance"
    assert manifest["vix_provenance"] == audit.to_dict()
