"""SAC training VIX repair is clipped to consumed actor cutoffs."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from brain_api.core.sac.trade_clock import build_sac_weekly_trade_clock
from brain_api.core.vix_fallback import VixFallbackAudit, VixFallbackError
from brain_api.routes.training.sac._market_history import (
    record_sac_vix_audit,
    repair_and_extract_sac_market_history,
)


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


def test_august_window_ignores_unconsumed_vix_holes(monkeypatch) -> None:
    clock = build_sac_weekly_trade_clock(date(2026, 7, 1), date(2026, 8, 28))
    completed_through = clock.transition_actor_cutoffs[-1].date()
    assert completed_through == date(2026, 8, 14)
    dates = pd.bdate_range("2026-07-01", "2026-08-28").date.tolist()
    prices = {"SPY": _frame(dates, 630.0), "^VIX": _frame(dates, 16.0)}
    prices["^VIX"].loc[["2026-08-21", "2026-08-28"], "close"] = float("nan")
    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: pytest.fail("unconsumed VIX holes must not call Cboe"),
    )

    repaired, market_dates, _spy, _vix, audit = repair_and_extract_sac_market_history(
        prices,
        start_date=date(2026, 7, 1),
        completed_through=completed_through,
    )

    assert market_dates[-1] == date(2026, 8, 14)
    assert pd.isna(repaired["^VIX"].loc["2026-08-21", "close"])
    assert audit.fallback_dates == ()


def test_required_sac_vix_gap_repairs_and_unresolved_gap_fails(monkeypatch) -> None:
    dates = pd.bdate_range("2026-07-01", "2026-08-14").date.tolist()
    prices = {"SPY": _frame(dates, 630.0), "^VIX": _frame(dates[:-1], 16.0)}
    cboe = _frame([dates[-1]], 15.5)

    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history", lambda: cboe
    )
    repaired, _dates, _spy, _vix, audit = repair_and_extract_sac_market_history(
        prices, start_date=dates[0], completed_through=dates[-1]
    )
    assert audit.fallback_dates == ("2026-08-14",)
    assert repaired["^VIX"].loc["2026-08-14", "close"] == 15.5

    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: _frame(dates[:-1], 15.5),
    )
    with pytest.raises(VixFallbackError, match="2026-08-14"):
        repair_and_extract_sac_market_history(
            prices, start_date=dates[0], completed_through=dates[-1]
        )


def test_vix_audit_is_added_to_every_sac_candidate() -> None:
    candidates = [
        SimpleNamespace(result=SimpleNamespace(audit_metadata={})) for _ in range(3)
    ]
    experiment = SimpleNamespace(candidates=candidates)
    audit = VixFallbackAudit(
        fallback_provider="cboe",
        fallback_dates=("2026-08-14",),
        source_url="https://example.test/vix.csv",
        retrieved_at="2026-08-17T13:00:00+00:00",
    )

    record_sac_vix_audit(experiment, audit)

    for candidate in candidates:
        assert candidate.result.audit_metadata["vix_fallback"] == audit.to_dict()
