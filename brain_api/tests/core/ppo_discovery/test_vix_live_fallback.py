"""Live PPO VIX repair, provenance, and HTTP failure contracts."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

from brain_api.main import app as _app  # noqa: F401 - initialize route imports
from brain_api.routes.signals.ppo_discovery import PPOStateRequest, build_state


def _frame(dates: list[str], value: float) -> pd.DataFrame:
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


def _request() -> PPOStateRequest:
    return PPOStateRequest(
        as_of="2026-08-17T09:00:00-04:00",
        run_id="paper:halal_new:2026-08-17",
        attempt=1,
        current_weights={"CASH": 1.0},
    )


def _common_patches(spy_vix: dict[str, pd.DataFrame]):
    snapshot = SimpleNamespace(sorted_symbols=("AAPL",))
    service = MagicMock()
    service.materialize.return_value = ([], [])
    artifacts = SimpleNamespace(feature_scalers={}, regime_hmm={"schema_version": 3})
    return (
        patch(
            "brain_api.routes.signals.ppo_discovery.resolve_universe_snapshot",
            return_value=snapshot,
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.get_news_service",
            return_value=service,
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.build_ppo_news_features",
            return_value={"AAPL": object()},
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.features_to_schema",
            return_value=object(),
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.load_current_artifacts_for_bucket",
            return_value=artifacts,
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.RegimeHMMArtifact.from_dict",
            return_value=SimpleNamespace(training_cutoff_date=date(2026, 8, 7)),
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.load_prices_yfinance",
            side_effect=[{"AAPL": _frame(["2026-08-14"], 100.0)}, spy_vix],
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.live_regime_probabilities",
            return_value=(0.7, 0.1),
        ),
    )


def test_live_state_uses_repaired_vix_and_keeps_audit(monkeypatch) -> None:
    dates = [
        "2026-08-07",
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
        "2026-08-14",
    ]
    spy_vix = {
        "SPY": _frame(dates, 630.0),
        "^VIX": _frame(dates[:-1], 16.0),
    }
    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: _frame(["2026-08-14"], 15.5),
    )
    built = SimpleNamespace(to_dict=lambda: {"ok": True})
    with ExitStack() as stack:
        for manager in _common_patches(spy_vix):
            stack.enter_context(manager)
        with patch(
            "brain_api.routes.signals.ppo_discovery.build_ppo_discovery_state",
            return_value=built,
        ) as build:
            assert build_state(_request()) == {"ok": True}

    state_request = build.call_args.args[0]
    assert state_request.market_history_provenance["vix_fallback"][
        "fallback_dates"
    ] == ["2026-08-14"]


def test_live_state_returns_503_when_cboe_cannot_repair(monkeypatch) -> None:
    dates = [
        "2026-08-07",
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
        "2026-08-14",
    ]
    spy_vix = {
        "SPY": _frame(dates, 630.0),
        "^VIX": _frame(dates[:-1], 16.0),
    }
    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: _frame(["2026-08-13"], 15.5),
    )
    with ExitStack() as stack:
        for manager in _common_patches(spy_vix):
            stack.enter_context(manager)
        with pytest.raises(HTTPException) as error:
            build_state(_request())
    assert error.value.status_code == 503


def test_live_state_keeps_spy_gap_as_422(monkeypatch) -> None:
    dates = [
        "2026-08-07",
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
        "2026-08-14",
    ]
    spy_vix = {
        "SPY": _frame(dates[:-1], 630.0),
        "^VIX": _frame(dates, 16.0),
    }
    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: pytest.fail("complete VIX must not call Cboe"),
    )
    with ExitStack() as stack:
        for manager in _common_patches(spy_vix):
            stack.enter_context(manager)
        with pytest.raises(HTTPException) as error:
            build_state(_request())
    assert error.value.status_code == 422
    assert "2026-08-14" in str(error.value.detail)
