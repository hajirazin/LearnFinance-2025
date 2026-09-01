"""Conditional Cboe repair for missing Yahoo VIX sessions."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests

from brain_api.core import vix_fallback as fallback


def _frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [0.0] * len(dates),
        },
        index=pd.DatetimeIndex(dates),
    )


def test_complete_yahoo_vix_does_not_call_cboe(monkeypatch) -> None:
    yahoo = _frame(["2026-08-27", "2026-08-28"], [14.51, 14.43])
    monkeypatch.setattr(
        fallback,
        "load_cboe_vix_history",
        lambda: pytest.fail("Cboe must not be called for complete Yahoo evidence"),
    )

    result = fallback.apply_cboe_vix_fallback(
        {"^VIX": yahoo},
        required_dates=[date(2026, 8, 27), date(2026, 8, 28)],
    )

    assert result.audit.fallback_provider is None
    assert result.audit.fallback_dates == ()
    pd.testing.assert_frame_equal(result.prices["^VIX"], yahoo)


@pytest.mark.parametrize("missing_as_nan", [False, True])
def test_missing_yahoo_session_is_repaired_without_overwrite(
    monkeypatch, missing_as_nan: bool
) -> None:
    dates = ["2026-08-27"]
    closes = [14.51]
    if missing_as_nan:
        dates.append("2026-08-28")
        closes.append(float("nan"))
    yahoo = _frame(dates, closes)
    cboe = pd.DataFrame(
        {
            "open": [99.0, 14.57, 15.24],
            "high": [99.0, 14.84, 15.48],
            "low": [99.0, 14.13, 14.86],
            "close": [99.0, 14.43, 14.92],
            "volume": [0.0, 0.0, 0.0],
        },
        index=pd.DatetimeIndex(["2026-08-27", "2026-08-28", "2026-08-31"]),
    )
    monkeypatch.setattr(fallback, "load_cboe_vix_history", lambda: cboe)

    result = fallback.apply_cboe_vix_fallback(
        {"^VIX": yahoo},
        required_dates=[date(2026, 8, 27), date(2026, 8, 28)],
    )

    repaired = result.prices["^VIX"]
    assert repaired.loc["2026-08-27", "close"] == pytest.approx(14.51)
    assert repaired.loc["2026-08-28", "close"] == pytest.approx(14.43)
    assert pd.Timestamp("2026-08-31") not in repaired.index
    assert result.audit.fallback_provider == "cboe"
    assert result.audit.fallback_dates == ("2026-08-28",)
    assert result.audit.source_url == fallback.CBOE_VIX_HISTORY_URL
    assert result.audit.retrieved_at is not None


def test_unresolved_cboe_session_fails_closed(monkeypatch) -> None:
    yahoo = _frame(["2026-08-27"], [14.51])
    monkeypatch.setattr(
        fallback,
        "load_cboe_vix_history",
        lambda: _frame(["2026-08-27"], [14.51]),
    )

    with pytest.raises(fallback.VixFallbackError, match="2026-08-28"):
        fallback.apply_cboe_vix_fallback(
            {"^VIX": yahoo}, required_dates=[date(2026, 8, 28)]
        )


def test_cboe_loader_rejects_malformed_csv(monkeypatch) -> None:
    class Response:
        text = "DATE,CLOSE\n08/28/2026,14.43\n"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(fallback.requests, "get", lambda *_args, **_kwargs: Response())

    with pytest.raises(fallback.VixFallbackError, match="columns"):
        fallback.load_cboe_vix_history()


def test_cboe_loader_wraps_network_failure(monkeypatch) -> None:
    def fail_request(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(fallback.requests, "get", fail_request)

    with pytest.raises(fallback.VixFallbackError, match=r"download failed.*offline"):
        fallback.load_cboe_vix_history()


def test_cboe_loader_uses_explicit_headers_and_date_format(monkeypatch) -> None:
    captured = {}

    class Response:
        text = "DATE,OPEN,HIGH,LOW,CLOSE\n08/28/2026,14.57,14.84,14.13,14.43\n"

        def raise_for_status(self) -> None:
            return None

    def request(*_args, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(fallback.requests, "get", request)

    loaded = fallback.load_cboe_vix_history()

    assert loaded.index.tolist() == [pd.Timestamp("2026-08-28")]
    assert captured["headers"] == fallback.CBOE_REQUEST_HEADERS
    assert captured["timeout"] == 30


def test_unused_duplicate_and_bad_rows_do_not_block_repair() -> None:
    yahoo = _frame(["2026-08-27"], [14.51])
    cboe = pd.DataFrame(
        {
            "open": [0.0, 0.0, 14.57],
            "high": [float("nan"), float("nan"), 14.84],
            "low": [0.0, 0.0, 14.13],
            "close": [0.0, 0.0, 14.43],
        },
        index=pd.DatetimeIndex(["1998-01-02", "1998-01-02", "2026-08-28"]),
    )

    result = fallback.apply_cboe_vix_fallback(
        {"^VIX": yahoo},
        required_dates=[date(2026, 8, 27), date(2026, 8, 28)],
        cboe_history=cboe,
    )

    assert result.prices["^VIX"].loc["2026-08-28", "close"] == pytest.approx(14.43)


def test_requested_duplicate_date_is_rejected() -> None:
    yahoo = _frame(["2026-08-27"], [14.51])
    cboe = _frame(["2026-08-28", "2026-08-28"], [14.43, 14.43])

    with pytest.raises(fallback.VixFallbackError, match="duplicates required date"):
        fallback.apply_cboe_vix_fallback(
            {"^VIX": yahoo},
            required_dates=[date(2026, 8, 28)],
            cboe_history=cboe,
        )


@pytest.mark.parametrize("bad_value", [float("nan"), 0.0])
def test_requested_nonpositive_or_nonfinite_ohlc_is_rejected(
    bad_value: float,
) -> None:
    yahoo = _frame(["2026-08-27"], [14.51])
    cboe = _frame(["2026-08-28"], [14.43])
    cboe.loc["2026-08-28", "high"] = bad_value

    with pytest.raises(fallback.VixFallbackError, match="finite and positive"):
        fallback.apply_cboe_vix_fallback(
            {"^VIX": yahoo},
            required_dates=[date(2026, 8, 28)],
            cboe_history=cboe,
        )


def test_loader_preserves_unused_duplicate_dates(monkeypatch) -> None:
    class Response:
        text = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "08/28/2026,14.57,14.84,14.13,14.43\n"
            "08/28/2026,14.57,14.84,14.13,14.43\n"
        )

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(fallback.requests, "get", lambda *_args, **_kwargs: Response())

    loaded = fallback.load_cboe_vix_history()
    assert loaded.index.duplicated().sum() == 1
