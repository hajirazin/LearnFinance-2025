"""Conditional Cboe backup for missing Yahoo VIX sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests

CBOE_VIX_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
CBOE_REQUEST_HEADERS = {"User-Agent": "LearnFinance-2025/1.0"}
_PRICE_COLUMNS = ("open", "high", "low", "close")


class VixFallbackError(RuntimeError):
    """Raised when required Yahoo VIX gaps cannot be repaired from Cboe."""


@dataclass(frozen=True)
class VixFallbackAudit:
    """Auditable provider selection for one VIX evidence request."""

    primary_provider: str = "yfinance"
    fallback_provider: str | None = None
    fallback_dates: tuple[str, ...] = ()
    source_url: str | None = None
    retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_provider": self.primary_provider,
            "fallback_provider": self.fallback_provider,
            "fallback_dates": list(self.fallback_dates),
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
        }


@dataclass(frozen=True)
class VixFallbackResult:
    """Price frames plus provenance after conditional VIX repair."""

    prices: dict[str, pd.DataFrame]
    audit: VixFallbackAudit


def _normalized_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise VixFallbackError(f"{source} VIX history is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise VixFallbackError(f"{source} VIX history must use a DatetimeIndex")
    normalized = frame.copy()
    index = normalized.index
    if index.tz is not None:
        index = index.tz_localize(None)
    normalized.index = index.normalize()
    if normalized.index.has_duplicates:
        raise VixFallbackError(f"{source} VIX history has duplicate dates")
    if not normalized.index.is_monotonic_increasing:
        raise VixFallbackError(f"{source} VIX history must be ordered")
    return normalized


def load_cboe_vix_history() -> pd.DataFrame:
    """Download Cboe's official daily VIX history and parse its date index."""
    try:
        response = requests.get(
            CBOE_VIX_HISTORY_URL,
            headers=CBOE_REQUEST_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise VixFallbackError(f"Cboe VIX download failed: {exc}") from exc
    try:
        raw = pd.read_csv(StringIO(response.text))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise VixFallbackError(f"Cboe VIX CSV cannot be parsed: {exc}") from exc

    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required = {"date", *_PRICE_COLUMNS}
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise VixFallbackError(
            f"Cboe VIX history is missing columns: {missing_columns}"
        )
    dates = pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="coerce")
    history = raw.loc[:, list(_PRICE_COLUMNS)].copy()
    history.index = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    return history


def _valid_close_dates(frame: pd.DataFrame | None) -> set[date]:
    if frame is None or frame.empty or "close" not in frame.columns:
        return set()
    normalized = _normalized_frame(frame, source="Yahoo")
    closes = pd.to_numeric(normalized["close"], errors="coerce").to_numpy(dtype=float)
    return {
        timestamp.date()
        for timestamp, value in zip(normalized.index, closes, strict=True)
        if np.isfinite(value) and value > 0
    }


def apply_cboe_vix_fallback(
    prices: Mapping[str, pd.DataFrame],
    *,
    required_dates: Sequence[date],
    cboe_history: pd.DataFrame | None = None,
) -> VixFallbackResult:
    """Repair only required missing VIX sessions, preserving valid Yahoo rows."""
    required = list(dict.fromkeys(required_dates))
    if required != sorted(required):
        raise VixFallbackError("required VIX dates must be unique and ordered")
    copied = {symbol: frame.copy() for symbol, frame in prices.items()}
    valid_dates = _valid_close_dates(copied.get("^VIX"))
    missing_dates = [session for session in required if session not in valid_dates]
    if not missing_dates:
        return VixFallbackResult(copied, VixFallbackAudit())

    cboe = cboe_history.copy() if cboe_history is not None else load_cboe_vix_history()
    if not isinstance(cboe.index, pd.DatetimeIndex):
        raise VixFallbackError("Cboe VIX history must use a DatetimeIndex")
    cboe.index = cboe.index.tz_localize(None).normalize()
    missing_index = pd.DatetimeIndex(missing_dates)
    rows: list[pd.Series] = []
    for session, timestamp in zip(missing_dates, missing_index, strict=True):
        matches = cboe.loc[cboe.index == timestamp]
        if matches.empty:
            raise VixFallbackError(
                f"Cboe VIX history lacks required dates: {[session.isoformat()]}"
            )
        if len(matches) != 1:
            raise VixFallbackError(
                f"Cboe VIX history duplicates required date: {session.isoformat()}"
            )
        rows.append(matches.iloc[0])
    repairs = pd.DataFrame(rows, index=missing_index)
    missing_columns = sorted(set(_PRICE_COLUMNS) - set(repairs.columns))
    if missing_columns:
        raise VixFallbackError(
            f"Cboe VIX history is missing columns: {missing_columns}"
        )
    try:
        repairs = repairs.loc[:, list(_PRICE_COLUMNS)].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise VixFallbackError(f"Cboe VIX repair rows cannot be parsed: {exc}") from exc
    repair_values = repairs.to_numpy(dtype=float)
    if not np.all(np.isfinite(repair_values)) or np.any(repair_values <= 0):
        raise VixFallbackError("Cboe VIX repair OHLC must be finite and positive")
    repairs["volume"] = 0.0
    yahoo = copied.get("^VIX")
    if yahoo is None or yahoo.empty:
        repaired = repairs
    else:
        normalized_yahoo = _normalized_frame(yahoo, source="Yahoo")
        normalized_yahoo = normalized_yahoo.drop(index=missing_index, errors="ignore")
        repaired = pd.concat([normalized_yahoo, repairs]).sort_index()
    if repaired.index.has_duplicates:
        raise VixFallbackError("repaired VIX history has duplicate dates")
    copied["^VIX"] = repaired
    retrieved_at = datetime.now(UTC).isoformat()
    audit = VixFallbackAudit(
        fallback_provider="cboe",
        fallback_dates=tuple(session.isoformat() for session in missing_dates),
        source_url=CBOE_VIX_HISTORY_URL,
        retrieved_at=retrieved_at,
    )
    return VixFallbackResult(copied, audit)


__all__ = [
    "CBOE_REQUEST_HEADERS",
    "CBOE_VIX_HISTORY_URL",
    "VixFallbackAudit",
    "VixFallbackError",
    "VixFallbackResult",
    "apply_cboe_vix_fallback",
    "load_cboe_vix_history",
]
