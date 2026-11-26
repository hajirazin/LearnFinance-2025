"""Common causal input windows and five-session labels for PatchTST."""

from dataclasses import dataclass
from datetime import date
from tempfile import TemporaryFile

import numpy as np
import pandas as pd

from brain_api.core.patchtst.calendar import (
    FORECAST_SESSIONS,
    PRICE_WARMUP_SESSIONS,
    PatchTSTCalendar,
    training_calendar,
)
from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.patchtst.india_prices import (
    NSE_RECOVERED_BAR,
    restore_nse_cash_bar_gaps,
)
from brain_api.core.patchtst.price_history import normalize_price_dates
from brain_api.news.patchtst import load_session_sentiment
from brain_api.news.store import NewsStore


@dataclass
class WeeklyPatchTSTData:
    X: np.ndarray
    y: np.ndarray
    anchor_dates: np.ndarray
    target_dates: np.ndarray
    symbols: np.ndarray


class InsufficientWeeklyHistory(ValueError):
    """An annual cutoff cannot establish both chronological folds."""


class WeeklyPatchTSTInputs:
    """Compute causal features once for the loaded symbol and session grid."""

    def __init__(
        self,
        config: PatchTSTConfig,
        calendar: PatchTSTCalendar,
        prices: dict[str, pd.DataFrame],
        store: NewsStore | None = None,
    ):
        self.config = config
        self.calendar = calendar
        index = calendar.sessions
        if "news_sentiment" in config.feature_names:
            if store is None:
                store = NewsStore()
            sentiment = load_session_sentiment(store, list(prices), calendar.closes)
        else:
            sentiment = None
        self.symbols = sorted(prices)
        self.features = {}
        self.close = {}
        self.valid_returns = {}
        self.recovered_bars = {}
        for symbol in self.symbols:
            try:
                bars = normalize_price_dates(prices[symbol]).reindex(index)
            except ValueError:
                continue
            close = bars["close"]
            volume = bars["volume"]
            usable = (
                np.isfinite(close) & np.isfinite(volume) & (close > 0) & (volume > 0)
            )
            close, volume = close.where(usable), volume.where(usable)
            logs = np.log(close).diff()
            values = {
                "momentum_20d": np.expm1(logs.rolling(20).sum()),
                "momentum_12_1": np.expm1(logs.rolling(231).sum().shift(21)),
                "volatility_20d": logs.rolling(20).std(ddof=1) * np.sqrt(252),
                "relative_volume_20d": volume / volume.shift(1).rolling(20).median(),
                "close": close,
                "volume": volume,
            }
            if sentiment is not None:
                values["news_sentiment"] = sentiment[symbol]
            self.features[symbol] = np.column_stack(
                [values[name] for name in config.feature_names]
            ).astype(np.float32)
            self.close[symbol] = close.to_numpy()
            self.valid_returns[symbol] = np.isfinite(logs.to_numpy())
            if NSE_RECOVERED_BAR in bars:
                self.recovered_bars[symbol] = (
                    bars[NSE_RECOVERED_BAR].eq(True).to_numpy()
                )

    def window(self, symbol: str, anchor: date) -> tuple[np.ndarray | None, int]:
        left, right = self.calendar.window(anchor, self.config.context_calendar_days)
        count = int(right - left)
        if count > self.config.context_length:
            raise ValueError(
                f"{self.config.market} context exceeds {self.config.context_length} sessions"
            )
        if (
            left < PRICE_WARMUP_SESSIONS
            or symbol not in self.features
            or count < 1
            or self.calendar.sessions[right - 1].date() != anchor
            or self._recovered_at(symbol, right - 1)
            or not self.valid_returns[symbol][left:right].all()
        ):
            return None, count
        raw = self.features[symbol][left:right]
        observed = np.isfinite(raw)
        if not observed[-1].all() or not (observed.mean(axis=0) >= 0.8).all():
            return None, count
        lengths = observed.sum(axis=0)
        averages = np.where(observed, raw, 0).sum(axis=0, dtype=np.float64) / lengths
        centered = np.where(observed, raw - averages, 0)
        variance = np.sum(centered**2, axis=0) / lengths
        normalized = np.zeros(
            (self.config.context_length, raw.shape[1]), dtype=np.float32
        )
        normalized[-count:] = centered / np.sqrt(variance + 1e-5)
        return normalized, count

    def _recovered_at(self, symbol: str, position: int) -> bool:
        recovered = self.recovered_bars.get(symbol)
        return recovered is not None and bool(recovered[position])

    def labeled(self, start: date, end: date) -> WeeklyPatchTSTData:
        sessions = self.calendar.sessions
        bounds = sessions.to_series().groupby(sessions.to_period("W-SUN")).last()
        anchors = bounds[
            (bounds >= pd.Timestamp(start)) & (bounds <= pd.Timestamp(end))
        ]
        if len(anchors) < 66:
            raise InsufficientWeeklyHistory(
                "PatchTST requires at least 52 fitting and 13 validation weeks after purging"
            )
        limit = sessions.searchsorted(pd.Timestamp(end), "right")
        capacity = len(anchors) * len(self.symbols)
        if not capacity:
            raise InsufficientWeeklyHistory("No prices available for weekly PatchTST")
        labels, dates, ends, symbols = [], [], [], []
        expected_end = None
        shape = (capacity, self.config.context_length, self.config.num_input_channels)
        # Keep historical panels off heap. The trainer copies only the current batch.
        with TemporaryFile() as backing:
            matrix = np.memmap(backing, mode="w+", shape=shape, dtype=np.float32)
            for stamp in anchors:
                anchor = stamp.date()
                idx = sessions.get_loc(stamp)
                ahead = idx + FORECAST_SESSIONS
                if ahead >= limit:
                    continue
                expected_end = sessions[ahead].date()
                for symbol in self.close:
                    # A repaired close was confirmed by the next session.
                    if self._recovered_at(symbol, ahead):
                        continue
                    if not self.valid_returns[symbol][idx + 1 : ahead + 1].all():
                        continue
                    window, _ = self.window(symbol, anchor)
                    if window is None:
                        continue
                    prices = self.close[symbol]
                    matrix[len(labels)] = window
                    labels.append(prices[ahead] / prices[idx] - 1.0)
                    dates.append(anchor)
                    ends.append(sessions[ahead].date())
                    symbols.append(symbol)
        if not labels:
            raise InsufficientWeeklyHistory("No eligible weekly PatchTST examples")
        latest_count = sum(day == expected_end for day in ends)
        required = min(10, len(self.symbols))
        if latest_count < required:
            raise ValueError(
                f"Stale PatchTST inputs at cutoff {end}: expected labels through "
                f"{expected_end} for at least {required} stocks, found {latest_count}; "
                f"latest usable label ends {max(ends)}. Repair price/news evidence."
            )
        return WeeklyPatchTSTData(
            matrix[: len(labels)],
            np.asarray(labels, dtype=np.float32),
            np.asarray(dates),
            np.asarray(ends),
            np.asarray(symbols),
        )


def load_weekly_training_inputs(
    symbols,
    start: date,
    end: date,
    config: PatchTSTConfig,
    price_loader,
    *,
    store: NewsStore | None = None,
) -> WeeklyPatchTSTInputs:
    calendar = training_calendar(config.market, start, end)
    prices = price_loader(symbols, calendar.sessions[0].date(), end)
    if config.market == "india":
        prices = restore_nse_cash_bar_gaps(prices, calendar.sessions)
    return WeeklyPatchTSTInputs(config, calendar, prices, store)
