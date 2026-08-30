"""Train-fold feature prep for ppo_discovery. Kept out of pipeline.py."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from brain_api.core.ppo_discovery.config import ENCODER_CHANNELS, HISTORY_BARS
from brain_api.core.ppo_discovery.environment import WeeklyTransition
from brain_api.core.ppo_discovery.pretraining import next_week_open_log_return
from brain_api.core.ppo_discovery.price_features import (
    apply_encoder_channel_scaler,
    encoder_channels_from_ohlcv,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError, UniverseSnapshot
from brain_api.core.ppo_discovery.weeks import open_to_open_return, prices_as_of


def with_regime(week: WeeklyTransition, regimes: dict) -> WeeklyTransition:
    calm, stress = regimes[week.cutoff.date()]
    return WeeklyTransition(
        cutoff=week.cutoff,
        rebalance_session=week.rebalance_session,
        next_rebalance_session=week.next_rebalance_session,
        news_by_symbol=week.news_by_symbol,
        p_calm=float(calm),
        p_stress=float(stress),
    )


def ohlcv_for_training(
    prices: dict[str, pd.DataFrame], symbols: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Require SPY and VIX; omit missing stock frames so they mask later."""
    missing_index = [name for name in ("SPY", "^VIX") if name not in prices]
    if missing_index:
        raise PPODiscoveryError(f"missing yfinance frames: {missing_index}")
    return {symbol: prices[symbol] for symbol in symbols if symbol in prices}


def eligible_count(ohlcv, symbols, cutoff: date) -> int:
    count = 0
    for symbol in symbols:
        frame = ohlcv.get(symbol)
        if frame is None:
            continue
        try:
            sliced = prices_as_of(frame, cutoff)
        except PPODiscoveryError:
            continue
        if len(sliced) >= HISTORY_BARS:
            count += 1
    return count


def fit_count_scaler(weeks: Sequence[WeeklyTransition]) -> dict[str, dict[str, float]]:
    values = [
        float(row.log1p_article_count)
        for week in weeks
        for row in week.news_by_symbol.values()
    ]
    if not values:
        raise PPODiscoveryError("cannot fit log1p_article_count scaler on empty news")
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    scale = float(array.std(ddof=0))
    if scale < 1e-12:
        scale = 1.0
    return {"log1p_article_count": {"mean": mean, "scale": scale}}


def fit_feature_scalers(
    weeks: Sequence[WeeklyTransition], snapshot: UniverseSnapshot, ohlcv
) -> dict[str, Any]:
    """Fit news-count and per-channel encoder scalers on the train fold only."""
    scalers: dict[str, Any] = dict(fit_count_scaler(weeks))
    channel_rows: list[np.ndarray] = []
    for week in weeks:
        cutoff = week.cutoff.date()
        for symbol in snapshot.sorted_symbols:
            frame = ohlcv.get(symbol)
            if frame is None:
                continue
            try:
                tensor = encoder_channels_from_ohlcv(prices_as_of(frame, cutoff))
            except PPODiscoveryError:
                continue
            channel_rows.append(tensor.reshape(-1, ENCODER_CHANNELS))
    if not channel_rows:
        raise PPODiscoveryError(
            "cannot fit encoder_channels scaler on an empty train fold"
        )
    stacked = np.concatenate(channel_rows, axis=0)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0, ddof=0)
    scale = np.maximum(scale, 1e-12)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise PPODiscoveryError("encoder_channels scaler is non-finite")
    scalers["encoder_channels"] = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    return scalers


def pretrain_arrays(weeks, snapshot, ohlcv, feature_scalers=None):
    histories = []
    targets = []
    symbols = list(snapshot.sorted_symbols)
    for week in weeks:
        cutoff = week.cutoff.date()
        history_rows = []
        target_rows = []
        for symbol in symbols:
            frame = ohlcv.get(symbol)
            if frame is None:
                continue
            try:
                sliced = prices_as_of(frame, cutoff)
                history = apply_encoder_channel_scaler(
                    encoder_channels_from_ohlcv(sliced), feature_scalers
                )
                start_open, simple = open_to_open_return(
                    frame,
                    week.rebalance_session,
                    week.next_rebalance_session,
                    symbol=symbol,
                )
                target = next_week_open_log_return(
                    start_open, start_open * (1.0 + simple)
                )
            except PPODiscoveryError:
                continue
            history_rows.append(history)
            target_rows.append(target)
        if not history_rows:
            continue
        histories.append(np.stack(history_rows, axis=0))
        targets.append(np.asarray(target_rows, dtype=np.float64))
    return histories, targets


__all__ = [
    "eligible_count",
    "fit_feature_scalers",
    "ohlcv_for_training",
    "pretrain_arrays",
    "with_regime",
]
