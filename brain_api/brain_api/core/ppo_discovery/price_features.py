"""Point-in-time OHLCV tensors and explicit price ranks for ppo_discovery."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from brain_api.core.portfolio_rl.state import cross_sectional_rank
from brain_api.core.ppo_discovery.config import (
    ENCODER_CHANNELS,
    ENCODER_SESSIONS,
    HISTORY_BARS,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.prices import repair_ohlc_envelope
from brain_api.core.sac.momentum_signals import (
    compute_momentum_1w,
    compute_momentum_4w,
    compute_momentum_12_1,
    compute_realized_vol_20d,
)

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate numeric OHLCV and envelope provider high/low around open/close."""
    if frame is None or frame.empty:
        raise PPODiscoveryError(f"{symbol} has no OHLCV evidence")
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise PPODiscoveryError(f"{symbol} OHLCV missing columns {missing}")
    ordered = frame.loc[:, list(REQUIRED_COLUMNS)].astype(np.float64)
    if len(ordered) < HISTORY_BARS:
        raise PPODiscoveryError(
            f"{symbol} has {len(ordered)} sessions; need {HISTORY_BARS}"
        )
    tail = ordered.iloc[-HISTORY_BARS:].copy()
    for column in ("open", "high", "low", "close"):
        values = tail[column].to_numpy()
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise PPODiscoveryError(f"{symbol} {column} must be finite and positive")
    volume = tail["volume"].to_numpy()
    if not np.all(np.isfinite(volume)) or np.any(volume < 0):
        raise PPODiscoveryError(f"{symbol} volume must be finite and nonnegative")
    return repair_ohlc_envelope(tail)


def encoder_channels_from_ohlcv(frame: pd.DataFrame) -> np.ndarray:
    """Build the 250 x 4 log-return tensor from the last 251 of 253 bars."""
    if len(frame) < HISTORY_BARS:
        raise PPODiscoveryError("OHLCV too short for encoder tensor")
    window = frame.iloc[-(ENCODER_SESSIONS + 1) :]
    if len(window) != ENCODER_SESSIONS + 1:
        raise PPODiscoveryError("encoder window must be 251 sessions")
    open_ = window["open"].to_numpy(dtype=np.float64)
    high = window["high"].to_numpy(dtype=np.float64)
    low = window["low"].to_numpy(dtype=np.float64)
    close = window["close"].to_numpy(dtype=np.float64)
    volume = window["volume"].to_numpy(dtype=np.float64)
    close_ret = np.log(close[1:] / close[:-1])
    hl_range = np.log(high[1:] / low[1:])
    co_ret = np.log(close[1:] / open_[1:])
    vol_chg = np.log1p(volume[1:]) - np.log1p(volume[:-1])
    tensor = np.stack([close_ret, hl_range, co_ret, vol_chg], axis=1)
    if tensor.shape != (ENCODER_SESSIONS, ENCODER_CHANNELS):
        raise PPODiscoveryError(f"encoder tensor shape {tensor.shape} is invalid")
    if not np.all(np.isfinite(tensor)):
        raise PPODiscoveryError("encoder tensor contains non-finite values")
    return tensor


def apply_encoder_channel_scaler(
    tensor: np.ndarray, scalers: Mapping[str, object] | None
) -> np.ndarray:
    """Standardize the 4 encoder channels with the train-fold scaler."""
    if scalers is None or "encoder_channels" not in scalers:
        return tensor
    payload = scalers["encoder_channels"]
    if not isinstance(payload, Mapping):
        raise PPODiscoveryError("encoder_channels scaler must be a mapping")
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    if mean.shape != (ENCODER_CHANNELS,) or scale.shape != (ENCODER_CHANNELS,):
        raise PPODiscoveryError("encoder_channels scaler width mismatch")
    if (
        np.any(scale <= 0)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
    ):
        raise PPODiscoveryError("invalid encoder_channels scaler")
    scaled = (tensor - mean) / scale
    if not np.all(np.isfinite(scaled)):
        raise PPODiscoveryError("scaled encoder tensor contains non-finite values")
    return scaled


def explicit_price_signals(closes: Sequence[float]) -> dict[str, float]:
    """Momentum and realized-vol using the existing SAC helpers."""
    as_of_index = len(closes) - 1
    return {
        "momentum_1w": compute_momentum_1w(closes, as_of_index=as_of_index),
        "momentum_4w": compute_momentum_4w(closes, as_of_index=as_of_index),
        "momentum_12_1": compute_momentum_12_1(closes, as_of_index=as_of_index),
        "realized_vol_20d": compute_realized_vol_20d(closes, as_of_index=as_of_index),
    }


def rank_eligible(
    values: Mapping[str, float], eligible: Sequence[str]
) -> dict[str, float]:
    """Average-tie rank over eligible names only."""
    ordered = list(eligible)
    vector = np.asarray([values[symbol] for symbol in ordered], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise PPODiscoveryError("cannot rank non-finite price features")
    ranks = cross_sectional_rank(vector)
    return {symbol: float(ranks[index]) for index, symbol in enumerate(ordered)}


def spy_return_20d(spy_closes: Sequence[float]) -> float:
    """Simple 20-session SPY return ending at the latest completed session."""
    if len(spy_closes) < 21:
        raise PPODiscoveryError("SPY history too short for 20-session return")
    end = float(spy_closes[-1])
    start = float(spy_closes[-21])
    if not math.isfinite(end) or not math.isfinite(start) or start <= 0 or end <= 0:
        raise PPODiscoveryError("SPY closes must be finite and positive")
    return end / start - 1.0


__all__ = [
    "apply_encoder_channel_scaler",
    "encoder_channels_from_ohlcv",
    "explicit_price_signals",
    "rank_eligible",
    "spy_return_20d",
    "validate_ohlcv_frame",
]
