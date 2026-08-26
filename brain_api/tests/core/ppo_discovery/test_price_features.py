"""OHLCV channel and momentum tests for ppo_discovery."""

from __future__ import annotations

import numpy as np
import pytest

from brain_api.core.ppo_discovery.price_features import (
    apply_encoder_channel_scaler,
    encoder_channels_from_ohlcv,
    explicit_price_signals,
    spy_return_20d,
    validate_ohlcv_frame,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.sac.momentum_signals import (
    compute_momentum_1w,
    compute_momentum_4w,
    compute_momentum_12_1,
    compute_realized_vol_20d,
)
from tests.core.ppo_discovery.factories import make_ohlcv


def test_encoder_channels_order_and_log1p_volume() -> None:
    frame = make_ohlcv()
    validated = validate_ohlcv_frame("AAPL", frame)
    tensor = encoder_channels_from_ohlcv(validated)
    assert tensor.shape == (250, 4)
    window = validated.iloc[-251:]
    close = window["close"].to_numpy()
    high = window["high"].to_numpy()
    low = window["low"].to_numpy()
    open_ = window["open"].to_numpy()
    volume = window["volume"].to_numpy()
    np.testing.assert_allclose(tensor[:, 0], np.log(close[1:] / close[:-1]))
    np.testing.assert_allclose(tensor[:, 1], np.log(high[1:] / low[1:]))
    np.testing.assert_allclose(tensor[:, 2], np.log(close[1:] / open_[1:]))
    np.testing.assert_allclose(
        tensor[:, 3], np.log1p(volume[1:]) - np.log1p(volume[:-1])
    )


def test_zero_volume_is_safe_for_log1p() -> None:
    frame = make_ohlcv()
    frame.loc[frame.index[-1], "volume"] = 0.0
    tensor = encoder_channels_from_ohlcv(validate_ohlcv_frame("AAPL", frame))
    assert np.isfinite(tensor[-1, 3])


def test_momentum_helpers_match_sac() -> None:
    frame = make_ohlcv()
    closes = frame["close"].to_numpy()
    signals = explicit_price_signals(closes)
    as_of = len(closes) - 1
    assert signals["momentum_1w"] == compute_momentum_1w(closes, as_of_index=as_of)
    assert signals["momentum_4w"] == compute_momentum_4w(closes, as_of_index=as_of)
    assert signals["momentum_12_1"] == compute_momentum_12_1(closes, as_of_index=as_of)
    assert signals["realized_vol_20d"] == compute_realized_vol_20d(
        closes, as_of_index=as_of
    )


def test_short_history_rejected() -> None:
    with pytest.raises(PPODiscoveryError, match="sessions"):
        validate_ohlcv_frame("AAPL", make_ohlcv(n=10))


def test_spy_return_20d() -> None:
    closes = [float(i) for i in range(1, 30)]
    assert spy_return_20d(closes) == pytest.approx(closes[-1] / closes[-21] - 1.0)


def test_encoder_channel_scaler_standardizes() -> None:
    tensor = np.ones((250, 4), dtype=np.float64)
    tensor[:, 1] = 3.0
    scaled = apply_encoder_channel_scaler(
        tensor,
        {
            "encoder_channels": {
                "mean": [1.0, 3.0, 1.0, 1.0],
                "scale": [2.0, 2.0, 2.0, 2.0],
            }
        },
    )
    np.testing.assert_allclose(scaled[:, 0], 0.0)
    np.testing.assert_allclose(scaled[:, 1], 0.0)
