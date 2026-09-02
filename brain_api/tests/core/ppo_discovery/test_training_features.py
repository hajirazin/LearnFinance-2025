from __future__ import annotations

import numpy as np
import pandas as pd

from brain_api.core.ppo_discovery.config import HISTORY_BARS
from brain_api.core.ppo_discovery.environment import WeeklyTransition
from brain_api.core.ppo_discovery.price_features import (
    encoder_channels_from_ohlcv,
    validate_ohlcv_frame,
)
from brain_api.core.ppo_discovery.training_features import (
    fit_feature_scalers,
    pretrain_arrays,
)
from brain_api.core.ppo_discovery.weeks import prices_as_of

from .factories import make_news, make_ohlcv, make_snapshot


def test_scaler_and_pretraining_use_canonical_repaired_ohlcv() -> None:
    snapshot = make_snapshot(1)
    symbol = snapshot.sorted_symbols[0]
    index = pd.bdate_range("2025-08-25", periods=HISTORY_BARS + 2)
    frame = make_ohlcv(n=len(index))
    frame.index = index
    cutoff = index[HISTORY_BARS - 1]
    frame.loc[cutoff, "low"] = frame.loc[cutoff, "open"] + 1.0
    frame.loc[cutoff, "high"] = frame.loc[cutoff, "close"] - 1.0
    week = WeeklyTransition(
        cutoff=cutoff,
        rebalance_session=index[HISTORY_BARS],
        next_rebalance_session=index[HISTORY_BARS + 1],
        news_by_symbol={symbol: make_news(symbol)},
        p_calm=0.8,
        p_stress=0.1,
    )
    prices = {symbol: frame}
    expected = encoder_channels_from_ohlcv(
        validate_ohlcv_frame(symbol, prices_as_of(frame, cutoff.date()))
    )

    scalers = fit_feature_scalers([week], snapshot, prices)
    histories, targets = pretrain_arrays([week], snapshot, prices)

    np.testing.assert_allclose(
        scalers["encoder_channels"]["mean"], expected.mean(axis=0)
    )
    np.testing.assert_allclose(histories[0][0], expected)
    assert targets[0].shape == (1,)
