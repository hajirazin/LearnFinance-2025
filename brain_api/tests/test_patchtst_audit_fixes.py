"""Unit tests for PatchTST audit fix phases C/D/E/F1."""

import datetime as dt
from datetime import date

import numpy as np
import pandas as pd

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import (
    DEFAULT_INDIA_EXCHANGE,
    compute_week_from_cutoff,
)
from brain_api.core.patchtst.config import DEFAULT_CONFIG, PatchTSTConfig
from brain_api.core.patchtst.dataset import _week_end_anchors, build_dataset
from brain_api.core.patchtst.training import _chrono_train_val_split


def test_patchtst_default_weight_decay_is_zero():
    """E8: after close-only MSE, prod wd=1e-4 dominated Adam (wd||theta||/||g|| ~ 40).

    Overfit-batch and full wd sweep (halal 10y->2026) only train at wd=0;
    any wd>=1e-5 early-stops barely-trained. Default must stay 0.
    training.py must keep reading config.weight_decay (not a hard-coded 1e-4).
    """
    assert PatchTSTConfig().weight_decay == 0.0
    assert DEFAULT_CONFIG.weight_decay == 0.0


def test_week_end_anchors_skip_midweek_gap():
    """ISO-week anchors: Tuesday before Thursday (Wed holiday) is not an anchor."""
    # Mon Tue (Wed holiday) Thu Fri
    idx = pd.to_datetime(["2024-06-17", "2024-06-18", "2024-06-20", "2024-06-21"])
    anchors = _week_end_anchors(pd.DatetimeIndex(idx), min_week_days=3)
    assert anchors == [3]  # Friday 2024-06-21 only


def test_compute_ohlcv_log_returns_does_not_zero_inf():
    """F1: Inf must not be silently replaced with 0."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [100.0, 0.0, 100.0],
        },
        index=idx,
    )
    out = compute_ohlcv_log_returns(df, use_returns=True)
    assert bool(np.isinf(out["volume_ret"].to_numpy()).any())


def test_chrono_split_val_after_train():
    """Phase D: chronological split keeps val after train with purge."""
    n = 20
    X = np.zeros((n, 4, 5), dtype=np.float32)
    y = np.zeros((n, 5, 5), dtype=np.float32)
    anchors = np.array(
        [date(2024, 1, 5) + dt.timedelta(days=7 * i) for i in range(n)],
        dtype=object,
    )
    X_tr, X_va, _y_tr, _y_va = _chrono_train_val_split(
        X, y, anchors, validation_split=0.2, horizon_purge_calendar_days=7
    )
    split_idx = int(n * 0.8)
    min_val = anchors[split_idx]
    purge_before = min_val - dt.timedelta(days=7)
    assert len(X_va) == n - split_idx
    assert len(X_tr) == sum(1 for a in anchors[:split_idx] if a < purge_before)


def test_india_mlk_week_start_includes_monday_on_xbom():
    """Phase E: India (XBOM) target week starts Mon 2025-01-20; US XNYS starts Tue."""
    cutoff = date(2025, 1, 17)  # Friday before MLK week
    us = compute_week_from_cutoff(cutoff, exchange="XNYS")
    india = compute_week_from_cutoff(cutoff, exchange=DEFAULT_INDIA_EXCHANGE)
    assert us.target_week_start == date(2025, 1, 21)
    assert india.target_week_start == date(2025, 1, 20)


def test_build_dataset_returns_sorted_anchor_dates():
    """Dataset samples are sorted by anchor date."""
    rng = np.random.default_rng(0)
    n = 80
    idx = pd.bdate_range("2023-01-02", periods=n)
    ohlcv = {
        "open_ret": rng.normal(0, 0.01, n),
        "high_ret": rng.normal(0, 0.01, n),
        "low_ret": rng.normal(0, 0.01, n),
        "close_ret": rng.normal(0, 0.01, n),
        "volume_ret": rng.normal(0, 0.1, n),
    }
    df = pd.DataFrame(ohlcv, index=idx)
    config = PatchTSTConfig(context_length=20, min_week_days=3)
    result = build_dataset({"AAA": df}, {"AAA": df}, config)
    assert len(result.anchor_dates) == len(result.X)
    assert list(result.anchor_dates) == sorted(result.anchor_dates)
