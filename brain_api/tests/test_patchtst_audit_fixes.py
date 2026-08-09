"""Unit tests for PatchTST audit fix phases C/D/E/F1."""

import datetime as dt
import warnings
from datetime import date

import numpy as np
import pandas as pd

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import (
    DEFAULT_INDIA_EXCHANGE,
    compute_week_from_cutoff,
)
from brain_api.core.patchtst.config import DEFAULT_CONFIG, PatchTSTConfig
from brain_api.core.patchtst.data_loaders import align_multivariate_data
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


def test_compute_ohlcv_log_returns_zero_volume_is_nan_not_inf():
    """Zero volume → NaN on volume_ret (no Inf, no silent zero-fill)."""
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compute_ohlcv_log_returns(df, use_returns=True)
    assert not any("divide" in str(w.message).lower() for w in caught)
    assert not any("invalid" in str(w.message).lower() for w in caught)
    vol = out["volume_ret"].to_numpy()
    assert not bool(np.isinf(vol).any())
    # Day with volume=0 and the day after (prev=0) are both non-positive pairs → NaN
    assert bool(np.isnan(vol[0]))  # 0 / 100
    assert bool(np.isnan(vol[1]))  # 100 / 0
    # No silent zero-fill of the bad bars
    assert not bool((vol == 0.0).any())


def test_compute_ohlcv_log_returns_positive_volume_matches_log_ratio():
    """Positive OHLCV bars keep standard finite log returns."""
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    df = pd.DataFrame(
        {
            "open": [10.0, 11.0, 10.5, 12.0],
            "high": [10.5, 11.5, 11.0, 12.5],
            "low": [9.5, 10.5, 10.0, 11.5],
            "close": [10.0, 11.0, 10.5, 12.0],
            "volume": [1000.0, 1100.0, 900.0, 1200.0],
        },
        index=idx,
    )
    out = compute_ohlcv_log_returns(df, use_returns=True)
    assert np.isfinite(out.to_numpy()).all()
    expected_close = np.log(
        df["close"].iloc[1:].to_numpy() / df["close"].iloc[:-1].to_numpy()
    )
    expected_vol = np.log(
        df["volume"].iloc[1:].to_numpy() / df["volume"].iloc[:-1].to_numpy()
    )
    np.testing.assert_allclose(out["close_ret"].to_numpy(), expected_close)
    np.testing.assert_allclose(out["volume_ret"].to_numpy(), expected_vol)


def test_compute_ohlcv_log_returns_nonpositive_ohlc_is_nan():
    """OHLC ≤ 0 yields NaN on that channel, not Inf."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "open": [10.0, 0.0, 10.0],
            "high": [10.0, -1.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [100.0, 100.0, 100.0],
        },
        index=idx,
    )
    out = compute_ohlcv_log_returns(df, use_returns=True)
    assert not bool(np.isinf(out.to_numpy()).any())
    assert bool(np.isnan(out["open_ret"].iloc[0]))
    assert bool(np.isnan(out["high_ret"].iloc[0]))
    assert bool(np.isnan(out["open_ret"].iloc[1]))  # recovery day still has prev≤0
    assert bool(np.isnan(out["high_ret"].iloc[1]))
    assert np.isfinite(out["close_ret"].to_numpy()).all()
    assert np.isfinite(out["volume_ret"].to_numpy()).all()


def _synthetic_ohlcv(n: int, idx: pd.DatetimeIndex, volume: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 110.0, n),
            "high": np.linspace(101.0, 111.0, n),
            "low": np.linspace(99.0, 109.0, n),
            "close": np.linspace(100.0, 110.0, n),
            "volume": volume,
        },
        index=idx,
    )


def test_align_multivariate_data_zero_volume_no_inf_warning(capsys):
    """FERG-like zero-volume days: Inf warning must not fire; NaN+skip is OK."""
    n = 80
    idx = pd.bdate_range("2023-01-02", periods=n)
    # One late zero-volume cluster (thin ADR pattern) so early week windows stay valid
    volume_dirty = np.full(n, 1_000_000.0)
    volume_dirty[[60, 61]] = 0.0
    volume_clean = np.full(n, 1_000_000.0)
    prices = {
        "FERG": _synthetic_ohlcv(n, idx, volume_dirty),
        "BDRFY": _synthetic_ohlcv(n, idx, volume_dirty.copy()),
        "CLEAN": _synthetic_ohlcv(n, idx, volume_clean),
    }
    config = PatchTSTConfig(context_length=20, min_week_days=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligned = align_multivariate_data(prices, config)
    captured = capsys.readouterr().out
    assert "Inf values" not in captured
    assert not any("divide" in str(w.message).lower() for w in caught)
    assert set(aligned) == {"FERG", "BDRFY", "CLEAN"}
    for sym in ("FERG", "BDRFY"):
        feat = aligned[sym]
        assert not bool(np.isinf(feat.to_numpy()).any()), sym
        assert bool(feat["volume_ret"].isna().any()), sym
    # Dataset still builds finite samples and skips windows that touch NaN volume days
    dirty_only = {k: aligned[k] for k in ("FERG", "BDRFY")}
    clean_only = {"CLEAN": aligned["CLEAN"]}
    dirty_result = build_dataset(dirty_only, dirty_only, config)
    clean_result = build_dataset(clean_only, clean_only, config)
    assert len(dirty_result.X) > 0
    assert len(dirty_result.X) < 2 * len(clean_result.X)
    assert not bool(np.isinf(dirty_result.X).any())
    assert not bool(np.isnan(dirty_result.X).any())


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
