"""Offline checks for causal price-input clipping."""

import numpy as np
import pandas as pd
from rolling_mad import clip_returns, price_features, reconstruct_close


def test_spike_is_capped_using_previous_observations_only():
    values = pd.DataFrame({"A": [-0.01, 0.0, 0.01, -0.01, 0.0, 0.01, 1.0, 0.02]})
    clipped, audit = clip_returns(values, window=6, minimum=6, multiplier=3)
    assert np.isclose(clipped.loc[6, "A"], 3 * 1.4826 * 0.01)
    assert len(audit) == 1
    modified = values.copy()
    modified.loc[7, "A"] = -100
    second, _ = clip_returns(modified, window=6, minimum=6, multiplier=3)
    np.testing.assert_array_equal(clipped.iloc[:7], second.iloc[:7])


def test_zero_mad_and_insufficient_history_are_explicitly_unchanged():
    values = pd.DataFrame({"A": [0.0] * 6 + [1.0], "B": [np.nan] * 6 + [1.0]})
    clipped, audit = clip_returns(values, window=6, minimum=6, multiplier=3)
    pd.testing.assert_frame_equal(clipped, values)
    assert audit.empty


def test_reconstruction_preserves_missing_prices_and_does_not_fill_returns():
    close = pd.DataFrame({"A": [100.0, 110.0, np.nan, 90.0, 99.0]})
    raw = np.log(close).diff()
    pd.testing.assert_frame_equal(reconstruct_close(close, raw, raw), close)
    clipped = raw.copy()
    clipped.loc[1, "A"] = 0
    result = reconstruct_close(close, raw, clipped)
    assert np.isclose(result.loc[1, "A"], 100)
    assert pd.isna(result.loc[2, "A"])
    assert np.isclose(result.loc[4, "A"], 90)
    pd.testing.assert_frame_equal(
        close, pd.DataFrame({"A": [100.0, 110.0, np.nan, 90.0, 99.0]})
    )


def test_price_features_retain_other_channels_and_observed_labels():
    rng = np.random.default_rng(12)
    logs = pd.DataFrame(rng.normal(0, 0.01, (280, 2)), columns=["A", "B"])
    close = 100 * np.exp(logs.cumsum())
    features = rng.normal(size=(280, 2, 7)).astype(np.float32)
    original = features.copy()
    changed = logs.clip(-0.005, 0.005)
    result = price_features(features, close, logs, changed)
    np.testing.assert_array_equal(result[..., [3, 4, 6]], original[..., [3, 4, 6]])
    np.testing.assert_array_equal(features, original)
    assert not np.allclose(result[..., 5], original[..., 5])
    assert np.isclose(result[-1, 0, 0], np.expm1(changed.iloc[-20:, 0].sum()))
