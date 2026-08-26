"""Data loading utilities for PatchTST training and inference."""

import numpy as np
import pandas as pd

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.patchtst.config import PatchTSTConfig


def align_multivariate_data(
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> dict[str, pd.DataFrame]:
    """Align OHLCV data into feature channels for PatchTST.

    Computes OHLCV log returns and filters to ``config.feature_names``.
    Locked production config uses one close-return channel.

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        config: PatchTST configuration (feature_names selects channels)

    Returns:
        Dict of symbol -> aligned DataFrame with config.num_input_channels columns
    """
    aligned: dict[str, pd.DataFrame] = {}

    for symbol, price_df in prices.items():
        if len(price_df) < config.context_length + 5:
            continue

        # OHLCV log returns; config.feature_names selects the model channels
        features_df = compute_ohlcv_log_returns(
            price_df, use_returns=config.use_returns
        )

        # Ensure column order matches config.feature_names
        features_df = features_df[config.feature_names]

        # CRITICAL VERIFICATION: Channel count
        assert len(features_df.columns) == config.num_input_channels, (
            f"CRITICAL: Expected {config.num_input_channels} channels, got {len(features_df.columns)}"
        )

        # Quick data quality check (no heavy stats computation)
        nan_count = features_df.isna().sum().sum()
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()

        # Only log warnings for problematic symbols
        if nan_count > 0:
            print(f"[PatchTST] WARNING: {symbol} has {nan_count} NaN values")
        if inf_count > 0:
            print(f"[PatchTST] WARNING: {symbol} has {inf_count} Inf values")

        if len(features_df) >= config.context_length:
            aligned[symbol] = features_df

    # Summary log at the end (not per-symbol)
    print(
        f"[PatchTST] Aligned {len(aligned)} symbols with {config.num_input_channels} channels each"
    )

    return aligned
