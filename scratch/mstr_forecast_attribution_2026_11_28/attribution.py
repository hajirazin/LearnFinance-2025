"""Explain a frozen forecast with exact seven-channel Shapley values; no fitting."""

import hashlib
import json
from datetime import date
from math import factorial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from brain_api.core.patchtst.calendar import inference_calendar
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTInputs
from brain_api.core.patchtst.weekly_training import predict_weekly_returns
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "brain_api/data"
VERSION = "v2026-11-27-611c30dfacf0"
CUTOFF = date(2026, 11, 27)


def exact_channel_shapley(values):
    """Average marginal contributions over every ordering of input channels."""
    values = np.asarray(values, dtype=float)
    size = len(values)
    if size < 2 or size & (size - 1) or not np.isfinite(values).all():
        raise ValueError("Require 2**n finite coalition predictions")
    channels = size.bit_length() - 1
    result = np.zeros(channels)
    for channel in range(channels):
        bit = 1 << channel
        for subset in range(size):
            if subset & bit:
                continue
            count = subset.bit_count()
            weight = (
                factorial(count) * factorial(channels - count - 1) / factorial(channels)
            )
            result[channel] += weight * (values[subset | bit] - values[subset])
    return result


class ReadOnlyNews:
    def _connect(self):
        return duckdb.connect(str(DATA / "news/news_v1.duckdb"), read_only=True)


def main():
    torch.set_num_threads(1)
    local = ROOT / "data"
    local.mkdir(exist_ok=True)
    artifact = PatchTSTHalalNewModelStorage(DATA).load_version_artifacts(VERSION)
    artifact.model.cpu().eval()
    calendar = inference_calendar("us", CUTOFF)
    anchor = calendar.anchor(CUTOFF)
    path = local / "mstr_prices.csv"
    if not path.exists():
        yf.set_tz_cache_location(str(local / "yfinance_cache"))
        raw = yf.download(
            "MSTR",
            start=calendar.sessions[0].date().isoformat(),
            end="2026-11-28",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=30,
        )
        if raw.empty:
            raise ValueError("No MSTR prices returned")
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs("MSTR", axis=1, level="Ticker")
        raw.columns = raw.columns.str.lower()
        raw.to_csv(path, index_label="date")
    prices = pd.read_csv(path, index_col="date", parse_dates=True)
    closes = calendar.closes.loc[: str(anchor)]
    inputs = WeeklyPatchTSTInputs(
        artifact.config, type(calendar)(closes), {"MSTR": prices}, ReadOnlyNews()
    )
    window, count = inputs.window("MSTR", anchor)
    if window is None:
        raise ValueError("MSTR input window is not eligible")
    channels = len(artifact.config.feature_names)
    membership = (
        np.arange(1 << channels)[:, None, None]
        & (1 << np.arange(channels))[None, None, :]
    ) > 0
    matrix = np.where(membership, window[None, :, :], 0).astype(np.float32)
    scores = (
        predict_weekly_returns(
            artifact.model, matrix, artifact.feature_scaler, batch_size=8
        )
        * 100
    )
    contributions = exact_channel_shapley(scores)
    assert np.isclose(contributions.sum(), scores[-1] - scores[0], atol=1e-7)
    records = []
    for index, feature in enumerate(artifact.config.feature_names):
        records.append(
            {
                "feature": feature,
                "latest_raw_value": float(inputs.features["MSTR"][-1, index]),
                "latest_normalized_value": float(window[-1, index]),
                "shapley_percentage_points": float(contributions[index]),
                "prediction_without_feature_pct": float(
                    scores[((1 << channels) - 1) ^ (1 << index)]
                ),
            }
        )
    cached = json.loads(
        (DATA / "cache/universe/halal_filtered_2026-11.json").read_text()
    )
    original = next(
        row["predicted_weekly_return_pct"]
        for row in cached["stocks"]
        if row["symbol"] == "MSTR"
    )
    result = {
        "model": VERSION,
        "cutoff": str(anchor),
        "context_sessions": count,
        "cached_selection_forecast_pct": original,
        "reconstructed_forecast_pct": float(scores[-1]),
        "reconstruction_difference_pp": float(scores[-1] - original),
        "all_channels_at_window_mean_forecast_pct": float(scores[0]),
        "features": records,
        "method": "Exact group Shapley over 128 channel coalitions, relative to zero normalized histories; pre-cutoff news revision policy.",
        "limitations": "Synthetic masked inputs can be out of distribution; correlated price-derived channels share information. This explains model sensitivity for one reconstructed forecast, not causation or general importance.",
        "prices_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(
            (DATA / "models/patchtst_halal_new" / VERSION / "weights.pt").read_bytes()
        ).hexdigest(),
    }
    (ROOT / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    np.savez(local / "coalitions.npz", normalized_window=window, predictions_pct=scores)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
