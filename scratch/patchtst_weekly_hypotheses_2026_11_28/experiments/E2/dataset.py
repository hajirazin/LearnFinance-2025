"""Freeze a matched offline panel using the production causal feature builder."""

import pickle
from datetime import date

import duckdb
import numpy as np
import pandas as pd
from evidence import DATA, REPO, code_hashes, read_json, sha256, write_json

from brain_api.core.patchtst.calendar import load_patchtst_calendar
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTData, WeeklyPatchTSTInputs
from brain_api.core.patchtst.weekly_training import weekly_fold_indices

PRICE_ROOT = REPO / "scratch/patchtst_geometry_sweep_2026_08_23/data"
ROSTER = REPO / (
    "brain_api/data/models/patchtst_halal_new/v2026-11-20-549d9adc9504/metadata.json"
)
NEWS = REPO / "brain_api/data/news/news_v1.duckdb"
# Previously verified collisions, not a volatility or asset-type exclusion.
COLLISIONS = {
    "ABX",
    "AI",
    "AMS",
    "CFR",
    "CNR",
    "CSL",
    "DOL",
    "ENR",
    "OR",
    "RKT",
    "SAN",
    "UCB",
    "WES",
}


class ReadOnlyNews:
    def _connect(self):
        return duckdb.connect(
            str(NEWS), read_only=True, config={"threads": 4, "memory_limit": "4GB"}
        )


def describe(panel, rows):
    anchors, ends = panel.anchor_dates[rows], panel.target_dates[rows]
    if len(rows) == 0:
        raise ValueError("Empty research fold")
    return {
        "rows": len(rows),
        "weeks": len(np.unique(anchors)),
        "symbols": len(np.unique(panel.symbols[rows])),
        "first_anchor": str(min(anchors)),
        "last_anchor": str(max(anchors)),
        "latest_label": str(max(ends)),
    }


def subset(panel, rows):
    # Arrays remain shared or mapped; selected training batches are copied later.
    return WeeklyPatchTSTData(
        panel.X[rows],
        panel.y[rows],
        panel.anchor_dates[rows],
        panel.target_dates[rows],
        panel.symbols[rows],
    )


def save_panel(panel, output):
    np.save(output / "X.npy", panel.X)
    np.save(output / "y.npy", panel.y)
    pd.DataFrame(
        {
            "anchor": panel.anchor_dates,
            "target_end": panel.target_dates,
            "symbol": panel.symbols,
        }
    ).to_csv(output / "index.csv", index=False)


def load_panel(output):
    index = pd.read_csv(output / "index.csv")
    return WeeklyPatchTSTData(
        np.load(output / "X.npy", mmap_mode="r"),
        np.load(output / "y.npy", mmap_mode="r"),
        pd.to_datetime(index.anchor).dt.date.to_numpy(),
        pd.to_datetime(index.target_end).dt.date.to_numpy(),
        index.symbol.to_numpy(),
    )


def prepare(config):
    output = DATA / "panel"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    source_identity = {
        "price_manifest": sha256(PRICE_ROOT / "adjusted_ohlcv_manifest.json"),
        "roster": sha256(ROSTER),
        "code": code_hashes(),
        "config": config.to_dict(),
    }
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != source_identity:
            raise ValueError(
                "Panel source/code identity changed; retain the old attempt"
            )
        for name, expected in manifest["artifacts"].items():
            if sha256(output / name) != expected:
                raise ValueError(f"Changed frozen panel artifact: {name}")
        return load_panel(output), manifest

    roster = sorted(read_json(ROSTER)["symbols"])
    manifest = read_json(PRICE_ROOT / "adjusted_ohlcv_manifest.json")
    prices, exclusions, audit = {}, [], []
    for symbol in roster:
        if symbol in COLLISIONS or symbol not in manifest["files"]:
            exclusions.append(
                {
                    "symbol": symbol,
                    "reason": "verified ticker identity collision"
                    if symbol in COLLISIONS
                    else "not in frozen price archive",
                }
            )
            continue
        item = manifest["files"][symbol]
        path = PRICE_ROOT / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Changed archived prices for {symbol}")
        bars = pd.read_csv(path, index_col=0, parse_dates=True)
        if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate/unordered prices for {symbol}")
        bars = bars.loc[:"2025-12-31"]
        prices[symbol] = bars
        audit.append(
            {
                "symbol": symbol,
                "rows": len(bars),
                "invalid_close": int(
                    (~np.isfinite(bars.close) | (bars.close <= 0)).sum()
                ),
                "nonpositive_volume": int((bars.volume <= 0).sum()),
                "large_daily_moves_retained": int(
                    (bars.close.pct_change(fill_method=None).abs() > 0.5).sum()
                ),
            }
        )
    if len(prices) < 300:
        raise ValueError(f"Insufficient archived roster breadth: {len(prices)}")
    print(
        f"Frozen roster: {len(prices)} retained, {len(exclusions)} excluded", flush=True
    )
    pd.DataFrame(exclusions).to_csv(output / "exclusions.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "price_audit.csv", index=False)
    calendar = load_patchtst_calendar("us", date(2015, 1, 1), date(2025, 12, 31))
    before = NEWS.stat()
    inputs = WeeklyPatchTSTInputs(
        config, calendar, prices, ReadOnlyNews(), news_revision_policy="point_in_time"
    )
    after = NEWS.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("News database changed during the read-only extraction")
    with (output / "features.pkl").open("wb") as handle:
        pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    panel = inputs.labeled(date(2017, 1, 1), date(2025, 12, 31))
    if not np.isfinite(panel.y).all():
        raise ValueError("Nonfinite raw labels")
    # labels ending in 2025 never enter either the fitting or validation panel.
    development_rows = np.flatnonzero(panel.target_dates <= date(2024, 12, 31))
    evaluation_rows = np.flatnonzero(panel.anchor_dates >= date(2025, 1, 1))
    development = subset(panel, development_rows)
    train, validation = weekly_fold_indices(development, config.validation_split)
    if max(development.target_dates[train]) >= min(
        development.anchor_dates[validation]
    ):
        raise ValueError("Fitting labels cross validation boundary")
    save_panel(panel, output)
    record = {
        "identity": source_identity,
        "roster": sorted(prices),
        "excluded": exclusions,
        "survivorship_bias": True,
        "news_revision_policy": "point_in_time",
        "news_evidence": inputs.news_evidence,
        "news_database": {
            "path": str(NEWS.relative_to(REPO)),
            "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
        },
        "folds": {
            "fitting": describe(development, train),
            "validation": describe(development, validation),
            "evaluation": describe(panel, evaluation_rows),
        },
        "artifacts": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    write_json(manifest_path, record)
    print(record["folds"], flush=True)
    return load_panel(output), record


def split_panel(panel):
    # Slice the date-sorted panel without copying its full feature matrix.
    end = int(np.searchsorted(panel.target_dates, date(2025, 1, 1)))
    start = int(np.searchsorted(panel.anchor_dates, date(2025, 1, 1)))
    return subset(panel, slice(0, end)), subset(panel, slice(start, None))
