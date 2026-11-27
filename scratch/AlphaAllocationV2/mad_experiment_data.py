"""Prepare a matched listing-age and rolling-MAD PatchTST experiment."""

import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from build_dataset import DATA, ROSTER, load_roster
from compare_patchtst import sha256
from rolling_mad import (
    MINIMUM,
    MULTIPLIER,
    WINDOW,
    clip_returns,
    price_features,
    reconstruct_close,
)
from train_v1_xgboost_comparison import OUTPUT as SHARED
from train_v1_xgboost_comparison import load_split, shared_inputs, write_json
from v1_hypothesis_data import SEVEN, daily_features, session_data, windows

ROOT = Path(__file__).resolve().parent
OUTPUT = DATA / "rolling_mad"
ARMS = ("raw", "mad")
IDENTITY_ERRORS = {
    "ABX": "Roster: Barrick; unqualified Yahoo ABX is not its US listing B/GOLD",
    "AI": "Roster: Air Liquide; downloaded AI is C3.ai",
    "CNR": "Roster: Canadian National Railway; US listing is CNI, not CNR",
    "OR": "Roster: L'Oreal; downloaded OR is a gold royalty company",
    "RKT": "Roster: Reckitt; downloaded RKT is Rocket Companies",
    "AMS": "Roster: Amadeus IT; downloaded AMS is American Shared Hospital Services",
    "CFR": "Roster: Richemont; downloaded CFR is Cullen/Frost Bankers",
    "CSL": "Roster: CSL Ltd; downloaded CSL is Carlisle Companies",
    "DOL": "Roster: Dollarama; downloaded DOL is a WisdomTree international ETF",
    "ENR": "Roster: Siemens Energy; downloaded ENR is Energizer Holdings",
    "SAN": "Roster: Sanofi; downloaded SAN is Banco Santander",
    "UCB": "Roster: UCB SA; downloaded UCB is United Community Banks",
    "WES": "Roster: Wesfarmers; downloaded WES is Western Midstream Partners",
}
LEGRAND_SOURCE = "https://www.adrbny.com/directory/dr-details/_jcr_content/root/drDetailsComponent.overview.overview.524671104.html"


def cohort_audit(manifest):
    rows = []
    for symbol in manifest["universe"]["symbols"]:
        first = manifest["symbols_with_data"].get(symbol, {}).get("first_session")
        evidence = "Frozen Yahoo OHLCV manifest; first row is not an independently verified IPO date"
        if symbol in IDENTITY_ERRORS:
            status, reason = "identity_exclusion", IDENTITY_ERRORS[symbol]
        elif first is None:
            status, reason = "missing_price_exclusion", "No archived OHLCV"
        elif symbol == "LGRDY":
            status, reason = (
                "retain",
                "ADR effective August 15, 2014; shorter Yahoo coverage is not a new IPO",
            )
            evidence = LEGRAND_SOURCE
        elif first > "2016-01-04":
            status, reason = (
                "later_series_exclusion",
                "Current price series begins after start-2016; listing-age proxy, not certified company IPO",
            )
        else:
            status, reason = (
                "retain",
                "Price evidence present at the archive's first 2016 session",
            )
        rows.append(
            {
                "symbol": symbol,
                "first_archived_session": first,
                "status": status,
                "reason": reason,
                "evidence": evidence,
            }
        )
    return pd.DataFrame(rows)


def complete_history(calendar, close, logs):
    """Extend the existing rebased source back to 2016 for causal MAD warmup."""
    archived = pd.read_csv(DATA / "daily_ohlcv_2016_2025.csv", parse_dates=["date"])
    historical = archived.pivot(index="date", columns="symbol", values="close").reindex(
        columns=close.columns
    )
    overlap = historical.index.intersection(close.index)
    factor = (
        (close.loc[overlap] / historical.loc[overlap])
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .iloc[-1]
    )
    sessions = calendar.sessions_in_range("2016-01-01", close.index[-1])
    full = historical.mul(factor).reindex(sessions)
    full.loc[close.index] = close
    full = full.where(np.isfinite(full) & full.gt(0))
    full_logs = np.log(full).diff()
    # Retain the existing unresolved-price and large-move masks, not new labels.
    full_logs.loc[logs.index] = logs
    flags = pd.read_csv(
        DATA / "friday/flagged_daily_returns.csv", parse_dates=["session"]
    )
    for row in flags.loc[flags.status.str.startswith("unresolved")].itertuples():
        if row.session in full_logs.index and row.symbol in full_logs:
            full_logs.loc[row.session, row.symbol] = np.nan
    return full, full_logs


def build_split(name, source_logs, features, symbols):
    raw, target, original = load_split(name)
    keep = original.get_level_values("symbol").isin(symbols)
    keys = original[keep]
    dates = keys.unique("friday").sort_values()
    matrices, observed_keys = [], []
    for start in range(0, len(dates), 10):
        index, x, mask, _ = windows(
            source_logs,
            features,
            dates[start : start + 10],
            use_features=True,
            year=True,
        )
        matched = index.isin(keys)
        matrices.append(shared_inputs(x[matched], mask[matched]))
        observed_keys.extend(index[matched])
    order = pd.MultiIndex.from_tuples(observed_keys, names=keys.names)
    if not order.is_unique or len(order) != len(keys) or not keys.isin(order).all():
        raise ValueError(f"MAD changed row eligibility in {name}")
    mad = np.concatenate(matrices)[order.get_indexer(keys)]
    if not np.isfinite(mad).all():
        raise ValueError(f"Nonfinite model inputs in {name}")
    baseline = np.asarray(raw[keep])
    np.testing.assert_array_equal(mad[..., [3, 4, 6]], baseline[..., [3, 4, 6]])
    np.save(OUTPUT / f"{name}_raw.npy", baseline)
    np.save(OUTPUT / f"{name}_mad.npy", mad)
    np.save(OUTPUT / f"{name}_y.npy", target[keep])
    original_index = pd.read_csv(SHARED / f"{name}_index.csv")
    original_index.loc[keep].to_csv(OUTPUT / f"{name}_index.csv", index=False)
    print(
        f"Prepared {name}: {len(keys):,} rows, {keys.unique('symbol').size} stocks",
        flush=True,
    )
    return {
        "rows": len(keys),
        "stocks": keys.unique("symbol").size,
        "first": str(dates.min().date()),
        "last": str(dates.max().date()),
    }


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = [
        OUTPUT / "identity_checks.json",
        ROSTER,
        DATA / "ohlcv_manifest.json",
        DATA / "daily_ohlcv_2016_2025.csv",
        DATA / "friday/flagged_daily_returns.csv",
        DATA / "news/news_v1.duckdb",
        SHARED / "dataset.json",
        ROOT.parent / "AlphaAllocationV1/data/v1_holdout/prices_2026.csv",
        ROOT.parent / "AlphaAllocationV1/data/v1_holdout/news/news_v1.duckdb",
    ]
    sources = {str(p.relative_to(ROOT.parents[1])): sha256(p) for p in files}
    sources.update(
        {p.name: sha256(p) for p in (Path(__file__), ROOT / "rolling_mad.py")}
    )
    path = OUTPUT / "dataset.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["sources"] != sources:
            raise ValueError("Frozen MAD dataset sources changed")
        for filename, digest in manifest["arrays"].items():
            if sha256(OUTPUT / filename) != digest:
                raise ValueError(f"Modified array: {filename}")
        return manifest
    frozen = json.loads((SHARED / "dataset.json").read_text())
    for filename, digest in frozen["arrays"].items():
        if sha256(SHARED / filename) != digest:
            raise ValueError(f"Original dataset changed: {filename}")
    listing = json.loads((DATA / "ohlcv_manifest.json").read_text())
    assert listing["universe"]["symbols"] == load_roster(ROSTER)["symbols"]
    audit = cohort_audit(listing)
    audit.to_csv(OUTPUT / "universe_audit.csv", index=False)
    symbols = audit.loc[audit.status.eq("retain"), "symbol"].tolist()
    write_json(OUTPUT / "retained_symbols.json", symbols)
    print(f"Universe audit: {audit.status.value_counts().to_dict()}", flush=True)
    calendar, _, close, volume, logs, _ = session_data()
    close, volume, logs = close[symbols], volume[symbols], logs[symbols]
    full_close, full_logs = complete_history(calendar, close, logs)
    clipped, adjustments = clip_returns(full_logs)
    adjustments.to_csv(OUTPUT / "clipped_returns.csv", index=False)
    model_close = reconstruct_close(full_close, full_logs, clipped)
    evidence = pd.concat(
        {
            "raw_close": full_close.stack(),
            "model_close": model_close.stack(),
            "raw_log_return": full_logs.stack(),
            "model_log_return": clipped.stack(),
        },
        axis=1,
    )
    evidence.index.names = ["session", "symbol"]
    with duckdb.connect() as db:
        db.from_df(evidence.reset_index()).write_parquet(
            str(OUTPUT / "model_prices.parquet")
        )
    # Preserve the original feature archive, including its known normalization.
    # Only the four price-derived inputs change; labels and eligibility do not.
    counts = {}
    for evaluation, names in ((False, ("train", "validation")), (True, ("test",))):
        features = daily_features(calendar, close, volume, logs, evaluation=evaluation)
        modified = price_features(
            features, full_close, full_logs, clipped, index=logs.index
        )
        for name in names:
            counts[name] = build_split(name, logs, modified, symbols)
        del features, modified
    manifest = {
        "sources": sources,
        "features": list(SEVEN),
        "shape": [255, 7],
        "splits": counts,
        "cohort_counts": {k: int(v) for k, v in audit.status.value_counts().items()},
        "window": WINDOW,
        "minimum": MINIMUM,
        "multiplier": MULTIPLIER,
        "seed": 42,
        "target_std": float(np.std(np.load(OUTPUT / "train_y.npy"), dtype=np.float64)),
        "clipped_observations": len(adjustments),
        "finite_daily_returns": int(full_logs.notna().sum().sum()),
        "arrays": {
            p.name: sha256(p)
            for p in OUTPUT.iterdir()
            if p.suffix == ".npy" or p.name.endswith("_index.csv")
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "listing_policy": "First archived series at start-2016, with verified pre-2016 LGRDY exception; IPO dates not independently certified",
    }
    write_json(path, manifest)
    return manifest


def load_arrays(split, arm):
    x = np.load(OUTPUT / f"{split}_{arm}.npy", mmap_mode="r")
    y = np.load(OUTPUT / f"{split}_y.npy")
    frame = pd.read_csv(
        OUTPUT / f"{split}_index.csv", parse_dates=["friday", "target_end"]
    )
    return x, y, pd.MultiIndex.from_frame(frame[["friday", "symbol"]])


if __name__ == "__main__":
    prepare()
