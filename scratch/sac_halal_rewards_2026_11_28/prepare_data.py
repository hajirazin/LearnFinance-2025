"""Build the real SAC training panel using read-only production evidence."""

import hashlib
import json
import os
import pickle
import socket
import subprocess
from datetime import date
from unittest.mock import patch

import duckdb
import pandas as pd
import torch
from experiment_settings import BENCHMARK, ROOT, UNIVERSE

from brain_api.core.forecaster_snapshot_identity import (
    expected_dec31_walkforward_snapshot_hash,
)
from brain_api.core.patchtst.config import US_WEEKLY_CONFIG
from brain_api.core.portfolio_rl.data_loading import build_rl_training_signals
from brain_api.core.portfolio_rl.weekly_patchtst import score_sac_forecasts
from brain_api.core.sac.market_history import extract_aligned_market_history
from brain_api.core.sac.trade_clock import (
    build_sac_weekly_trade_clock,
    extract_session_open_prices,
)
from brain_api.core.sac.training import build_training_data
from brain_api.news.store import NewsStore
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

REPO = ROOT.parents[1]
PRODUCTION_DATA = REPO / "brain_api/data"


class ReadOnlyNewsStore(NewsStore):
    def __init__(self, base_path=None):
        self.base_path = PRODUCTION_DATA
        self.db_path = self.base_path / "news/news_v1.duckdb"
        self.lock_path = self.db_path.with_suffix(".lock")

    def _connect(self):
        return duckdb.connect(str(self.db_path), read_only=True)

    def _with_write_lock(self, fn):
        raise RuntimeError("Experiment forbids production news writes")


def require_local_snapshot(*, snapshot_storage, cutoff_date):
    digest = expected_dec31_walkforward_snapshot_hash(
        forecaster_bucket="patchtst_halal_new",
        cutoff_date=cutoff_date,
        config_dict=US_WEEKLY_CONFIG.to_dict(),
    )
    if not snapshot_storage.snapshot_exists(cutoff_date, digest):
        raise ValueError(f"Required local snapshot missing: {cutoff_date}-{digest}")
    return True


def forbid_network(*args, **kwargs):
    raise RuntimeError("Preparation must use saved prices and local news/snapshots")


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(1)
    raw_path = ROOT / "evidence/adjusted_ohlcv.parquet"
    raw = pd.read_parquet(raw_path)
    roster = json.loads((ROOT / "evidence/roster.json").read_text())
    symbols = sorted(stock["symbol"] for stock in roster["stocks"])
    prices = {}
    for symbol in sorted({*symbols, "SPY", "^VIX", BENCHMARK}):
        frame = raw[symbol].copy()
        frame.columns = frame.columns.str.lower()
        prices[symbol] = frame.dropna(subset=["open", "close", "volume"])

    start, end = date(2018, 1, 1), date(2026, 11, 27)
    clock = build_sac_weekly_trade_clock(start, end)
    market_dates, spy, vix = extract_aligned_market_history(
        prices,
        start_date=date(2016, 1, 4),
        completed_through=clock.transition_actor_cutoffs[-1].date(),
    )
    weekly_prices = {
        symbol: extract_session_open_prices(
            prices[symbol], clock.rebalance_sessions, symbol=symbol
        )
        for symbol in symbols
    }

    # Provider boundary is explicit: no HF resolution, no downloads, no ETL.
    def saved_prices(requested_symbols, *_args, **_kwargs):
        return {symbol: prices[symbol] for symbol in requested_symbols}

    def local_storage(forecaster_type):
        return SnapshotLocalStorage(forecaster_type, base_path=PRODUCTION_DATA)

    evidence = {}
    with (
        patch.object(socket.socket, "connect", forbid_network),
        patch(
            "brain_api.core.portfolio_rl.weekly_patchtst.load_prices_yfinance",
            saved_prices,
        ),
        patch(
            "brain_api.core.portfolio_rl.weekly_patchtst.SnapshotLocalStorage",
            local_storage,
        ),
        patch(
            "brain_api.core.portfolio_rl.weekly_patchtst.ensure_snapshot_for_bucket",
            require_local_snapshot,
        ),
        patch("brain_api.core.patchtst.weekly_data.NewsStore", ReadOnlyNewsStore),
    ):
        signals = build_rl_training_signals(
            prices,
            symbols,
            start,
            end,
            weekly_cutoffs=clock.transition_actor_cutoffs,
            store=ReadOnlyNewsStore(),
        )
        forecasts = score_sac_forecasts(
            symbols,
            clock.transition_actor_cutoffs,
            clock.transition_start_sessions,
            start=start,
            end=end,
            evidence=evidence,
        )
    data = build_training_data(
        weekly_prices,
        signals,
        forecasts,
        symbols,
        weekly_dates=list(clock.transition_actor_cutoffs),
        market_dates=market_dates,
        spy_adjusted_closes=spy,
        vix_closes=vix,
    )
    benchmark = extract_session_open_prices(
        prices[BENCHMARK], clock.rebalance_sessions, symbol=BENCHMARK
    )
    spy_open = extract_session_open_prices(
        prices["SPY"], clock.rebalance_sessions, symbol="SPY"
    )
    panel = {
        "data": data,
        f"{BENCHMARK.lower()}_returns": benchmark[1:] / benchmark[:-1] - 1,
        "spy_returns": spy_open[1:] / spy_open[:-1] - 1,
        f"{BENCHMARK.lower()}_prices": benchmark[:-1],
        "sessions": [value.date().isoformat() for value in clock.rebalance_sessions],
    }
    panel_path = ROOT / "evidence/panel.pkl"
    panel_path.write_bytes(pickle.dumps(panel))
    manifest = {
        "run_id": f"scratch:sac-{UNIVERSE}-rewards:{BENCHMARK}:2026-11-28",
        "universe": UNIVERSE,
        "benchmark": BENCHMARK,
        "attempt": 1,
        "as_of": "2026-11-28",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "symbols": symbols,
        "n_weeks": data.n_weeks,
        "train_weeks": data.n_weeks - 104,
        "test_weeks": 104,
        "test_start": panel["sessions"][-105],
        "test_end": panel["sessions"][-1],
        "min_eligible_assets": int(data.asset_masks.sum(axis=1).min()),
        "price_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "price_returns": "Yahoo auto_adjust=True OHLCV; dividend/split-adjusted open-to-next-open",
        "universe_bias": f"November 2026 {UNIVERSE} roster applied to history; survivorship/selection bias",
        "forecast_evidence": evidence,
        "training_hyperparameters": "Production SAC v3 defaults; 50000 steps; seeds 42,123,2026",
        "test_policy": "Same trailing 104 weeks for every run; no test-based hyperparameter selection",
    }
    (ROOT / "evidence/manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
