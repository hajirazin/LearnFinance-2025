"""Pre-2026-only feature/HMM fitting and reusable causal state templates."""

import gc
import json
import pickle
from datetime import date

import numpy as np
import pandas as pd

from research_support import (
    ROOT,
    ReadOnlyNewsStore,
    file_hash,
    protected_hashes,
    write_json,
)

from brain_api.core.ppo_discovery.news_adapter import load_historical_ppo_news_features
from brain_api.core.ppo_discovery.regime import (
    fit_ppo_regime_hmm,
    weekly_regime_probabilities,
)
from brain_api.core.ppo_discovery.schemas import UniverseSnapshot
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    news_window_starts_at_or_after_archive,
    weekly_trade_clock,
)
from brain_api.core.prices import repair_ohlc_envelope


def main():
    if (ROOT / "market_dataset.pkl").exists():
        raise RuntimeError(
            "Prepared evidence already exists; do not overwrite a training dataset"
        )
    write_json("protected_before.json", protected_hashes())
    prices = {
        s: repair_ohlc_envelope(f)
        for s, f in pd.read_pickle(ROOT / "market_prices.pkl").items()
    }
    payload = json.loads((ROOT / "market_roster.json").read_text())
    payload["sorted_symbols"] = tuple(payload["sorted_symbols"])
    snapshot = UniverseSnapshot(**payload)
    symbols = snapshot.sorted_symbols
    clock = weekly_trade_clock(date(2019, 1, 1), date(2026, 9, 14))
    rows = []
    for i, cutoff in enumerate(actor_cutoff_datetimes(clock)[:-1]):
        if cutoff.date() > date(
            2026, 9, 4
        ) or not news_window_starts_at_or_after_archive(cutoff):
            continue
        eligible = sum(
            len(f) > 0
            and f.index.searchsorted(pd.Timestamp(cutoff.date()), side="right") >= 253
            for s, f in prices.items()
            if s in symbols
        )
        if eligible < 10:
            continue
        start, end = clock.rebalance_sessions[i : i + 2]
        split = (
            "train"
            if end.date() <= date(2025, 12, 31)
            else "test"
            if start.date() >= date(2026, 1, 1)
            else "bridge_excluded"
        )
        rows.append({"cutoff": cutoff, "start": start, "end": end, "split": split})
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    print(
        f"Causal split train={len(train)} test={len(test)}, fitting HMM through {train[-1]['cutoff'].date()}",
        flush=True,
    )
    hmm = fit_ppo_regime_hmm(
        prices, start_date=date(2019, 1, 1), cutoff=train[-1]["cutoff"].date()
    )
    active = [r for r in rows if r["split"] != "bridge_excluded"]
    regimes = weekly_regime_probabilities(
        hmm,
        prices,
        start_date=date(2019, 1, 1),
        completed_through=active[-1]["cutoff"].date(),
        weekly_cutoffs=[r["cutoff"].date() for r in active],
    )
    print("Reading complete news windows", flush=True)
    news = load_historical_ppo_news_features(
        [r["cutoff"] for r in active], symbols, store=ReadOnlyNewsStore()
    )
    total = np.zeros(4)
    total_sq = np.zeros(4)
    count = 0
    log_counts = []
    templates = []
    for i, row in enumerate(active):
        cutoff = row["cutoff"]
        sliced = {
            s: f.loc[: pd.Timestamp(cutoff.date())]
            for s, f in prices.items()
            if not f.empty and f.index.min().date() <= cutoff.date()
        }
        calm, stress = regimes[cutoff.date()]
        state = build_ppo_discovery_state(
            StateBuildRequest(
                as_of=cutoff,
                universe_snapshot=snapshot,
                ohlcv_by_symbol=sliced,
                news_by_symbol=news[cutoff],
                current_weights={"CASH": 1.0},
                p_calm=calm,
                p_stress=stress,
                spy_closes=sliced["SPY"]["close"].to_numpy(),
            )
        )
        if row["split"] == "train":
            raw = state.price_history[state.asset_mask].reshape(-1, 4)
            total += raw.sum(axis=0)
            total_sq += (raw * raw).sum(axis=0)
            count += len(raw)
            log_counts.extend(n.log1p_article_count for n in news[cutoff].values())
        templates.append({**row, "state": state})
        if (i + 1) % 25 == 0:
            print(f"Prepared causal state {i + 1}/{len(active)}", flush=True)
    channel_mean = total / count
    channel_scale = np.sqrt(np.maximum(total_sq / count - channel_mean**2, 1e-24))
    news_mean = float(np.mean(log_counts))
    news_scale = float(np.std(log_counts))
    if news_scale < 1e-12:
        news_scale = 1.0
    scalers = {
        "encoder_channels": {
            "mean": channel_mean.tolist(),
            "scale": channel_scale.tolist(),
        },
        "log1p_article_count": {"mean": news_mean, "scale": news_scale},
    }
    for row in templates:
        state = row["state"]
        mask = state.asset_mask
        state.price_history[mask] = (
            state.price_history[mask] - channel_mean
        ) / channel_scale
        state.price_history = state.price_history.astype(np.float32)
        state.asset_features[mask, 6] = (
            state.asset_features[mask, 6] - news_mean
        ) / news_scale
        state.state_digest = ""  # Internal research template; never sent to an API.
    starts = pd.DatetimeIndex([r["start"] for r in templates])
    ends = pd.DatetimeIndex([r["end"] for r in templates])
    opens = {
        s: f["open"].reindex(starts).to_numpy(dtype=float) for s, f in prices.items()
    }
    next_opens = {
        s: f["open"].reindex(ends).to_numpy(dtype=float) for s, f in prices.items()
    }
    for index, row in enumerate(templates):
        row["opens"] = {
            s: float(v[index])
            for s, v in opens.items()
            if np.isfinite(v[index]) and v[index] > 0
        }
        row["returns"] = {
            s: float(next_opens[s][index] / v[index] - 1)
            for s, v in opens.items()
            if np.isfinite(v[index])
            and v[index] > 0
            and np.isfinite(next_opens[s][index])
            and next_opens[s][index] > 0
        }
    # Array caching must not change the canonical feature calculations.
    for sample in [templates[0], templates[len(train) - 1], templates[-1]]:
        cached = sample["state"]
        sliced = {
            s: f.loc[: pd.Timestamp(sample["cutoff"].date())]
            for s, f in prices.items()
            if not f.empty and f.index.min().date() <= sample["cutoff"].date()
        }
        proper = build_ppo_discovery_state(
            StateBuildRequest(
                as_of=sample["cutoff"],
                universe_snapshot=snapshot,
                ohlcv_by_symbol=sliced,
                news_by_symbol=cached.news_by_symbol,
                current_weights={"CASH": 1.0},
                p_calm=float(cached.globals[0]),
                p_stress=float(cached.globals[1]),
                spy_closes=sliced["SPY"]["close"].to_numpy(),
                feature_scalers=scalers,
            )
        )
        np.testing.assert_array_equal(cached.asset_mask, proper.asset_mask)
        np.testing.assert_allclose(
            cached.price_history,
            proper.price_history.astype(np.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            cached.asset_features, proper.asset_features, rtol=1e-10, atol=1e-10
        )
    print(
        "Canonical feature-cache parity passed on first train, last train and last test dates",
        flush=True,
    )
    manifest = {
        "outer_cutoff": "2025-12-31",
        "train_weeks": len(train),
        "test_weeks": len(test),
        "train_first_actor": str(train[0]["cutoff"]),
        "train_last_actor": str(train[-1]["cutoff"]),
        "last_training_outcome": str(train[-1]["end"]),
        "test_first_decision_open": str(test[0]["start"]),
        "test_last_decision_open": str(test[-1]["start"]),
        "test_last_outcome": str(test[-1]["end"]),
        "hmm_cutoff": str(hmm.training_cutoff_date),
        "scalers_fit": "train only, streaming population moments",
        "prices_sha256": file_hash(ROOT / "market_prices.pkl"),
        "snapshot": snapshot.to_dict(),
        "news_missing_fallback": False,
        "retrospective_test": True,
        "current_roster_survivorship_bias": True,
        "excluded_bridge_weeks": [
            str(r["cutoff"]) for r in rows if r["split"] == "bridge_excluded"
        ],
    }
    data = {
        "templates": templates,
        "snapshot": snapshot,
        "scalers": scalers,
        "hmm": hmm,
        "manifest": manifest,
    }
    with (ROOT / "market_dataset.pkl").open("wb") as handle:
        pickle.dump(data, handle, protocol=5)
    write_json("market_data_manifest.json", manifest)
    write_json("market_scalers.json", scalers)
    write_json("market_hmm.json", hmm.to_dict())
    del news, templates, data
    gc.collect()
    print("Historical causal dataset ready", flush=True)


if __name__ == "__main__":
    main()
