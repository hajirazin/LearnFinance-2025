"""Score the untouched comparison weeks and run the shared HRP simulation."""

import json
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd
import torch
from build_dataset import DATA
from build_friday_dataset import friday_news, plt
from build_v1_holdout_data import OUTPUT
from compare_feature_sets import SEVEN
from compare_patchtst import sha256, write_json
from matplotlib.ticker import PercentFormatter
from patchtst_replay_data import WEEKLY_CONTEXT, weekly_windows
from patchtst_replay_models import build_model, model_output, simple_returns
from train_v1_holdout import MODELS
from v1_holdout_portfolio import (
    INITIAL_NAV,
    SPREAD_SLIPPAGE_BPS,
    allocation_path,
    simulate,
)

FRIDAYS = pd.date_range("2026-10-02", "2026-10-30", freq="W-FRI")
TERMINAL = pd.Timestamp("2026-11-09")
BASELINE_CONTEXT = 60


def load_prices(symbols, calendar):
    historical = pd.read_csv(
        DATA / "daily_ohlcv_2016_2025.csv",
        usecols=["date", "symbol", "close", "volume"],
        parse_dates=["date"],
    )
    recent = pd.read_csv(OUTPUT / "prices_2026.csv", parse_dates=["date"])
    archived = historical.pivot(index="date", columns="symbol", values="close").reindex(
        columns=symbols
    )
    latest = recent.pivot(index="date", columns="symbol", values="close").reindex(
        columns=symbols
    )
    anchor = archived.index.intersection(latest.index)
    # Rebase the archive to the recent adjustment convention using each symbol's
    # last valid overlapping close. This preserves historical price ratios.
    ratio = (latest.loc[anchor] / archived.loc[anchor]).where(
        lambda x: np.isfinite(x) & x.gt(0)
    )
    factor = ratio.ffill().iloc[-1]
    history = historical.loc[historical["symbol"].isin(symbols)].copy()
    history["close"] *= history["symbol"].map(factor)
    bars = pd.concat(
        [history.loc[history["date"].lt(latest.index.min())], recent], ignore_index=True
    )
    sessions = calendar.sessions_in_range("2019-01-01", TERMINAL)
    frames = {
        key: bars.pivot(index="date", columns="symbol", values=key).reindex(
            index=sessions, columns=symbols
        )
        for key in ("close", "volume", "open", "raw_open")
    }
    return frames


def forecast_inputs(frames, calendar):
    close, volume = frames["close"], frames["volume"]
    usable = np.isfinite(close) & close.gt(0) & np.isfinite(volume) & volume.gt(0)
    clean = close.where(usable)
    logs = np.log(clean).diff()
    # Exclude unresolved, unusually large daily moves from trailing model inputs.
    large_move = np.expm1(logs).abs().ge(0.40)
    logs = logs.mask(large_move)
    grid = close.index[
        (close.index.day_of_week == 4)
        & (close.index >= pd.Timestamp("2026-07-17"))
        & (close.index <= FRIDAYS[-1])
    ]
    news = friday_news(
        grid, calendar, close.columns, db_path=OUTPUT / "news/news_v1.duckdb"
    )
    news_frame = (
        news["news_sentiment"]
        .unstack("symbol")
        .reindex(index=grid, columns=close.columns)
    )
    features = {
        "momentum_20d": np.expm1(logs.rolling(20).sum()),
        "momentum_12_1": np.expm1(logs.rolling(231).sum().shift(21)),
        "volatility_20d": logs.rolling(20).std(ddof=1) * np.sqrt(252),
        "relative_volume_20d": volume.where(usable)
        / volume.where(usable).shift().rolling(20).median(),
        "news_sentiment": news_frame,
        "close": clean,
        "volume": volume.where(usable),
    }
    weekly = np.stack([features[name].loc[grid].to_numpy() for name in SEVEN], axis=-1)
    weekly = weekly_windows(weekly, WEEKLY_CONTEXT)[grid.get_indexer(FRIDAYS)]
    daily = np.stack(
        [logs.loc[:friday].tail(BASELINE_CONTEXT).to_numpy().T for friday in FRIDAYS],
        axis=0,
    )
    weekly_ok = np.isfinite(weekly).all(axis=(2, 3))
    daily_ok = np.isfinite(daily).all(axis=2)
    common = weekly_ok & daily_ok
    coverage = pd.DataFrame(
        {
            "daily_eligible": daily_ok.sum(axis=1),
            "weekly_eligible": weekly_ok.sum(axis=1),
            "common_eligible": common.sum(axis=1),
            "verified_empty_news": news["news_status"]
            .eq("verified_empty")
            .groupby(level="friday")
            .sum()
            .reindex(FRIDAYS),
        },
        index=FRIDAYS,
    ).rename_axis("friday")
    if coverage["common_eligible"].lt(300).any():
        raise ValueError(f"Incomplete shared coverage:\n{coverage}")
    keys = pd.MultiIndex.from_product(
        [FRIDAYS, close.columns], names=["friday", "symbol"]
    )[common.ravel()]
    return keys, daily[common, :, None], weekly[common], coverage


def forecast(name, x, *, daily):
    metadata = json.loads((OUTPUT / f"{name}.json").read_text())
    weights = OUTPUT / f"{name}.pt"
    if sha256(weights) != metadata["weights_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {name}")
    if not daily:
        x = (x - np.array(metadata["feature_means"])) / np.array(
            metadata["feature_scales"]
        )
    model = build_model(x, daily=daily).eval()
    if model.config.context_length != metadata["config"]["context_length"]:
        raise ValueError(f"Context mismatch for {name}")
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        inputs = torch.as_tensor(x, dtype=torch.float32)
        values = torch.cat(
            [
                model_output(
                    model, batch, torch.ones_like(batch, dtype=torch.bool), daily=daily
                )
                for batch in inputs.split(256)
            ]
        )
        scores = simple_returns(values * metadata["target_scale"], daily=daily).numpy()
    if not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite holdout forecasts for {name}")
    return scores


def main():
    torch.set_num_threads(4)
    panel_path = DATA / "friday/friday_dataset.parquet"
    with duckdb.connect() as db:
        symbols = (
            db.execute(
                "SELECT DISTINCT symbol FROM read_parquet(?) ORDER BY symbol",
                [str(panel_path)],
            )
            .df()["symbol"]
            .tolist()
        )
    calendar = xcals.get_calendar("XNYS")
    frames = load_prices(symbols, calendar)
    keys, daily, weekly, coverage = forecast_inputs(frames, calendar)
    scores = pd.DataFrame(
        {
            MODELS[0]: forecast(MODELS[0], daily, daily=True),
            MODELS[1]: forecast(MODELS[1], weekly, daily=False),
        },
        index=keys,
    )
    coverage.to_csv(OUTPUT / "coverage.csv")
    scores.to_csv(OUTPUT / "scores.csv")
    allocations, trades, navs, ledgers = [], [], [], []
    metrics = {}
    for name in MODELS:
        path = allocation_path(scores[name], frames["close"])
        for friday, weights, selection in path:
            allocations.extend(
                {
                    "model": name,
                    "friday": friday,
                    "symbol": symbol,
                    "weight": weight,
                    "selection_reason": selection.reasons[symbol],
                }
                for symbol, weight in weights.items()
            )
        metrics[name] = {}
        for bps in SPREAD_SLIPPAGE_BPS:
            result, ledger, nav, orders = simulate(
                path, frames["open"], frames["raw_open"], frames["close"], TERMINAL, bps
            )
            metrics[name][str(bps)] = result
            ledgers.append(ledger.assign(model=name, bps=bps))
            navs.append(nav.assign(model=name, bps=bps))
            trades.append(orders.assign(model=name, bps=bps))
    pd.DataFrame(allocations).to_csv(OUTPUT / "weights.csv", index=False)
    pd.concat(trades).to_csv(OUTPUT / "trades.csv", index=False)
    weekly_results = pd.concat(ledgers)
    weekly_results.to_csv(OUTPUT / "weekly_results.csv", index=False)
    paired = weekly_results.loc[weekly_results["bps"].eq(10)].pivot(
        index="fill", columns="model", values="net_return"
    )
    differences = paired[MODELS[1]] - paired[MODELS[0]]
    daily_nav = pd.concat(navs)
    daily_nav.to_csv(OUTPUT / "nav.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name, label in zip(
        MODELS, ("V1 close-only PatchTST", "Seven-feature PatchTST"), strict=True
    ):
        rows = daily_nav.loc[daily_nav["model"].eq(name) & daily_nav["bps"].eq(10)]
        ax.plot(rows["date"], rows["nav"] / INITIAL_NAV - 1, label=label)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(
        title="V1 holdout after simulated trading costs", ylabel="Cumulative net return"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT / "net_nav.png", dpi=150)
    plt.close(fig)
    write_json(
        OUTPUT / "summary.json",
        {
            "start_open": "2026-10-05",
            "terminal_open": TERMINAL,
            "rebalances": len(FRIDAYS),
            "initial_nav": INITIAL_NAV,
            "main_spread_and_slippage_bps_per_side": 10,
            "universe_size": len(symbols),
            "metrics": metrics,
            "challenger_weeks_ahead": int(differences.gt(0).sum()),
            "mean_paired_weekly_difference": differences.mean(),
            "compounded_net_difference_by_bps": {
                str(bps): metrics[MODELS[1]][str(bps)]["net_return"]
                - metrics[MODELS[0]][str(bps)]["net_return"]
                for bps in SPREAD_SLIPPAGE_BPS
            },
            "sha256": {
                p.name: sha256(p)
                for p in [
                    panel_path,
                    OUTPUT / "prices_2026.csv",
                    OUTPUT / "news/news_v1.duckdb",
                    *[OUTPUT / f"{model}.pt" for model in MODELS],
                    Path(__file__),
                    Path(__file__).with_name("v1_holdout_portfolio.py"),
                    Path(__file__).with_name("V1_HOLDOUT_PROTOCOL.md"),
                ]
            },
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
