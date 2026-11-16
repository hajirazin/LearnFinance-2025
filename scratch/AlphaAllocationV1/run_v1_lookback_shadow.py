"""Freeze close-only 60-session and one-year forecasts and settle the comparison."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd
import torch
from brain_api.core.sticky_selection import select_with_rank_band
from build_dataset import ROSTER, load_roster
from compare_patchtst import sha256, write_json
from fetch_v1_lookback_prices import RESULT
from run_v1_hypotheses import scores_for
from v1_hypothesis_data import DATA, OUTPUT, windows

ARMS = {"60-session": "current", "one-year": "h2_year"}
QTY = 15
HOLD_RANK = 20
LARGE_DAILY_MOVE = 0.40


def snapshot(path):
    meta = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != meta["sha256"] or meta["roster_sha256"] != sha256(ROSTER):
        raise ValueError(f"Snapshot or roster hash mismatch: {path}")
    with duckdb.connect() as db:
        prices = db.read_parquet(str(path)).df()
    prices["date"] = prices.date.dt.as_unit("ns")
    if prices.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate symbol-session rows in snapshot")
    return prices, meta


def price_panel(prices, days, symbols):
    close = prices.pivot(index="date", columns="symbol", values="close").reindex(
        index=days, columns=symbols
    )
    volume = prices.pivot(index="date", columns="symbol", values="volume").reindex_like(
        close
    )
    valid = np.isfinite(close) & close.gt(0) & np.isfinite(volume) & volume.gt(0)
    return close.where(valid)


def input_log_returns(prices, decision, cal, symbols):
    # Materialize the as-of cut before any returns or eligibility are computed.
    before = prices.loc[prices.date.le(decision)]
    if not before.date.eq(decision).any():
        raise ValueError(f"Snapshot has no closes for {decision.date()}")
    if before.date.min() > decision - pd.Timedelta(days=365):
        raise ValueError("Snapshot lacks the one-year context warmup")
    days = cal.sessions_in_range(before.date.min(), decision)
    logs = np.log(price_panel(before, days, symbols)).diff()
    audit = pd.read_csv(
        DATA / "friday/flagged_daily_returns.csv", parse_dates=["session"]
    )
    for row in audit.loc[audit.status.str.startswith("unresolved")].itertuples():
        if row.session in logs.index and row.symbol in logs.columns:
            logs.loc[row.session, row.symbol] = np.nan
    recent = logs.index.year >= 2026
    logs.loc[recent] = logs.loc[recent].mask(
        np.expm1(logs.loc[recent]).abs().ge(LARGE_DAILY_MOVE)
    )
    return logs


def chosen(score, prior):
    selection = select_with_rank_band(score.to_dict(), prior, QTY, HOLD_RANK)
    if len(selection.selected) != QTY:
        raise ValueError("Incomplete sticky basket")
    return selection.selected


def archived_prior(decision):
    source = pd.read_csv(OUTPUT / "scores.csv", parse_dates=["friday"])
    shared = (
        source.loc[source.friday.lt(decision)]
        .set_index(["friday", "symbol"])[list(ARMS.values())]
        .dropna()
    )
    prior = dict.fromkeys(ARMS)
    last = None
    for day, frame in shared.groupby(level="friday", sort=True):
        frame = frame.droplevel("friday")
        prior = {arm: set(chosen(frame[key], prior[arm])) for arm, key in ARMS.items()}
        last = day.date().isoformat()
    return prior, last


def prior_picks(decision):
    runs = sorted(
        p
        for p in RESULT.iterdir()
        if p.name < decision.date().isoformat() and (p / "forecast.json").is_file()
    )
    if runs:
        row = json.loads((runs[-1] / "forecast.json").read_text())
        return {arm: set(picks) for arm, picks in row["picks"].items()}, row[
            "decision_date"
        ]
    return archived_prior(decision)


def freeze(decision, price_path):
    if decision <= pd.Timestamp("2026-10-30"):
        raise ValueError(
            "Shadow forecasts begin after the archived H1–H27 decision period"
        )
    cal = xcals.get_calendar("XNYS")
    if decision.day_of_week != 4 or not cal.is_session(decision):
        raise ValueError("Forecast decisions must be trading Fridays")
    if cal.session_close(decision) > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Decision session has not closed")
    target = cal.session_offset(decision, 5)
    output = RESULT / decision.date().isoformat()
    if output.exists():
        raise FileExistsError(f"Forecast already frozen: {output}")
    prices, source = snapshot(price_path)
    symbols = load_roster(ROSTER)["symbols"]
    logs = input_log_returns(prices, decision, cal, symbols)
    predictions, context_counts, checkpoints = {}, {}, {}
    for arm, key in ARMS.items():
        index, x, mask, y = windows(
            logs,
            None,
            pd.DatetimeIndex([decision]),
            use_features=False,
            year=arm == "one-year",
        )
        if np.isfinite(y).any():
            raise ValueError("Forecast inputs unexpectedly include labels")
        values = scores_for(key, x, mask, "cpu")
        predictions[arm] = pd.Series(values, index=index.get_level_values("symbol"))
        context_counts[arm] = sorted(np.unique(mask[:, :, 0].sum(axis=1)).tolist())
        checkpoints[arm] = {
            "file": f"{key}.pt",
            "sha256": sha256(OUTPUT / f"{key}.pt"),
            "scored_candidates": len(index),
        }
    common = pd.DataFrame(predictions).dropna().rename_axis("symbol").sort_index()
    if len(common) < QTY:
        raise ValueError("Too few common eligible stocks")
    prior, prior_date = prior_picks(decision)
    picks = {arm: chosen(common[arm], prior[arm]) for arm in ARMS}
    fixed = list(picks["60-session"])
    saved_at = pd.Timestamp.now(tz="UTC")
    with TemporaryDirectory(dir=RESULT) as temporary:
        stage = Path(temporary) / "forecast"
        stage.mkdir()
        common.to_csv(stage / "stock_forecasts.csv")
        write_json(
            stage / "forecast.json",
            {
                "decision_date": decision.date().isoformat(),
                "target_date": target.date().isoformat(),
                "frozen_at_utc": saved_at.isoformat(),
                "classification": "prospective"
                if saved_at < cal.session_open(cal.session_offset(decision, 1))
                else "retrospective",
                "price_snapshot": str(price_path.resolve()),
                "price_snapshot_sha256": source["sha256"],
                "snapshot_retrieved_at_utc": source["retrieved_at_utc"],
                "roster_sha256": source["roster_sha256"],
                "checkpoints": checkpoints,
                "actual_context_sessions": context_counts,
                "common_candidates": len(common),
                "excluded_from_common_candidates": sorted(
                    set(symbols) - set(common.index)
                ),
                "prior_selection_date": prior_date,
                "picks": {arm: list(picks[arm]) for arm in ARMS},
                "primary_basket": fixed,
                "primary_basket_forecast": {
                    arm: float(common.loc[fixed, arm].mean()) for arm in ARMS
                },
                "own_basket_forecast": {
                    arm: float(common.loc[list(picks[arm]), arm].mean()) for arm in ARMS
                },
                "forecasts_sha256": sha256(stage / "stock_forecasts.csv"),
            },
        )
        stage.rename(output)
    print((output / "forecast.json").read_text())


def settle(decision, price_path):
    output = RESULT / decision.date().isoformat()
    frozen = json.loads((output / "forecast.json").read_text())
    forecasts = output / "stock_forecasts.csv"
    if sha256(forecasts) != frozen["forecasts_sha256"]:
        raise ValueError("Frozen forecasts were modified")
    cal = xcals.get_calendar("XNYS")
    target = pd.Timestamp(frozen["target_date"])
    if pd.Timestamp.now(tz="UTC") < cal.session_close(target):
        raise ValueError(f"Outcome pending until the {target.date()} US close")
    destination = output / "settlement.json"
    if destination.exists():
        raise FileExistsError(f"Settlement already recorded: {destination}")
    prices, source = snapshot(price_path)
    if pd.Timestamp(source["through"]) < target:
        raise ValueError(f"Snapshot has not reached the {target.date()} outcome")
    score = pd.read_csv(forecasts).set_index("symbol")
    symbols = [*score.index, "SPY"]
    days = cal.sessions_in_range(decision, target)
    closes = price_panel(prices.loc[prices.date.isin(days)], days, symbols)
    moves = closes.div(closes.shift()) - 1
    valid = closes.notna().all(axis=0) & ~moves.abs().ge(LARGE_DAILY_MOVE).any(axis=0)
    actual = (closes.loc[target] / closes.loc[decision] - 1).where(valid)
    if not np.isfinite(actual["SPY"]):
        raise ValueError("SPY outcome is missing")
    score["actual"] = actual.reindex(score.index)
    score = score.assign(Zero=0.0)
    fixed = score.loc[frozen["primary_basket"]]
    actual_basket = fixed.actual.mean(skipna=False)
    observed = score.dropna(subset=["actual"])
    stats = {}
    for arm in (*ARMS, "Zero"):
        error = observed[arm] - observed.actual
        own = score.loc[frozen["picks"][arm]] if arm in ARMS else fixed
        own_actual = own.actual.mean(skipna=False)
        arm_stats = {
            "primary_basket_forecast": fixed[arm].mean(),
            "primary_basket_mae_pp": 100 * abs(fixed[arm].mean() - actual_basket),
            "stock_mae_pp": 100 * error.abs().mean(),
            "stock_rmse_pp": 100 * np.sqrt(error.pow(2).mean()),
            "rank_ic": observed[arm].rank().corr(observed.actual.rank())
            if arm in ARMS
            else None,
            "own_basket_actual": own_actual,
            "own_basket_forecast": own[arm].mean(),
        }
        stats[arm] = {
            metric: float(value) if value is not None and np.isfinite(value) else None
            for metric, value in arm_stats.items()
        }
    result = {
        "classification": frozen["classification"],
        "decision_date": frozen["decision_date"],
        "target_date": frozen["target_date"],
        "matched_stock_outcomes": len(observed),
        "missing_stock_outcomes": score.index[score.actual.isna()].tolist(),
        "primary_basket_actual": float(actual_basket)
        if np.isfinite(actual_basket)
        else None,
        "spy_actual": float(actual["SPY"]),
        "score_by_arm": stats,
        "outcome_snapshot_sha256": source["sha256"],
        "frozen_forecasts_sha256": frozen["forecasts_sha256"],
    }
    score.to_csv(output / "stock_outcomes.csv")
    write_json(destination, result)
    print(destination.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "settle"))
    parser.add_argument("--date", type=pd.Timestamp, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    action = freeze if args.action == "freeze" else settle
    action(args.date, args.prices)
