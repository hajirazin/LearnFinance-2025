"""Stress test frozen H28, H1, and H5 with a one-session sentiment shock."""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from brain_api.core.sticky_selection import select_with_rank_band
from build_friday_dataset import (
    CONFIDENCE_RECENCY_TAU_HOURS,
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
)
from compare_patchtst import sha256, write_json
from compare_v1_forecast_accuracy import paired_interval
from patchtst_replay_models import model_output, simple_returns
from run_v1_holdout import OUTPUT as HOLDOUT_DATA
from train_v1_h28 import RESULT
from train_v1_hypotheses import model_factory
from v1_hypothesis_data import (
    DATA,
    OUTPUT,
    ROSTER,
    SEVEN,
    daily_features,
    decision_dates,
    session_data,
    windows,
)

NEWS = SEVEN.index("news_sentiment")
SHOCK = -0.50
NEAR_UNIVERSAL = 0.95
FORECAST_EVIDENCE = OUTPUT.parent / "forecast_accuracy"
CHECKPOINTS = {
    "H1": (OUTPUT, "h1_features"),
    "H5": (OUTPUT, "h5_features_training_filter"),
    "H28": (RESULT, "h28"),
}


def load_model(name, x, device):
    directory, checkpoint = CHECKPOINTS[name]
    meta = pd.read_json(directory / f"{checkpoint}.json", typ="series")
    weights = directory / f"{checkpoint}.pt"
    if sha256(weights) != meta["weights_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {name}")
    model = model_factory(x, daily=True)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, x, mask, device, *, lag=None, override=None):
    views = []
    if lag is not None:
        if not mask[:, -1 - lag, NEWS].all():
            raise ValueError("Stress case requires an observed feature")
        override = np.broadcast_to(override, (len(x),))
    for start in range(0, len(x), 512):
        end = start + 512
        inp = torch.tensor(x[start:end], device=device)
        if lag is not None:
            inp[:, -1 - lag, NEWS] = torch.tensor(
                override[start:end], device=device, dtype=inp.dtype
            )
        observed = torch.as_tensor(mask[start:end], device=device)
        views.append(
            simple_returns(model_output(model, inp, observed, daily=True), daily=True)
            .cpu()
            .numpy()
        )
    scores = np.concatenate(views)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite checkpoint score")
    return scores


def article_weight(index, calendar, original_sentiment):
    """Get the original denominator for the archived seven-day sentiment feature."""
    windows_df = index.to_frame(index=False)
    dates = index.get_level_values("friday").unique()
    cutoff = pd.Series([calendar.session_close(day) for day in dates], index=dates)
    windows_df["cutoff"] = windows_df.friday.map(cutoff)
    windows_df["window_start"] = windows_df.cutoff.dt.tz_convert(
        "America/New_York"
    ) - pd.DateOffset(days=7)
    with duckdb.connect(
        str(HOLDOUT_DATA / "news/news_v1.duckdb"), read_only=True
    ) as db:
        db.register("decisions", windows_df)
        existing = db.execute(
            """
            WITH revisions AS (
                SELECT d.friday, d.symbol, e.sentiment_score,
                       e.confidence * exp((epoch(e.created_at) - epoch(d.cutoff)) /
                                          (3600 * $tau)) AS weight
                FROM decisions d JOIN news_events e
                ON d.symbol = e.symbol AND e.created_at > d.window_start
                    AND e.created_at <= d.cutoff AND e.updated_at <= d.cutoff
                WHERE e.provider = $provider AND e.schema_version = $schema
                    AND e.sentiment_model = $model AND e.sentiment_model_revision = $revision
                QUALIFY row_number() OVER (
                    PARTITION BY d.friday, d.symbol, e.provider_article_id
                    ORDER BY e.updated_at DESC
                ) = 1
            )
            SELECT friday, symbol, sum(weight) AS weight,
                   sum(weight * sentiment_score) AS numerator
            FROM revisions GROUP BY friday, symbol
            """,
            {
                "tau": CONFIDENCE_RECENCY_TAU_HOURS,
                "provider": NEWS_PROVIDER,
                "schema": NEWS_SCHEMA_VERSION,
                "model": NEWS_SENTIMENT_MODEL,
                "revision": NEWS_SENTIMENT_REVISION,
            },
        ).df()
    existing = existing.set_index(["friday", "symbol"]).reindex(index, fill_value=0)
    reconstructed = (existing.numerator / existing.weight).fillna(0)
    np.testing.assert_allclose(reconstructed, original_sentiment, atol=1e-6, rtol=1e-5)
    return existing.weight.to_numpy()


def select(values, symbols, previous):
    scores = dict(zip(symbols, values, strict=True))
    return set(select_with_rank_band(scores, previous, 15, 20).selected)


def replaced_rank(background, override, positions):
    """Rank each candidate against other stocks at their archived scores."""
    ordinal = np.arange(len(background))
    above = background[None, :] > override[:, None]
    ties_first = (background[None, :] == override[:, None]) & (
        ordinal[None, :] < positions[:, None]
    )
    above[np.arange(len(override)), positions] = False
    return 1 + (above | ties_first).sum(axis=1)


def individually_selected(background, override, rank, positions, symbols, previous):
    selected = np.zeros(len(override), dtype=bool)
    for i in np.flatnonzero(rank <= 20):
        column = positions[i]
        values = background.copy()
        values[column] = override[i]
        selected[i] = symbols[column] in select(values, symbols, previous)
    return selected


def compare_selection(frame, baseline, previous):
    # Symbols are explicitly alphabetical, matching the production rank tiebreak.
    frame = frame.sort_index(level="symbol").copy()
    baseline = (
        baseline.droplevel("friday").sort_index()
        if isinstance(baseline.index, pd.MultiIndex)
        else baseline.sort_index()
    )
    symbols = baseline.index.to_numpy()
    background = baseline.to_numpy()
    positions = baseline.index.get_indexer(frame.index.get_level_values("symbol"))
    if (positions < 0).any():
        raise ValueError("Stress symbols missing from the baseline universe")
    for label in ("neutral", "negative"):
        forecast = frame[f"{label}_score"].to_numpy()
        rank = replaced_rank(background, forecast, positions)
        frame[f"{label}_rank"] = rank
        frame[f"{label}_selected"] = individually_selected(
            background, forecast, rank, positions, symbols, previous
        )
    frame["original_selected"] = frame.index.get_level_values("symbol").isin(
        select(background, symbols, previous)
    )
    frame["previously_held"] = frame.index.get_level_values("symbol").isin(
        previous or ()
    )
    frame["removed"] = frame.neutral_selected & ~frame.negative_selected
    frame["entered"] = ~frame.neutral_selected & frame.negative_selected
    return frame


def summarize_paired(trials):
    grouped = trials.groupby(["model", "scenario", "friday"], sort=True)
    weekly = grouped.agg(
        candidates=("removed", "size"),
        neutral_selected=("neutral_selected", "sum"),
        negative_selected=("negative_selected", "sum"),
        removed=("removed", "sum"),
        entered=("entered", "sum"),
        mean_score_change_pp=("score_change_pp", "mean"),
        mean_rank_change=("rank_change", "mean"),
    )
    weekly["removed_fraction"] = weekly.removed.div(
        weekly.neutral_selected.where(weekly.neutral_selected.gt(0))
    )
    weekly["selection_change_pp"] = (
        (weekly.negative_selected - weekly.neutral_selected) / weekly.candidates * 100
    )
    result = {}
    for (model, scenario), group in weekly.groupby(level=["model", "scenario"]):
        removal = group.removed_fraction.dropna()
        # Lower-tail tests at zero signed change and at a 95% removal threshold.
        delta = paired_interval(group.selection_change_pp)
        near = paired_interval(removal - NEAR_UNIVERSAL)
        subset = trials.loc[trials.model.eq(model) & trials.scenario.eq(scenario)]
        result.setdefault(model, {})[scenario] = {
            "weeks": len(group),
            "candidates": int(group.candidates.sum()),
            "neutral_selected": int(group.neutral_selected.sum()),
            "negative_selected": int(group.negative_selected.sum()),
            "removed": int(group.removed.sum()),
            "survivors": int(group.neutral_selected.sum() - group.removed.sum()),
            "entered": int(group.entered.sum()),
            "conditional_exit_pct_week_mean": 100 * removal.mean(),
            "conditional_exit_ci95_pct": [
                (near[1] + NEAR_UNIVERSAL) * 100,
                (near[2] + NEAR_UNIVERSAL) * 100,
            ],
            "p_against_at_least_95pct_exit": near[3],
            "net_selection_change_pp": delta[0],
            "net_selection_change_ci95_pp": [delta[1], delta[2]],
            "p_for_net_selection_decrease": delta[3],
            "mean_score_change_pp": float(group.mean_score_change_pp.mean()),
            "mean_rank_change": group.mean_rank_change.mean(),
            "rank_worsened_pct": 100 * subset.rank_change.gt(0).mean(),
            "rank_improved_pct": 100 * subset.rank_change.lt(0).mean(),
        }
    return weekly, result


def forecast_comparison(baselines):
    """Use the identical 33-week stocks and V1 baskets from the H1–H27 report."""
    with duckdb.connect() as db:
        panel = db.read_parquet(str(FORECAST_EVIDENCE / "stock_forecasts.parquet")).df()
    panel = panel.set_index(["friday", "symbol"])
    panel = panel.join(baselines[["H28"]], how="left").dropna(subset=["actual"])
    if not panel.H28.notna().all():
        raise ValueError("H28 forecasts are missing on the original common sample")
    fixed = pd.read_csv(
        FORECAST_EVIDENCE / "weekly_forecasts.csv", parse_dates=["friday"]
    )
    fixed = fixed.loc[fixed.method.eq("Native") & fixed.model.eq("V1")].set_index(
        "friday"
    )
    records = []
    for date, week in panel.groupby(level="friday", sort=True):
        picks = fixed.loc[date, "fixed_symbols"].split(",")
        basket = week.droplevel("friday").loc[picks]
        for name in ["V1", "H1", "H5", "H28", "Zero"]:
            stock_error = week[name] - week.actual if name != "Zero" else -week.actual
            basket_error = (
                basket[name].mean() - basket.actual.mean()
                if name != "Zero"
                else -basket.actual.mean()
            )
            records.append(
                {
                    "friday": date,
                    "model": name,
                    "observed_stocks": len(week),
                    "stock_mae_pp": 100 * stock_error.abs().mean(),
                    "basket_error_pp": 100 * basket_error,
                    "basket_absolute_error_pp": 100 * abs(basket_error),
                    "basket_actual": basket.actual.mean(),
                    "basket_forecast": basket.actual.mean() + basket_error,
                    "spy_close_return": fixed.loc[date, "spy"],
                }
            )
    records = pd.DataFrame(records)
    records.to_csv(RESULT / "forecast_weekly.csv", index=False)
    means = records.groupby("model")[
        ["stock_mae_pp", "basket_absolute_error_pp"]
    ].mean()
    paired = records.pivot(
        index="friday", columns="model", values="basket_absolute_error_pp"
    )
    return {
        "weeks": len(fixed),
        "stock_outcomes": len(panel),
        "mean_errors_pp": means.to_dict(orient="index"),
        "h28_minus_h1_mae": paired_interval(paired.H28 - paired.H1)[:3],
        "h28_minus_h5_mae": paired_interval(paired.H28 - paired.H5)[:3],
        "h28_minus_zero_mae": paired_interval(paired.H28 - paired.Zero)[:3],
    }


def run(models, device):
    calendar, _, close, volume, logs, _ = session_data()
    dates = decision_dates(close.index, evaluation=True)
    features = daily_features(calendar, close, volume, logs, evaluation=True)
    index, x, mask, _ = windows(logs, features, dates, use_features=True, year=False)
    weight = article_weight(index, calendar, x[:, -1, NEWS])
    # Synthetic articles both have confidence 1 and are added at the cutoff.
    neutral_article = x[:, -1, NEWS] * weight / (weight + 1)
    negative_article = neutral_article + SHOCK / (weight + 1)
    scenarios = {
        "friday_channel": (0, 0.0, SHOCK),
        "friday_article": (0, neutral_article, negative_article),
        "previous_session_channel": (1, 0.0, SHOCK),
    }
    baselines = pd.DataFrame(index=index)
    trials = []
    for model_name in models:
        print(f"Scoring {model_name}", flush=True)
        model = load_model(model_name, x, device)
        base = pd.Series(predict(model, x, mask, device), index=index)
        baselines[model_name] = base
        prior_by_day = {}
        previous = None
        for date, week in base.groupby(level="friday", sort=True):
            prior_by_day[date] = previous
            previous = select(
                week.to_numpy(), week.index.get_level_values("symbol"), previous
            )
        for scenario, (lag, neutral, negative) in scenarios.items():
            eligible = mask[:, -1 - lag, NEWS]
            column_neutral = np.broadcast_to(neutral, (len(x),))[eligible]
            column_negative = np.broadcast_to(negative, (len(x),))[eligible]
            pairs = pd.DataFrame(
                {
                    "neutral_score": predict(
                        model,
                        x[eligible],
                        mask[eligible],
                        device,
                        lag=lag,
                        override=column_neutral,
                    ),
                    "negative_score": predict(
                        model,
                        x[eligible],
                        mask[eligible],
                        device,
                        lag=lag,
                        override=column_negative,
                    ),
                    "original_sentiment": x[eligible, -1 - lag, NEWS],
                    "neutral_sentiment": column_neutral,
                    "negative_sentiment": column_negative,
                },
                index=index[eligible],
            )
            for date, frame in pairs.groupby(level="friday", sort=True):
                trials.append(
                    compare_selection(frame, base.loc[date], prior_by_day[date]).assign(
                        model=model_name, scenario=scenario
                    )
                )
    trials = (
        pd.concat(trials)
        .reset_index()
        .assign(
            score_change_pp=lambda d: 100 * (d.negative_score - d.neutral_score),
            rank_change=lambda d: d.negative_rank - d.neutral_rank,
        )
    )
    weekly, summary = summarize_paired(trials)
    suffix = "" if "H28" in models else "_reference"
    with duckdb.connect() as db:
        db.from_df(trials).write_parquet(
            str(RESULT / f"sentiment_trials{suffix}.parquet")
        )
    weekly.to_csv(RESULT / f"sentiment_weekly{suffix}.csv")
    baselines.to_csv(RESULT / f"scores{suffix}.csv")
    result = {
        "first_decision": dates[0].date().isoformat(),
        "last_decision": dates[-1].date().isoformat(),
        "shock_sentiment": SHOCK,
        "near_universal_threshold": NEAR_UNIVERSAL,
        "bootstrap": {"block_weeks": 4, "draws": 20_000, "seed": 42},
        "scoring_device": str(device),
        "checkpoints_sha256": {
            name: sha256(directory / f"{stem}.pt")
            for name, (directory, stem) in CHECKPOINTS.items()
            if name in models
        },
        "by_model": summary,
    }
    if "H28" in models:
        result["normal_forecast_accuracy"] = forecast_comparison(baselines)
        repo = Path(__file__).resolve().parents[2]
        sources = [
            ROSTER,
            DATA / "daily_ohlcv_2016_2025.csv",
            DATA / "friday/friday_dataset.parquet",
            DATA / "friday/flagged_daily_returns.csv",
            DATA / "news/news_v1.duckdb",
            HOLDOUT_DATA / "prices_2026.csv",
            HOLDOUT_DATA / "news/news_v1.duckdb",
            FORECAST_EVIDENCE / "stock_forecasts.parquet",
            FORECAST_EVIDENCE / "weekly_forecasts.csv",
            Path(__file__).resolve(),
            Path(__file__).with_name("train_v1_h28.py"),
        ]
        result["sources_sha256"] = {
            str(path.relative_to(repo)): sha256(path) for path in sources
        }
    write_json(RESULT / f"summary{suffix}.json", result)
    print(f"Saved {len(trials):,} paired trials to {RESULT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    RESULT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    run(
        ("H1", "H5") if args.reference_only else ("H1", "H5", "H28"),
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
