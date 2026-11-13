"""Replay Ridge and two PatchTST screens on matching 2025 Fridays."""

import argparse
import hashlib
import json
import platform
import subprocess
from itertools import combinations, pairwise
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import transformers
from build_friday_dataset import DATA, EVIDENCE, OUTPUT
from compare_feature_sets import (
    RIDGE_ALPHA,
    SEVEN,
    TOP_K,
    TRAIN_YEARS,
    evaluate,
    forecast_accuracy,
    interval,
    plot_basket_forecasts,
)
from patchtst_replay_data import WEEKLY_CONTEXT, YEAR_DAYS, load_replay, monthly_folds
from patchtst_replay_models import SEED, fit_patchtst
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARM_TITLES = {
    "ridge": "Seven-feature Ridge",
    "weekly_patchtst": "Seven-feature PatchTST",
    "daily_patchtst": "Close-only PatchTST, 365 calendar days",
}


def write_json(path, value):
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    staged.replace(path)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fold_predictions(data, fold, *, device, epochs, batch_size, output):
    cutoff, historical, train, validation, score = fold
    y = data.labels["forward_return"].to_numpy()
    latest = data.weekly_x[:, -1, :]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    ridge.fit(latest[historical], y[historical])
    scores = pd.DataFrame(
        {"ridge": ridge.predict(latest[score])}, index=data.labels.index[score]
    )
    metadata = {
        "refit": cutoff,
        "train_start": cutoff - pd.DateOffset(years=TRAIN_YEARS),
        "last_observed_label": data.labels.loc[historical, "next_friday"].max(),
        "last_neural_train_label": data.labels.loc[train, "next_friday"].max(),
        "first_validation_friday": data.labels.index[validation]
        .get_level_values("friday")
        .min(),
        "ridge_train_rows": int(historical.sum()),
        "neural_train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "scored_rows": int(score.sum()),
        "ridge_feature_means": ridge.named_steps["standardscaler"].mean_.tolist(),
        "ridge_feature_scales": ridge.named_steps["standardscaler"].scale_.tolist(),
        "ridge_coefficients": ridge.named_steps["ridge"].coef_.tolist(),
        "ridge_intercept": ridge.named_steps["ridge"].intercept_,
    }
    for name, x, targets, mask, daily in (
        (
            "weekly_patchtst",
            data.weekly_x,
            y[:, None],
            np.zeros(len(y), dtype=bool),
            False,
        ),
        ("daily_patchtst", data.daily_x, data.daily_y, data.daily_mask, True),
    ):
        prediction, state, details = fit_patchtst(
            x,
            targets,
            mask,
            train,
            validation,
            score,
            data.labels,
            daily=daily,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )
        scores[name] = prediction
        weights = output / f"{name}.pt"
        torch.save(state, weights)
        metadata[name] = {**details, "weights_sha256": sha256(weights)}
    return scores.join(data.labels), metadata


def save_results(predictions, metadata, output):
    metrics, picks, stocks = evaluate(predictions, ARM_TITLES)
    paired = metrics.loc[metrics["paired"]]
    if len(paired) < 4:
        raise ValueError("Insufficient matched realized weeks")
    models = {}
    for name in ARM_TITLES:
        actual = paired[f"{name}_top15_return"]
        stock = stocks.loc[stocks["model"].eq(name) & stocks["paired"]]
        by_week = (
            picks.loc[picks["model"].eq(name) & picks["friday"].isin(paired.index)]
            .groupby("friday")["symbol"]
            .agg(set)
            .tolist()
        )
        models[name] = {
            "basket": forecast_accuracy(
                paired[f"{name}_top15_predicted_return"], actual
            ),
            "stock": forecast_accuracy(
                stock["predicted_return"], stock["forward_return"]
            ),
            "mean_basket_excess": (actual - paired["universe_return"]).mean(),
            "mean_rank_ic": paired[f"{name}_rank_ic"].mean(),
            "fraction_replaced": np.mean(
                [len(new - old) / TOP_K for old, new in pairwise(by_week)]
            ),
        }
    comparisons = {}
    for left, right in combinations(ARM_TITLES, 2):
        difference = paired[f"{left}_top15_return"] - paired[f"{right}_top15_return"]
        comparisons[f"{left}_minus_{right}"] = {
            "mean_return_difference": difference.mean(),
            "ci95": interval(difference),
            "weeks_left_won": int(difference.gt(0).sum()),
            "weeks_right_won": int(difference.lt(0).sum()),
        }
    metadata.update(
        scored_weeks=len(metrics),
        matched_weeks=len(paired),
        stock_forecast_rows=len(stocks),
        matched_fridays=paired.index.strftime("%Y-%m-%d").tolist(),
        omitted_fridays=metrics.index[~metrics["paired"]].strftime("%Y-%m-%d").tolist(),
        median_candidates=float(metrics["scorable_stocks"].median()),
        mean_matched_universe_return=paired["universe_return"].mean(),
        models=models,
        comparisons=comparisons,
    )
    metrics.to_csv(output / "weekly_results.csv")
    picks.to_csv(output / "top_15_picks.csv", index=False)
    with duckdb.connect() as db:
        db.from_df(stocks).write_parquet(str(output / "stock_forecasts.parquet"))
    plot_basket_forecasts(metrics, ARM_TITLES, output, titles=ARM_TITLES)
    write_json(output / "summary.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fit only the first month and skip final statistics.",
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch size must be positive")
    device = torch.device(args.device)
    torch.empty(0, device=device)
    torch.set_num_threads(4)
    output = OUTPUT / (
        "patchtst_comparison_smoke" if args.smoke else "patchtst_comparison"
    )
    output.mkdir(exist_ok=True)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("patchtst_replay_data.py"),
        Path(__file__).with_name("patchtst_replay_models.py"),
        Path(__file__).with_name("compare_feature_sets.py"),
        Path(__file__).with_name("build_friday_dataset.py"),
    ]
    inputs = [
        OUTPUT / "friday_dataset.parquet",
        DATA / "daily_ohlcv_2016_2025.csv",
        EVIDENCE,
    ]
    metadata = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sha256": {path.name: sha256(path) for path in [*source_files, *inputs]},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "ridge_alpha": RIDGE_ALPHA,
        "top_k": TOP_K,
        "rolling_training_years": TRAIN_YEARS,
        "weekly_features": list(SEVEN),
        "weekly_context_fridays": WEEKLY_CONTEXT,
        "daily_context_calendar_days": YEAR_DAYS,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "seed": SEED,
        "refits": [],
    }
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    data = load_replay()
    data.coverage.loc[data.evaluation_dates].rename_axis("friday").to_csv(
        output / "coverage.csv"
    )
    predictions = []
    for fold in monthly_folds(data):
        cutoff, _, train, validation, score = fold
        print(
            f"{cutoff.date()} | {train.sum():,} train, {validation.sum():,} validation, {score.sum():,} score",
            flush=True,
        )
        fold_output = output / "refits" / cutoff.strftime("%Y-%m-%d")
        fold_output.mkdir(parents=True, exist_ok=True)
        fold_metadata = fold_output / "metadata.json"
        forecasts = fold_output / "forecasts.csv"
        previous = (
            json.loads(fold_metadata.read_text()) if fold_metadata.exists() else {}
        )
        if (
            previous.get("identity") == identity
            and forecasts.exists()
            and previous.get("forecasts_sha256") == sha256(forecasts)
        ):
            scores = pd.read_csv(
                forecasts, parse_dates=["friday", "next_friday"]
            ).set_index(["friday", "symbol"])
            details = previous
            print("  reused completed refit", flush=True)
        else:
            scores, details = fold_predictions(
                data,
                fold,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output=fold_output,
            )
            scores.to_csv(forecasts)
            details.update(identity=identity, forecasts_sha256=sha256(forecasts))
            write_json(fold_metadata, details)
        predictions.append(scores)
        metadata["refits"].append(details)
        if args.smoke:
            return
    save_results(pd.concat(predictions).sort_index(), metadata, output)
    print(f"Completed {len(predictions)} refits. Results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
