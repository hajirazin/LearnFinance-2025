"""Compare news-disabled and seven-feature PatchTST selections and forecasts."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from compare_patchtst import sha256
from compare_v1_forecast_accuracy import decorate, line, paired_interval, plt, set_style
from report_v1_xgboost_comparison import align_predictions, table, weekly_metrics
from run_v1_h28 import FORECAST_EVIDENCE
from sklearn.metrics import r2_score
from train_patchtst_news_ablation import OUTPUT
from train_v1_xgboost_comparison import OUTPUT as SHARED
from train_v1_xgboost_comparison import SEEDS, completed, load_split, write_json

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures/patchtst_news_ablation"
PRIMARY = ("Six active", "Seven active")


def arm_name(active, seed):
    return f"{active} active" if seed == 42 else f"{active} active seed {seed}"


def paired_test(delta, block=4):
    mean, low, high, lower = paired_interval(delta, block=block)
    _, _, _, upper = paired_interval(-delta, block=block)
    return {
        "block_observed_weeks": block,
        "mean_six_minus_seven_pp": float(mean),
        "ci95_low_pp": float(low),
        "ci95_high_pp": float(high),
        "two_sided_p": min(1.0, 2 * min(lower, upper)),
    }


def load_forecasts():
    _, _, index = load_split("test")
    forecasts = {}
    for seed in SEEDS:
        for active, directory in (("Six", OUTPUT), ("Seven", SHARED)):
            name = f"patchtst_{seed}"
            if not completed(name, output=directory):
                raise ValueError(f"Incomplete {active} reference for seed {seed}")
            metadata = json.loads((directory / f"{name}.json").read_text())
            if metadata.get("disable_news", False) != (active == "Six"):
                raise ValueError(
                    f"Incorrect sentiment policy for {active}, seed {seed}"
                )
            values = np.load(directory / f"{name}_predictions.npy")
            if len(values) != len(index) or not np.isfinite(values).all():
                raise ValueError("Forecasts do not cover the shared input rows")
            forecasts[arm_name(active, seed)] = values
    scores = pd.DataFrame(forecasts, index=index).assign(Zero=0.0)
    with duckdb.connect() as db:
        evidence = db.read_parquet(
            str(FORECAST_EVIDENCE / "stock_forecasts.parquet")
        ).df()
    evidence.friday = evidence.friday.dt.as_unit("ns")
    actuals = evidence.set_index(["friday", "symbol"])[["actual"]].dropna()
    panel = align_predictions(actuals, scores)
    if len(panel) != 13712 or panel.index.unique("friday").size != 33:
        raise ValueError("Outcome panel drifted from the seven-feature experiment")
    fixed = pd.read_csv(
        FORECAST_EVIDENCE / "weekly_forecasts.csv", parse_dates=["friday"]
    )
    fixed = fixed.loc[fixed.method.eq("Native") & fixed.model.eq("V1")].set_index(
        "friday"
    )
    return scores, panel, fixed


def summarize(panel, weekly):
    metrics = []
    for name, frame in weekly.groupby("model", sort=False):
        metrics.append(
            {
                "Arm": name,
                "Stock RMSE (pp)": np.sqrt(frame.stock_mse_pp2.mean()),
                "Stock MAE (pp)": frame.stock_mae_pp.mean(),
                "Stock R²": r2_score(panel.actual, panel[name]),
                "Mean rank IC": frame.rank_ic.mean(),
                "Fixed-basket MAE (pp)": frame.basket_mae_pp.mean(),
            }
        )
    selections, tests, omissions = [], [], {}
    for seed in SEEDS:
        names = [arm_name(active, seed) for active in ("Six", "Seven")]
        returns = weekly.loc[weekly.model.isin(names)].pivot(
            index="friday", columns="model", values="own_actual_gross"
        )
        common = returns.notna().all(axis=1)
        omissions[str(seed)] = [str(day.date()) for day in returns.index[~common]]
        returns = returns.loc[common]
        spy = (
            weekly.loc[weekly.model.eq(names[1])]
            .set_index("friday")
            .spy.reindex(returns.index)
        )
        for name in names:
            excess = 100 * (returns[name] - spy)
            mean, low, high, _ = paired_interval(excess)
            selections.append(
                {
                    "Arm": name,
                    "Paired weeks": len(returns),
                    "Mean return (%)": 100 * returns[name].mean(),
                    "Mean excess over SPY (pp)": mean,
                    "Weeks ahead of SPY": int((excess > 0).sum()),
                    "Excess CI low (pp)": low,
                    "Excess CI high (pp)": high,
                }
            )
        delta = 100 * (returns[names[0]] - returns[names[1]])
        for block in (2, 4, 8):
            tests.append(
                {
                    "seed": seed,
                    "paired_weeks": len(delta),
                    "six_ahead_of_seven_weeks": int((delta > 0).sum()),
                    **paired_test(delta, block),
                }
            )
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(selections),
        pd.DataFrame(tests),
        omissions,
    )


def overlap_table(selections):
    days = sorted({day for day, _ in selections})
    rows = []
    for seed in SEEDS:
        previous = {"Six": set(), "Seven": set()}
        for position, day in enumerate(days):
            names = {
                active: set(selections[day, arm_name(active, seed)])
                for active in ("Six", "Seven")
            }
            rows.append(
                {
                    "friday": day,
                    "seed": seed,
                    "common_selected_names": len(names["Six"] & names["Seven"]),
                    "six_replaced": len(names["Six"] - previous["Six"])
                    if position
                    else np.nan,
                    "seven_replaced": len(names["Seven"] - previous["Seven"])
                    if position
                    else np.nan,
                }
            )
            previous = names
    return pd.DataFrame(rows)


def verify_seven_reference(weekly):
    reference = pd.read_csv(SHARED / "weekly_forecasts.csv", parse_dates=["friday"])
    reference = (
        reference.loc[reference.model.eq("PatchTST")].set_index("friday").sort_index()
    )
    current = (
        weekly.loc[weekly.model.eq("Seven active")].set_index("friday").sort_index()
    )
    for field in ("own_actual_gross", "stock_mse_pp2", "basket_forecast", "spy"):
        np.testing.assert_allclose(current[field], reference[field], rtol=0, atol=1e-12)


def plot(weekly):
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_style()
    frames = {
        name: weekly.loc[weekly.model.eq(name)].set_index("friday").sort_index()
        for name in (*PRIMARY, "Zero")
    }
    six, seven = (frames[name] for name in PRIMARY)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        "PatchTST · six versus seven informative features",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.943,
        "News disabled versus news supplied · identical architecture and rows · primary seed 42",
    )
    colors = {"Six active": "#126f73", "Seven active": "#7860aa", "Zero": "#748396"}
    for name in PRIMARY:
        line(axes[0], frames[name].own_actual_gross, name, colors[name])
    line(axes[0], seven.spy, "SPY", "#c37224", style="--")
    decorate(axes[0], "Own equal-weight sticky 15 · gross weekly returns", "Return (%)")
    line(
        axes[1],
        six.own_actual_gross - seven.own_actual_gross,
        "Six minus seven",
        "#126f73",
    )
    decorate(axes[1], "Paired gross selection-return difference", "Difference (pp)")
    for name in (*PRIMARY, "Zero"):
        line(axes[2], np.sqrt(frames[name].stock_mse_pp2), name, colors[name], scale=1)
    decorate(axes[2], "Individual-stock forecast error", "RMSE (pp)", zero=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.06, hspace=0.62)
    fig.savefig(FIGURES / "six_vs_seven.png", dpi=150)
    plt.close(fig)


def update_report(metrics, selection, tests, overlap):
    path = ROOT / "V1_PATCHTST_NEWS_ABLATION.md"
    protocol = path.read_text().split("\n## Results\n")[0]
    primary_test = tests.loc[tests.seed.eq(42) & tests.block_observed_weeks.eq(4)].iloc[
        0
    ]
    primary_metrics = metrics.loc[metrics.Arm.isin([*PRIMARY, "Zero"])]
    primary_selection = selection.loc[selection.Arm.isin(PRIMARY)]
    seed_tests = tests.loc[
        tests.block_observed_weeks.eq(4),
        [
            "seed",
            "paired_weeks",
            "mean_six_minus_seven_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "two_sided_p",
        ],
    ]
    seed_tests = seed_tests.round(5)
    primary_overlap = overlap.loc[overlap.seed.eq(42)]
    path.write_text(
        protocol + "\n## Results\n\n" + table(primary_selection.round(4)) + "\n\n"
        f"The primary six-minus-seven mean weekly return difference was {primary_test.mean_six_minus_seven_pp:+.4f} pp "
        f"(95% block interval {primary_test.ci95_low_pp:+.4f} to {primary_test.ci95_high_pp:+.4f} pp, "
        f"two-sided p = {primary_test.two_sided_p:.4g}). "
        f"The comparison used {int(primary_test.paired_weeks)} paired observed weeks.\n\n"
        + table(primary_metrics.round(5))
        + "\n\n"
        f"The two primary selections shared an average of {primary_overlap.common_selected_names.mean():.2f} "
        "names out of 15 across all 40 decision Fridays. "
        f"Mean replacements after the first decision were {primary_overlap.six_replaced.mean():.2f} "
        f"for six and {primary_overlap.seven_replaced.mean():.2f} for seven.\n\n"
        "![News ablation returns and forecast errors](figures/patchtst_news_ablation/six_vs_seven.png)\n\n"
        "### Seed sensitivity\n\n"
        + table(selection.round(4))
        + "\n\n"
        + table(seed_tests)
        + "\n\n"
        "Only seed 42 is primary. The same historical outcomes have informed earlier experiments. "
        "The six-feature description means six informative inputs and a constant-zero sentiment slot, "
        "with the original seven-slot network retained. These results use gross adjusted-close returns.\n\n"
        "[Weekly comparisons](data/patchtst_news_ablation/weekly_forecasts.csv) · "
        "[Stock forecasts and actuals](data/patchtst_news_ablation/stock_forecasts.parquet) · "
        "[Primary test and block sensitivity](data/patchtst_news_ablation/paired_tests.csv) · "
        "[Selection overlap](data/patchtst_news_ablation/overlap.csv) · "
        "[Source manifest](data/patchtst_news_ablation/manifest.json) · "
        "[Summary](data/patchtst_news_ablation/summary.json)\n\n"
        "### Run commands\n\n```sh\n"
        "export PYTHONPATH=brain_api:scratch/AlphaAllocationV2\n"
        "python scratch/AlphaAllocationV1/train_patchtst_news_ablation.py --device mps\n"
        "python scratch/AlphaAllocationV1/report_patchtst_news_ablation.py\n```\n"
    )


def main():
    scores, panel, fixed = load_forecasts()
    arms = tuple(name for name in scores if name != "Zero")
    weekly, selections = weekly_metrics(scores, panel, fixed, arms=arms)
    verify_seven_reference(weekly)
    metrics, selection, tests, omissions = summarize(panel, weekly)
    overlap = overlap_table(selections)
    for filename, frame in (
        ("weekly_forecasts.csv", weekly),
        ("metrics.csv", metrics),
        ("selection.csv", selection),
        ("paired_tests.csv", tests),
        ("overlap.csv", overlap),
    ):
        frame.to_csv(OUTPUT / filename, index=False)
    pd.DataFrame(
        [
            {"friday": day, "Arm": name, "symbols": ",".join(names)}
            for (day, name), names in selections.items()
        ]
    ).to_csv(OUTPUT / "selections.csv", index=False)
    with duckdb.connect() as db:
        db.from_df(panel.reset_index()).write_parquet(
            str(OUTPUT / "stock_forecasts.parquet")
        )
        db.from_df(scores.reset_index()).write_parquet(
            str(OUTPUT / "all_forecasts.parquet")
        )
    write_json(
        OUTPUT / "summary.json",
        {
            "primary_seed": 42,
            "stock_outcomes": len(panel),
            "network_slots": 7,
            "six_informative_features": "Sentiment zeroed at training, validation, and inference",
            "metrics": json.loads(metrics.to_json(orient="records")),
            "selection": json.loads(selection.to_json(orient="records")),
            "paired_tests": json.loads(tests.to_json(orient="records")),
            "omitted_selection_dates_by_seed": omissions,
            "manifest_sha256": sha256(OUTPUT / "manifest.json"),
            "fit_metadata_sha256": {
                f"{active}_{seed}": sha256(directory / f"patchtst_{seed}.json")
                for active, directory in (("six", OUTPUT), ("seven", SHARED))
                for seed in SEEDS
            },
            "shared_dataset_sha256": sha256(SHARED / "dataset.json"),
            "reference_weekly_sha256": sha256(SHARED / "weekly_forecasts.csv"),
            "reporter_sha256": sha256(Path(__file__)),
        },
    )
    plot(weekly)
    update_report(metrics, selection, tests, overlap)
    print(metrics.to_string(index=False))
    print(selection.to_string(index=False))
    print(tests.loc[tests.block_observed_weeks.eq(4)].to_string(index=False))


if __name__ == "__main__":
    main()
