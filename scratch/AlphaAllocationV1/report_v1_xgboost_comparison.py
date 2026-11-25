"""Evaluate the frozen matched weekly forecasts without changing their fits."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from brain_api.core.sticky_selection import select_with_rank_band
from compare_patchtst import sha256
from compare_v1_forecast_accuracy import decorate, line, paired_interval, plt, set_style
from run_v1_h28 import FORECAST_EVIDENCE
from sklearn.metrics import r2_score
from train_v1_xgboost_comparison import OUTPUT, SEEDS, completed, load_split, write_json

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures/xgboost_comparison"
MAIN = ("Ridge", "XGBoost", "PatchTST")
BASELINES = ("Zero", "Training mean")


def align_predictions(actuals, scores):
    if not actuals.index.is_unique or not scores.index.is_unique:
        raise ValueError("Stock-week keys must be unique")
    panel = actuals.join(scores, how="left", validate="one_to_one")
    if not np.isfinite(panel.to_numpy()).all():
        raise ValueError("A required stock-week forecast or outcome is missing")
    return panel.sort_index()


def load_predictions():
    _, _, index = load_split("test")
    scores = {}
    for name in (
        "ridge",
        *(f"xgboost_{seed}" for seed in SEEDS),
        *(f"patchtst_{seed}" for seed in SEEDS),
    ):
        if not completed(name):
            raise ValueError(f"Fit is incomplete: {name}")
        values = np.load(OUTPUT / f"{name}_predictions.npy")
        if len(values) != len(index) or not np.isfinite(values).all():
            raise ValueError(f"Invalid frozen forecasts: {name}")
        scores[name] = values
    scores = pd.DataFrame(scores, index=index).rename(
        columns={"ridge": "Ridge", "xgboost_42": "XGBoost", "patchtst_42": "PatchTST"}
    )
    scores["Zero"] = 0.0
    scores["Training mean"] = json.loads((OUTPUT / "dataset.json").read_text())[
        "target_mean"
    ]
    with duckdb.connect() as db:
        evidence = db.read_parquet(
            str(FORECAST_EVIDENCE / "stock_forecasts.parquet")
        ).df()
    evidence.friday = evidence.friday.dt.as_unit("ns")
    actuals = evidence.set_index(["friday", "symbol"])[["actual"]].dropna()
    panel = align_predictions(actuals, scores)
    if len(panel) != 13712 or panel.index.unique("friday").size != 33:
        raise ValueError("Evaluation stock-weeks drifted from the frozen archive")
    fixed = pd.read_csv(
        FORECAST_EVIDENCE / "weekly_forecasts.csv", parse_dates=["friday"]
    )
    fixed = fixed.loc[fixed.method.eq("Native") & fixed.model.eq("V1")].set_index(
        "friday"
    )
    return scores, panel, fixed


def selected_baskets(scores, *, arms=MAIN):
    """Choose from all eligible forecasts before joining future observations."""
    previous = dict.fromkeys(arms)
    selections = {}
    for day, group in scores.groupby(level="friday", sort=True):
        group = group.droplevel("friday")
        for arm in arms:
            result = select_with_rank_band(group[arm].to_dict(), previous[arm], 15, 20)
            if len(result.selected) != 15:
                raise ValueError(f"Incomplete selection for {arm} on {day}")
            previous[arm] = set(result.selected)
            selections[day, arm] = list(result.selected)
    return selections


def weekly_metrics(scores, panel, fixed, *, arms=MAIN):
    selections = selected_baskets(scores, arms=arms)
    records = []
    for day, group in panel.groupby(level="friday", sort=True):
        group = group.droplevel("friday")
        names = fixed.loc[day, "fixed_symbols"].split(",")
        basket = group.loc[names]
        if len(basket) != 15 or not np.isclose(
            basket.actual.mean(), fixed.loc[day, "fixed_actual"], atol=1e-12, rtol=0
        ):
            raise ValueError("Fixed basket labels changed")
        for arm in scores:
            error = 100 * (group[arm] - group.actual)
            forecast = basket[arm].mean()
            own = selections.get((day, arm), [])
            observed = group.actual.reindex(own)
            own_actual = (
                observed.mean() if len(own) == 15 and observed.notna().all() else np.nan
            )
            records.append(
                {
                    "friday": day,
                    "model": arm,
                    "stocks": len(group),
                    "stock_mse_pp2": float((error**2).mean()),
                    "stock_mae_pp": float(error.abs().mean()),
                    "rank_ic": group[arm].rank().corr(group.actual.rank())
                    if group[arm].nunique() > 1
                    else np.nan,
                    "basket_forecast": forecast,
                    "basket_actual": basket.actual.mean(),
                    "basket_mse_pp2": float(
                        (100 * (forecast - basket.actual.mean())) ** 2
                    ),
                    "basket_mae_pp": float(100 * abs(forecast - basket.actual.mean())),
                    "spy": fixed.loc[day, "spy"],
                    "own_actual_gross": own_actual,
                    "own_excess_gross_pp": 100 * (own_actual - fixed.loc[day, "spy"]),
                    "own_missing_outcomes": int(observed.isna().sum())
                    if own
                    else np.nan,
                    "own_symbols": ",".join(own),
                }
            )
    return pd.DataFrame(records), selections


def summarize(panel, weekly):
    rows = []
    for arm, group in weekly.groupby("model", sort=False):
        rows.append(
            {
                "model": arm,
                "weeks": len(group),
                "stock_mse_pp2": group.stock_mse_pp2.mean(),
                "stock_rmse_pp": np.sqrt(group.stock_mse_pp2.mean()),
                "stock_mae_pp": group.stock_mae_pp.mean(),
                "stock_r_squared": r2_score(panel.actual, panel[arm]),
                "rank_ic": group.rank_ic.mean(),
                "basket_rmse_pp": np.sqrt(group.basket_mse_pp2.mean()),
                "basket_mae_pp": group.basket_mae_pp.mean(),
                "basket_r_squared": r2_score(
                    group.basket_actual, group.basket_forecast
                ),
            }
        )
    metrics = pd.DataFrame(rows).set_index("model")
    comparisons = []
    pivot = weekly.pivot(index="friday", columns="model", values="stock_mse_pp2")
    for reference in ("Ridge", "PatchTST"):
        delta = pivot.XGBoost - pivot[reference]
        for block in (2, 4, 8):
            mean, low, high, p = paired_interval(delta, block=block)
            comparisons.append(
                {
                    "reference": reference,
                    "block": block,
                    "mean_xgboost_minus_reference_pp2": mean,
                    "ci95_low_pp2": low,
                    "ci95_high_pp2": high,
                    "one_sided_lower_error_p": p,
                }
            )
    tests = pd.DataFrame(comparisons)
    pvalues = tests.loc[tests.block.eq(4), "one_sided_lower_error_p"].sort_values()
    adjusted = (pvalues * np.arange(len(pvalues), 0, -1)).cummax().clip(upper=1)
    tests["holm_primary_p"] = adjusted.reindex(tests.index)
    own = weekly.loc[weekly.model.isin(MAIN)].pivot(
        index="friday", columns="model", values="own_actual_gross"
    )
    common = own.notna().all(axis=1)
    own = own.loc[common]
    spy = (
        weekly.loc[weekly.model.eq("Ridge")].set_index("friday").spy.reindex(own.index)
    )
    selection = pd.DataFrame(
        {
            "common_weeks": len(own),
            "mean_weekly_return_pct": 100 * own.mean(),
            "mean_weekly_excess_pp": 100 * own.sub(spy, axis=0).mean(),
            "weeks_ahead_of_spy": own.gt(spy, axis=0).sum(),
        }
    )
    for arm in MAIN:
        _, low, high, _ = paired_interval(100 * (own[arm] - spy))
        selection.loc[arm, "excess_ci95_low_pp"] = low
        selection.loc[arm, "excess_ci95_high_pp"] = high
    return metrics, tests, selection


def plot(weekly):
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_style()
    frames = {
        arm: weekly.loc[weekly.model.eq(arm)].set_index("friday").sort_index()
        for arm in (*MAIN, *BASELINES)
    }
    colors = {
        "Ridge": "#7860aa",
        "XGBoost": "#126f73",
        "PatchTST": "#b65b3c",
        "Zero": "#748396",
        "Training mean": "#59623e",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.8), sharex=True)
    fig.suptitle(
        "Matched weekly forecasts · XGBoost, ridge, and PatchTST",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.932,
        "Seven channels · one-year histories · primary seed 42 · 33 archived Fridays",
    )
    line(axes[0], frames["Ridge"].basket_actual, "Actual fixed basket", "#172b4d")
    line(axes[0], frames["Ridge"].spy, "Realized SPY", "#c37224", style="--")
    for arm in MAIN:
        line(axes[0], frames[arm].basket_forecast, arm, colors[arm])
    decorate(
        axes[0], "Forecasts for the same 15 stocks and realized returns", "Return (%)"
    )
    for arm in (*MAIN, "Zero"):
        line(axes[1], np.sqrt(frames[arm].stock_mse_pp2), arm, colors[arm], scale=1)
    decorate(axes[1], "Weekly individual-stock RMSE", "Error (pp)", zero=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.08, hspace=0.55)
    fig.savefig(FIGURES / "forecast_accuracy.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.8), sharex=True)
    fig.suptitle(
        "Forecast error differences and selected-stock outcomes",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.932,
        "Paired weeks · lower error differences favor XGBoost · selection returns before costs",
    )
    for arm in ("Ridge", "PatchTST"):
        line(
            axes[0],
            frames["XGBoost"].stock_mse_pp2 - frames[arm].stock_mse_pp2,
            f"XGBoost minus {arm}",
            colors[arm],
            scale=1,
        )
    decorate(axes[0], "Difference in mean stock squared error", "Difference (pp²)")
    for arm in MAIN:
        line(axes[1], frames[arm].own_actual_gross - frames[arm].spy, arm, colors[arm])
    decorate(
        axes[1],
        "Own 15/20 sticky baskets · weekly gross excess over SPY",
        "Excess (pp)",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.08, hspace=0.55)
    fig.savefig(FIGURES / "paired_and_selection.png", dpi=150)
    plt.close(fig)


def table(frame):
    frame = frame.astype(object).where(pd.notna(frame), "—")
    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |",
            *(
                "| " + " | ".join(map(str, row)) + " |"
                for row in frame.itertuples(index=False, name=None)
            ),
        ]
    )


def spy_regressions(weekly):
    ridge = weekly.loc[weekly.model.eq("Ridge")].set_index("friday").sort_index()
    spy = ridge.spy.to_numpy()
    design = np.column_stack([np.ones(len(spy)), spy])
    rows = []
    for arm in (*MAIN, "Actual fixed basket"):
        values = (
            ridge.basket_actual.to_numpy()
            if arm == "Actual fixed basket"
            else weekly.loc[weekly.model.eq(arm)]
            .sort_values("friday")
            .basket_forecast.to_numpy()
        )
        intercept, beta = np.linalg.lstsq(design, values, rcond=None)[0]
        r_squared = r2_score(values, design @ np.array([intercept, beta]))
        rows.append(
            {
                "Series": arm,
                "Market R²": r_squared,
                "Adjusted R²": 1 - (1 - r_squared) * (len(spy) - 1) / (len(spy) - 2),
                "SPY beta": beta,
            }
        )
    return pd.DataFrame(rows)


def update_report(metrics, tests, selection, market):
    columns = [
        "stock_mse_pp2",
        "stock_rmse_pp",
        "stock_mae_pp",
        "stock_r_squared",
        "rank_ic",
        "basket_mae_pp",
    ]
    primary = metrics.loc[[*MAIN, *BASELINES], columns].round(4).reset_index()
    primary.columns = [
        "Model",
        "Stock MSE (pp²)",
        "Stock RMSE (pp)",
        "Stock MAE (pp)",
        "Stock R²",
        "Mean rank IC",
        "Fixed-basket MAE (pp)",
    ]
    tests_table = tests.loc[
        tests.block.eq(4),
        [
            "reference",
            "mean_xgboost_minus_reference_pp2",
            "ci95_low_pp2",
            "ci95_high_pp2",
            "holm_primary_p",
        ],
    ].round(4)
    tests_table.columns = [
        "XGBoost versus",
        "MSE difference (pp²)",
        "95% low",
        "95% high",
        "Holm p",
    ]
    tests_table["Holm p"] = tests.loc[tests.block.eq(4), "holm_primary_p"].map(
        lambda p: f"{p:.4g}"
    )
    seeds = (
        metrics.loc[
            [f"{arm}_{seed}" for arm in ("xgboost", "patchtst") for seed in (43, 44)],
            columns,
        ]
        .round(4)
        .reset_index()
    )
    seeds.columns = primary.columns
    selection_table = selection.loc[list(MAIN)].round(4).reset_index()
    selection_table.columns = [
        "Model",
        "Weeks",
        "Mean weekly return (%)",
        "Mean excess over SPY (pp)",
        "Weeks ahead",
        "95% excess low (pp)",
        "95% excess high (pp)",
    ]
    report = ROOT / "V1_XGBOOST_COMPARISON.md"
    protocol = report.read_text().split("\n## Results\n")[0]
    report.write_text(
        protocol + "\n## Results\n\n" + table(primary) + "\n\n"
        "Stock MSE and MAE give each observed week equal weight. Stock R² pools the "
        "13,712 observed stock returns. Primary XGBoost and PatchTST forecasts use seed 42.\n\n"
        + table(tests_table)
        + "\n\n"
        "Negative MSE differences favor XGBoost. The intervals and Holm p-values "
        "use the two predefined primary comparisons.\n\n"
        "![Matched forecasts and errors](figures/xgboost_comparison/forecast_accuracy.png)\n\n"
        "### Seed sensitivity\n\n" + table(seeds) + "\n\n"
        "### Own selections and SPY\n\n" + table(selection_table) + "\n\n"
        "Selections use the full common forecast universe before future return "
        "availability is examined. The table pairs all three strategies on weeks "
        "with all 15 selected outcomes observed. These are gross Friday-close "
        "diagnostics on sampled weeks. Execution costs and tradable portfolio NAV "
        "are outside this forecast comparison. Selection intervals use the same "
        "four-observed-week block bootstrap and are descriptive secondary intervals.\n\n"
        "### Descriptive SPY regression\n\n" + table(market.round(5)) + "\n\n"
        "Each row fits the basket forecast or actual return against realized SPY "
        "with an intercept on the same 33 dates. Adjusted R² uses one SPY slope. "
        "This describes co-movement within these observations.\n\n"
        "![Paired forecast differences and selected baskets](figures/xgboost_comparison/paired_and_selection.png)\n\n"
        "[Metrics](data/xgboost_comparison/metrics.csv) · "
        "[Weekly forecasts](data/xgboost_comparison/weekly_forecasts.csv) · "
        "[Stock forecasts and actuals](data/xgboost_comparison/stock_forecasts.parquet) · "
        "[Statistical tests and block sensitivity](data/xgboost_comparison/paired_tests.csv) · "
        "[Manifest and selected fits](data/xgboost_comparison/summary.json)\n\n"
        "### Run commands\n\n```sh\n"
        "export PYTHONPATH=brain_api:scratch/AlphaAllocationV2\n"
        "python scratch/AlphaAllocationV1/train_v1_xgboost_comparison.py --stage all --device mps\n"
        "python scratch/AlphaAllocationV1/report_v1_xgboost_comparison.py\n```\n"
    )


def main():
    scores, panel, fixed = load_predictions()
    weekly, selections = weekly_metrics(scores, panel, fixed)
    metrics, tests, selection = summarize(panel, weekly)
    market = spy_regressions(weekly)
    weekly.to_csv(OUTPUT / "weekly_forecasts.csv", index=False)
    metrics.to_csv(OUTPUT / "metrics.csv")
    tests.to_csv(OUTPUT / "paired_tests.csv", index=False)
    selection.to_csv(OUTPUT / "selection.csv")
    market.to_csv(OUTPUT / "spy_regressions.csv", index=False)
    pd.DataFrame(
        [
            {"friday": day, "model": model, "symbols": ",".join(names)}
            for (day, model), names in selections.items()
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
            "primary_models": list(MAIN),
            "primary_seed": 42,
            "evaluation_weeks": 33,
            "stock_outcomes": len(panel),
            "metrics": json.loads(metrics.reset_index().to_json(orient="records")),
            "paired_tests": json.loads(tests.to_json(orient="records")),
            "selection": json.loads(selection.reset_index().to_json(orient="records")),
            "dataset_sha256": sha256(OUTPUT / "dataset.json"),
            "fit_manifest_sha256": {
                path.name: sha256(path)
                for path in OUTPUT.glob("*.json")
                if path.stem
                in (
                    "ridge",
                    *(
                        f"{arm}_{seed}"
                        for arm in ("xgboost", "patchtst")
                        for seed in SEEDS
                    ),
                )
            },
            "sources": {
                str(path): sha256(path)
                for path in (
                    FORECAST_EVIDENCE / "stock_forecasts.parquet",
                    FORECAST_EVIDENCE / "weekly_forecasts.csv",
                    Path(__file__),
                )
            },
            "interpretation": "Exploratory historical forecast experiment; gross selection diagnostics",
        },
    )
    plot(weekly)
    update_report(metrics, tests, selection, market)
    print(metrics.to_string())
    print(tests.loc[tests.block.eq(4)].to_string(index=False))
    print(selection.to_string())


if __name__ == "__main__":
    main()
