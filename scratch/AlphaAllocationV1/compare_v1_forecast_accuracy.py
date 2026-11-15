"""Compare archived H1–H27 on exact Friday-close forecast error and draw charts."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brain_api.core.sticky_selection import select_with_rank_band

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/v1_hypotheses"
KEYS = ["friday", "symbol"]
BASE = {
    "current": "V1",
    "h1_features": "H1",
    "h2_year": "H2",
    "h3_training_filter": "H3",
    "h4_combined": "H4",
    "h5_features_training_filter": "H5",
}
TITLES = {
    1: "Seven features",
    2: "One-year input window",
    3: "Fit on 356 stocks",
    4: "Seven features, one year, and 356 stocks",
    5: "Seven features and 356 stocks",
    21: "Five-session return MSE",
    22: "Pairwise ranking loss",
    23: "SPY-relative selection objective",
    24: "Five-session MSE with IBKR fees",
    25: "Ranking with prior membership and IBKR fees",
    26: "Seven daily forecast targets",
    27: "Seven weekly forecast targets",
}
SCORE_ONLY = {"H22", "H23", "H25"}
MODELS = [*BASE.values(), *(f"H{i}" for i in range(21, 28)), "H21 chronological"]
NATIVE = [name for name in MODELS if name not in SCORE_ONLY]
MIN_HISTORY = 8
COLORS = {"model": "#126f73", "v1": "#7860aa", "actual": "#172b4d", "spy": "#c37224"}
SCORE_FILES = {
    "scores.csv": BASE,
    "loss_comparison/stock_scores.csv": {
        "h21": "H21",
        "h22": "H22",
        "h23": "H23",
        "actual_five_session_return": "actual_five",
        "actual_friday_return": "actual",
    },
    "loss_comparison/h24_ibkr/stock_scores.csv": {
        "h24": "H24",
        "h21_chronological": "H21 chronological",
    },
    "loss_comparison/h25_rank_ibkr/stock_ranks.csv": {"h25": "H25"},
    "loss_comparison/multitarget/stock_scores.csv": {"h26": "H26", "h27": "H27"},
}


def read(relative, **kwargs):
    return pd.read_csv(SOURCE / relative, **kwargs)


def unique_index(frame, keys):
    indexed = frame.set_index(keys)
    if not indexed.index.is_unique:
        raise ValueError(f"Duplicate input keys for {keys}")
    return indexed


def load_scores():
    frames = [
        unique_index(
            read(file, usecols=[*KEYS, *columns], parse_dates=["friday"]), KEYS
        ).rename(columns=columns)
        for file, columns in SCORE_FILES.items()
    ]
    scores = pd.concat(frames, axis=1, join="inner").sort_index()
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna(subset=MODELS)
    if scores.groupby(level="friday").size().min() < 15:
        raise ValueError("Insufficient shared candidates for a top-15 comparison")
    return scores


def exact_weeks(scores, spy):
    dates = scores.index.get_level_values("friday").unique()
    calendar = xcals.get_calendar("XNYS")
    end = dates + pd.Timedelta(days=7)
    exact = [
        date
        for date, next_friday in zip(dates, end, strict=True)
        if calendar.is_session(date)
        and calendar.is_session(next_friday)
        and calendar.sessions_distance(date, next_friday) == 6
    ]
    result = scores.loc[exact].copy()
    both = result[["actual", "actual_five"]].dropna()
    if not np.allclose(both.actual, both.actual_five, rtol=1e-5, atol=1e-7):
        raise ValueError("Observed Friday and five-session outcomes diverge")
    # The five-session labels include the archived daily-return quality checks.
    result["actual"] = result["actual_five"]
    ends = pd.DatetimeIndex(exact) + pd.Timedelta(days=7)
    spy_return = (
        spy.close.reindex(ends).to_numpy() / spy.close.loc[exact].to_numpy()
    ) - 1
    if not np.isfinite(spy_return).all():
        raise ValueError("Missing SPY closes for the forecast horizon")
    return result, pd.Series(spy_return, index=exact, name="spy")


def calibration(panel):
    """Map score percentiles to returns using only completed earlier windows."""
    percentile = panel[MODELS].groupby(level="friday").rank(pct=True) - 0.5
    dates = panel.index.get_level_values("friday")
    observed = panel.actual.notna()
    calibrated = pd.DataFrame(np.nan, index=panel.index, columns=MODELS)
    market_mean = pd.Series(np.nan, index=panel.index)
    coefficients = []
    for date in dates.unique():
        history = (dates + pd.Timedelta(days=7) <= date) & observed
        weeks = dates[history].nunique()
        if weeks < MIN_HISTORY:
            continue
        x = percentile.loc[history]
        y = panel.loc[history, "actual"]
        x_centered = x - x.mean()
        slope = x_centered.mul(y - y.mean(), axis=0).sum() / x_centered.pow(2).sum()
        # Enforce the archived convention that a higher score implies higher rank.
        slope = slope.fillna(0.0).clip(lower=0.0)
        intercept = y.mean() - slope * x.mean()
        decision = dates == date
        calibrated.loc[decision] = percentile.loc[decision] * slope + intercept
        market_mean.loc[decision] = y.mean()
        for name in MODELS:
            coefficients.append(
                {
                    "friday": date,
                    "model": name,
                    "history_weeks": weeks,
                    "last_label_available": dates[history].max() + pd.Timedelta(days=7),
                    "intercept": intercept[name],
                    "slope": slope[name],
                }
            )
    calibrated["Past universe mean"] = market_mean
    return calibrated, pd.DataFrame(coefficients)


def sticky_baskets(scores):
    """Choose all baskets from scores before checking any realized outcomes."""
    previous = dict.fromkeys(MODELS)
    results = {}
    for date, week in scores.groupby(level="friday", sort=True):
        week = week.droplevel("friday")
        for name in MODELS:
            picks = select_with_rank_band(week[name].to_dict(), previous[name], 15, 20)
            if len(picks.selected) != 15:
                raise ValueError(f"Incomplete selection for {name} on {date}")
            previous[name] = set(picks.selected)
            results[date, name] = picks.selected
    return results


def build_weekly(panel, forecasts, selections, spy, method):
    rows = []
    for date, week in panel.groupby(level="friday", sort=True):
        week = week.droplevel("friday")
        prediction = forecasts.loc[date]
        if prediction.isna().all().all():
            continue
        if not np.isfinite(prediction.to_numpy()).all():
            raise ValueError(f"Partial predictions in {method} on {date}")
        prediction = prediction.assign(Zero=0.0)
        fixed = selections[date, "V1"]
        actual = week.actual
        realized = actual.dropna()
        if len(realized) < 15:
            raise ValueError(f"Insufficient observed returns on {date}")
        winners = set(
            realized.sort_values(ascending=False, kind="stable").head(15).index
        )
        for name in prediction:
            predicted = prediction[name]
            error = predicted.loc[realized.index] - realized
            own = selections.get((date, name), fixed)
            fixed_actual = actual.loc[fixed].mean(skipna=False)
            own_actual = actual.loc[own].mean(skipna=False)
            rows.append(
                {
                    "friday": date,
                    "method": method,
                    "model": name,
                    "candidates": len(week),
                    "observed_stocks": len(realized),
                    "spy": spy.loc[date],
                    "stock_mae": error.abs().mean(),
                    "stock_mse": error.pow(2).mean(),
                    "stock_bias": error.mean(),
                    "stock_direction_hit": (
                        np.sign(predicted.loc[realized.index]) == np.sign(realized)
                    ).mean(),
                    "rank_ic": (
                        week.loc[realized.index, name].rank().corr(realized.rank())
                        if name in MODELS
                        else np.nan
                    ),
                    "own_top15_hits": len(winners.intersection(own))
                    if name in MODELS
                    else np.nan,
                    "fixed_predicted": predicted.loc[fixed].mean(),
                    "fixed_actual": fixed_actual,
                    "own_predicted": predicted.loc[own].mean(),
                    "own_actual": own_actual,
                    "v1_on_own": prediction.loc[own, "V1"].mean(),
                    "own_symbols": ",".join(own),
                    "fixed_symbols": ",".join(fixed),
                }
            )
    rows = pd.DataFrame(rows)
    for basket in ("fixed", "own"):
        rows[f"{basket}_error"] = rows[f"{basket}_predicted"] - rows[f"{basket}_actual"]
        rows[f"{basket}_absolute_error"] = rows[f"{basket}_error"].abs()
    return rows


def paired_interval(values, block=4):
    values = np.asarray(values, dtype=float)
    if len(values) <= block or not np.isfinite(values).all():
        raise ValueError("Not enough paired weekly errors")
    rng = np.random.default_rng(42)
    starts = rng.integers(
        0, len(values), size=(20_000, (len(values) + block - 1) // block)
    )
    draws = (starts[..., None] + np.arange(block)) % len(values)
    distribution = values[draws.reshape(20_000, -1)[:, : len(values)]].mean(axis=1)
    mean = values.mean()
    p_lower = (1 + np.count_nonzero(distribution - mean <= mean)) / (
        len(distribution) + 1
    )
    return mean, *np.quantile(distribution, [0.025, 0.975]), p_lower


def holm(pvalues):
    ordered = pvalues.dropna().sort_values()
    return (ordered * np.arange(len(ordered), 0, -1)).cummax().clip(upper=1.0)


def summarize(weekly):
    records = []
    for method, frame in weekly.groupby("method", sort=False):
        fixed_errors = frame.pivot(
            index="friday", columns="model", values="fixed_absolute_error"
        ).dropna()
        own_errors = frame.pivot(
            index="friday", columns="model", values="own_absolute_error"
        ).dropna()
        for name, rows in frame.groupby("model", sort=False):
            fixed = rows.dropna(subset=["fixed_error"])
            own = rows.dropna(subset=["own_error"])
            record = {
                "method": method,
                "model": name,
                "stock_weeks": len(rows),
                "stock_observations": rows.observed_stocks.sum(),
                "stock_mae_pp": 100 * rows.stock_mae.mean(),
                "stock_rmse_pp": 100 * np.sqrt(rows.stock_mse.mean()),
                "stock_bias_pp": 100 * rows.stock_bias.mean(),
                "stock_direction_pct": 100 * rows.stock_direction_hit.mean(),
                "fixed_weeks": len(fixed),
                "fixed_mae_pp": 100 * fixed.fixed_absolute_error.mean(),
                "fixed_rmse_pp": 100 * np.sqrt(fixed.fixed_error.pow(2).mean()),
                "fixed_bias_pp": 100 * fixed.fixed_error.mean(),
                "fixed_spy_down_mae_pp": 100
                * fixed.loc[fixed.spy.lt(0), "fixed_absolute_error"].mean(),
                "fixed_spy_up_mae_pp": 100
                * fixed.loc[fixed.spy.ge(0), "fixed_absolute_error"].mean(),
                "fixed_spy_down_weeks": int(fixed.spy.lt(0).sum()),
                "own_weeks": len(own),
                "own_mae_pp": 100 * own.own_absolute_error.mean(),
                "zero_on_same_own_mae_pp": 100 * own.own_actual.abs().mean(),
                "own_common_weeks": len(own_errors),
                "own_common_mae_pp": 100 * own_errors[name].mean(),
                "v1_on_same_own_mae_pp": 100
                * (own.v1_on_own - own.own_actual).abs().mean(),
                "rank_ic": rows.rank_ic.mean(),
                "own_top15_hits": rows.own_top15_hits.mean(),
            }
            for reference in ("V1", "Zero"):
                if name == reference:
                    continue
                mean, low, high, p = paired_interval(
                    100 * (fixed_errors[name] - fixed_errors[reference])
                )
                record.update(
                    {
                        f"fixed_delta_vs_{reference}_pp": mean,
                        f"fixed_delta_vs_{reference}_ci_low_pp": low,
                        f"fixed_delta_vs_{reference}_ci_high_pp": high,
                        f"fixed_p_vs_{reference}": p,
                    }
                )
            records.append(record)
    metrics = pd.DataFrame(records)
    for reference in ("V1", "Zero"):
        field = f"fixed_p_vs_{reference}"
        # One exploratory family per method and reference.
        metrics[field + "_holm"] = metrics.groupby("method")[field].transform(holm)
    return metrics


def archive_tables(spy):
    def open_returns(frame):
        frame = frame.assign(
            spy=frame.next_mark.map(spy.open) / frame.fill.map(spy.open) - 1
        )
        if not np.isfinite(frame.spy).all():
            raise ValueError("Missing SPY open for an archived execution interval")
        return frame

    paths = {
        "hrp": "weekly_results.csv",
        "equal": "equal_weight/weekly_results.csv",
        "loss": "loss_comparison/portfolio_weekly.csv",
        "cost": "loss_comparison/h24_ibkr/portfolio_weekly.csv",
        "rank_cost": "loss_comparison/h25_rank_ibkr/portfolio_weekly.csv",
        "multitarget": "loss_comparison/multitarget/portfolio_weekly.csv",
    }
    archive = {
        group: open_returns(read(path, parse_dates=["friday", "fill", "next_mark"]))
        for group, path in paths.items()
    }
    archive["ranking"] = read("diagnostics/ranking_weekly.csv", parse_dates=["friday"])
    path = "loss_comparison/rank_weekly.csv"
    archive["loss_ranking"] = read(path, parse_dates=["friday"])
    return archive, [*paths.values(), path]


def hypothesis(number):
    model = (number - 1) % 5 + 1 if number <= 20 else number
    label = (
        "HRP replay"
        if number <= 5
        else "Equal-weight replay"
        if number <= 10
        else "Top-15 ranking comparison"
        if number <= 15
        else "Equal-weight SPY comparison"
        if number <= 20
        else "Archived portfolio comparison"
    )
    return f"H{model}", TITLES[model], label


def original_comparison(number, archive):
    if number <= 20:
        key = list(BASE)[(number - 1) % 5 + 1]
        group = "hrp" if number <= 5 else "ranking" if 11 <= number <= 15 else "equal"
        source = archive[group]
        rows = source.loc[source.comparison.eq(key)]
        value = "sticky_top15_hits" if group == "ranking" else "net_return"
        frame = rows.pivot(index="friday", columns="arm", values=value)
        if group != "ranking":
            frame["SPY"] = rows.groupby("friday").spy.first()
        else:
            frame["Random"] = rows.groupby("friday").random_expected_hits.mean()
        return frame.rename(columns={"current": "V1", key: "Model"}), group

    if number == 22:
        rows = archive["loss_ranking"]
        rows = rows.loc[rows.arm.isin(["current", "h22"])]
        frame = rows.pivot(index="friday", columns="arm", values="sticky_top15_hits")
        frame["Random"] = rows.groupby("friday").random_expected_hits.mean()
        return frame.rename(columns={"current": "V1", "h22": "Model"}), "ranking"

    group = (
        "loss"
        if number < 24
        else "cost"
        if number == 24
        else "rank_cost"
        if number == 25
        else "multitarget"
    )
    rows = archive[group]
    if group in ("rank_cost", "multitarget"):
        rows = rows.loc[rows.weighting.eq("hrp") & ~rows.arm.eq("spy")]
    frame = rows.pivot(index="friday", columns="arm", values="net_return")
    frame["SPY"] = rows.groupby("friday").spy.first()
    reference = "v1" if group in ("rank_cost", "multitarget") else "current"
    keys = {reference: "V1", f"h{number}": "Model"}
    if number == 24:
        keys["h21"] = "H21"
        keys["h21_chronological"] = "H21 chronological"
    if number == 25:
        keys["oracle"] = "Hindsight top 15"
    if number in (26, 27):
        keys["h5" if number == 26 else "h21"] = "H5" if number == 26 else "H21"
    return frame[[*keys, "SPY"]].rename(columns=keys), group


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#42536a",
            "axes.titleweight": "bold",
            "xtick.color": "#42536a",
            "ytick.color": "#42536a",
            "axes.edgecolor": "#acb8c5",
            "grid.color": "#e1e7ee",
            "figure.facecolor": "#f6f8fb",
            "axes.facecolor": "#f6f8fb",
            "savefig.facecolor": "#f6f8fb",
        }
    )


def line(ax, values, label, color, *, style="-", scale=100, show_gaps=True):
    if show_gaps:
        values = values.asfreq("W-FRI")
    ax.plot(
        values.index,
        scale * values,
        linestyle=style,
        linewidth=1.65,
        marker="o" if show_gaps else None,
        markersize=2.4,
        label=label,
        color=color,
    )


def decorate(ax, title, units, *, zero=True):
    ax.set_title(title, loc="left", fontsize=11, pad=23)
    ax.set_ylabel(units)
    ax.grid(axis="y", linewidth=0.65)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if zero:
        ax.axhline(0, color="#a1aab6", linewidth=0.8, zorder=0)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncols=5,
        fontsize=8,
        frameon=False,
        borderaxespad=0,
    )


def plot_one(number, weekly, metrics, original, group, target):
    name, title, label = hypothesis(number)
    method = "Calibrated" if name in SCORE_ONLY else "Native"
    frame = weekly.loc[weekly.method.eq(method)]
    model = frame.loc[frame.model.eq(name)].set_index("friday")
    v1 = frame.loc[frame.model.eq("V1")].set_index("friday")
    stat = metrics.loc[metrics.method.eq(method) & metrics.model.eq(name)].iloc[0]
    ref = metrics.loc[metrics.method.eq(method) & metrics.model.eq("V1")].iloc[0]
    _, axes = plt.subplots(5, 1, figsize=(12.5, 12.9))
    fig = axes[0].figure
    fig.suptitle(
        f"H{number:02d}   {title}",
        x=0.09,
        y=0.979,
        ha="left",
        fontsize=21,
        color="#172b4d",
        fontweight="bold",
    )
    method_label = (
        "Past-only calibration of rank scores"
        if method == "Calibrated"
        else "Native return forecasts"
    )
    fig.text(
        0.09,
        0.946,
        f"{label}   •   {method_label}   •   Underlying model {name}",
        fontsize=10,
        color="#42536a",
    )
    fig.text(
        0.09,
        0.925,
        f"Same-basket MAE  {stat.fixed_mae_pp:.2f} pp   |   V1 refit  {ref.fixed_mae_pp:.2f} pp   |   Stock MAE  {stat.stock_mae_pp:.2f} pp   |   {int(stat.fixed_weeks)} matched basket weeks",
        fontsize=10,
        fontweight="bold",
        color=COLORS["model"],
    )

    fixed = model.fixed_actual.notna()
    line(axes[0], model.fixed_actual, "Actual basket", COLORS["actual"])
    line(axes[0], model.fixed_predicted.where(fixed), "Model forecast", COLORS["model"])
    line(axes[0], v1.fixed_predicted.where(fixed), "V1 forecast", COLORS["v1"])
    line(axes[0], model.spy, "Realized SPY", COLORS["spy"], style="--")
    decorate(axes[0], "Same 15 stocks selected by V1 for both forecasts", "Return (%)")

    line(axes[1], model.fixed_absolute_error, "Model absolute error", COLORS["model"])
    line(axes[1], v1.fixed_absolute_error, "V1 absolute error", COLORS["v1"])
    line(
        axes[1],
        model.fixed_actual.abs(),
        "Zero forecast absolute error",
        "#748396",
        style=":",
    )
    decorate(
        axes[1], "Weekly forecast error on the same basket", "Error (pp)", zero=False
    )

    own = model.own_actual.notna()
    line(axes[2], model.own_actual, "Actual model basket", COLORS["actual"])
    line(axes[2], model.own_predicted.where(own), "Model forecast", COLORS["model"])
    line(axes[2], model.v1_on_own.where(own), "V1 on those same names", COLORS["v1"])
    line(axes[2], model.spy, "Realized SPY", COLORS["spy"], style="--")
    decorate(
        axes[2],
        "Each model’s own 15 picks, with both forecasts evaluated on those names",
        "Return (%)",
    )

    line(axes[3], model.stock_mae, "Model stock MAE", COLORS["model"])
    line(axes[3], v1.stock_mae, "V1 stock MAE", COLORS["v1"])
    zero = frame.loc[frame.model.eq("Zero")].set_index("friday")
    line(axes[3], zero.stock_mae, "Zero forecast MAE", "#748396", style=":")
    decorate(
        axes[3],
        "Mean absolute error across all matched stocks each week",
        "Error (pp)",
        zero=False,
    )

    colors = {
        "Model": COLORS["model"],
        "V1": COLORS["v1"],
        "SPY": COLORS["spy"],
        "Random": "#748396",
        "H21": "#748396",
        "H21 chronological": "#636341",
    }
    if group == "ranking":
        for name_, series in original.items():
            line(
                axes[4],
                series,
                name_,
                colors[name_],
                scale=1,
                style="--" if name_ == "Random" else "-",
                show_gaps=False,
            )
        decorate(axes[4], "Original comparison · realized top-15 winners found", "Hits")
    elif 16 <= number <= 20 or number == 23:
        cumulative = (1 + original).cumprod() - 1
        for name_ in ("Model", "V1"):
            line(
                axes[4],
                cumulative[name_] - cumulative.SPY,
                name_,
                colors[name_],
                show_gaps=False,
            )
        decorate(
            axes[4],
            "Original comparison · cumulative lead over SPY at next opens",
            "Lead (pp)",
        )
    else:
        cumulative = (1 + original).cumprod() - 1
        oracle = cumulative.pop("Hindsight top 15") if number == 25 else None
        for name_, series in cumulative.items():
            line(
                axes[4],
                series,
                name_,
                colors.get(name_, "#748396"),
                style="--" if name_ == "SPY" else "-",
                show_gaps=False,
            )
        weighting = (
            "HRP" if group in ("hrp", "rank_cost", "multitarget") else "equal weight"
        )
        decorate(
            axes[4],
            f"Original comparison · next-open net return ({weighting})"
            + (
                f"  |  Hindsight top 15 finished {oracle.iloc[-1]:+,.0%}"
                if oracle is not None
                else ""
            ),
            "Cumulative (%)",
        )

    for ax in axes:
        ax.set_xlim(pd.Timestamp("2026-01-06"), pd.Timestamp("2026-11-05"))
    fig.text(
        0.09,
        0.025,
        "Forecast panels use matched five-session Friday closes. Gaps mark omitted dates. SPY is a realized market reference. V1 is the archived research refit.",
        fontsize=8.4,
        color="#42536a",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.872, bottom=0.06, hspace=0.92)
    fig.savefig(target, dpi=130)
    plt.close(fig)


def plot_overview(metrics, target):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5))
    charts = [
        ("Native", "fixed_mae_pp", "Same-basket MAE"),
        ("Native", "stock_mae_pp", "Individual-stock MAE"),
        ("Calibrated", "fixed_mae_pp", "Calibrated basket MAE"),
    ]
    for ax, (method, metric, title) in zip(axes, charts, strict=True):
        rows = metrics.loc[metrics.method.eq(method)].sort_values(
            metric, ascending=False
        )
        colors = [
            COLORS["v1"]
            if name == "V1"
            else "#748396"
            if name in ("Zero", "Past universe mean")
            else COLORS["model"]
            for name in rows.model
        ]
        bars = ax.barh(rows.model, rows[metric], color=colors, height=0.62)
        ax.bar_label(
            bars, labels=[f"{v:.2f}" for v in rows[metric]], padding=4, fontsize=8
        )
        ax.set_xlim(0, rows[metric].max() * 1.18)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean absolute error (percentage points)")
        ax.grid(axis="x", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        "V1 forecast accuracy on matched 2026 weeks",
        fontsize=20,
        x=0.05,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.89,
        "Lower is better. Native and calibrated comparisons cover different date ranges.",
        color="#42536a",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.82, bottom=0.10, wspace=0.67)
    fig.savefig(target, dpi=145)
    plt.close(fig)


def record_original(frame, group):
    if group == "ranking":
        return {
            "original_weeks": len(frame),
            "original_metric": "mean top-15 hits",
            "original_model": frame.Model.mean(),
            "original_v1": frame.V1.mean(),
            "original_spy": np.nan,
            "original_extra": f"Random {frame.Random.mean():.2f} hits",
        }
    weighting = (
        "HRP" if group in ("hrp", "rank_cost", "multitarget") else "equal weight"
    )
    totals = ((1 + frame).prod() - 1) * 100
    extras = ", ".join(
        f"{n} {v:+.2f}%" for n, v in totals.drop(["Model", "V1", "SPY"]).items()
    )
    down = frame.SPY.lt(0)
    return {
        "original_weeks": len(frame),
        "original_metric": f"{weighting} compounded net %",
        "original_model": totals.Model,
        "original_v1": totals.V1,
        "original_spy": totals.SPY,
        "original_extra": extras,
        "original_weeks_ahead_spy": int(frame.Model.gt(frame.SPY).sum()),
        "original_spy_down_weeks": int(down.sum()),
        "original_ahead_on_down_weeks": int(
            frame.loc[down, "Model"].gt(frame.loc[down, "SPY"]).sum()
        ),
    }


def model_table(frame):
    header = (
        "| Model | Same 15 MAE | Stock MAE | Stock RMSE | Stock bias | Own 15 MAE* | V1 on those same picks* | Zero on those same picks* | SPY down MAE | SPY up MAE |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    columns = [
        "fixed_mae_pp",
        "stock_mae_pp",
        "stock_rmse_pp",
        "stock_bias_pp",
        "own_mae_pp",
        "v1_on_same_own_mae_pp",
        "zero_on_same_own_mae_pp",
        "fixed_spy_down_mae_pp",
        "fixed_spy_up_mae_pp",
    ]
    return header + "".join(
        f"| {row.model} | " + " | ".join(f"{row[key]:.3f}" for key in columns) + " |\n"
        for _, row in frame.sort_values("fixed_mae_pp").iterrows()
    )


def write_report(output, metrics, directory, panel):
    native = metrics.loc[metrics.method.eq("Native")].set_index("model", drop=False)
    calibrated = metrics.loc[metrics.method.eq("Calibrated")].set_index(
        "model", drop=False
    )
    winner = native.loc[NATIVE].sort_values("fixed_mae_pp").iloc[0]
    stock_winner = native.loc[NATIVE].sort_values("stock_mae_pp").iloc[0]
    own_winner = native.loc[NATIVE].sort_values("own_common_mae_pp").iloc[0]
    rmse_winner = native.loc[NATIVE].sort_values("fixed_rmse_pp").iloc[0]
    calibration_winner = calibrated.loc[MODELS].sort_values("fixed_mae_pp").iloc[0]
    dates = panel.index.get_level_values("friday").unique()
    first, last = dates.min().date(), dates.max().date()
    stock_rows = int(panel.actual.notna().sum())
    report = f"""# V1 forecast accuracy · H1–H27

**{winner.model} had the lowest native return forecast error on the same 15 stocks each week ({winner.fixed_mae_pp:.3f} pp MAE).** {stock_winner.model} had the lowest native stock-level MAE ({stock_winner.stock_mae_pp:.3f} pp). The reference is the saved V1 research refit.

[Overview chart](overview.png) · [All metrics and intervals](model_metrics.csv) · [Weekly forecast detail](weekly_forecasts.csv) · [Per-stock forecasts and actuals](stock_forecasts.parquet) · [Inputs and run settings](manifest.json)

## What was compared

- **Forecast target:** adjusted Friday close to the following Friday close, restricted to exactly five subsequent XNYS sessions. This matches the five-session forecasts and H27's weekly forecast. The common score panel had {len(dates)} eligible weeks ({first} to {last}, decision Fridays), with {stock_rows:,} observed stock outcomes. Exclude missing or unresolved archived actuals.
- **Primary metric:** mean absolute prediction error in percentage points on the same weekly equal-weight set of 15 stocks. Select using the V1 15/20 sticky rule on the common score universe before inspecting returns. Carry selections through all archived decisions, including weeks omitted from forecast scoring. All 15 realized returns must be available for a basket comparison. The stock metrics average each week's errors before averaging across weeks.
- **Own picks:** use each model's own 15/20 selections on that same shared universe. Also score the V1 forecast on exactly those names. Original archived portfolio and rank comparisons use their own eligibility and selection rules. H25's archived scores retain the prior-membership state from its original scoring run.
- **SPY:** show realized dividend-adjusted SPY on the same close-to-close horizon and split fixed-basket MAE by whether SPY fell. SPY is market context. Forecast errors are relative to the realized stocks or basket. The original trading panels use SPY adjusted opens over the archived executable intervals.
- **Calibration:** H22, H23, and H25 are selection scores. For a comparable secondary test, convert each model's score to a within-week percentile and fit a linear mapping with a nonnegative slope using only previously completed windows. Start once eight weeks of outcomes are known. Compare against zero and the expanding mean of past universe returns. This calibration is a separate forecasting rule.
- **Inference:** paired differences in same-basket absolute error, with 95% circular block-bootstrap intervals (4 adjacent evaluated weeks, 20,000 draws, seed 42). One-sided lower-error p-values are adjusted with Holm within each method and reference. Earlier experiments already used the evaluation period. Treat all results as exploratory.

## Native return forecasts

{model_table(native)}
Values are in percentage points. V1's fixed-basket MAE was {native.loc["V1", "fixed_mae_pp"]:.3f} pp and the zero-return forecast was {native.loc["Zero", "fixed_mae_pp"]:.3f} pp. The best native model's paired MAE difference versus V1 was {winner.fixed_delta_vs_V1_pp:+.3f} pp (95% CI {winner.fixed_delta_vs_V1_ci_low_pp:+.3f} to {winner.fixed_delta_vs_V1_ci_high_pp:+.3f}). Against zero it was {winner.fixed_delta_vs_Zero_pp:+.3f} pp (95% CI {winner.fixed_delta_vs_Zero_ci_low_pp:+.3f} to {winner.fixed_delta_vs_Zero_ci_high_pp:+.3f}). A negative difference favors the candidate.

Zero had a stock MAE of {native.loc["Zero", "stock_mae_pp"]:.3f} pp, compared with {stock_winner.stock_mae_pp:.3f} pp for the best trained model. {own_winner.model} had the smallest native own-pick MAE ({own_winner.own_mae_pp:.3f} pp). On exactly those picks, V1 scored {own_winner.v1_on_same_own_mae_pp:.3f} pp and zero scored {own_winner.zero_on_same_own_mae_pp:.3f} pp. {rmse_winner.model} had the lowest fixed-basket RMSE ({rmse_winner.fixed_rmse_pp:.3f} pp), a metric that penalizes larger misses more heavily.

In the {int(winner.fixed_spy_down_weeks)} weeks SPY fell, fixed-basket MAE was {winner.fixed_spy_down_mae_pp:.3f} pp for {winner.model}, {native.loc["V1", "fixed_spy_down_mae_pp"]:.3f} pp for V1, and {native.loc["Zero", "fixed_spy_down_mae_pp"]:.3f} pp for zero. A correctly predicted negative basket return counts as an accurate forecast. The sign of SPY describes the market week.

## Past-only calibration

{model_table(calibrated)}
{calibration_winner.model} led the calibrated model forecasts on the fixed basket at {calibration_winner.fixed_mae_pp:.3f} pp over {int(calibration_winner.fixed_weeks)} weeks, compared with {calibrated.loc["Zero", "fixed_mae_pp"]:.3f} pp for zero and {calibrated.loc["Past universe mean", "fixed_mae_pp"]:.3f} pp for the past mean. Native and calibrated metrics cover different periods.

*Own-pick MAEs may differ in stocks and available weeks across models. Each model's comparison against V1 on its own picks uses the identical baskets and weeks. The [metrics file](model_metrics.csv) also includes own-pick MAE on the intersection of observable weeks, counts, signed errors, directional accuracy, ranking diagnostics, and paired intervals.

## All 27 comparisons

Each chart shows the same-basket forecast and actual, weekly errors, each model's own picks, errors across individual stocks, and the original ranking or trading comparison. H6–H20 reuse the H1–H5 forecasts, so their forecast errors repeat by design. Historical trading outcomes are included for context.

| Chart | Model change | Forecast | Same 15 MAE | Stock MAE | Original metric | Model | V1 | SPY | Other original reference |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
"""
    for row in directory.itertuples(index=False):
        spy = "—" if pd.isna(row.original_spy) else f"{row.original_spy:+.2f}"
        report += (
            f"| [{row.hypothesis}](charts/{row.hypothesis}.png) | {row.title} | {row.method} | "
            f"{row.fixed_mae_pp:.3f} | {row.stock_mae_pp:.3f} | {row.original_metric} | "
            f"{row.original_model:.2f} | {row.original_v1:.2f} | {spy} | {row.original_extra or '—'} |\n"
        )
    report += f"""
[Original comparison data and market-down counts](hypothesis_metrics.csv) · [Calibration coefficients with label availability](calibration.csv)

## Final assessment

{winner.model} and H21 had the two lowest native MAEs on the fixed basket. Their advantage over the V1 refit supports a fresh forecast evaluation. None of the native or calibrated models established lower same-basket MAE than zero in the paired tests at the adjusted 5% level. The own-pick results answer a separate question because the realized basket changes with the model. Use the same-basket and stock-level errors, with the zero baseline, to evaluate whether a candidate improves prediction.
"""
    (output / "REPORT.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/forecast_accuracy")
    args = parser.parse_args()
    output = args.output
    charts = output / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    spy_path = "diagnostics/spy_adjusted_prices.csv"
    spy = unique_index(read(spy_path, parse_dates=["date"]), "date")
    scores = load_scores()
    panel, spy_close = exact_weeks(scores, spy)
    chosen = sticky_baskets(scores)
    calibrated, coefficients = calibration(panel)
    weekly = pd.concat(
        [
            build_weekly(panel, panel[NATIVE], chosen, spy_close, "Native"),
            build_weekly(panel, calibrated, chosen, spy_close, "Calibrated"),
        ],
        ignore_index=True,
    )
    metrics = summarize(weekly)
    archive, original_sources = archive_tables(spy)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    weekly.to_csv(output / "weekly_forecasts.csv", index=False)
    coefficients.to_csv(output / "calibration.csv", index=False)
    export = panel.drop(columns="actual_five").rename(
        columns={k: f"score_{k}" for k in SCORE_ONLY}
    )
    export = export.join(calibrated.add_prefix("calibrated_"))
    with duckdb.connect() as db:
        db.from_df(export.reset_index()).write_parquet(
            str(output / "stock_forecasts.parquet")
        )
    set_style()
    plot_overview(metrics, output / "overview.png")
    all_hypotheses = []
    for number in range(1, 28):
        hypothesis_id = f"H{number:02d}"
        name, title, label = hypothesis(number)
        original, group = original_comparison(number, archive)
        plot_one(
            number, weekly, metrics, original, group, charts / f"{hypothesis_id}.png"
        )
        method = "Calibrated" if name in SCORE_ONLY else "Native"
        row = metrics.loc[metrics.method.eq(method) & metrics.model.eq(name)].iloc[0]
        all_hypotheses.append(
            {
                "hypothesis": hypothesis_id,
                "title": title,
                "archived_comparison": label,
                **row.to_dict(),
                **record_original(original, group),
            }
        )
    all_hypotheses = pd.DataFrame(all_hypotheses)
    all_hypotheses.to_csv(output / "hypothesis_metrics.csv", index=False)
    write_report(output, metrics, all_hypotheses, panel)
    manifest = {
        "input_sha256": {
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in [
                *SCORE_FILES,
                *original_sources,
                spy_path,
                "diagnostics/ranking_weekly.csv",
            ]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_return_models": NATIVE,
        "score_only_models": sorted(SCORE_ONLY),
        "evaluation_fridays": [x.date().isoformat() for x in spy_close.index],
        "minimum_calibration_history_weeks": MIN_HISTORY,
        "bootstrap_block": 4,
        "bootstrap_draws": 20_000,
        "seed": 42,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(all_hypotheses)} hypothesis charts and {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
