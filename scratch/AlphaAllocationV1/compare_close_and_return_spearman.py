"""Add Friday's one-session close return to the archived V2 Spearman study."""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from build_friday_dataset import (
    ACTION_DATES,
    DEVELOPMENT_END,
    END,
    FEATURES,
    OHLCV_FEATURES,
    START,
    audit_candidates,
    join_friday_ohlcv,
    load_prices,
    plt,
    stack,
    verify_prices,
    weekly_spearman,
)
from build_friday_dataset import (
    OUTPUT as FRIDAY_SOURCE,
)
from compare_v1_forecast_accuracy import paired_interval

OUTPUT = Path(__file__).resolve().parent / "data/close_return_spearman"
DAILY = "close_log_return_1d"
FEATURE_LABELS = {**FEATURES, **OHLCV_FEATURES, DAILY: "Close return 1d"}


def load_panel():
    calendar, close, _ = load_prices()
    grid = pd.date_range(START, END, freq="W-FRI")
    _, weekly_audit, daily_audit = audit_candidates(close, grid)
    adjusted, _ = verify_prices(weekly_audit, daily_audit, close)
    daily = np.log(adjusted).diff()
    for day, symbol in daily_audit.index[
        daily_audit.status.str.startswith("unresolved")
    ]:
        daily.loc[day, symbol] = np.nan
    for symbol, day in ACTION_DATES:
        daily.loc[day, symbol] = np.nan
    with duckdb.connect() as db:
        archived = db.read_parquet(str(FRIDAY_SOURCE / "friday_dataset.parquet")).df()
    panel = join_friday_ohlcv(archived.set_index(["friday", "symbol"]).sort_index())
    fridays = panel.index.get_level_values("friday").unique()
    if len(close.columns) != panel.index.get_level_values("symbol").nunique():
        raise ValueError("The close-return cohort must match the original")
    if not all(calendar.is_session(day) for day in fridays):
        raise ValueError("Non-trading dates in the archived panel")
    panel[DAILY] = stack(daily.loc[fridays]).reindex(panel.index)
    return panel


def draw_heatmap(table):
    labels = list(FEATURE_LABELS.values())
    keys = list(FEATURE_LABELS)
    matrix = pd.DataFrame(np.nan, index=keys, columns=keys)
    for row in table.itertuples(index=False):
        matrix.loc[row.left, row.right] = matrix.loc[row.right, row.left] = (
            row.mean_weekly_spearman
        )
    if matrix.isna().any().any():
        raise ValueError("Incomplete expanded Spearman matrix")
    fig, ax = plt.subplots(figsize=(14.5, 11), layout="constrained")
    fig.suptitle(
        "Friday feature overlap with one-day close return",
        fontsize=21,
        fontweight="bold",
    )
    ax.set_title(
        "Mean weekly cross-sectional Spearman • 356-stock cohort • 2016–2024", pad=12
    )
    image = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)), labels, rotation=42, ha="right")
    ax.set_yticks(range(len(keys)), labels)
    for i, j in np.ndindex(matrix.shape):
        value = matrix.iat[i, j]
        ax.text(
            j,
            i,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=9.5,
            color="white" if abs(value) >= 0.6 else "#172b4d",
            fontweight="bold" if i == len(keys) - 1 or j == len(keys) - 1 else "normal",
        )
    fig.colorbar(image, ax=ax, label="Spearman rank correlation", shrink=0.82)
    fig.savefig(OUTPUT / "feature_spearman_with_close_return_heatmap.png", dpi=150)
    plt.close(fig)


def estimate(weekly):
    mean, low, high, _ = paired_interval(weekly.spearman)
    return {
        "mean_spearman": mean,
        "ci95": [low, high],
        "weeks": len(weekly),
        "median_paired_stocks": float(weekly.paired_stocks.median()),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    dates = panel.index.get_level_values("friday")
    data = panel.loc[panel.price_usable_at_cutoff & (dates <= DEVELOPMENT_END)]
    new = {name: weekly_spearman(data, name, DAILY) for name in FEATURE_LABELS}
    archive = pd.read_csv(FRIDAY_SOURCE / "feature_spearman_with_ohlcv.csv")
    added = pd.DataFrame(
        [
            {
                "left": name,
                "right": DAILY,
                "mean_weekly_spearman": value.spearman.mean(),
                "weeks": len(value),
                "median_paired_stocks": value.paired_stocks.median(),
            }
            for name, value in new.items()
        ]
    )
    table = pd.concat([archive, added], ignore_index=True)
    table.to_csv(OUTPUT / "feature_spearman_with_close_return.csv", index=False)
    pd.concat(new, names=["feature"]).to_csv(OUTPUT / "weekly_overlap.csv")

    matched = data.loc[
        data.next_friday.le(DEVELOPMENT_END), ["close", DAILY, "realized_return_rank"]
    ].dropna()
    matched["rank_score"] = -matched.realized_return_rank
    rank = {
        name: weekly_spearman(matched, name, "rank_score") for name in ("close", DAILY)
    }
    if not rank["close"].index.equals(rank[DAILY].index):
        raise ValueError("Close and daily-return rank IC dates differ")
    rank_delta = rank[DAILY].assign(
        spearman=rank[DAILY].spearman - rank["close"].spearman
    )

    pd.concat(rank, names=["feature"]).to_csv(OUTPUT / "weekly_next_friday_rank_ic.csv")
    source = FRIDAY_SOURCE / "feature_spearman_with_ohlcv.csv"
    summary = {
        "original_matrix_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_pairs_preserved": len(archive),
        "close_vs_daily_return": estimate(new["close"]),
        "next_friday_rank_ic": {name: estimate(value) for name, value in rank.items()},
        "rank_ic_difference_daily_return_minus_close": estimate(rank_delta),
        "bootstrap": {"block_week_observations": 4, "draws": 20_000, "seed": 42},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw_heatmap(table)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
