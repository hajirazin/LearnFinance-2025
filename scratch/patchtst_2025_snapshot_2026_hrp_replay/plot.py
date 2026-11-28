"""Export all four original comparisons plus HRP in the original visual style."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parent
COLORS = {
    "PatchTST": "#003D4F",
    "PatchTST + HRP": "#634F7D",
    "SPY": "#F2617A",
    "QQQ": "#CC850A",
    "S&P 500": "#6B9E78",
}
STYLES = {
    "PatchTST": "-",
    "PatchTST + HRP": "--",
    "SPY": "--",
    "QQQ": ":",
    "S&P 500": "-.",
}


def main():
    fonts = ROOT.parent / "patchtst_2025_snapshot_2026_replay/data/fonts"
    for path in fonts.glob("*"):
        fm.fontManager.addfont(path)
    title = fm.FontProperties(fname=fonts / "Bitter-Bold.ttf")
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 11,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#EDF1F3",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
            "text.parse_math": False,
        }
    )
    summary = json.loads((ROOT / "summary.json").read_text())
    nav = pd.read_csv(ROOT / "weekly_nav.csv", index_col="date", parse_dates=True)
    assert len(nav) == summary["weeks"] + 1
    assert set(nav.columns) == set(COLORS)
    fig, (ax, dd) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.76, bottom=0.21, hspace=0.20)
    fig.text(
        0.08,
        0.95,
        "The same PatchTST stocks: equal weight vs HRP",
        fontproperties=title,
        fontsize=26,
    )
    fig.text(
        0.08,
        0.911,
        "January 5 to November 25, 2026  |  $10,000 start  |  Top 15, sticky 15 / 20  |  Zero trading costs",
        fontsize=12,
    )
    for x, label in zip([0.08, 0.26, 0.44, 0.62, 0.80], COLORS, strict=True):
        metric = summary["metrics"][label]
        name = "PatchTST equal weight" if label == "PatchTST" else label
        fig.text(x, 0.861, name, fontsize=12)
        fig.text(
            x,
            0.820,
            f"${metric['ending_value']:,.0f}  {metric['total_return']:+.2%}",
            fontproperties=title,
            fontsize=20,
            color=COLORS[label],
        )
    for label, color in COLORS.items():
        curve = nav[label].to_numpy()
        assert np.isclose(
            curve[-1], summary["metrics"][label]["ending_value"], atol=1e-8
        )
        ax.plot(
            nav.index,
            curve,
            label="PatchTST equal weight" if label == "PatchTST" else label,
            color=color,
            linestyle=STYLES[label],
            linewidth=2.5 if "PatchTST" in label else 1.8,
            marker="o",
            markersize=3,
        )
        ax.plot(
            nav.index[-1],
            curve[-1],
            marker="D",
            markersize=6,
            markerfacecolor="#FFFFFF",
            markeredgecolor=color,
            linestyle="none",
        )
        dd.plot(
            nav.index,
            curve / np.maximum.accumulate(curve) - 1,
            color=color,
            linestyle=STYLES[label],
            linewidth=1.8,
        )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio value (USD)")
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=10)
    ax.axhline(10000, color="#EDF1F3", linewidth=1, zorder=0)
    dd.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    dd.set_ylabel("Drawdown\n(weekly marks)")
    dd.xaxis.set_major_locator(mdates.MonthLocator())
    dd.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    dd.set_xlim(pd.Timestamp("2026-01-01"), nav.index[-1] + pd.Timedelta(days=5))
    for axis in (ax, dd):
        axis.grid(axis="y", color="#EDF1F3", linewidth=1)
    notes = [
        "HRP uses 252 daily adjusted-close returns through each forecast anchor. Both strategies hold exactly the same 15 stocks.",
        "46 completed weeks plus the November 23 week to date, marked November 25 (open diamond); original frozen evidence retained.",
        "Holdings carry over weekends; fractional units rebalance at the next weekly open. No added capital, fees or cash reserve.",
        "SPY and QQQ are adjusted-price buy-and-hold comparisons; S&P 500 is the price-only index. Original four curves are preserved.",
        "Retrospective 417-name roster has survivorship bias. Snapshot fitted in November 2026 using labels ending in 2025.",
    ]
    for y, note in zip([0.155, 0.127, 0.099, 0.071, 0.043], notes, strict=True):
        fig.text(0.08, y, note, fontsize=10)
    fig.text(0.08, 0.016, "© Thoughtworks 2026   |   Confidential", fontsize=8)
    fig.text(0.96, 0.016, "1", fontsize=8, ha="right")
    fig.savefig(ROOT / "weekly_performance_2026.png", dpi=180)
    fig.savefig(ROOT / "weekly_performance_2026.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
