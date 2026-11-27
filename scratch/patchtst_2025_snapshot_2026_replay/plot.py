"""Export a weekly-point comparison and drawdown chart with the saved style."""

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
    "SPY": "#F2617A",
    "QQQ": "#CC850A",
    "S&P 500": "#6B9E78",
}
STYLES = {"PatchTST": "-", "SPY": "--", "QQQ": ":", "S&P 500": "-."}


def main():
    fonts = ROOT / "data/fonts"
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
    final = nav.index[-1].strftime("%B %d").replace(" 0", " ")
    fig, (ax, dd) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.subplots_adjust(left=0.085, right=0.94, top=0.76, bottom=0.21, hspace=0.20)
    fig.text(
        0.085,
        0.95,
        "2026: $10,000 with the frozen PatchTST snapshot",
        fontproperties=title,
        fontsize=25,
    )
    fig.text(
        0.085,
        0.911,
        f"January 5 to {final}, 2026  |  Top 15, equal weight  |  Sticky rank band: 15 / 20  |  Zero trading costs",
        fontsize=11,
    )
    for x, label in zip([0.085, 0.31, 0.535, 0.76], COLORS, strict=True):
        metric = summary["metrics"][label]
        description = (
            " · weekly rebalance" if label == "PatchTST" else " · buy and hold"
        )
        if label == "S&P 500":
            description = " · price index"
        fig.text(
            x,
            0.859,
            label + description,
            fontsize=11,
        )
        fig.text(
            x,
            0.819,
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
            label=label,
            color=color,
            linestyle=STYLES[label],
            linewidth=2.4 if label == "PatchTST" else 1.9,
            marker="o",
            markersize=3.5,
        )
        if not summary["final_week_complete"]:
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
    ax.legend(loc="upper left", ncol=4, frameon=False)
    ax.axhline(10000, color="#EDF1F3", linewidth=1, zorder=0)
    dd.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    dd.set_ylabel("Drawdown\n(weekly marks)")
    dd.xaxis.set_major_locator(mdates.MonthLocator())
    dd.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    dd.set_xlim(pd.Timestamp("2026-01-01"), nav.index[-1] + pd.Timedelta(days=5))
    for axis in (ax, dd):
        axis.grid(axis="y", color="#EDF1F3", linewidth=1)
    final_note = (
        f"46 completed weeks plus November 23 week to date, valued at {final} close (open diamond); November 27 was still trading."
        if not summary["final_week_complete"]
        else "Every point marks the week's final trading-session close."
    )
    fig.text(0.085, 0.155, final_note, fontsize=10)
    fig.text(
        0.085,
        0.127,
        "Hold stocks over weekends; rebalance at the next weekly open. Fractional adjusted-price units; no additional cash.",
        fontsize=10,
    )
    fig.text(
        0.085,
        0.099,
        "SPY and QQQ include adjusted-price dividends; S&P 500 (^GSPC) is a price-only index. The SPY and index lines nearly overlap.",
        fontsize=10,
    )
    fig.text(
        0.085,
        0.071,
        "Retrospective experiment: fixed 417-name roster has survivorship bias. Model fitted in November 2026 with labels ending in 2025.",
        fontsize=10,
    )
    fig.text(
        0.085,
        0.043,
        "The model predicts five trading sessions; holiday-week targets can extend beyond Friday. No training, broker orders, or promotion.",
        fontsize=10,
    )
    fig.text(0.085, 0.016, "© Thoughtworks 2026   |   Confidential", fontsize=8)
    fig.savefig(ROOT / "weekly_performance_2026.png", dpi=180)
    fig.savefig(ROOT / "weekly_performance_2026.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
