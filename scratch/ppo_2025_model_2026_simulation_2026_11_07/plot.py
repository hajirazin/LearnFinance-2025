"""Publication-ready weekly account values for the frozen 2025 model."""

import json
from datetime import datetime
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
FONTS = ROOT.parent / "ppo_prod_refresh_2026_11_07/data/chart_fonts"
for path in FONTS.glob("*-static.ttf"):
    fm.fontManager.addfont(path)
TITLE = fm.FontProperties(fname=FONTS / "Bitter-static.ttf")
COLORS = {"PPO": "#003D4F", "SPY": "#F2617A", "QQQ": "#CC850A"}
STYLES = {"PPO": "-", "SPY": "--", "QQQ": ":"}
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


def main():
    result = json.loads((ROOT / "summary.json").read_text())
    values = pd.read_csv(ROOT / "weekly_nav.csv", index_col="date", parse_dates=True)
    assert len(values) == result["weeks"] + 1 == 45
    fig, (ax, dd) = plt.subplots(
        2, 1, figsize=(12, 8.8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    fig.subplots_adjust(left=0.09, right=0.94, top=0.76, bottom=0.205, hspace=0.24)
    fig.text(
        0.09,
        0.95,
        "2026: $10,000 with the frozen PPO model",
        fontproperties=TITLE,
        fontsize=24,
        color=COLORS["PPO"],
    )
    fig.text(
        0.09,
        0.909,
        "5 January–6 November 2026  |  44 weekly rebalances  |  No retraining on 2026",
        fontsize=11,
    )
    for x, name, description in zip(
        [0.09, 0.40, 0.71],
        COLORS,
        ["PPO · trained through 2025", "SPY · S&P 500", "QQQ · Nasdaq-100"],
        strict=True,
    ):
        metric = result["metrics"][name]
        fig.text(x, 0.858, description, fontsize=11)
        fig.text(
            x,
            0.815,
            f"${metric['ending_nav']:,.2f}   {metric['total_return']:+.2%}",
            fontproperties=TITLE,
            fontsize=19,
            color=COLORS[name],
        )
    for name in COLORS:
        curve = values[name].to_numpy()
        assert np.isclose(
            curve[-1], result["metrics"][name]["ending_nav"], rtol=1e-12, atol=1e-9
        )
        ax.plot(
            values.index,
            curve,
            label=name,
            color=COLORS[name],
            linestyle=STYLES[name],
            linewidth=2.3,
            marker="o",
            markersize=3,
        )
        drawdown = curve / np.maximum.accumulate(curve) - 1
        dd.plot(
            values.index,
            drawdown,
            color=COLORS[name],
            linestyle=STYLES[name],
            linewidth=1.8,
        )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.set_ylabel("Account value after charges")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    dd.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    dd.set_ylabel("Drawdown\n(weekly marks)")
    dd.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(datetime(2026, 1, 1), datetime(2026, 11, 10))
    for axis in [ax, dd]:
        axis.grid(axis="y", color="#EDF1F3", linewidth=1)
    fig.text(
        0.09,
        0.144,
        f"PPO fees: ${result['metrics']['PPO']['fees_usd']:.2f}.   Ending cash: ${result['cash']:,.2f}.   Holdings: BSX, INTC, NVDA, SNDK.",
        fontsize=11,
    )
    fig.text(
        0.09,
        0.112,
        "Weekly points mark Friday close, or the week's last trading day. Holdings carry into Monday delta rebalances.",
        fontsize=9,
    )
    fig.text(
        0.09,
        0.087,
        "IBKR Singapore Tiered charges use the account's changing NAV. Benchmarks buy and hold after their entry charge.",
        fontsize=9,
    )
    fig.text(
        0.09,
        0.062,
        "Retrospective simulation: fixed September-2026 roster creates survivorship bias. Adjusted prices; fractional positions.",
        fontsize=9,
    )
    fig.text(
        0.09,
        0.038,
        "No slippage, tax or cash interest; no final liquidation. Frozen model's last training outcome: 29 December 2025.",
        fontsize=8.5,
    )
    fig.text(0.09, 0.014, "© Thoughtworks 2026   |   Confidential", fontsize=8)
    fig.savefig(ROOT / "weekly_performance_2026.png", dpi=180)
    fig.savefig(ROOT / "weekly_performance_2026.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
