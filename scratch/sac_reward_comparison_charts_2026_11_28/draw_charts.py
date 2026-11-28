"""Render the two six-line experiments and their twelve-line comparison."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from chart_data import load_experiment
from matplotlib import font_manager, ticker

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COLORS = ("#003D4F", "#F2617A", "#CC850A", "#6B9E78", "#47A1AD", "#634F7D")


def set_style():
    fonts = Path.home() / "Library/Fonts"
    for name in (
        "Inter-Regular.otf",
        "Inter-SemiBold.otf",
        "Inter-Bold.otf",
        "Bitter-Bold.ttf",
    ):
        font_manager.fontManager.addfont(fonts / name)
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 13,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#EDF1F3",
            "axes.linewidth": 1,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "svg.fonttype": "path",
            "savefig.facecolor": "#FFFFFF",
        }
    )


def export_csv(curves, stem, output_root=ROOT):
    source_dates = curves[0].dates
    if any(curve.dates != source_dates for curve in curves):
        raise ValueError("All chart curves must share the exact same dates")
    with (output_root / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["date", *[f"{curve.universe} | {curve.label}" for curve in curves]]
        )
        for index, day in enumerate(source_dates):
            writer.writerow(
                [day.isoformat(), *[curve.wealth[index] for curve in curves]]
            )


def draw(
    curves, title, subtitle, stem, page_number, *, output_root=ROOT, extra_note=None
):
    combined = len(curves) == 12
    fig, ax = plt.subplots(figsize=(16, 12 if combined else 10.5))
    fig.subplots_adjust(
        left=0.085, right=0.97, top=0.80, bottom=0.34 if combined else 0.30
    )
    handles, labels = [], []
    for curve in curves:
        filtered = curve.universe == "halal_filtered"
        (line,) = ax.plot(
            curve.dates,
            curve.wealth,
            color=(*COLORS, "#000000")[curve.category],
            linewidth=3.0 if curve.category == 6 else 2.25,
            linestyle=(0, (5, 3)) if filtered else "-",
            marker="D" if filtered else None,
            markevery=13,
            markersize=3.2,
        )
        handles.append(line)
        prefix = ("Filtered: " if filtered else "Halal: ") if combined else ""
        seed = f" (s{curve.seed})" if curve.seed is not None else ""
        labels.append(
            f"{prefix}{curve.label}{seed} | ${curve.wealth[-1]:,.0f} | {curve.cagr:.2%} CAGR"
        )
    ax.set_ylabel("Simulated portfolio value (USD)", labelpad=12)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(curves[0].dates[0], curves[0].dates[-1])
    ax.grid(axis="y", color="#EDF1F3", linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=9)
    fig.text(0.085, 0.94, title, fontfamily="Bitter", fontweight="bold", fontsize=25)
    fig.text(0.085, 0.898, subtitle, fontsize=13)
    fig.text(
        0.085,
        0.861,
        "Growth of $10,000 after modeled fees | November 25, 2024 to November 23, 2026",
        fontsize=12,
    )
    if combined:
        fig.text(
            0.085,
            0.824,
            "Solid lines: SAC halal / QQQ     Dashed lines with diamonds: SAC halal_filtered / SPY",
            fontsize=12,
            fontweight="semibold",
        )
    elif extra_note:
        fig.text(0.085, 0.824, extra_note, fontsize=12, fontweight="semibold")
    legend = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.078, 0.285 if combined else 0.245),
        ncol=2,
        frameon=False,
        fontsize=10.5 if combined else 11,
        handlelength=3.6,
        columnspacing=2.5,
        labelspacing=0.9,
    )
    note = fig.text(
        0.085,
        0.086,
        "SAC lines use the median-test-CAGR seed from three 50,000-step runs; they are actual policy paths, not seed averages.",
        fontsize=10.5,
    )
    fig.text(
        0.085,
        0.060,
        "Exploratory: frozen current rosters, later-edited news inputs, and IBKR fees modeled at a fixed $10,000 NAV.",
        fontsize=10.5,
    )
    fig.text(0.085, 0.024, "© Thoughtworks 2026   |   Confidential", fontsize=10)
    fig.text(0.97, 0.024, str(page_number), ha="right", fontsize=10)
    fig.canvas.draw()
    if legend.get_window_extent().overlaps(note.get_window_extent()):
        raise ValueError("Chart legend overlaps its methodology note")
    if len(ax.lines) != len(curves):
        raise ValueError("Chart scenario count is incorrect")
    fig.savefig(output_root / f"{stem}.png", dpi=160)
    fig.savefig(output_root / f"{stem}.svg")
    plt.close(fig)
    export_csv(curves, stem, output_root)


def main():
    set_style()
    halal = load_experiment(
        REPO / "scratch/sac_halal_rewards_2026_11_28", "halal", "QQQ"
    )
    filtered = load_experiment(
        REPO / "scratch/sac_halal_filtered_spy_rewards_2026_11_28",
        "halal_filtered",
        "SPY",
    )
    if len(halal) != 6 or len(filtered) != 6:
        raise ValueError(
            "Expected four reward variants plus two benchmarks per experiment"
        )
    draw(
        halal,
        "SAC halal: six-scenario comparison",
        "13-stock halal universe | Four SAC reward variants plus QQQ and the equal-weight basket",
        "01_sac_halal",
        1,
    )
    draw(
        filtered,
        "SAC: six-scenario comparison",
        "15-stock halal_filtered universe | Four SAC reward variants plus SPY and the equal-weight basket",
        "02_sac_halal_filtered",
        2,
    )
    draw(
        [*halal, *filtered],
        "All 12 scenarios: SAC halal and SAC",
        "Both frozen universes compared over the same 104 weekly transitions",
        "03_all_12_scenarios",
        3,
    )
    records = []
    for curve in [*halal, *filtered]:
        record = asdict(curve)
        record["source"] = str(Path(curve.source).relative_to(REPO))
        record.pop("wealth")
        record["dates"] = [day.isoformat() for day in curve.dates]
        record["final_value_usd"] = curve.wealth[-1]
        records.append(record)
    (ROOT / "series_manifest.json").write_text(json.dumps(records, indent=2))
    print(
        "Generated three charts: 6, 6, and 12 scenario lines; all 105 endpoints align."
    )
    for curve in [*halal, *filtered]:
        print(
            curve.universe,
            curve.label,
            curve.seed,
            f"${curve.wealth[-1]:,.2f}",
            f"CAGR={curve.cagr:.4%}",
            f"Sharpe={curve.sharpe:.4f}",
        )


if __name__ == "__main__":
    main()
