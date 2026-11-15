"""Compare the saved V1 research replay with production on common dates."""

import json

import numpy as np
import pandas as pd
from build_friday_dataset import plt
from compare_patchtst import write_json
from fetch_v1_production_comparison import OUTPUT, START
from matplotlib.ticker import PercentFormatter
from train_v1_loss_comparison import RESULT as REPLAY
from v1_holdout_portfolio import INITIAL_NAV

AUDIT = OUTPUT.parents[1] / "alpha_hrp_audit_2026_10_03/analysis.json"


def replay_close_nav():
    marks = pd.read_csv(REPLAY / "h25_rank_ibkr/nav.csv", parse_dates=["date"])
    close = marks.loc[marks.arm.eq("v1") & marks.mark.eq("close")]
    return close.pivot(index="date", columns="weighting", values="nav")


def target_comparison(actual, replay):
    audit = json.loads(AUDIT.read_text())
    picks = pd.read_csv(
        REPLAY / "multitarget/picks_and_weights.csv", parse_dates=["friday"]
    )
    picks = picks.loc[picks.arm.eq("v1") & picks.weighting.eq("hrp")].set_index(
        ["friday", "symbol"]
    )
    marks = pd.read_csv(OUTPUT / "adjusted_close.csv", parse_dates=["date"]).set_index(
        "date"
    )
    rows = []
    for report in audit["weeks"]:
        scheduled = pd.Timestamp(report["week"])
        start = scheduled - pd.Timedelta(days=3)
        end = start + pd.Timedelta(days=7)
        production = pd.Series(report["weights"], dtype=float)
        production /= production.sum()
        research = picks.loc[start, "weight"]
        returns = marks.loc[end] / marks.loc[start] - 1
        # Audit returns cover the September roster, including names outside
        # the frozen August research archive.
        audited = pd.Series(report["selected_actual_calendar_week"], dtype=float) / 100
        returns = returns.combine_first(audited)
        target = returns.reindex(production.index)
        refit = returns.reindex(research.index)
        if not np.isfinite(target).all() or not np.isfinite(refit).all():
            raise ValueError(f"Missing target-basket return for {end.date()}")
        rows.append(
            {
                "production_checkpoint": audit["model"],
                "week_ending": end,
                "names_in_both_baskets": len(
                    production.index.intersection(research.index)
                ),
                "production_only": ",".join(
                    sorted(production.index.difference(research.index))
                ),
                "replay_only": ",".join(
                    sorted(research.index.difference(production.index))
                ),
                "production_target_at_prior_close_gross": production.dot(target),
                "refit_target_at_prior_close_gross": research.dot(refit),
                "refit_hrp_after_cost": replay.loc[end, "hrp"]
                / replay.loc[start, "hrp"]
                - 1,
                "actual_alpaca": actual.loc[end, "v1"] / actual.loc[start, "v1"] - 1,
                "spy": actual.loc[end, "spy"] / actual.loc[start, "spy"] - 1,
            }
        )
    return pd.DataFrame(rows)


def main():
    actual = pd.read_csv(
        OUTPUT / "daily_comparison.csv", parse_dates=["date"]
    ).set_index("date")
    replay = replay_close_nav()
    common = actual[["v1", "spy"]].join(replay, how="inner").loc[START:]
    if common.index[0] != START or not np.isfinite(common).all().all():
        raise ValueError("Missing evidence on the comparison grid")
    growth = common.div(common.iloc[0])
    growth.to_csv(OUTPUT / "matched_replay_daily.csv")
    targets = target_comparison(actual, replay)
    targets.to_csv(OUTPUT / "matched_replay_targets.csv", index=False)
    summary = {
        "matched_start": START.date().isoformat(),
        "matched_end": common.index[-1].date().isoformat(),
        "matched_return": (growth.iloc[-1] - 1).to_dict(),
        "research_hrp_return_through_switch_baseline": replay.loc[START, "hrp"]
        / INITIAL_NAV
        - 1,
        "target_weeks": len(targets),
        "mean_top15_overlap": targets.names_in_both_baskets.mean(),
        "target_comparison": (
            (1 + targets.loc[:, "production_target_at_prior_close_gross":]).prod() - 1
        ).to_dict(),
    }
    write_json(OUTPUT / "matched_replay_summary.json", summary)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    labels = {
        "hrp": ("Research refit · HRP", "#92562d"),
        "v1": ("Production · Alpaca paper", "#315c88"),
        "spy": ("SPY", "#4a7754"),
    }
    for name, (label, color) in labels.items():
        ax.plot(growth.index, growth[name] - 1, label=label, color=color, linewidth=2)
    ax.set_title(
        f"V1 research vs. production on matched dates · {growth.index[0]:%b %d}–{growth.index[-1]:%b %d, %Y}"
    )
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "matched_replay.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
