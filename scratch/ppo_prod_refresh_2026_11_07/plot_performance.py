"""Chart saved candidate test returns and an explicitly retrospective week."""

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.ppo_discovery.evaluator import evaluate_policy_weeks
from brain_api.core.ppo_discovery.weeks import weekly_trade_clock
from brain_api.core.sac.experience_accounting import build_rebalance_arrays
from brain_api.core.sac.trade_clock import extract_session_open_prices

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODEL = REPO / "brain_api/data/models/ppo_discovery_halal_new/v2026-11-07-91ba085dd8c6"
PROMOTED = json.loads((MODEL / "metadata.json").read_text()).get("promoted") is True
CAPITAL = 10_000.0
COLORS = {"PPO": "#003D4F", "SPY": "#F2617A", "QQQ": "#CC850A"}
GRID = "#EDF1F3"
FONTS = ROOT / "data/chart_fonts"
for font in FONTS.glob("*-static.ttf"):
    fm.fontManager.addfont(font)
TITLE = fm.FontProperties(fname=FONTS / "Bitter-static.ttf")
plt.rcParams.update(
    {
        "font.family": "Inter",
        "font.size": 11,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRID,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
    }
)


def prices(symbol):
    frame = pd.read_csv(
        ROOT / "data/chart_prices" / f"{symbol}.csv", index_col="date", parse_dates=True
    )
    assert not frame.index.has_duplicates
    return frame


def entry_cost(weights, opens):
    symbols, current, target, values = build_rebalance_arrays(
        {"CASH": 1.0}, weights, opens
    )
    return compute_ibkr_rebalance_cost(
        symbol_order=symbols,
        current_weights=current,
        target_weights=target,
        prices=values,
        cfg=IBKRSingaporeCostConfig.default().with_nav(CAPITAL),
        weight_epsilon=1e-9,
    ).total_fraction


def save(fig, name):
    fig.savefig(ROOT / f"{name}.png", dpi=170)
    fig.savefig(ROOT / f"{name}.svg")
    plt.close(fig)


def week():
    allocation_file = (
        "promoted_http_inference.json"
        if PROMOTED
        else "local_candidate_http_inference.json"
    )
    allocation = json.loads((ROOT / allocation_file).read_text())
    assert allocation["model_version"] == MODEL.name
    weights = allocation["percentage_weights"]
    sessions = pd.bdate_range("2026-11-02", "2026-11-06")
    series, metrics, contributions = {}, {}, []
    gross = np.ones(len(sessions))
    opens = {}
    for symbol, weight in weights.items():
        if symbol == "CASH":
            continue
        frame = prices(symbol)
        closes = frame["close"].reindex(sessions).to_numpy()
        open_ = float(frame.loc[sessions[0], "open"])
        assert np.all(np.isfinite(closes)) and np.all(closes > 0) and open_ > 0
        returns = closes / open_ - 1
        gross += weight * returns
        opens[symbol] = open_
        contributions.append(
            {
                "symbol": symbol,
                "weight": weight,
                "monday_open": open_,
                "friday_close": float(closes[-1]),
                "stock_return": float(returns[-1]),
                "portfolio_contribution": float(weight * returns[-1]),
            }
        )
    cost = entry_cost(weights, opens)
    series["PPO"] = np.r_[CAPITAL, CAPITAL * (gross - cost)]
    metrics["PPO"] = {
        "return": float(gross[-1] - cost - 1),
        "gross_return": float(gross[-1] - 1),
        "entry_cost_usd": float(cost * CAPITAL),
        "ending_value": float(series["PPO"][-1]),
    }
    assert np.isclose(
        sum(r["portfolio_contribution"] for r in contributions), gross[-1] - 1
    )
    for symbol in ["SPY", "QQQ"]:
        frame = prices(symbol)
        start = float(frame.loc[sessions[0], "open"])
        closes = frame["close"].reindex(sessions).to_numpy()
        assert np.all(np.isfinite(closes)) and np.all(closes > 0) and start > 0
        fee = entry_cost({symbol: 1.0, "CASH": 0.0}, {symbol: start})
        series[symbol] = np.r_[CAPITAL, CAPITAL * (closes / start - fee)]
        metrics[symbol] = {
            "return": float(closes[-1] / start - fee - 1),
            "entry_cost_usd": float(fee * CAPITAL),
            "ending_value": float(series[symbol][-1]),
        }
    result = {
        "start": "2026-11-02 open",
        "end": "2026-11-06 close",
        "starting_capital": CAPITAL,
        "retrospective_not_live": True,
        "lookahead_monday_close_in_input": True,
        "cash_weight": weights["CASH"],
        "metrics": metrics,
        "ppo_minus_spy_pp": (metrics["PPO"]["return"] - metrics["SPY"]["return"]) * 100,
        "ppo_minus_qqq_pp": (metrics["PPO"]["return"] - metrics["QQQ"]["return"]) * 100,
        "entry_cost_assumption": "Start entirely in cash; normal IBKR Singapore Tiered model; no exit trade",
        "dividends": "Yahoo adjusted OHLC; fractional notional positions; no cash interest",
    }
    (ROOT / "weekly_performance_summary.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(contributions).sort_values("portfolio_contribution").to_csv(
        ROOT / "weekly_stock_contributions.csv", index=False
    )
    pd.DataFrame(
        series,
        index=[
            "Mon open",
            "Mon close",
            "Tue close",
            "Wed close",
            "Thu close",
            "Fri close",
        ],
    ).to_csv(ROOT / "weekly_portfolio_values.csv")

    fig = plt.figure(figsize=(12, 10.4))
    fig.text(
        0.08,
        0.955,
        "What these weights would have done this week",
        fontproperties=TITLE,
        fontsize=22,
        color=COLORS["PPO"],
    )
    fig.text(
        0.08,
        0.918,
        "2–6 November 2026  |  $10,000 starting capital  |  21.02% held in cash",
        fontsize=11,
    )
    fig.text(
        0.08,
        0.884,
        "RETROSPECTIVE ILLUSTRATION — allocation generated later and includes Monday's close",
        fontsize=10,
        weight="bold",
    )
    for x, symbol, label in zip(
        [0.08, 0.395, 0.71],
        ["PPO", "SPY", "QQQ"],
        [
            "PPO (promoted)" if PROMOTED else "PPO candidate",
            "SPY · S&P 500",
            "QQQ · Nasdaq-100",
        ],
        strict=True,
    ):
        m = metrics[symbol]
        fig.text(x, 0.829, label, color=COLORS[symbol], fontsize=12, weight="bold")
        fig.text(
            x,
            0.794,
            f"{m['return']:+.2%}   ${m['ending_value']:,.0f}",
            fontsize=19,
            fontproperties=TITLE,
        )
    ax = fig.add_axes([0.08, 0.445, 0.85, 0.305])
    for symbol in ["PPO", "SPY", "QQQ"]:
        ax.plot(
            range(6),
            series[symbol],
            color=COLORS[symbol],
            lw=2.7,
            linestyle={"PPO": "-", "SPY": "--", "QQQ": ":"}[symbol],
            marker="o",
            markersize=4,
            label=symbol,
        )
    ax.set_xticks(
        range(6),
        ["Mon open", "Mon close", "Tue close", "Wed close", "Thu close", "Fri close"],
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Portfolio value")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="lower left", ncol=3)

    rows = sorted(
        contributions, key=lambda r: abs(r["portfolio_contribution"]), reverse=True
    )
    selected = rows[:7]
    selected.append(
        {
            "symbol": "Other 7 stocks",
            "portfolio_contribution": sum(
                r["portfolio_contribution"] for r in rows[7:]
            ),
        }
    )
    selected.sort(key=lambda r: r["portfolio_contribution"])
    bar = fig.add_axes([0.15, 0.135, 0.7, 0.22])
    vals = [r["portfolio_contribution"] * 100 for r in selected]
    bar.barh(
        [r["symbol"] for r in selected],
        vals,
        color=[COLORS["SPY"] if v < 0 else COLORS["PPO"] for v in vals],
    )
    bar.invert_yaxis()
    bar.axvline(0, color=COLORS["PPO"], linewidth=0.7)
    bar.set_xlabel(
        "Contribution to portfolio return (percentage points, before entry costs)",
        fontsize=10,
    )
    bar.set_title(
        "Largest contributors to the portfolio's move", loc="left", fontsize=12, pad=10
    )
    bar.grid(axis="x", color=GRID, linewidth=1)
    bar.set_axisbelow(True)
    fig.text(
        0.08,
        0.059,
        "This is not a valid Monday forecast test: Monday closing prices entered the retrospective state.",
        fontsize=9,
    )
    fig.text(
        0.08,
        0.038,
        "Source: Yahoo adjusted OHLC. Entry costs modeled for all three portfolios; no exit fee or cash interest.",
        fontsize=8.5,
    )
    fig.text(0.08, 0.016, "© Thoughtworks 2026   |   Confidential", fontsize=8)
    save(fig, "weekly_performance")
    return result


def historical():
    evaluation = json.loads((MODEL / "evaluation.json").read_text())
    metadata = json.loads((MODEL / "metadata.json").read_text())
    logs = np.asarray(evaluation["test_weekly_net_log"], dtype=float)
    cutoffs = (
        pd.to_datetime(
            [row["as_of"] for row in evaluation["portfolio_diagnostics"]["weekly"]],
            utc=True,
        )
        .tz_localize(None)
        .normalize()
    )
    clock = weekly_trade_clock(
        date.fromisoformat(metadata["data_window"]["start"]),
        date.fromisoformat(metadata["end_date"]),
    )
    indices = clock.actor_cutoffs.get_indexer(cutoffs)
    assert (
        len(indices) == len(logs)
        and np.all(indices >= 0)
        and np.all(np.diff(indices) == 1)
    )
    dates = clock.rebalance_sessions[indices[0] : indices[-1] + 2]
    assert len(dates) == len(logs) + 1
    values = {"PPO": CAPITAL * np.r_[1, np.exp(np.cumsum(logs))]}
    metrics = {"PPO": evaluate_policy_weeks(logs)}
    for symbol in ["SPY", "QQQ"]:
        opens = extract_session_open_prices(prices(symbol), dates, symbol=symbol)
        values[symbol] = CAPITAL * opens / opens[0]
        metrics[symbol] = evaluate_policy_weeks(np.diff(np.log(opens)))
    for key in ["cagr", "sharpe", "max_drawdown"]:
        assert np.isclose(metrics["PPO"][key], evaluation["test_" + key], atol=1e-10)
    for symbol in values:
        metrics[symbol]["total_return"] = float(values[symbol][-1] / CAPITAL - 1)
        metrics[symbol]["ending_value"] = float(values[symbol][-1])
    result = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "n_weeks": len(logs),
        "starting_capital": CAPITAL,
        "metrics": metrics,
        "ppo_minus_spy_pp": (
            metrics["PPO"]["total_return"] - metrics["SPY"]["total_return"]
        )
        * 100,
        "survivorship_bias": True,
        "costs": "Saved PPO after modeled costs; benchmarks adjusted buy-and-hold before trading costs",
    }
    (ROOT / "historical_performance_summary.json").write_text(
        json.dumps(result, indent=2)
    )
    pd.DataFrame(values, index=dates).to_csv(
        ROOT / "historical_portfolio_values.csv", index_label="date"
    )
    fig, (ax, dd) = plt.subplots(
        2, 1, figsize=(12, 8.3), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True
    )
    fig.subplots_adjust(left=0.09, right=0.94, top=0.79, bottom=0.15, hspace=0.27)
    fig.text(
        0.09,
        0.95,
        "Historical test: PPO against the market",
        fontproperties=TITLE,
        fontsize=23,
        color=COLORS["PPO"],
    )
    fig.text(
        0.09,
        0.91,
        f"{dates[0]:%d %b %Y}–{dates[-1]:%d %b %Y}  |  {len(logs)} weekly returns  |  $10,000 starting capital",
        fontsize=11,
    )
    for x, symbol in zip([0.09, 0.40, 0.71], ["PPO", "SPY", "QQQ"], strict=True):
        m = metrics[symbol]
        fig.text(
            x,
            0.855,
            f"{symbol}   {m['total_return']:+.1%}   ${m['ending_value']:,.0f}",
            color=COLORS[symbol],
            fontsize=15,
            weight="bold",
        )
    for symbol in values:
        wealth = values[symbol]
        style = {"PPO": "-", "SPY": "--", "QQQ": ":"}[symbol]
        ax.plot(
            dates, wealth, label=symbol, color=COLORS[symbol], lw=2.4, linestyle=style
        )
        drawdown = wealth / np.maximum.accumulate(wealth) - 1
        dd.plot(dates, drawdown, color=COLORS[symbol], lw=1.8, linestyle=style)
    ax.set_ylabel("Portfolio value")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(frameon=False, loc="upper left", ncol=3)
    dd.set_ylabel("Drawdown from peak")
    dd.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for axis in [ax, dd]:
        axis.grid(axis="y", color=GRID, linewidth=1)
    fig.text(
        0.09,
        0.085,
        "Historical simulation with current-roster survivorship bias; "
        + (
            "model is promoted by explicit override."
            if PROMOTED
            else "candidate is not promoted."
        ),
        fontsize=9,
    )
    fig.text(
        0.09,
        0.06,
        "PPO uses saved net returns after modeled costs. SPY/QQQ use adjusted open-to-open buy-and-hold before trading costs.",
        fontsize=8.5,
    )
    fig.text(0.09, 0.025, "© Thoughtworks 2026   |   Confidential", fontsize=8)
    save(fig, "historical_performance")
    return result


if __name__ == "__main__":
    print(json.dumps({"week": week(), "historical": historical()}, indent=2))
