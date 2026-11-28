"""Add causal weekly one-stage HRP to the frozen experiment rosters."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain_api.core.hrp import compute_hrp_weights
from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.portfolio_rl.eval import compute_cagr, compute_sharpe_ratio
from brain_api.core.portfolio_rl.rewards import RebalanceTransition
from brain_api.core.sac.market_sessions import xnys_session_dates

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXPERIMENTS = (
    ("halal", "sac_halal_rewards_2026_11_28"),
    ("halal_filtered", "sac_halal_filtered_spy_rewards_2026_11_28"),
)


def target_from_history(closes, decision, *, cash_buffer=0.02, lookback=252):
    """Only the prior calendar Friday and earlier bars may affect the target."""
    decision = pd.Timestamp(decision).normalize()
    cutoff = decision - pd.Timedelta(days=decision.weekday() + 3)
    history = closes.loc[closes.index <= cutoff].tail(lookback + 1)
    if len(history) != lookback + 1:
        raise ValueError(f"Need {lookback + 1} complete prior closes at {decision}")
    values = history.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("HRP history must be complete and positive for every stock")
    returns = history.pct_change(fill_method=None).iloc[1:]
    weights = compute_hrp_weights(returns)
    if set(weights) != set(closes.columns):
        raise ValueError("HRP may not silently exclude a frozen-roster stock")
    risky = np.asarray([weights[symbol] for symbol in closes.columns])
    if (
        not np.all(np.isfinite(risky))
        or np.any(risky < 0)
        or not np.isclose(risky.sum(), 1)
    ):
        raise ValueError("HRP must produce a finite nonnegative stock simplex")
    target = np.r_[risky * (1 - cash_buffer), cash_buffer]
    evidence = {
        "actor_cutoff": cutoff.date().isoformat(),
        "history_start": history.index[0].date().isoformat(),
        "history_end": history.index[-1].date().isoformat(),
        "daily_returns_used": len(returns),
    }
    return target, evidence


def run_hrp(universe, directory):
    folder = REPO / "scratch" / directory
    manifest = json.loads((folder / "evidence/manifest.json").read_text())
    price_path = folder / "evidence/adjusted_ohlcv.parquet"
    if hashlib.sha256(price_path.read_bytes()).hexdigest() != manifest["price_sha256"]:
        raise ValueError("Frozen experiment prices have changed")
    raw = pd.read_parquet(price_path)
    symbols = manifest["symbols"]
    closes = pd.DataFrame({symbol: raw[symbol]["Close"] for symbol in symbols})
    # Yahoo's combined panel includes a few VIX-only, non-XNYS dates. Count
    # actual stock-market sessions, retaining any genuine missing stock bar
    # so strict history validation can reject it.
    closes = closes.reindex(
        pd.DatetimeIndex(
            xnys_session_dates(closes.index.min().date(), closes.index.max().date())
        )
    )
    reference = pd.read_csv(folder / "results_50000/benchmarks.csv")
    sessions = pd.to_datetime([*reference.session, manifest["test_end"]])
    opens = np.column_stack(
        [
            raw[symbol]["Open"].reindex(sessions).to_numpy(dtype=float)
            for symbol in symbols
        ]
    )
    if not np.all(np.isfinite(opens)) or np.any(opens <= 0):
        raise ValueError("Every stock requires a positive open at every rebalance")
    weekly_returns = opens[1:] / opens[:-1] - 1
    config = json.loads(
        (folder / "results_50000/current/seed_42/metrics.json").read_text()
    )["config"]
    costs = IBKRSingaporeCostConfig.from_dict(config["cost_config"])
    current = np.r_[np.zeros(len(symbols)), 1.0]
    rows = []
    for week, session in enumerate(sessions[:-1]):
        target, evidence = target_from_history(
            closes, session, cash_buffer=config["cash_buffer"]
        )
        cost = compute_ibkr_rebalance_cost(symbols, current, target, opens[week], costs)
        transition = RebalanceTransition.calculate(
            target, weekly_returns[week], cost.total_fraction
        )
        rows.append(
            {
                "session": session.date().isoformat(),
                "end_session": sessions[week + 1].date().isoformat(),
                "net_return": transition.net_growth - 1,
                "gross_return": transition.gross_return,
                "cost_usd": cost.total_usd,
                "turnover": float(np.abs(target - current).sum() / 2),
                "cash": float(target[-1]),
                **evidence,
                **{
                    f"weight_{symbol}": float(target[index])
                    for index, symbol in enumerate(symbols)
                },
            }
        )
        current = transition.post_weights
    frame = pd.DataFrame(rows)
    if len(frame) != 104 or not frame.session.equals(reference.session):
        raise ValueError("HRP and original curves must align over all 104 weeks")
    net = frame.net_return.to_numpy()
    wealth = np.r_[1.0, np.cumprod(1 + net)]
    report = {
        "run_id": f"scratch:hrp-252:{universe}:2026-11-28",
        "attempt": 1,
        "as_of": "2026-11-28",
        "universe": universe,
        "symbols": symbols,
        "lookback_daily_returns": 252,
        "weekly_periods": 104,
        "test_start": manifest["test_start"],
        "test_end": manifest["test_end"],
        "cash_buffer": config["cash_buffer"],
        "cost_config": costs.to_dict(),
        "cost_weight_epsilon": 0.005,
        "cagr": compute_cagr(net),
        "sharpe": compute_sharpe_ratio(net),
        "max_drawdown": float((1 - wealth / np.maximum.accumulate(wealth)).max()),
        "final_value_usd": float(wealth[-1] * 10_000),
        "fees_usd_fixed_nav": float(frame.cost_usd.sum()),
        "mean_weekly_turnover": float(frame.turnover.mean()),
        "price_sha256": manifest["price_sha256"],
        "signal_policy": "252 adjusted close returns through prior calendar Friday; one-stage HRP on all roster stocks",
    }
    frame.to_csv(ROOT / f"{universe}_hrp_252_returns.csv", index=False)
    (ROOT / f"{universe}_hrp_252_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    for experiment in EXPERIMENTS:
        run_hrp(*experiment)
