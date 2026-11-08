"""Frozen pre-2026 PPO: causal Monday decisions, rolling account and Friday marks."""

import hashlib
import json
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import torch

from book import PortfolioBook, buy_and_hold_book

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.news_adapter import load_historical_ppo_news_features
from brain_api.core.ppo_discovery.policy import (
    PPODiscoveryActorCritic,
    validate_inference_weights,
)
from brain_api.core.ppo_discovery.regime import (
    live_regime_probabilities,
    spy_vix_rows_after_cutoff,
)
from brain_api.core.ppo_discovery.schemas import UniverseSnapshot
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    weekly_trade_clock,
)
from brain_api.core.sac.market_sessions import completed_xnys_session_dates
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact
from brain_api.core.vix_fallback import apply_cboe_vix_fallback
from brain_api.core.weekly_decision import monday_decision_cutoff
from brain_api.news.store import NewsStore

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = REPO / "scratch/ppo_allocator_learning_2026_11_06"
INITIAL_NAV = 10_000.0
END = date(2026, 11, 6)


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def quote(prices, symbol, session, column):
    if symbol not in prices or session not in prices[symbol].index:
        raise ValueError(f"Missing {column} for {symbol} on {session.date()}")
    value = float(prices[symbol].loc[session, column])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"Nonpositive/nonfinite {column} for {symbol} on {session.date()}"
        )
    return value


def past_frames(prices, decision_date):
    """All features stop before the pre-open decision date; never forward-fill."""
    return {s: f.loc[f.index.date < decision_date].copy() for s, f in prices.items()}


def schedule():
    clock = weekly_trade_clock(date(2026, 1, 5), END)
    calendar = xcals.get_calendar("XNYS")
    all_sessions = calendar.sessions_in_range("2025-12-19", str(END))
    rows = []
    for execution, actor in zip(
        clock.rebalance_sessions, actor_cutoff_datetimes(clock), strict=True
    ):
        monday = execution.date() - timedelta(days=execution.date().weekday())
        as_of = monday_decision_cutoff(monday)
        prior = all_sessions[all_sessions.date < monday][-1]
        week = all_sessions[
            (all_sessions.date >= monday)
            & (all_sessions.date <= monday + timedelta(days=4))
        ]
        assert execution == week[0]
        rows.append(
            {
                "decision": as_of,
                "actor": actor,
                "execution": execution,
                "prior_session": prior,
                "mark": week[-1],
            }
        )
    assert len(rows) == 44 and rows[0]["execution"] == pd.Timestamp("2026-01-05")
    assert rows[-1]["mark"] == pd.Timestamp("2026-11-06")
    return rows


def main():
    torch.set_num_threads(4)
    identity = json.loads((ROOT / "model_identity.json").read_text())
    for name, expected in identity["files"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
    manifest = json.loads((SOURCE / "market_training_manifest.json").read_text())
    assert manifest["data"]["last_training_outcome"][:10] <= "2025-12-31"
    config = PPODiscoveryConfig.from_dict(manifest["config"])
    policy = PPODiscoveryActorCritic(config)
    policy.load_state_dict(
        torch.load(
            SOURCE / "market_seed42_final.pt", map_location="cpu", weights_only=True
        )
    )
    policy.eval()
    scalers = json.loads((SOURCE / "market_scalers.json").read_text())
    hmm_payload = json.loads((SOURCE / "market_hmm.json").read_text())
    hmm = RegimeHMMArtifact.from_dict(hmm_payload)
    assert hmm.training_cutoff_date <= date(2025, 12, 31)
    raw = json.loads((SOURCE / "market_roster.json").read_text())
    raw["sorted_symbols"] = tuple(raw["sorted_symbols"])
    snapshot = UniverseSnapshot(**raw)
    with (ROOT / "data/prices.pkl").open("rb") as file:
        prices = pickle.load(file)
    for frame in prices.values():
        index = (
            frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
        )
        frame.index = index.normalize()
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
    plan = schedule()
    required = completed_xnys_session_dates(
        hmm.training_cutoff_date + timedelta(days=1), date(2026, 11, 7)
    )
    fixed = apply_cboe_vix_fallback(prices, required_dates=required)
    prices = fixed.prices
    write("vix_provenance.json", fixed.audit.to_dict())
    print("Loading complete news for 44 Mondays / 432 symbols", flush=True)
    news = load_historical_ppo_news_features(
        [r["actor"] for r in plan],
        snapshot.sorted_symbols,
        store=NewsStore(ROOT / "data"),
    )
    book = PortfolioBook(INITIAL_NAV)
    benchmarks, benchmark_fees = {}, {}
    for symbol in ["SPY", "QQQ"]:
        benchmarks[symbol], benchmark_fees[symbol] = buy_and_hold_book(
            symbol, quote(prices, symbol, plan[0]["execution"], "open"), INITIAL_NAV
        )
    nav_rows = [
        {
            "date": "2026-01-05",
            "PPO": INITIAL_NAV,
            "SPY": INITIAL_NAV,
            "QQQ": INITIAL_NAV,
        }
    ]
    weeks, trade_rows = [], []
    fees_total = 0.0
    causal_check = None
    for index, row in enumerate(plan):
        as_of, prior = row["decision"], row["prior_session"]
        # Decision portfolio weights use only completed-session closes.
        decision_marks = {s: quote(prices, s, prior, "close") for s in book.shares}
        decision_nav = book.value(decision_marks)
        current_weights = book.weights(decision_marks)
        sliced = past_frames(prices, as_of.date())
        regime_rows = spy_vix_rows_after_cutoff(
            prices, cutoff=hmm.training_cutoff_date, decision_date=as_of.date()
        )
        assert all(
            datetime.fromisoformat(r["date"]).date() < as_of.date() for r in regime_rows
        )
        calm, stress = live_regime_probabilities(
            hmm_payload, spy_vix_rows=regime_rows, decision_date=as_of.date()
        )

        def build(frames):
            return build_ppo_discovery_state(
                StateBuildRequest(
                    as_of=as_of,
                    universe_snapshot=snapshot,
                    ohlcv_by_symbol=frames,
                    news_by_symbol=news[row["actor"]],
                    current_weights=current_weights,
                    p_calm=calm,
                    p_stress=stress,
                    spy_closes=frames["SPY"]["close"].to_numpy(),
                    feature_scalers=scalers,
                    market_history_provenance={
                        "completed_session": str(prior.date()),
                        "decision_price_cutoff_strict": True,
                    },
                )
            )

        state = build(sliced)
        eligible = {
            s for s, mask in zip(state.symbols, state.asset_mask, strict=True) if mask
        }
        with torch.no_grad():
            weights, order = policy.infer_decision(state)
        weights = validate_inference_weights(weights, eligible)
        if index == 0:
            # Perturb future prices and prove they cannot enter the first state.
            perturbed = {s: f.copy() for s, f in prices.items()}
            for frame in perturbed.values():
                frame.loc[
                    frame.index.date >= as_of.date(), ["open", "high", "low", "close"]
                ] *= 7
            check = build(past_frames(perturbed, as_of.date()))
            assert check.state_digest == state.state_digest
            with torch.no_grad():
                same_weights, _ = policy.infer_decision(check)
            assert same_weights == weights
            causal_check = {
                "future_price_multiplier": 7,
                "state_digest_unchanged": True,
                "weights_unchanged": True,
            }
        # Only now, after freezing the allocation, observe the execution opens.
        needed = set(book.shares) | {
            s for s, w in weights.items() if s != "CASH" and w > 0
        }
        opens = {s: quote(prices, s, row["execution"], "open") for s in needed}
        book, rebalance = book.rebalance(
            weights, opens, weight_epsilon=config.rebalance_weight_epsilon
        )
        fees_total += rebalance["fees_usd"]
        marks = {s: quote(prices, s, row["mark"], "close") for s in book.shares}
        friday_nav = book.value(marks)
        values = {"PPO": friday_nav}
        for symbol, benchmark in benchmarks.items():
            values[symbol] = benchmark.value(
                {symbol: quote(prices, symbol, row["mark"], "close")}
            )
        nav_rows.append({"date": str(row["mark"].date()), **values})
        for trade in rebalance["trades"]:
            trade_rows.append(
                {
                    "week": index + 1,
                    "decision": as_of.isoformat(),
                    "execution_session": str(row["execution"].date()),
                    **trade,
                }
            )
        weeks.append(
            {
                "week": index + 1,
                "decision": as_of.isoformat(),
                "last_feature_session": str(prior.date()),
                "execution_session": str(row["execution"].date()),
                "friday_mark_session": str(row["mark"].date()),
                "decision_nav": decision_nav,
                "decision_current_weights": current_weights,
                "state_digest": state.state_digest,
                "eligible_assets": len(eligible),
                "selection_order": list(order),
                "target_weights": weights,
                "monday_nav_before": rebalance["nav_before"],
                "monday_nav_after_fees": rebalance["nav_after"],
                "fees_usd": rebalance["fees_usd"],
                "fees_ytd_usd": fees_total,
                "friday_nav": friday_nav,
                "friday_cash": book.cash,
                "friday_weights": book.weights(marks),
                "positions": dict(book.shares),
                "fee_breakdown": rebalance["fee_breakdown"],
            }
        )
        # The pre-decision NAV is the preceding Friday mark, before any weekend gap.
        assert np.isclose(
            decision_nav,
            INITIAL_NAV if index == 0 else weeks[-2]["friday_nav"],
            atol=1e-7,
        )
        progress = {
            "weeks_completed": index + 1,
            "total_weeks": len(plan),
            "mark": str(row["mark"].date()),
            "nav": friday_nav,
            "fees_ytd": fees_total,
            "selected": list(order),
        }
        write("progress.json", progress)
        print(json.dumps(progress), flush=True)
    frame = pd.DataFrame(nav_rows).set_index("date")
    frame.to_csv(ROOT / "weekly_nav.csv")
    pd.DataFrame(trade_rows).to_csv(ROOT / "trades.csv", index=False)
    write("weekly_ledger.json", weeks)
    final_date = plan[-1]["mark"]
    final_nav = nav_rows[-1]["PPO"]
    holdings = [
        {
            "symbol": s,
            "shares": q,
            "close_price": quote(prices, s, final_date, "close"),
            "market_value": q * quote(prices, s, final_date, "close"),
        }
        for s, q in book.shares.items()
    ]
    holdings.sort(key=lambda r: r["market_value"], reverse=True)
    for holding in holdings:
        holding["weight"] = holding["market_value"] / final_nav
    assert np.isclose(sum(h["market_value"] for h in holdings) + book.cash, final_nav)
    assert np.isclose(sum(t["total"] for t in trade_rows), fees_total)
    metrics = {}
    for name in ["PPO", "SPY", "QQQ"]:
        curve = frame[name].to_numpy()
        metrics[name] = {
            "ending_nav": float(curve[-1]),
            "total_return": float(curve[-1] / INITIAL_NAV - 1),
            "weekly_max_drawdown": float(
                np.max(1 - curve / np.maximum.accumulate(curve))
            ),
            "fees_usd": fees_total if name == "PPO" else benchmark_fees[name],
        }
    pd.DataFrame(holdings).to_csv(ROOT / "ending_holdings.csv", index=False)
    for name, expected in identity["files"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
    current = (
        (REPO / "brain_api/data/models/ppo_discovery_halal_new/current")
        .read_text()
        .strip()
    )
    assert current == identity["production_current_before"]
    source_news = REPO / "brain_api/data/news/news_v1.duckdb"
    assert source_news.stat().st_size == identity["source_news_stat_before"]["size"]
    assert (
        source_news.stat().st_mtime_ns
        == identity["source_news_stat_before"]["mtime_ns"]
    )
    result = {
        "as_of": "2026-11-07",
        "last_valuation_session": str(final_date.date()),
        "first_execution": "2026-01-05",
        "initial_nav": INITIAL_NAV,
        "weeks": len(weeks),
        "metrics": metrics,
        "holdings": holdings,
        "cash": book.cash,
        "model_checkpoint_sha256": identity["files"]["market_seed42_final.pt"],
        "model_training_cutoff": "2025-12-31",
        "last_training_outcome": identity["last_training_outcome"],
        "causality_check": causal_check,
        "cost_model": "Configured IBKR Singapore Tiered with actual rolling NAV per rebalance",
        "book_convention": "Friday marks, positions held across weekend, net delta rebalances at next week's first XNYS open; fees deducted from cash",
        "price_convention": "Adjusted OHLCV; fractional positions; dividend-reinvestment approximation; no slippage, tax or cash interest; no final liquidation",
        "survivorship_bias": True,
        "roster_retrieved_at": snapshot.retrieved_at,
        "production_current_unchanged": current,
        "production_news_unchanged": True,
        "pi_accessed": False,
        "orders_submitted": False,
    }
    write("summary.json", result)
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
