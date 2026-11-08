"""Compare pinned old/new PPO locally, with no model-pointer changes."""

import json
import pickle
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.ppo_discovery.inference import run_ppo_discovery_inference
from brain_api.core.ppo_discovery.regime import (
    live_regime_probabilities,
    spy_vix_rows_after_cutoff,
)
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.sac.experience_accounting import build_rebalance_arrays
from brain_api.core.sac.market_sessions import completed_xnys_session_dates
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact
from brain_api.core.vix_fallback import apply_cboe_vix_fallback
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

ROOT = Path(__file__).resolve().parent
CAPITAL = 10_000.0
DECISION = datetime.fromisoformat("2026-11-02T09:00:00-05:00")
SESSIONS = pd.bdate_range("2026-11-02", "2026-11-06")
VERSIONS = {
    "Old PPO": "v2026-09-12-13a430da1914-seed42",
    "New PPO": "v2026-11-07-91ba085dd8c6",
}


def history_before_decision(frame, decision_date):
    """Exclude the entire decision session and all subsequent evidence."""
    index = frame.index
    if index.tz is not None:
        index = index.tz_localize(None)
    mask = index.normalize().date < decision_date
    past = frame.loc[mask].copy()
    past.index = index[mask].normalize()
    assert past.empty or past.index.max().date() < decision_date
    return past


def evaluate_week(weights, prices):
    """Fresh cash book, fractional notionals, entry fees, Friday valuation."""
    gross = np.ones(len(SESSIONS))
    opens = {}
    contributions = []
    for symbol, weight in weights.items():
        if symbol == "CASH" or weight == 0:
            continue
        frame = prices[symbol]
        open_ = float(frame.loc[SESSIONS[0], "open"])
        closes = frame["close"].reindex(SESSIONS).to_numpy()
        assert np.isfinite(open_) and open_ > 0
        assert np.all(np.isfinite(closes)) and np.all(closes > 0)
        changes = closes / open_ - 1
        gross += weight * changes
        opens[symbol] = open_
        contributions.append(
            {
                "symbol": symbol,
                "weight": weight,
                "open": open_,
                "friday_close": float(closes[-1]),
                "stock_return": float(changes[-1]),
                "portfolio_contribution": float(weight * changes[-1]),
            }
        )
    symbols, prior, target, values = build_rebalance_arrays(
        {"CASH": 1.0}, weights, opens
    )
    cost = compute_ibkr_rebalance_cost(
        symbol_order=symbols,
        current_weights=prior,
        target_weights=target,
        prices=values,
        cfg=IBKRSingaporeCostConfig.default().with_nav(CAPITAL),
        weight_epsilon=1e-9,
    ).total_fraction
    wealth = np.r_[CAPITAL, CAPITAL * (gross - cost)]
    return {
        "return": float(wealth[-1] / CAPITAL - 1),
        "gross_return": float(gross[-1] - 1),
        "ending_value": float(wealth[-1]),
        "entry_cost_usd": float(CAPITAL * cost),
        "cash_weight": float(weights.get("CASH", 0)),
        "k": sum(symbol != "CASH" and weight > 0 for symbol, weight in weights.items()),
        "percentage_weights": weights,
        "wealth": wealth.tolist(),
        "contributions": contributions,
    }


def plot(result):
    import matplotlib.pyplot as plt

    from plot_performance import TITLE, save

    colors = {
        "Old PPO": "#003D4F",
        "New PPO": "#F2617A",
        "SPY": "#CC850A",
        "QQQ": "#6B9E78",
    }
    styles = {"Old PPO": "-", "New PPO": "--", "SPY": ":", "QQQ": "-."}
    fig = plt.figure(figsize=(12, 7.5))
    fig.text(
        0.08,
        0.94,
        "Old versus new PPO: last week's replay",
        fontproperties=TITLE,
        fontsize=24,
        color=colors["Old PPO"],
    )
    fig.text(
        0.08,
        0.89,
        "2–6 November 2026  |  $10,000 each  |  Prices available through Friday, 30 October",
        fontsize=11,
    )
    for x, name in zip([0.08, 0.30, 0.52, 0.74], colors, strict=True):
        row = result["price_cutoff_corrected"][name]
        fig.text(x, 0.82, name, fontsize=12, weight="bold")
        fig.text(
            x,
            0.77,
            f"{row['return']:+.2%}",
            fontproperties=TITLE,
            fontsize=23,
            color=colors[name],
        )
        fig.text(x, 0.73, f"${row['ending_value']:,.0f} ending value", fontsize=10)
    ax = fig.add_axes([0.09, 0.24, 0.85, 0.43])
    for name in colors:
        ax.plot(
            range(6),
            result["price_cutoff_corrected"][name]["wealth"],
            color=colors[name],
            linestyle=styles[name],
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=name,
        )
    ax.set_xticks(
        range(6),
        ["Mon open", "Mon close", "Tue close", "Wed close", "Thu close", "Fri close"],
    )
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(axis="y", color="#EDF1F3", linewidth=1)
    ax.legend(frameon=False, loc="lower left", ncol=4)
    old, new = (
        result["price_cutoff_corrected"][name] for name in ["Old PPO", "New PPO"]
    )
    fig.text(
        0.09,
        0.16,
        f"Old: {old['k']} stocks, {old['cash_weight']:.2%} cash.   New: {new['k']} stocks, {new['cash_weight']:.2%} cash.",
        fontsize=11,
    )
    fig.text(
        0.09,
        0.11,
        "Retrospective, not executed. Both use their own saved scalers and HMM, and the same Monday news window.",
        fontsize=9,
    )
    fig.text(
        0.09,
        0.08,
        "Same 417-stock roster resolved on 7 November; historical Monday roster not recovered. IBKR entry fees included.",
        fontsize=9,
    )
    fig.text(
        0.09,
        0.04,
        "Source: Yahoo adjusted OHLCV. No exit fee or cash interest.   © Thoughtworks 2026   |   Confidential",
        fontsize=8.5,
    )
    save(fig, "old_new_week_comparison")


def main():
    storage = PPODiscoveryHalalNewModelStorage(
        base_path=ROOT.parents[1] / "brain_api/data"
    )
    pointer = storage.read_current_version()
    original = CanonicalPPOState.from_dict(
        json.loads((ROOT / "data/local_state_response.json").read_text())
    )
    with (ROOT / "data/old_new_all_prices.pkl").open("rb") as file:
        assets = pickle.load(file)
    with (ROOT / "data/old_new_market_prices.pkl").open("rb") as file:
        market = pickle.load(file)
    past_assets = {
        s: history_before_decision(f, DECISION.date()) for s, f in assets.items()
    }
    past_market = {
        s: history_before_decision(f, DECISION.date()) for s, f in market.items()
    }
    assert max(
        f.index.max().date() for f in past_assets.values() if not f.empty
    ) == date(2026, 10, 30)
    required = completed_xnys_session_dates(date(2024, 3, 16), DECISION.date())
    repaired = apply_cboe_vix_fallback(past_market, required_dates=required)
    past_market = repaired.prices
    results = {}
    decisions = {}
    masks = []
    for name, version in VERSIONS.items():
        artifact = storage.load_artifacts(version)
        hmm = RegimeHMMArtifact.from_dict(artifact.regime_hmm)
        rows = spy_vix_rows_after_cutoff(
            past_market, cutoff=hmm.training_cutoff_date, decision_date=DECISION.date()
        )
        calm, stress = live_regime_probabilities(
            artifact.regime_hmm, spy_vix_rows=rows, decision_date=DECISION.date()
        )
        state = build_ppo_discovery_state(
            StateBuildRequest(
                as_of=DECISION,
                universe_snapshot=original.universe_snapshot,
                ohlcv_by_symbol=past_assets,
                news_by_symbol=original.news_by_symbol,
                current_weights={"CASH": 1.0},
                p_calm=calm,
                p_stress=stress,
                spy_closes=past_market["SPY"]["close"].to_numpy(),
                feature_scalers=artifact.feature_scalers,
                market_history_provenance={
                    "price_cutoff": "2026-10-30",
                    "vix_fallback": repaired.audit.to_dict(),
                    "replay": "scratch pre-open price filtering",
                },
            )
        )
        masks.append(state.asset_mask)
        inference = run_ppo_discovery_inference(
            state, expected_digest=state.state_digest, artifacts=artifact
        )
        decisions[name] = asdict(inference)
        outcome = evaluate_week(inference.percentage_weights, assets)
        outcome.update(
            {
                "model_version": version,
                "eligible_assets": int(state.asset_mask.sum()),
                "state_digest": state.state_digest,
                "hmm_cutoff": str(hmm.training_cutoff_date),
            }
        )
        results[name] = outcome
        (
            ROOT
            / "data"
            / (
                "old_preopen_state.json"
                if name == "Old PPO"
                else "new_preopen_state.json"
            )
        ).write_text(json.dumps(state.to_dict(), separators=(",", ":")))
    assert np.array_equal(masks[0], masks[1]), (
        "Models must receive identical eligible universes"
    )
    for benchmark in ["SPY", "QQQ"]:
        results[benchmark] = evaluate_week({benchmark: 1.0, "CASH": 0.0}, market)
    mechanical = {}
    for name, filename in [
        ("Old PPO", "old_same_saved_state_inference.json"),
        ("New PPO", "local_candidate_http_inference.json"),
    ]:
        inference = json.loads((ROOT / filename).read_text())
        mechanical[name] = evaluate_week(inference["percentage_weights"], assets)
        mechanical[name]["model_version"] = inference["model_version"]
    assert storage.read_current_version() == pointer
    result = {
        "start": "2026-11-02 open",
        "end": "2026-11-06 close",
        "price_cutoff": "2026-10-30",
        "starting_capital": CAPITAL,
        "price_cutoff_corrected": results,
        "same_saved_state_diagnostic": mechanical,
        "same_saved_state_limitation": "Uses Monday close and the new artifact's preprocessing; not the fair comparison",
        "roster_limitation": "Same 417-symbol current roster resolved 7 November, not recovered historical Monday roster",
        "no_orders": True,
        "pi_accessed": False,
        "production_current_unchanged": pointer,
    }
    (ROOT / "old_new_week_summary.json").write_text(json.dumps(result, indent=2))
    (ROOT / "old_new_preopen_inference.json").write_text(
        json.dumps(decisions, indent=2)
    )
    pd.DataFrame(
        {name: row["wealth"] for name, row in results.items()},
        index=[
            "Mon open",
            "Mon close",
            "Tue close",
            "Wed close",
            "Thu close",
            "Fri close",
        ],
    ).to_csv(ROOT / "old_new_week_values.csv")
    plot(result)
    print(
        json.dumps(
            {
                name: {
                    key: row[key]
                    for key in ["return", "ending_value", "cash_weight", "k"]
                }
                for name, row in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
