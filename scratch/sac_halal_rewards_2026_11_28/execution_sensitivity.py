"""Evaluation-only checks of fee accounting, genuine deadbands, and slippage."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from run_experiment import FACTORY, ROOT, metrics

from brain_api.core.portfolio_rl import env as env_module
from brain_api.core.portfolio_rl.broker_costs import compute_ibkr_rebalance_cost
from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    enforce_constraints,
)
from brain_api.core.portfolio_rl.sac_networks import GaussianActor
from brain_api.core.sac import training
from brain_api.core.sac.config import SACConfig


def apply_deadband(current, target, cash_buffer, band):
    """Freeze small stock deltas; fund accepted buys from sales and spare cash."""
    delta = np.asarray(target[:-1]) - current[:-1]
    delta[np.abs(delta) < band] = 0
    sells = np.minimum(delta, 0)
    buys = np.maximum(delta, 0)
    available = float(current[-1] - cash_buffer - sells.sum())
    if available < 0:
        # The cash floor overrides the band when returns have depleted cash.
        remaining = current[:-1] + sells
        sells -= remaining * (-available / remaining.sum())
        available = 0.0
    if buys.sum() > max(available, 0):
        buys *= max(available, 0) / buys.sum()
    stocks = current[:-1] + sells + buys
    return np.r_[stocks, 1 - stocks.sum()]


@dataclass
class CostWithSlippage:
    base: object
    slippage_fraction: float

    @property
    def total_fraction(self):
        return self.base.total_fraction + self.slippage_fraction

    @property
    def total_usd(self):
        return self.total_fraction * self.base.nav_usd

    def breakdown(self):
        return {
            **self.base.breakdown(),
            "slippage": self.slippage_fraction * self.base.nav_usd,
        }


def evaluate(candidate: Path, panel, *, band, slippage_bps):
    record = json.loads((candidate / "metrics.json").read_text())
    config = SACConfig.from_dict(record["config"])
    data = panel["data"]
    env = FACTORY(data, config, start_week=data.n_weeks - 104, end_week=data.n_weeks)
    env.max_episode_weeks = None
    actor = GaussianActor(
        env.state_dim, env.action_dim, config.hidden_sizes, config.activation
    )
    actor.load_state_dict(
        torch.load(candidate / "model.pt", map_location="cpu", weights_only=True)[
            "actor"
        ]
    )
    actor.eval()
    scaler = pickle.loads((candidate / "scaler.pkl").read_bytes())
    normalized = training.NormalizedEnv(env, scaler)

    def target_for_execution(action, mask):
        target = enforce_constraints(
            apply_softmax_to_weights(action, mask), cash_buffer=config.cash_buffer
        )
        return (
            apply_deadband(env.current_weights, target, config.cash_buffer, band)
            if band
            else target
        )

    def execution_cost(**kwargs):
        # Charge all executed deltas, including those production exempts below 0.5pp.
        base = compute_ibkr_rebalance_cost(**kwargs, weight_epsilon=1e-12)
        delta = np.abs(
            kwargs["target_weights"][:-1] - kwargs["current_weights"][:-1]
        ).sum()
        return CostWithSlippage(base, float(delta) * slippage_bps / 10_000)

    state = normalized.reset(start_week=0)
    rows = []
    done = False
    with (
        patch.object(env_module, "apply_softmax_to_weights", target_for_execution),
        patch.object(env_module, "compute_ibkr_rebalance_cost", execution_cost),
    ):
        while not done:
            step = normalized.step(actor.get_action(state, deterministic=True))
            rows.append(
                {
                    "return": step.info["net_portfolio_return"],
                    "cost": step.info["transaction_cost_usd"],
                    "turnover": step.info["turnover"],
                }
            )
            state, done = step.next_state, step.done
    frame = pd.DataFrame(rows)
    return {
        "variant": record["variant"],
        "seed": record["seed"],
        "deadband": band,
        "slippage_bps": slippage_bps,
        **metrics(frame["return"]),
        "total_cost_usd_fixed_nav": float(frame.cost.sum()),
        "avg_turnover": float(frame.turnover.mean()),
    }


def main():
    torch.set_num_threads(1)
    panel = pickle.loads((ROOT / "evidence/panel.pkl").read_bytes())
    # Restore the train-only HMM features used by the trained policies.
    first = next((ROOT / "results_50000").glob("*/seed_*/regime_hmm.pkl"))
    hmm = pickle.loads(first.read_bytes())
    from brain_api.core.sac.regime_hmm import (
        causal_filter,
        market_observations,
        regime_probabilities,
    )

    data = panel["data"]
    observations = market_observations(data.spy_adjusted_closes, data.vix_closes)
    dates = data.market_dates[20:]
    cutoff = data.weekly_dates[data.n_weeks - 105]
    count = sum(day <= cutoff for day in dates)
    probabilities = np.vstack(
        (
            causal_filter(observations[:count], hmm),
            causal_filter(observations[count:], hmm, hmm.terminal_posterior),
        )
    )
    data.regime_probabilities = np.array(
        [
            regime_probabilities(
                probabilities[np.searchsorted(dates, day, side="right") - 1], hmm
            )
            for day in data.weekly_dates
        ]
    )
    rows = []
    for candidate in sorted((ROOT / "results_50000").glob("*/seed_*")):
        if not (candidate / "metrics.json").exists():
            continue
        for band, slippage in ((0, 0), (0.01, 0), (0.02, 0), (0, 10)):
            rows.append(evaluate(candidate, panel, band=band, slippage_bps=slippage))
    target = ROOT / "results_50000/execution_sensitivity.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f"Saved {len(rows)} evaluation-only sensitivity runs to {target}")


if __name__ == "__main__":
    main()
