"""The actual broker-cost/HHI reward on a controlled one-step market."""

import numpy as np

from controlled_policy import CASH, N, SYMBOLS

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.rewards import ppo_discovery_reward

PRIOR = {**dict.fromkeys(SYMBOLS, (1 - CASH) / N), "CASH": CASH}
PRICES = dict.fromkeys(SYMBOLS, 100.0)
CONFIG = PPODiscoveryConfig()


def reward(q, returns):
    target = {s: float((1 - CASH) * q[i]) for i, s in enumerate(SYMBOLS)}
    target["CASH"] = CASH
    r, gross, cost, net = ppo_discovery_reward(
        prior_weights=PRIOR,
        target_weights=target,
        symbol_returns=dict(zip(SYMBOLS, returns, strict=True)),
        symbol_prices=PRICES,
        nav_usd=10000.0,
        config=CONFIG,
    )
    return {"reward": r, "net_log": net, "gross": gross, "cost": cost}


def feasible_comparator():
    returns = np.full(N, -0.005)
    returns[0] = 0.04
    choices = []
    for p in np.linspace(0.10, 0.84, 1481):
        q = np.full(N, (1 - p) / (N - 1))
        q[0] = p
        choices.append({"winner_weight": float(p), **reward(q, returns)})
    best = max(choices, key=lambda r: r["reward"])
    return {
        "best": best,
        "equal": reward(np.full(N, 1 / N), returns),
        "grid_range": [0.10, 0.84],
        "grid_step": 0.0005,
    }
