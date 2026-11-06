"""Independent noisy future outcomes for each already-trained conditional policy."""

import json

import numpy as np
import torch

from controlled_policy import (
    CASH,
    N,
    AllocationContext,
    ConditionalAllocationPolicy,
    context_returns,
)
from controlled_rewards import reward
from research_support import ROOT, write_json

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig


def main():
    torch.set_num_threads(2)
    result = []
    for source in sorted(ROOT.glob("controlled_*_seed*.json")):
        saved = json.loads(source.read_text())
        policy = ConditionalAllocationPolicy(
            PPODiscoveryConfig(), saved["representation"]
        ).eval()
        policy.load_state_dict(
            torch.load(source.with_suffix(".pt"), map_location="cpu", weights_only=True)
        )
        contexts = [AllocationContext(i, saved["signal_visible"]) for i in range(N)]
        q, _ = policy.deterministic_batch(contexts)
        costs = np.array([reward(weights, np.zeros(N))["cost"] for weights in q])
        hhi_penalty = 0.4 * np.maximum((1 - CASH) ** 2 * (q * q).sum(axis=1) - 0.2, 0)
        rng = np.random.default_rng(9152026)
        slots = rng.integers(0, N, 10000)
        test_contexts = [
            AllocationContext(int(i), saved["signal_visible"]) for i in slots
        ]
        future = context_returns(test_contexts, rng, noise=0.01)
        portfolio = (1 - CASH) * (q[slots] * future).sum(axis=1)
        shaped = np.log1p(portfolio - costs[slots]) - hhi_penalty[slots]
        equal = np.log1p((1 - CASH) * future.mean(axis=1))
        diffs = shaped - equal
        for index in range(10):
            direct = reward(q[slots[index]], future[index])["reward"]
            np.testing.assert_allclose(direct, shaped[index], atol=1e-10, rtol=1e-6)
        mean = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        result.append(
            {
                "case": saved["case"],
                "seed": saved["seed"],
                "future_outcomes": len(slots),
                "mean_paired_reward_advantage": mean,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                "fraction_outcomes_beats_equal": float(np.mean(diffs > 0)),
                "note": "Independent synthetic outcomes; this is not market-return evidence.",
            }
        )
    write_json("controlled_noisy_holdout.json", result)
    for row in result:
        print(row, flush=True)


if __name__ == "__main__":
    main()
