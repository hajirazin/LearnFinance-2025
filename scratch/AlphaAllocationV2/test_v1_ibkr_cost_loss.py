"""Check the differentiable fee calculation against SAC's broker math."""

import unittest

import numpy as np
import pandas as pd
import torch
from brain_api.core.portfolio_rl.broker_costs import compute_ibkr_leg_cost
from v1_ibkr_cost_loss import (
    COST_CONFIG,
    COST_WEIGHT,
    CostAwareWeeklyMSE,
    ibkr_leg_fees,
    ibkr_transaction_fees,
)
from v1_loss_objectives import weekly_mse


class IBKRCostLossTest(unittest.TestCase):
    def test_tensor_and_execution_fees_match_sac(self):
        deltas = [0, 1, -1, 500, -500, 100_000, -100_000]
        prices = [10, 10, 10, 100, 100, 1, 1]
        expected = [
            compute_ibkr_leg_cost(
                str(i), abs(d), abs(d) / p, "buy" if d > 0 else "sell", COST_CONFIG
            ).total
            if d
            else 0
            for i, (d, p) in enumerate(zip(deltas, prices, strict=True))
        ]
        tensor = ibkr_leg_fees(
            torch.tensor(deltas, dtype=torch.float64),
            torch.tensor(prices, dtype=torch.float64),
        )
        np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-12)
        legs = pd.Series(deltas, index=[str(i) for i in range(len(deltas))])
        friction, broker = ibkr_transaction_fees(
            legs, pd.Series(prices, index=legs.index), 10
        )
        self.assertAlmostEqual(friction, sum(map(abs, deltas)) * 0.001)
        self.assertAlmostEqual(broker, sum(expected))

    def test_unchanged_holdings_at_unchanged_prices_do_not_pay_again(self):
        dates = pd.to_datetime(["2025-01-03", "2025-01-10"])
        symbols = [f"S{i:02}" for i in range(25)]
        index = pd.MultiIndex.from_product([dates, symbols], names=["friday", "symbol"])
        closes = pd.DataFrame(100.0, index=dates, columns=symbols)
        objective = CostAwareWeeklyMSE(
            index, np.ones(len(index), dtype=bool), closes, "cpu"
        )
        output = torch.linspace(0.02, -0.02, len(symbols))[:, None].requires_grad_()
        target = output.detach().clone()
        objective.reset(training=True)
        loss = objective(output, target, 0)
        opening_fee = (
            compute_ibkr_leg_cost(
                "test", 10_000 / 15, 100 / 15, "buy", COST_CONFIG
            ).total
            * 15
        )
        self.assertAlmostEqual(
            loss.item(), COST_WEIGHT * opening_fee / 10_000, places=8
        )
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertAlmostEqual(
            objective(output, target, 1).item(),
            weekly_mse(output, target).item(),
            places=8,
        )
