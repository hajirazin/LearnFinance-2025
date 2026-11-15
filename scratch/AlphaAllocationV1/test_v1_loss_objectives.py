"""Check weekly compounding, within-week ordering, and SPY shortfall weights."""

import unittest

import torch
from v1_loss_objectives import excess_loss, pairwise_ranking, soft_top15, weekly_mse


class LossObjectivesTest(unittest.TestCase):
    def test_weekly_mse_uses_compounded_total(self):
        target = torch.tensor([[0.10, -0.10, 0, 0, 0]])
        self.assertEqual(weekly_mse(torch.zeros_like(target), target).item(), 0)

    def test_pairwise_loss_rewards_correct_order_and_ignores_ties(self):
        target = torch.tensor([[0.01, 0], [0.01, 0], [-0.01, 0]])
        self.assertLess(
            pairwise_ranking(target, target), pairwise_ranking(-target, target)
        )
        self.assertEqual(pairwise_ranking(target, torch.zeros_like(target)).item(), 0)

    def test_spy_loss_has_zero_at_parity_and_double_shortfall_slope(self):
        x = torch.tensor([-0.02, 0.0, 0.02], requires_grad=True)
        loss = excess_loss(x)
        torch.testing.assert_close(loss, torch.tensor([0.04, 0, -0.02]))
        loss.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([-2.0, -1.0, -1.0]))

    def test_soft_top15_allocates_and_backpropagates(self):
        scores = torch.linspace(-0.05, 0.05, 100, requires_grad=True)
        weights = soft_top15(scores)
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
        self.assertLessEqual(weights.max().item(), 1 / 15 + 1e-6)
        self.assertTrue(weights.diff().ge(0).all())
        (weights @ scores.detach()).backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(scores.grad[-1].item(), 0)
        self.assertAlmostEqual(scores.grad.sum().item(), 0, places=5)
