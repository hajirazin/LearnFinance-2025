"""Check that removing sentiment changes only its supplied information."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from report_patchtst_news_ablation import paired_test
from report_v1_xgboost_comparison import selected_baskets
from train_v1_xgboost_comparison import (
    SEVEN,
    make_patchtst,
    patch_input,
    patch_predictions,
)


class PatchTSTNewsAblationTest(unittest.TestCase):
    def test_news_mask_preserves_other_inputs_and_source(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(2, 255, 7)).astype(np.float32)
        original = values.copy()
        full = patch_input(values, torch.device("cpu"))
        ablated = patch_input(values, torch.device("cpu"), disable_news=True)
        news = list(SEVEN).index("news_sentiment")
        retained = [column for column in range(7) if column != news]
        np.testing.assert_array_equal(values, original)
        torch.testing.assert_close(
            ablated[..., retained], full[..., retained], rtol=0, atol=0
        )
        self.assertEqual(torch.count_nonzero(ablated[..., news]).item(), 0)
        self.assertEqual(ablated.shape, full.shape)

    def test_network_and_initial_weights_are_identical_within_seed(self):
        torch.manual_seed(42)
        seven = make_patchtst()
        torch.manual_seed(42)
        six = make_patchtst()
        self.assertEqual(seven.config.to_dict(), six.config.to_dict())
        for name, value in seven.state_dict().items():
            torch.testing.assert_close(value, six.state_dict()[name], rtol=0, atol=0)

    def test_both_arms_use_the_same_forecast_only_selection_rule(self):
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2026-01-09")], [f"S{i:02}" for i in range(25)]],
            names=["friday", "symbol"],
        )
        scores = pd.DataFrame(
            {
                name: np.arange(25, dtype=float)
                for name in ("Six active", "Seven active")
            },
            index=index,
        )
        selected = selected_baskets(scores, arms=("Six active", "Seven active"))
        day = pd.Timestamp("2026-01-09")
        self.assertEqual(
            set(selected[day, "Six active"]), set(selected[day, "Seven active"])
        )

    def test_two_sided_comparison_is_invariant_to_reversing_the_arms(self):
        delta = np.linspace(0.1, 0.3, 33)
        forward = paired_test(delta)
        reverse = paired_test(-delta)
        self.assertAlmostEqual(forward["two_sided_p"], reverse["two_sided_p"])
        self.assertAlmostEqual(
            forward["mean_six_minus_seven_pp"], -reverse["mean_six_minus_seven_pp"]
        )
        self.assertAlmostEqual(forward["ci95_low_pp"], -reverse["ci95_high_pp"])
        self.assertLess(forward["two_sided_p"], 0.05)

    def test_inference_removes_the_news_information(self):
        news = list(SEVEN).index("news_sentiment")

        class ReadNews(torch.nn.Module):
            def forward(self, past_values):
                return SimpleNamespace(
                    regression_outputs=past_values[..., news].mean(dim=1, keepdim=True)
                )

        values = np.zeros((3, 255, 7), dtype=np.float32)
        values[..., news] = 0.5
        model = ReadNews()
        full = patch_predictions(model, values, torch.device("cpu"))
        ablated = patch_predictions(
            model, values, torch.device("cpu"), disable_news=True
        )
        np.testing.assert_array_equal(full, [0.5] * 3)
        np.testing.assert_array_equal(ablated, [0.0] * 3)
        np.testing.assert_array_equal(values[..., news], 0.5)


if __name__ == "__main__":
    unittest.main()
