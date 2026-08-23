#!/usr/bin/env python3
"""Deterministic contract tests for the corrected PatchTST experiment harness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation import pesaran_timmermann, weekly_statistics
from experiment_core import (
    ARCHITECTURES,
    SPLITS,
    build_model,
    build_weekly_panel,
    control_predictions,
    json_dump,
    load_model_artifact,
    model_sensitivity,
    patch_count,
    validate_split_contract,
)
from run_experiments import _prediction_frame, _select_with_declared_tie_break


def _synthetic_prices(n: int = 900) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=n)
    out: dict[str, pd.DataFrame] = {}
    for j, symbol in enumerate([f"S{i:02d}" for i in range(10)]):
        base = 100.0 * np.exp(np.linspace(0, 0.3 + j / 100, n))
        wave = 1.0 + 0.01 * np.sin(np.arange(n) / (8 + j))
        close = base * wave
        out[symbol] = pd.DataFrame(
            {
                "open": close * (1 - 0.001),
                "high": close * (1 + 0.004),
                "low": close * (1 - 0.004),
                "close": close,
                "volume": 1_000_000 + j * 1_000 + np.arange(n),
            },
            index=dates,
        )
    return out


class ArchitectureContractTest(unittest.TestCase):
    def test_corrected_stride_produces_six_patches(self) -> None:
        self.assertEqual(patch_count(60, 16, 8), 6)
        self.assertEqual(patch_count(60, 16, 1), 45)
        corrected = build_model(ARCHITECTURES["canonical_independent_5ch"])
        legacy = build_model(ARCHITECTURES["legacy_effective"])
        self.assertEqual(corrected.config.patch_stride, 8)
        self.assertEqual(legacy.config.patch_stride, 1)

    def test_only_channel_mixing_uses_non_close_inputs(self) -> None:
        torch.manual_seed(7)
        x = np.random.default_rng(7).normal(0, 0.01, (2, 60, 5)).astype(np.float32)
        independent = build_model(ARCHITECTURES["canonical_independent_5ch"])
        mixing = build_model(ARCHITECTURES["canonical_mixing_5ch"])
        independent_result = model_sensitivity(independent, x, close_idx=3)
        mixing_result = model_sensitivity(mixing, x, close_idx=3)
        self.assertEqual(independent_result["forecast_max_abs_delta"], 0.0)
        self.assertEqual(independent_result["non_close_grad_l1"], 0.0)
        self.assertGreater(mixing_result["forecast_max_abs_delta"], 1e-12)
        self.assertGreater(mixing_result["non_close_grad_l1"], 1e-12)


class MetricContractTest(unittest.TestCase):
    def test_pesaran_timmermann_uses_marginal_variance(self) -> None:
        pred = np.array([1, 1, 1, -1, -1, 1, -1, 1], dtype=float)
        actual = np.array([1, -1, 1, -1, 1, 1, -1, -1], dtype=float)
        n = len(pred)
        px, py = np.mean(pred > 0), np.mean(actual > 0)
        observed = np.mean((pred > 0) == (actual > 0))
        expected = px * py + (1 - px) * (1 - py)
        variance_observed = expected * (1 - expected) / n
        variance_expected = (
            (2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)
        ) / n + 4 * px * py * (1 - px) * (1 - py) / (n * n)
        self.assertAlmostEqual(
            pesaran_timmermann(pred, actual),
            (observed - expected) / np.sqrt(variance_observed - variance_expected),
        )

    def test_reported_magnitude_is_compounded_simple_return(self) -> None:
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2025-01-06").date()],
                "symbol": ["S00"],
                "split": ["test"],
                "actual_weekly_return": [np.log(1.10)],
            }
        )
        frame = _prediction_frame(panel, "test", np.array([np.log(1.20)]), "arm", 1)
        self.assertAlmostEqual(frame.iloc[0]["actual_weekly_return"], 0.10)
        self.assertAlmostEqual(frame.iloc[0]["predicted_weekly_return"], 0.20)

    def test_top_three_metrics_are_invariant_to_row_order_with_ties(self) -> None:
        frame = pd.DataFrame(
            {
                "decision_date": ["2025-01-06"] * 6,
                "symbol": list("ABCDEF"),
                "predicted_weekly_return": [0.3, 0.2, 0.2, 0.2, -0.1, -0.1],
                "actual_weekly_return": [0.4, 0.1, 0.1, -0.2, -0.2, -0.2],
            }
        )
        original = weekly_statistics(frame).iloc[0]
        shuffled = weekly_statistics(frame.sample(frac=1, random_state=9)).iloc[0]
        for metric in ["top3_excess", "top3_bottom3_spread", "ndcg_at_3"]:
            self.assertAlmostEqual(original[metric], shuffled[metric], msg=metric)

    def test_selection_has_explicit_tie_break(self) -> None:
        scores = {"close": 0.1, "independent": 0.1, "mixing": 0.0}
        self.assertEqual(
            _select_with_declared_tie_break(scores, ["close", "independent", "mixing"]),
            "close",
        )

    def test_model_loader_rejects_tampered_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            model = build_model(ARCHITECTURES["canonical_close_only"])
            torch.save(model.state_dict(), model_dir / "weights.pt")
            json_dump(model_dir / "meta.json", {"weights_sha256": "0" * 64})
            with self.assertRaisesRegex(RuntimeError, "weight hash mismatch"):
                load_model_artifact(ARCHITECTURES["canonical_close_only"], model_dir)


class PanelContractTest(unittest.TestCase):
    def test_split_dates_are_disjoint_and_embargoed(self) -> None:
        validate_split_contract(SPLITS)
        eras = [
            set(pd.date_range(v[0], v[1], freq="W-MON").date) for v in SPLITS.values()
        ]
        self.assertFalse(eras[0] & eras[1])
        self.assertFalse(eras[1] & eras[2])

    def test_test_labels_stay_absent_until_explicit_unlock(self) -> None:
        prices = _synthetic_prices()
        custom = {
            "train": (
                pd.Timestamp("2020-05-04").date(),
                pd.Timestamp("2021-12-20").date(),
            ),
            "validation": (
                pd.Timestamp("2022-01-10").date(),
                pd.Timestamp("2022-06-20").date(),
            ),
            "test": (
                pd.Timestamp("2022-07-11").date(),
                pd.Timestamp("2023-05-29").date(),
            ),
        }
        locked = build_weekly_panel(prices, custom, include_test_labels=False)
        test_rows = locked[locked["split"] == "test"]
        self.assertGreater(len(test_rows), 0)
        self.assertTrue(test_rows["actual_weekly_return"].isna().all())
        unlocked = build_weekly_panel(prices, custom, include_test_labels=True)
        self.assertTrue(np.isfinite(unlocked["actual_weekly_return"]).all())
        for split_name, (_start, _end) in list(custom.items())[:-1]:
            next_name = list(custom)[list(custom).index(split_name) + 1]
            split_rows = unlocked[unlocked["split"] == split_name]
            self.assertTrue(
                (
                    pd.to_datetime(split_rows["target_end"]).dt.date
                    < custom[next_name][0]
                ).all()
            )

    def test_misaligned_symbol_target_calendar_removes_the_whole_week(self) -> None:
        prices = _synthetic_prices()
        missing_session = prices["S00"].index[700]
        prices["S00"] = prices["S00"].drop(missing_session)
        custom = {
            "train": (
                pd.Timestamp("2020-05-04").date(),
                pd.Timestamp("2021-12-20").date(),
            ),
            "validation": (
                pd.Timestamp("2022-01-10").date(),
                pd.Timestamp("2022-06-20").date(),
            ),
            "test": (
                pd.Timestamp("2022-07-11").date(),
                pd.Timestamp("2023-05-29").date(),
            ),
        }
        panel = build_weekly_panel(prices, custom, include_test_labels=True)
        target_counts = panel.groupby("decision_date")["target_end"].nunique()
        self.assertTrue((target_counts == 1).all())
        self.assertGreater(
            panel.attrs["exclusions"].get("misaligned_or_short_target_calendar", 0), 0
        )

    def test_controls_do_not_use_current_or_future_labels(self) -> None:
        prices = _synthetic_prices()
        custom = {
            "train": (
                pd.Timestamp("2020-05-04").date(),
                pd.Timestamp("2021-12-20").date(),
            ),
            "validation": (
                pd.Timestamp("2022-01-10").date(),
                pd.Timestamp("2022-06-20").date(),
            ),
            "test": (
                pd.Timestamp("2022-07-11").date(),
                pd.Timestamp("2023-05-29").date(),
            ),
        }
        panel = build_weekly_panel(prices, custom, include_test_labels=True)
        original = control_predictions(panel)
        majority_by_week = original.groupby("decision_date")["majority_sign"].nunique()
        self.assertTrue((majority_by_week == 1).all())
        mutated = panel.copy()
        first_test = mutated.loc[mutated["split"] == "test", "decision_date"].min()
        mutated.loc[
            mutated["decision_date"] >= first_test, "actual_weekly_return"
        ] *= -100
        changed = control_predictions(mutated)
        keys = [
            "zero_return",
            "historical_mean",
            "majority_sign",
            "persistence_1w",
            "reversal_1w",
            "momentum_4w",
        ]
        mask = original["decision_date"] == first_test
        for key in keys:
            self.assertTrue(np.isfinite(original.loc[mask, key]).all(), key)
            np.testing.assert_allclose(original.loc[mask, key], changed.loc[mask, key])

        # A five-session target can cross the next Monday in holiday weeks;
        # that still-pending label must not enter expanding controls.
        decisions = sorted(panel["decision_date"].unique())
        previous, current = decisions[10], decisions[11]
        pending_panel = panel.copy()
        pending_panel.loc[pending_panel["decision_date"] == previous, "target_end"] = (
            current
        )
        pending_original = control_predictions(pending_panel)
        mutated_pending = pending_panel.copy()
        mutated_pending.loc[
            mutated_pending["decision_date"] == previous, "actual_weekly_return"
        ] *= -100
        pending_changed = control_predictions(mutated_pending)
        current_mask = pending_original["decision_date"] == current
        for key in ["historical_mean", "majority_sign"]:
            np.testing.assert_allclose(
                pending_original.loc[current_mask, key],
                pending_changed.loc[current_mask, key],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
