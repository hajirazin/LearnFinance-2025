"""Regression tests for the isolated full-universe PatchTST experiment."""

from __future__ import annotations

import math
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch
from experiment_data import load_halal_new_universe_cache
from experiment_metrics import aggregate_metrics, paired_block_bootstrap
from experiment_panel import SplitWindow, build_weekly_panel, panel_arrays
from experiment_spec import ARMS, build_model, patch_count
from experiment_training import fit_target_scales, scaled_channel_mse


def _prices(index: pd.DatetimeIndex, offset: float = 0.0) -> pd.DataFrame:
    base = np.arange(len(index), dtype=float) + 100.0 + offset
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": 1_000_000.0 + 100.0 * np.arange(len(index)),
        },
        index=index,
    )


def test_arm_configs_are_explicit_and_have_expected_patch_counts() -> None:
    control = build_model(ARMS["corrected_control_16_8"]).config
    candidate = build_model(ARMS["coherent_candidate_10_5"]).config

    assert patch_count(60, control.patch_length, control.patch_stride) == 6
    assert patch_count(60, candidate.patch_length, candidate.patch_stride) == 11
    assert not hasattr(control, "dropout")
    assert not hasattr(candidate, "dropout")
    assert control.pooling_type == "mean"
    assert candidate.pooling_type is None
    assert candidate.scaling == "std"
    assert candidate.channel_attention is False
    assert candidate.share_embedding is True
    assert candidate.share_projection is True
    assert candidate.positional_encoding_type == "sincos"
    assert candidate.use_cls_token is False
    assert candidate.attention_dropout == 0.0
    assert candidate.positional_dropout == 0.05
    assert candidate.path_dropout == 0.05
    assert candidate.ff_dropout == 0.05
    assert candidate.head_dropout == 0.0


def test_panel_requires_exact_exchange_sessions_per_symbol_week() -> None:
    sessions = pd.bdate_range("2022-09-01", "2023-04-30")
    prices = {"AAA": _prices(sessions), "BBB": _prices(sessions, 10.0)}
    missing = pd.Timestamp("2023-03-15")
    prices["BBB"] = prices["BBB"].drop(index=missing)
    splits = {
        "train": SplitWindow(date(2023, 1, 9), date(2023, 2, 27)),
        "validation": SplitWindow(date(2023, 3, 20), date(2023, 3, 27)),
        "test": SplitWindow(date(2023, 4, 17), date(2023, 4, 24)),
    }

    panel = build_weekly_panel(
        prices,
        sessions=sessions,
        splits=splits,
        include_test_labels=False,
        context_length=60,
        prediction_length=5,
    )

    affected = panel[panel["decision_date"] == date(2023, 3, 20)]
    assert set(affected["symbol"]) == {"AAA"}
    assert panel.attrs["exclusion_counts"]["missing_exact_session"] >= 1
    assert panel.loc[panel["split"] == "test", "y"].isna().all()
    with pytest.raises(RuntimeError, match="labels are locked"):
        panel_arrays(panel, "test")


def test_target_scales_use_training_targets_only() -> None:
    train = np.ones((4, 5, 5), dtype=np.float32)
    validation = np.full((2, 5, 5), 10_000.0, dtype=np.float32)

    before = fit_target_scales(train)
    validation *= -3.0
    after = fit_target_scales(train)

    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(before, np.ones(5, dtype=np.float32))


def test_scaled_channel_loss_gives_equal_weight_to_equal_standardized_errors() -> None:
    targets = torch.zeros((2, 5, 5))
    predictions = torch.zeros_like(targets)
    predictions[:, :, 0] = 0.01
    predictions[:, :, 4] = 10.0
    scales = torch.tensor([0.01, 1.0, 1.0, 1.0, 10.0])

    loss = scaled_channel_mse(predictions, targets, scales)

    assert loss.item() == pytest.approx(0.4)


def test_top15_metrics_and_turnover_use_each_week_cross_section() -> None:
    rows = []
    for week, shift in [(date(2025, 1, 6), 0), (date(2025, 1, 13), 1)]:
        for index in range(40):
            rows.append(
                {
                    "decision_date": week,
                    "symbol": f"S{index:02d}",
                    "predicted_weekly_return": float((index + shift) % 40),
                    "actual_weekly_return": float(index),
                }
            )
    frame = pd.DataFrame(rows)

    metrics = aggregate_metrics(frame, top_k=15)

    assert metrics["n_weeks"] == 2
    assert metrics["weekly_rank_ic"] > 0.7
    assert metrics["top15_turnover"] == pytest.approx(1.0 / 15.0)
    assert metrics["top15_overlap"] == pytest.approx(14.0 / 15.0)


def test_constant_cross_section_has_no_rank_or_top_k_metrics() -> None:
    rows = []
    for week_index in range(2):
        day = date(2025, 1, 6) + pd.Timedelta(days=7 * week_index)
        for symbol_index in range(40):
            rows.append(
                {
                    "decision_date": day,
                    "symbol": f"S{symbol_index:02d}",
                    "predicted_weekly_return": 0.0,
                    "actual_weekly_return": float(symbol_index),
                }
            )

    metrics = aggregate_metrics(pd.DataFrame(rows), top_k=15)

    for name in (
        "weekly_rank_ic",
        "rank_ic_information_ratio",
        "top15_excess",
        "top15_bottom15_spread",
        "top15_overlap",
        "top15_turnover",
    ):
        assert math.isnan(float(metrics[name]))


def test_paired_block_bootstrap_is_deterministic() -> None:
    rows = []
    for week_index in range(8):
        day = date(2025, 1, 6) + pd.Timedelta(days=7 * week_index)
        for symbol_index in range(40):
            actual = float(symbol_index - 20) / 100.0
            rows.append(
                {
                    "decision_date": day,
                    "symbol": f"S{symbol_index:02d}",
                    "actual_weekly_return": actual,
                    "predicted_weekly_return": actual,
                }
            )
    challenger = pd.DataFrame(rows)
    reference = challenger.copy()
    reference["predicted_weekly_return"] *= -1

    first = paired_block_bootstrap(
        challenger, reference, seed=7, repetitions=100, block_weeks=2, top_k=15
    )
    second = paired_block_bootstrap(
        challenger, reference, seed=7, repetitions=100, block_weeks=2, top_k=15
    )

    assert first == second
    assert first["weekly_rank_ic"]["delta"] > 1.9


def test_paired_bootstrap_handles_undefined_rank_metrics_without_warning() -> None:
    rows = []
    for week_index in range(8):
        day = date(2025, 1, 6) + pd.Timedelta(days=7 * week_index)
        for symbol_index in range(40):
            rows.append(
                {
                    "decision_date": day,
                    "symbol": f"S{symbol_index:02d}",
                    "actual_weekly_return": float(symbol_index),
                    "predicted_weekly_return": 0.0,
                }
            )
    constant = pd.DataFrame(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = paired_block_bootstrap(
            constant,
            constant,
            seed=7,
            repetitions=100,
            block_weeks=2,
            top_k=15,
        )

    assert math.isnan(float(result["weekly_rank_ic"]["delta"]))
    assert all(math.isnan(float(value)) for value in result["weekly_rank_ic"]["ci95"])


def test_universe_cache_loader_preserves_all_symbols_and_provenance(tmp_path) -> None:
    path = tmp_path / "halal_new_2026-08.json"
    path.write_text(
        '{"total_stocks": 3, "fetched_at": "2026-08-08T00:00:00+00:00", '
        '"etfs_used": ["SPUS"], "stocks": '
        '[{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}]}'
    )

    symbols, manifest = load_halal_new_universe_cache(path, minimum_symbols=3)

    assert symbols == ["AAA", "BBB", "CCC"]
    assert manifest["halal_new_count"] == 3
    assert manifest["source_kind"] == "existing_repository_cache"
    assert manifest["source_sha256"]
