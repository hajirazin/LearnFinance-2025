"""Regression tests for the isolated PatchTST pooling-head sweep."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiment_data import load_halal_new_universe_cache, load_or_download_prices
from pooling_metrics import (
    aggregate_metrics,
    build_causal_control_frames,
    paired_block_bootstrap,
    research_clearance,
)
from pooling_panel import build_fold_panel, panel_arrays
from pooling_training import (
    TrainingRunLock,
    mps_smoke_result_is_sanitized,
    training_fingerprint,
    training_jobs,
)
from run_experiment import (
    SequentialModelLedger,
    confirmatory_unlock_payload,
    research_side_effects_manifest,
)
from pooling_spec import (
    CONTEXT_LENGTH,
    PATCH_LENGTH,
    PATCH_STRIDE,
    DateWindow,
    EvaluationFold,
    EVALUATION_FOLDS,
    POOLING_HEADS,
    SEEDS,
    hf_config_for_head,
    patch_count,
)


def _prices(index: pd.DatetimeIndex, offset: float = 0.0) -> pd.DataFrame:
    base = np.arange(len(index), dtype=float) + 100.0 + offset
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.zeros(len(index)),
        },
        index=index,
    )


def _small_fold() -> EvaluationFold:
    return EvaluationFold(
        name="fixture",
        evidence_kind="confirmatory",
        train=DateWindow(date(2023, 1, 9), date(2023, 2, 6)),
        validation=DateWindow(date(2023, 2, 27), date(2023, 3, 6)),
        evaluation=DateWindow(date(2023, 3, 27), date(2023, 4, 3)),
    )


def test_frozen_geometry_is_10_5_eleven_tokens() -> None:
    assert patch_count(CONTEXT_LENGTH, PATCH_LENGTH, PATCH_STRIDE) == 11
    configs = [hf_config_for_head(head) for head in POOLING_HEADS.values()]
    assert all(config.patch_length == 10 for config in configs)
    assert all(config.patch_stride == 5 for config in configs)
    assert all(config.num_input_channels == 1 for config in configs)


def test_pooling_type_is_the_only_arm_difference() -> None:
    configs = [hf_config_for_head(head) for head in POOLING_HEADS.values()]
    normalized = []
    for config in configs:
        values = config.to_dict()
        values.pop("pooling_type")
        normalized.append(values)

    assert list(POOLING_HEADS) == ["mean", "flatten"]
    assert POOLING_HEADS["mean"].pooling_type == "mean"
    assert POOLING_HEADS["flatten"].pooling_type is None
    assert configs[0].pooling_type == "mean"
    assert configs[1].pooling_type is None
    assert normalized[1:] == normalized[:-1]
    assert all(config.attention_dropout == 0.2 for config in configs)
    assert all(config.positional_dropout == 0.2 for config in configs)


def test_expanding_folds_are_chronological_and_embargoed() -> None:
    folds = list(EVALUATION_FOLDS.values())
    assert [fold.name for fold in folds] == [
        "development_2024",
        "development_2025",
        "confirmatory_2026",
    ]
    assert [fold.train.start for fold in folds] == [date(2015, 5, 4)] * 3
    assert [fold.train.end.year for fold in folds] == [2022, 2023, 2024]
    assert [fold.validation.end.year for fold in folds] == [2023, 2024, 2025]
    assert [fold.evaluation.start.year for fold in folds] == [2024, 2025, 2026]
    for fold in folds:
        assert (fold.validation.start - fold.train.end).days == 21
        assert (fold.evaluation.start - fold.validation.end).days == 21
        assert fold.train.end < fold.validation.start
        assert fold.validation.end < fold.evaluation.start
    assert asdict(folds[-1])["evaluation"]["end"] == date(2026, 8, 17)


def test_close_only_panel_requires_exact_sessions() -> None:
    sessions = pd.bdate_range("2022-09-01", "2023-04-30")
    prices = {"AAA": _prices(sessions), "BBB": _prices(sessions, 10.0)}
    prices["BBB"] = prices["BBB"].drop(index=pd.Timestamp("2023-03-15"))

    panel = build_fold_panel(
        prices,
        sessions=sessions,
        fold=_small_fold(),
        include_evaluation_labels=False,
    )

    affected = panel[panel["decision_date"] == date(2023, 3, 27)]
    assert set(affected["symbol"]) == {"AAA"}
    assert panel.attrs["exclusion_counts"]["missing_exact_session"] >= 1


def test_non_close_fields_cannot_change_close_only_eligibility() -> None:
    sessions = pd.bdate_range("2022-09-01", "2023-04-30")
    first = _prices(sessions)
    second = first.copy()
    second[["open", "high", "low", "volume"]] = np.nan

    panel = build_fold_panel(
        {"AAA": first, "BBB": second},
        sessions=sessions,
        fold=_small_fold(),
        include_evaluation_labels=True,
    )

    counts = panel.groupby("symbol").size().to_dict()
    assert counts["AAA"] == counts["BBB"]


def test_confirmatory_labels_remain_locked() -> None:
    sessions = pd.bdate_range("2022-09-01", "2023-04-30")
    panel = build_fold_panel(
        {"AAA": _prices(sessions)},
        sessions=sessions,
        fold=_small_fold(),
        include_evaluation_labels=False,
    )

    assert panel.loc[panel["split"] == "evaluation", "y"].isna().all()
    with pytest.raises(RuntimeError, match="labels are locked"):
        panel_arrays(panel, "evaluation")


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_panel_rejects_nonpositive_or_nonfinite_close(bad_value: float) -> None:
    sessions = pd.bdate_range("2022-09-01", "2023-04-30")
    bad = _prices(sessions)
    bad.loc[pd.Timestamp("2023-03-15"), "close"] = bad_value

    panel = build_fold_panel(
        {"AAA": _prices(sessions), "BAD": bad},
        sessions=sessions,
        fold=_small_fold(),
        include_evaluation_labels=True,
    )

    affected = panel[panel["decision_date"] == date(2023, 3, 27)]
    assert set(affected["symbol"]) == {"AAA"}
    assert panel.attrs["exclusion_counts"]["nonpositive_or_nonfinite_close"] >= 1


def test_universe_cache_loader_preserves_all_symbols_and_hash(tmp_path: Path) -> None:
    cache = tmp_path / "halal_new_2026-08.json"
    cache.write_text(
        json.dumps(
            {
                "total_stocks": 3,
                "fetched_at": "2026-08-08T00:00:00+00:00",
                "etfs_used": ["SPUS"],
                "stocks": [
                    {"symbol": "AAA"},
                    {"symbol": "BBB"},
                    {"symbol": "CCC"},
                ],
            }
        )
    )

    symbols, manifest = load_halal_new_universe_cache(cache, minimum_symbols=3)

    assert symbols == ["AAA", "BBB", "CCC"]
    assert manifest["halal_new_count"] == 3
    assert len(manifest["source_sha256"]) == 64


def test_verified_price_cache_rejects_request_mismatch(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def downloader(
        symbols: list[str], start: date, end: date, *, log_prefix: str
    ) -> dict[str, pd.DataFrame]:
        calls.append(symbols)
        index = pd.bdate_range(start, end, inclusive="left")
        return {symbol: _prices(index) for symbol in symbols}

    load_or_download_prices(
        ["AAA"],
        data_dir=tmp_path,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 2, 1),
        downloader=downloader,
    )
    load_or_download_prices(
        ["AAA"],
        data_dir=tmp_path,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 1),
        downloader=downloader,
    )

    assert calls == [["AAA"], ["AAA"]]


def test_price_manifest_records_missing_symbols_and_hashes(tmp_path: Path) -> None:
    def downloader(
        symbols: list[str], start: date, end: date, *, log_prefix: str
    ) -> dict[str, pd.DataFrame]:
        del symbols, log_prefix
        return {"AAA": _prices(pd.bdate_range(start, end, inclusive="left"))}

    prices, manifest = load_or_download_prices(
        ["AAA", "MISSING"],
        data_dir=tmp_path,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 2, 1),
        downloader=downloader,
    )

    assert set(prices) == {"AAA"}
    assert manifest["missing_symbols"] == ["MISSING"]
    assert manifest["files"]["AAA"]["sha256"]
    assert (tmp_path / manifest["files"]["AAA"]["file"]).exists()


def test_training_fingerprint_changes_for_fold_pooling_seed_or_data() -> None:
    train_x = np.zeros((2, 60, 1), dtype=np.float32)
    train_y = np.zeros((2, 5, 1), dtype=np.float32)
    validation_x = train_x.copy()
    validation_y = train_y.copy()
    base = training_fingerprint(
        EVALUATION_FOLDS["development_2024"],
        POOLING_HEADS["mean"],
        20260823,
        (train_x, train_y, validation_x, validation_y),
        max_epochs=60,
        patience=8,
    )

    variants = [
        training_fingerprint(
            EVALUATION_FOLDS["development_2025"],
            POOLING_HEADS["mean"],
            20260823,
            (train_x, train_y, validation_x, validation_y),
            max_epochs=60,
            patience=8,
        ),
        training_fingerprint(
            EVALUATION_FOLDS["development_2024"],
            POOLING_HEADS["flatten"],
            20260823,
            (train_x, train_y, validation_x, validation_y),
            max_epochs=60,
            patience=8,
        ),
        training_fingerprint(
            EVALUATION_FOLDS["development_2024"],
            POOLING_HEADS["mean"],
            20260824,
            (train_x, train_y, validation_x, validation_y),
            max_epochs=60,
            patience=8,
        ),
    ]
    changed = train_x.copy()
    changed[0, 0, 0] = 1.0
    variants.append(
        training_fingerprint(
            EVALUATION_FOLDS["development_2024"],
            POOLING_HEADS["mean"],
            20260823,
            (changed, train_y, validation_x, validation_y),
            max_epochs=60,
            patience=8,
        )
    )

    assert all(value != base for value in variants)


def test_training_lock_rejects_a_second_runner(tmp_path: Path) -> None:
    lock_path = tmp_path / "training.lock"
    with TrainingRunLock(lock_path):
        with pytest.raises(RuntimeError, match="already active"):
            with TrainingRunLock(lock_path):
                pass


def test_training_job_order_is_strictly_fold_arm_seed() -> None:
    jobs = list(training_jobs())

    assert len(jobs) == 18
    assert len({(job.fold_name, job.pooling_name, job.seed) for job in jobs}) == 18
    assert jobs[0] == ("development_2024", "mean", 20260823)
    assert jobs[1] == ("development_2024", "mean", 20260824)
    assert jobs[3] == ("development_2024", "flatten", 20260823)
    assert jobs[-1] == ("confirmatory_2026", "flatten", 20260825)


def test_mps_smoke_result_schema_is_sanitized() -> None:
    result = {
        "passed": True,
        "device": "mps",
        "state_sha256": "a" * 64,
        "prediction_sha256": "b" * 64,
        "hardware": {"chip": "Apple M5 Pro", "unified_memory_gb": 48},
    }

    assert mps_smoke_result_is_sanitized(result)
    result["hardware"]["serial_number"] = "secret"
    assert not mps_smoke_result_is_sanitized(result)


def _evaluation_frame(*, reverse: bool = False) -> pd.DataFrame:
    rows = []
    for week_index in range(8):
        decision = date(2025, 1, 6) + pd.Timedelta(days=7 * week_index)
        for symbol_index in range(40):
            actual = float(symbol_index - 20) / 100.0
            predicted = -actual if reverse else actual
            rows.append(
                {
                    "fold": "fixture",
                    "decision_date": decision,
                    "symbol": f"S{symbol_index:02d}",
                    "actual_weekly_return": actual,
                    "predicted_weekly_return": predicted,
                }
            )
    return pd.DataFrame(rows)


def test_causal_mean_never_reads_future_evaluation_rows() -> None:
    rows = []
    for split, decision, actual in [
        ("train", date(2024, 1, 8), 0.01),
        ("validation", date(2024, 1, 15), 0.03),
        ("evaluation", date(2024, 1, 22), 0.50),
        ("evaluation", date(2024, 1, 29), -0.50),
    ]:
        for index in range(30):
            rows.append(
                {
                    "fold": "fixture",
                    "split": split,
                    "decision_date": decision,
                    "symbol": f"S{index:02d}",
                    "actual_weekly_log_return": actual + index / 10_000,
                    "past_week_log_return": index / 1_000,
                    "momentum_4w_log_return": index / 500,
                    "context_log_return": index / 100,
                    "volatility_4w": 0.01 + index / 10_000,
                }
            )
    controls = build_causal_control_frames(pd.DataFrame(rows))
    mean = controls["causal_historical_mean"]
    first = mean[mean["decision_date"] == date(2024, 1, 22)]
    second = mean[mean["decision_date"] == date(2024, 1, 29)]

    assert first.iloc[0]["predicted_weekly_log_return"] == pytest.approx(0.02)
    assert second.iloc[0]["predicted_weekly_log_return"] == pytest.approx(0.18)
    assert controls["ridge"].attrs["ridge_feature_columns"] == [
        "past_week_log_return",
        "momentum_4w_log_return",
        "context_log_return",
        "volatility_4w",
    ]


def test_paired_block_uncertainty_is_deterministic_and_includes_stability() -> None:
    challenger = _evaluation_frame()
    reference = _evaluation_frame(reverse=True)

    first = paired_block_bootstrap(
        challenger,
        reference,
        seed=7,
        repetitions=100,
        block_weeks=2,
        top_k=15,
    )
    second = paired_block_bootstrap(
        challenger,
        reference,
        seed=7,
        repetitions=100,
        block_weeks=2,
        top_k=15,
    )

    assert first == second
    assert first["weekly_rank_ic"]["delta"] > 1.9
    assert set(first) >= {"top15_overlap", "top15_turnover"}


def test_aggregate_metrics_prioritizes_requested_business_metrics() -> None:
    metrics = aggregate_metrics(_evaluation_frame(), top_k=15)

    assert metrics["weekly_rank_ic"] == pytest.approx(1.0)
    assert metrics["top15_bottom15_spread"] > 0
    assert metrics["top15_excess"] > 0
    assert metrics["top15_overlap"] == pytest.approx(1.0)
    assert metrics["top15_turnover"] == pytest.approx(0.0)


def test_clearance_fails_when_either_gate_is_not_credibly_beaten() -> None:
    arm_metrics = {
        "development_2024": {"weekly_rank_ic": 0.1},
        "development_2025": {"weekly_rank_ic": 0.1},
        "confirmatory_2026": {"weekly_rank_ic": 0.1},
    }
    comparisons = {
        "development": {
            gate: {
                "weekly_rank_ic": {"delta": 0.01, "ci95": [-0.01, 0.03]},
                "top15_excess": {"delta": 0.01, "ci95": [-0.01, 0.03]},
                "top15_bottom15_spread": {
                    "delta": 0.01,
                    "ci95": [-0.01, 0.03],
                },
            }
            for gate in ("causal_historical_mean", "ridge")
        },
        "confirmatory": {
            gate: {
                "weekly_rank_ic": {"delta": 0.01, "ci95": [0.001, 0.03]},
                "top15_excess": {"delta": 0.01, "ci95": [-0.01, 0.03]},
                "top15_bottom15_spread": {
                    "delta": 0.01,
                    "ci95": [-0.01, 0.03],
                },
            }
            for gate in ("causal_historical_mean", "ridge")
        },
    }
    comparisons["confirmatory"]["ridge"]["weekly_rank_ic"]["ci95"][0] = -0.001

    verdict = research_clearance(arm_metrics, comparisons)

    assert verdict["passed"] is False
    assert any("ridge" in reason for reason in verdict["failure_reasons"])


def test_confirmatory_unlock_requires_all_six_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    training: dict[str, dict[str, object]] = {}
    for pooling_name in POOLING_HEADS:
        for seed in SEEDS:
            path = tmp_path / pooling_name / str(seed) / "weights.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{pooling_name}/{seed}".encode())
            training[f"confirmatory_2026/{pooling_name}/{seed}"] = {
                "weights_path": str(path),
                "weights_sha256": __import__("hashlib")
                .sha256(path.read_bytes())
                .hexdigest(),
                "created_at_utc": "2026-08-23T00:00:00+00:00",
            }

    payload = confirmatory_unlock_payload(
        training,
        attempt_id="fixture",
        locked_non_evaluation_panel_sha256="a" * 64,
    )

    assert payload["checkpoint_count"] == 6
    assert payload["evaluation_labels_read_before_unlock"] is False
    training.pop(next(iter(training)))
    with pytest.raises(RuntimeError, match="exactly 6"):
        confirmatory_unlock_payload(
            training,
            attempt_id="fixture",
            locked_non_evaluation_panel_sha256="a" * 64,
        )


def test_runner_never_has_more_than_one_live_model() -> None:
    ledger = SequentialModelLedger()
    first = ("development_2024", "mean", 20260823)
    second = ("development_2024", "mean", 20260824)

    ledger.begin(first)
    with pytest.raises(RuntimeError, match="still active"):
        ledger.begin(second)
    ledger.finish(first)
    ledger.begin(second)
    ledger.finish(second)

    assert ledger.max_active_models == 1
    assert ledger.completed_jobs == [first, second]


def test_manifest_declares_no_pointer_or_production_side_effects() -> None:
    manifest = research_side_effects_manifest()

    assert manifest == {
        "production_current_pointers_touched": False,
        "artifacts_promoted": False,
        "trades_submitted": False,
        "temporal_workflows_triggered": False,
        "production_cache_written": False,
    }
