"""Tests for SAC promotion guardrails (full + finetune).

Per AGENTS.md rule #2 (math correctness over DRY), SAC and forecaster
guardrails live in separate modules even though both return the same
:class:`ArtifactHealthCheck` dataclass. These tests cover only the
SAC math; forecaster guardrails are tested in ``test_training_utils.py``.
"""

import math
from pathlib import Path

import pytest

from brain_api.core.sac.promotion import (
    _SAC_REQUIRED_FILES,
    SAC_PROMOTION_CAGR_FLOOR,
    evaluate_sac_artifact_health,
    evaluate_sac_finetune_artifact_health,
)


def _materialize_sac_files(tmp_path: Path) -> Path:
    """Write every artifact file SAC's storage layer persists."""
    for filename in _SAC_REQUIRED_FILES:
        (tmp_path / filename).write_bytes(b"x")
    return tmp_path


def _healthy_full_args(artifact_dir: Path) -> dict:
    return {
        "actor_loss": 0.5,
        "critic_loss": 0.7,
        "eval_cagr": SAC_PROMOTION_CAGR_FLOOR + 0.05,
        "eval_sharpe": 1.2,
        "eval_max_drawdown": -0.15,
        "expected_symbol_count": 15,
        "actual_symbol_count": 15,
        "artifact_dir": artifact_dir,
    }


def _healthy_finetune_args(artifact_dir: Path) -> dict:
    return {
        "actor_loss": 0.5,
        "critic_loss": 0.7,
        "eval_cagr": SAC_PROMOTION_CAGR_FLOOR + 0.05,
        "eval_sharpe": 1.2,
        "eval_max_drawdown": -0.15,
        "prior_symbol_order": ["AAPL", "MSFT", "GOOGL"],
        "actual_symbol_order": ["AAPL", "MSFT", "GOOGL"],
        "artifact_dir": artifact_dir,
    }


# ---------------------------------------------------------------------------
# evaluate_sac_artifact_health (full)
# ---------------------------------------------------------------------------


class TestSACFullHealth:
    def test_happy_path_promotes(self, tmp_path: Path):
        artifact_dir = _materialize_sac_files(tmp_path)
        health = evaluate_sac_artifact_health(**_healthy_full_args(artifact_dir))
        assert health.is_healthy is True
        assert health.failure_reasons == []

    def test_eval_cagr_below_floor_rejects(self, tmp_path: Path):
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_full_args(artifact_dir)
        args["eval_cagr"] = SAC_PROMOTION_CAGR_FLOOR - 0.01  # 0.11 < 0.12
        health = evaluate_sac_artifact_health(**args)
        assert health.is_healthy is False
        assert any(
            "eval_cagr" in r and "below floor" in r for r in health.failure_reasons
        )

    def test_eval_cagr_exactly_at_floor_promotes(self, tmp_path: Path):
        """The locked policy rejects only CAGR strictly below 12%."""
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_full_args(artifact_dir)
        args["eval_cagr"] = SAC_PROMOTION_CAGR_FLOOR
        health = evaluate_sac_artifact_health(**args)
        assert health.is_healthy is True

    def test_nan_eval_cagr_rejects(self, tmp_path: Path):
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_full_args(artifact_dir)
        args["eval_cagr"] = float("nan")
        health = evaluate_sac_artifact_health(**args)
        assert "eval_cagr is not finite" in health.failure_reasons

    @pytest.mark.parametrize(
        "metric_name",
        ["eval_sharpe", "eval_max_drawdown", "actor_loss", "critic_loss"],
    )
    def test_other_metrics_do_not_gate_full_promotion(
        self, tmp_path: Path, metric_name: str
    ):
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_full_args(artifact_dir)
        args[metric_name] = math.nan
        health = evaluate_sac_artifact_health(**args)
        assert health.is_healthy is True
        assert health.failure_reasons == []

    def test_symbol_count_does_not_gate_full_promotion(self, tmp_path: Path):
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_full_args(artifact_dir)
        args["actual_symbol_count"] = 14
        health = evaluate_sac_artifact_health(**args)
        assert health.is_healthy is True

    @pytest.mark.parametrize("missing_file", _SAC_REQUIRED_FILES)
    def test_artifact_presence_does_not_gate_full_promotion(
        self, tmp_path: Path, missing_file: str
    ):
        for filename in _SAC_REQUIRED_FILES:
            if filename == missing_file:
                continue
            (tmp_path / filename).write_bytes(b"x")
        health = evaluate_sac_artifact_health(**_healthy_full_args(tmp_path))
        assert health.is_healthy is True

    def test_only_cagr_failure_is_reported(self, tmp_path: Path):
        (tmp_path / "actor.pt").write_bytes(b"x")
        args = _healthy_full_args(tmp_path)
        args["eval_cagr"] = SAC_PROMOTION_CAGR_FLOOR - 0.05
        args["actor_loss"] = math.nan
        args["actual_symbol_count"] = 10
        health = evaluate_sac_artifact_health(**args)
        assert health.is_healthy is False
        reasons = health.failure_reasons
        assert len(reasons) == 1
        assert "eval_cagr" in reasons[0] and "below floor" in reasons[0]


# ---------------------------------------------------------------------------
# evaluate_sac_finetune_artifact_health
# ---------------------------------------------------------------------------


class TestSACFinetuneHealth:
    def test_happy_path_promotes(self, tmp_path: Path):
        artifact_dir = _materialize_sac_files(tmp_path)
        health = evaluate_sac_finetune_artifact_health(
            **_healthy_finetune_args(artifact_dir)
        )
        assert health.is_healthy is True
        assert health.failure_reasons == []

    def test_eval_cagr_below_floor_rejects(self, tmp_path: Path):
        """Same floor as full -- finetune doesn't get a free pass."""
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_finetune_args(artifact_dir)
        args["eval_cagr"] = SAC_PROMOTION_CAGR_FLOOR - 0.01
        health = evaluate_sac_finetune_artifact_health(**args)
        assert health.is_healthy is False
        assert any("eval_cagr" in r for r in health.failure_reasons)

    def test_symbol_order_reordered_rejects(self, tmp_path: Path):
        """Same set of symbols, different order. Action space is
        positional, so this MUST reject -- otherwise the finetuned
        actor would output weights for the wrong stock."""
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_finetune_args(artifact_dir)
        args["prior_symbol_order"] = ["AAPL", "MSFT", "GOOGL"]
        args["actual_symbol_order"] = ["MSFT", "AAPL", "GOOGL"]
        health = evaluate_sac_finetune_artifact_health(**args)
        assert health.is_healthy is False
        assert any(
            "actual_symbol_order" in r and "does not match" in r
            for r in health.failure_reasons
        )

    def test_symbol_set_changed_rejects(self, tmp_path: Path):
        """Different set entirely (e.g. delisted symbol dropped) ->
        finetune isn't valid; operator must run a full retrain."""
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_finetune_args(artifact_dir)
        args["prior_symbol_order"] = ["AAPL", "MSFT", "GOOGL"]
        args["actual_symbol_order"] = ["AAPL", "MSFT"]  # GOOGL dropped
        health = evaluate_sac_finetune_artifact_health(**args)
        assert health.is_healthy is False
        assert any("actual_symbol_order" in r for r in health.failure_reasons)

    def test_symbol_order_identical_passes(self, tmp_path: Path):
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_finetune_args(artifact_dir)
        args["prior_symbol_order"] = ["A", "B", "C", "D", "E"]
        args["actual_symbol_order"] = ["A", "B", "C", "D", "E"]
        health = evaluate_sac_finetune_artifact_health(**args)
        assert health.is_healthy is True

    @pytest.mark.parametrize(
        "metric_name",
        ["eval_sharpe", "eval_max_drawdown", "actor_loss", "critic_loss"],
    )
    def test_nan_other_metric_rejects(self, tmp_path: Path, metric_name: str):
        artifact_dir = _materialize_sac_files(tmp_path)
        args = _healthy_finetune_args(artifact_dir)
        args[metric_name] = math.nan
        health = evaluate_sac_finetune_artifact_health(**args)
        assert f"{metric_name} is not finite" in health.failure_reasons

    @pytest.mark.parametrize("missing_file", _SAC_REQUIRED_FILES)
    def test_missing_artifact_file_rejects(self, tmp_path: Path, missing_file: str):
        for filename in _SAC_REQUIRED_FILES:
            if filename == missing_file:
                continue
            (tmp_path / filename).write_bytes(b"x")
        health = evaluate_sac_finetune_artifact_health(
            **_healthy_finetune_args(tmp_path)
        )
        assert f"{missing_file} missing or zero bytes" in health.failure_reasons

    def test_combined_failures_accumulate(self, tmp_path: Path):
        (tmp_path / "actor.pt").write_bytes(b"x")
        args = _healthy_finetune_args(tmp_path)
        args["eval_cagr"] = SAC_PROMOTION_CAGR_FLOOR - 0.05
        args["actor_loss"] = math.nan
        args["actual_symbol_order"] = ["B", "A", "C"]  # reordered
        health = evaluate_sac_finetune_artifact_health(**args)
        assert health.is_healthy is False
        reasons = health.failure_reasons
        assert any("eval_cagr" in r for r in reasons)
        assert "actor_loss is not finite" in reasons
        assert any("actual_symbol_order" in r for r in reasons)
