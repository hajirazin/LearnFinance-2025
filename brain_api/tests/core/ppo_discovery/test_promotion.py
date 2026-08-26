"""Promotion gates and comparator safety tests."""

from __future__ import annotations

import pytest

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    REQUIRED_ABLATIONS,
)
from brain_api.core.ppo_discovery.evaluator import (
    mark_ablations,
    reject_current_patchtst_on_old_weeks,
    weekly_net_cagr,
)
from brain_api.core.ppo_discovery.promotion import evaluate_ppo_discovery_promotion
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError


def _meta(**overrides):
    payload = {
        "config_hash": "abc123",
        "experiment_variant": "full",
        "asset_feature_names": list(ASSET_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "news_required": True,
    }
    payload.update(overrides)
    return payload


def _eval(**overrides):
    payload = {
        "test_cagr": 0.20,
        "alpha_hrp_test_cagr": 0.15,
        "test_max_drawdown": 0.10,
        "alpha_hrp_test_max_drawdown": 0.12,
        "paired_vs_alpha_hrp_point": 0.001,
        "ablations": {
            name: {"status": "ok", "cagr": 0.18} for name in REQUIRED_ABLATIONS
        },
        "failed_seeds": [],
    }
    payload.update(overrides)
    return payload


def test_promotion_requires_approved_by_and_hash() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(),
        approved_by="",
        expected_config_hash="abc123",
    )
    assert check.is_healthy is False
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(),
        approved_by="razin",
        expected_config_hash="nope",
    )
    assert check.is_healthy is False


def test_promotion_rejects_cagr_floor_and_alpha_hrp_underrun() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=0.05),
        approved_by="razin",
        expected_config_hash="abc123",
    )
    assert any("12%" in reason for reason in check.failure_reasons)
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=0.13, alpha_hrp_test_cagr=0.14),
        approved_by="razin",
        expected_config_hash="abc123",
    )
    assert any("below Alpha-HRP" in reason for reason in check.failure_reasons)


def test_no_news_variant_cannot_promote() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(experiment_variant="no_news_features"),
        evaluation=_eval(),
        approved_by="razin",
        expected_config_hash="abc123",
    )
    assert check.is_healthy is False


def test_healthy_full_variant_passes() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(),
        approved_by="razin",
        expected_config_hash="abc123",
    )
    assert check.is_healthy is True


def test_reject_current_patchtst_on_old_weeks() -> None:
    with pytest.raises(PPODiscoveryError, match="PatchTST current"):
        reject_current_patchtst_on_old_weeks(True)
    reject_current_patchtst_on_old_weeks(False)


def test_ablations_marked_unavailable() -> None:
    report = mark_ablations({"full_ppo": {"cagr": 0.2}})
    assert report["no_news_features"]["status"] == "unavailable"
    assert "full_ppo" in report


def test_promotion_rejects_unavailable_ablations() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(ablations=mark_ablations({})),
        approved_by="razin",
        expected_config_hash="abc123",
    )
    assert check.is_healthy is False
    assert any(
        "unavailable" in reason or "ablation" in reason
        for reason in check.failure_reasons
    )


def test_cagr_formula() -> None:
    weekly = [0.01] * 52
    assert weekly_net_cagr(weekly) == pytest.approx(
        float(__import__("numpy").expm1(0.52))
    )
