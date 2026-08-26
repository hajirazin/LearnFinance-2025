"""Manual promotion gates for ppo_discovery. Training never auto-promotes."""

from __future__ import annotations

from typing import Any

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    PROMOTION_CAGR_FLOOR,
    REQUIRED_ABLATIONS,
)
from brain_api.core.training_health import ArtifactHealthCheck
from brain_api.storage.ppo_discovery.huggingface import maybe_upload_ppo_discovery
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

FULL_VARIANT = "full"


def evaluate_ppo_discovery_promotion(
    *,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    approved_by: str,
    expected_config_hash: str,
) -> ArtifactHealthCheck:
    """Hard gates from the research spec. Failures never write ``current``."""
    reasons: list[str] = []
    if not approved_by or not str(approved_by).strip():
        reasons.append("approved_by is required")
    if metadata.get("config_hash") != expected_config_hash:
        reasons.append("expected_config_hash does not match artifact config_hash")
    if metadata.get("experiment_variant") != FULL_VARIANT:
        reasons.append("only experiment_variant='full' may be promoted")
    if metadata.get("asset_feature_names") != list(ASSET_FEATURE_NAMES):
        reasons.append("asset feature schema mismatch")
    if metadata.get("global_feature_names") != list(GLOBAL_FEATURE_NAMES):
        reasons.append("global feature schema mismatch")
    if metadata.get("news_required") is not True:
        reasons.append("news_required must be true")
    cagr = float(evaluation.get("test_cagr", float("nan")))
    if not (cagr >= PROMOTION_CAGR_FLOOR):
        reasons.append(f"test CAGR {cagr} is below the 12% floor")
    alpha_cagr = evaluation.get("alpha_hrp_test_cagr")
    if alpha_cagr is None:
        reasons.append("Alpha-HRP test CAGR is unavailable; primary comparison failed")
    elif cagr < float(alpha_cagr):
        reasons.append("test CAGR is below Alpha-HRP")
    max_dd = evaluation.get("test_max_drawdown")
    alpha_dd = evaluation.get("alpha_hrp_test_max_drawdown")
    if max_dd is None or alpha_dd is None:
        reasons.append("max drawdown comparison is incomplete")
    elif float(max_dd) > float(alpha_dd) + 1e-12:
        reasons.append("maximum drawdown is worse than Alpha-HRP")
    paired = evaluation.get("paired_vs_alpha_hrp_point")
    if paired is None or float(paired) <= 0:
        reasons.append("paired PPO-minus-Alpha-HRP point estimate is not positive")
    ablations = evaluation.get("ablations") or {}
    missing = [name for name in REQUIRED_ABLATIONS if name not in ablations]
    if missing:
        reasons.append(f"missing required ablations: {missing}")
    if evaluation.get("failed_seeds"):
        reasons.append("one or more seeds failed")
    if reasons:
        return ArtifactHealthCheck(is_healthy=False, failure_reasons=reasons)
    return ArtifactHealthCheck(is_healthy=True, failure_reasons=[])


def promote_ppo_discovery(
    storage: PPODiscoveryHalalNewModelStorage,
    version: str,
    *,
    approved_by: str,
    expected_config_hash: str,
) -> dict[str, Any]:
    """Promote a candidate only after the locked gates pass."""
    artifacts = storage.load_artifacts(version)
    evaluation = _load_json(artifacts.artifact_dir / "evaluation.json")
    check = evaluate_ppo_discovery_promotion(
        metadata=artifacts.metadata,
        evaluation=evaluation,
        approved_by=approved_by,
        expected_config_hash=expected_config_hash,
    )
    if not check.is_healthy:
        raise ValueError("; ".join(check.failure_reasons))
    storage.promote_version(version)
    maybe_upload_ppo_discovery(storage, version, make_current=True)
    return {
        "version": version,
        "approved_by": approved_by,
        "promoted": True,
        "failure_reasons": [],
    }


def _load_json(path):
    import json

    return json.loads(path.read_text())


__all__ = [
    "FULL_VARIANT",
    "evaluate_ppo_discovery_promotion",
    "promote_ppo_discovery",
]
