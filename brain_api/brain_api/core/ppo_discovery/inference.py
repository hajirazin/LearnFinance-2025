"""Deterministic inference for a promoted ppo_discovery artifact."""

from __future__ import annotations

from typing import Any

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    MODEL_TYPE,
    UNIVERSE_NAME,
)
from brain_api.core.ppo_discovery.explanations import build_explanations
from brain_api.core.ppo_discovery.policy import (
    PPODiscoveryActorCritic,
    validate_inference_weights,
)
from brain_api.core.ppo_discovery.schemas import (
    CanonicalPPOState,
    PPODiscoveryError,
    PPOInferenceResult,
    sha256_digest,
    state_to_digest_payload,
)
from brain_api.storage.policy import load_current_artifacts_for_bucket
from brain_api.storage.ppo_discovery.local import PPODiscoveryArtifacts


def load_policy_from_artifacts(
    artifacts: PPODiscoveryArtifacts,
) -> PPODiscoveryActorCritic:
    policy = PPODiscoveryActorCritic(artifacts.config)
    policy.load_state_dict(artifacts.policy_state_dict)
    policy.eval()
    return policy


def reject_schema_mismatch(metadata: dict[str, Any]) -> None:
    if metadata.get("asset_feature_names") != list(ASSET_FEATURE_NAMES):
        raise PPODiscoveryError("artifact asset feature schema mismatch")
    if metadata.get("global_feature_names") != list(GLOBAL_FEATURE_NAMES):
        raise PPODiscoveryError("artifact global feature schema mismatch")
    if metadata.get("news_required") is not True:
        raise PPODiscoveryError("artifact is not a news-required full variant")
    if metadata.get("experiment_variant") != "full":
        raise PPODiscoveryError(
            "only the full experiment variant may be used for inference"
        )


def run_ppo_discovery_inference(
    state: CanonicalPPOState,
    *,
    expected_digest: str,
    artifacts: PPODiscoveryArtifacts | None = None,
) -> PPOInferenceResult:
    """Deterministic inference. Incomplete news never reaches this function."""
    recomputed = sha256_digest(state_to_digest_payload(state))
    state.state_digest = recomputed
    if expected_digest != recomputed:
        raise PPODiscoveryError("state_digest mismatch")
    if artifacts is None:
        bucket = get_bucket(ModelType.PPO_DISCOVERY, UNIVERSE_NAME)
        artifacts = load_current_artifacts_for_bucket(
            bucket=bucket, model_label=bucket.model_label
        )
    reject_schema_mismatch(artifacts.metadata)
    policy = load_policy_from_artifacts(artifacts)
    eligible = {
        symbol
        for symbol, flag in zip(state.symbols, state.asset_mask.tolist(), strict=True)
        if flag and symbol
    }
    weights = validate_inference_weights(policy.infer_weights(state), eligible)
    stocks = tuple(
        sorted(symbol for symbol in weights if symbol != "CASH" and weights[symbol] > 0)
    )
    k = len(stocks)
    explanations = build_explanations(state, weights, artifacts.metadata)
    return PPOInferenceResult(
        model_type=MODEL_TYPE,
        model_version=artifacts.version,
        universe=UNIVERSE_NAME,
        selected_symbols=stocks,
        selection_order=stocks,
        k=k,
        percentage_weights=weights,
        state_digest=state.state_digest,
        evidence_manifest_sha256=sha256_digest(state.evidence_manifest),
        explanations=explanations,
    )


__all__ = [
    "load_policy_from_artifacts",
    "reject_schema_mismatch",
    "run_ppo_discovery_inference",
]
