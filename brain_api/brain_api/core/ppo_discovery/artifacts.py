"""Write a candidate ppo_discovery artifact without touching ``current``."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from brain_api.core.ppo_discovery.checkpoints import model_config_hash
from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    MODEL_TYPE,
    PPO_DISCOVERY_ARCHITECTURE,
    PPO_DISCOVERY_SCHEMA_VERSION,
    PPODiscoveryConfig,
    ppo_discovery_cost_contract,
)
from brain_api.core.ppo_discovery.news_adapter import news_adapter_revision
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.promotion import (
    FULL_VARIANT,
    ppo_discovery_source_digest,
    protocol_file_digest,
    result_hash,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.news.models import NEWS_SCHEMA_VERSION, NEWS_SENTIMENT_REVISION
from brain_api.storage.ppo_discovery.huggingface import maybe_upload_ppo_discovery
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

_DATASET_HASH_KEYS = (
    "training_dataset_hash",
    "validation_dataset_hash",
    "evaluation_dataset_hash",
)


def _data_window_start(
    news_manifest: dict[str, Any],
    price_manifest: dict[str, Any] | None,
) -> str | None:
    if price_manifest and price_manifest.get("start"):
        return str(price_manifest["start"])
    weeks = news_manifest.get("weeks") or []
    if not weeks:
        return None
    first = weeks[0]
    if isinstance(first, dict) and first.get("cutoff"):
        return str(first["cutoff"])
    raise PPODiscoveryError("news_manifest weeks[0] cutoff must be a string")


def create_ppo_discovery_metadata(
    *,
    version: str,
    config: PPODiscoveryConfig,
    evaluation: dict[str, Any],
    universe_manifest: dict[str, Any],
    news_manifest: dict[str, Any],
    experiment_id: str,
    experiment_variant: str,
    end_date: str,
    code_revision: str,
    protocol_digest: str,
    config_hash_value: str,
    promoted: bool = False,
    prior_version: str | None = None,
    failure_reasons: list[str] | None = None,
    approved_by: str | None = None,
    price_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """SAC-parity auditable metadata for a candidate or promoted artifact."""
    cost_contract = ppo_discovery_cost_contract()
    symbols = list(
        universe_manifest.get("sorted_symbols")
        or universe_manifest.get("symbols")
        or []
    )
    trained_at = datetime.now(UTC).isoformat()
    metadata: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "ppo_discovery_schema_version": PPO_DISCOVERY_SCHEMA_VERSION,
        "architecture": PPO_DISCOVERY_ARCHITECTURE,
        "version": version,
        "config_hash": config_hash_value,
        "experiment_id": experiment_id,
        "experiment_variant": experiment_variant,
        "asset_feature_names": list(ASSET_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "asset_feature_count": 9,
        "global_feature_count": 7,
        "asset_news_feature_count": 4,
        "global_news_feature_count": 2,
        "news_required": True,
        "trained_at": trained_at,
        "training_timestamp": trained_at,
        "end_date": end_date,
        "data_window": {
            "start": _data_window_start(news_manifest, price_manifest),
            "end": end_date,
        },
        "symbols": symbols,
        "config": config.to_dict(),
        "promoted": promoted,
        "prior_version": prior_version,
        "failure_reasons": list(failure_reasons or []),
        "metrics": {
            "test_cagr": evaluation.get("test_cagr"),
            "test_max_drawdown": evaluation.get("test_max_drawdown"),
            "test_sharpe": evaluation.get("test_sharpe"),
            "val_cagr_median": (
                (evaluation.get("seed_aggregates") or {}).get("val_cagr") or {}
            ).get("median"),
        },
        "training_seed": evaluation.get("selected_seed"),
        "experiment_seeds": list(config.seeds),
        "seed_metrics": evaluation.get("seed_metrics") or {},
        "seed_aggregates": evaluation.get("seed_aggregates") or {},
        "failed_seeds": evaluation.get("failed_seeds") or [],
        "code_revision": code_revision,
        "protocol_digest": protocol_digest,
        "result_hash": result_hash(evaluation),
        "news_schema_version": NEWS_SCHEMA_VERSION,
        "finbert_revision": NEWS_SENTIMENT_REVISION,
        "news_adapter_revision": news_adapter_revision(),
        "training_dataset_hash": news_manifest["training_dataset_hash"],
        "validation_dataset_hash": news_manifest["validation_dataset_hash"],
        "evaluation_dataset_hash": news_manifest["evaluation_dataset_hash"],
        "model_config_hash": model_config_hash(config),
        "train_recipe_hash": evaluation.get("train_recipe_hash"),
        **cost_contract,
    }
    if approved_by is not None:
        metadata["approved_by"] = approved_by
    return metadata


def config_hash(config: PPODiscoveryConfig, extra: dict[str, Any]) -> str:
    payload = {**config.to_dict(), **extra}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def write_candidate_artifact(
    storage: PPODiscoveryHalalNewModelStorage,
    policy: PPODiscoveryActorCritic,
    *,
    config: PPODiscoveryConfig,
    evaluation: dict[str, Any],
    universe_manifest: dict[str, Any],
    feature_scalers: dict[str, Any] | None = None,
    regime_hmm: dict[str, Any] | None = None,
    news_manifest: dict[str, Any] | None = None,
    price_manifest: dict[str, Any] | None = None,
    experiment_lock: dict[str, Any] | None = None,
    experiment_id: str = "ppo-discovery-default",
    end_date: str | None = None,
    experiment_variant: str = FULL_VARIANT,
    pretrained_encoder_state_dict: dict[str, Any],
    seeds_ledger: dict[str, Any] | None = None,
) -> str:
    """Persist a candidate version. ``promote`` stays false."""
    if news_manifest is None or price_manifest is None:
        raise PPODiscoveryError("news_manifest and price_manifest are required")
    if news_manifest.get("complete") is not True:
        raise PPODiscoveryError("news_manifest.complete must be true")
    if price_manifest.get("complete") is not True:
        raise PPODiscoveryError("price_manifest.complete must be true")
    if not regime_hmm:
        raise PPODiscoveryError("regime_hmm artifact is required")
    for key in _DATASET_HASH_KEYS:
        if not news_manifest.get(key) or not price_manifest.get(key):
            raise PPODiscoveryError(
                f"{key} is required on news_manifest and price_manifest"
            )
        if news_manifest[key] != price_manifest[key]:
            raise PPODiscoveryError(f"{key} mismatch between news and price manifests")
    end_date = end_date or datetime.now(UTC).date().isoformat()
    cost_contract = ppo_discovery_cost_contract()
    evaluation.update(cost_contract)
    code_revision = ppo_discovery_source_digest()
    digest = config_hash(
        config,
        {
            "universe_snapshot_sha256": universe_manifest.get("snapshot_sha256"),
            "experiment_id": experiment_id,
            "experiment_variant": experiment_variant,
            "seeds": list(config.seeds),
            "news_manifest": news_manifest,
            "price_manifest": price_manifest,
            "code_revision": code_revision,
        },
    )
    version = f"v{end_date}-{digest}"
    metadata = create_ppo_discovery_metadata(
        version=version,
        config=config,
        evaluation=evaluation,
        universe_manifest=universe_manifest,
        news_manifest=news_manifest,
        experiment_id=experiment_id,
        experiment_variant=experiment_variant,
        end_date=end_date,
        code_revision=code_revision,
        protocol_digest=protocol_file_digest(),
        config_hash_value=digest,
        promoted=False,
        prior_version=storage.read_current_version(),
        failure_reasons=[],
        price_manifest=price_manifest,
    )
    from brain_api.core.ppo_discovery.promotion import (
        evaluate_ppo_discovery_candidate,
    )

    health = evaluate_ppo_discovery_candidate(metadata, evaluation)
    metadata["failure_reasons"] = list(health.failure_reasons)
    storage.write_artifacts(
        version,
        policy_state_dict=policy.state_dict(),
        pretrained_encoder_state_dict=pretrained_encoder_state_dict,
        config=config,
        feature_scalers=feature_scalers or {},
        regime_hmm=regime_hmm,
        metadata=metadata,
        universe_manifest=universe_manifest,
        news_manifest=news_manifest,
        price_manifest=price_manifest,
        experiment_lock=experiment_lock or {"experiment_id": experiment_id},
        evaluation=evaluation,
        promote=False,
        seeds_ledger=seeds_ledger or {"schema_version": 1, "seeds": {}},
    )
    maybe_upload_ppo_discovery(storage, version, make_current=False)
    return version


__all__ = [
    "config_hash",
    "create_ppo_discovery_metadata",
    "ppo_discovery_source_digest",
    "write_candidate_artifact",
]
