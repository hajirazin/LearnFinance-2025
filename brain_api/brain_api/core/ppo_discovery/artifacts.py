"""Write a candidate ppo_discovery artifact without touching ``current``."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.promotion import FULL_VARIANT
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.storage.ppo_discovery.huggingface import maybe_upload_ppo_discovery
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage


def ppo_discovery_source_digest() -> str:
    """Hash of the ppo_discovery package sources. Included in the version id."""
    root = Path(__file__).resolve().parent
    payload = b"".join(path.read_bytes() for path in sorted(root.glob("*.py")))
    return hashlib.sha256(payload).hexdigest()[:12]


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
    end_date = end_date or datetime.now(UTC).date().isoformat()
    code_revision = ppo_discovery_source_digest()
    digest = config_hash(
        config,
        {
            "universe_snapshot_sha256": universe_manifest.get("snapshot_sha256"),
            "experiment_id": experiment_id,
            "seeds": list(config.seeds),
            "news_manifest": news_manifest,
            "price_manifest": price_manifest,
            "code_revision": code_revision,
        },
    )
    version = f"v{end_date}-{digest}"
    metadata = {
        "version": version,
        "config_hash": digest,
        "experiment_id": experiment_id,
        "experiment_variant": FULL_VARIANT,
        "asset_feature_names": list(ASSET_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "asset_feature_count": 9,
        "global_feature_count": 7,
        "asset_news_feature_count": 4,
        "global_news_feature_count": 2,
        "news_required": True,
        "trained_at": datetime.now(UTC).isoformat(),
        "end_date": end_date,
        "code_revision": code_revision,
    }
    storage.write_artifacts(
        version,
        policy_state_dict=policy.state_dict(),
        pretrained_encoder_state_dict=policy.temporal.state_dict(),
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
    )
    maybe_upload_ppo_discovery(storage, version, make_current=False)
    return version


__all__ = ["config_hash", "ppo_discovery_source_digest", "write_candidate_artifact"]
