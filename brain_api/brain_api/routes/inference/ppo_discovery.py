"""``POST /inference/ppo-discovery``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
    list_universes_for,
)
from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.inference import run_ppo_discovery_inference
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState, PPODiscoveryError
from brain_api.storage.policy import load_current_artifacts_for_bucket

router = APIRouter()


class PPOInferenceRequest(BaseModel):
    state: dict[str, Any]
    state_digest: str
    universe: str = UNIVERSE_NAME


@router.post("/ppo-discovery")
def infer_ppo_discovery(request: PPOInferenceRequest) -> dict[str, Any]:
    if request.universe != UNIVERSE_NAME:
        try:
            get_bucket(ModelType.PPO_DISCOVERY, request.universe)
        except UnknownBucketError as exc:
            allowed = sorted(list_universes_for(ModelType.PPO_DISCOVERY))
            raise HTTPException(
                status_code=422,
                detail=f"Unknown universe '{request.universe}' for ppo_discovery. Allowed: {allowed}",
            ) from exc
    try:
        state = CanonicalPPOState.from_dict(request.state)
        bucket = get_bucket(ModelType.PPO_DISCOVERY, UNIVERSE_NAME)
        artifacts = load_current_artifacts_for_bucket(
            bucket=bucket, model_label=bucket.model_label
        )
        result = run_ppo_discovery_inference(
            state, expected_digest=request.state_digest, artifacts=artifacts
        )
    except HTTPException:
        raise
    except PPODiscoveryError as exc:
        status = 422
        if "digest" in str(exc):
            status = 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="no promoted ppo_discovery artifact"
        ) from exc
    return {
        "model_type": result.model_type,
        "model_version": result.model_version,
        "universe": result.universe,
        "selected_symbols": list(result.selected_symbols),
        "selection_order": list(result.selection_order),
        "k": result.k,
        "percentage_weights": result.percentage_weights,
        "state_digest": result.state_digest,
        "evidence_manifest_sha256": result.evidence_manifest_sha256,
        "explanations": result.explanations,
        "warnings": list(result.warnings),
    }
