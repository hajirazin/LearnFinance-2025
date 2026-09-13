"""``POST /train/ppo-discovery/promote`` and ``/reevaluate``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.promotion import (
    promote_ppo_discovery,
    reevaluate_ppo_discovery,
)
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

router = APIRouter()


class PPOPromoteRequest(BaseModel):
    version: str
    approved_by: str = Field(min_length=1)
    expected_current_version: str


class PPOReevaluateRequest(BaseModel):
    version: str


@router.post("/ppo-discovery/promote")
def promote_ppo_discovery_endpoint(request: PPOPromoteRequest) -> dict:
    storage = PPODiscoveryHalalNewModelStorage(base_path=DEFAULT_DATA_PATH)
    try:
        return promote_ppo_discovery(
            storage,
            request.version,
            approved_by=request.approved_by,
            expected_current_version=request.expected_current_version,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status = 503 if "HF_" in detail or "Hugging Face" in detail else 422
        raise HTTPException(status_code=status, detail=detail) from exc


@router.post("/ppo-discovery/reevaluate")
def reevaluate_ppo_discovery_endpoint(request: PPOReevaluateRequest) -> dict:
    storage = PPODiscoveryHalalNewModelStorage(base_path=DEFAULT_DATA_PATH)
    try:
        return reevaluate_ppo_discovery(storage, request.version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
