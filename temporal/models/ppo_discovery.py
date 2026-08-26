"""Pydantic payloads for ppo_discovery Temporal activities."""

from pydantic import BaseModel, Field


class PPOStateResponse(BaseModel):
    state: dict
    state_digest: str
    run_id: str | None = None
    attempt: int | None = None


class PPOInferenceResponse(BaseModel):
    model_type: str
    model_version: str
    universe: str
    selected_symbols: list[str]
    selection_order: list[str]
    k: int
    percentage_weights: dict[str, float]
    state_digest: str
    evidence_manifest_sha256: str
    explanations: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
