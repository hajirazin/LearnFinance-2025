"""Domain objects for the ppo_discovery decision state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    AUDIT_NEWS_FIELDS,
    ENCODER_CHANNELS,
    ENCODER_SESSIONS,
    GLOBAL_FEATURE_NAMES,
    MAX_ASSETS,
)


class PPODiscoveryError(ValueError):
    """Raised when ppo_discovery evidence or math invariants fail."""


def canonical_json_bytes(payload: Any) -> bytes:
    """UTF-8 JSON with sorted keys, no NaN/Infinity, compact separators."""

    def _default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"cannot canonicalize {type(value)!r}")

    encoded = json.dumps(
        payload,
        default=_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return encoded.encode("utf-8")


def sha256_digest(payload: Any) -> str:
    """Return a ``sha256:`` prefixed digest of canonical JSON."""
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class UniverseSnapshot:
    """Immutable frozen ``halal_new`` roster for one experiment or live run."""

    universe: str
    retrieved_at: str
    sorted_symbols: tuple[str, ...]
    symbol_count: int
    cache_path: str | None
    cache_sha256: str | None
    source_provenance: dict[str, Any]
    snapshot_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "retrieved_at": self.retrieved_at,
            "sorted_symbols": list(self.sorted_symbols),
            "symbol_count": self.symbol_count,
            "cache_path": self.cache_path,
            "cache_sha256": self.cache_sha256,
            "source_provenance": self.source_provenance,
            "snapshot_sha256": self.snapshot_sha256,
        }


@dataclass(frozen=True)
class SymbolNewsFeatures:
    """Per-symbol news aggregates. Compact model fields plus audit extras."""

    symbol: str
    raw_sentiment: float
    article_count: int
    average_confidence: float
    sentiment_dispersion: float
    hours_since_latest: float
    unique_source_count: int
    has_news: int
    query_complete: bool
    news_recency: float
    log1p_article_count: float
    article_ids_sha256: str
    request_manifest_sha256: str


@dataclass
class CanonicalPPOState:
    """Packed tensors plus evidence needed for train and live inference."""

    symbols: tuple[str, ...]
    asset_mask: np.ndarray
    price_history: np.ndarray
    asset_features: np.ndarray
    globals: np.ndarray
    current_weights: dict[str, float]
    news_by_symbol: dict[str, SymbolNewsFeatures]
    audit_news: dict[str, Any]
    exclusions: dict[str, str]
    universe_snapshot: UniverseSnapshot
    evidence_manifest: dict[str, Any]
    state_digest: str
    as_of: str
    held_symbols: tuple[str, ...]

    def packed_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "price_history": tuple(self.price_history.shape),
            "asset_features": tuple(self.asset_features.shape),
            "globals": tuple(self.globals.shape),
            "asset_mask": tuple(self.asset_mask.shape),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "asset_mask": self.asset_mask.astype(int).tolist(),
            "price_history": np.asarray(self.price_history, dtype=np.float64).tolist(),
            "asset_features": np.asarray(
                self.asset_features, dtype=np.float64
            ).tolist(),
            "globals": np.asarray(self.globals, dtype=np.float64).tolist(),
            "current_weights": self.current_weights,
            "audit_news": self.audit_news,
            "exclusions": self.exclusions,
            "universe_snapshot": self.universe_snapshot.to_dict(),
            "evidence_manifest": self.evidence_manifest,
            "state_digest": self.state_digest,
            "as_of": self.as_of,
            "held_symbols": list(self.held_symbols),
            "news_by_symbol": {
                symbol: {
                    "symbol": row.symbol,
                    "raw_sentiment": row.raw_sentiment,
                    "article_count": row.article_count,
                    "average_confidence": row.average_confidence,
                    "sentiment_dispersion": row.sentiment_dispersion,
                    "hours_since_latest": row.hours_since_latest,
                    "unique_source_count": row.unique_source_count,
                    "has_news": row.has_news,
                    "query_complete": row.query_complete,
                    "news_recency": row.news_recency,
                    "log1p_article_count": row.log1p_article_count,
                    "article_ids_sha256": row.article_ids_sha256,
                    "request_manifest_sha256": row.request_manifest_sha256,
                }
                for symbol, row in self.news_by_symbol.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CanonicalPPOState:
        snapshot = payload["universe_snapshot"]
        news = {
            symbol: SymbolNewsFeatures(**row)
            for symbol, row in payload.get("news_by_symbol", {}).items()
        }
        state = cls(
            symbols=tuple(payload["symbols"]),
            asset_mask=np.asarray(payload["asset_mask"], dtype=bool),
            price_history=np.asarray(payload["price_history"], dtype=np.float64),
            asset_features=np.asarray(payload["asset_features"], dtype=np.float64),
            globals=np.asarray(payload["globals"], dtype=np.float64),
            current_weights=dict(payload["current_weights"]),
            news_by_symbol=news,
            audit_news=payload.get("audit_news") or {},
            exclusions=payload.get("exclusions") or {},
            universe_snapshot=UniverseSnapshot(
                universe=snapshot["universe"],
                retrieved_at=snapshot["retrieved_at"],
                sorted_symbols=tuple(snapshot["sorted_symbols"]),
                symbol_count=int(snapshot["symbol_count"]),
                cache_path=snapshot.get("cache_path"),
                cache_sha256=snapshot.get("cache_sha256"),
                source_provenance=snapshot.get("source_provenance") or {},
                snapshot_sha256=snapshot["snapshot_sha256"],
            ),
            evidence_manifest=payload.get("evidence_manifest") or {},
            state_digest="",
            as_of=payload["as_of"],
            held_symbols=tuple(payload.get("held_symbols") or ()),
        )
        recomputed = sha256_digest(state_to_digest_payload(state))
        state.state_digest = recomputed
        claimed = payload.get("state_digest")
        if claimed and claimed != recomputed:
            raise PPODiscoveryError("state_digest does not match reconstructed tensors")
        return state


@dataclass(frozen=True)
class SampledAction:
    """Factored PPO action with the exact sequence used for log-probability."""

    k: int
    selection_order: tuple[str, ...]
    selection_indices: tuple[int, ...]
    z_cash: float | None
    dirichlet_weights: tuple[float, ...] | None
    percentage_weights: dict[str, float]
    log_p_k: float
    log_p_selection: float
    log_p_cash: float
    log_p_dirichlet: float
    log_p_total: float


@dataclass(frozen=True)
class ActionLogProb:
    """Recomputed component log-probabilities for a stored action."""

    log_p_k: float
    log_p_selection: float
    log_p_cash: float
    log_p_dirichlet: float
    log_p_total: float


@dataclass(frozen=True)
class PPOInferenceResult:
    """Deterministic inference output."""

    model_type: str
    model_version: str
    universe: str
    selected_symbols: tuple[str, ...]
    selection_order: tuple[str, ...]
    k: int
    percentage_weights: dict[str, float]
    state_digest: str
    evidence_manifest_sha256: str
    explanations: dict[str, Any]
    warnings: tuple[str, ...] = ()


def assert_feature_contract() -> None:
    """Fail fast if a future edit silently changes packed widths."""
    if any(name in ASSET_FEATURE_NAMES for name in AUDIT_NEWS_FIELDS):
        raise PPODiscoveryError("audit news fields must not enter asset features")
    if any(name in GLOBAL_FEATURE_NAMES for name in AUDIT_NEWS_FIELDS):
        raise PPODiscoveryError("audit news fields must not enter globals")


def empty_padded_history() -> np.ndarray:
    return np.zeros((MAX_ASSETS, ENCODER_SESSIONS, ENCODER_CHANNELS), dtype=np.float64)


def state_to_digest_payload(state: CanonicalPPOState) -> dict[str, Any]:
    """JSON-ready payload whose hash is the canonical state digest."""
    manifest = dict(state.evidence_manifest)
    manifest.pop("state_digest", None)
    return {
        "as_of": state.as_of,
        "symbols": list(state.symbols),
        "asset_mask": state.asset_mask.astype(int).tolist(),
        "price_history": np.asarray(state.price_history, dtype=np.float64).tolist(),
        "asset_features": np.asarray(state.asset_features, dtype=np.float64).tolist(),
        "globals": np.asarray(state.globals, dtype=np.float64).tolist(),
        "current_weights": state.current_weights,
        "universe_snapshot_sha256": state.universe_snapshot.snapshot_sha256,
        "evidence_manifest": manifest,
        "exclusions": state.exclusions,
        "held_symbols": list(state.held_symbols),
        "asset_feature_names": list(ASSET_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
    }


__all__ = [
    "ActionLogProb",
    "CanonicalPPOState",
    "PPODiscoveryError",
    "PPOInferenceResult",
    "SampledAction",
    "SymbolNewsFeatures",
    "UniverseSnapshot",
    "assert_feature_contract",
    "canonical_json_bytes",
    "empty_padded_history",
    "sha256_digest",
    "state_to_digest_payload",
]
