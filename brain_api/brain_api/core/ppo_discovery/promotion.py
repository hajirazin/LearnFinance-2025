"""Manual promotion gates for ppo_discovery. Training never auto-promotes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    PPO_DISCOVERY_ARCHITECTURE,
    PPO_DISCOVERY_SCHEMA_VERSION,
    PROMOTION_CAGR_FLOOR,
    ppo_discovery_cost_contract,
)
from brain_api.core.ppo_discovery.schemas import canonical_json_bytes
from brain_api.core.training_health import ArtifactHealthCheck
from brain_api.storage.ppo_discovery.huggingface import maybe_upload_ppo_discovery
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

FULL_VARIANT = "full"
_PPO_DIR = Path(__file__).resolve().parent
_CORE_DIR = _PPO_DIR.parent
_PACKAGE_DIR = _CORE_DIR.parent

_PROTOCOL_FILES = (
    _CORE_DIR / "portfolio_rl" / "rewards.py",
    _CORE_DIR / "portfolio_rl" / "broker_costs.py",
    _CORE_DIR / "weekly_decision.py",
    _PPO_DIR / "config.py",
    _PPO_DIR / "evaluator.py",
    _PPO_DIR / "rewards.py",
    _PPO_DIR / "environment.py",
    _PPO_DIR / "policy.py",
    _PPO_DIR / "distributions.py",
    _PPO_DIR / "splits.py",
    _PPO_DIR / "weeks.py",
    _PPO_DIR / "news_adapter.py",
    _PACKAGE_DIR / "news" / "models.py",
)


def protocol_file_digest() -> str:
    """Hash of reward, cost, policy, split, and news-formula sources."""
    payload = b"".join(path.read_bytes() for path in _PROTOCOL_FILES)
    return hashlib.sha256(payload).hexdigest()


def ppo_discovery_source_digest() -> str:
    """Hash of the ppo_discovery package sources. Included in the version id."""
    payload = b"".join(path.read_bytes() for path in sorted(_PPO_DIR.glob("*.py")))
    return hashlib.sha256(payload).hexdigest()[:12]


def result_hash(evaluation: dict[str, Any]) -> str:
    payload = _without_result_hash(evaluation)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _without_result_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_result_hash(item)
            for key, item in value.items()
            if key != "result_hash"
        }
    if isinstance(value, list):
        return [_without_result_hash(item) for item in value]
    return value


def _finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def evaluate_ppo_discovery_candidate(
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
) -> ArtifactHealthCheck:
    """Intrinsic candidate checks. Missing approved_by is not a failure."""
    reasons: list[str] = []
    if metadata.get("ppo_discovery_schema_version") != PPO_DISCOVERY_SCHEMA_VERSION:
        reasons.append("ppo_discovery_schema_version mismatch")
    if metadata.get("architecture") != PPO_DISCOVERY_ARCHITECTURE:
        reasons.append("architecture mismatch")
    if metadata.get("asset_feature_names") != list(ASSET_FEATURE_NAMES):
        reasons.append("asset feature schema mismatch")
    if metadata.get("global_feature_names") != list(GLOBAL_FEATURE_NAMES):
        reasons.append("global feature schema mismatch")
    for key, expected in ppo_discovery_cost_contract().items():
        if metadata.get(key) != expected:
            reasons.append(f"metadata.{key} does not match locked PPO cost contract")
        if evaluation.get(key) != expected:
            reasons.append(f"evaluation.{key} does not match locked PPO cost contract")
    if metadata.get("news_required") is not True:
        reasons.append("news_required must be true")
    cagr_raw = evaluation.get("test_cagr")
    if not _finite_number(cagr_raw):
        reasons.append("test CAGR is missing or non-finite")
    else:
        cagr = float(cagr_raw)
        if not (cagr > PROMOTION_CAGR_FLOOR):
            reasons.append(f"test CAGR {cagr} is not above the 12% floor")
    sharpe_raw = evaluation.get("test_sharpe")
    if not _finite_number(sharpe_raw):
        reasons.append("test Sharpe is missing or non-finite")
    drawdown = evaluation.get("test_max_drawdown")
    if not _finite_number(drawdown) or not (0.0 <= float(drawdown) <= 1.0):
        reasons.append("test_max_drawdown must be finite in [0, 1]")
    if reasons:
        return ArtifactHealthCheck(is_healthy=False, failure_reasons=reasons)
    return ArtifactHealthCheck(is_healthy=True, failure_reasons=[])


def evaluate_ppo_discovery_promotion(
    *,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    approved_by: str,
    incumbent_cagr: float | None = None,
    incumbent_sharpe: float | None = None,
) -> ArtifactHealthCheck:
    """Promote when test CAGR > 12% and, if an incumbent exists, both
    test CAGR and test Sharpe are strictly greater than the incumbent.
    """
    if incumbent_cagr is not None and incumbent_sharpe is None:
        raise ValueError("incumbent test_sharpe is required")
    candidate = evaluate_ppo_discovery_candidate(metadata, evaluation)
    reasons = list(candidate.failure_reasons)
    if not approved_by or not str(approved_by).strip():
        reasons.append("approved_by is required")
    cagr_raw = evaluation.get("test_cagr")
    sharpe_raw = evaluation.get("test_sharpe")
    if incumbent_cagr is not None:
        if _finite_number(cagr_raw) and not (float(cagr_raw) > float(incumbent_cagr)):
            reasons.append("test CAGR is not strictly greater than the incumbent")
        if _finite_number(sharpe_raw) and not (
            float(sharpe_raw) > float(incumbent_sharpe)
        ):
            reasons.append("test Sharpe is not strictly greater than the incumbent")
    if reasons:
        return ArtifactHealthCheck(is_healthy=False, failure_reasons=reasons)
    return ArtifactHealthCheck(is_healthy=True, failure_reasons=[])


def reevaluate_ppo_discovery(
    storage: PPODiscoveryHalalNewModelStorage, version: str
) -> dict[str, Any]:
    """Recompute CAGR/drawdown from stored test weekly logs."""
    from brain_api.core.ppo_discovery.evaluator import evaluate_policy_weeks

    artifacts = storage.load_artifacts(version)
    evaluation = _load_json(artifacts.artifact_dir / "evaluation.json")
    logs = evaluation.get("test_weekly_net_log")
    if not logs:
        raise ValueError("evaluation.json has no test_weekly_net_log")
    metrics = evaluate_policy_weeks(logs)
    evaluation["test_cagr"] = metrics["cagr"]
    evaluation["test_sharpe"] = metrics["sharpe"]
    evaluation["test_max_drawdown"] = metrics["max_drawdown"]
    evaluation["result_hash"] = result_hash(evaluation)
    (artifacts.artifact_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True)
    )
    metadata = dict(artifacts.metadata)
    metadata["result_hash"] = evaluation["result_hash"]
    (artifacts.artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    storage.write_checksums(version)
    return evaluation


def promote_ppo_discovery(
    storage: PPODiscoveryHalalNewModelStorage,
    version: str,
    *,
    approved_by: str,
    expected_current_version: str,
) -> dict[str, Any]:
    """Promote a candidate only after the locked gates pass.

    Ledger ``pending`` is written before Hugging Face or local ``current``.
    """
    artifacts = storage.load_artifacts(version)
    _smoke_load_candidate(artifacts)
    if _pending_promotion_version(storage) == version:
        _commit_promotion(
            storage,
            version,
            approved_by=approved_by,
            expected_current_version=expected_current_version,
            config_changed=False,
            unpaired_acknowledged=False,
        )
        return {
            "version": version,
            "approved_by": approved_by,
            "promoted": True,
            "failure_reasons": [],
        }
    evaluation = _load_json(artifacts.artifact_dir / "evaluation.json")
    incumbent = storage.read_current_version()
    incumbent_cagr = None
    incumbent_sharpe = None
    if incumbent:
        incumbent_eval = _load_json(
            storage._version_path(incumbent) / "evaluation.json"
        )
        incumbent_cagr_raw = incumbent_eval.get("test_cagr")
        incumbent_sharpe_raw = incumbent_eval.get("test_sharpe")
        if not _finite_number(incumbent_sharpe_raw):
            raise ValueError("incumbent test_sharpe is required")
        if not _finite_number(incumbent_cagr_raw):
            raise ValueError("incumbent test_cagr is required")
        incumbent_cagr = float(incumbent_cagr_raw)
        incumbent_sharpe = float(incumbent_sharpe_raw)
    check = evaluate_ppo_discovery_promotion(
        metadata=artifacts.metadata,
        evaluation=evaluation,
        approved_by=approved_by,
        incumbent_cagr=incumbent_cagr,
        incumbent_sharpe=incumbent_sharpe,
    )
    if not check.is_healthy:
        raise ValueError("; ".join(check.failure_reasons))
    _commit_promotion(
        storage,
        version,
        approved_by=approved_by,
        expected_current_version=expected_current_version,
        config_changed=False,
        unpaired_acknowledged=False,
    )
    return {
        "version": version,
        "approved_by": approved_by,
        "promoted": True,
        "failure_reasons": [],
    }


def _smoke_load_candidate(artifacts: Any) -> None:
    from brain_api.core.ppo_discovery.inference import (
        load_policy_from_artifacts,
        reject_schema_mismatch,
    )

    reject_schema_mismatch(artifacts.metadata)
    policy = load_policy_from_artifacts(artifacts)
    for name, parameter in policy.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"non-finite parameter {name} in candidate artifact")


def _pending_promotion_version(
    storage: PPODiscoveryHalalNewModelStorage,
) -> str | None:
    conn = _ledger(storage)
    try:
        row = conn.execute(
            "SELECT version FROM promotions WHERE status = 'pending'"
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def _commit_promotion(
    storage: PPODiscoveryHalalNewModelStorage,
    version: str,
    *,
    approved_by: str,
    expected_current_version: str,
    config_changed: bool,
    unpaired_acknowledged: bool,
) -> None:
    pointer = storage.read_current_version() or ""
    conn = _ledger(storage)
    try:
        conn.execute("BEGIN IMMEDIATE")
        pending = conn.execute(
            "SELECT version FROM promotions WHERE status = 'pending'"
        ).fetchall()
        pending_versions = [row[0] for row in pending]
        if version in pending_versions and pointer == version:
            _rewrite_promoted_metadata(storage, version, approved_by=approved_by)
            maybe_upload_ppo_discovery(storage, version, make_current=True)
            _mark_ledger_promoted(conn, version)
            conn.commit()
            return
        others = [item for item in pending_versions if item != version]
        if others:
            other = others[0]
            if pointer == other:
                _mark_ledger_promoted(conn, other)
            else:
                conn.rollback()
                raise ValueError(
                    f"promotion pending for {other!r}; aborting {version!r}"
                )
        if expected_current_version != pointer:
            conn.rollback()
            raise ValueError(
                f"expected_current_version {expected_current_version!r} does not "
                f"match current pointer {pointer!r}"
            )
        existing = conn.execute(
            "SELECT status FROM promotions WHERE version = ?", (version,)
        ).fetchone()
        if existing and existing[0] == "promoted":
            conn.commit()
            return
        if existing is None:
            conn.execute(
                "INSERT INTO promotions(version, approved_by, "
                "expected_current_version, promoted_at, status, config_changed, "
                "unpaired_acknowledged) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (
                    version,
                    approved_by,
                    expected_current_version,
                    datetime.now(UTC).isoformat(),
                    int(config_changed),
                    int(unpaired_acknowledged),
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    _rewrite_promoted_metadata(storage, version, approved_by=approved_by)
    maybe_upload_ppo_discovery(storage, version, make_current=True)
    storage.promote_version(version)
    conn = _ledger(storage)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _mark_ledger_promoted(conn, version)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _rewrite_promoted_metadata(
    storage: PPODiscoveryHalalNewModelStorage,
    version: str,
    *,
    approved_by: str,
) -> None:
    artifacts = storage.load_artifacts(version)
    metadata = dict(artifacts.metadata)
    metadata["promoted"] = True
    metadata["failure_reasons"] = []
    metadata["approved_by"] = approved_by
    metadata["promoted_at"] = datetime.now(UTC).isoformat()
    (artifacts.artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    storage.write_checksums(version)


def _mark_ledger_promoted(conn: sqlite3.Connection, version: str) -> None:
    conn.execute(
        "UPDATE promotions SET status = 'promoted', promoted_at = ? WHERE version = ?",
        (datetime.now(UTC).isoformat(), version),
    )


def _ledger(storage: PPODiscoveryHalalNewModelStorage) -> sqlite3.Connection:
    path = storage._model_path / "promotion.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS promotions ("
        "version TEXT PRIMARY KEY, "
        "approved_by TEXT NOT NULL, "
        "expected_current_version TEXT, "
        "promoted_at TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'promoted', "
        "config_changed INTEGER NOT NULL DEFAULT 0, "
        "unpaired_acknowledged INTEGER NOT NULL DEFAULT 0)"
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promotions)")}
    if "status" not in columns:
        conn.execute(
            "ALTER TABLE promotions ADD COLUMN status TEXT NOT NULL DEFAULT 'promoted'"
        )
    if "config_changed" not in columns:
        conn.execute(
            "ALTER TABLE promotions ADD COLUMN config_changed INTEGER NOT NULL DEFAULT 0"
        )
    if "unpaired_acknowledged" not in columns:
        conn.execute(
            "ALTER TABLE promotions ADD COLUMN unpaired_acknowledged "
            "INTEGER NOT NULL DEFAULT 0"
        )
    return conn


def _load_json(path):
    return json.loads(path.read_text())


__all__ = [
    "FULL_VARIANT",
    "evaluate_ppo_discovery_candidate",
    "evaluate_ppo_discovery_promotion",
    "ppo_discovery_source_digest",
    "promote_ppo_discovery",
    "protocol_file_digest",
    "reevaluate_ppo_discovery",
    "result_hash",
]
