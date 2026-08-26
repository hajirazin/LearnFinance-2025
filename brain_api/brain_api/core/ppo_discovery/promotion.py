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
    PROMOTION_CAGR_FLOOR,
    REQUIRED_ABLATIONS,
)
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
    payload = {
        "test_cagr": evaluation.get("test_cagr"),
        "test_max_drawdown": evaluation.get("test_max_drawdown"),
        "test_weekly_net_log": evaluation.get("test_weekly_net_log"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def evaluate_ppo_discovery_promotion(
    *,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    approved_by: str,
    expected_config_hash: str,
    incumbent_cagr: float | None = None,
    incumbent_protocol_digest: str | None = None,
    incumbent_evaluation_dataset_hash: str | None = None,
    incumbent_model_config_hash: str | None = None,
    acknowledge_unpaired_evaluation: bool = False,
) -> ArtifactHealthCheck:
    """Hard gates from the research spec. Failures never write ``current``."""
    _ = incumbent_protocol_digest, incumbent_model_config_hash
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
    expected_protocol = protocol_file_digest()
    if metadata.get("protocol_digest") != expected_protocol:
        reasons.append("protocol_digest does not match current reward/cost/evaluator")
    if metadata.get("code_revision") != ppo_discovery_source_digest():
        reasons.append("code_revision does not match current ppo_discovery sources")
    if not metadata.get("evaluation_dataset_hash"):
        reasons.append("evaluation_dataset_hash is required")
    if not metadata.get("model_config_hash"):
        reasons.append("model_config_hash is required")
    cagr_raw = evaluation.get("test_cagr")
    if not _finite_number(cagr_raw):
        reasons.append("test CAGR is missing or non-finite")
        cagr = float("nan")
    else:
        cagr = float(cagr_raw)
        if cagr < PROMOTION_CAGR_FLOOR:
            reasons.append(f"test CAGR {cagr} is below the 12% floor")
    drawdown = evaluation.get("test_max_drawdown")
    if not _finite_number(drawdown) or not (0.0 <= float(drawdown) <= 1.0):
        reasons.append("test_max_drawdown must be finite in [0, 1]")
    eval_hash = metadata.get("evaluation_dataset_hash")
    paired = (
        incumbent_cagr is not None
        and incumbent_evaluation_dataset_hash is not None
        and eval_hash == incumbent_evaluation_dataset_hash
    )
    if incumbent_cagr is not None and not paired:
        if not acknowledge_unpaired_evaluation:
            reasons.append(
                "incumbent evaluation_dataset_hash differs; pass "
                "acknowledge_unpaired_evaluation"
            )
    elif paired and _finite_number(cagr_raw) and cagr < float(incumbent_cagr):
        reasons.append("test CAGR is below the incumbent")
    ablations = evaluation.get("ablations") or {}
    for name in REQUIRED_ABLATIONS:
        row = ablations.get(name)
        if not isinstance(row, dict) or row.get("status") != "ok":
            reasons.append(
                f"required ablation {name!r} is missing, failed, or unavailable"
            )
            continue
        ablation_cagr = row.get("cagr")
        if ablation_cagr is None or not _finite_number(ablation_cagr):
            reasons.append(f"required ablation {name!r} has a non-finite CAGR")
    if evaluation.get("failed_seeds"):
        reasons.append("one or more seeds failed")
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
    evaluation["test_max_drawdown"] = metrics["max_drawdown"]
    evaluation["result_hash"] = result_hash(evaluation)
    (artifacts.artifact_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True)
    )
    storage.write_checksums(version)
    return evaluation


def promote_ppo_discovery(
    storage: PPODiscoveryHalalNewModelStorage,
    version: str,
    *,
    approved_by: str,
    expected_config_hash: str,
    expected_current_version: str,
    acknowledge_unpaired_evaluation: bool = False,
) -> dict[str, Any]:
    """Promote a candidate only after the locked gates pass.

    Ledger ``pending`` is written before Hugging Face or local ``current``.
    """
    artifacts = storage.load_artifacts(version)
    _smoke_load_candidate(artifacts)
    evaluation = _load_json(artifacts.artifact_dir / "evaluation.json")
    incumbent = storage.read_current_version()
    incumbent_cagr = None
    incumbent_protocol = None
    incumbent_eval_hash = None
    incumbent_model_hash = None
    if incumbent:
        incumbent_eval = _load_json(
            storage._version_path(incumbent) / "evaluation.json"
        )
        incumbent_meta = _load_json(storage._version_path(incumbent) / "metadata.json")
        incumbent_cagr = incumbent_eval.get("test_cagr")
        incumbent_protocol = incumbent_meta.get("protocol_digest")
        incumbent_eval_hash = incumbent_meta.get("evaluation_dataset_hash")
        incumbent_model_hash = incumbent_meta.get("model_config_hash")
    check = evaluate_ppo_discovery_promotion(
        metadata=artifacts.metadata,
        evaluation=evaluation,
        approved_by=approved_by,
        expected_config_hash=expected_config_hash,
        incumbent_cagr=None if incumbent_cagr is None else float(incumbent_cagr),
        incumbent_protocol_digest=incumbent_protocol,
        incumbent_evaluation_dataset_hash=incumbent_eval_hash,
        incumbent_model_config_hash=incumbent_model_hash,
        acknowledge_unpaired_evaluation=acknowledge_unpaired_evaluation,
    )
    if not check.is_healthy:
        raise ValueError("; ".join(check.failure_reasons))
    config_changed = bool(
        incumbent_model_hash
        and incumbent_model_hash != artifacts.metadata.get("model_config_hash")
    )
    _commit_promotion(
        storage,
        version,
        approved_by=approved_by,
        expected_current_version=expected_current_version,
        config_changed=config_changed,
        unpaired_acknowledged=acknowledge_unpaired_evaluation,
    )
    return {
        "version": version,
        "approved_by": approved_by,
        "promoted": True,
        "failure_reasons": [],
        "config_changed": config_changed,
        "unpaired_acknowledged": acknowledge_unpaired_evaluation,
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


def _commit_promotion(
    storage: PPODiscoveryHalalNewModelStorage,
    version: str,
    *,
    approved_by: str,
    expected_current_version: str,
    config_changed: bool,
    unpaired_acknowledged: bool,
) -> None:
    conn = _ledger(storage)
    try:
        conn.execute("BEGIN IMMEDIATE")
        pending = conn.execute(
            "SELECT version, status FROM promotions WHERE status = 'pending'"
        ).fetchall()
        if any(row[0] != version for row in pending):
            other = next(row[0] for row in pending if row[0] != version)
            conn.rollback()
            raise ValueError(f"promotion pending for {other!r}; aborting {version!r}")
        promoted_row = conn.execute(
            "SELECT version FROM promotions WHERE status = 'promoted' "
            "ORDER BY promoted_at DESC LIMIT 1"
        ).fetchone()
        ledger_current = promoted_row[0] if promoted_row else ""
        if expected_current_version != ledger_current:
            conn.rollback()
            raise ValueError(
                f"expected_current_version {expected_current_version!r} does not "
                f"match ledger {ledger_current!r}"
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
    maybe_upload_ppo_discovery(storage, version, make_current=True)
    storage.promote_version(version)
    conn = _ledger(storage)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE promotions SET status = 'promoted', promoted_at = ? "
            "WHERE version = ?",
            (datetime.now(UTC).isoformat(), version),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
    "evaluate_ppo_discovery_promotion",
    "ppo_discovery_source_digest",
    "promote_ppo_discovery",
    "protocol_file_digest",
    "reevaluate_ppo_discovery",
    "result_hash",
]
