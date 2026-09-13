"""Promotion gates: CAGR > 12% and beat incumbent CAGR and Sharpe."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_api.core.ppo_discovery import promotion as promotion_mod
from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    PPODiscoveryConfig,
    ppo_discovery_cost_contract,
)
from brain_api.core.ppo_discovery.evaluator import (
    reject_current_patchtst_on_old_weeks,
    weekly_net_cagr,
)
from brain_api.core.ppo_discovery.promotion import (
    evaluate_ppo_discovery_candidate,
    evaluate_ppo_discovery_promotion,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage


def _meta(**overrides):
    payload = {
        "ppo_discovery_schema_version": 1,
        "architecture": "temporal_set_factored",
        "asset_feature_names": list(ASSET_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "news_required": True,
        **ppo_discovery_cost_contract(),
    }
    payload.update(overrides)
    return payload


def _eval(**overrides):
    payload = {
        "test_cagr": 0.20,
        "test_sharpe": 1.5,
        "test_max_drawdown": 0.10,
        **ppo_discovery_cost_contract(),
    }
    payload.update(overrides)
    return payload


def test_inaugural_promotion_requires_cagr_above_floor_and_approved_by() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=0.13, test_sharpe=0.8),
        approved_by="razin",
    )
    assert check.is_healthy is True
    floor = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=0.12, test_sharpe=0.8),
        approved_by="razin",
    )
    assert floor.is_healthy is False
    assert any("12%" in reason for reason in floor.failure_reasons)
    missing = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(),
        approved_by="",
    )
    assert missing.is_healthy is False
    assert any("approved_by" in reason for reason in missing.failure_reasons)


def test_incumbent_requires_strictly_greater_cagr_and_sharpe() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=0.21, test_sharpe=1.6),
        approved_by="razin",
        incumbent_cagr=0.20,
        incumbent_sharpe=1.5,
    )
    assert check.is_healthy is True
    equal_cagr = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=0.20, test_sharpe=1.6),
        approved_by="razin",
        incumbent_cagr=0.20,
        incumbent_sharpe=1.5,
    )
    assert equal_cagr.is_healthy is False
    assert any("CAGR" in reason for reason in equal_cagr.failure_reasons)
    equal_sharpe = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=0.21, test_sharpe=1.5),
        approved_by="razin",
        incumbent_cagr=0.20,
        incumbent_sharpe=1.5,
    )
    assert equal_sharpe.is_healthy is False
    assert any("Sharpe" in reason for reason in equal_sharpe.failure_reasons)


def test_missing_incumbent_sharpe_raises() -> None:
    with pytest.raises(ValueError, match="incumbent test_sharpe is required"):
        evaluate_ppo_discovery_promotion(
            metadata=_meta(),
            evaluation=_eval(),
            approved_by="razin",
            incumbent_cagr=0.14,
            incumbent_sharpe=None,
        )


def test_candidate_health_does_not_require_approved_by() -> None:
    check = evaluate_ppo_discovery_candidate(_meta(), _eval())
    assert check.is_healthy is True
    failed = evaluate_ppo_discovery_candidate(
        _meta(), _eval(test_cagr=float("nan"), test_sharpe=1.0)
    )
    assert failed.is_healthy is False
    assert any("CAGR" in reason for reason in failed.failure_reasons)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("metadata", "broker_cost_model", "alpaca_us"),
        ("metadata", "training_nav_usd", 100_000.0),
        ("evaluation", "broker_cost_config", {}),
    ],
)
def test_promotion_rejects_cost_contract_mismatch(
    location: str, field: str, value: object
) -> None:
    evaluation = _eval()
    metadata = _meta()
    if location == "metadata":
        metadata[field] = value
    else:
        evaluation[field] = value

    check = evaluate_ppo_discovery_promotion(
        metadata=metadata,
        evaluation=evaluation,
        approved_by="razin",
    )

    assert check.is_healthy is False
    assert any(field in reason for reason in check.failure_reasons)


def test_reject_current_patchtst_on_old_weeks() -> None:
    with pytest.raises(PPODiscoveryError, match="PatchTST current"):
        reject_current_patchtst_on_old_weeks(True)
    reject_current_patchtst_on_old_weeks(False)


def test_cagr_formula() -> None:
    weekly = [0.01] * 52
    assert weekly_net_cagr(weekly) == pytest.approx(
        float(__import__("numpy").expm1(0.52))
    )


def test_promotion_rejects_non_finite_cagr_and_drawdown() -> None:
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_cagr=float("inf")),
        approved_by="razin",
    )
    assert check.is_healthy is False
    assert any("non-finite" in reason for reason in check.failure_reasons)
    check = evaluate_ppo_discovery_promotion(
        metadata=_meta(),
        evaluation=_eval(test_max_drawdown=1.5),
        approved_by="razin",
    )
    assert check.is_healthy is False
    assert any("drawdown" in reason for reason in check.failure_reasons)


def _storage_with_pointer(
    tmp_path: Path, version: str | None
) -> PPODiscoveryHalalNewModelStorage:
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    storage._model_path.mkdir(parents=True, exist_ok=True)
    if version is not None:
        (storage._model_path / "current").write_text(version)
    return storage


def _write_stub_version(
    storage: PPODiscoveryHalalNewModelStorage, version: str
) -> None:
    from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic

    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=8)
    policy = PPODiscoveryActorCritic(config)
    storage.write_artifacts(
        version,
        policy_state_dict=policy.state_dict(),
        pretrained_encoder_state_dict=policy.temporal.state_dict(),
        config=config,
        feature_scalers={},
        regime_hmm={"schema_version": 3},
        metadata={"config_hash": "stub", "promoted": False},
        universe_manifest={},
        news_manifest={},
        price_manifest={},
        experiment_lock={},
        evaluation={},
        seeds_ledger={"schema_version": 1, "seeds": {}},
    )


def test_cas_empty_ledger_uses_current_pointer(tmp_path: Path, monkeypatch) -> None:
    storage = _storage_with_pointer(tmp_path, "v1")
    _write_stub_version(storage, "v2")
    monkeypatch.setattr(
        promotion_mod, "maybe_upload_ppo_discovery", lambda *a, **k: None
    )
    monkeypatch.setattr(storage, "promote_version", lambda version: None)
    promotion_mod._commit_promotion(
        storage,
        "v2",
        approved_by="razin",
        expected_current_version="v1",
        config_changed=False,
        unpaired_acknowledged=False,
    )
    conn = promotion_mod._ledger(storage)
    row = conn.execute(
        "SELECT status FROM promotions WHERE version = ?", ("v2",)
    ).fetchone()
    assert row[0] == "promoted"


def test_cas_empty_expected_fails_when_pointer_is_set(
    tmp_path: Path, monkeypatch
) -> None:
    storage = _storage_with_pointer(tmp_path, "v1")
    monkeypatch.setattr(
        promotion_mod, "maybe_upload_ppo_discovery", lambda *a, **k: None
    )
    with pytest.raises(ValueError, match="current pointer"):
        promotion_mod._commit_promotion(
            storage,
            "v2",
            approved_by="razin",
            expected_current_version="",
            config_changed=False,
            unpaired_acknowledged=False,
        )


def test_pending_for_self_with_pointer_on_candidate_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    storage = _storage_with_pointer(tmp_path, "v2")
    _write_stub_version(storage, "v2")
    conn = promotion_mod._ledger(storage)
    conn.execute(
        "INSERT INTO promotions(version, approved_by, expected_current_version, "
        "promoted_at, status, config_changed, unpaired_acknowledged) "
        "VALUES (?, ?, ?, ?, 'pending', 0, 0)",
        ("v2", "razin", "v1", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    promoted: list[str] = []
    monkeypatch.setattr(
        promotion_mod, "maybe_upload_ppo_discovery", lambda *a, **k: None
    )
    monkeypatch.setattr(
        storage, "promote_version", lambda version: promoted.append(version)
    )
    promotion_mod._commit_promotion(
        storage,
        "v2",
        approved_by="razin",
        expected_current_version="v1",
        config_changed=False,
        unpaired_acknowledged=False,
    )
    assert promoted == []
    conn = promotion_mod._ledger(storage)
    row = conn.execute(
        "SELECT status FROM promotions WHERE version = ?", ("v2",)
    ).fetchone()
    assert row[0] == "promoted"
    rewritten = storage.load_artifacts("v2").metadata
    assert rewritten["promoted"] is True
    assert rewritten["approved_by"] == "razin"
    assert rewritten["failure_reasons"] == []


def test_promote_resumes_pending_without_rechecking_gates(
    tmp_path: Path, monkeypatch
) -> None:
    storage = _storage_with_pointer(tmp_path, "v2")
    _write_stub_version(storage, "v2")
    conn = promotion_mod._ledger(storage)
    conn.execute(
        "INSERT INTO promotions(version, approved_by, expected_current_version, "
        "promoted_at, status, config_changed, unpaired_acknowledged) "
        "VALUES (?, ?, ?, ?, 'pending', 0, 0)",
        ("v2", "razin", "", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(promotion_mod, "_smoke_load_candidate", lambda artifacts: None)
    monkeypatch.setattr(
        promotion_mod, "maybe_upload_ppo_discovery", lambda *a, **k: None
    )

    def _fail_health(*_args, **_kwargs):
        raise AssertionError("pending resume must not re-run promotion gates")

    monkeypatch.setattr(promotion_mod, "evaluate_ppo_discovery_promotion", _fail_health)
    result = promotion_mod.promote_ppo_discovery(
        storage,
        "v2",
        approved_by="razin",
        expected_current_version="",
    )
    assert result["promoted"] is True
    conn = promotion_mod._ledger(storage)
    row = conn.execute(
        "SELECT status FROM promotions WHERE version = ?", ("v2",)
    ).fetchone()
    assert row[0] == "promoted"


def test_inaugural_promote_uses_empty_expected(tmp_path: Path, monkeypatch) -> None:
    storage = _storage_with_pointer(tmp_path, None)
    _write_stub_version(storage, "v1")
    monkeypatch.setattr(
        promotion_mod, "maybe_upload_ppo_discovery", lambda *a, **k: None
    )
    monkeypatch.setattr(storage, "promote_version", lambda version: None)
    promotion_mod._commit_promotion(
        storage,
        "v1",
        approved_by="razin",
        expected_current_version="",
        config_changed=False,
        unpaired_acknowledged=False,
    )
    conn = promotion_mod._ledger(storage)
    row = conn.execute(
        "SELECT status FROM promotions WHERE version = ?", ("v1",)
    ).fetchone()
    assert row[0] == "promoted"
