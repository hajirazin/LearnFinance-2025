"""Seed ledger, OOM fatal-job, and pipeline delegation tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from brain_api.core.ppo_discovery.checkpoints import (
    load_seed_checkpoint,
    load_seed_partial_checkpoint,
    save_seed_checkpoint,
    save_seed_partial_checkpoint,
    train_recipe_hash,
)
from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.seed_ledger import (
    empty_seeds_ledger,
    failed_seed_ids,
    load_seeds_ledger,
    upsert_seed_row,
    write_seeds_ledger,
)
from brain_api.core.ppo_discovery.seed_training import train_ppo_discovery_seeds
from tests.core.ppo_discovery.factories import make_snapshot


def _config() -> PPODiscoveryConfig:
    return PPODiscoveryConfig(
        dropout=0.0,
        total_timesteps=4,
        ppo_epochs=1,
        minibatch_size=8,
        ppo_microbatch_size=8,
        seeds=(42, 123),
    )


def test_recipe_directory_uses_train_recipe_hash(tmp_path: Path) -> None:
    from dataclasses import replace

    from brain_api.core.ppo_discovery.checkpoints import seed_checkpoint_dir

    config = _config()
    directory = seed_checkpoint_dir(
        tmp_path,
        experiment_id="exp",
        snapshot_hash="sha256:abc",
        recipe_hash=train_recipe_hash(config),
    )
    assert train_recipe_hash(config) in str(directory)
    assert train_recipe_hash(replace(config, seeds=(7,))) == train_recipe_hash(config)
    frozen = replace(config, freeze_encoder_updates=10**9)
    assert train_recipe_hash(frozen) != train_recipe_hash(config)


def test_accelerator_oom_fails_entire_job_and_skips_remaining_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    policy = PPODiscoveryActorCritic(config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    save_seed_partial_checkpoint(
        tmp_path,
        seed=42,
        policy=policy,
        optimizer=optimizer,
        device=torch.device("cpu"),
        steps_done=2,
        episode_index=1,
        update_index=0,
        metadata={"train_recipe_hash": train_recipe_hash(config)},
    )
    calls: list[int] = []

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del policy, episode_fn, config, kwargs
        calls.append(int(seed))
        raise MemoryError("mps allocator oom")

    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.train_ppo_discovery",
        fake_train,
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.collect_closed_loop_rollout",
        lambda *args, **kwargs: [],
    )
    snapshot = make_snapshot(4)
    with pytest.raises(PPODiscoveryError, match="accelerator_out_of_memory"):
        train_ppo_discovery_seeds(
            pretrained_state=policy.state_dict(),
            train_weeks=[],
            val_weeks=[],
            snapshot=snapshot,
            ohlcv={},
            spy=MagicMock(),
            scalers={},
            config=config,
            ckpt_dir=tmp_path,
            checkpoint_expected={"train_recipe_hash": train_recipe_hash(config)},
            experiment_id="exp",
            device=torch.device("cpu"),
        )
    assert calls == [42]
    ledger = load_seeds_ledger(tmp_path)
    row = ledger["seeds"]["42"]
    assert row["status"] == "failed"
    assert row["fatal"] is True
    assert row["failure_scope"] == "job"
    partial = load_seed_partial_checkpoint(
        tmp_path,
        seed=42,
        expected={"train_recipe_hash": train_recipe_hash(config)},
    )
    assert partial is not None
    assert config.ppo_microbatch_size == 8


def test_non_oom_seed_failure_continues_to_next_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    policy = PPODiscoveryActorCritic(config)
    calls: list[int] = []

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del episode_fn, kwargs
        calls.append(int(seed))
        if int(seed) == 42:
            raise PPODiscoveryError("math blew up")
        return {"ppo_loss": 0.0, "timesteps": float(config.total_timesteps)}

    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.train_ppo_discovery",
        fake_train,
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.eval_weeks",
        lambda *args, **kwargs: {"cagr": 0.2, "sharpe": 0.1},
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.collect_closed_loop_rollout",
        lambda *args, **kwargs: [],
    )
    result = train_ppo_discovery_seeds(
        pretrained_state=policy.state_dict(),
        train_weeks=[],
        val_weeks=[],
        snapshot=make_snapshot(4),
        ohlcv={},
        spy=MagicMock(),
        scalers={},
        config=config,
        ckpt_dir=tmp_path,
        checkpoint_expected={},
        experiment_id="exp",
        device=torch.device("cpu"),
    )
    assert calls == [42, 123]
    assert result.failed_seeds == [42]
    assert result.selected_seed == 123
    assert result.ledger["seeds"]["42"]["status"] == "failed"
    assert result.ledger["seeds"]["123"]["status"] == "complete"


def test_undeclared_ledger_seed_is_ignored_by_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    policy = PPODiscoveryActorCritic(config)
    write_seeds_ledger(
        tmp_path,
        {
            "schema_version": 1,
            "seeds": {
                "7": {
                    "status": "complete",
                    "val_cagr": 0.99,
                    "val_sharpe": 9.0,
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            },
        },
    )

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del episode_fn, kwargs
        if int(seed) == 42:
            raise PPODiscoveryError("math blew up")
        return {"ppo_loss": 0.0, "timesteps": float(config.total_timesteps)}

    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.train_ppo_discovery",
        fake_train,
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.eval_weeks",
        lambda *args, **kwargs: {"cagr": 0.2, "sharpe": 0.1},
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.collect_closed_loop_rollout",
        lambda *args, **kwargs: [],
    )
    result = train_ppo_discovery_seeds(
        pretrained_state=policy.state_dict(),
        train_weeks=[],
        val_weeks=[],
        snapshot=make_snapshot(4),
        ohlcv={},
        spy=MagicMock(),
        scalers={},
        config=config,
        ckpt_dir=tmp_path,
        checkpoint_expected={},
        experiment_id="exp",
        device=torch.device("cpu"),
    )
    assert result.selected_seed == 123
    assert "7" not in result.seed_metrics
    assert result.failed_seeds == [42]
    assert result.ledger["seeds"]["7"]["status"] == "complete"


def _stub_seed_loop(monkeypatch: pytest.MonkeyPatch, fake_train, fake_eval) -> None:
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.train_ppo_discovery",
        fake_train,
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.eval_weeks",
        fake_eval,
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.seed_training.collect_closed_loop_rollout",
        lambda *args, **kwargs: [],
    )


def test_failed_seed_ids_include_validation_failed() -> None:
    ids = failed_seed_ids(
        {
            "schema_version": 1,
            "seeds": {
                "42": {"status": "validation_failed"},
                "123": {"status": "complete"},
                "7": {"status": "failed"},
            },
        }
    )
    assert ids == [7, 42]


def test_validation_failed_seed_is_reported_and_not_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    policy = PPODiscoveryActorCritic(config)

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del policy, episode_fn, kwargs
        return {"ppo_loss": 0.0, "timesteps": float(config.total_timesteps)}

    eval_calls = {"n": 0}

    def fake_eval(*args, **kwargs):
        del args, kwargs
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            raise PPODiscoveryError("val blew up")
        return {"cagr": 0.2, "sharpe": 0.1}

    _stub_seed_loop(monkeypatch, fake_train, fake_eval)
    result = train_ppo_discovery_seeds(
        pretrained_state=policy.state_dict(),
        train_weeks=[],
        val_weeks=[],
        snapshot=make_snapshot(4),
        ohlcv={},
        spy=MagicMock(),
        scalers={},
        config=config,
        ckpt_dir=tmp_path,
        checkpoint_expected={"protocol_digest": "job"},
        experiment_id="exp",
        device=torch.device("cpu"),
    )
    assert result.failed_seeds == [42]
    assert result.selected_seed == 123
    assert result.ledger["seeds"]["42"]["status"] == "validation_failed"
    assert "42" not in result.seed_metrics


def test_stale_trained_checkpoint_does_not_skip_retrain_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    policy = PPODiscoveryActorCritic(config)
    expected = {"protocol_digest": "job"}
    save_seed_checkpoint(
        tmp_path,
        seed=42,
        policy=policy,
        metadata={"status": "trained", "protocol_digest": "stale"},
    )
    calls: list[int] = []

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del policy, episode_fn, kwargs
        calls.append(int(seed))
        if int(seed) == 42:
            raise PPODiscoveryError("math blew up")
        return {"ppo_loss": 0.0, "timesteps": float(config.total_timesteps)}

    _stub_seed_loop(
        monkeypatch, fake_train, lambda *a, **k: {"cagr": 0.2, "sharpe": 0.1}
    )
    kwargs = {
        "pretrained_state": policy.state_dict(),
        "train_weeks": [],
        "val_weeks": [],
        "snapshot": make_snapshot(4),
        "ohlcv": {},
        "spy": MagicMock(),
        "scalers": {},
        "config": config,
        "ckpt_dir": tmp_path,
        "checkpoint_expected": expected,
        "experiment_id": "exp",
        "device": torch.device("cpu"),
    }
    result = train_ppo_discovery_seeds(**kwargs)
    assert result.ledger["seeds"]["42"]["status"] == "failed"
    assert result.failed_seeds == [42]
    assert load_seed_checkpoint(tmp_path, seed=42, expected=expected) is None
    train_ppo_discovery_seeds(**kwargs)
    assert calls == [42, 123, 42]


def test_undeclared_validation_failed_seed_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    policy = PPODiscoveryActorCritic(config)
    write_seeds_ledger(
        tmp_path,
        {
            "schema_version": 1,
            "seeds": {
                "7": {
                    "status": "validation_failed",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            },
        },
    )

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del policy, episode_fn, seed, kwargs
        return {"ppo_loss": 0.0, "timesteps": float(config.total_timesteps)}

    _stub_seed_loop(
        monkeypatch, fake_train, lambda *a, **k: {"cagr": 0.2, "sharpe": 0.1}
    )
    result = train_ppo_discovery_seeds(
        pretrained_state=policy.state_dict(),
        train_weeks=[],
        val_weeks=[],
        snapshot=make_snapshot(4),
        ohlcv={},
        spy=MagicMock(),
        scalers={},
        config=config,
        ckpt_dir=tmp_path,
        checkpoint_expected={},
        experiment_id="exp",
        device=torch.device("cpu"),
    )
    assert 7 not in result.failed_seeds
    assert result.ledger["seeds"]["7"]["status"] == "validation_failed"


def test_retry_progress_does_not_report_current_seed_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PPODiscoveryConfig(
        dropout=0.0,
        total_timesteps=4,
        ppo_epochs=1,
        minibatch_size=8,
        ppo_microbatch_size=8,
        seeds=(42, 123, 2026),
    )
    policy = PPODiscoveryActorCritic(config)
    ledger = empty_seeds_ledger()
    for seed in (42, 123, 2026):
        ledger = upsert_seed_row(ledger, seed, status="failed")
    write_seeds_ledger(tmp_path, ledger)
    captured: list[dict] = []

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del policy, episode_fn, seed, kwargs
        return {"ppo_loss": 0.0, "timesteps": float(config.total_timesteps)}

    _stub_seed_loop(
        monkeypatch, fake_train, lambda *a, **k: {"cagr": 0.2, "sharpe": 0.1}
    )
    train_ppo_discovery_seeds(
        pretrained_state=policy.state_dict(),
        train_weeks=[],
        val_weeks=[],
        snapshot=make_snapshot(4),
        ohlcv={},
        spy=MagicMock(),
        scalers={},
        config=config,
        ckpt_dir=tmp_path,
        checkpoint_expected={},
        experiment_id="exp",
        device=torch.device("cpu"),
        progress=captured.append,
    )
    first = next(
        payload
        for payload in captured
        if payload.get("seed_status") == "in_progress" and payload.get("seed") == 42
    )
    assert 42 not in first["failed_seeds"]
    assert 123 in first["failed_seeds"]
    assert 2026 in first["failed_seeds"]
