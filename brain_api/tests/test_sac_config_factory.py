"""Tests for the per-bucket SAC config factory.

The factory ``make_sac_config_for_n_stocks`` resizes the SAC action
space to match a bucket-resolved symbol slate. Two parallel A/B SAC
buckets share the same FastAPI process: ``halal_filtered`` (fixed
15 names, byte-equivalent to the legacy default) and ``halal``
(yfinance ETF top-holdings, variable size). These tests pin the
math contract:

- ``target_entropy = -(n_stocks + 1)`` for any ``n_stocks``.
- ``n_stocks == base.n_stocks`` returns a config byte-equivalent to
  ``base`` (regression guard for halal_filtered's compute_version
  hash and ``current`` artifact lineage).
- The factory must NOT mutate the input or the global
  ``DEFAULT_SAC_CONFIG`` (two parallel runs share a process).
"""

import copy

import pytest

from brain_api.core.portfolio_rl.sac_config import (
    DEFAULT_SAC_BASE_CONFIG,
    SACBaseConfig,
    make_sac_base_config_for_n_stocks,
)
from brain_api.core.sac.config import (
    DEFAULT_SAC_CONFIG,
    SACConfig,
    make_sac_config_for_n_stocks,
)


class TestMakeSACBaseConfigForNStocks:
    """Pure-math contract for the base helper."""

    @pytest.mark.parametrize("n_stocks", [5, 12, 13, 14, 15, 20, 30])
    def test_target_entropy_equals_negative_action_dim(self, n_stocks: int) -> None:
        """Persisted fallback reflects the fixed 31-way network envelope."""
        cfg = make_sac_base_config_for_n_stocks(DEFAULT_SAC_BASE_CONFIG, n_stocks)
        assert cfg.n_stocks == n_stocks
        assert cfg.target_entropy == -31.0
        assert cfg.action_dim == 31

    def test_n_stocks_equal_to_base_round_trips_byte_equivalent(self) -> None:
        """Calling with the base's own n_stocks must reproduce the base.

        This is the halal_filtered byte-equivalence regression guard:
        the existing ``DEFAULT_SAC_BASE_CONFIG`` has ``n_stocks=15``
        and ``target_entropy=-16.0``, so the halal_filtered bucket
        (slate pinned to 15) must keep producing the exact same
        config object after the refactor.
        """
        cfg = make_sac_base_config_for_n_stocks(
            DEFAULT_SAC_BASE_CONFIG, DEFAULT_SAC_BASE_CONFIG.n_stocks
        )
        assert cfg == DEFAULT_SAC_BASE_CONFIG

    def test_returns_new_instance_does_not_mutate_input(self) -> None:
        """Per AGENTS.md rule #2 (math correctness): no shared mutable state.

        Two parallel A/B training runs must not race on the global
        default. Verify both that the input is unchanged AND that the
        return value is a distinct dataclass instance.
        """
        snapshot = copy.deepcopy(DEFAULT_SAC_BASE_CONFIG)
        cfg = make_sac_base_config_for_n_stocks(DEFAULT_SAC_BASE_CONFIG, 13)
        assert snapshot == DEFAULT_SAC_BASE_CONFIG
        assert cfg is not DEFAULT_SAC_BASE_CONFIG

    def test_inherits_all_other_hyperparameters(self) -> None:
        """Only n_stocks and target_entropy change; everything else is verbatim."""
        cfg = make_sac_base_config_for_n_stocks(DEFAULT_SAC_BASE_CONFIG, 13)
        for field in (
            "hidden_sizes",
            "activation",
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "tau",
            "gamma",
            "auto_entropy_tuning",
            "init_alpha",
            "buffer_size",
            "batch_size",
            "gradient_steps_per_env_step",
            "warmup_steps",
            "total_timesteps",
            "weight_decay",
            "max_grad_norm",
            "q_value_clip",
            "normalize_rewards",
            "cost_bps",
            "cash_buffer",
            "reward_scale",
            "seed",
            "validation_years",
            "min_cagr_improvement",
            "cost_config",
        ):
            assert getattr(cfg, field) == getattr(DEFAULT_SAC_BASE_CONFIG, field), (
                f"{field} unexpectedly differs"
            )

    def test_zero_or_negative_n_stocks_raises(self) -> None:
        """No silent fallback for impossible action spaces (AGENTS.md rule #1)."""
        with pytest.raises(ValueError, match=r"\[1, 30\]"):
            make_sac_base_config_for_n_stocks(DEFAULT_SAC_BASE_CONFIG, 0)
        with pytest.raises(ValueError, match=r"\[1, 30\]"):
            make_sac_base_config_for_n_stocks(DEFAULT_SAC_BASE_CONFIG, -3)


class TestMakeSACConfigForNStocks:
    """SACConfig-layer wrapper; preserves SACConfig-only fields."""

    def test_preserves_training_years_and_n_eval_folds(self) -> None:
        """The wrapper must not drop SACConfig's training-only fields."""
        custom = SACConfig(training_years=7, n_eval_folds=5)
        cfg = make_sac_config_for_n_stocks(custom, 12)
        assert cfg.training_years == 7
        assert cfg.n_eval_folds == 5
        assert cfg.n_stocks == 12
        assert cfg.target_entropy == -31.0
        assert isinstance(cfg, SACConfig)

    def test_default_round_trips_at_n_stocks_15(self) -> None:
        """Halal_filtered byte-equivalence at the SACConfig layer."""
        cfg = make_sac_config_for_n_stocks(
            DEFAULT_SAC_CONFIG, DEFAULT_SAC_CONFIG.n_stocks
        )
        assert cfg == DEFAULT_SAC_CONFIG

    @pytest.mark.parametrize("n_stocks", [12, 13, 14, 15])
    def test_typical_halal_sizes_resize_correctly(self, n_stocks: int) -> None:
        """Cover the realistic halal yfinance-top-holdings range."""
        cfg = make_sac_config_for_n_stocks(DEFAULT_SAC_CONFIG, n_stocks)
        assert cfg.n_stocks == n_stocks
        assert cfg.target_entropy == -31.0

    def test_zero_or_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, 30\]"):
            make_sac_config_for_n_stocks(DEFAULT_SAC_CONFIG, 0)


class TestPreservedDefaults:
    """Snapshot-style guards on the defaults the factory expects."""

    def test_default_base_config_n_stocks_is_15(self) -> None:
        """halal_filtered's contract relies on this default staying at 15."""
        assert DEFAULT_SAC_BASE_CONFIG.n_stocks == 15
        assert DEFAULT_SAC_BASE_CONFIG.target_entropy == -31.0

    def test_default_sac_config_inherits_base_defaults(self) -> None:
        assert isinstance(DEFAULT_SAC_CONFIG, SACBaseConfig)
        assert DEFAULT_SAC_CONFIG.n_stocks == DEFAULT_SAC_BASE_CONFIG.n_stocks
        assert (
            DEFAULT_SAC_CONFIG.target_entropy == DEFAULT_SAC_BASE_CONFIG.target_entropy
        )
