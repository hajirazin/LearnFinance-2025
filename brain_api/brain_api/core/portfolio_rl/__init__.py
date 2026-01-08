"""Portfolio RL shared core module.

This module provides the shared components for PPO/SAC-based portfolio allocation:
- Weekly portfolio environment (long-only, simplex weights)
- Transaction cost modeling
- Constraint enforcement (cash buffer, max position size)
- State/feature building with injectable forecast feature
- Evaluation helpers (walk-forward, metrics)
- Data loading (news sentiment, fundamentals)
- Walk-forward forecast generation

Used by `ppo_lstm`, `ppo_patchtst`, `sac_lstm`, and `sac_patchtst` variants.
"""

from brain_api.core.portfolio_rl.config import PPOConfig, DEFAULT_PPO_CONFIG
from brain_api.core.portfolio_rl.constraints import (
    enforce_constraints,
    compute_turnover,
    apply_softmax_to_weights,
)
from brain_api.core.portfolio_rl.rewards import (
    compute_portfolio_return,
    compute_transaction_cost,
    compute_reward,
)
from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.state import (
    build_state_vector,
    StateSchema,
    PortfolioState,
)
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.eval import (
    compute_sharpe_ratio,
    compute_cagr,
    compute_max_drawdown,
    compute_hit_rate,
    EvaluationMetrics,
    evaluate_policy,
    compute_baseline_metrics,
    compute_hrp_baseline_metrics,
    evaluate_ppo_for_promotion,
)
from brain_api.core.portfolio_rl.data_loading import (
    load_historical_news_sentiment,
    load_historical_fundamentals,
    align_signals_to_weekly,
    build_rl_training_signals,
)
from brain_api.core.portfolio_rl.walkforward import (
    build_forecast_features,
    generate_walkforward_forecasts_simple,
)

__all__ = [
    # Config
    "PPOConfig",
    "DEFAULT_PPO_CONFIG",
    # Constraints
    "enforce_constraints",
    "compute_turnover",
    "apply_softmax_to_weights",
    # Rewards
    "compute_portfolio_return",
    "compute_transaction_cost",
    "compute_reward",
    # Environment
    "PortfolioEnv",
    # State
    "build_state_vector",
    "StateSchema",
    "PortfolioState",
    # Scaler
    "PortfolioScaler",
    # Evaluation
    "compute_sharpe_ratio",
    "compute_cagr",
    "compute_max_drawdown",
    "compute_hit_rate",
    "EvaluationMetrics",
    "evaluate_policy",
    "compute_baseline_metrics",
    "compute_hrp_baseline_metrics",
    "evaluate_ppo_for_promotion",
    # Data loading
    "load_historical_news_sentiment",
    "load_historical_fundamentals",
    "align_signals_to_weekly",
    "build_rl_training_signals",
    # Walk-forward forecasts
    "build_forecast_features",
    "generate_walkforward_forecasts_simple",
]

