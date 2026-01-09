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

from brain_api.core.portfolio_rl.config import DEFAULT_PPO_CONFIG, PPOConfig
from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    compute_turnover,
    enforce_constraints,
)
from brain_api.core.portfolio_rl.data_loading import (
    align_signals_to_weekly,
    build_rl_training_signals,
    load_historical_fundamentals,
    load_historical_news_sentiment,
)
from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.eval import (
    EvaluationMetrics,
    compute_baseline_metrics,
    compute_cagr,
    compute_hit_rate,
    compute_hrp_baseline_metrics,
    compute_max_drawdown,
    compute_sharpe_ratio,
    evaluate_policy,
    evaluate_ppo_for_promotion,
)
from brain_api.core.portfolio_rl.rewards import (
    compute_portfolio_return,
    compute_reward,
    compute_transaction_cost,
)
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import (
    PortfolioState,
    StateSchema,
    build_state_vector,
)
from brain_api.core.portfolio_rl.walkforward import (
    build_forecast_features,
    generate_walkforward_forecasts_simple,
)

__all__ = [
    "DEFAULT_PPO_CONFIG",
    "EvaluationMetrics",
    # Config
    "PPOConfig",
    # Environment
    "PortfolioEnv",
    # Scaler
    "PortfolioScaler",
    "PortfolioState",
    "StateSchema",
    "align_signals_to_weekly",
    "apply_softmax_to_weights",
    # Walk-forward forecasts
    "build_forecast_features",
    "build_rl_training_signals",
    # State
    "build_state_vector",
    "compute_baseline_metrics",
    "compute_cagr",
    "compute_hit_rate",
    "compute_hrp_baseline_metrics",
    "compute_max_drawdown",
    # Rewards
    "compute_portfolio_return",
    "compute_reward",
    # Evaluation
    "compute_sharpe_ratio",
    "compute_transaction_cost",
    "compute_turnover",
    # Constraints
    "enforce_constraints",
    "evaluate_policy",
    "evaluate_ppo_for_promotion",
    "generate_walkforward_forecasts_simple",
    "load_historical_fundamentals",
    # Data loading
    "load_historical_news_sentiment",
]
