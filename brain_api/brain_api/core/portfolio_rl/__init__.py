"""Portfolio RL shared core module.

This module provides the shared components for RL-based portfolio allocation:
- Weekly portfolio environment (long-only, simplex weights)
- Transaction cost modeling
- Constraint enforcement (long-only simplex and cash buffer)
- State/feature building with dual forecast features (LSTM + PatchTST)
- Evaluation helpers (walk-forward, metrics)
- Data loading (news sentiment, fundamentals)
- Walk-forward forecast generation

Used by `sac` unified agent.
"""

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    LegCost,
    RebalanceCost,
    compute_ibkr_leg_cost,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.portfolio_rl.config import DEFAULT_RL_BASE_CONFIG, RLBaseConfig
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
    build_state_vector_strict_v2,
)
from brain_api.core.portfolio_rl.walkforward import (
    SnapshotInferenceError,
    SnapshotUnavailableError,
    build_dual_forecast_features,
    build_forecast_features,
)

__all__ = [
    "DEFAULT_RL_BASE_CONFIG",
    "EvaluationMetrics",
    "IBKRSingaporeCostConfig",
    "LegCost",
    "PortfolioEnv",
    "PortfolioScaler",
    "PortfolioState",
    "RLBaseConfig",
    "RebalanceCost",
    "SnapshotInferenceError",
    "SnapshotUnavailableError",
    "StateSchema",
    "align_signals_to_weekly",
    "apply_softmax_to_weights",
    "build_dual_forecast_features",
    "build_forecast_features",
    "build_rl_training_signals",
    "build_state_vector",
    "build_state_vector_strict_v2",
    "compute_baseline_metrics",
    "compute_cagr",
    "compute_hit_rate",
    "compute_hrp_baseline_metrics",
    "compute_ibkr_leg_cost",
    "compute_ibkr_rebalance_cost",
    "compute_max_drawdown",
    "compute_portfolio_return",
    "compute_reward",
    "compute_sharpe_ratio",
    "compute_transaction_cost",
    "compute_turnover",
    "enforce_constraints",
    "evaluate_policy",
    "load_historical_fundamentals",
    "load_historical_news_sentiment",
]
