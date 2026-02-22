"""Task modules for weekly forecast email workflow."""

from flows.tasks.execution import (
    generate_orders_hrp,
    generate_orders_ppo,
    generate_orders_sac,
    store_experience_ppo,
    store_experience_sac,
    update_execution_ppo,
    update_execution_sac,
)
from flows.tasks.inference import (
    allocate_hrp,
    get_fundamentals,
    get_lstm_forecast,
    get_news_sentiment,
    get_patchtst_forecast,
    infer_ppo,
    infer_sac,
)
from flows.tasks.portfolio import (
    get_active_symbols,
    get_hrp_portfolio,
    get_order_history_ppo,
    get_order_history_sac,
    get_ppo_portfolio,
    get_sac_portfolio,
    submit_orders_hrp,
    submit_orders_ppo,
    submit_orders_sac,
)
from flows.tasks.reporting import (
    generate_summary,
    send_weekly_email,
)

__all__ = [
    # Portfolio
    "get_active_symbols",
    "get_ppo_portfolio",
    "get_sac_portfolio",
    "get_hrp_portfolio",
    "submit_orders_ppo",
    "submit_orders_sac",
    "submit_orders_hrp",
    "get_order_history_ppo",
    "get_order_history_sac",
    # Inference
    "get_fundamentals",
    "get_news_sentiment",
    "get_lstm_forecast",
    "get_patchtst_forecast",
    "infer_ppo",
    "infer_sac",
    "allocate_hrp",
    # Execution
    "generate_orders_ppo",
    "generate_orders_sac",
    "generate_orders_hrp",
    "store_experience_ppo",
    "store_experience_sac",
    "update_execution_ppo",
    "update_execution_sac",
    # Reporting
    "generate_summary",
    "send_weekly_email",
]
