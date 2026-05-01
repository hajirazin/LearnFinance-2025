"""Temporal worker entry point.

Registers all workflows and activities, then starts polling the task queue.

Usage:
    cd temporal && uv run python -m worker
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities.execution import (
    generate_orders_alpha_hrp,
    generate_orders_dhrp,
    generate_orders_sac,
    store_experience_sac,
    update_execution_sac,
)
from activities.inference import (
    allocate_hrp,
    get_fundamentals,
    get_lstm_forecast,
    get_news_sentiment,
    get_patchtst_forecast,
    infer_sac,
    record_final_weights,
    score_halal_india_with_patchtst,
    score_halal_new_with_patchtst,
    select_rank_band_top_n,
    select_sticky_top_n,
)
from activities.portfolio import (
    check_order_statuses,
    get_active_symbols,
    get_dhrp_portfolio,
    get_hrp_portfolio,
    get_order_history_sac,
    get_order_history_sac_halal,
    get_sac_halal_portfolio,
    get_sac_portfolio,
    resolve_next_attempt,
    submit_orders_dhrp,
    submit_orders_hrp,
    submit_orders_sac,
    submit_orders_sac_halal,
)
from activities.reporting import (
    generate_double_hrp_summary,
    generate_india_alpha_hrp_summary,
    generate_summary,
    generate_us_alpha_hrp_summary,
    generate_us_double_hrp_summary,
    send_double_hrp_email,
    send_india_alpha_hrp_email,
    send_us_alpha_hrp_email,
    send_us_double_hrp_email,
    send_weekly_email,
)
from activities.training import (
    fetch_halal_filtered_universe,
    fetch_halal_new_universe,
    fetch_halal_universe,
    fetch_nifty_shariah_500_universe,
    generate_forecasters_training_summary,
    generate_india_training_summary,
    generate_sac_training_summary,
    refresh_training_data,
    send_forecasters_training_email,
    send_india_training_email,
    send_sac_training_email,
    train_india_patchtst,
    train_lstm,
    train_patchtst,
    train_sac,
)
from activities.training import (
    fetch_halal_india_universe as fetch_halal_india_universe_training,
)
from workflows.india_double_hrp import IndiaDoubleHRPWorkflow
from workflows.india_weekly_allocation import IndiaWeeklyAllocationWorkflow
from workflows.india_weekly_training import IndiaWeeklyTrainingWorkflow
from workflows.us_alpha_hrp import USAlphaHRPWorkflow
from workflows.us_double_hrp import USDoubleHRPWorkflow
from workflows.us_forecasters_training import USForecastersTrainingWorkflow
from workflows.us_sac_halal_allocation import USSACHalalAllocationWorkflow
from workflows.us_sac_halal_training import USSACHalalTrainingWorkflow
from workflows.us_sac_training import USSACTrainingWorkflow
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow

TASK_QUEUE = os.environ.get("TEMPORAL_TASK_QUEUE", "learnfinance")
TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
MAX_CONCURRENT_ACTIVITIES = int(
    os.environ.get("TEMPORAL_MAX_CONCURRENT_ACTIVITIES", "10")
)

ALL_WORKFLOWS = [
    IndiaDoubleHRPWorkflow,
    IndiaWeeklyAllocationWorkflow,
    IndiaWeeklyTrainingWorkflow,
    USAlphaHRPWorkflow,
    USDoubleHRPWorkflow,
    USForecastersTrainingWorkflow,
    USSACHalalAllocationWorkflow,
    USSACHalalTrainingWorkflow,
    USSACTrainingWorkflow,
    USWeeklyAllocationWorkflow,
]

ALL_ACTIVITIES = [
    # Inference / signals / allocators
    get_fundamentals,
    get_news_sentiment,
    get_lstm_forecast,
    get_patchtst_forecast,
    infer_sac,
    allocate_hrp,
    select_sticky_top_n,
    select_rank_band_top_n,
    score_halal_new_with_patchtst,
    score_halal_india_with_patchtst,
    record_final_weights,
    # Portfolio / orders
    get_active_symbols,
    get_sac_portfolio,
    get_sac_halal_portfolio,
    get_hrp_portfolio,
    get_dhrp_portfolio,
    submit_orders_sac,
    submit_orders_sac_halal,
    submit_orders_hrp,
    submit_orders_dhrp,
    get_order_history_sac,
    get_order_history_sac_halal,
    check_order_statuses,
    resolve_next_attempt,
    # Execution / experience
    generate_orders_sac,
    generate_orders_dhrp,
    generate_orders_alpha_hrp,
    store_experience_sac,
    update_execution_sac,
    # Reporting
    generate_summary,
    send_weekly_email,
    generate_india_alpha_hrp_summary,
    send_india_alpha_hrp_email,
    generate_double_hrp_summary,
    send_double_hrp_email,
    generate_us_double_hrp_summary,
    send_us_double_hrp_email,
    generate_us_alpha_hrp_summary,
    send_us_alpha_hrp_email,
    # Training
    fetch_halal_new_universe,
    fetch_halal_filtered_universe,
    fetch_halal_universe,
    fetch_nifty_shariah_500_universe,
    fetch_halal_india_universe_training,
    refresh_training_data,
    train_lstm,
    train_patchtst,
    train_sac,
    train_india_patchtst,
    generate_forecasters_training_summary,
    send_forecasters_training_email,
    generate_sac_training_summary,
    send_sac_training_email,
    generate_india_training_summary,
    send_india_training_email,
]


async def main():
    client = await Client.connect(
        TEMPORAL_ADDRESS, data_converter=pydantic_data_converter
    )
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
        activity_executor=ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ACTIVITIES),
        max_concurrent_activities=MAX_CONCURRENT_ACTIVITIES,
    )
    print(f"Worker started on task queue '{TASK_QUEUE}' ({TEMPORAL_ADDRESS})")
    print(f"  Max concurrent activities: {MAX_CONCURRENT_ACTIVITIES}")
    print(f"  Workflows: {len(ALL_WORKFLOWS)}")
    print(f"  Activities: {len(ALL_ACTIVITIES)}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
