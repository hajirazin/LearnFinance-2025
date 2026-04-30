"""Time-skipping ``WorkflowEnvironment`` + ``Worker`` boilerplate.

Encapsulated here so each test scenario can ``async with`` it without
re-declaring the executor / data converter setup.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker


@asynccontextmanager
async def worker_with_activities(workflow_classes: list, activities: Iterable[Any]):
    """Yield a started ``WorkflowEnvironment`` + ``Worker`` for tests."""
    env = await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    )
    try:
        worker = Worker(
            env.client,
            task_queue="test-queue",
            workflows=list(workflow_classes),
            activities=list(activities),
            activity_executor=ThreadPoolExecutor(),
        )
        async with worker:
            yield env
    finally:
        await env.shutdown()
