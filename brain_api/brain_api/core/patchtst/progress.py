"""Framework-independent progress and cancellation for weekly PatchTST."""

import threading
from collections.abc import Callable
from typing import Any

from brain_api.core.training_utils import TrainingCancelledError

ProgressCallback = Callable[[dict[str, Any]], None]


def report_progress(callback: ProgressCallback | None, **details) -> None:
    if callback is not None:
        callback(details)


def check_cancelled(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise TrainingCancelledError("Weekly PatchTST training cancelled")
