"""Sidecar heartbeats for sync activities that block on I/O or sleep."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from temporalio import activity

DEFAULT_HEARTBEAT_SECONDS = 30.0


@contextmanager
def heartbeat_until_done(
    *details: Any, interval_seconds: float = DEFAULT_HEARTBEAT_SECONDS
) -> Iterator[None]:
    """Keep heartbeats flowing while the caller blocks.

    Sync activities run in a thread pool. ``activity.heartbeat()`` from
    that thread is recorded, but a blocking ``httpx`` read or
    ``time.sleep`` prevents the next beat. Temporal then fires
    ``heartbeat_timeout`` even though ``start_to_close`` still has hours
    left. A sidecar thread on a copied activity context beats during
    those blocks.
    """
    stop = threading.Event()
    ctx = contextvars.copy_context()

    def _beat() -> None:
        while not stop.wait(interval_seconds):
            activity.heartbeat(*details)

    worker = threading.Thread(target=ctx.run, args=(_beat,), daemon=True)
    worker.start()
    activity.heartbeat(*details)
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=1.0)
