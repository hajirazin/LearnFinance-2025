"""Process-wide SEC EDGAR fair-access throttle.

SEC fair-access ceiling is 10 requests/second aggregate per requester across
all EDGAR hosts. This module targets ~8 req/s (0.125s min interval) shared by
eligibility, CompanyFacts, and filings enrichment clients in one process.
"""

from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_LAST_REQUEST_MONOTONIC: float | None = None
DEFAULT_MIN_INTERVAL_SECONDS = 0.125


def wait_for_sec_slot(
    *, min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS
) -> None:
    """Block until the shared SEC request slot is available, then claim it."""
    global _LAST_REQUEST_MONOTONIC
    with _LOCK:
        now = time.monotonic()
        if _LAST_REQUEST_MONOTONIC is not None:
            elapsed = now - _LAST_REQUEST_MONOTONIC
            if elapsed < min_interval_seconds:
                time.sleep(min_interval_seconds - elapsed)
                now = time.monotonic()
        _LAST_REQUEST_MONOTONIC = now


def reset_sec_rate_limit_for_tests() -> None:
    """Reset shared throttle state (tests only)."""
    global _LAST_REQUEST_MONOTONIC
    with _LOCK:
        _LAST_REQUEST_MONOTONIC = None
