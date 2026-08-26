"""In-process materialization coalescing for a shared (window, scorer) key."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_REVISION,
    NewsWindow,
)


@dataclass(frozen=True)
class MaterializationKey:
    provider: str
    start_exclusive: str
    end_inclusive: str
    schema_version: int
    sentiment_model_revision: str


def materialization_key(
    window: NewsWindow,
    *,
    provider: str = NEWS_PROVIDER,
    schema_version: int = NEWS_SCHEMA_VERSION,
    sentiment_model_revision: str = NEWS_SENTIMENT_REVISION,
) -> MaterializationKey:
    return MaterializationKey(
        provider=provider,
        start_exclusive=window.start_exclusive.isoformat(),
        end_inclusive=window.end_inclusive.isoformat(),
        schema_version=schema_version,
        sentiment_model_revision=sentiment_model_revision,
    )


@dataclass
class _InFlight:
    event: threading.Event = field(default_factory=threading.Event)
    symbols: set[str] = field(default_factory=set)
    done: set[str] = field(default_factory=set)
    error: BaseException | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class MaterializationCoordinator:
    """One in-flight job per MaterializationKey; waiters union extra symbols."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[MaterializationKey, _InFlight] = {}

    def run(
        self,
        key: MaterializationKey,
        symbols: Sequence[str],
        worker: Callable[[list[str], _InFlight], None],
    ) -> None:
        requested = set(symbols)
        with self._lock:
            job = self._inflight.get(key)
            leader = job is None
            if job is None:
                job = _InFlight(symbols=set(requested))
                self._inflight[key] = job
            else:
                with job.lock:
                    job.symbols.update(requested)
        if leader:
            try:
                while True:
                    with job.lock:
                        todo = sorted(job.symbols - job.done)
                    if not todo:
                        break
                    worker(todo, job)
                    with job.lock:
                        if not (job.symbols - job.done):
                            break
            except BaseException as exc:
                job.error = exc
                raise
            finally:
                job.event.set()
                with self._lock:
                    if self._inflight.get(key) is job:
                        del self._inflight[key]
        else:
            job.event.wait()
            if job.error is not None:
                raise job.error
            leftover = requested - job.done
            if leftover:
                self.run(key, sorted(leftover), worker)


COORDINATOR = MaterializationCoordinator()
