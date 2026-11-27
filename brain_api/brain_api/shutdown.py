"""Signal training cancellation before the ASGI server drains background work."""

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def relay_shutdown_signals(event: threading.Event) -> Iterator[None]:
    """Wrap installed server handlers while preserving their shutdown behavior.

    ASGI lifespan shutdown runs after Uvicorn waits for background tasks, too
    late to stop a fit cooperatively. Set its event at the original signal.
    Test clients may run lifespan in a worker thread, where signal installation
    is prohibited; they can still set the event explicitly.
    """
    installed = []
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(signum)
            if not callable(previous):
                continue

            def relay(received, frame, handler=previous):
                event.set()
                handler(received, frame)

            signal.signal(signum, relay)
            installed.append((signum, previous, relay))
    try:
        yield
    finally:
        for signum, previous, relay in installed:
            if signal.getsignal(signum) is relay:
                signal.signal(signum, previous)
