"""Training cancellation must precede the server's background-task drain."""

import signal
import threading
from unittest.mock import Mock

from brain_api.shutdown import relay_shutdown_signals


def test_shutdown_signal_sets_event_before_delegating_and_restores_handlers(
    monkeypatch,
):
    event = threading.Event()
    observed = []

    def server_handler(signum, frame):
        observed.append((signum, event.is_set()))

    handlers = dict.fromkeys((signal.SIGINT, signal.SIGTERM), server_handler)
    monkeypatch.setattr(signal, "getsignal", handlers.__getitem__)
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)
    with relay_shutdown_signals(event):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert observed == [(signal.SIGTERM, True)]
    assert all(handler is server_handler for handler in handlers.values())


def test_worker_thread_does_not_install_signal_handlers(monkeypatch):
    install = Mock()
    monkeypatch.setattr(signal, "signal", install)

    def run():
        with relay_shutdown_signals(threading.Event()):
            pass

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    install.assert_not_called()


def test_default_or_ignored_signal_handlers_are_preserved(monkeypatch):
    install = Mock()
    monkeypatch.setattr(signal, "getsignal", lambda _: signal.SIG_IGN)
    monkeypatch.setattr(signal, "signal", install)
    with relay_shutdown_signals(threading.Event()):
        pass
    install.assert_not_called()
