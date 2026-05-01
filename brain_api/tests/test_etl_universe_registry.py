"""Pure-function tests for the ETL universe registry.

Endpoint-level (HTTP 422) behaviour for unknown universes is exercised
in ``tests/test_etl_news_sentiment.py``; this module covers the
underlying registry contract.
"""

import threading
from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from brain_api.etl import universe_registry
from brain_api.etl.universe_registry import (
    ETLUniverseConfig,
    UnknownETLUniverseError,
    get_etl_symbols,
    list_universes,
)


@contextmanager
def _patched_resolver(universe: str, resolver: Callable):
    """Swap a registered universe's resolver for the duration of the test.

    ``BucketConfig`` is frozen and the resolver is captured at import
    time, so we replace the whole ``ETLUniverseConfig`` entry to mimic
    what the real registry would do for a future universe.
    """
    original = universe_registry._REGISTRY[universe]
    universe_registry._REGISTRY[universe] = ETLUniverseConfig(
        universe=universe,
        symbols_resolver=resolver,
        accepts_shutdown_event=original.accepts_shutdown_event,
    )
    try:
        yield
    finally:
        universe_registry._REGISTRY[universe] = original


class TestListUniverses:
    """Tests for ``list_universes`` -- shape of the registered set."""

    def test_includes_all_known_universes(self) -> None:
        """All currently expected universes are registered."""
        registered = list_universes()
        assert "halal" in registered
        assert "halal_new" in registered
        assert "halal_filtered" in registered
        assert "nifty_shariah_500" in registered
        assert "sp500" in registered

    def test_returns_immutable_frozenset(self) -> None:
        """Caller cannot mutate the registry through the returned set."""
        result = list_universes()
        assert isinstance(result, frozenset)


class TestGetETLSymbolsHappyPath:
    """Tests that the registry dispatches to the correct resolver."""

    def test_halal_dispatches_to_registered_resolver(self) -> None:
        """Asking for halal calls the resolver registered for halal."""
        mock_resolver = MagicMock(return_value=["AAPL", "MSFT"])
        with _patched_resolver("halal", mock_resolver):
            result = get_etl_symbols("halal")
        mock_resolver.assert_called_once_with()
        assert result == ["AAPL", "MSFT"]

    def test_halal_new_dispatches_to_registered_resolver(self) -> None:
        """halal_new -> registered resolver, no shutdown_event."""
        mock_resolver = MagicMock(return_value=["NVDA"])
        with _patched_resolver("halal_new", mock_resolver):
            result = get_etl_symbols("halal_new")
        mock_resolver.assert_called_once_with()
        assert result == ["NVDA"]

    def test_sp500_dispatches_to_registered_resolver(self) -> None:
        """sp500 -> registered resolver, no shutdown_event."""
        mock_resolver = MagicMock(return_value=["GOOG"])
        with _patched_resolver("sp500", mock_resolver):
            result = get_etl_symbols("sp500")
        mock_resolver.assert_called_once_with()
        assert result == ["GOOG"]

    def test_nifty_shariah_500_dispatches_to_india_resolver(self) -> None:
        """nifty_shariah_500 -> registered resolver, no shutdown_event."""
        mock_resolver = MagicMock(return_value=["RELIANCE.NS"])
        with _patched_resolver("nifty_shariah_500", mock_resolver):
            result = get_etl_symbols("nifty_shariah_500")
        mock_resolver.assert_called_once_with()
        assert result == ["RELIANCE.NS"]

    def test_halal_filtered_forwards_shutdown_event(self) -> None:
        """halal_filtered resolver receives shutdown_event kwarg.

        This is the ONLY universe that opts in to shutdown_event so the
        sticky-selection rebuild can be cancelled mid-PatchTST scoring.
        """
        event = threading.Event()
        mock_resolver = MagicMock(return_value=["AAPL"])
        with _patched_resolver("halal_filtered", mock_resolver):
            result = get_etl_symbols("halal_filtered", shutdown_event=event)
        mock_resolver.assert_called_once_with(shutdown_event=event)
        assert result == ["AAPL"]

    def test_halal_filtered_default_shutdown_event_is_none(self) -> None:
        """halal_filtered without explicit shutdown_event passes None."""
        mock_resolver = MagicMock(return_value=[])
        with _patched_resolver("halal_filtered", mock_resolver):
            get_etl_symbols("halal_filtered")
        mock_resolver.assert_called_once_with(shutdown_event=None)


class TestGetETLSymbolsUnknownUniverse:
    """Tests for the unknown-universe error path (maps to HTTP 422)."""

    def test_unknown_universe_raises_unknown_etl_universe_error(self) -> None:
        """Unregistered universe raises the dedicated subclass."""
        with pytest.raises(UnknownETLUniverseError) as exc:
            get_etl_symbols("does_not_exist")
        message = str(exc.value)
        assert "does_not_exist" in message
        # Sorted list of valid universes is included so callers can
        # self-diagnose typos.
        assert "halal_filtered" in message

    def test_unknown_universe_error_subclasses_value_error(self) -> None:
        """``UnknownETLUniverseError`` IS a ``ValueError``.

        Routes catch ``UnknownETLUniverseError`` specifically, but the
        ValueError parent keeps backwards-compat for any caller that
        uses the broader exception type.
        """
        assert issubclass(UnknownETLUniverseError, ValueError)
