"""ETL universe registry: maps a universe string to a symbol resolver.

Mirrors :mod:`brain_api.core.model_buckets` for the ETL surface. The
ETL endpoints (`/etl/news-sentiment`, `/etl/sentiment-gaps`,
`/etl/refresh-training-data`) take ``universe`` in the request body
instead of reading the ``ETL_UNIVERSE`` env var, so two parallel
Temporal workflows can refresh different slates against the same
brain_api deployment without env-var contention.

Why a registry? It collapses three duplicated ``if/elif`` ladders that
previously dispatched ``UniverseType -> get_<universe>_symbols`` in
:mod:`brain_api.etl.gap_fill`, :mod:`brain_api.etl.symbol_filter`, and
``routes.training.dependencies.get_etl_symbols``. Adding a new universe
becomes one ``_register(...)`` call.

Math correctness note (per AGENTS.md rule #2): this layer only routes
``universe -> symbol resolver``. The resolvers themselves
(``get_halal_filtered_symbols`` etc.) are unchanged, so per-universe
selection math (PatchTST scores, sticky bands, ETF holdings) stays in
its own modules.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from brain_api.universe.halal import get_halal_symbols
from brain_api.universe.halal_filtered import get_halal_filtered_symbols
from brain_api.universe.halal_new import get_halal_new_symbols
from brain_api.universe.nifty_shariah_500 import get_nifty_shariah_500_symbols
from brain_api.universe.sp500 import get_sp500_symbols


class UnknownETLUniverseError(ValueError):
    """Raised when ``universe`` is not registered.

    Routes translate this to HTTP 422 so the caller can correct its
    body without retrying blindly. Mirrors
    :class:`brain_api.core.model_buckets.UnknownBucketError`.
    """


# A resolver may optionally accept a ``shutdown_event`` kwarg (used by
# ``halal_filtered`` so cooperative cancellation can stop the cache
# rebuild mid-PatchTST scoring). The registry passes the kwarg only to
# resolvers that accept it.
SymbolsResolver = Callable[..., list[str]]


@dataclass(frozen=True)
class ETLUniverseConfig:
    """Routing entry for one ETL universe.

    Attributes:
        universe: Stable string identifier matching the request body
            field on ``/etl/*`` endpoints.
        symbols_resolver: Callable returning the symbol slate.
        accepts_shutdown_event: True if the resolver accepts a
            ``shutdown_event`` keyword argument (currently only the
            ``halal_filtered`` rebuilder).
    """

    universe: str
    symbols_resolver: SymbolsResolver
    accepts_shutdown_event: bool = False


_REGISTRY: dict[str, ETLUniverseConfig] = {}


def _register(cfg: ETLUniverseConfig) -> None:
    """Add a universe config; collisions raise so duplicates fail fast."""
    if cfg.universe in _REGISTRY:
        raise RuntimeError(f"Duplicate ETL universe registration for {cfg.universe!r}")
    _REGISTRY[cfg.universe] = cfg


def list_universes() -> frozenset[str]:
    """Return the set of registered ETL universe strings."""
    return frozenset(_REGISTRY)


def get_etl_symbols(
    universe: str,
    shutdown_event: threading.Event | None = None,
) -> list[str]:
    """Resolve the symbol slate for ``universe``.

    Args:
        universe: One of the registered universe strings (see
            :func:`list_universes`).
        shutdown_event: Optional cooperative-cancellation event;
            forwarded only to resolvers that accept it.

    Raises:
        UnknownETLUniverseError: if ``universe`` is not registered.
    """
    cfg = _REGISTRY.get(universe)
    if cfg is None:
        valid = sorted(list_universes())
        raise UnknownETLUniverseError(
            f"No ETL universe registered for {universe!r}. Valid universes: {valid}"
        )
    if cfg.accepts_shutdown_event:
        return cfg.symbols_resolver(shutdown_event=shutdown_event)
    return cfg.symbols_resolver()


# ---------------------------------------------------------------------------
# Initial registrations.
#
# Adding a future universe (e.g. a sticky-15 slate for a SAC-on-Halal
# A/B workflow) is one ``_register(...)`` call here -- no other ETL
# code edits required.
# ---------------------------------------------------------------------------

_register(
    ETLUniverseConfig(
        universe="halal",
        symbols_resolver=get_halal_symbols,
    )
)

_register(
    ETLUniverseConfig(
        universe="halal_new",
        symbols_resolver=get_halal_new_symbols,
    )
)

_register(
    ETLUniverseConfig(
        universe="halal_filtered",
        symbols_resolver=get_halal_filtered_symbols,
        accepts_shutdown_event=True,
    )
)

_register(
    ETLUniverseConfig(
        universe="nifty_shariah_500",
        symbols_resolver=get_nifty_shariah_500_symbols,
    )
)

_register(
    ETLUniverseConfig(
        universe="sp500",
        symbols_resolver=get_sp500_symbols,
    )
)


__all__ = [
    "ETLUniverseConfig",
    "UnknownETLUniverseError",
    "get_etl_symbols",
    "list_universes",
]
