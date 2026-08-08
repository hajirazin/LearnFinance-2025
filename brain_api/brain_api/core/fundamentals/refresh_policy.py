"""Refresh decision policy for historical fundamentals (no fetch-today)."""

from __future__ import annotations

from enum import Enum


class RefreshAction(str, Enum):
    """Action selected for one symbol during a refresh pass."""

    PULL = "pull"
    ENRICH_ONLY = "enrich_only"
    SKIP = "skip"
    PENDING_NEW_FILING = "pending_new_filing"


class SymbolCacheState(str, Enum):
    """Queue classification for AV backlog ordering."""

    MISSING = "missing"
    FILING_STALE = "filing_stale"
    UNPROVENANCED = "unprovenanced"
    COMPLETE_FRESH = "complete_fresh"
    PENDING = "pending"


def decide_refresh_action(
    *,
    force_refresh: bool,
    has_usable_quarters: bool,
    has_cik: bool,
    behind_head: bool,
    unprovenanced: bool,
) -> RefreshAction:
    """Return refresh action using locked priority order."""
    if force_refresh:
        return RefreshAction.PULL
    if not has_usable_quarters:
        return RefreshAction.PULL
    if has_cik and behind_head:
        return RefreshAction.PULL
    if unprovenanced:
        return RefreshAction.ENRICH_ONLY
    return RefreshAction.SKIP


def order_av_pull_queue(
    items: list[tuple[str, SymbolCacheState]],
    *,
    forced: set[str] | None = None,
) -> list[str]:
    """Order AV pulls: missing first, then forced, then filing-stale."""
    forced = forced or set()
    missing = [sym for sym, state in items if state == SymbolCacheState.MISSING]
    forced_syms = [
        sym
        for sym, state in items
        if sym in forced and state != SymbolCacheState.MISSING
    ]
    stale = [
        sym
        for sym, state in items
        if state == SymbolCacheState.FILING_STALE and sym not in forced
    ]
    # Deduplicate preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for sym in missing + forced_syms + stale:
        if sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    return ordered
