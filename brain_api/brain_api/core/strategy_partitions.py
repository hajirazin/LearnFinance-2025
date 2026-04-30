"""Canonical sticky-history partition keys used by multi-strategy pipelines.

A "partition key" scopes sticky-history rows for one strategy so it cannot
collide with another strategy's prior-period state. It is passed as the
``partition`` (or legacy ``universe``) column in the appropriate
sticky-history table:

- Two-stage strategies (HRP-backed, weekly cadence) -> ``stage1_weight_history``
  via ``StickyHistoryRepository``.
- Single-stage screening strategies (no Stage 2, monthly cadence) ->
  ``screening_history`` via ``ScreeningHistoryRepository``.

Partition strings MUST be unique across the union of both tables so reads
from either repository cannot accidentally overlap a different strategy's
state.

Definitions:

- ``HALAL_NEW_ALPHA_PARTITION = "halal_new_alpha"`` -- US Alpha-HRP
  (PatchTST predicted weekly return -> rank-band sticky -> HRP). The
  tradable universe is still ``halal_new``; the suffix ``_alpha``
  indicates "screened by alpha signal". Two-stage; lives in
  ``stage1_weight_history``.

- ``HALAL_NEW_PARTITION = "halal_new"`` -- US Double HRP (weight-band
  sticky on Stage 1 HRP weights). Reuses the universe label as the
  partition key because this strategy was the first user of the
  sticky_history table. Two-stage; lives in ``stage1_weight_history``.

- ``HALAL_INDIA_ALPHA_PARTITION = "halal_india_alpha"`` -- India
  Alpha-HRP (PatchTST predicted weekly return -> rank-band sticky ->
  HRP) on the Nifty Shariah 500 universe. Mirrors the US Alpha-HRP
  policy on Indian equities. Two-stage; lives in
  ``stage1_weight_history``.

- ``HALAL_FILTERED_ALPHA_PARTITION = "halal_filtered_alpha"`` -- the
  ``halal_filtered`` universe builder (PatchTST predicted weekly return
  -> rank-band sticky -> top 15). Same selector core as US Alpha-HRP but
  **monthly cadence** (driven by the monthly universe cache), so its
  period-key uses the YYYYWW of the month's first Monday (see
  ``iso_year_week_of_month_anchor`` in
  ``brain_api.core.sticky_selection``). Single-stage; lives in the
  sibling ``screening_history`` table -- it has no Stage 2 HRP, so the
  Stage 1 / Stage 2 weight columns of ``stage1_weight_history`` would
  always be NULL and ``selected_in_final`` would carry different
  semantics. Physically separating the table prevents cross-cadence
  collision and removes ambiguity at read time.

- ``HALAL_INDIA_FILTERED_ALPHA_PARTITION = "halal_india_filtered_alpha"``
  -- the India counterpart of ``HALAL_FILTERED_ALPHA_PARTITION``. The
  ``halal_india`` universe builder (India PatchTST predicted weekly
  return on the Nifty Shariah 500 base -> rank-band sticky -> top 15)
  uses this partition. Single-stage, **monthly cadence**, lives in the
  ``screening_history`` sibling table -- isolated from the weekly
  ``halal_india_alpha`` partition (which lives in
  ``stage1_weight_history`` and is driven by the weekly India Alpha-HRP
  workflow). Period-key follows the same first-Monday-of-month YYYYWW
  convention. Stocks in this partition retain the ``.NS`` yfinance
  suffix end-to-end (no append/strip transformations).

DO NOT use the same partition string for two strategies; doing so would
corrupt sticky carry-sets across strategies even when they live in
different tables. New strategies should reserve a fresh partition key
here before being deployed.
"""

from __future__ import annotations

# US strategies on the halal_new universe (two-stage; stage1_weight_history).
HALAL_NEW_PARTITION = "halal_new"
HALAL_NEW_ALPHA_PARTITION = "halal_new_alpha"

# India strategies on the nifty_shariah_500 universe (two-stage; stage1_weight_history).
HALAL_INDIA_ALPHA_PARTITION = "halal_india_alpha"

# Single-stage screening strategies (monthly cadence; screening_history).
HALAL_FILTERED_ALPHA_PARTITION = "halal_filtered_alpha"
HALAL_INDIA_FILTERED_ALPHA_PARTITION = "halal_india_filtered_alpha"


ALL_PARTITIONS: tuple[str, ...] = (
    HALAL_NEW_PARTITION,
    HALAL_NEW_ALPHA_PARTITION,
    HALAL_INDIA_ALPHA_PARTITION,
    HALAL_FILTERED_ALPHA_PARTITION,
    HALAL_INDIA_FILTERED_ALPHA_PARTITION,
)


def _assert_partitions_unique() -> None:
    """Module-import-time guard that all partition strings are distinct.

    Sharing a partition string between two strategies corrupts carry-sets
    even across separate tables (because partition strings are how callers
    look up "previous round"). Catching duplicates at import time gives a
    loud failure instead of silent data drift.
    """
    if len(set(ALL_PARTITIONS)) != len(ALL_PARTITIONS):
        duplicates = [p for p in ALL_PARTITIONS if ALL_PARTITIONS.count(p) > 1]
        raise AssertionError(
            f"Duplicate partition keys in strategy_partitions: {sorted(set(duplicates))}"
        )


_assert_partitions_unique()
