"""Canonical sticky-history partition keys used by multi-strategy pipelines.

A "partition key" is the value passed as ``universe`` into
``StickyHistoryRepository`` / ``/allocation/sticky-top-n`` /
``/allocation/rank-band-top-n``. It scopes sticky-history rows so that
strategies which screen the same tradable universe (e.g. halal_new) do
not stomp on each other's prior-week selection state.

Definitions:

- ``HALAL_NEW_ALPHA_PARTITION = "halal_new_alpha"`` -- US Alpha-HRP
  (PatchTST predicted weekly return -> rank-band sticky -> HRP). The
  tradable universe is still ``halal_new``; the suffix ``_alpha``
  indicates "screened by alpha signal".

- ``HALAL_NEW_PARTITION = "halal_new"`` -- US Double HRP (weight-band
  sticky on Stage 1 HRP weights). Reuses the universe label as the
  partition key because this strategy was the first user of the
  sticky_history table.

- ``HALAL_INDIA_ALPHA_PARTITION = "halal_india_alpha"`` -- India
  Alpha-HRP (PatchTST predicted weekly return -> rank-band sticky ->
  HRP) on the Nifty Shariah 500 universe. Mirrors the US Alpha-HRP
  policy on Indian equities. The ``_alpha`` suffix indicates
  "screened by alpha signal" and keeps the rank-band sticky rows
  isolated from any future India weight-band variant.

DO NOT use the same partition string for two strategies; doing so
would corrupt sticky carry-sets across strategies. New strategies
should reserve a fresh partition key here before being deployed.
"""

from __future__ import annotations

# US strategies on the halal_new universe.
HALAL_NEW_PARTITION = "halal_new"
HALAL_NEW_ALPHA_PARTITION = "halal_new_alpha"

# India strategies on the nifty_shariah_500 universe.
HALAL_INDIA_ALPHA_PARTITION = "halal_india_alpha"
