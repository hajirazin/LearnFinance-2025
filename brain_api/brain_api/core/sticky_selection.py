"""Sticky-selection (rebalance-band) primitive for top-N stock selection.

Given this week's stage 1 HRP weights for a full universe and last week's
stage 1 weights + final selected set, decide which N stocks to allocate
this week. Stocks that were held last week and whose stage 1 weight has
moved by less than ``threshold_pp`` absolute percentage points are kept
("sticky"). Remaining slots are filled by top current-week weight.

Math note
---------
The threshold is in **absolute percentage points** on the stage 1 HRP
weight. For a ~410-stock halal_new universe the average per-stock weight
is ~0.24%, so a 1.0pp threshold is a generous no-trade band: only stocks
whose weight changed by ~4x the average get evicted on the weight signal.
This is intentional — sticky selection is a turnover damper, not a signal
filter.

When the universe shrinks/grows materially (ETF rebalance), every weight
mechanically renormalises and stable risk contributors can trip the
threshold. Acceptable for top-15 portfolios with default 1.0pp threshold;
revisit with a relative or universe-size-aware band if anomalous evictions
are observed.

This module is pure: no I/O, no DB, no globals. Persistence is the
responsibility of ``brain_api.storage.sticky_history``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

# Selection reasons (audit values, mirrored in SQLite ``selection_reason``)
REASON_STICKY = "sticky"
REASON_TOP_RANK = "top_rank"

# Eviction reasons (why a previously-held stock was dropped this week)
EVICTION_WEIGHT_DIFF = "weight_diff"
EVICTION_DROPPED_FROM_UNIVERSE = "dropped_from_universe"


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of one week's sticky selection.

    Attributes:
        selected: Symbols chosen for this week's final portfolio (length
            ``top_n``), ordered with sticky-kept stocks first (preserving
            their previous-week ranking by current weight) followed by
            fillers in descending current-weight order.
        reasons: Map from selected symbol to ``"sticky"`` or ``"top_rank"``.
        kept_count: How many of last week's final stocks were retained.
        fillers_count: How many slots were filled with new top-rank stocks.
        evicted_from_previous: Map from previously-held symbols that did
            NOT carry over to ``"weight_diff"`` (weight moved by >=
            threshold) or ``"dropped_from_universe"`` (no longer in the
            current universe).
    """

    selected: list[str]
    reasons: dict[str, str]
    kept_count: int
    fillers_count: int
    evicted_from_previous: dict[str, str] = field(default_factory=dict)


def iso_year_week(d: date | datetime) -> str:
    """Return ISO year+week as a 6-character ``YYYYWW`` string.

    Uses ``%G%V`` so the year prefix is the ISO week-numbering year (which
    can differ from the calendar year on the boundaries of week 1 / week
    52-53). Lexicographic comparison gives correct chronological ordering
    across year boundaries.

    Examples:
        >>> from datetime import date
        >>> iso_year_week(date(2026, 2, 23))
        '202609'
        >>> iso_year_week(date(2025, 12, 29))   # ISO week 1 of 2026
        '202601'
    """
    return d.strftime("%G%V")


def select_with_stickiness(
    current_stage1: dict[str, float],
    previous_stage1: dict[str, float] | None,
    previous_final_set: set[str] | None,
    top_n: int,
    threshold_pp: float = 1.0,
) -> SelectionResult:
    """Pick ``top_n`` symbols using a no-trade band against last week.

    Algorithm (deterministic, no I/O):

    1. **Cold start**: if ``previous_stage1`` or ``previous_final_set`` is
       ``None``, return the top ``top_n`` of ``current_stage1`` by weight
       descending; every reason is ``"top_rank"``.
    2. **Sticky candidates**: for each stock in ``previous_final_set``:

       - Not in this week's universe -> evict with reason
         ``"dropped_from_universe"``.
       - Otherwise compute
         ``diff_pp = abs(current[s] - previous[s, default=0])``. If
         ``diff_pp < threshold_pp``, keep as sticky candidate; else
         evict with reason ``"weight_diff"``.
    3. **Trim sticky to top_n** if more than ``top_n`` stocks qualify
       (improbable for top_n=15 since at most 15 candidates exist, but
       general for larger ``top_n``). Sort by current weight descending
       and keep the top.
    4. **Fill remaining slots** with the highest current-week weights
       among non-kept stocks.
    5. Return ``SelectionResult`` with sticky stocks first (ordered by
       current weight desc) followed by fillers (ordered by current
       weight desc).

    Args:
        current_stage1: Symbol -> stage 1 HRP weight (in %) for this week.
        previous_stage1: Same shape for last week, or ``None`` for cold
            start.
        previous_final_set: Symbols that were in last week's final
            ``top_n``, or ``None`` for cold start.
        top_n: Target count of selected symbols.
        threshold_pp: Maximum absolute pp move allowed to remain sticky.
            Comparison uses strict ``<`` against the threshold, so a
            stock at exactly the threshold IS evicted.

    Raises:
        ValueError: If ``top_n`` < 1, ``threshold_pp`` < 0, or
            ``current_stage1`` is empty.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if threshold_pp < 0:
        raise ValueError(f"threshold_pp must be >= 0, got {threshold_pp}")
    if not current_stage1:
        raise ValueError("current_stage1 must not be empty")

    # Stable sort key: by weight desc, then by symbol asc for determinism.
    def by_weight_desc(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))

    if previous_stage1 is None or previous_final_set is None:
        ranked = by_weight_desc(list(current_stage1.items()))
        selected = [s for s, _ in ranked[:top_n]]
        reasons = dict.fromkeys(selected, REASON_TOP_RANK)
        return SelectionResult(
            selected=selected,
            reasons=reasons,
            kept_count=0,
            fillers_count=len(selected),
            evicted_from_previous={},
        )

    # 1. Classify previous-week holdings as sticky-eligible vs evicted.
    sticky_pairs: list[tuple[str, float]] = []
    evicted: dict[str, str] = {}
    for stock in previous_final_set:
        cur = current_stage1.get(stock)
        if cur is None:
            evicted[stock] = EVICTION_DROPPED_FROM_UNIVERSE
            continue
        prev = previous_stage1.get(stock, 0.0)
        diff_pp = abs(cur - prev)
        if diff_pp < threshold_pp:
            sticky_pairs.append((stock, cur))
        else:
            evicted[stock] = EVICTION_WEIGHT_DIFF

    # 2. Trim sticky if it ever exceeds top_n (defensive for top_n != 15).
    sticky_pairs = by_weight_desc(sticky_pairs)[:top_n]
    kept = [s for s, _ in sticky_pairs]
    kept_set = set(kept)

    # 3. Fill remaining slots from the current universe, excluding kept.
    remaining = top_n - len(kept)
    if remaining > 0:
        candidates = by_weight_desc(
            [(s, w) for s, w in current_stage1.items() if s not in kept_set]
        )
        fillers = [s for s, _ in candidates[:remaining]]
    else:
        fillers = []

    selected = kept + fillers
    reasons: dict[str, str] = dict.fromkeys(kept, REASON_STICKY)
    reasons.update(dict.fromkeys(fillers, REASON_TOP_RANK))

    return SelectionResult(
        selected=selected,
        reasons=reasons,
        kept_count=len(kept),
        fillers_count=len(fillers),
        evicted_from_previous=evicted,
    )
