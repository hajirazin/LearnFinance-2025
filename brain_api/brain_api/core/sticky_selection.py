"""Selection policies (rebalance-band) for top-N stock selection.

This module hosts two pure, deterministic top-N selection policies.
Both are turnover dampers: stocks held last week stay in the basket
unless they breach a configurable band, and remaining slots are filled
from this week's best candidates. The two policies differ in *what*
the band is measured on:

- :func:`select_with_stickiness` -- **weight-band** policy. Operates
  on Stage 1 HRP percentage weights. A previously-held stock survives
  if ``abs(current_weight - previous_weight) < threshold_pp``. Used by
  US Double HRP and India Double HRP, where Stage 1 already produces
  a weight, not a score.

- :func:`select_with_rank_band` -- **rank-band** policy. Operates on
  rank derived from any numeric signal (e.g. PatchTST predicted
  weekly return). A previously-held stock survives if its current
  rank is ``<= hold_threshold`` (``K_hold``). Entry remains capped
  at the top ``top_n`` (``K_in``). Used by US Alpha-HRP, where the
  Stage 1 input is a forecast, not a portfolio weight.

Math correctness invariants (do NOT merge):

- The two policies must NOT share their eviction or trim math. A
  weight diff and a rank diff are different scales with different
  noise characteristics; the rank-band policy is regime-stable
  (Grinold-Kahn, "Active Portfolio Management") whereas the
  weight-band policy is calibrated against HRP weight magnitudes.
  Code reuse is allowed only for ranking and assembly helpers
  (:func:`rank_by_score`) -- never for the policy itself.

- Both policies preserve a deterministic ranking when scores tie:
  ``(-score, symbol asc)``. Persistence layers must use the same
  key for ``stage1_rank`` so that the rank stored in
  ``sticky_history.db`` matches the rank the selection function
  actually saw. :func:`rank_by_score` is the single source of truth
  for this convention.

This module is pure: no I/O, no DB, no globals. Persistence is the
responsibility of ``brain_api.storage.sticky_history``.

Math note -- weight-band threshold
----------------------------------
The threshold is in **absolute percentage points** on the stage 1 HRP
weight. For a ~410-stock halal_new universe the average per-stock
weight is ~0.24%, so a 1.0pp threshold is a generous no-trade band:
only stocks whose weight changed by ~4x the average get evicted on
the weight signal. This is intentional -- sticky selection is a
turnover damper, not a signal filter.

When the universe shrinks/grows materially (ETF rebalance), every
weight mechanically renormalises and stable risk contributors can
trip the threshold. Acceptable for top-15 portfolios with default
1.0pp threshold; revisit with a relative or universe-size-aware band
if anomalous evictions are observed.

Math note -- rank-band asymmetry (K_in vs K_hold)
-------------------------------------------------
The asymmetric band ``K_in <= K_hold`` is a textbook turnover damper
for ranked-alpha portfolios. Names enter only when their rank is in
the top ``K_in`` (strong signal); they are retained while rank stays
``<= K_hold`` (moderate signal). Using rank rather than score makes
the band scale-invariant and regime-stable: a 25 bps shift in
predicted return is meaningless in a high-vol week and large in a
calm week, but a rank shift from 14 -> 28 means the same thing in
both regimes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

# ----------------------------------------------------------------------------
# Selection reasons (audit values, mirrored in SQLite ``selection_reason``)
# ----------------------------------------------------------------------------
# Both policies emit the same reason codes for *selected* symbols.
REASON_STICKY = "sticky"
REASON_TOP_RANK = "top_rank"

# ----------------------------------------------------------------------------
# Eviction reasons (why a previously-held stock was dropped this week)
# ----------------------------------------------------------------------------
# Shared across policies:
EVICTION_DROPPED_FROM_UNIVERSE = "dropped_from_universe"
# Weight-band policy only (select_with_stickiness):
EVICTION_WEIGHT_DIFF = "weight_diff"
# Rank-band policy only (select_with_rank_band):
EVICTION_RANK_OUT_OF_HOLD = "rank_out_of_hold"


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of one week's sticky selection (either policy).

    Attributes:
        selected: Symbols chosen for this week's final portfolio (length
            ``top_n``), ordered with sticky-kept stocks first (preserving
            their previous-week ranking by current weight/rank) followed
            by fillers in descending current-week order.
        reasons: Map from selected symbol to ``REASON_STICKY`` or
            ``REASON_TOP_RANK``.
        kept_count: How many of last week's final stocks were retained.
        fillers_count: How many slots were filled with new top-rank stocks.
        evicted_from_previous: Map from previously-held symbols that did
            NOT carry over to one of the policy-specific eviction
            reasons. Valid values:

            * Both policies: ``EVICTION_DROPPED_FROM_UNIVERSE``
            * Weight-band only (:func:`select_with_stickiness`):
              ``EVICTION_WEIGHT_DIFF``
            * Rank-band only (:func:`select_with_rank_band`):
              ``EVICTION_RANK_OUT_OF_HOLD``

            Trim losses (sticky overflow trimmed to ``top_n`` by best
            current rank/weight) are NOT recorded here; they are
            slot-budget decisions, not policy evictions.
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


def rank_by_score(scores: dict[str, float]) -> list[tuple[str, int, float]]:
    """Return ``[(symbol, rank, score), ...]`` ordered by rank ascending.

    Rank is 1-indexed by ``score`` descending with symbol ascending as
    deterministic tie-break. This is the **single source of truth** for
    the ranking convention used by:

    - :func:`select_with_rank_band` (selection)
    - The persistence adapter in
      ``brain_api/routes/allocation.py`` (writes ``stage1_rank`` to
      ``sticky_history.db``)
    - Reporting code that wants to show the ranks the selector actually
      saw (e.g. the alpha-HRP email's top-25 table).

    Keeping a single helper means the persisted ``stage1_rank`` cannot
    drift from the rank used inside the selector, even if either layer
    is refactored independently.

    Args:
        scores: Symbol -> numeric signal (higher is better). Must be
            non-empty and contain only finite floats.

    Raises:
        ValueError: If ``scores`` is empty or contains a non-finite
            value (NaN, +inf, -inf).
    """
    if not scores:
        raise ValueError("scores must not be empty")
    bad = [s for s, v in scores.items() if not math.isfinite(v)]
    if bad:
        raise ValueError(f"scores must be finite; non-finite values for: {sorted(bad)}")
    sorted_pairs = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        (symbol, rank, score)
        for rank, (symbol, score) in enumerate(sorted_pairs, start=1)
    ]


def _cold_start_top_n_ranked(
    ranked_symbols: list[str],
    top_n: int,
) -> SelectionResult:
    """Build a cold-start :class:`SelectionResult` from a pre-ranked list.

    Shared by both selection policies for the case where the caller
    has no previous-week state (``previous_*`` is ``None``). The math
    each policy uses to *produce* ``ranked_symbols`` is intentionally
    different (weight desc vs rank-band score desc); this helper only
    handles the result-assembly mechanics so that math change in one
    policy cannot accidentally bleed into the other.
    """
    selected = ranked_symbols[:top_n]
    reasons = dict.fromkeys(selected, REASON_TOP_RANK)
    return SelectionResult(
        selected=selected,
        reasons=reasons,
        kept_count=0,
        fillers_count=len(selected),
        evicted_from_previous={},
    )


def _assemble_reasons(kept: list[str], fillers: list[str]) -> dict[str, str]:
    """Combine kept and filler reason dicts into one map.

    Pure assembly; no policy-specific logic.
    """
    reasons: dict[str, str] = dict.fromkeys(kept, REASON_STICKY)
    reasons.update(dict.fromkeys(fillers, REASON_TOP_RANK))
    return reasons


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
    # Kept local to this policy because weight-band ranks weights, not
    # signal scores -- we deliberately do NOT call ``rank_by_score`` so
    # the two policies' math stays independent.
    def by_weight_desc(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))

    if previous_stage1 is None or previous_final_set is None:
        ranked = by_weight_desc(list(current_stage1.items()))
        return _cold_start_top_n_ranked([s for s, _ in ranked], top_n)

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

    return SelectionResult(
        selected=kept + fillers,
        reasons=_assemble_reasons(kept, fillers),
        kept_count=len(kept),
        fillers_count=len(fillers),
        evicted_from_previous=evicted,
    )


def select_with_rank_band(
    current_scores: dict[str, float],
    previous_selected_set: set[str] | None,
    top_n: int,
    hold_threshold: int,
) -> SelectionResult:
    """Pick ``top_n`` symbols using an asymmetric rank-band against last week.

    Algorithm (deterministic, no I/O):

    1. **Rank** every symbol in ``current_scores`` by score descending; ties
       broken by symbol ascending. The 1-indexed rank is used for both
       hold-threshold checks and filler selection.
    2. **Cold start**: if ``previous_selected_set`` is ``None``, return the
       top ``top_n`` by current rank; every reason is ``"top_rank"``.
    3. **Sticky candidates**: for each stock in ``previous_selected_set``:

       - Not in ``current_scores`` -> evict with reason
         ``"dropped_from_universe"``.
       - Rank > ``hold_threshold`` -> evict with reason
         ``"rank_out_of_hold"``.
       - Otherwise keep as sticky candidate.
    4. **Trim sticky to top_n** if more than ``top_n`` qualify (possible
       when ``hold_threshold`` is wider than ``top_n``): keep stocks with
       the best (lowest) current rank.
    5. **Fill remaining slots** with the highest-ranked non-kept stocks.
    6. Return ``SelectionResult`` with sticky stocks first (ordered by
       current rank ascending) followed by fillers (ordered by current
       rank ascending).

    Asymmetry — entry vs hold:

    The "entry" threshold is implicitly ``top_n`` (a stock can only be
    *added* this week if its rank is in the top ``top_n``). The "hold"
    threshold ``hold_threshold`` is wider, so a previously-held stock is
    retained as long as its rank stays inside ``hold_threshold`` even if
    it has slipped out of the top ``top_n``. This is the textbook
    rebalance-band turnover damper for ranked-alpha portfolios
    (Grinold-Kahn): names are added on strong signal but held on
    moderate signal, so we capture alpha without paying friction every
    time the middle of the rank list reshuffles.

    Math note — why rank, not score:

    The signal in scope is a noisy weekly-return forecast. Using rank
    rather than score makes the band scale-invariant and regime-stable;
    a 25 bps shift in predicted return is meaningless in a high-vol
    week and large in a calm week, but a rank shift from 14 -> 28 means
    the same thing in both regimes. Rank-band selection has no impact
    on the Stage 2 allocator's math — it only changes which symbols
    enter Stage 2.

    Args:
        current_scores: Symbol -> numeric signal for this week (e.g.
            PatchTST ``predicted_weekly_return_pct``). Higher is better.
        previous_selected_set: Symbols held last week, or ``None`` for
            cold start.
        top_n: Target count of selected symbols (entry threshold ``K_in``).
        hold_threshold: Maximum rank a previously-held stock may have
            and still be retained (``K_hold``). Must be ``>= top_n`` so
            that any stock currently inside the entry zone is also
            inside the hold zone.

    Raises:
        ValueError: If ``top_n`` < 1, ``hold_threshold`` < ``top_n``,
            ``current_scores`` is empty, or any score is non-finite
            (NaN, +inf, -inf). Non-finite scores break the strict-weak
            ordering rank-band selection requires; ``sorted`` on NaN
            is undefined and ±inf would dominate without economic
            meaning, so we surface the corruption loudly rather than
            silently producing an arbitrary ordering.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if hold_threshold < top_n:
        raise ValueError(
            f"hold_threshold ({hold_threshold}) must be >= top_n ({top_n})"
        )
    if not current_scores:
        raise ValueError("current_scores must not be empty")

    # Single source of truth for ranking. Raises on non-finite scores.
    ranked = rank_by_score(current_scores)
    ranked_symbols = [symbol for symbol, _, _ in ranked]
    rank_of: dict[str, int] = {symbol: rank for symbol, rank, _ in ranked}

    if previous_selected_set is None:
        return _cold_start_top_n_ranked(ranked_symbols, top_n)

    # 1. Classify previous-week holdings as sticky-eligible vs evicted.
    #    Math note: eviction predicate is rank > hold_threshold. Do NOT
    #    reuse the weight-band predicate here -- weight-diff and
    #    rank-out-of-hold are different statistics with different noise
    #    behaviour.
    sticky_ranks: list[tuple[str, int]] = []
    evicted: dict[str, str] = {}
    for stock in previous_selected_set:
        rank = rank_of.get(stock)
        if rank is None:
            evicted[stock] = EVICTION_DROPPED_FROM_UNIVERSE
            continue
        if rank > hold_threshold:
            evicted[stock] = EVICTION_RANK_OUT_OF_HOLD
        else:
            sticky_ranks.append((stock, rank))

    # 2. Trim sticky to top_n by best (lowest) current rank if it ever
    # exceeds top_n. Possible when hold_threshold is meaningfully wider
    # than top_n and last week's basket happens to all stay inside hold.
    sticky_ranks.sort(key=lambda kv: (kv[1], kv[0]))
    sticky_ranks = sticky_ranks[:top_n]
    kept = [s for s, _ in sticky_ranks]
    kept_set = set(kept)

    # 3. Fill remaining slots from current-week ranking, excluding kept.
    #    Asymmetry invariant: fillers are taken from the *prefix* of the
    #    global rank order; combined with ``len(kept) <= top_n`` this
    #    guarantees every filler has rank <= top_n (K_in). New names
    #    cannot enter via the K_hold band.
    remaining = top_n - len(kept)
    if remaining > 0:
        fillers = [s for s in ranked_symbols if s not in kept_set][:remaining]
    else:
        fillers = []

    return SelectionResult(
        selected=kept + fillers,
        reasons=_assemble_reasons(kept, fillers),
        kept_count=len(kept),
        fillers_count=len(fillers),
        evicted_from_previous=evicted,
    )
