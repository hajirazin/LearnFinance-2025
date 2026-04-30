"""Two-stage rank-band selection orchestration (read-previous + select + persist).

This module is the I/O glue for the **two-stage** rank-band path used by
``/allocation/rank-band-top-n`` (US Alpha-HRP, India Alpha-HRP). It sits
on top of:

- :func:`brain_api.core.sticky_selection.select_with_rank_band` -- math.
- :class:`brain_api.storage.sticky_history.StickyHistoryRepository` --
  persistence in ``stage1_weight_history`` (the two-stage table).

It contains NO selection math. It only:

1. Reads ``previous_final_set`` from the repo using the **two-stage**
   carry-set rule (``final_allocation_pct IS NOT NULL``).
2. Runs the rank-band selector on this round's scores.
3. Persists Stage 1 rows for the full universe -- exactly one row per
   candidate symbol -- with the audit-grade ``stage1_rank`` from
   :func:`brain_api.core.sticky_selection.rank_by_score`.

Math / DDD invariants
---------------------
- This module never touches selection math. Math lives in
  ``select_with_rank_band``. Two-stage and single-stage paths share that
  selector via two thin orchestrators (this one + the screening
  orchestrator); they do NOT share persistence.
- Persisted ``stage1_rank`` is computed by ``rank_by_score(scores)`` --
  the same convention the selector saw. Splitting that ranking call
  across two layers is forbidden because it could let the persisted
  rank drift from the selector's rank.
- Non-finite scores (NaN, +/-inf) MUST raise. Both ``rank_by_score`` and
  ``select_with_rank_band`` already do; the orchestrator propagates.
- Single-stage strategies MUST NOT use this module. They use
  :mod:`brain_api.core.screening_orchestration` against the sibling
  ``screening_history`` table.
"""

from __future__ import annotations

from typing import Literal

from brain_api.core.sticky_selection import (
    SelectionResult,
    rank_by_score,
    select_with_rank_band,
)
from brain_api.storage.sticky_history import StickyHistoryRepository, WeightRow

StageOneColumn = Literal["initial_allocation_pct", "signal_score"]


def persist_stage1_rows(
    *,
    repo: StickyHistoryRepository,
    universe: str,
    year_week: str,
    as_of_date: str,
    run_id: str,
    signals: dict[str, float],
    selected_set: set[str],
    selection_reasons: dict[str, str],
    column: StageOneColumn,
) -> None:
    """Persist Stage 1 rows for either two-stage selection policy.

    Used by both ``select_sticky_top_n_endpoint`` (weight-band) and
    ``select_rank_band_top_n_endpoint`` (rank-band). The two endpoints
    differ only in *which* Stage 1 column the signal lands in
    (``initial_allocation_pct`` for HRP weights vs ``signal_score`` for
    raw forecast scores). Both are exclusive -- rows persist with exactly
    one of the two columns populated, never both, never neither.

    Math/DDD note: ``rank_by_score`` is used here even though the
    weight-band primitive does NOT use rank-based math internally. For
    *persistence* the rank serves as an audit number; the math inside
    the two ``select_with_*`` functions remains independent.
    """
    rows = [
        WeightRow(
            universe=universe,
            year_week=year_week,
            as_of_date=as_of_date,
            stock=symbol,
            stage1_rank=rank,
            initial_allocation_pct=value
            if column == "initial_allocation_pct"
            else None,
            signal_score=value if column == "signal_score" else None,
            final_allocation_pct=None,
            selected_in_final=(symbol in selected_set),
            selection_reason=selection_reasons.get(symbol),
            run_id=run_id,
        )
        for symbol, rank, value in rank_by_score(signals)
    ]
    repo.persist_stage1(rows)


def select_rank_band_with_persistence(
    *,
    repo: StickyHistoryRepository,
    universe: str,
    year_week: str,
    as_of_date: str,
    run_id: str,
    current_scores: dict[str, float],
    top_n: int,
    hold_threshold: int,
) -> tuple[SelectionResult, str | None]:
    """Read previous final set, run rank-band, persist Stage 1 rows.

    Used by ``/allocation/rank-band-top-n`` (US/India Alpha-HRP). Reads
    the carry-set with the **two-stage** rule (post-HRP-shrinkage:
    symbols whose ``final_allocation_pct`` is non-null in the most recent
    prior week). For single-stage screening strategies use
    :func:`brain_api.core.screening_orchestration.select_rank_band_for_screening`
    against the sibling ``screening_history`` table.

    Args:
        repo: Two-stage repository pointed at ``stage1_weight_history``.
        universe: Partition key (e.g. ``halal_new_alpha``).
        year_week: Period key for THIS round (ISO YYYYWW).
        as_of_date: ISO YYYY-MM-DD; persisted on every row for audit.
        run_id: Build identifier; persisted on every row.
        current_scores: Symbol -> numeric signal for THIS round.
        top_n: K_in (entry threshold).
        hold_threshold: K_hold (retention threshold). Must be >= top_n.

    Returns:
        ``(selection_result, previous_year_week_used)`` where the second
        element is the prior week the carry-set was read from, or
        ``None`` on cold start.

    Raises:
        ValueError: If ``current_scores`` is empty, ``hold_threshold``
            < ``top_n``, ``top_n`` < 1, or any score is non-finite. No
            silent fallback to top-N -- the rank-band contract is
            strict on inputs.
    """
    previous = repo.read_previous_final_set(
        universe=universe,
        current_year_week=year_week,
    )
    previous_final_set = previous.final_set if previous is not None else None
    previous_year_week_used = previous.year_week if previous is not None else None

    result = select_with_rank_band(
        current_scores=current_scores,
        previous_selected_set=previous_final_set,
        top_n=top_n,
        hold_threshold=hold_threshold,
    )

    persist_stage1_rows(
        repo=repo,
        universe=universe,
        year_week=year_week,
        as_of_date=as_of_date,
        run_id=run_id,
        signals=current_scores,
        selected_set=set(result.selected),
        selection_reasons=result.reasons,
        column="signal_score",
    )

    return result, previous_year_week_used
