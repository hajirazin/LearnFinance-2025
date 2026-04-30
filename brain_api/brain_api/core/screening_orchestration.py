"""Single-stage rank-band screening orchestration (read-previous + select + persist).

This module is the I/O glue for the **single-stage** rank-band path used
by the ``halal_filtered`` universe builder. It sits on top of:

- :func:`brain_api.core.sticky_selection.select_with_rank_band` -- math
  (the SAME selector the two-stage path uses; math reuse is by sharing
  this single function, not by sharing persistence code or schemas).
- :class:`brain_api.storage.screening_history.ScreeningHistoryRepository`
  -- persistence in the sibling ``screening_history`` table.

It contains NO selection math. It only:

1. Reads ``previous_selected_set`` from the screening repo using the
   **single-stage** carry-set rule (``selected = 1`` in last round).
2. Runs the rank-band selector on this round's scores.
3. Persists screening rows for the full universe -- exactly one row per
   candidate symbol -- with the audit-grade ``rank`` from
   :func:`brain_api.core.sticky_selection.rank_by_score`.

Math / DDD invariants
---------------------
- This module never touches selection math. Math lives in
  ``select_with_rank_band``. The two-stage path
  (:mod:`brain_api.core.rank_band_orchestration`) and this module both
  call that function; persistence stays separate per bounded context.
- Persisted ``rank`` is computed by ``rank_by_score(scores)`` -- the
  same convention the selector saw. The two layers MUST use the same
  ranking call so the audit rank cannot drift from the selector's rank.
- Non-finite scores MUST raise (no silent fallback to top-N). Both
  ``rank_by_score`` and ``select_with_rank_band`` already raise; the
  orchestrator propagates.
- This module MUST NOT touch ``stage1_weight_history`` or
  :class:`StickyHistoryRepository`. Two-stage strategies use the
  rank-band orchestration module instead.
"""

from __future__ import annotations

from brain_api.core.sticky_selection import (
    SelectionResult,
    rank_by_score,
    select_with_rank_band,
)
from brain_api.storage.screening_history import (
    ScreeningHistoryRepository,
    ScreeningRow,
)


def persist_screening_rows(
    *,
    repo: ScreeningHistoryRepository,
    partition: str,
    period_key: str,
    as_of_date: str,
    run_id: str,
    scores: dict[str, float],
    selected_set: set[str],
    selection_reasons: dict[str, str],
) -> None:
    """Persist one round of screening rows: full candidate universe.

    Builds one row per scored symbol with ``rank`` matching the rank the
    selector saw (single source of truth: ``rank_by_score``). The
    selected set drives the ``selected`` flag; the reasons map drives
    ``selection_reason``. Symbols not in ``selected_set`` are still
    persisted with ``selected = 0`` so the full audit trail of the
    candidate population is preserved (matches the two-stage table's
    convention of storing the whole universe per round).

    ``signal_score`` is REQUIRED (non-NULL) on every row -- by repository
    schema and by single-stage math: rank-band cannot run without a score
    per candidate, so no row should claim a missing score.
    """
    rows = [
        ScreeningRow(
            partition=partition,
            period_key=period_key,
            as_of_date=as_of_date,
            stock=symbol,
            rank=rank,
            signal_score=value,
            selected=(symbol in selected_set),
            selection_reason=selection_reasons.get(symbol),
            run_id=run_id,
        )
        for symbol, rank, value in rank_by_score(scores)
    ]
    repo.persist_screening_round(rows)


def select_rank_band_for_screening(
    *,
    repo: ScreeningHistoryRepository,
    partition: str,
    period_key: str,
    as_of_date: str,
    run_id: str,
    current_scores: dict[str, float],
    top_n: int,
    hold_threshold: int,
) -> tuple[SelectionResult, str | None]:
    """Read previous selected set, run rank-band, persist screening rows.

    Used by the ``halal_filtered`` universe builder. Reads the carry-set
    with the **single-stage** rule: symbols flagged ``selected = 1`` in
    the most recent prior round for this partition. For two-stage
    strategies use
    :func:`brain_api.core.rank_band_orchestration.select_rank_band_with_persistence`
    against the ``stage1_weight_history`` table.

    Args:
        repo: Screening repository pointed at ``screening_history``.
        partition: Partition key (e.g. ``halal_filtered_alpha``).
        period_key: Period key for THIS round. For monthly-cadence
            strategies use
            :func:`brain_api.core.sticky_selection.iso_year_week_of_month_anchor`.
        as_of_date: ISO YYYY-MM-DD; persisted on every row for audit.
        run_id: Build identifier; persisted on every row.
        current_scores: Symbol -> numeric signal for THIS round.
        top_n: K_in (entry threshold).
        hold_threshold: K_hold (retention threshold). Must be >= top_n.

    Returns:
        ``(selection_result, previous_period_key_used)`` where the second
        element is the prior period_key the carry-set was read from, or
        ``None`` on cold start.

    Raises:
        ValueError: If ``current_scores`` is empty, ``hold_threshold``
            < ``top_n``, ``top_n`` < 1, or any score is non-finite. No
            silent fallback to top-N -- a single-stage screen with a
            broken score map is a hard error, not a degraded mode.
    """
    previous = repo.read_previous_selected_set(
        partition=partition,
        current_period_key=period_key,
    )
    previous_selected_set = previous.selected_set if previous is not None else None
    previous_period_key_used = previous.period_key if previous is not None else None

    result = select_with_rank_band(
        current_scores=current_scores,
        previous_selected_set=previous_selected_set,
        top_n=top_n,
        hold_threshold=hold_threshold,
    )

    persist_screening_rows(
        repo=repo,
        partition=partition,
        period_key=period_key,
        as_of_date=as_of_date,
        run_id=run_id,
        scores=current_scores,
        selected_set=set(result.selected),
        selection_reasons=result.reasons,
    )

    return result, previous_period_key_used
