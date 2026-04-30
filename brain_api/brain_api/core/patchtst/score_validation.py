"""Score validation policy for PatchTST batch inference -> rank-band selection.

The rank-band selector (``brain_api.core.sticky_selection.select_with_rank_band``)
requires strict-weak ordering on scores so that ranks are deterministic across
runs. Two failure modes break that contract and must be raised loudly rather
than silently degraded:

1. **Non-finite scores** (NaN, +inf, -inf): break strict-weak ordering and
   produce nondeterministic ranks. They typically indicate model corruption
   (e.g. exploded gradients leaking into inference) and should never be fed
   into selection. Per the AGENTS.md "no silent fallbacks" rule we surface
   them rather than dropping them as if they were missing data.

2. **Below-floor count**: rank-band semantics assume the universe is large
   enough that ``top_n`` selection is meaningful. Returning fewer than
   ``min_predictions`` valid scores means the batch is effectively broken
   (most symbols failed inference) and the resulting basket would not
   represent the strategy's intent.

This module is the **single canonical home** of these invariants. Both US
(halal_new) and India (nifty_shariah_500) Alpha-HRP score paths go through
it via the ``/inference/patchtst/score-batch`` endpoint, so the contract
cannot drift between markets. Per AGENTS.md "math/policy in core, thin
endpoints", this is not infrastructure plumbing -- it is a math invariant
of the rank-band selector and belongs next to the model code.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from brain_api.core.patchtst.inference import SymbolPrediction


def validate_and_collect_finite_scores(
    predictions: Iterable[SymbolPrediction],
    requested_count: int,
    min_predictions: int,
) -> tuple[dict[str, float], list[str]]:
    """Filter PatchTST batch predictions to finite scores; enforce math invariants.

    Splits predictions into three buckets:

    * **finite scores** -> returned as ``{symbol: predicted_weekly_return_pct}``
    * **None predictions** (insufficient history / missing data) -> returned in
      the ``excluded`` list, NOT raised. These are an expected and benign
      outcome of running batch inference over a noisy universe.
    * **non-finite predictions** (NaN, +inf, -inf) -> raise immediately.

    After bucketing, also enforce the ``min_predictions`` floor.

    Args:
        predictions: Per-symbol PatchTST results from
            :func:`brain_api.core.patchtst.inference.run_batch_inference`.
        requested_count: Total number of symbols originally requested
            (used only for the error message; lets the caller surface
            "got X / Y").
        min_predictions: Minimum count of finite scores required for
            rank-band selection to be meaningful. Below this floor the
            batch is rejected.

    Returns:
        A pair ``(scores, excluded)``:

        * ``scores`` -- dict of symbol -> finite ``predicted_weekly_return_pct``.
        * ``excluded`` -- list of symbols whose prediction was ``None``.
          (Non-finite symbols never reach this list because they raise.)

    Raises:
        RuntimeError: If any prediction is non-finite, OR if the count of
            finite scores is below ``min_predictions``. The two messages
            match the original wording so existing log scrapes / runbooks
            keep working.
    """
    scores: dict[str, float] = {}
    excluded: list[str] = []
    non_finite: list[str] = []

    for prediction in predictions:
        score = prediction.predicted_weekly_return_pct
        if score is None:
            excluded.append(prediction.symbol)
        elif not math.isfinite(score):
            # NaN / +inf / -inf break the rank-band selector's strict-weak
            # ordering and would produce nondeterministic ranks. Surface
            # the corruption loudly rather than silently degrading.
            non_finite.append(prediction.symbol)
            excluded.append(prediction.symbol)
        else:
            scores[prediction.symbol] = score

    if non_finite:
        raise RuntimeError(
            f"PatchTST batch produced non-finite scores for symbols: "
            f"{non_finite}. Refusing to feed NaN/inf into rank-band "
            f"selection -- investigate the model output before rerunning."
        )

    if len(scores) < min_predictions:
        raise RuntimeError(
            f"PatchTST batch returned only {len(scores)} valid predictions "
            f"({requested_count} requested), below "
            f"min_predictions={min_predictions}. Excluded={len(excluded)}."
        )

    return scores, excluded
