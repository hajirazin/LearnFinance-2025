"""Universe-keyed model bucket registry.

A *bucket* is the storage namespace for a single ``(model_type,
universe)`` pair. Each bucket has its own:

* on-disk path (``data/models/{bucket_name}/``) with an independent
  ``current`` pointer and promotion lineage,
* HuggingFace repo (``HF_{MODEL}_{UNIVERSE}_MODEL_REPO``),
* in-process symbol resolver (the universe builder that produces the
  training symbol list),
* optional symbol validator (e.g. enforce ``.NS`` suffix for India).

Why a registry? Two parallel Temporal workflows can hit the same training
endpoint with different ``universe`` values for an A/B comparison without
clobbering each other's ``current`` pointer. Env-var dispatch could not
support this because env vars are process-wide.

Math correctness note (per AGENTS.md): the registry only routes
``(model_type, universe) -> bucket config``. The shared core training
functions (``_train_patchtst_core`` etc.) still receive the bucket's
storage class and HF repo getter as parameters, so per-algorithm math
remains in the algorithm-specific code paths -- this layer adds zero
math coupling between LSTM/PatchTST/SAC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from brain_api.core.config import (
    get_hf_lstm_halal_new_model_repo,
    get_hf_patchtst_halal_new_model_repo,
    get_hf_patchtst_nifty_shariah_500_model_repo,
    get_hf_sac_halal_filtered_model_repo,
    get_hf_sac_halal_model_repo,
)
from brain_api.storage.lstm.huggingface import HuggingFaceModelStorage
from brain_api.storage.lstm.local import LSTMHalalNewModelStorage
from brain_api.storage.patchtst.huggingface import (
    PatchTSTHalalNewHuggingFaceModelStorage,
    PatchTSTNiftyShariah500HuggingFaceModelStorage,
)
from brain_api.storage.patchtst.local import (
    PatchTSTHalalNewModelStorage,
    PatchTSTNiftyShariah500ModelStorage,
)
from brain_api.storage.sac.huggingface import SACHuggingFaceModelStorage
from brain_api.storage.sac.local import (
    SACHalalFilteredModelStorage,
    SACHalalModelStorage,
)
from brain_api.universe.halal import get_halal_symbols
from brain_api.universe.halal_filtered import get_halal_filtered_symbols
from brain_api.universe.halal_new import get_halal_new_symbols
from brain_api.universe.nifty_shariah_500 import get_nifty_shariah_500_symbols

if TYPE_CHECKING:
    pass


class ModelType(StrEnum):
    """Model families that own a bucket registry entry per universe."""

    LSTM = "lstm"
    PATCHTST = "patchtst"
    SAC = "sac"


class UnknownBucketError(ValueError):
    """Raised when a ``(model_type, universe)`` pair is not registered.

    Surfaced by training endpoints as a 422 so the caller can correct
    its ``universe`` field without retrying blindly.
    """


def _validate_ns_suffix(symbols: list[str]) -> None:
    """Enforce ``.NS`` suffix on every Indian NSE symbol.

    India PatchTST trains and infers on yfinance NSE tickers
    (``RELIANCE.NS`` etc.); a missing suffix would fetch the wrong
    instrument and silently produce garbage forecasts. Per AGENTS.md
    rule #1 (no silent fallbacks), we raise instead of stripping.
    """
    bad = [s for s in symbols if not s.endswith(".NS")]
    if bad:
        sample = bad[:5]
        raise ValueError(
            f"India universe symbols must end with .NS suffix. "
            f"Got {len(bad)} without suffix (sample: {sample})."
        )


def _validate_halal_filtered_count(symbols: list[str]) -> None:
    """Enforce the halal_filtered sticky-15 invariant.

    The ``halal_filtered`` bucket is contractually fixed at 15 names
    (rank-band sticky ``K_in=15`` on top of PatchTST scores). SAC's
    actor/critic dim and ``compute_version`` hash are baked at training
    time, so a slate of any other size would silently train a
    different-shaped network and break the bucket's ``current``
    artifact lineage. Per AGENTS.md rule #1 we raise rather than
    truncate or pad.

    The sibling ``halal`` bucket intentionally has NO count validator
    -- its size is whatever yfinance's ETF top-holdings produced this
    month (typical range 12-15) and the SAC config factory resizes the
    network to fit.
    """
    expected = 15
    if len(symbols) != expected:
        raise ValueError(
            f"halal_filtered bucket requires exactly {expected} symbols "
            f"(rank-band sticky K_in=15 invariant), got {len(symbols)}."
        )


def _validate_halal_min_count(symbols: list[str]) -> None:
    """Lower-bound check for the variable-size ``halal`` SAC bucket.

    ``halal`` is sourced from yfinance ETF top-holdings (SPUS, HLAL,
    SPTE) which can fluctuate month-to-month. A single-digit slate
    cannot meaningfully diversify a portfolio agent (and the existing
    ``_run_sac_full_training`` pipeline drops further if any symbol
    lacks price data, see ``routes/training/sac.py:215``), so we fail
    fast at the bucket layer instead of letting training crash later.
    """
    minimum = 5
    if len(symbols) < minimum:
        raise ValueError(
            f"halal bucket requires at least {minimum} symbols for SAC "
            f"training, got {len(symbols)}."
        )


@dataclass(frozen=True)
class BucketConfig:
    """Routing information for a ``(model_type, universe)`` bucket.

    Threaded by training endpoints through to the shared core training
    functions -- those functions consume ``local_storage_class``,
    ``hf_storage_class`` and ``hf_repo_getter`` to write artifacts and
    optionally upload to HF, and they receive ``bucket_name`` as the
    snapshot/job-namespace key.
    """

    model_type: ModelType
    universe: str
    bucket_name: str  # e.g. "patchtst_halal_new"; matches storage model_type
    model_label: (
        str  # human-readable, e.g. "LSTM halal_new"; used in 503 detail strings
    )
    local_storage_class: type
    hf_storage_class: type
    hf_repo_getter: Callable[[], str | None]
    symbols_resolver: Callable[[], list[str]]
    symbol_validator: Callable[[list[str]], None] | None = None


_BUCKETS: dict[tuple[ModelType, str], BucketConfig] = {}


def _register(cfg: BucketConfig) -> None:
    """Add a bucket config; collisions raise so duplicates fail fast."""
    key = (cfg.model_type, cfg.universe)
    if key in _BUCKETS:
        existing = _BUCKETS[key].bucket_name
        raise RuntimeError(
            f"Duplicate bucket registration for {key}: "
            f"existing={existing!r}, new={cfg.bucket_name!r}"
        )
    _BUCKETS[key] = cfg


def get_bucket(model_type: ModelType, universe: str) -> BucketConfig:
    """Look up the bucket config for a ``(model_type, universe)`` pair.

    Raises:
        UnknownBucketError: if the pair is not registered. Endpoints
            translate this to HTTP 422.
    """
    try:
        return _BUCKETS[(model_type, universe)]
    except KeyError as e:
        valid = sorted(list_universes_for(model_type))
        raise UnknownBucketError(
            f"No {model_type.value} bucket registered for universe "
            f"{universe!r}. Valid universes for {model_type.value}: {valid}"
        ) from e


def list_universes_for(model_type: ModelType) -> frozenset[str]:
    """Return the universes that have a bucket for this model type.

    Used as the endpoint allowlist when validating the request body's
    ``universe`` field.
    """
    return frozenset(u for (m, u) in _BUCKETS if m == model_type)


# ---------------------------------------------------------------------------
# Initial bucket registrations.
#
# Forecasters (LSTM, PatchTST) train on the *broad* universes
# (``halal_new`` for US, ``nifty_shariah_500`` for India). SAC trains on
# the sticky-15 ``halal_filtered`` slate that comes from running PatchTST
# + rank-band sticky on top of ``halal_new`` (see the universe-to-tier
# mapping in the plan / AGENTS.md). Adding a new bucket -- e.g.
# ``sac_halal`` for an A/B vs ``sac_halal_filtered`` -- is one new
# ``_register`` call plus a sibling storage subclass + new env var.
# ---------------------------------------------------------------------------

_register(
    BucketConfig(
        model_type=ModelType.LSTM,
        universe="halal_new",
        bucket_name="lstm_halal_new",
        model_label="LSTM halal_new",
        local_storage_class=LSTMHalalNewModelStorage,
        hf_storage_class=HuggingFaceModelStorage,
        hf_repo_getter=get_hf_lstm_halal_new_model_repo,
        symbols_resolver=get_halal_new_symbols,
    )
)

_register(
    BucketConfig(
        model_type=ModelType.PATCHTST,
        universe="halal_new",
        bucket_name="patchtst_halal_new",
        model_label="PatchTST halal_new",
        local_storage_class=PatchTSTHalalNewModelStorage,
        hf_storage_class=PatchTSTHalalNewHuggingFaceModelStorage,
        hf_repo_getter=get_hf_patchtst_halal_new_model_repo,
        symbols_resolver=get_halal_new_symbols,
    )
)

_register(
    BucketConfig(
        model_type=ModelType.PATCHTST,
        universe="nifty_shariah_500",
        bucket_name="patchtst_nifty_shariah_500",
        model_label="PatchTST nifty_shariah_500",
        local_storage_class=PatchTSTNiftyShariah500ModelStorage,
        hf_storage_class=PatchTSTNiftyShariah500HuggingFaceModelStorage,
        hf_repo_getter=get_hf_patchtst_nifty_shariah_500_model_repo,
        symbols_resolver=get_nifty_shariah_500_symbols,
        symbol_validator=_validate_ns_suffix,
    )
)

_register(
    BucketConfig(
        model_type=ModelType.SAC,
        universe="halal_filtered",
        bucket_name="sac_halal_filtered",
        model_label="SAC halal_filtered",
        local_storage_class=SACHalalFilteredModelStorage,
        hf_storage_class=SACHuggingFaceModelStorage,
        hf_repo_getter=get_hf_sac_halal_filtered_model_repo,
        symbols_resolver=get_halal_filtered_symbols,
        # The halal_filtered SAC slate is contractually fixed at 15
        # (rank-band sticky K_in=15). The validator pins this at the
        # bucket layer so the SAC training endpoint no longer needs a
        # process-wide ``config.n_stocks`` equality check (which would
        # otherwise prevent the parallel halal bucket from running).
        symbol_validator=_validate_halal_filtered_count,
    )
)

_register(
    BucketConfig(
        model_type=ModelType.SAC,
        universe="halal",
        bucket_name="sac_halal",
        model_label="SAC halal",
        local_storage_class=SACHalalModelStorage,
        hf_storage_class=SACHuggingFaceModelStorage,
        hf_repo_getter=get_hf_sac_halal_model_repo,
        symbols_resolver=get_halal_symbols,
        # The legacy halal universe is variable-size (yfinance ETF top
        # holdings; typical range 12-15 after dedup + US filter), so
        # we only enforce a lower bound here. The endpoint resizes
        # SAC's ``n_stocks`` and ``target_entropy`` via
        # ``make_sac_config_for_n_stocks`` to match the resolved slate.
        symbol_validator=_validate_halal_min_count,
    )
)


__all__ = [
    "BucketConfig",
    "ModelType",
    "UnknownBucketError",
    "get_bucket",
    "list_universes_for",
]
