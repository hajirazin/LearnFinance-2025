"""Shared metadata and bucket bindings for isolated HF reuse tests."""

from __future__ import annotations

from dataclasses import replace

from brain_api.core.model_buckets import ModelType, get_bucket

_FAKE_METADATA: dict = {
    "data_window": {"start": "2016-01-01", "end": "2026-04-30"},
    "metrics": {"loss": 0.42},
    "training_evidence": {"latest_label_end": "2026-04-24", "latest_label_symbols": 2},
    "promoted": True,
    "prior_version": "v2026-04-24-prev",
    "failure_reasons": [],
}


def _fake_metadata_with(**overrides) -> dict:
    """Return a copy of the canonical fake metadata with overrides applied.

    Tests that need extra fields (``symbols`` for SAC) layer them on
    top so they cannot accidentally mutate the shared baseline.
    """
    out = dict(_FAKE_METADATA)
    out.update(overrides)
    return out


def _fail_if_live_yfinance(*_args, **_kwargs):
    raise AssertionError(
        "test_sac_full_endpoint_force_true_bypasses_hf_helper must not "
        "call load_prices_yfinance; stub _run_sac_full_training"
    )


def _bind_lstm_bucket(monkeypatch, tmpdir, *, symbols=None):
    """Helper: rebind the LSTM halal_new bucket to a tmpdir storage."""
    from brain_api.core import model_buckets
    from brain_api.storage.lstm.local import LSTMHalalNewModelStorage

    storage = LSTMHalalNewModelStorage(base_path=tmpdir)
    original = get_bucket(ModelType.LSTM, "halal_new")
    patched = replace(
        original,
        local_storage_class=lambda: storage,
        symbols_resolver=lambda: list(symbols or ["AAPL", "MSFT"]),
    )
    monkeypatch.setitem(
        model_buckets._BUCKETS,
        (ModelType.LSTM, "halal_new"),
        patched,
    )
    return storage
