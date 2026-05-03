"""Tests for deterministic model / snapshot hashing."""

from datetime import date

from brain_api.core.version import compute_model_hash, compute_model_version


def test_compute_model_hash_symbol_order_invariant() -> None:
    a = compute_model_hash(
        "lstm_halal_new",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["MSFT", "AAPL"],
        {"hidden": 64},
    )
    b = compute_model_hash(
        "lstm_halal_new",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["AAPL", "MSFT"],
        {"hidden": 64},
    )
    assert a == b
    assert len(a) == 12


def test_compute_model_hash_config_key_order_invariant() -> None:
    a = compute_model_hash(
        "m",
        date(2020, 1, 1),
        date(2020, 1, 5),
        [],
        {"zeta": 1, "alpha": 2},
    )
    b = compute_model_hash(
        "m",
        date(2020, 1, 1),
        date(2020, 1, 5),
        [],
        {"alpha": 2, "zeta": 1},
    )
    assert a == b


def test_compute_model_version_wraps_digest() -> None:
    digest = compute_model_hash(
        "lstm",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["AAPL"],
        {"k": 1},
    )
    ver = compute_model_version(
        "lstm",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["AAPL"],
        {"k": 1},
    )
    assert ver == f"v2020-12-31-{digest}"
