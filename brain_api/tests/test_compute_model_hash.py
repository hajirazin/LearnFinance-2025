"""Tests for deterministic model and snapshot hashing."""

import hashlib
import json
import re
from datetime import date

from brain_api.core.version import (
    compute_model_hash,
    compute_model_version,
    compute_snapshot_identity_hash,
)


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


def test_compute_model_hash_changes_when_symbols_change() -> None:
    a = compute_model_hash(
        "lstm_halal_new",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["AAPL"],
        {"hidden": 64},
    )
    b = compute_model_hash(
        "lstm_halal_new",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["MSFT"],
        {"hidden": 64},
    )
    assert a != b


def test_compute_snapshot_identity_hash_matches_canonical_payload() -> None:
    canonical = {
        "model": "patchtst_halal_new",
        "cutoff_date": "2020-12-31",
        "config": {"alpha": 2, "zeta": 1},
    }
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()
    ).hexdigest()[:12]

    digest = compute_snapshot_identity_hash(
        "patchtst_halal_new",
        date(2020, 12, 31),
        {"zeta": 1, "alpha": 2},
    )

    assert digest == expected


def test_compute_snapshot_identity_hash_is_twelve_lowercase_hex() -> None:
    digest = compute_snapshot_identity_hash(
        "lstm_halal_new", date(2020, 12, 31), {"hidden": 64}
    )
    assert re.fullmatch(r"[0-9a-f]{12}", digest)


def test_compute_snapshot_identity_hash_is_config_key_order_invariant() -> None:
    a = compute_snapshot_identity_hash("m", date(2020, 12, 31), {"zeta": 1, "alpha": 2})
    b = compute_snapshot_identity_hash("m", date(2020, 12, 31), {"alpha": 2, "zeta": 1})
    assert a == b


def test_compute_snapshot_identity_hash_changes_with_bucket() -> None:
    cutoff = date(2020, 12, 31)
    config = {"hidden": 64}
    assert compute_snapshot_identity_hash(
        "lstm_halal_new", cutoff, config
    ) != compute_snapshot_identity_hash("patchtst_halal_new", cutoff, config)


def test_compute_snapshot_identity_hash_changes_with_cutoff() -> None:
    config = {"hidden": 64}
    assert compute_snapshot_identity_hash(
        "lstm_halal_new", date(2020, 12, 31), config
    ) != compute_snapshot_identity_hash("lstm_halal_new", date(2021, 12, 31), config)


def test_compute_snapshot_identity_hash_changes_with_config() -> None:
    cutoff = date(2020, 12, 31)
    assert compute_snapshot_identity_hash(
        "lstm_halal_new", cutoff, {"hidden": 64}
    ) != compute_snapshot_identity_hash("lstm_halal_new", cutoff, {"hidden": 128})


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
