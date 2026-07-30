"""File storage helpers for fundamentals data."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_fundamentals_dir(base_path: Path, symbol: str) -> Path:
    """Get directory for a symbol's fundamental data.

    Args:
        base_path: Base data directory
        symbol: Stock ticker

    Returns:
        Path to symbol's fundamentals directory
    """
    return base_path / "raw" / "fundamentals" / symbol


def get_legacy_nested_fundamentals_dir(base_path: Path, symbol: str) -> Path:
    """Return the path produced by the former double-appending refresh bug."""
    return base_path / "raw" / "fundamentals" / "raw" / "fundamentals" / symbol


def _migrate_legacy_file(legacy_path: Path, canonical_path: Path) -> Path:
    """Copy a legacy cache file into the canonical tree without overwriting."""
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = canonical_path.with_name(
        f".{canonical_path.name}.{os.getpid()}.tmp"
    )
    try:
        shutil.copy2(legacy_path, temporary_path)
        with contextlib.suppress(FileExistsError):
            temporary_path.rename(canonical_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return canonical_path


def save_raw_response(
    base_path: Path,
    symbol: str,
    endpoint: str,
    data: dict[str, Any],
) -> Path:
    """Save raw API response to JSON file.

    Args:
        base_path: Base data directory
        symbol: Stock ticker
        endpoint: "income_statement" or "balance_sheet"
        data: Raw API response

    Returns:
        Path where file was saved
    """
    dir_path = get_fundamentals_dir(base_path, symbol)
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"{endpoint}.json"

    # Add metadata to the saved file
    wrapped_data = {
        "symbol": symbol,
        "endpoint": endpoint,
        "fetched_at": datetime.now(UTC).isoformat(),
        "response": data,
    }

    with open(file_path, "w") as f:
        json.dump(wrapped_data, f, indent=2)

    return file_path


def load_raw_response(
    base_path: Path,
    symbol: str,
    endpoint: str,
) -> dict[str, Any] | None:
    """Load raw API response from JSON file.

    Args:
        base_path: Base data directory
        symbol: Stock ticker
        endpoint: "income_statement" or "balance_sheet"

    Returns:
        Wrapped data dict with "response" key, or None if not found
    """
    file_path = get_fundamentals_dir(base_path, symbol) / f"{endpoint}.json"

    if not file_path.exists():
        legacy_path = (
            get_legacy_nested_fundamentals_dir(base_path, symbol) / f"{endpoint}.json"
        )
        if not legacy_path.exists():
            return None
        file_path = _migrate_legacy_file(legacy_path, file_path)

    with open(file_path) as f:
        return json.load(f)
