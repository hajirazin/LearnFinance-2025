"""File storage helpers for fundamentals data."""

from __future__ import annotations

import json
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
        return None
    
    with open(file_path) as f:
        return json.load(f)


