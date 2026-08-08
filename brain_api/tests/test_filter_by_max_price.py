from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from brain_api.core.filters.filter_by_max_price import (
    MaxPriceExclusion,
    filter_symbols_by_max_price,
)


@pytest.fixture
def mock_load_prices():
    with patch(
        "brain_api.core.filters.filter_by_max_price.load_prices_yfinance"
    ) as mock:
        yield mock


def test_filter_symbols_by_max_price_excludes_correctly(mock_load_prices):
    """Above-max and missing-price are distinct exclusion reasons."""
    df_a = pd.DataFrame({"close": [3900.0, 4000.0]})
    df_b = pd.DataFrame({"close": [5900.0, 6000.0]})
    df_c = pd.DataFrame()

    mock_load_prices.return_value = {
        "SYM_A": df_a,
        "SYM_B": df_b,
        "SYM_C": df_c,
    }

    symbols = ["SYM_A", "SYM_B", "SYM_C"]
    as_of = date(2026, 1, 9)

    qualifying, excluded = filter_symbols_by_max_price(symbols, as_of=as_of)

    assert qualifying == ["SYM_A"]
    assert excluded == [
        MaxPriceExclusion(symbol="SYM_B", price=6000.0, reason="above_max"),
        MaxPriceExclusion(symbol="SYM_C", price=None, reason="missing_price"),
    ]

    mock_load_prices.assert_called_once()
    args, _kwargs = mock_load_prices.call_args
    assert args[0] == symbols
    assert args[1] == date(2026, 1, 2)
    assert args[2] == as_of


def test_filter_symbols_empty_input(mock_load_prices):
    """Empty list returns empty results immediately."""
    qualifying, excluded = filter_symbols_by_max_price([])

    assert qualifying == []
    assert excluded == []
    mock_load_prices.assert_not_called()
