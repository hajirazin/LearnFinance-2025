from unittest.mock import patch

import pandas as pd
import pytest

from brain_api.core.filters.filter_by_max_price import filter_symbols_by_max_price


@pytest.fixture
def mock_load_prices():
    with patch(
        "brain_api.core.filters.filter_by_max_price.load_prices_yfinance"
    ) as mock:
        yield mock


def test_filter_symbols_by_max_price_excludes_correctly(mock_load_prices):
    """Test that symbols exceeding the max price are excluded, while others qualify."""
    # Setup mock prices
    # We create DataFrames with a 'close' column to mimic yfinance output
    df_a = pd.DataFrame({"close": [3900.0, 4000.0]})  # Under 5000
    df_b = pd.DataFrame({"close": [5900.0, 6000.0]})  # Over 5000
    df_c = pd.DataFrame()  # Missing data

    mock_load_prices.return_value = {
        "SYM_A": df_a,
        "SYM_B": df_b,
        "SYM_C": df_c,
    }

    symbols = ["SYM_A", "SYM_B", "SYM_C"]

    qualifying, excluded = filter_symbols_by_max_price(symbols)

    assert qualifying == ["SYM_A"]
    # SYM_C has missing data, so its actual_price is evaluated as None in the logic, falling back to 0.0
    assert excluded == [("SYM_B", 6000.0), ("SYM_C", 0.0)]

    mock_load_prices.assert_called_once()
    args, _ = mock_load_prices.call_args
    assert args[0] == symbols


def test_filter_symbols_empty_input(mock_load_prices):
    """Test that an empty list returns empty results immediately."""
    qualifying, excluded = filter_symbols_by_max_price([])

    assert qualifying == []
    assert excluded == []
    mock_load_prices.assert_not_called()
