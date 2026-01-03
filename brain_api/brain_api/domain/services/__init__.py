"""Domain services - pure business logic with no external dependencies.

These services contain the core algorithms and business logic.
They depend only on domain entities and standard library types.
"""

from brain_api.domain.services.fundamentals_computation import (
    compute_ratios,
    get_statement_as_of,
    parse_quarterly_statements,
)
from brain_api.domain.services.hrp import (
    compute_hrp_weights,
)
from brain_api.domain.services.lstm_computation import (
    build_inference_features_from_data,
    classify_direction,
    compute_version,
    compute_week_boundaries_simple,
    compute_weekly_return,
    extract_trading_weeks,
)
from brain_api.domain.services.sentiment_aggregation import (
    aggregate_symbol_sentiment,
    compute_recency_weight,
)

__all__ = [
    # HRP
    "compute_hrp_weights",
    # Sentiment
    "compute_recency_weight",
    "aggregate_symbol_sentiment",
    # LSTM
    "compute_version",
    "compute_week_boundaries_simple",
    "extract_trading_weeks",
    "compute_weekly_return",
    "build_inference_features_from_data",
    "classify_direction",
    # Fundamentals
    "parse_quarterly_statements",
    "get_statement_as_of",
    "compute_ratios",
]

