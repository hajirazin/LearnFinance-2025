"""Domain entities - pure dataclasses with no external dependencies.

These entities represent the core business objects in the domain model.
"""

from brain_api.domain.entities.allocation import HRPResult
from brain_api.domain.entities.fundamentals import (
    FetchRecord,
    FundamentalRatios,
    QuarterlyStatement,
)
from brain_api.domain.entities.lstm import (
    DatasetResult,
    InferenceFeatures,
    LSTMConfig,
    SymbolPrediction,
    TrainingResult,
    WeekBoundaries,
)
from brain_api.domain.entities.sentiment import (
    Article,
    FinBERTResult,
    ScoredArticle,
    SymbolSentiment,
)

__all__ = [
    # LSTM
    "LSTMConfig",
    "DatasetResult",
    "TrainingResult",
    "WeekBoundaries",
    "InferenceFeatures",
    "SymbolPrediction",
    # Allocation
    "HRPResult",
    # Sentiment
    "Article",
    "FinBERTResult",
    "ScoredArticle",
    "SymbolSentiment",
    # Fundamentals
    "QuarterlyStatement",
    "FundamentalRatios",
    "FetchRecord",
]

