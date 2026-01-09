"""Hierarchical Risk Parity (HRP) portfolio allocation.

Implementation of López de Prado's HRP algorithm (2016) for
risk-based portfolio allocation using hierarchical clustering.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# ============================================================================
# Data structures
# ============================================================================


@dataclass
class HRPResult:
    """Result of HRP allocation computation."""

    # Percentage weights (sum to 100)
    percentage_weights: dict[str, float]

    # Symbols actually used in allocation
    symbols_used: list[str]

    # Symbols excluded (insufficient data)
    symbols_excluded: list[str]

    # Parameters used
    lookback_days: int
    as_of_date: str  # ISO format


# ============================================================================
# Core HRP Algorithm
# ============================================================================


def _compute_correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation matrix to distance matrix.

    Distance formula: d[i,j] = sqrt(0.5 * (1 - corr[i,j]))

    This maps correlation:
    - corr = 1.0 (perfect positive) -> d = 0
    - corr = 0.0 (uncorrelated) -> d = 0.707
    - corr = -1.0 (perfect negative) -> d = 1.0

    Args:
        corr: Correlation matrix (DataFrame)

    Returns:
        Distance matrix (DataFrame)
    """
    return np.sqrt(0.5 * (1 - corr))


def _get_quasi_diagonal_order(link: np.ndarray, n_items: int) -> list[int]:
    """Get quasi-diagonal ordering of items from hierarchical clustering.

    This reorders items so that similar items are adjacent,
    making the covariance matrix approximately block-diagonal.

    Args:
        link: Linkage matrix from scipy hierarchical clustering
        n_items: Number of original items

    Returns:
        List of indices in quasi-diagonal order
    """
    return list(leaves_list(link))


def _get_cluster_variance(cov: pd.DataFrame, symbols: list[str]) -> float:
    """Compute variance of an inverse-variance weighted cluster.

    For a cluster of assets, the optimal (minimum variance) weights
    are proportional to inverse variances. This computes the
    resulting cluster variance.

    Args:
        cov: Covariance matrix
        symbols: Symbols in the cluster

    Returns:
        Cluster variance
    """
    if len(symbols) == 1:
        return cov.loc[symbols[0], symbols[0]]

    # Subset covariance
    cov_slice = cov.loc[symbols, symbols]

    # Inverse variance weights within cluster
    ivp = 1.0 / np.diag(cov_slice)
    ivp = ivp / ivp.sum()

    # Cluster variance = w' * Cov * w
    return float(np.dot(ivp, np.dot(cov_slice, ivp)))


def _recursive_bisection(
    cov: pd.DataFrame,
    sorted_symbols: list[str],
) -> dict[str, float]:
    """Allocate weights via recursive bisection.

    The algorithm:
    1. Start with all weight (1.0) on the full cluster
    2. Split cluster in half
    3. Allocate weight to each half inversely proportional to variance
    4. Recursively apply to each half until reaching individual assets

    Args:
        cov: Covariance matrix
        sorted_symbols: Symbols in quasi-diagonal order

    Returns:
        Dict mapping symbol -> weight (sum to 1.0)
    """
    # Initialize weights
    weights = pd.Series(1.0, index=sorted_symbols)

    # Track clusters to process: list of (start_idx, end_idx)
    clusters = [(0, len(sorted_symbols))]

    while clusters:
        start, end = clusters.pop(0)

        if end - start <= 1:
            # Single item, no split needed
            continue

        # Split in half
        mid = (start + end) // 2
        left_symbols = sorted_symbols[start:mid]
        right_symbols = sorted_symbols[mid:end]

        # Compute cluster variances
        left_var = _get_cluster_variance(cov, left_symbols)
        right_var = _get_cluster_variance(cov, right_symbols)

        # Allocate inversely proportional to variance
        total_inv_var = 1.0 / left_var + 1.0 / right_var
        left_weight = (1.0 / left_var) / total_inv_var
        right_weight = (1.0 / right_var) / total_inv_var

        # Update weights
        current_weight = weights[
            sorted_symbols[start]
        ]  # All in cluster have same weight at this point
        weights[left_symbols] = current_weight * left_weight
        weights[right_symbols] = current_weight * right_weight

        # Add sub-clusters for further processing
        if len(left_symbols) > 1:
            clusters.append((start, mid))
        if len(right_symbols) > 1:
            clusters.append((mid, end))

    return weights.to_dict()


def compute_hrp_weights(
    returns: pd.DataFrame,
) -> dict[str, float]:
    """Compute HRP portfolio weights from returns data.

    Full HRP algorithm:
    1. Compute correlation matrix
    2. Convert to distance matrix
    3. Hierarchical clustering (single linkage)
    4. Quasi-diagonalize covariance matrix
    5. Recursive bisection for weight allocation

    Args:
        returns: DataFrame with columns = symbols, rows = dates,
                 values = daily returns

    Returns:
        Dict mapping symbol -> weight (weights sum to 1.0)
    """
    if returns.empty or len(returns.columns) == 0:
        return {}

    # Handle single asset case
    if len(returns.columns) == 1:
        return {returns.columns[0]: 1.0}

    # Handle two asset case (HRP degenerates to inverse vol)
    if len(returns.columns) == 2:
        vols = returns.std()
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        return weights.to_dict()

    # Step 1: Compute correlation and covariance matrices
    corr = returns.corr()
    cov = returns.cov()

    # Step 2: Convert correlation to distance
    dist = _compute_correlation_distance(corr)

    # Step 3: Hierarchical clustering
    # Convert to condensed distance matrix for scipy
    dist_condensed = squareform(dist.values, checks=False)
    link = linkage(dist_condensed, method="single")

    # Step 4: Get quasi-diagonal ordering
    sort_idx = _get_quasi_diagonal_order(link, len(returns.columns))
    sorted_symbols = [returns.columns[i] for i in sort_idx]

    # Step 5: Recursive bisection
    weights = _recursive_bisection(cov, sorted_symbols)

    return weights


# ============================================================================
# High-level API
# ============================================================================


def compute_hrp_allocation(
    prices: dict[str, pd.DataFrame],
    lookback_days: int = 252,
    min_data_days: int = 60,
    as_of_date: date | None = None,
) -> HRPResult:
    """Compute HRP allocation from price data.

    This is the main entry point for HRP allocation.

    Args:
        prices: Dict mapping symbol -> DataFrame with 'close' column
                (case-insensitive column names supported)
        lookback_days: Number of trading days for return calculation
        min_data_days: Minimum days of data required to include a symbol
        as_of_date: Reference date (defaults to today)

    Returns:
        HRPResult with percentage weights and metadata
    """
    if as_of_date is None:
        as_of_date = date.today()

    # Build returns DataFrame
    returns_dict = {}
    symbols_excluded = []

    for symbol, df in prices.items():
        if df is None or df.empty:
            symbols_excluded.append(symbol)
            continue

        # Find close column (case-insensitive)
        close_col = None
        for col in df.columns:
            if col.lower() == "close":
                close_col = col
                break

        if close_col is None:
            symbols_excluded.append(symbol)
            continue

        # Get close prices and compute returns
        close = df[close_col].dropna()

        if len(close) < min_data_days:
            symbols_excluded.append(symbol)
            continue

        # Use only the most recent lookback_days
        close = close.tail(lookback_days + 1)  # +1 because pct_change drops first row
        returns = close.pct_change().dropna()

        if len(returns) < min_data_days - 1:
            symbols_excluded.append(symbol)
            continue

        returns_dict[symbol] = returns

    # Build returns DataFrame (align on dates)
    if not returns_dict:
        return HRPResult(
            percentage_weights={},
            symbols_used=[],
            symbols_excluded=symbols_excluded,
            lookback_days=lookback_days,
            as_of_date=as_of_date.isoformat(),
        )

    returns_df = pd.DataFrame(returns_dict).dropna()

    # Check if we still have data after alignment
    if returns_df.empty or len(returns_df.columns) == 0:
        return HRPResult(
            percentage_weights={},
            symbols_used=[],
            symbols_excluded=list(prices.keys()),
            lookback_days=lookback_days,
            as_of_date=as_of_date.isoformat(),
        )

    # Any symbols that got dropped during alignment
    used_symbols = list(returns_df.columns)
    for symbol in returns_dict:
        if symbol not in used_symbols and symbol not in symbols_excluded:
            symbols_excluded.append(symbol)

    # Compute HRP weights
    weights = compute_hrp_weights(returns_df)

    # Convert to percentage (multiply by 100)
    percentage_weights = {
        symbol: round(weight * 100, 4) for symbol, weight in weights.items()
    }

    return HRPResult(
        percentage_weights=percentage_weights,
        symbols_used=used_symbols,
        symbols_excluded=symbols_excluded,
        lookback_days=lookback_days,
        as_of_date=as_of_date.isoformat(),
    )
