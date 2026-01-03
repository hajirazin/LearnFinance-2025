"""Hierarchical Risk Parity (HRP) domain service.

Implementation of López de Prado's HRP algorithm (2016) for
risk-based portfolio allocation using hierarchical clustering.

This module contains pure mathematical functions with no external dependencies
beyond numpy/pandas/scipy.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


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
        current_weight = weights[sorted_symbols[start]]
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

    This is a pure function - it takes returns data and produces weights.

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

