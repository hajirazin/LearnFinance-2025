"""Tests for strategy partition constants.

Partition strings drive sticky-history scoping. Sharing a partition
between two strategies corrupts the carry-set even when the strategies
sit in different sticky-history tables. These tests are the safety net
that catches accidental duplication at PR review time.
"""

from __future__ import annotations

from brain_api.core import strategy_partitions as sp


def test_all_partitions_pairwise_distinct():
    assert len(sp.ALL_PARTITIONS) == len(set(sp.ALL_PARTITIONS))


def test_halal_filtered_alpha_value():
    assert sp.HALAL_FILTERED_ALPHA_PARTITION == "halal_filtered_alpha"


def test_halal_filtered_alpha_in_all_partitions():
    assert sp.HALAL_FILTERED_ALPHA_PARTITION in sp.ALL_PARTITIONS


def test_known_partitions_present():
    assert sp.HALAL_NEW_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_NEW_ALPHA_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_INDIA_ALPHA_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_FILTERED_ALPHA_PARTITION in sp.ALL_PARTITIONS
