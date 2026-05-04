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


def test_halal_india_filtered_alpha_value():
    assert sp.HALAL_INDIA_FILTERED_ALPHA_PARTITION == "halal_india_filtered_alpha"


def test_halal_filtered_alpha_in_all_partitions():
    assert sp.HALAL_FILTERED_ALPHA_PARTITION in sp.ALL_PARTITIONS


def test_halal_india_filtered_alpha_in_all_partitions():
    assert sp.HALAL_INDIA_FILTERED_ALPHA_PARTITION in sp.ALL_PARTITIONS


def test_halal_india_double_hrp_value():
    # Pin the exact partition string -- changing this constant after a
    # production write silently re-cold-starts the carry-set on the
    # next run.
    assert sp.HALAL_INDIA_DOUBLE_HRP_PARTITION == "halal_india_double_hrp"


def test_halal_india_double_hrp_in_all_partitions():
    assert sp.HALAL_INDIA_DOUBLE_HRP_PARTITION in sp.ALL_PARTITIONS


def test_known_partitions_present():
    assert sp.HALAL_NEW_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_NEW_ALPHA_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_INDIA_ALPHA_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_INDIA_DOUBLE_HRP_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_FILTERED_ALPHA_PARTITION in sp.ALL_PARTITIONS
    assert sp.HALAL_INDIA_FILTERED_ALPHA_PARTITION in sp.ALL_PARTITIONS


def test_monthly_screening_partitions_distinct_from_weekly_two_stage():
    """Single-stage monthly partitions must not collide with two-stage weekly partitions.

    The two halal_india* partitions in particular are intentionally
    distinct: ``halal_india_alpha`` lives in ``stage1_weight_history``
    (weekly Alpha-HRP, two-stage), while ``halal_india_filtered_alpha``
    lives in ``screening_history`` (monthly halal_india universe build,
    single-stage). Sharing a string would let one cadence's writes
    corrupt the other's carry-set even though the tables are separate.
    """
    monthly_screening = {
        sp.HALAL_FILTERED_ALPHA_PARTITION,
        sp.HALAL_INDIA_FILTERED_ALPHA_PARTITION,
    }
    weekly_two_stage = {
        sp.HALAL_NEW_PARTITION,
        sp.HALAL_NEW_ALPHA_PARTITION,
        sp.HALAL_INDIA_ALPHA_PARTITION,
        sp.HALAL_INDIA_DOUBLE_HRP_PARTITION,
    }
    assert monthly_screening.isdisjoint(weekly_two_stage)


def test_india_weight_band_and_rank_band_partitions_distinct():
    """India Double HRP (weight-band) and India Alpha-HRP (rank-band)
    MUST live in different partitions.

    The two selectors are different mathematical primitives -- merging
    their carry-sets would mean a stock kept by rank-band semantics one
    week could falsely be treated as a weight-band hold the next week
    (or vice versa), corrupting both signals. AGENTS.md rule #2 +
    sticky-history isolation rule both force this separation.
    """
    assert sp.HALAL_INDIA_DOUBLE_HRP_PARTITION != sp.HALAL_INDIA_ALPHA_PARTITION
