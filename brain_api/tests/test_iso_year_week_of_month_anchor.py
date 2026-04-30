"""Tests for ``iso_year_week_of_month_anchor``.

Period-key convention used by single-stage monthly screening
strategies (e.g. ``halal_filtered_alpha``). The helper MUST be
idempotent across days within the same calendar month and resolve
to a real ISO YYYYWW so lex-comparison still gives chronological
ordering across year boundaries.
"""

from __future__ import annotations

from datetime import date

from brain_api.core.sticky_selection import iso_year_week_of_month_anchor


def test_april_2026_first_monday_is_apr_6():
    assert iso_year_week_of_month_anchor(date(2026, 4, 1)) == "202615"
    assert iso_year_week_of_month_anchor(date(2026, 4, 30)) == "202615"


def test_january_2026_first_monday_is_jan_5():
    assert iso_year_week_of_month_anchor(date(2026, 1, 1)) == "202602"
    assert iso_year_week_of_month_anchor(date(2026, 1, 15)) == "202602"


def test_june_2026_first_is_monday():
    assert iso_year_week_of_month_anchor(date(2026, 6, 1)) == "202623"


def test_idempotent_across_days_in_same_month():
    anchors = {iso_year_week_of_month_anchor(date(2026, 4, d)) for d in range(1, 31)}
    assert anchors == {"202615"}


def test_year_boundary_january_2027():
    """Jan 2027: 1st is Friday, first Monday is Jan 4 (ISO week 1 of 2027)."""
    assert iso_year_week_of_month_anchor(date(2027, 1, 1)) == "202701"
    assert iso_year_week_of_month_anchor(date(2027, 1, 31)) == "202701"


def test_lex_ordering_across_year_boundary():
    dec_2026 = iso_year_week_of_month_anchor(date(2026, 12, 15))
    jan_2027 = iso_year_week_of_month_anchor(date(2027, 1, 15))
    assert dec_2026 < jan_2027
