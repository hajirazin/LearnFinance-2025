"""Check self-financing, drift, trade-side fees, and fee-funded benchmarks."""

import pytest

from book import PortfolioBook, buy_and_hold_book


def test_first_buy_is_self_financing_and_fees_reduce_cash():
    book = PortfolioBook(10_000)
    after, ledger = book.rebalance({"A": 0.8, "CASH": 0.2}, {"A": 100})
    assert after.shares == {"A": 80}
    assert after.cash == pytest.approx(2000 - ledger["fees_usd"])
    assert after.value({"A": 100}) == pytest.approx(10_000 - ledger["fees_usd"])
    assert [t["side"] for t in ledger["trades"]] == ["buy"]
    assert ledger["fees_usd"] > 0
    assert book.cash == 10_000 and not book.shares


def test_marking_and_weekend_gap_preserve_shares_without_resetting_nav():
    book = PortfolioBook(2000, {"A": 80})
    assert book.value({"A": 110}) == 10_800
    assert book.value({"A": 90}) == 9200
    assert book.shares["A"] == 80
    assert book.weights({"A": 110})["A"] != book.weights({"A": 90})["A"]


def test_unchanged_book_has_no_trades_or_extra_fee():
    book = PortfolioBook(2000, {"A": 80})
    prices = {"A": 110}
    after, ledger = book.rebalance(book.weights(prices), prices)
    assert not ledger["trades"] and ledger["fees_usd"] == 0
    assert after.shares["A"] == pytest.approx(80)
    assert after.cash == pytest.approx(2000)


def test_rotation_charges_both_sell_and_buy_at_actual_nav():
    book = PortfolioBook(2000, {"A": 100})
    after, ledger = book.rebalance({"B": 0.8, "CASH": 0.2}, {"A": 80, "B": 40})
    assert after.shares == {"B": 200}
    assert {t["side"] for t in ledger["trades"]} == {"sell", "buy"}
    assert len(ledger["trades"]) == 2
    assert after.cash == pytest.approx(2000 - ledger["fees_usd"])
    assert sum(t["total"] for t in ledger["trades"]) == pytest.approx(
        ledger["fees_usd"]
    )


def test_held_asset_missing_mark_fails_instead_of_imputing():
    with pytest.raises(ValueError, match="held asset A"):
        PortfolioBook(1000, {"A": 5}).value({})


def test_benchmark_entry_funds_its_own_fee_without_negative_cash():
    book, fee = buy_and_hold_book("SPY", 500, 10_000)
    assert book.cash >= 0 and book.shares["SPY"] < 20
    assert book.value({"SPY": 500}) + fee == pytest.approx(10_000)
    assert book.value({"SPY": 550}) == pytest.approx(
        book.cash + book.shares["SPY"] * 550
    )
