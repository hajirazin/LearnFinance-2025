"""Recover missing India PatchTST bars from the NSE cash-market archive."""

from datetime import date
from io import BytesIO
from math import isclose
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import requests

from brain_api.core.patchtst.price_history import normalize_price_dates

_ARCHIVE = "https://nsearchives.nseindia.com/content"
_UDIFF_START = date(2024, 7, 8)
_FIELDS = ["close", "volume"]
NSE_RECOVERED_BAR = "nse_recovered_bar"


def _cash_report(day: date) -> pd.DataFrame:
    if day < _UDIFF_START:
        month = f"{day:%b}".upper()
        filename = f"historical/EQUITIES/{day:%Y}/{month}/cm{day:%d}{month}{day:%Y}bhav.csv.zip"
        columns = {
            "SYMBOL": "symbol",
            "SERIES": "series",
            "TIMESTAMP": "day",
            "CLOSE": "close",
            "TOTTRDQTY": "volume",
        }
    else:
        filename = f"cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
        columns = {
            "TckrSymb": "symbol",
            "SctySrs": "series",
            "TradDt": "day",
            "ClsPric": "close",
            "TtlTradgVol": "volume",
        }
    try:
        response = requests.get(
            f"{_ARCHIVE}/{filename}", headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        )
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            csv_files = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if len(csv_files) != 1:
                raise ValueError("Expected one cash-market CSV")
            with archive.open(csv_files[0]) as csv:
                rows = pd.read_csv(csv)
        rows.columns = rows.columns.str.strip()
        rows = rows.rename(columns=columns)
        if not pd.to_datetime(rows["day"]).dt.date.eq(day).all():
            raise ValueError("Unexpected trading date")
        equities = rows.loc[rows["series"].str.strip().eq("EQ")].copy()
        equities["symbol"] = equities["symbol"].str.strip() + ".NS"
        return equities.set_index("symbol", verify_integrity=True)[_FIELDS]
    except (requests.RequestException, BadZipFile, KeyError, ValueError) as exc:
        raise ValueError(
            f"NSE cash-market report unavailable for {day}: {exc}"
        ) from exc


def restore_nse_cash_bar_gaps(
    prices: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    """Recover isolated market-wide feed gaps using matching surrounding closes."""
    if not prices:
        return prices
    frames = {
        symbol: normalize_price_dates(frame).reindex(sessions)[_FIELDS]
        for symbol, frame in prices.items()
    }
    valid = pd.DataFrame(
        {
            symbol: (np.isfinite(frame) & frame.gt(0)).all(axis=1)
            for symbol, frame in frames.items()
        }
    )
    for frame in frames.values():
        frame[NSE_RECOVERED_BAR] = False
    comparable = valid.shift(fill_value=False) & valid.shift(-1, fill_value=False)
    missing = comparable & ~valid
    widespread = missing.sum(axis=1) > comparable.sum(axis=1) / 2
    reports = {}
    for position in np.flatnonzero(widespread.to_numpy()):
        prior, current, following = sessions[position - 1 : position + 2]
        for session in (prior, current, following):
            if session not in reports:
                reports[session] = _cash_report(session.date())
        for symbol in missing.columns[missing.loc[current]]:
            if not all(
                symbol in reports[session].index
                for session in (prior, current, following)
            ):
                continue
            frame = frames[symbol]
            row = reports[current].loc[symbol]
            raw_prior = reports[prior].at[symbol, "close"]
            raw_following = reports[following].at[symbol, "close"]
            quotes = [raw_prior, raw_following, row["close"], row["volume"]]
            if not all(np.isfinite(quote) and quote > 0 for quote in quotes):
                continue
            factor = frame.at[prior, "close"] / raw_prior
            following_factor = frame.at[following, "close"] / raw_following
            if isclose(factor, following_factor, rel_tol=0.001):
                frame.loc[current, _FIELDS] = [row["close"] * factor, row["volume"]]
                frame.at[current, NSE_RECOVERED_BAR] = True
    return frames
