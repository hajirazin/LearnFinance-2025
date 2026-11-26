"""Read and cache NSE capital-market holiday evidence."""

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from brain_api.storage.base import DEFAULT_DATA_PATH

HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
_BUNDLED_HOLIDAYS = Path(__file__).with_name("nse_cash_holidays.json")


@dataclass(frozen=True)
class NseCashHolidays:
    start: date
    end: date
    holidays: frozenset[date]


def _read_holidays(path: Path) -> NseCashHolidays:
    try:
        data = json.loads(path.read_text())
        first = date.fromisoformat(data["start"])
        last = date.fromisoformat(data["end"])
        if last < first or not data["sources"]:
            raise ValueError("Missing calendar coverage or sources")
        holidays = frozenset(date.fromisoformat(day) for day in data["holidays"])
        if any(not first <= day <= last for day in holidays):
            raise ValueError("Holiday outside calendar coverage")
        return NseCashHolidays(first, last, holidays)
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ValueError(
            f"Unable to read NSE cash holidays from {path}: {exc}"
        ) from exc


def _fetch_nse_cash_holidays(year: int) -> frozenset[date]:
    try:
        response = requests.get(
            HOLIDAY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        )
        response.raise_for_status()
        holidays = frozenset(
            day
            for row in response.json()["CM"]
            if (day := datetime.strptime(row["tradingDate"], "%d-%b-%Y").date()).year
            == year
        )
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to fetch NSE cash holidays for {year}: {exc}"
        ) from exc
    if not holidays:
        raise ValueError(f"NSE has no cash-market holiday list available for {year}")
    return holidays


def _cache_year(path: Path, coverage: NseCashHolidays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(
                {
                    "start": coverage.start.isoformat(),
                    "end": coverage.end.isoformat(),
                    "sources": [HOLIDAY_URL],
                    "holidays": sorted(day.isoformat() for day in coverage.holidays),
                },
                stream,
            )
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_nse_cash_holidays(
    end: date, *, refresh_missing_years: bool
) -> NseCashHolidays:
    override = os.environ.get("NSE_CASH_CALENDAR_PATH")
    coverage = _read_holidays(Path(override) if override else _BUNDLED_HOLIDAYS)
    if override:
        return coverage

    for year in range(coverage.end.year + 1, end.year + 1):
        first, last = date(year, 1, 1), date(year, 12, 31)
        path = DEFAULT_DATA_PATH / "cache" / "nse_holidays" / f"{year}.json"
        if path.exists():
            current = _read_holidays(path)
            if current.start != first or current.end != last:
                raise ValueError(f"Incomplete cached NSE cash holidays for {year}")
        elif refresh_missing_years:
            current = NseCashHolidays(first, last, _fetch_nse_cash_holidays(year))
            _cache_year(path, current)
        else:
            break
        coverage = NseCashHolidays(
            coverage.start, current.end, coverage.holidays | current.holidays
        )
    return coverage
