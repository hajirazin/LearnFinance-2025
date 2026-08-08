"""Build AV-shaped statement payloads from SEC CompanyFacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests

from brain_api.core.fundamentals.sec_rate_limit import wait_for_sec_slot

SAC_INCOME_FIELDS = (
    "totalRevenue",
    "grossProfit",
    "operatingIncome",
    "netIncome",
)
SAC_BALANCE_FIELDS = (
    "totalCurrentAssets",
    "totalCurrentLiabilities",
    "shortLongTermDebtTotal",
    "totalShareholderEquity",
)

TAG_CHAINS: dict[str, tuple[str, ...]] = {
    "totalRevenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ),
    "grossProfit": ("GrossProfit",),
    "operatingIncome": ("OperatingIncomeLoss",),
    "netIncome": ("NetIncomeLoss",),
    "totalCurrentAssets": ("AssetsCurrent",),
    "totalCurrentLiabilities": ("LiabilitiesCurrent",),
    # Debt uses first-match via resolve_debt_points — chain listed for docs only.
    "shortLongTermDebtTotal": (
        "DebtLongtermAndShorttermCombinedAmount",
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "CommercialPaper",
        "NotesPayableCurrent",
        "ConvertibleDebtCurrent",
        "LongTermDebtNoncurrent",
        "ConvertibleDebtNoncurrent",
        "LongTermNotesPayable",
        "OtherLongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
    ),
    "totalShareholderEquity": ("StockholdersEquity",),
}

DEBT_CURRENT_TAGS = (
    "DebtCurrent",
    "ShortTermBorrowings",
    "LongTermDebtCurrent",
    "CommercialPaper",
    "NotesPayableCurrent",
    "ConvertibleDebtCurrent",
)
DEBT_NONCURRENT_TAGS = (
    "LongTermDebtNoncurrent",
    "ConvertibleDebtNoncurrent",
    "LongTermNotesPayable",
    "OtherLongTermDebtNoncurrent",
)
GROSS_PROFIT_COST_TAGS = (
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    "CostOfServicesExcludingDepreciationDepletionAndAmortization",
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
)

FLOW_FIELDS = frozenset(SAC_INCOME_FIELDS)
INSTANT_FIELDS = frozenset(SAC_BALANCE_FIELDS)
PERIODIC_FORMS = frozenset(
    {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F", "20-F/A", "40-F/A"}
)


class SECStatementError(RuntimeError):
    """Raised when CompanyFacts cannot supply required SAC fields."""


@dataclass(frozen=True)
class _FactPoint:
    end: str
    start: str | None
    value: float
    filed: str
    form: str
    accession: str
    fy: int | None
    fp: str | None


def _parse_day(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _duration_days(start: str | None, end: str) -> int | None:
    if not start:
        return None
    return (_parse_day(end) - _parse_day(start)).days


class CompanyFactsClient:
    """Thin SEC CompanyFacts client."""

    _FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(
        self,
        user_agent: str,
        timeout_seconds: float = 60.0,
    ):
        if not user_agent.strip():
            raise ValueError("SEC user agent must identify the requesting application")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def fetch_companyfacts(self, cik: str) -> dict[str, Any]:
        """Download CompanyFacts JSON for a CIK."""
        wait_for_sec_slot()
        url = self._FACTS_URL.format(cik=cik)
        response = requests.get(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise SECStatementError(f"CompanyFacts returned non-object for CIK {cik}")
        return payload


def _extract_tag_points(facts: dict[str, Any], tag: str) -> list[_FactPoint]:
    us_gaap = ((facts.get("facts") or {}).get("us-gaap") or {}).get(tag)
    if not isinstance(us_gaap, dict):
        return []
    units = us_gaap.get("units") or {}
    series = units.get("USD")
    if not isinstance(series, list):
        return []
    points: list[_FactPoint] = []
    for row in series:
        if not isinstance(row, dict):
            continue
        form = str(row.get("form") or "")
        if form not in PERIODIC_FORMS:
            continue
        end = row.get("end")
        val = row.get("val")
        filed = row.get("filed")
        accn = row.get("accn")
        if end is None or val is None or filed is None or accn is None:
            continue
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        fy_raw = row.get("fy")
        fy = int(fy_raw) if fy_raw is not None else None
        fp = str(row["fp"]) if row.get("fp") is not None else None
        points.append(
            _FactPoint(
                end=str(end)[:10],
                start=str(row["start"])[:10] if row.get("start") else None,
                value=value,
                filed=str(filed)[:10],
                form=form,
                accession=str(accn),
                fy=fy,
                fp=fp,
            )
        )
    return points


def _pick_instant_points(points: list[_FactPoint]) -> dict[str, _FactPoint]:
    """Latest filed fact per period end for balance-sheet (instant) concepts."""
    by_end: dict[str, _FactPoint] = {}
    for point in sorted(points, key=lambda p: (p.end, p.filed, p.accession)):
        by_end[point.end] = point
    return by_end


def _standalone_flow_value(
    points_for_fy: list[_FactPoint],
    *,
    fp: str,
) -> _FactPoint | None:
    """Return standalone quarter value for fp within one fiscal year."""
    ordered = sorted(points_for_fy, key=lambda p: (p.end, p.filed, p.accession))
    # Prefer short-duration (~quarter) facts for this fp
    matching = [p for p in ordered if (p.fp or "").upper() == fp.upper()]
    short = [
        p
        for p in matching
        if (d := _duration_days(p.start, p.end)) is not None and 60 <= d <= 120
    ]
    if short:
        return short[-1]
    if fp.upper() == "Q1":
        # Q1 YTD == standalone
        if matching:
            return matching[-1]
        return None

    # YTD differencing within fy using fp order
    order = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}
    target = order.get(fp.upper())
    if target is None:
        return None
    by_fp: dict[str, _FactPoint] = {}
    for point in ordered:
        key = (point.fp or "").upper()
        if key in order:
            by_fp[key] = point
    if fp.upper() not in by_fp:
        return None
    current = by_fp[fp.upper()]
    if fp.upper() == "FY":
        # Q4 = FY - Q3 YTD if Q3 present, else FY if duration ~365
        q3 = by_fp.get("Q3")
        if q3 is not None:
            return _FactPoint(
                end=current.end,
                start=q3.end,
                value=current.value - q3.value,
                filed=current.filed,
                form=current.form,
                accession=current.accession,
                fy=current.fy,
                fp="Q4",
            )
        return current

    prev_fp = {2: "Q1", 3: "Q2"}.get(target)
    if prev_fp is None or prev_fp not in by_fp:
        return None
    prev = by_fp[prev_fp]
    return _FactPoint(
        end=current.end,
        start=prev.end,
        value=current.value - prev.value,
        filed=max(current.filed, prev.filed),
        form=current.form,
        accession=current.accession,
        fy=current.fy,
        fp=fp.upper(),
    )


def _pick_flow_quarterly(points: list[_FactPoint]) -> dict[str, _FactPoint]:
    """Map fiscalDateEnding -> standalone quarterly flow fact."""
    by_fy: dict[int, list[_FactPoint]] = defaultdict(list)
    for point in points:
        if point.fy is None:
            continue
        by_fy[point.fy].append(point)

    result: dict[str, _FactPoint] = {}
    for _fy, group in by_fy.items():
        for fp in ("Q1", "Q2", "Q3", "FY"):
            standalone = _standalone_flow_value(group, fp=fp)
            if standalone is None:
                continue
            # Represent FY-derived Q4 under its end date; skip pure FY annual row
            # for quarterlyReports (loader uses quarterly). Store Q1-Q3 and Q4.
            if fp == "FY" and (standalone.fp or "").upper() != "Q4":
                # annual only — keep for annualReports separately via end date
                result[f"annual:{standalone.end}"] = standalone
                continue
            result[standalone.end] = standalone
    return result


def _sum_tag_values(
    tags: dict[str, _FactPoint], names: tuple[str, ...]
) -> tuple[float, _FactPoint] | None:
    """Sum present tags from ``names``; return (total, provenance) or None."""
    present = [tags[n] for n in names if n in tags]
    if not present:
        return None
    total = sum(p.value for p in present)
    provenance = max(present, key=lambda p: (p.filed, p.accession))
    return total, provenance


def _compose_current_leg(
    tags: dict[str, _FactPoint],
) -> tuple[float, _FactPoint] | None:
    """Resolve current-leg debt value (first-match within expanded current set)."""
    if "DebtCurrent" in tags:
        return tags["DebtCurrent"].value, tags["DebtCurrent"]
    has_st = "ShortTermBorrowings" in tags
    has_ltd_current = "LongTermDebtCurrent" in tags
    if has_st and has_ltd_current:
        st = tags["ShortTermBorrowings"]
        ltdc = tags["LongTermDebtCurrent"]
        provenance = ltdc if ltdc.filed > st.filed else st
        return st.value + ltdc.value, provenance
    if has_ltd_current:
        return tags["LongTermDebtCurrent"].value, tags["LongTermDebtCurrent"]
    if has_st:
        return tags["ShortTermBorrowings"].value, tags["ShortTermBorrowings"]
    # Remaining expanded current tags (CommercialPaper, NotesPayable, Convertible)
    extras = (
        "CommercialPaper",
        "NotesPayableCurrent",
        "ConvertibleDebtCurrent",
    )
    return _sum_tag_values(tags, extras)


def _compose_noncurrent_leg(
    tags: dict[str, _FactPoint],
) -> tuple[float, _FactPoint] | None:
    """Resolve noncurrent-leg debt (prefer LongTermDebtNoncurrent, else expanded)."""
    if "LongTermDebtNoncurrent" in tags:
        return tags["LongTermDebtNoncurrent"].value, tags["LongTermDebtNoncurrent"]
    return _sum_tag_values(
        tags,
        (
            "ConvertibleDebtNoncurrent",
            "LongTermNotesPayable",
            "OtherLongTermDebtNoncurrent",
        ),
    )


def _compose_debt_point(
    *,
    end: str,
    tags: dict[str, _FactPoint],
) -> _FactPoint:
    """Resolve total debt for one period end (first-match).

    B is entered only when both current and noncurrent legs resolve; incomplete B
    falls through to C then D (does not hard-fail). Noncurrent-only uses noncurrent
    as total. Unresolvable ends raise so the caller can period-skip.
    """
    if "DebtLongtermAndShorttermCombinedAmount" in tags:
        return tags["DebtLongtermAndShorttermCombinedAmount"]

    current = _compose_current_leg(tags)
    noncurrent = _compose_noncurrent_leg(tags)

    if current is not None and noncurrent is not None:
        current_val, current_pt = current
        noncurrent_val, noncurrent_pt = noncurrent
        provenance = (
            current_pt if current_pt.filed >= noncurrent_pt.filed else noncurrent_pt
        )
        return _FactPoint(
            end=end,
            start=None,
            value=current_val + noncurrent_val,
            filed=provenance.filed,
            form=provenance.form,
            accession=provenance.accession,
            fy=provenance.fy,
            fp=provenance.fp,
        )

    if current is None and noncurrent is not None:
        # Implicit current = 0
        _val, provenance = noncurrent
        return _FactPoint(
            end=end,
            start=None,
            value=_val,
            filed=provenance.filed,
            form=provenance.form,
            accession=provenance.accession,
            fy=provenance.fy,
            fp=provenance.fp,
        )

    # Incomplete B (current without noncurrent) or no legs → try C then D
    if "LongTermDebtAndCapitalLeaseObligations" in tags:
        return tags["LongTermDebtAndCapitalLeaseObligations"]
    if "LongTermDebt" in tags:
        return tags["LongTermDebt"]
    raise SECStatementError(f"no debt tags for end {end}")


def resolve_debt_points(facts: dict[str, Any]) -> dict[str, _FactPoint]:
    """First-match total-debt mapping; skip unresolvable period ends.

    Empty dict means zero-debt mode (no interest-bearing tags for the CIK) —
    callers fill ``0`` on otherwise-complete balance rows.
    """
    tag_names = (
        "DebtLongtermAndShorttermCombinedAmount",
        *DEBT_CURRENT_TAGS,
        *DEBT_NONCURRENT_TAGS,
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
    )
    by_tag: dict[str, dict[str, _FactPoint]] = {}
    ends: set[str] = set()
    for tag in tag_names:
        points = _extract_tag_points(facts, tag)
        if not points:
            continue
        by_end = _pick_instant_points(points)
        by_tag[tag] = by_end
        ends |= set(by_end)

    if not ends:
        return {}

    resolved: dict[str, _FactPoint] = {}
    for end in ends:
        tags_here = {
            tag: by_end[end] for tag, by_end in by_tag.items() if end in by_end
        }
        try:
            resolved[end] = _compose_debt_point(end=end, tags=tags_here)
        except SECStatementError:
            continue
    return resolved


def _resolve_gross_profit_points(facts: dict[str, Any]) -> dict[str, _FactPoint]:
    """GrossProfit tag first; else revenue - first-match cost tag (YTD->quarter)."""
    direct: list[_FactPoint] = []
    for tag in TAG_CHAINS["grossProfit"]:
        direct.extend(_extract_tag_points(facts, tag))
    if direct:
        return _pick_flow_quarterly(direct)

    rev_merged: list[_FactPoint] = []
    for tag in TAG_CHAINS["totalRevenue"]:
        rev_merged.extend(_extract_tag_points(facts, tag))
    if not rev_merged:
        raise SECStatementError("No CompanyFacts USD points for field grossProfit")
    rev_by_end = _pick_flow_quarterly(rev_merged)

    cost_by_end: dict[str, _FactPoint] | None = None
    for cost_tag in GROSS_PROFIT_COST_TAGS:
        cost_pts = _extract_tag_points(facts, cost_tag)
        if not cost_pts:
            continue
        cost_by_end = _pick_flow_quarterly(cost_pts)
        if cost_by_end:
            break
    if not cost_by_end:
        raise SECStatementError("No CompanyFacts USD points for field grossProfit")

    derived: dict[str, _FactPoint] = {}
    for end, rev_pt in rev_by_end.items():
        if end.startswith("annual:"):
            cost_pt = cost_by_end.get(end)
            if cost_pt is None:
                continue
            derived[end] = _FactPoint(
                end=rev_pt.end,
                start=rev_pt.start,
                value=rev_pt.value - cost_pt.value,
                filed=max(rev_pt.filed, cost_pt.filed),
                form=rev_pt.form,
                accession=rev_pt.accession,
                fy=rev_pt.fy,
                fp=rev_pt.fp,
            )
            continue
        cost_pt = cost_by_end.get(end)
        if cost_pt is None:
            continue
        derived[end] = _FactPoint(
            end=end,
            start=rev_pt.start,
            value=rev_pt.value - cost_pt.value,
            filed=max(rev_pt.filed, cost_pt.filed),
            form=rev_pt.form,
            accession=rev_pt.accession,
            fy=rev_pt.fy,
            fp=rev_pt.fp,
        )
    if not derived:
        raise SECStatementError("No CompanyFacts USD points for field grossProfit")
    return derived


def _resolve_field_points(facts: dict[str, Any], field: str) -> dict[str, _FactPoint]:
    if field == "shortLongTermDebtTotal":
        return resolve_debt_points(facts)
    if field == "grossProfit":
        return _resolve_gross_profit_points(facts)
    merged: list[_FactPoint] = []
    for tag in TAG_CHAINS[field]:
        merged.extend(_extract_tag_points(facts, tag))
    if not merged:
        raise SECStatementError(f"No CompanyFacts USD points for field {field}")
    if field in INSTANT_FIELDS:
        return _pick_instant_points(merged)
    return _pick_flow_quarterly(merged)


def _period_report(
    end: str,
    values: dict[str, float],
    provenance: _FactPoint,
    *,
    cik: str,
    currency: str = "USD",
    field_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "fiscalDateEnding": end,
        "reportedCurrency": currency,
        "filingDate": provenance.filed,
        "accessionNumber": provenance.accession,
        "filingForm": provenance.form,
        "filingSource": (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{provenance.accession.replace('-', '')}"
        ),
    }
    report.update({k: str(v) for k, v in values.items()})
    if field_source:
        report["fieldSource"] = dict(field_source)
    return report


def build_statement_payloads_from_companyfacts(
    facts: dict[str, Any],
    *,
    symbol: str,
    cik: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (income_statement, balance_sheet) AV-shaped provider payloads."""
    income_points = {f: _resolve_field_points(facts, f) for f in SAC_INCOME_FIELDS}
    balance_points = {
        f: _resolve_field_points(facts, f)
        for f in SAC_BALANCE_FIELDS
        if f != "shortLongTermDebtTotal"
    }
    debt_points = resolve_debt_points(facts)
    zero_debt = not debt_points
    balance_points["shortLongTermDebtTotal"] = debt_points

    # Quarterly income: intersection of ends that look like quarters (not annual:)
    income_ends = set()
    for _field, by_end in income_points.items():
        income_ends |= {e for e in by_end if not e.startswith("annual:")}
    if not income_ends:
        raise SECStatementError(f"No quarterly income periods for {symbol}")

    quarterly_income: list[dict[str, Any]] = []
    gross_profit_derived = not _extract_tag_points(facts, "GrossProfit")
    for end in sorted(income_ends, reverse=True):
        values: dict[str, float] = {}
        provenance: _FactPoint | None = None
        field_source: dict[str, str] = {}
        missing = False
        for field in SAC_INCOME_FIELDS:
            point = income_points[field].get(end)
            if point is None:
                missing = True
                break
            values[field] = point.value
            if field == "grossProfit" and gross_profit_derived:
                field_source[field] = "sec_derived"
            else:
                field_source[field] = "sec_companyfacts"
            if provenance is None or point.filed > provenance.filed:
                provenance = point
        if missing or provenance is None:
            continue
        quarterly_income.append(
            _period_report(end, values, provenance, cik=cik, field_source=field_source)
        )

    if not quarterly_income:
        raise SECStatementError(
            f"Could not build complete quarterly income rows for {symbol}"
        )

    balance_ends: set[str] = set()
    for field, by_end in balance_points.items():
        if field == "shortLongTermDebtTotal" and zero_debt:
            continue
        balance_ends |= set(by_end)

    quarterly_balance: list[dict[str, Any]] = []
    for end in sorted(balance_ends, reverse=True):
        values = {}
        provenance = None
        field_source = {}
        missing = False
        for field in SAC_BALANCE_FIELDS:
            if field == "shortLongTermDebtTotal" and zero_debt:
                values[field] = 0.0
                field_source[field] = "sec_zero_debt"
                continue
            point = balance_points[field].get(end)
            if point is None:
                missing = True
                break
            values[field] = point.value
            field_source[field] = "sec_companyfacts"
            if provenance is None or point.filed > provenance.filed:
                provenance = point
        if missing or provenance is None:
            continue
        if zero_debt:
            # provenance from another balance field already set
            pass
        quarterly_balance.append(
            _period_report(end, values, provenance, cik=cik, field_source=field_source)
        )

    if not quarterly_balance:
        raise SECStatementError(
            f"Could not build complete quarterly balance rows for {symbol}"
        )

    # Annual income from annual: keys when present
    annual_income: list[dict[str, Any]] = []
    annual_ends = set()
    for by_end in income_points.values():
        annual_ends |= {e[7:] for e in by_end if e.startswith("annual:")}
    for end in sorted(annual_ends, reverse=True):
        values = {}
        provenance = None
        missing = False
        for field in SAC_INCOME_FIELDS:
            point = income_points[field].get(f"annual:{end}")
            if point is None:
                missing = True
                break
            values[field] = point.value
            provenance = point
        if missing or provenance is None:
            continue
        annual_income.append(_period_report(end, values, provenance, cik=cik))

    income_payload = {
        "symbol": symbol.upper(),
        "annualReports": annual_income,
        "quarterlyReports": quarterly_income,
        "provider": "sec_companyfacts",
    }
    balance_payload = {
        "symbol": symbol.upper(),
        "annualReports": [],
        "quarterlyReports": quarterly_balance,
        "provider": "sec_companyfacts",
    }
    return income_payload, balance_payload
