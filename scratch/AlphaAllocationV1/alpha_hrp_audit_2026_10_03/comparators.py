"""Gmail target transcriptions; common price/horizon comparison only, not broker NAV."""

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
prices = pickle.loads((OUT / "prices.pkl").read_bytes())
DHRP = [
    (
        "2026-08-24",
        "CHT:23.05 LIN:10.84 GWW:8.88 SGAPY:8.29 EQIX:8.15 VLTO:7.43 T:6.32 HSHCY:4.93 NVZMY:4.80 KAOOY:4.53 CTVA:4.53 ALPMY:3.23 EBAY:2.67 CCO:2.23 MNST:0.13",
    ),
    (
        "2026-08-31",
        "CHT:25.12 SGAPY:9.83 GWW:9.74 EQIX:8.97 LIN:8.17 T:6.99 VLTO:6.08 HSHCY:5.16 KAOOY:4.55 NVZMY:4.45 CTVA:3.65 ALPMY:3.27 EBAY:2.16 CCO:1.73 MNST:0.14",
    ),
    (
        "2026-09-07",
        "CHT:23.36 SGAPY:10.06 GWW:8.46 EQIX:7.44 LIN:7.20 MNST:6.39 T:5.86 VLTO:5.68 HSHCY:4.97 NVZMY:4.85 KAOOY:4.23 CTVA:3.48 ALPMY:3.00 CCO:2.62 EBAY:2.38",
    ),
    (
        "2026-09-14",
        "CHT:23.31 SGAPY:10.06 GWW:8.18 EQIX:7.81 LIN:7.09 MNST:6.45 T:5.88 VLTO:5.69 KAOOY:4.93 HSHCY:4.38 NVZMY:4.24 ALPMY:3.47 CTVA:3.44 CCO:2.69 EBAY:2.36",
    ),
    (
        "2026-09-21",
        "CHT:23.35 SGAPY:9.81 GWW:8.28 EQIX:7.73 LIN:6.85 MNST:6.51 VLTO:6.06 T:5.77 KAOOY:5.04 HSHCY:4.78 NVZMY:3.95 CTVA:3.35 ALPMY:3.21 CCO:2.81 EBAY:2.49",
    ),
    (
        "2026-09-28",
        "CHT:23.70 SGAPY:9.78 GWW:8.32 EQIX:7.47 LIN:6.80 MNST:6.50 T:5.80 VLTO:5.62 KAOOY:4.54 HSHCY:4.23 ALPMY:4.14 NVZMY:3.98 CTVA:3.31 CCO:2.95 EBAY:2.87",
    ),
]
SAC = [
    (
        "2026-08-24",
        "AMD:15.26 BBY:11.78 CDW:8.38 CTSH:12.53 DLTR:13.38 FTNT:2.27 FUJIY:3.68 GRMN:2.16 ILMN:2.10 MSFT:2.15 NTAP:2.31 PANW:2.19 SAP:7.51 WDC:9.04 ZBRA:2.17 CASH:3.08",
    ),
    (
        "2026-09-08",
        "ATEYY:4.48 AXON:14.00 CRL:3.39 CSLLY:13.76 EXPE:2.52 IBM:3.31 ICLR:4.37 ILMN:2.03 IMPUY:3.80 INCY:8.28 MMYT:13.82 PANW:2.64 SSDOY:12.04 WPM:2.37 ZBRA:7.17 CASH:2.03",
    ),
    (
        "2026-09-14",
        "ATEYY:1.72 AXON:11.15 CRL:6.73 CSLLY:2.06 EXPE:10.35 IBM:11.99 ICLR:1.71 ILMN:5.78 IMPUY:8.27 INCY:3.85 MMYT:9.22 PANW:2.79 SSDOY:10.91 WPM:1.66 ZBRA:8.92 CASH:2.90",
    ),
    (
        "2026-09-21",
        "ATEYY:1.78 AXON:12.99 CRL:3.06 CSLLY:12.74 EXPE:3.64 IBM:6.03 ICLR:5.29 ILMN:1.79 IMPUY:4.55 INCY:12.56 MMYT:9.06 PANW:1.99 SSDOY:12.29 WPM:2.50 ZBRA:7.37 CASH:2.35",
    ),
    (
        "2026-09-28",
        "ATEYY:2.31 AXON:5.37 CRL:4.81 CSLLY:10.91 EXPE:13.13 IBM:4.10 ICLR:2.08 ILMN:1.78 IMPUY:10.81 INCY:7.64 MMYT:13.12 PANW:12.99 SSDOY:4.58 WPM:1.91 ZBRA:1.78 CASH:2.69",
    ),
]


def analyze(data):
    rows = []
    prev = None
    for d, w in data:
        weights = {s: float(v) for s, v in (p.split(":") for p in w.split())}
        monday = date.fromisoformat(d)
        fri = monday + timedelta(days=4 - monday.weekday())
        contrib = {}
        for s, v in weights.items():
            if s == "CASH":
                continue
            df = prices[s]
            b = df[df.index.date < monday]
            a = df[(df.index.date >= monday) & (df.index.date <= fri)]
            contrib[s] = v / 100 * (a.close.iloc[-1] / b.close.iloc[-1] - 1) * 100
        rows.append(
            {
                "week": d,
                "weights_pct": weights,
                "new_names": None if prev is None else len(set(weights) - set(prev)),
                "calendar_week_gross_return_pct": sum(contrib.values()),
            }
        )
        prev = weights
    return rows


out = {
    "double_hrp": analyze(DHRP),
    "sac_halal_filtered": analyze(SAC),
    "limitations": [
        "No August 31 SAC email found in this search, so SAC six-week cumulative return is not computed.",
        "Prices as of October 3; targets rounded in Gmail. Prior-close to Friday-close diagnostic excludes intraday executions, costs, idle cash and unfilled orders.",
        "SAC slate changed sharply on September 8 and its allocator model differs; not a controlled PatchTST ablation.",
    ],
}
(OUT / "comparator_analysis.json").write_text(json.dumps(out, indent=2))
print(
    [
        (r["week"], round(r["calendar_week_gross_return_pct"], 4), r["new_names"])
        for r in out["double_hrp"]
    ]
)
print(
    "DHRP compounded",
    float(
        np.prod(
            [1 + r["calendar_week_gross_return_pct"] / 100 for r in out["double_hrp"]]
        )
        * 100
        - 100
    ),
)
print(
    "SAC",
    [
        (r["week"], round(r["calendar_week_gross_return_pct"], 4))
        for r in out["sac_halal_filtered"]
    ],
)
