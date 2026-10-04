"""Read-only forward audit; report inputs transcribed from Gmail AX tables."""

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from brain_api.core.inference_utils import compute_week_boundaries
from brain_api.core.patchtst.inference import build_inference_features, run_inference
from brain_api.storage.patchtst.local import PatchTSTModelStorage

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
torch.set_num_threads(4)
prices = pickle.loads((OUT / "prices.pkl").read_bytes())
art = PatchTSTModelStorage(ROOT / "brain_api/data").load_current_artifacts()
assert art.version == "v2026-08-21-0af698826abd"
DATA = [
    (
        "2026-08-24",
        6,
        "PSX:17.82 VLO:13.10 CSLLY:11.16 SSDOY:9.07 CRL:8.91 ZBRA:7.66 STX:5.70 PANW:5.30 EXPE:3.92 RCRUY:3.58 PINS:3.50 WDAY:3.21 OKTA:2.90 AXON:2.62 TEAM:1.55",
        "CRL:4.48 ZBRA:4.30 CSLLY:4.26 WDAY:3.47 STX:3.23 VLO:3.14 OKTA:3.09 AXON:3.06 PANW:2.99 RCRUY:2.80 PINS:2.75 TEAM:2.73 PSX:2.71 DASH:2.48 ILMN:2.42 EXPE:2.40 TGT:2.36 BBY:2.34 SSMXY:2.28 SSDOY:2.27 GPC:2.20 REGN:2.17 ICLR:2.16 SMCI:2.16 HPE:2.15 COO:2.15 TMO:2.14 DNPLY:2.13 NDEKY:2.11 LH:2.09",
    ),
    (
        "2026-08-31",
        6,
        "INCY:16.71 ONC:14.03 CSLLY:10.41 CRL:8.71 SSDOY:8.51 PANW:7.45 ZBRA:5.82 DASH:5.18 RCRUY:5.04 EXPE:4.42 MMYT:3.72 AXON:3.10 ATEYY:2.84 TEAM:2.10 SMCI:1.95",
        "CRL:5.30 CSLLY:4.70 ATEYY:4.53 DASH:4.01 SMCI:3.98 EXPE:3.88 ZBRA:3.82 INCY:3.79 TEAM:3.57 MMYT:3.46 AXON:3.43 SSDOY:3.37 RCRUY:3.35 ONC:3.33 ICLR:3.31 WPM:3.31 CVNA:3.26 CRWD:3.26 IMPUY:3.17 PANW:3.14 IBM:3.12 ILMN:3.07 RVTY:3.05 OKTA:3.05 NEM:3.02 GPC:2.86 TGT:2.84 GFI:2.81 PSX:2.78 TMO:2.74",
    ),
    (
        "2026-09-07",
        8,
        "CTVA:26.91 VLO:15.42 ONC:10.72 CSLLY:8.74 CRL:5.90 ILMN:5.28 PANW:4.04 CTSH:3.90 ACN:3.90 ATEYY:3.62 ZBRA:3.38 GDDY:2.97 DOCU:2.43 OKTA:1.83 TEAM:0.97",
        "CSLLY:4.69 CRL:4.36 ACN:4.11 CTSH:3.83 ZBRA:3.67 TEAM:3.58 GDDY:3.48 ILMN:3.31 PANW:3.30 CTVA:3.23 ONC:3.13 OKTA:3.06 DOCU:3.06 VLO:3.02 PSX:2.94 WDAY:2.85 GFI:2.64 CF:2.60 ATEYY:2.56 CRWD:2.53 DASH:2.48 MMYT:2.48 RVTY:2.48 COP:2.47 RBLX:2.46 CRM:2.45 REGN:2.43 MRK:2.43 LH:2.31 TMO:2.29",
    ),
    (
        "2026-09-14",
        7,
        "PSX:19.15 VLO:13.81 EXPGY:10.73 CSLLY:10.21 CRL:8.52 ILMN:7.31 ZBRA:7.06 ADSK:5.00 IBM:4.37 DELL:3.21 CRM:2.64 CRWD:2.59 DOCU:2.45 OKTA:1.82 TEAM:1.14",
        "TEAM:5.83 ADSK:5.03 IBM:4.52 DELL:4.27 CSLLY:4.12 ILMN:4.09 PSX:3.98 DOCU:3.85 CRWD:3.76 CRL:3.75 EXPGY:3.71 VLO:3.67 CRM:3.67 MSTR:3.65 ZBRA:3.50 OKTA:3.42 HPQ:3.41 SSDOY:3.34 RCRUY:3.24 NOW:3.07 BZ:2.89 MMYT:2.83 NTAP:2.71 TMO:2.71 IT:2.62 ADBE:2.60 CF:2.60 ONC:2.60 NCLTY:2.59 GDDY:2.52",
    ),
    (
        "2026-09-21",
        6,
        "PSX:15.13 CSLLY:14.70 VLO:11.13 EXPGY:10.63 SSDOY:9.14 NTAP:7.89 NCLTY:6.05 CTSH:4.57 ACN:4.46 IBM:3.76 MSTR:3.03 CRM:2.84 DELL:2.75 DOCU:2.63 TEAM:1.28",
        "IBM:5.78 TEAM:5.66 PSX:4.64 VLO:4.30 CSLLY:4.13 NTAP:3.86 ACN:3.76 CTSH:3.75 SSDOY:3.51 NCLTY:3.48 MSTR:3.38 DELL:3.38 NTOIY:3.34 HPQ:3.30 DOCU:3.28 HPE:3.19 EXPGY:3.00 GFI:2.91 WDAY:2.90 CRM:2.86 SMCI:2.84 VGNT:2.79 IMPUY:2.79 IT:2.75 SWKS:2.70 RCRUY:2.63 SSMXY:2.61 NOW:2.57 CF:2.56 SOUHY:2.50",
    ),
    (
        "2026-09-28",
        4,
        "COP:21.00 CSLLY:11.04 PSX:10.42 NCLTY:8.34 ILMN:7.75 VLO:7.49 SSDOY:6.68 NTAP:5.30 HPQ:4.98 CTSH:4.20 IBM:4.17 ACN:3.49 MSTR:1.96 TEAM:1.64 SMCI:1.55",
        "IBM:4.77 TEAM:4.70 CSLLY:4.52 PSX:4.23 HPQ:4.07 SMCI:3.90 NTAP:3.73 ILMN:3.69 VLO:3.58 SSDOY:3.31 COP:3.30 NCLTY:3.26 UI:3.20 CTSH:3.19 NTOIY:3.08 HPE:3.07 SCCO:2.99 ACN:2.97 RVTY:2.97 MSTR:2.96 WPM:2.88 STLD:2.84 DELL:2.82 IMPUY:2.80 SU:2.77 TMO:2.76 NEM:2.70 DOCU:2.69 FCX:2.66 IT:2.60",
    ),
]


def parse(s):
    return {a: float(b) for a, b in (v.split(":") for v in s.split())}


def close_return(sym, monday, end=None, n=5):
    df = prices[sym]
    before = df[df.index.date < monday]
    after = df[df.index.date >= monday]
    if end is not None:
        after = after[after.index.date <= end]
    if end is None:
        after = after.iloc[:n]
    if before.empty or after.empty:
        return np.nan
    return float((after.close.iloc[-1] / before.close.iloc[-1] - 1) * 100)


rows = []
saved = []
previous = None
for d, new, w, sc in DATA:
    monday = date.fromisoformat(d)
    weights = parse(w)
    email_scores = parse(sc)
    week = compute_week_boundaries(monday)
    # August 31 email scores reproduce only when Friday August 28 is excluded.
    # This is a diagnostic reconstruction, not a claim that original inputs were saved.
    feature_cutoff = date(2026, 8, 28) if d == "2026-08-31" else week.target_week_start
    feats = [
        build_inference_features(s, p, art.config, feature_cutoff)
        for s, p in prices.items()
        if s not in {"SPY", "QQQ"}
    ]
    preds = run_inference(art.model, art.feature_scaler, feats, week, art.config)
    scores = {
        p.symbol: p.predicted_weekly_return_pct
        for p in preds
        if p.predicted_weekly_return_pct is not None
        and np.isfinite(p.predicted_weekly_return_pct)
    }
    actual = {s: close_return(s, monday) for s in scores}
    friday = monday + timedelta(days=4)
    week_actual = {s: close_return(s, monday, friday) for s in scores}
    selected_actual = {s: week_actual[s] for s in weights}
    top15 = sorted(scores, key=lambda s: (-scores[s], s))[:15]
    errs = [abs(scores.get(s, np.nan) - v) for s, v in email_scores.items()]

    def corr(a, b):
        pairs = [
            (a[s], b[s])
            for s in a.keys() & b.keys()
            if np.isfinite(a[s]) and np.isfinite(b[s])
        ]
        return float(spearmanr(np.array(pairs)[:, 0], np.array(pairs)[:, 1]).statistic)

    mean60 = {
        f.symbol: float(np.expm1(f.features.mean() * 5) * 100)
        for f in feats
        if f.features is not None
    }
    vol60 = {f.symbol: float(f.features.std()) for f in feats if f.features is not None}
    rule_ok = None
    if previous is None:
        previous = "GPC SSDOY PANW NTAP CRL TECH ZBRA RCRUY EXPE HPE MMYT WDAY AXON ATEYY TEAM".split()
    if previous is not None:
        # Displayed email scores lack full precision but ties cannot affect these selections.
        kept = [
            s
            for s in previous
            if s in email_scores and list(email_scores).index(s) < 20
        ]
        expected = (
            kept + [s for s in list(email_scores) if s not in kept][: 15 - len(kept)]
        )
        rule_ok = set(expected) == set(weights)
    row = {
        "week": d,
        "new_names": new,
        "largest_weight_pct": max(weights.values()),
        "largest_symbol": max(weights, key=weights.get),
        "email_rule_ok": rule_ok,
        "recomputed_valid_scores": len(scores),
        "max_email_score_error_pp": max(errs),
        "median_email_score_error_pp": float(np.nanmedian(errs)),
        "rank_ic_next5": corr(scores, actual),
        "rank_ic_calendar_week": corr(scores, week_actual),
        "corr_forecast_mean60": corr(scores, mean60),
        "corr_forecast_vol60": corr(scores, vol60),
        "model_top15_mean_next5_pct": float(np.nanmean([actual[s] for s in top15])),
        "selected_equal_calendar_return_pct": float(
            np.mean(list(selected_actual.values()))
        ),
        "selected_hrp_calendar_return_pct": sum(
            weights[s] / 100 * selected_actual[s] for s in weights
        ),
        "selected_forecast_weighted_pct": sum(
            weights[s] / 100 * email_scores[s] for s in weights
        ),
        "SPY_calendar_return_pct": close_return("SPY", monday, friday),
        "QQQ_calendar_return_pct": close_return("QQQ", monday, friday),
        "universe_equal_calendar_return_pct": float(
            np.nanmean(list(week_actual.values()))
        ),
    }
    row["feature_cutoff"] = str(feature_cutoff)
    row["largest_contributor_pp"] = max(
        (weights[s] / 100 * selected_actual[s], s) for s in weights
    )
    row["worst_contributor_pp"] = min(
        (weights[s] / 100 * selected_actual[s], s) for s in weights
    )
    rows.append(row)
    previous = list(weights)
    saved.append(
        {
            "week": d,
            "weights": weights,
            "email_top30_scores": email_scores,
            "selected_actual_calendar_week": selected_actual,
            "full_recomputed_scores": scores,
            "full_actual_next5": actual,
        }
    )
print(pd.DataFrame(rows).round(4).to_string(index=False))
(OUT / "analysis.json").write_text(
    json.dumps(
        {
            "model": art.version,
            "rows": rows,
            "weeks": saved,
            "limitations": [
                "Prices freshly downloaded, not original run snapshots. Email scores rounded to 2 decimals.",
                "Calendar-week returns are prior completed close to Friday close; diagnostic portfolio, not broker NAV or actual execution PnL.",
                "Labor Day week next5 horizon extends to September 14.",
            ],
        },
        indent=2,
    )
)
