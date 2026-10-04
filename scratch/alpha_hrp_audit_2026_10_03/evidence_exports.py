"""Offline reconciliation, exports and controlled gross portfolio diagnostics."""

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from brain_api.core.hrp import compute_hrp_allocation
from brain_api.core.patchtst.inference import build_inference_features
from brain_api.core.sticky_selection import select_with_rank_band
from brain_api.storage.patchtst.local import PatchTSTModelStorage

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
torch.set_num_threads(4)
prices = pickle.loads((OUT / "full_price_evidence.pkl").read_bytes())
archive = json.loads((OUT / "full_order_archive.json").read_text())
order_rows = []
for account, a in archive["accounts"].items():
    for o in a["orders"]:
        row = {"account": account, **o}
        row["filled_notional_usd"] = float(o["filled_qty"] or 0) * float(
            o["filled_avg_price"] or 0
        )
        order_rows.append(row)
pd.DataFrame(order_rows).to_csv(OUT / "all_orders_2026.csv", index=False)
pd.concat(prices, names=["symbol", "date"]).to_csv(OUT / "all_adjusted_ohlcv.csv")
closes = pd.DataFrame({s: f.close for s, f in prices.items()})
closes.index.name = "date"
closes.to_csv(OUT / "all_adjusted_closes.csv")
nav = pd.read_csv(OUT / "broker_daily_nav.csv", index_col="date", parse_dates=True)
monthly = (
    pd.concat([nav[["hrp", "sac"]], closes[["SPY", "QQQ"]]], axis=1)
    .resample("ME")
    .last()
    .pct_change(fill_method=None)
    * 100
)
monthly = monthly.loc["2026-01-01":"2026-09-30"]
monthly.to_csv(OUT / "monthly_account_benchmark_returns.csv")


def calendar_return(s, monday):
    series = prices[s].close
    before = series.loc[series.index.date < monday]
    after = series.loc[
        (series.index.date >= monday)
        & (series.index.date <= monday + timedelta(days=4))
    ]
    return float(after.iloc[-1] / before.iloc[-1] - 1)


storage = PatchTSTModelStorage(ROOT / "brain_api/data")
pointer = storage.read_current_version()
samples = json.loads((OUT / "pre_migration_email_samples.json").read_text())["samples"]
pre_rows = []
for row in samples:
    monday = date.fromisoformat(row["week"])
    art = storage.load_version_artifacts(row["model"])
    features = [
        build_inference_features(s, f, art.config, monday)
        for s, f in prices.items()
        if s not in {"SPY", "QQQ", "SPUS", "HLAL"}
    ]
    good = [f for f in features if f.features is not None]
    with torch.no_grad():
        outputs = (
            art.model(
                past_values=torch.from_numpy(
                    np.array([f.features for f in good], dtype=np.float32)
                )
            )
            .prediction_outputs[:, :, 3]
            .sum(1)
            .numpy()
        )
    scores = {f.symbol: float(np.expm1(p) * 100) for f, p in zip(good, outputs)}
    errors = {
        s: scores[s] - v for s, v in row["email_top25_scores"].items() if s in scores
    }
    means = np.array([f.features[:, 3].mean() * 5 for f in good])
    ret = (
        sum(w / 100 * calendar_return(s, monday) for s, w in row["weights_pct"].items())
        * 100
    )
    friday = monday + timedelta(days=4)
    before_nav = nav.loc[nav.index.date < monday, "hrp"].dropna().iloc[-1]
    after_nav = (
        nav.loc[(nav.index.date >= monday) & (nav.index.date <= friday), "hrp"]
        .dropna()
        .iloc[-1]
    )
    fills = [
        o
        for o in archive["accounts"]["hrp"]["orders"]
        if o["filled_at"] and monday <= pd.Timestamp(o["filled_at"]).date() <= friday
    ]
    pre_rows.append(
        {
            "week": row["week"],
            "version": row["model"],
            "email_k_hold": 30,
            "reproduced_email_scores": len(errors),
            "email_score_count": len(row["email_top25_scores"]),
            "max_email_forecast_error_pp": max(abs(v) for v in errors.values()),
            "corr_model_prediction_mean60": float(spearmanr(outputs, means).statistic),
            "actual_nav_calendar_return_pct": float((after_nav / before_nav - 1) * 100),
            "email_weight_gross_calendar_return_pct": ret,
            "SPY_calendar_return_pct": calendar_return("SPY", monday) * 100,
            "QQQ_calendar_return_pct": calendar_return("QQQ", monday) * 100,
            "filled_orders": len(fills),
            "gross_filled_notional_usd": sum(
                float(o["filled_qty"]) * float(o["filled_avg_price"]) for o in fills
            ),
        }
    )

decomp = json.loads((OUT / "math_prediction_decomposition.json").read_text())
versions = sorted({r["version"] for r in decomp})
replay = []
initial = (
    "GPC SSDOY PANW NTAP CRL TECH ZBRA RCRUY EXPE HPE MMYT WDAY AXON ATEYY TEAM".split()
)
for version in versions:
    rows = sorted(
        [r for r in decomp if r["version"] == version], key=lambda r: r["week"]
    )
    for band in [20, 30]:
        previous = initial
        for row in rows:
            monday = date.fromisoformat(row["week"])
            scores = dict(zip(row["symbols"], row["predicted_weekly_logs"]))
            selected = select_with_rank_band(
                scores, previous, top_n=15, hold_threshold=band
            )
            frames = {
                s: prices[s].loc[prices[s].index.date < monday]
                for s in selected.selected
            }
            hrp = compute_hrp_allocation(frames, lookback_days=252, as_of_date=monday)
            weights = hrp.percentage_weights
            returns = {s: calendar_return(s, monday) for s in weights}
            returns_pp = sum(weights[s] / 100 * returns[s] for s in weights) * 100
            replay.append(
                {
                    "version": version,
                    "band": band,
                    "week": row["week"],
                    "new_names": selected.fillers_count,
                    "largest_weight_pct": max(weights.values()),
                    "gross_hrp_calendar_return_pct": returns_pp,
                    "selected": selected.selected,
                    "weights_pct": weights,
                    "retained_outside_entry": [
                        s
                        for s in selected.selected
                        if sorted(scores, key=lambda s: (-scores[s], s)).index(s) >= 15
                    ],
                }
            )
            previous = selected.selected
summaries = []
for version in versions:
    for band in [20, 30]:
        rows = [r for r in replay if r["version"] == version and r["band"] == band]
        summaries.append(
            {
                "version": version,
                "band": band,
                "gross_compounded_hrp_calendar_pct": float(
                    (
                        np.prod(
                            [1 + r["gross_hrp_calendar_return_pct"] / 100 for r in rows]
                        )
                        - 1
                    )
                    * 100
                ),
                "new_names_total": sum(r["new_names"] for r in rows),
            }
        )
assert storage.read_current_version() == pointer
result = {
    "pre_migration_samples": pre_rows,
    "replay_summary": summaries,
    "replay_weeks": replay,
    "orders_by_account": {a: len(v["orders"]) for a, v in archive["accounts"].items()},
    "price_series": len(prices),
    "current_pointer_unchanged": True,
    "caveats": [
        "Replays use a frozen September roster, common OHLCV-eligible panel and one observed August17 carry-set.",
        "Gross close-to-close hypothetical portfolios exclude market fills, delays, costs and corporate-action reconciliation; they are not actual account NAV.",
        "Every decision uses only prices before that date. New histories are retrospective Yahoo adjusted evidence.",
        "Pre-migration email samples were deliberately spaced, not statistically random.",
    ],
}
(OUT / "evidence_reconciliation.json").write_text(json.dumps(result, indent=2))
pd.DataFrame(pre_rows).to_csv(OUT / "pre_migration_weekly_samples.csv", index=False)
pd.DataFrame(summaries).to_csv(OUT / "checkpoint_hrp_replay_summary.csv", index=False)
print(json.dumps({"pre": pre_rows, "replays": summaries}, indent=2))
