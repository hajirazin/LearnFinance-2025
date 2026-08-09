#!/usr/bin/env python3
"""E10: Non-ML factor controls + IC significance (Claude follow-up).

Same 12-name / 2026-week slate and IC / topK→HRP / Sharpe pipeline as E9.

Controls (zero ML):
  1) mom_4w  — close return over prior 20 trading days (as-of before week)
  2) mom_1w  — close return over prior 5 trading days
  3) rev_1w  — negative of mom_1w (1-week reversal)

Also: IC mean ± SE and t-stat for each factor and for E9 ListNet scores
(reloaded from exp_e9_listnet/results/weekly_scores.json).

HRP-252 is pure covariance risk-parity (no return forecast) — stated in summary.

Artifacts: scratch/.../exp_e10_factor_controls/
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from brain_api.core.hrp import compute_hrp_allocation

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_e10_factor_controls"
RESULTS_DIR = OUT_DIR / "results"
PRICE_CACHE = BASE_DIR / "exp_e123" / "cache" / "prices.pkl"
E9_SCORES = BASE_DIR / "exp_e9_listnet" / "results" / "weekly_scores.json"
E8_DIR = BASE_DIR / "exp_e8_wd_sweep" / "results"

EVAL_YEAR = 2026
TOP_K = 6
HRP_LOOKBACK = 252


def _iter_weeks(prices, symbols):
    ref = prices[symbols[0]]
    days = list(ref.index[ref.index.year == EVAL_YEAR])
    by_period: dict = {}
    for ts in days:
        by_period.setdefault(ts.to_period("W"), []).append(ts)
    weeks = []
    for _p, sessions in sorted(by_period.items(), key=lambda kv: kv[1][0]):
        if len(sessions) < 5:
            continue
        five = sessions[:5]
        ok = all(
            all(d in prices[s].index for d in five)
            and len(prices[s][prices[s].index < five[0]]) >= 25
            for s in symbols
        )
        if ok:
            weeks.append(five)
    return weeks


def _asset_week_ret(pdf, five):
    prior = pdf[pdf.index < five[0]]["close"]
    start = float(prior.iloc[-1])
    end = float(pdf.loc[five[-1], "close"])
    return end / start - 1.0


def _mom(pdf, as_of, lookback_bars: int) -> float | None:
    hist = pdf[pdf.index < as_of]["close"].dropna()
    if len(hist) < lookback_bars + 1:
        return None
    return float(hist.iloc[-1] / hist.iloc[-(lookback_bars + 1)] - 1.0)


def _truncate_prices(prices, as_of):
    return {
        s: df[df.index < as_of] for s, df in prices.items() if len(df[df.index < as_of])
    }


def _hrp_weights(prices_asof, symbols, as_of: date):
    subset = {s: prices_asof[s] for s in symbols if s in prices_asof}
    res = compute_hrp_allocation(
        subset, lookback_days=HRP_LOOKBACK, min_data_days=60, as_of_date=as_of
    )
    w = {s: v / 100.0 for s, v in res.percentage_weights.items()}
    if not w:
        return {s: 1.0 / len(symbols) for s in symbols}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def _metrics(weekly_rets: list[float]) -> dict:
    r = np.array(weekly_rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(52) if sd > 1e-12 else float("nan")
    wealth = np.cumprod(1.0 + r)
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "cum_return_pct": cum * 100,
        "sharpe_ann_approx": float(sharpe),
        "hit_rate": float(np.mean(r > 0)),
        "max_drawdown_pct": float(np.min(dd) * 100),
        "n_weeks": len(r),
    }


def _ic_stats(weekly_ics: list[float]) -> dict:
    x = np.array(weekly_ics, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    # SE of mean IC across weeks
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = mean / se if se > 1e-15 else float("nan")
    return {
        "n_weeks": n,
        "ic_mean": mean,
        "ic_std": float(np.std(x, ddof=1)) if n > 1 else float("nan"),
        "ic_se": se,
        "ic_tstat": float(t),
        "ic_mean_pm_se": f"{mean:+.4f} ± {se:.4f}",
    }


def _eval_factor(name: str, scores_by_week: dict, prices, symbols, weeks) -> dict:
    """scores_by_week: week_start_iso -> {symbol: score}"""
    ics = []
    top3_ov = []
    arm_rets = {
        "HRP-252": [],
        f"{name}_top6_HRP": [],
        f"{name}_top6_EW": [],
        "EW-12": [],
    }
    rows = []

    for five in weeks:
        ws = str(five[0].date())
        scores = scores_by_week.get(ws)
        if not scores or len(scores) < TOP_K:
            continue
        asset_rets = {s: _asset_week_ret(prices[s], five) for s in scores}
        sc = [scores[s] for s in scores]
        ar = [asset_rets[s] for s in scores]
        rho, _ = spearmanr(sc, ar)
        if np.isfinite(rho):
            ics.append(float(rho))

        order_p = sorted(scores, key=scores.get, reverse=True)
        order_a = sorted(asset_rets, key=asset_rets.get, reverse=True)
        top3_p, top3_a = set(order_p[:3]), set(order_a[:3])
        top3_ov.append(len(top3_p & top3_a) / 3.0)

        for s in scores:
            rows.append(
                {
                    "week_start": ws,
                    "symbol": s,
                    "score": scores[s],
                    "actual_week_frac": asset_rets[s],
                }
            )

        top = order_p[:TOP_K]
        prices_asof = _truncate_prices(prices, five[0])
        as_of = five[0].date()
        all_s = list(scores.keys())
        arms = {
            "EW-12": {s: 1.0 / len(all_s) for s in all_s},
            "HRP-252": _hrp_weights(prices_asof, all_s, as_of),
            f"{name}_top6_HRP": _hrp_weights(prices_asof, top, as_of),
            f"{name}_top6_EW": {s: 1.0 / TOP_K for s in top},
        }
        for arm, w in arms.items():
            arm_rets[arm].append(float(sum(w[s] * asset_rets[s] for s in w)))

    arms_m = {a: _metrics(rs) for a, rs in arm_rets.items()}
    hrp_s = arms_m["HRP-252"]["sharpe_ann_approx"]
    ic = _ic_stats(ics)
    return {
        "name": name,
        "ic": ic,
        "mean_top3_overlap": float(np.mean(top3_ov)) if top3_ov else None,
        "random_top3_overlap": 3.0 / len(symbols),
        "arms": arms_m,
        "beats_hrp_sharpe": arms_m[f"{name}_top6_HRP"]["sharpe_ann_approx"] > hrp_s,
        "rows": rows,
    }


def _verify_e8_preds() -> dict:
    files = sorted(E8_DIR.glob("E8_wd_*_oos_rows.json"))
    arms = {}
    for f in files:
        rows = json.loads(f.read_text())
        key = f.name.replace("_oos_rows.json", "")
        pred = np.array([r["pred_pct"] for r in rows], dtype=float)
        arms[key] = pred
    out: dict = {"n_arms": len(arms), "pairs": [], "per_arm": {}}
    keys = list(arms.keys())
    for f in files:
        rows = json.loads(f.read_text())
        key = f.name.replace("_oos_rows.json", "")
        pred = np.array([r["pred_pct"] for r in rows], dtype=float)
        act = np.array([r["actual_pct"] for r in rows], dtype=float)
        out["per_arm"][key] = {
            "mae": float(np.mean(np.abs(pred - act))),
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std()),
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
        }
    for i, a in enumerate(keys):
        pa = arms[a]
        for b in keys[i + 1 :]:
            pb = arms[b]
            diff = pa - pb
            out["pairs"].append(
                {
                    "a": a,
                    "b": b,
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "mean_abs_diff": float(np.mean(np.abs(diff))),
                    "corr": float(np.corrcoef(pa, pb)[0, 1]),
                    "identical": bool(np.allclose(pa, pb)),
                }
            )

    wd0 = next(k for k in keys if k == "E8_wd_0")
    wd_prod = next(k for k in keys if "0p0001" in k)
    pair = next(p for p in out["pairs"] if {p["a"], p["b"]} == {wd0, wd_prod})
    out["verdict"] = (
        "PREDS_DIFFER — MAE ~tie is real (OOS near-noise), not a harness bug"
        if not pair["identical"] and pair["mean_abs_diff"] > 1e-6
        else "SUSPECT — preds nearly identical; investigate eval harness"
    )
    out["wd0_vs_prod"] = pair
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    e8_check = _verify_e8_preds()
    print("=== E8 pred integrity ===")
    print(
        json.dumps(
            {
                "verdict": e8_check["verdict"],
                "wd0_vs_prod": e8_check["wd0_vs_prod"],
                "per_arm": e8_check["per_arm"],
            },
            indent=2,
        )
    )

    prices = pd.read_pickle(PRICE_CACHE)
    symbols = sorted(prices.keys())
    weeks = _iter_weeks(prices, symbols)
    print(f"symbols={len(symbols)} weeks={len(weeks)}")

    # Build factor scores per week
    factors = {"mom_4w": {}, "mom_1w": {}, "rev_1w": {}}
    for five in weeks:
        ws = str(five[0].date())
        as_of = five[0]
        for fname, lb, sign in [
            ("mom_4w", 20, 1.0),
            ("mom_1w", 5, 1.0),
            ("rev_1w", 5, -1.0),
        ]:
            scores = {}
            for s in symbols:
                v = _mom(prices[s], as_of, lb)
                if v is not None:
                    scores[s] = sign * v
            factors[fname][ws] = scores

    results = {}
    for fname, by_week in factors.items():
        print(f"=== eval {fname} ===")
        r = _eval_factor(fname, by_week, prices, symbols, weeks)
        # drop bulky rows from printed summary; save separately
        rows = r.pop("rows")
        (RESULTS_DIR / f"{fname}_rows.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        results[fname] = r
        print(
            f"  IC {r['ic']['ic_mean_pm_se']} t={r['ic']['ic_tstat']:.2f} "
            f"Sharpe tilt={r['arms'][f'{fname}_top6_HRP']['sharpe_ann_approx']:.2f} "
            f"HRP={r['arms']['HRP-252']['sharpe_ann_approx']:.2f}"
        )

    # E9 IC significance from saved scores
    e9_ic = None
    if E9_SCORES.exists():
        rows = json.loads(E9_SCORES.read_text())
        by_week: dict[str, list] = {}
        for r in rows:
            by_week.setdefault(r["week_start"], []).append(r)
        ics = []
        for ws, items in sorted(by_week.items()):
            sc = [i["pred_weekly_frac"] for i in items]
            ac = [i["actual_weekly_frac"] for i in items]
            rho, _ = spearmanr(sc, ac)
            if np.isfinite(rho):
                ics.append(float(rho))
        e9_ic = _ic_stats(ics)
        print(
            f"=== E9 ListNet IC {e9_ic['ic_mean_pm_se']} t={e9_ic['ic_tstat']:.2f} ==="
        )

    # Interpretation
    factor_any_ic = any(abs(results[f]["ic"]["ic_mean"]) > 0.05 for f in results)
    factor_any_sig = any(abs(results[f]["ic"]["ic_tstat"]) >= 1.96 for f in results)
    e9_sig = e9_ic is not None and abs(e9_ic["ic_tstat"]) >= 1.96

    out = {
        "e8_pred_integrity": e8_check,
        "hrp_note": (
            "HRP-252 is López de Prado hierarchical risk parity on trailing "
            "covariance only — no return forecast. A return tilt should only "
            "add value if it has positive risk-adjusted alpha; losing to HRP "
            "means the tilt adds forecast error, not that PatchTST lost to "
            "another forecaster."
        ),
        "factors": results,
        "e9_listnet_ic_significance": e9_ic,
        "interpretation": {
            "any_factor_abs_ic_gt_0_05": factor_any_ic,
            "any_factor_ic_t_ge_1_96": factor_any_sig,
            "e9_ic_t_ge_1_96": e9_sig,
            "read": (
                "If naive factors also ~0 IC / insignificant: slate/window may "
                "not support technical return tilts — shift from model-fix to "
                "feature/universe design. If a naive factor works and E9 doesn't: "
                "failure is model/representation, not missing alpha in the window."
            ),
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n======== E10 SUMMARY ========")
    # compact print without nested rows
    print(json.dumps(out, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
