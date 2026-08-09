#!/usr/bin/env python3
"""Portfolio bake-off: PatchTST screen vs HRP on fixed 12-name halal slate.

Arms (same 2026 weeks, no sticky, no costs):
  1) EW-12          — equal weight all 12
  2) HRP-252        — HRP on all 12 (lookback 252, as-of before week)
  3) PTST-top6→HRP  — close_only PatchTST rank → top 6 → HRP-252
  4) PTST-top6→EW   — close_only PatchTST rank → top 6 equal weight
  5) MULTI-top6→HRP — multitask PatchTST rank → top 6 → HRP-252

Also reports Spearman IC of PatchTST weekly scores vs realized weekly %.

NOT live Alpha-HRP / Double-HRP (those use ~400 names + sticky).
Artifacts: scratch/.../exp_ptst_vs_hrp/
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.hrp import compute_hrp_allocation
from brain_api.core.patchtst.config import PatchTSTConfig

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_ptst_vs_hrp"
RESULTS_DIR = OUT_DIR / "results"
PRICE_CACHE = BASE_DIR / "exp_e123" / "cache" / "prices.pkl"
MODELS = {
    "close_only": BASE_DIR / "models" / "close_only" / "weights.pt",
    "multitask": BASE_DIR / "models" / "multitask" / "weights.pt",
}

EVAL_YEAR = 2026
CLOSE_IDX = 3
TOP_K = 6
HRP_LOOKBACK = 252
SEED = 20260809


@dataclass
class WeekPort:
    week_start: str
    week_end: str
    arm: str
    ret_frac: float
    n_names: int
    selected: list[str]


def _ohlcv_rets(price_df: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(price_df, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


def _create_ohlcv_model(config: PatchTSTConfig) -> PatchTSTForPrediction:
    hf = HFPatchTSTConfig(
        num_input_channels=config.num_input_channels,
        context_length=config.context_length,
        patch_length=config.patch_length,
        stride=config.stride,
        d_model=config.d_model,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        prediction_length=config.prediction_length,
        attention_dropout=config.dropout,
        positional_dropout=config.dropout,
        use_cls_token=False,
        pooling_type="mean",
    )
    return PatchTSTForPrediction(hf)


def _load_model(path: Path, config: PatchTSTConfig, device: torch.device):
    model = _create_ohlcv_model(config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _predict_weekly_pct(
    model: PatchTSTForPrediction, context: np.ndarray, device: torch.device
) -> float:
    with torch.no_grad():
        x = torch.from_numpy(context[None, ...]).float().to(device)
        daily = model(past_values=x).prediction_outputs[0, :, CLOSE_IDX].cpu().numpy()
    return float((np.exp(np.sum(daily)) - 1.0) * 100.0)


def _iter_weeks(prices: dict[str, pd.DataFrame], symbols: list[str], ctx_len: int):
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
        week_start = five[0]
        ok = True
        for sym in symbols:
            pdf = prices[sym]
            if any(d not in pdf.index for d in five):
                ok = False
                break
            if len(pdf[pdf.index < week_start]) < ctx_len + 1:
                ok = False
                break
        if ok:
            weeks.append((five[0], five[-1], five))
    return weeks


def _truncate_prices(
    prices: dict[str, pd.DataFrame], as_of: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Prices strictly before as_of (no look-ahead into the hold week)."""
    out = {}
    for sym, df in prices.items():
        cut = df[df.index < as_of]
        if len(cut) > 0:
            out[sym] = cut
    return out


def _asset_week_ret(pdf: pd.DataFrame, five: list[pd.Timestamp]) -> float:
    """Simple close-to-close week: close[day5]/close[day0_prev] - 1.

    day0_prev = last close before week_start; if missing use open of day0.
    """
    week_start = five[0]
    prior = pdf[pdf.index < week_start]["close"]
    if len(prior) == 0:
        start = float(pdf.loc[five[0], "close"])
    else:
        start = float(prior.iloc[-1])
    end = float(pdf.loc[five[-1], "close"])
    return end / start - 1.0


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    s = sum(w.values())
    if s <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w} if n else {}
    return {k: v / s for k, v in w.items()}


def _hrp_weights(
    prices_asof: dict[str, pd.DataFrame], symbols: list[str], as_of: date
) -> dict[str, float]:
    subset = {s: prices_asof[s] for s in symbols if s in prices_asof}
    res = compute_hrp_allocation(
        subset, lookback_days=HRP_LOOKBACK, min_data_days=60, as_of_date=as_of
    )
    # percentage_weights are in percent; convert to fraction
    w = {s: v / 100.0 for s, v in res.percentage_weights.items()}
    if not w:
        # fail loud fallback would mask bugs — use EW on requested symbols with data
        have = [s for s in symbols if s in subset]
        return {s: 1.0 / len(have) for s in have} if have else {}
    return _normalize_weights(w)


def _port_return(weights: dict[str, float], asset_rets: dict[str, float]) -> float:
    return float(sum(weights[s] * asset_rets[s] for s in weights if s in asset_rets))


def _metrics(weekly_rets: list[float]) -> dict:
    r = np.array(weekly_rets, dtype=float)
    if len(r) == 0:
        return {}
    cum = float(np.prod(1.0 + r) - 1.0)
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe_w = (mu / sd) if sd > 1e-12 else float("nan")
    # annualize ~52 weeks
    sharpe_ann = sharpe_w * np.sqrt(52) if sd > 1e-12 else float("nan")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return {
        "n_weeks": len(r),
        "cum_return_frac": cum,
        "cum_return_pct": cum * 100,
        "mean_weekly_frac": mu,
        "std_weekly_frac": sd,
        "sharpe_weekly": sharpe_w,
        "sharpe_ann_approx": float(sharpe_ann),
        "hit_rate": float(np.mean(r > 0)),
        "max_drawdown_frac": float(np.min(dd)),
        "max_drawdown_pct": float(np.min(dd) * 100),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing model {name}: {path}")

    prices: dict[str, pd.DataFrame] = pd.read_pickle(PRICE_CACHE)
    symbols = sorted(prices.keys())
    print(f"symbols ({len(symbols)}): {symbols}")

    config = PatchTSTConfig()
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    models = {name: _load_model(path, config, device) for name, path in MODELS.items()}

    weeks = _iter_weeks(prices, symbols, config.context_length)
    print(f"2026 weeks: {len(weeks)}")

    arm_rets: dict[str, list[float]] = {
        "EW-12": [],
        "HRP-252": [],
        "PTST_close_top6_HRP": [],
        "PTST_close_top6_EW": [],
        "PTST_multi_top6_HRP": [],
    }
    week_rows: list[WeekPort] = []
    ic_close: list[float] = []
    ic_multi: list[float] = []
    score_rows: list[dict] = []

    for week_start, week_end, five in weeks:
        ws = str(week_start.date())
        we = str(week_end.date())
        as_of = week_start.date()
        prices_asof = _truncate_prices(prices, week_start)

        # realized asset weekly returns
        asset_rets = {s: _asset_week_ret(prices[s], five) for s in symbols}

        # PatchTST scores (close_only + multitask)
        scores_c: dict[str, float] = {}
        scores_m: dict[str, float] = {}
        for sym in symbols:
            hist = prices[sym][prices[sym].index < week_start]
            rets = _ohlcv_rets(hist)
            ctx = rets.iloc[-config.context_length :].values.astype(np.float32)
            if ctx.shape != (config.context_length, 5) or np.isnan(ctx).any():
                continue
            scores_c[sym] = _predict_weekly_pct(models["close_only"], ctx, device)
            scores_m[sym] = _predict_weekly_pct(models["multitask"], ctx, device)

        if len(scores_c) < TOP_K:
            print(f"skip {ws}: only {len(scores_c)} scores")
            continue

        # IC vs realized (fraction * 100 for same scale as score)
        actual_pct = {s: asset_rets[s] * 100 for s in scores_c}
        sc = [scores_c[s] for s in scores_c]
        sm = [scores_m[s] for s in scores_m]
        ac = [actual_pct[s] for s in scores_c]
        rho_c, _ = spearmanr(sc, ac)
        rho_m, _ = spearmanr(sm, ac)
        if np.isfinite(rho_c):
            ic_close.append(float(rho_c))
        if np.isfinite(rho_m):
            ic_multi.append(float(rho_m))

        for s in scores_c:
            score_rows.append(
                {
                    "week_start": ws,
                    "symbol": s,
                    "score_close_only_pct": scores_c[s],
                    "score_multitask_pct": scores_m[s],
                    "actual_weekly_pct": actual_pct[s],
                }
            )

        top_c = sorted(scores_c, key=scores_c.get, reverse=True)[:TOP_K]
        top_m = sorted(scores_m, key=scores_m.get, reverse=True)[:TOP_K]

        # Arm weights
        arms: dict[str, dict[str, float]] = {}
        arms["EW-12"] = {s: 1.0 / len(symbols) for s in symbols}
        arms["HRP-252"] = _hrp_weights(prices_asof, symbols, as_of)
        arms["PTST_close_top6_HRP"] = _hrp_weights(prices_asof, top_c, as_of)
        arms["PTST_close_top6_EW"] = {s: 1.0 / TOP_K for s in top_c}
        arms["PTST_multi_top6_HRP"] = _hrp_weights(prices_asof, top_m, as_of)

        for arm, w in arms.items():
            r = _port_return(w, asset_rets)
            arm_rets[arm].append(r)
            week_rows.append(
                WeekPort(
                    week_start=ws,
                    week_end=we,
                    arm=arm,
                    ret_frac=r,
                    n_names=len(w),
                    selected=sorted(w.keys()),
                )
            )

        print(
            f"{ws}: EW={arm_rets['EW-12'][-1]:+.3%} "
            f"HRP={arm_rets['HRP-252'][-1]:+.3%} "
            f"PTST→HRP={arm_rets['PTST_close_top6_HRP'][-1]:+.3%} "
            f"IC_c={rho_c:+.3f}"
        )

    summary_arms = {arm: _metrics(rets) for arm, rets in arm_rets.items()}

    # Relative to HRP
    hrp_cum = summary_arms["HRP-252"]["cum_return_frac"]
    vs_hrp = {}
    for arm, m in summary_arms.items():
        vs_hrp[arm] = {
            "cum_minus_hrp_pp": (m["cum_return_frac"] - hrp_cum) * 100,
            "sharpe_minus_hrp": m["sharpe_ann_approx"]
            - summary_arms["HRP-252"]["sharpe_ann_approx"],
            "beats_hrp_cum": m["cum_return_frac"] > hrp_cum,
            "beats_hrp_sharpe": m["sharpe_ann_approx"]
            > summary_arms["HRP-252"]["sharpe_ann_approx"],
        }

    out = {
        "universe": "legacy_halal_12",
        "n_symbols": len(symbols),
        "symbols": symbols,
        "top_k": TOP_K,
        "hrp_lookback": HRP_LOOKBACK,
        "n_weeks": len(weeks),
        "models": {k: str(v) for k, v in MODELS.items()},
        "arms": summary_arms,
        "vs_hrp": vs_hrp,
        "rank_ic": {
            "close_only_spearman_mean": float(np.mean(ic_close)) if ic_close else None,
            "close_only_spearman_std": float(np.std(ic_close)) if ic_close else None,
            "multitask_spearman_mean": float(np.mean(ic_multi)) if ic_multi else None,
            "multitask_spearman_std": float(np.std(ic_multi)) if ic_multi else None,
            "n_weeks_ic": len(ic_close),
        },
        "caveat": (
            "Fixed 12-name slate; not production Alpha-HRP (halal_new screen) "
            "or Double HRP. No costs/turnover. close_only model was barely_trained "
            "(best_epoch=2) in prior walkforward."
        ),
    }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS_DIR / "weekly_portfolio.json").write_text(
        json.dumps([asdict(r) for r in week_rows], indent=2) + "\n"
    )
    (RESULTS_DIR / "weekly_scores.json").write_text(
        json.dumps(score_rows, indent=2) + "\n"
    )

    print("\n======== PTST vs HRP ========")
    print(json.dumps(out, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
