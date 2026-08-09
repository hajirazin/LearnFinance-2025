#!/usr/bin/env python3
"""E7: Cross-sectional Z-score targets (Alpha-HRP ranking alignment).

Change vs E6 (time-series z-score):
  Target for day t = z-score of that day's simple return *across the 12
  halal names on calendar day t* (not vs each name's own 60d history).

  If AAPL beat the universe that day → positive target; if MSFT lagged → negative.

Input: 60d close simple-return fractions (stationary; no TS z-score).
Output: 5 next-day cross-sectional z-scores.
Eval denorm not needed for ranking — use mean/sum of 5 predicted z as score.
Also: portfolio bake-off top6→HRP vs HRP-252 / EW (same harness as exp_ptst_vs_hrp).

Artifacts: scratch/.../exp_e7_xs_zscore/
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
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.hrp import compute_hrp_allocation
from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.universe.halal import get_halal_symbols

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_e7_xs_zscore"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
CACHE_DIR = OUT_DIR / "cache"
SHARED_CACHE = BASE_DIR / "exp_e123" / "cache" / "prices.pkl"

TRAIN_START = date(2015, 1, 1)
TRAIN_END = date(2025, 12, 31)
EVAL_YEAR = 2026
SEED = 20260809
PATIENCE = 15
MAX_EPOCHS = 100
LR = 3e-4
EPS = 1e-8
TOP_K = 6
HRP_LOOKBACK = 252


@dataclass
class DayRow:
    week_start: str
    symbol: str
    day_i: int
    date: str
    actual_frac: float
    actual_xs_z: float
    pred_xs_z: float
    abs_err_z: float


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _download_or_load(symbols: list[str]) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "prices.pkl"
    if cache_path.exists():
        print(f"loading cached prices {cache_path}")
        obj = pd.read_pickle(cache_path)
        return {k: v for k, v in obj.items() if k in symbols}
    if SHARED_CACHE.exists():
        print(f"copying shared cache {SHARED_CACHE}")
        obj = pd.read_pickle(SHARED_CACHE)
        prices = {k: v for k, v in obj.items() if k in symbols}
        pd.to_pickle(prices, cache_path)
        return prices
    raise FileNotFoundError(f"no price cache at {cache_path} or {SHARED_CACHE}")


def _panel_simple_returns(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide panel: index=date, columns=symbol, values=daily simple return."""
    series = {}
    for sym, pdf in prices.items():
        series[sym] = pdf["close"].astype(float).pct_change()
    panel = pd.DataFrame(series).sort_index()
    return panel


def _cross_section_z(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score across symbols (skip NaNs per row)."""
    mu = panel.mean(axis=1)
    sigma = panel.std(axis=1, ddof=0)
    sigma = sigma.mask(sigma < EPS, EPS)
    return panel.sub(mu, axis=0).div(sigma, axis=0)


def _build_train_xy(
    panel_r: pd.DataFrame,
    panel_z: pd.DataFrame,
    symbols: list[str],
    config: PatchTSTConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Samples aligned by calendar week across the universe.

    For each (symbol, week-anchor) with full context + 5 forward days:
      X = that symbol's 60d raw simple returns
      y = that symbol's next-5 cross-sectional z-scores
    """
    lo, hi = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    # Common calendar where >= half universe has returns
    idx = panel_r.index[(panel_r.index >= lo) & (panel_r.index <= hi)]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    # week anchors: last day of each ISO week with >= min_week_days sessions
    periods = idx.to_period("W")
    i, n = 0, len(idx)
    anchors: list[pd.Timestamp] = []
    while i < n:
        p = periods[i]
        j = i + 1
        while j < n and periods[j] == p:
            j += 1
        if j - i >= config.min_week_days:
            anchors.append(idx[j - 1])
        i = j

    for anchor in anchors:
        # need 5 trading days after anchor in panel
        pos = panel_r.index.get_loc(anchor)
        if isinstance(pos, slice):
            continue
        if pos < config.context_length - 1:
            continue
        if pos + 5 >= len(panel_r.index):
            continue
        fwd_dates = panel_r.index[pos + 1 : pos + 6]
        if len(fwd_dates) < 5:
            continue

        for sym in symbols:
            if sym not in panel_r.columns:
                continue
            ctx = panel_r[sym].iloc[pos - config.context_length + 1 : pos + 1].values
            tgt = panel_z[sym].loc[fwd_dates].values
            if ctx.shape != (config.context_length,) or tgt.shape != (5,):
                continue
            if (
                np.isnan(ctx).any()
                or np.isinf(ctx).any()
                or np.isnan(tgt).any()
                or np.isinf(tgt).any()
            ):
                continue
            xs.append(ctx.astype(np.float32)[:, None])
            ys.append(tgt.astype(np.float32)[:, None])

    if not xs:
        raise RuntimeError("no training samples")
    X, y = np.stack(xs), np.stack(ys)
    stats = {
        "n": int(len(X)),
        "n_anchors": len(anchors),
        "y_z_mean": float(np.mean(y)),
        "y_z_std": float(np.std(y)),
        "y_z_p01": float(np.quantile(y, 0.01)),
        "y_z_p99": float(np.quantile(y, 0.99)),
    }
    print(f"train set X={X.shape} y={y.shape} stats={stats}")
    return X, y, stats


def _create_model(config: PatchTSTConfig) -> PatchTSTForPrediction:
    hf = HFPatchTSTConfig(
        num_input_channels=1,
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
        scaling=None,
    )
    return PatchTSTForPrediction(hf)


def _mean_target_baseline(y_va: np.ndarray) -> float:
    t = y_va[:, :, 0]
    mean = np.mean(t, axis=0, keepdims=True)
    return float(np.mean((t - mean) ** 2))


def _train(
    name: str, X: np.ndarray, y: np.ndarray, config: PatchTSTConfig
) -> tuple[PatchTSTForPrediction, dict]:
    _set_seeds()
    split = int(len(X) * (1 - config.validation_split))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    baseline = _mean_target_baseline(y_va)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=min(config.batch_size, len(X_tr)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=min(config.batch_size, max(len(X_va), 1)),
        shuffle=False,
    )

    model = _create_model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history: list[dict] = []

    print(
        f"[{name}] lr={LR} n_train={len(X_tr)} n_val={len(X_va)} "
        f"baseline_mse={baseline:.6e} device={device}"
    )

    for epoch in range(config.epochs):
        model.train()
        tot, n = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(past_values=bx).prediction_outputs
            loss = F.mse_loss(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            opt.step()
            tot += float(loss.detach())
            n += 1
        train_loss = tot / max(n, 1)

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                preds = model(past_values=vx).prediction_outputs
                vloss = F.mse_loss(preds, vy)
                vtot += float(vloss.detach())
                vn += 1
        val_loss = vtot / max(vn, 1)
        sched.step(val_loss)

        history.append(
            {
                "epoch": epoch + 1,
                "train_mse_xs_z": train_loss,
                "val_mse_xs_z": val_loss,
                "beats_baseline": val_loss < baseline,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[{name}] ep{epoch + 1}: train={train_loss:.6e} val={val_loss:.6e} "
                f"beat_bl={val_loss < baseline} pat={patience}/{PATIENCE} "
                f"best_ep={best_epoch}"
            )

        if patience >= PATIENCE:
            print(
                f"[{name}] early-stop ep{epoch + 1} best_ep={best_epoch} "
                f"best_val={best_val:.6e}"
            )
            break
    else:
        print(f"[{name}] finished all {config.epochs} epochs best_ep={best_epoch}")

    assert best_state is not None
    model_cpu = _create_model(config)
    model_cpu.load_state_dict(best_state)
    path = MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path / "weights.pt")
    meta = {
        "name": name,
        "representation": "cross_sectional_z_of_daily_simple_return",
        "input": "60d_raw_simple_return_frac",
        "target": "next_5_xs_z_across_universe",
        "lr": LR,
        "best_epoch": best_epoch,
        "best_val_mse_xs_z": best_val,
        "baseline_mse_xs_z": baseline,
        "beats_baseline_best": best_val < baseline,
        "stopped_epoch": history[-1]["epoch"],
        "ran_full_100": history[-1]["epoch"] == MAX_EPOCHS,
        "barely_trained": best_epoch <= 2,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[{name}] meta={json.dumps(meta)}")
    return model_cpu, meta


def _predict_5_z(
    model: PatchTSTForPrediction, ctx: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    if ctx.ndim == 1:
        ctx = ctx[:, None]
    with torch.no_grad():
        x = torch.from_numpy(ctx[None, ...].astype(np.float32)).to(device)
        out = model(past_values=x).prediction_outputs[0, :, 0].cpu().numpy()
    return out.astype(np.float64)


def _iter_weeks(panel_r: pd.DataFrame, symbols: list[str], ctx_len: int):
    days = list(panel_r.index[panel_r.index.year == EVAL_YEAR])
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
            hist = panel_r[sym].loc[panel_r.index < week_start].dropna()
            if len(hist) < ctx_len:
                ok = False
                break
            if any(
                d not in panel_r.index or pd.isna(panel_r.loc[d, sym]) for d in five
            ):
                ok = False
                break
        if ok:
            weeks.append((five[0], five[-1], five))
    return weeks


def _truncate_prices(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp):
    return {
        s: df[df.index < as_of] for s, df in prices.items() if len(df[df.index < as_of])
    }


def _asset_week_ret(pdf: pd.DataFrame, five: list[pd.Timestamp]) -> float:
    week_start = five[0]
    prior = pdf[pdf.index < week_start]["close"]
    start = float(prior.iloc[-1]) if len(prior) else float(pdf.loc[five[0], "close"])
    end = float(pdf.loc[five[-1], "close"])
    return end / start - 1.0


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    s = sum(w.values())
    if s <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w} if n else {}
    return {k: v / s for k, v in w.items()}


def _hrp_weights(prices_asof, symbols, as_of: date) -> dict[str, float]:
    subset = {s: prices_asof[s] for s in symbols if s in prices_asof}
    res = compute_hrp_allocation(
        subset, lookback_days=HRP_LOOKBACK, min_data_days=60, as_of_date=as_of
    )
    w = {s: v / 100.0 for s, v in res.percentage_weights.items()}
    if not w:
        have = [s for s in symbols if s in subset]
        return {s: 1.0 / len(have) for s in have} if have else {}
    return _normalize_weights(w)


def _port_return(weights: dict[str, float], asset_rets: dict[str, float]) -> float:
    return float(sum(weights[s] * asset_rets[s] for s in weights if s in asset_rets))


def _metrics(weekly_rets: list[float]) -> dict:
    r = np.array(weekly_rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe_w = (mu / sd) if sd > 1e-12 else float("nan")
    sharpe_ann = sharpe_w * np.sqrt(52) if sd > 1e-12 else float("nan")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return {
        "n_weeks": len(r),
        "cum_return_pct": cum * 100,
        "sharpe_ann_approx": float(sharpe_ann),
        "hit_rate": float(np.mean(r > 0)),
        "max_drawdown_pct": float(np.min(dd) * 100),
        "mean_weekly_frac": mu,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = get_halal_symbols()
    prices = _download_or_load(symbols)
    symbols = [s for s in symbols if s in prices]
    print(f"symbols ({len(symbols)}): {symbols}")

    panel_r = _panel_simple_returns(prices)
    panel_z = _cross_section_z(panel_r)
    print(
        f"panel {panel_r.shape}, XS-z nan rate={float(panel_z.isna().mean().mean()):.3f}"
    )

    config = PatchTSTConfig()
    config.num_input_channels = 1
    config.epochs = MAX_EPOCHS
    config.early_stopping_patience = PATIENCE
    config.learning_rate = LR

    X, y, y_stats = _build_train_xy(panel_r, panel_z, symbols, config)
    name = "E7_xs_zscore"
    model, meta = _train(name, X, y, config)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)

    weeks = _iter_weeks(panel_r, symbols, config.context_length)
    print(f"2026 weeks: {len(weeks)}")

    day_rows: list[DayRow] = []
    ic_score_vs_week_ret: list[float] = []
    ic_score_vs_week_xs: list[float] = []
    arm_rets: dict[str, list[float]] = {
        "EW-12": [],
        "HRP-252": [],
        "E7_top6_HRP": [],
        "E7_top6_EW": [],
    }
    score_rows: list[dict] = []

    for week_start, week_end, five in weeks:
        ws = str(week_start.date())
        as_of = week_start.date()
        prices_asof = _truncate_prices(prices, week_start)
        asset_rets = {s: _asset_week_ret(prices[s], five) for s in symbols}

        # actual weekly XS-z of compound week return across universe
        week_frac = np.array([asset_rets[s] for s in symbols])
        wmu, wsd = float(week_frac.mean()), float(week_frac.std())
        if wsd < EPS:
            wsd = EPS
        actual_week_xs = {s: (asset_rets[s] - wmu) / wsd for s in symbols}

        scores: dict[str, float] = {}
        for sym in symbols:
            hist = panel_r[sym].loc[panel_r.index < week_start].dropna()
            ctx = hist.iloc[-config.context_length :].values.astype(np.float64)
            if len(ctx) < config.context_length or np.isnan(ctx).any():
                continue
            pred_z = _predict_5_z(model, ctx, device)
            # ranking score = mean predicted daily XS-z over the horizon
            scores[sym] = float(np.mean(pred_z))

            for i, d in enumerate(five):
                a_frac = float(panel_r.loc[d, sym])
                a_z = float(panel_z.loc[d, sym])
                day_rows.append(
                    DayRow(
                        week_start=ws,
                        symbol=sym,
                        day_i=i + 1,
                        date=str(d.date()),
                        actual_frac=a_frac,
                        actual_xs_z=a_z,
                        pred_xs_z=float(pred_z[i]),
                        abs_err_z=abs(float(pred_z[i]) - a_z),
                    )
                )

            score_rows.append(
                {
                    "week_start": ws,
                    "symbol": sym,
                    "score_mean_xs_z": scores[sym],
                    "actual_week_frac": asset_rets[sym],
                    "actual_week_xs_z": actual_week_xs[sym],
                }
            )

        if len(scores) < TOP_K:
            continue

        sc = [scores[s] for s in scores]
        ar = [asset_rets[s] for s in scores]
        ax = [actual_week_xs[s] for s in scores]
        rho_r, _ = spearmanr(sc, ar)
        rho_x, _ = spearmanr(sc, ax)
        if np.isfinite(rho_r):
            ic_score_vs_week_ret.append(float(rho_r))
        if np.isfinite(rho_x):
            ic_score_vs_week_xs.append(float(rho_x))

        top = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
        arms = {
            "EW-12": {s: 1.0 / len(symbols) for s in symbols},
            "HRP-252": _hrp_weights(prices_asof, symbols, as_of),
            "E7_top6_HRP": _hrp_weights(prices_asof, top, as_of),
            "E7_top6_EW": {s: 1.0 / TOP_K for s in top},
        }
        for arm, w in arms.items():
            arm_rets[arm].append(_port_return(w, asset_rets))

        print(
            f"{ws}: HRP={arm_rets['HRP-252'][-1]:+.3%} "
            f"E7→HRP={arm_rets['E7_top6_HRP'][-1]:+.3%} "
            f"IC_ret={rho_r:+.3f} IC_xs={rho_x:+.3f}"
        )

    mae_z = float(np.mean([r.abs_err_z for r in day_rows])) if day_rows else None
    summary_arms = {a: _metrics(rs) for a, rs in arm_rets.items()}
    hrp_sharpe = summary_arms["HRP-252"]["sharpe_ann_approx"]
    vs_hrp = {
        a: {
            "cum_minus_hrp_pp": summary_arms[a]["cum_return_pct"]
            - summary_arms["HRP-252"]["cum_return_pct"],
            "sharpe_minus_hrp": summary_arms[a]["sharpe_ann_approx"] - hrp_sharpe,
            "beats_hrp_sharpe": summary_arms[a]["sharpe_ann_approx"] > hrp_sharpe,
        }
        for a in summary_arms
    }

    # compare to prior bake-off if present
    prior = None
    prior_path = BASE_DIR / "exp_ptst_vs_hrp" / "results" / "summary.json"
    if prior_path.exists():
        p = json.loads(prior_path.read_text())
        prior = {
            "PTST_close_top6_HRP": p["arms"].get("PTST_close_top6_HRP"),
            "HRP-252": p["arms"].get("HRP-252"),
            "rank_ic_close_only_mean": p["rank_ic"].get("close_only_spearman_mean"),
        }

    out = {
        "y_train_stats": y_stats,
        "train": meta,
        "forecast": {
            "mae_xs_z": mae_z,
            "n_day_rows": len(day_rows),
        },
        "rank_ic": {
            "score_vs_week_return_mean": float(np.mean(ic_score_vs_week_ret))
            if ic_score_vs_week_ret
            else None,
            "score_vs_week_return_std": float(np.std(ic_score_vs_week_ret))
            if ic_score_vs_week_ret
            else None,
            "score_vs_week_xs_z_mean": float(np.mean(ic_score_vs_week_xs))
            if ic_score_vs_week_xs
            else None,
            "score_vs_week_xs_z_std": float(np.std(ic_score_vs_week_xs))
            if ic_score_vs_week_xs
            else None,
            "n_weeks": len(ic_score_vs_week_ret),
        },
        "arms": summary_arms,
        "vs_hrp": vs_hrp,
        "prior_ptst_vs_hrp": prior,
        "note": (
            "Targets = cross-sectional z of daily simple returns across 12 names. "
            "Ranking score = mean of 5 predicted daily XS-z. Not production Alpha-HRP."
        ),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS_DIR / "daily_rows.json").write_text(
        json.dumps([asdict(r) for r in day_rows], indent=2) + "\n"
    )
    (RESULTS_DIR / "weekly_scores.json").write_text(
        json.dumps(score_rows, indent=2) + "\n"
    )

    print("\n======== E7 SUMMARY ========")
    print(
        json.dumps(
            {k: out[k] for k in ("train", "rank_ic", "arms", "vs_hrp")},
            indent=2,
        )
    )
    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
