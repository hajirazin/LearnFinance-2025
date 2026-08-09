#!/usr/bin/env python3
"""E5: Z-score close *prices* (levels), not %.

Same halal 10y + 2026 WF harness. Uses raw Close from the OHLCV series
main already downloads — no return transform, no %-zscore.

Per sample (RevIN-style, causal):
  mu, sigma = mean/std of the 60-day close context
  X = (close_ctx - mu) / sigma
  y = (close_next5 - mu) / sigma   # same context stats
Model predicts 5 z-scored closes; RevIN inside HF is OFF (we pre-zscore).
Eval: inverse to price, then daily simple-return frac vs actual.

Baseline weekly_pp JSON under results/ left untouched.
Artifacts: scratch/.../exp_e5_zscore_close_price/
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.universe.halal import get_halal_symbols

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_e5_zscore_close_price"
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


@dataclass
class DayRow:
    week_start: str
    week_end: str
    symbol: str
    day_i: int
    date: str
    actual_price: float
    pred_price: float
    actual_frac: float  # simple return vs prior close
    pred_frac: float
    abs_err_frac: float
    abs_err_z: float
    dir_correct: bool
    context_end: str
    ctx_mu: float
    ctx_sigma: float


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


def _zscore(
    ctx: np.ndarray, future: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mu = float(np.mean(ctx))
    sigma = float(np.std(ctx))
    if sigma < EPS:
        sigma = EPS
    return (ctx - mu) / sigma, (future - mu) / sigma, mu, sigma


def _build_train_xy(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    lo, hi = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    for _sym, pdf in prices.items():
        pdf = pdf[(pdf.index >= lo) & (pdf.index <= hi)]
        close = pdf["close"].astype(float)
        if len(close) < config.context_length + 5:
            continue
        periods = close.index.to_period("W")
        i, n = 0, len(close)
        while i < n:
            p = periods[i]
            j = i + 1
            while j < n and periods[j] == p:
                j += 1
            if j - i >= config.min_week_days:
                t = j - 1
                if t >= config.context_length - 1 and t + 5 < n:
                    ctx = close.iloc[t - config.context_length + 1 : t + 1].values
                    fut = close.iloc[t + 1 : t + 6].values
                    if ctx.shape == (config.context_length,) and fut.shape == (5,):
                        if not (
                            np.isnan(ctx).any()
                            or np.isinf(ctx).any()
                            or np.isnan(fut).any()
                            or np.isinf(fut).any()
                        ):
                            xz, yz, _mu, _sig = _zscore(
                                ctx.astype(np.float64), fut.astype(np.float64)
                            )
                            xs.append(xz.astype(np.float32)[:, None])
                            ys.append(yz.astype(np.float32)[:, None])
            i = j
    X, y = np.stack(xs), np.stack(ys)
    stats = {
        "n": int(len(X)),
        "y_z_mean": float(np.mean(y)),
        "y_z_std": float(np.std(y)),
        "y_z_p01": float(np.quantile(y, 0.01)),
        "y_z_p99": float(np.quantile(y, 0.99)),
        "y_z_min": float(np.min(y)),
        "y_z_max": float(np.max(y)),
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
        scaling=None,  # pre-zscored close prices
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
                "train_mse_z": train_loss,
                "val_mse_z": val_loss,
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
        "representation": "zscore_close_price_levels",
        "zscore": "per_sample_context_mean_std",
        "num_input_channels": 1,
        "scaling": None,
        "lr": LR,
        "best_epoch": best_epoch,
        "best_val_mse_z": best_val,
        "baseline_mse_z": baseline,
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
    model: PatchTSTForPrediction, ctx_z: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    if ctx_z.ndim == 1:
        ctx_z = ctx_z[:, None]
    with torch.no_grad():
        x = torch.from_numpy(ctx_z[None, ...].astype(np.float32)).to(device)
        out = model(past_values=x).prediction_outputs[0, :, 0].cpu().numpy()
    return out.astype(np.float64)


def _iter_weeks(prices: dict[str, pd.DataFrame], symbols: list[str]):
    ref = prices[symbols[0]]
    days = list(ref.index[ref.index.year == EVAL_YEAR])
    by_period: dict = {}
    for ts in days:
        by_period.setdefault(ts.to_period("W"), []).append(ts)
    for _p, sessions in sorted(by_period.items(), key=lambda kv: kv[1][0]):
        if len(sessions) < 5:
            continue
        yield sessions[:5]


def _eval_walkforward(
    model: PatchTSTForPrediction,
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
    symbols: list[str],
) -> tuple[list[DayRow], dict]:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    model = model.to(device)

    rows: list[DayRow] = []
    for week_days in _iter_weeks(prices, symbols):
        week_start = week_days[0].date().isoformat()
        week_end = week_days[-1].date().isoformat()
        for sym in symbols:
            pdf = prices.get(sym)
            if pdf is None:
                continue
            close = pdf["close"].astype(float)
            anchor = week_days[0]
            hist = close.loc[close.index < anchor].dropna()
            if len(hist) < config.context_length:
                continue
            ctx_px = hist.iloc[-config.context_length :].values.astype(np.float64)
            if np.isnan(ctx_px).any() or np.isinf(ctx_px).any():
                continue
            fut_px = []
            ok = True
            for d in week_days:
                if d not in close.index or pd.isna(close.loc[d]):
                    ok = False
                    break
                fut_px.append(float(close.loc[d]))
            if not ok:
                continue
            fut_px_a = np.array(fut_px, dtype=np.float64)
            ctx_z, fut_z, mu, sigma = _zscore(ctx_px, fut_px_a)
            pred_z = _predict_5_z(model, ctx_z, device)
            pred_px = pred_z * sigma + mu

            # daily simple returns vs prior close (last ctx close for day1)
            prior = float(ctx_px[-1])
            ctx_end = hist.index[-1].date().isoformat()
            for i in range(5):
                a_px = fut_px_a[i]
                p_px = float(pred_px[i])
                a_frac = a_px / prior - 1.0
                p_frac = p_px / prior - 1.0
                # chain prior for next day using actual (teacher path for return def)
                # For fair day-i return: return vs previous *actual* close
                if i == 0:
                    prev_a = prior
                else:
                    prev_a = fut_px_a[i - 1]
                a_frac = a_px / prev_a - 1.0
                # pred day-i return vs same previous actual (apples-to-apples level error
                # translated at the actual path) — also report vs prev predicted below
                p_frac = p_px / prev_a - 1.0
                rows.append(
                    DayRow(
                        week_start=week_start,
                        week_end=week_end,
                        symbol=sym,
                        day_i=i + 1,
                        date=week_days[i].date().isoformat(),
                        actual_price=float(a_px),
                        pred_price=p_px,
                        actual_frac=float(a_frac),
                        pred_frac=float(p_frac),
                        abs_err_frac=abs(p_frac - a_frac),
                        abs_err_z=abs(float(pred_z[i]) - float(fut_z[i])),
                        dir_correct=bool(
                            (np.sign(p_frac) == np.sign(a_frac))
                            if a_frac != 0
                            else (p_frac == 0)
                        ),
                        context_end=ctx_end,
                        ctx_mu=mu,
                        ctx_sigma=sigma,
                    )
                )
                prior = a_px  # unused after redefine; keep for clarity

    if not rows:
        raise RuntimeError("no walk-forward rows")

    abs_err = np.array([r.abs_err_frac for r in rows])
    actual = np.array([r.actual_frac for r in rows])
    pred = np.array([r.pred_frac for r in rows])
    naive0_mae = float(np.mean(np.abs(actual)))
    mae = float(np.mean(abs_err))
    nonzero = actual != 0
    dir_acc = float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero])))

    # weekly compound from daily fracs (actual path priors already day-wise)
    # Better: weekly from prices — last pred/actual vs close before week
    weekly_rows = []
    keys = {(r.week_start, r.symbol) for r in rows}
    for ws, sym in sorted(keys):
        sub = sorted(
            [r for r in rows if r.week_start == ws and r.symbol == sym],
            key=lambda r: r.day_i,
        )
        if len(sub) != 5:
            continue
        # reconstruct prior from day1: actual_price / (1+actual_frac)
        prior0 = sub[0].actual_price / (1.0 + sub[0].actual_frac)
        a_week = sub[-1].actual_price / prior0 - 1.0
        p_week = sub[-1].pred_price / prior0 - 1.0
        weekly_rows.append((a_week, p_week))
    wa = np.array([w[0] for w in weekly_rows])
    wp = np.array([w[1] for w in weekly_rows])
    weekly_mae = float(np.mean(np.abs(wp - wa)))
    weekly_naive0 = float(np.mean(np.abs(wa)))

    by_day = {}
    for d in range(1, 6):
        sub = [r for r in rows if r.day_i == d]
        by_day[f"day{d}_mae_frac"] = float(np.mean([r.abs_err_frac for r in sub]))
        by_day[f"day{d}_dir_acc"] = float(np.mean([r.dir_correct for r in sub]))
        by_day[f"day{d}_mae_z"] = float(np.mean([r.abs_err_z for r in sub]))

    summary = {
        "n_rows": len(rows),
        "n_weeks": len({r.week_start for r in rows}),
        "mae_frac": mae,
        "mae_as_pct_points_display": mae * 100,
        "rmse_frac": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "mean_pred_frac": float(np.mean(pred)),
        "mean_actual_frac": float(np.mean(actual)),
        "std_pred_frac": float(np.std(pred)),
        "std_actual_frac": float(np.std(actual)),
        "corr_frac": float(np.corrcoef(pred, actual)[0, 1]) if len(rows) > 1 else None,
        "dir_acc": dir_acc,
        "naive_predict0_mae_frac": naive0_mae,
        "beats_naive0_daily": mae < naive0_mae,
        "mae_z": float(np.mean([r.abs_err_z for r in rows])),
        "weekly_mae_frac": weekly_mae,
        "weekly_naive0_mae_frac": weekly_naive0,
        "beats_naive0_weekly": weekly_mae < weekly_naive0,
        "weekly_mae_pp_display": weekly_mae * 100,
        "pred_z_min": float(
            np.min([r.pred_price for r in rows])
        ),  # placeholder replaced
        **by_day,
    }
    # fix pred z range from rows via recomputing isn't stored — add from prices
    # store pred frac range instead
    summary["pred_frac_min"] = float(np.min(pred))
    summary["pred_frac_max"] = float(np.max(pred))
    del summary["pred_z_min"]
    return rows, summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = get_halal_symbols()
    print(f"symbols ({len(symbols)}): {symbols}")
    prices = _download_or_load(symbols)
    symbols = [s for s in symbols if s in prices]
    print(f"with prices: {symbols}")

    config = PatchTSTConfig()
    config.num_input_channels = 1
    config.epochs = MAX_EPOCHS
    config.early_stopping_patience = PATIENCE
    config.learning_rate = LR

    X, y, y_stats = _build_train_xy(prices, config)
    name = "E5_zscore_close_price"
    model, meta = _train(name, X, y, config)
    rows, wf = _eval_walkforward(model, prices, config, symbols)

    samples = []
    first_week = min(r.week_start for r in rows)
    for sym in symbols[:3]:
        sub = [r for r in rows if r.week_start == first_week and r.symbol == sym]
        if not sub:
            continue
        samples.append(
            {
                "week_start": first_week,
                "symbol": sym,
                "ctx_mu": round(sub[0].ctx_mu, 4),
                "ctx_sigma": round(sub[0].ctx_sigma, 4),
                "days": [
                    {
                        "day_i": r.day_i,
                        "date": r.date,
                        "actual_price": round(r.actual_price, 4),
                        "pred_price": round(r.pred_price, 4),
                        "actual_frac": round(r.actual_frac, 6),
                        "pred_frac": round(r.pred_frac, 6),
                    }
                    for r in sorted(sub, key=lambda x: x.day_i)
                ],
            }
        )

    # Compare blurb vs E4 if present
    e4_path = BASE_DIR / "exp_e4_daily_frac" / "results" / "summary.json"
    e4_cmp = None
    if e4_path.exists():
        e4 = json.loads(e4_path.read_text())
        e4_cmp = {
            "e4_mae_frac": e4["walkforward"]["mae_frac"],
            "e4_dir_acc": e4["walkforward"]["dir_acc"],
            "e4_beats_naive0": e4["walkforward"]["beats_naive0"],
            "e5_mae_frac": wf["mae_frac"],
            "e5_dir_acc": wf["dir_acc"],
            "e5_beats_naive0_daily": wf["beats_naive0_daily"],
            "e5_better_mae_than_e4": wf["mae_frac"] < e4["walkforward"]["mae_frac"],
        }

    out = {
        "y_train_stats": y_stats,
        "train": meta,
        "walkforward": wf,
        "vs_e4": e4_cmp,
        "samples_first_week_3_symbols": samples,
        "note": (
            "Z-score close PRICE levels using per-sample 60d context mean/std. "
            "Predict next 5 closes in z-space; inverse to $; score daily % vs actual. "
            "Not z-score of %. Main today uses log-returns+RevIN — this is price-level."
        ),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS_DIR / "daily_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n"
    )

    print("\n======== E5 SUMMARY ========")
    print(json.dumps({k: out[k] for k in ("train", "walkforward", "vs_e4")}, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
