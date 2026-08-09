#!/usr/bin/env python3
"""E4: PatchTST on daily close *simple returns as fractions* (0.02 = 2%).

- Input: 60d close-only context of r_t = close_t/close_{t-1} - 1
- Target / pred: next 5 trading-day close simple returns (5 numbers)
- No pp scaling, no extra map to [-1,1]; RevIN off so values stay as fractions
- Eval: day-by-day pred vs actual on 2026 weeks (teacher-forced context)

Baseline weekly_pp JSON under results/ is left untouched.
Artifacts: scratch/.../exp_e4_daily_frac/
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
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.universe.halal import get_halal_symbols

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_e4_daily_frac"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
CACHE_DIR = OUT_DIR / "cache"
# Reuse E123 price cache if present
SHARED_CACHE = BASE_DIR / "exp_e123" / "cache" / "prices.pkl"

TRAIN_START = date(2015, 1, 1)
TRAIN_END = date(2025, 12, 31)
EVAL_YEAR = 2026
SEED = 20260809
PATIENCE = 15
MAX_EPOCHS = 100
LR = 3e-4


@dataclass
class DayRow:
    week_start: str
    week_end: str
    symbol: str
    day_i: int  # 1..5 within the week forecast
    date: str
    actual_frac: float
    pred_frac: float
    err_pred_minus_actual: float
    abs_err: float
    dir_correct: bool
    context_end: str


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _close_simple_frac(price_df: pd.DataFrame) -> pd.Series:
    """Daily close simple return as fraction: 0.02 == +2%."""
    close = price_df["close"].astype(float)
    return close.pct_change()


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

    prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.download(
            sym,
            start=TRAIN_START.isoformat(),
            end="2026-12-31",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or len(df) == 0:
            print(f"  SKIP {sym}: empty")
            continue
        if getattr(df.columns, "nlevels", 1) > 1:
            df.columns = df.columns.get_level_values(0)
        need = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in need):
            print(f"  SKIP {sym}: cols")
            continue
        out = df[need].copy()
        out.columns = ["open", "high", "low", "close", "volume"]
        out = out.dropna()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        if len(out) < 400:
            print(f"  SKIP {sym}: short")
            continue
        prices[sym] = out
        print(f"  {sym}: {len(out)} rows")
    pd.to_pickle(prices, cache_path)
    return prices


def _build_train_xy(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """X: (N, ctx, 1), y: (N, 5, 1) — close simple-return fractions."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    lo, hi = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    for _sym, pdf in prices.items():
        pdf = pdf[(pdf.index >= lo) & (pdf.index <= hi)]
        rets = _close_simple_frac(pdf).dropna()
        if len(rets) < config.context_length + 5:
            continue
        periods = rets.index.to_period("W")
        i, n = 0, len(rets)
        while i < n:
            p = periods[i]
            j = i + 1
            while j < n and periods[j] == p:
                j += 1
            if j - i >= config.min_week_days:
                t = j - 1
                if t >= config.context_length - 1 and t + 5 < n:
                    seq = rets.iloc[t - config.context_length + 1 : t + 1].values
                    tgt = rets.iloc[t + 1 : t + 6].values
                    if seq.shape == (config.context_length,) and tgt.shape == (5,):
                        if not (
                            np.isnan(seq).any()
                            or np.isinf(seq).any()
                            or np.isnan(tgt).any()
                            or np.isinf(tgt).any()
                        ):
                            xs.append(seq.astype(np.float32)[:, None])
                            ys.append(tgt.astype(np.float32)[:, None])
            i = j
    X, y = np.stack(xs), np.stack(ys)
    stats = {
        "n": int(len(X)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "y_p01": float(np.quantile(y, 0.01)),
        "y_p99": float(np.quantile(y, 0.99)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "frac_abs_gt_0_05": float(np.mean(np.abs(y) > 0.05)),
        "frac_abs_gt_0_10": float(np.mean(np.abs(y) > 0.10)),
    }
    print(f"train set X={X.shape} y={y.shape} stats={stats}")
    return X, y, stats


def _create_model(config: PatchTSTConfig) -> PatchTSTForPrediction:
    """1-channel close-frac PatchTST; RevIN off so fractions stay as-is."""
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
    """MSE of predicting mean daily frac per horizon slot on val."""
    close = y_va[:, :, 0]
    mean = np.mean(close, axis=0, keepdims=True)
    return float(np.mean((close - mean) ** 2))


def _train(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    config: PatchTSTConfig,
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
        tot = 0.0
        n = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(past_values=bx).prediction_outputs  # (B, 5, 1)
            loss = F.mse_loss(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            opt.step()
            tot += float(loss.detach())
            n += 1
        train_loss = tot / max(n, 1)

        model.eval()
        vtot = 0.0
        vn = 0
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
                "train_mse": train_loss,
                "val_mse": val_loss,
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
        "representation": "close_simple_return_fraction",
        "example": "2% -> 0.02",
        "num_input_channels": 1,
        "scaling": None,
        "lr": LR,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "baseline_mse": baseline,
        "beats_baseline_best": best_val < baseline,
        "stopped_epoch": history[-1]["epoch"],
        "ran_full_100": history[-1]["epoch"] == MAX_EPOCHS,
        "barely_trained": best_epoch <= 2,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[{name}] meta={json.dumps(meta)}")
    return model_cpu, meta


def _predict_5(
    model: PatchTSTForPrediction, context: np.ndarray, device: torch.device
) -> np.ndarray:
    """context (60,) or (60,1) -> pred (5,) fractions."""
    model.eval()
    if context.ndim == 1:
        context = context[:, None]
    with torch.no_grad():
        x = torch.from_numpy(context[None, ...]).float().to(device)
        out = model(past_values=x).prediction_outputs[0, :, 0].cpu().numpy()
    return out.astype(np.float64)


def _iter_weeks(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig, symbols: list[str]
):
    ref = prices[symbols[0]]
    days = list(ref.index[ref.index.year == EVAL_YEAR])
    by_period: dict = {}
    for ts in days:
        by_period.setdefault(ts.to_period("W"), []).append(ts)
    for _p, sessions in sorted(by_period.items(), key=lambda kv: kv[1][0]):
        if len(sessions) < 5:
            continue
        # Use first 5 sessions of the ISO week as the forecast window
        week_days = sessions[:5]
        yield week_days


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
    for week_days in _iter_weeks(prices, config, symbols):
        week_start = week_days[0].date().isoformat()
        week_end = week_days[-1].date().isoformat()
        for sym in symbols:
            pdf = prices.get(sym)
            if pdf is None:
                continue
            rets = _close_simple_frac(pdf)
            # Context ends on the trading day before week_days[0]
            anchor = week_days[0]
            hist = rets.loc[rets.index < anchor].dropna()
            if len(hist) < config.context_length:
                continue
            ctx = hist.iloc[-config.context_length :].values.astype(np.float32)
            if np.isnan(ctx).any() or np.isinf(ctx).any():
                continue
            # Actuals: simple return on each of the 5 week days
            actuals = []
            ok = True
            for d in week_days:
                if d not in rets.index or pd.isna(rets.loc[d]):
                    ok = False
                    break
                actuals.append(float(rets.loc[d]))
            if not ok or len(actuals) != 5:
                continue
            preds = _predict_5(model, ctx, device)
            ctx_end = hist.index[-1].date().isoformat()
            for i in range(5):
                a, p = actuals[i], float(preds[i])
                rows.append(
                    DayRow(
                        week_start=week_start,
                        week_end=week_end,
                        symbol=sym,
                        day_i=i + 1,
                        date=week_days[i].date().isoformat(),
                        actual_frac=a,
                        pred_frac=p,
                        err_pred_minus_actual=p - a,
                        abs_err=abs(p - a),
                        dir_correct=bool(
                            (np.sign(p) == np.sign(a)) if a != 0 else (p == 0)
                        ),
                        context_end=ctx_end,
                    )
                )

    if not rows:
        raise RuntimeError("no walk-forward rows")

    abs_err = np.array([r.abs_err for r in rows])
    actual = np.array([r.actual_frac for r in rows])
    pred = np.array([r.pred_frac for r in rows])
    # naive predict-0
    naive0_mae = float(np.mean(np.abs(actual)))
    mae = float(np.mean(abs_err))
    # direction: ignore exact zeros in actual
    nonzero = actual != 0
    dir_acc = float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero])))
    # per-day-slot MAE
    by_day = {}
    for d in range(1, 6):
        sub = [r for r in rows if r.day_i == d]
        by_day[f"day{d}_mae_frac"] = float(np.mean([r.abs_err for r in sub]))
        by_day[f"day{d}_dir_acc"] = float(np.mean([r.dir_correct for r in sub]))

    summary = {
        "n_rows": len(rows),
        "n_weeks": len({r.week_start for r in rows}),
        "mae_frac": mae,
        "mae_as_pct_points_display": mae * 100,  # for human read only
        "rmse_frac": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "mean_pred_frac": float(np.mean(pred)),
        "mean_actual_frac": float(np.mean(actual)),
        "std_pred_frac": float(np.std(pred)),
        "std_actual_frac": float(np.std(actual)),
        "corr": float(np.corrcoef(pred, actual)[0, 1]) if len(rows) > 1 else None,
        "dir_acc": dir_acc,
        "naive_predict0_mae_frac": naive0_mae,
        "beats_naive0": mae < naive0_mae,
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        **by_day,
    }
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
    name = "E4_close_simple_frac"
    model, meta = _train(name, X, y, config)
    rows, wf = _eval_walkforward(model, prices, config, symbols)

    # Sample: first week, all symbols, show 5 pred vs actual
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
                "days": [
                    {
                        "day_i": r.day_i,
                        "date": r.date,
                        "actual_frac": round(r.actual_frac, 6),
                        "pred_frac": round(r.pred_frac, 6),
                        "actual_pct_display": round(r.actual_frac * 100, 3),
                        "pred_pct_display": round(r.pred_frac * 100, 3),
                    }
                    for r in sorted(sub, key=lambda x: x.day_i)
                ],
            }
        )

    out = {
        "y_train_stats": y_stats,
        "train": meta,
        "walkforward": wf,
        "samples_first_week_3_symbols": samples,
        "note": (
            "Fractions: 0.02 == +2%. Model is 1-channel close-only, RevIN off, "
            "predicts 5 next-day close simple returns. Baseline weekly_pp JSON kept."
        ),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS_DIR / "daily_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n"
    )

    # Compact readable by week
    by_week: dict[str, list] = {}
    for r in rows:
        by_week.setdefault(r.week_start, []).append(r)
    readable = {
        "overall": {
            "mae_frac": round(wf["mae_frac"], 6),
            "naive0_mae_frac": round(wf["naive_predict0_mae_frac"], 6),
            "beats_naive0": wf["beats_naive0"],
            "dir_acc": round(wf["dir_acc"], 4),
            "corr": None if wf["corr"] is None else round(wf["corr"], 4),
        },
        "weeks": [],
    }
    for ws in sorted(by_week):
        sub = by_week[ws]
        readable["weeks"].append(
            {
                "week_start": ws,
                "mae_frac": round(float(np.mean([r.abs_err for r in sub])), 6),
                "dir_acc": round(float(np.mean([r.dir_correct for r in sub])), 4),
                "examples": [
                    {
                        "symbol": r.symbol,
                        "day_i": r.day_i,
                        "date": r.date,
                        "actual": round(r.actual_frac, 5),
                        "pred": round(r.pred_frac, 5),
                    }
                    for r in sub[:6]
                ],
            }
        )
    (RESULTS_DIR / "daily_readable.json").write_text(
        json.dumps(readable, indent=2) + "\n"
    )

    print("\n======== E4 SUMMARY ========")
    print(json.dumps(out, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
