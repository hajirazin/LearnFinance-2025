#!/usr/bin/env python3
"""Halal-universe + 10y walk-forward: close_only vs multitask.

- Universe: legacy ``get_halal_symbols()`` (~12 names)
- Train window: 2015-01-01 .. 2025-12-31 (matches DEFAULT_LOOKBACK_YEARS=10
  anchored to Jan 1 of end_year-10, with TRAIN_END fixed to 2025-12-31)
- Eval: every 2026 week with 5 sessions; context = actual OHLCV rets ending
  day before week_start (prior weeks teacher-forced)
- Artifacts only under scratch/ — does not touch data/models/
"""

from __future__ import annotations

import csv
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

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.universe.halal import get_halal_symbols

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"

TRAIN_START = date(2015, 1, 1)  # 10y lookback vs end_year 2025
TRAIN_END = date(2025, 12, 31)
EVAL_YEAR = 2026
SEED = 20260809
CLOSE_IDX = 3
PROD_LR = 3e-4
PATIENCE = 15
MAX_EPOCHS = 100


@dataclass
class WeekScore:
    week_start: str
    week_end: str
    symbol: str
    actual_weekly_pct: float
    pred_close_only_pct: float
    pred_multitask_pct: float
    err_close_only: float
    err_multitask: float
    abs_err_close_only: float
    abs_err_multitask: float
    context_end: str
    n_context_days: int


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _download_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    start = TRAIN_START.isoformat()
    end = "2026-12-31"
    prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.download(
            sym, start=start, end=end, progress=False, auto_adjust=False, threads=False
        )
        if df is None or len(df) == 0:
            print(f"  SKIP {sym}: empty download")
            continue
        if getattr(df.columns, "nlevels", 1) > 1:
            df.columns = df.columns.get_level_values(0)
        need = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in need):
            print(f"  SKIP {sym}: missing OHLCV cols")
            continue
        out = df[need].copy()
        out.columns = ["open", "high", "low", "close", "volume"]
        out = out.dropna()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        # Need enough history for context_length before 2026
        if len(out) < 400:
            print(f"  SKIP {sym}: only {len(out)} rows")
            continue
        prices[sym] = out
        print(
            f"  {sym}: {len(out)} rows  {out.index[0].date()} .. {out.index[-1].date()}"
        )
    if len(prices) < 3:
        raise RuntimeError(f"too few symbols downloaded: {list(prices)}")
    return prices


def _ohlcv_rets(price_df: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(price_df, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


def _build_train_xy(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    # Keep bars from TRAIN_START through TRAIN_END (targets must fit in window)
    lo = pd.Timestamp(TRAIN_START)
    hi = pd.Timestamp(TRAIN_END)

    for sym, pdf in prices.items():
        pdf = pdf[(pdf.index >= lo) & (pdf.index <= hi)]
        rets = _ohlcv_rets(pdf)
        if len(rets) < config.context_length + 5:
            print(f"  skip train {sym}: short history {len(rets)}")
            continue
        periods = rets.index.to_period("W")
        i = 0
        n = len(rets)
        n_sym = 0
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
                    if seq.shape == (config.context_length, 5) and tgt.shape == (5, 5):
                        if not (
                            np.isnan(seq).any()
                            or np.isinf(seq).any()
                            or np.isnan(tgt).any()
                            or np.isinf(tgt).any()
                        ):
                            xs.append(seq.astype(np.float32))
                            ys.append(tgt.astype(np.float32))
                            n_sym += 1
            i = j
        print(f"  {sym}: {n_sym} train samples")

    if not xs:
        raise RuntimeError("no training samples")
    X = np.stack(xs)
    y = np.stack(ys)
    print(f"train set X={X.shape} y={y.shape}")
    return X, y


def _create_model(config: PatchTSTConfig) -> PatchTSTForPrediction:
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


def _train(
    name: str,
    loss_mode: str,
    X: np.ndarray,
    y: np.ndarray,
    config: PatchTSTConfig,
) -> tuple[PatchTSTForPrediction, dict]:
    _set_seeds()
    split = int(len(X) * (1 - config.validation_split))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    bs = min(config.batch_size, len(X_tr))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=bs,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=min(config.batch_size, max(len(X_va), 1)),
        shuffle=False,
    )

    model = _create_model(config).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history: list[dict] = []

    print(
        f"[{name}] n_train={len(X_tr)} n_val={len(X_va)} device={device} "
        f"mode={loss_mode} bs={bs} lr={config.learning_rate}"
    )
    for epoch in range(config.epochs):
        model.train()
        tot = 0.0
        n = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(past_values=bx, future_values=by)
            preds = out.prediction_outputs
            if loss_mode == "close_only":
                loss = F.mse_loss(preds[:, :, CLOSE_IDX], by[:, :, CLOSE_IDX])
            elif loss_mode == "multitask":
                loss = out.loss
            else:
                raise ValueError(loss_mode)
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
                out = model(past_values=vx, future_values=vy)
                preds = out.prediction_outputs
                if loss_mode == "close_only":
                    vloss = F.mse_loss(preds[:, :, CLOSE_IDX], vy[:, :, CLOSE_IDX])
                else:
                    vloss = out.loss
                vtot += float(vloss.detach())
                vn += 1
        val_loss = vtot / max(vn, 1)
        sched.step(val_loss)
        history.append(
            {"epoch": epoch + 1, "train_obj": train_loss, "val_obj": val_loss}
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
            if patience >= config.early_stopping_patience:
                print(
                    f"[{name}] early-stop ep{epoch + 1} best_ep={best_epoch} "
                    f"train={train_loss:.6e} val={val_loss:.6e}"
                )
                break
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[{name}] ep{epoch + 1}: train={train_loss:.6e} val={val_loss:.6e} "
                f"pat={patience}/{config.early_stopping_patience} best_ep={best_epoch}"
            )

    assert best_state is not None
    if best_epoch <= 2:
        print(
            f"[{name}] WARNING: best_epoch={best_epoch} — barely trained "
            f"(same pathology as production close-only)"
        )

    model_cpu = _create_model(config)
    model_cpu.load_state_dict(best_state)
    path = MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path / "weights.pt")
    meta = {
        "name": name,
        "loss_mode": loss_mode,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "stopped_epoch": history[-1]["epoch"] if history else None,
        "train_start": TRAIN_START.isoformat(),
        "train_end": TRAIN_END.isoformat(),
        "n_train": len(X_tr),
        "n_val": len(X_va),
        "seed": SEED,
        "barely_trained": best_epoch <= 2,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[{name}] saved {path} best_epoch={best_epoch} best_val={best_val:.6e}")
    return model_cpu, meta


def _predict_weekly_close_pct(
    model: PatchTSTForPrediction, context: np.ndarray, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(context[None, ...]).float().to(device)
        preds = model(past_values=x).prediction_outputs
        daily = preds[0, :, CLOSE_IDX].cpu().numpy()
    return float((np.exp(np.sum(daily)) - 1.0) * 100.0)


def _iter_2026_weeks(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig, symbols: list[str]
) -> list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]]:
    ref = prices[symbols[0]]
    days = list(ref.index[ref.index.year == EVAL_YEAR])
    if not days:
        raise RuntimeError("no 2026 trading days")
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
            for d in five:
                if d not in pdf.index:
                    ok = False
                    break
            if not ok:
                break
            if len(pdf[pdf.index < week_start]) < config.context_length + 1:
                ok = False
                break
        if ok:
            weeks.append((five[0], five[-1], five))
    print(f"2026 evaluable weeks: {len(weeks)}")
    return weeks


def _context_and_actual(
    pdf: pd.DataFrame, five: list[pd.Timestamp], config: PatchTSTConfig
) -> tuple[np.ndarray, float, str]:
    week_start = five[0]
    hist = pdf[pdf.index < week_start]
    rets = _ohlcv_rets(hist)
    context = rets.iloc[-config.context_length :].values.astype(np.float32)
    full_rets = _ohlcv_rets(pdf)
    daily = [float(full_rets.loc[d, "close_ret"]) for d in five]
    actual_pct = float((np.exp(np.sum(daily)) - 1.0) * 100.0)
    return context, actual_pct, str(rets.index[-1].date())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== legacy halal universe ===")
    symbols = get_halal_symbols()
    print(f"symbols ({len(symbols)}): {symbols}")

    print(f"=== download {TRAIN_START} .. present ===")
    prices = _download_prices(symbols)
    symbols = sorted(prices.keys())
    print(f"usable symbols ({len(symbols)}): {symbols}")

    config = PatchTSTConfig()  # prod defaults: batch 256, lr 3e-4, patience 15

    print(f"=== build train set {TRAIN_START} .. {TRAIN_END} ===")
    X, y = _build_train_xy(prices, config)

    print("=== train close_only ===")
    model_c, meta_c = _train("close_only", "close_only", X, y, config)
    print("=== train multitask ===")
    model_m, meta_m = _train("multitask", "multitask", X, y, config)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model_c = model_c.to(device)
    model_m = model_m.to(device)

    print("=== walk-forward 2026 ===")
    weeks = _iter_2026_weeks(prices, config, symbols)
    rows: list[WeekScore] = []
    for week_start, week_end, five in weeks:
        for sym in symbols:
            ctx, actual, ctx_end = _context_and_actual(prices[sym], five, config)
            pc = _predict_weekly_close_pct(model_c, ctx, device)
            pm = _predict_weekly_close_pct(model_m, ctx, device)
            rows.append(
                WeekScore(
                    week_start=str(week_start.date()),
                    week_end=str(week_end.date()),
                    symbol=sym,
                    actual_weekly_pct=actual,
                    pred_close_only_pct=pc,
                    pred_multitask_pct=pm,
                    err_close_only=pc - actual,
                    err_multitask=pm - actual,
                    abs_err_close_only=abs(pc - actual),
                    abs_err_multitask=abs(pm - actual),
                    context_end=ctx_end,
                    n_context_days=int(ctx.shape[0]),
                )
            )
        print(f"  week {week_start.date()} .. {week_end.date()}")

    mae_c = float(np.mean([r.abs_err_close_only for r in rows]))
    mae_m = float(np.mean([r.abs_err_multitask for r in rows]))

    def dir_acc(attr: str) -> float:
        ok = sum(
            1
            for r in rows
            if (getattr(r, attr) >= 0) == (r.actual_weekly_pct >= 0)
            or (getattr(r, attr) == 0 and r.actual_weekly_pct == 0)
        )
        # simpler sign match
        ok = 0
        for r in rows:
            pred = getattr(r, attr)
            if (pred >= 0 and r.actual_weekly_pct >= 0) or (
                pred < 0 and r.actual_weekly_pct < 0
            ):
                ok += 1
        return ok / max(len(rows), 1)

    summary = {
        "universe": "halal",
        "symbols": symbols,
        "train_start": TRAIN_START.isoformat(),
        "train_end": TRAIN_END.isoformat(),
        "n_train_samples": int(X.shape[0]),
        "n_weeks": len(weeks),
        "n_rows": len(rows),
        "close_only_meta": meta_c,
        "multitask_meta": meta_m,
        "mae_close_only_pct": mae_c,
        "mae_multitask_pct": mae_m,
        "dir_acc_close_only": dir_acc("pred_close_only_pct"),
        "dir_acc_multitask": dir_acc("pred_multitask_pct"),
        "winner_mae": "close_only"
        if mae_c < mae_m
        else ("multitask" if mae_m < mae_c else "tie"),
        "caveat": (
            "If close_only best_epoch<=2, MAE comparison is against a barely-trained "
            "model (production pathology reproduced)."
        ),
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "weekly_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n"
    )
    with (RESULTS_DIR / "weekly_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "brain_api"))
    raise SystemExit(main())
