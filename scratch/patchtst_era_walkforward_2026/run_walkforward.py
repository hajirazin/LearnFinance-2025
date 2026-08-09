#!/usr/bin/env python3
"""Train close-only vs multi-task PatchTST (scratch) and walk-forward 2026 weeks.

- Does NOT touch data/models/ artifacts.
- 4 US stocks; train samples end on/before 2025-12-31.
- For each 2026 week: context = last 60 trading days of *actual* OHLCV log
  returns ending the day before the week starts (so week N gets week 1..N-1
  actuals in the window — yes, PatchTST sees true prior-week numbers).
- Predict next 5 trading-day close log-returns; compound to weekly %;
  compare to actual compounded close return that week.

Outputs under scratch/patchtst_era_walkforward_2026/.
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

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.patchtst.config import PatchTSTConfig

ROOT = Path(__file__).resolve().parents[2]  # repo root if under scratch/...
# Allow running from scratch/ directly
if not (ROOT / "brain_api").exists():
    ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "JNJ"]
TRAIN_END = date(2025, 12, 31)
EVAL_YEAR = 2026
SEED = 20260809
CLOSE_IDX = 3
PROD_LR = 3e-4
PATIENCE = 15
MAX_EPOCHS = 100
BATCH = 64  # small panel


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


def _download_prices() -> dict[str, pd.DataFrame]:
    # Need history for context before 2026 + all 2026 actuals for labels
    start = "2018-01-01"
    end = "2026-12-31"  # yfinance end is exclusive-ish; download past today is fine
    prices: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = yf.download(
            sym, start=start, end=end, progress=False, auto_adjust=False, threads=False
        )
        if df is None or len(df) == 0:
            raise RuntimeError(f"empty download {sym}")
        if getattr(df.columns, "nlevels", 1) > 1:
            df.columns = df.columns.get_level_values(0)
        need = ["Open", "High", "Low", "Close", "Volume"]
        out = df[need].copy()
        out.columns = ["open", "high", "low", "close", "volume"]
        out = out.dropna()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        prices[sym] = out
        print(
            f"  {sym}: {len(out)} rows  {out.index[0].date()} .. {out.index[-1].date()}"
        )
    return prices


def _ohlcv_rets(price_df: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(price_df, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


def _build_train_xy(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Week-aligned samples with target window entirely on/before TRAIN_END."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    train_end_ts = pd.Timestamp(TRAIN_END)

    for sym, pdf in prices.items():
        pdf = pdf[pdf.index <= train_end_ts]
        rets = _ohlcv_rets(pdf)
        if len(rets) < config.context_length + 5:
            continue
        # ISO-week last trading day anchors (same idea as current main)
        periods = rets.index.to_period("W")
        i = 0
        n = len(rets)
        while i < n:
            p = periods[i]
            j = i + 1
            while j < n and periods[j] == p:
                j += 1
            if j - i >= config.min_week_days:
                t = j - 1  # week-end index
                if t >= config.context_length - 1 and t + 5 < n:
                    # target days must be <= TRAIN_END (already filtered pdf)
                    seq = rets.iloc[t - config.context_length + 1 : t + 1].values
                    tgt = rets.iloc[t + 1 : t + 6].values
                    if seq.shape == (config.context_length, 5) and tgt.shape == (5, 5):
                        if not (np.isnan(seq).any() or np.isinf(seq).any()):
                            if not (np.isnan(tgt).any() or np.isinf(tgt).any()):
                                xs.append(seq.astype(np.float32))
                                ys.append(tgt.astype(np.float32))
            i = j
        print(f"  train samples from {sym}: running total {len(xs)}")

    if not xs:
        raise RuntimeError("no training samples")
    X = np.stack(xs)
    y = np.stack(ys)
    # chronological order approximate: already built in date order per symbol then concat
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
) -> PatchTSTForPrediction:
    """loss_mode: 'close_only' | 'multitask' (337a8af)."""
    _set_seeds()
    split = int(len(X) * (1 - config.validation_split))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=min(BATCH, len(X_tr)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=min(BATCH, max(len(X_va), 1)),
        shuffle=False,
    )

    model = _create_model(config).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=PROD_LR, weight_decay=config.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0

    print(
        f"[{name}] train {len(X_tr)} val {len(X_va)} device={device} mode={loss_mode}"
    )
    for epoch in range(MAX_EPOCHS):
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
                loss = out.loss  # all 5 channels (337a8af)
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

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience += 1
            if patience >= PATIENCE:
                print(
                    f"[{name}] early-stop ep{epoch + 1} best_ep={best_epoch} "
                    f"train={train_loss:.6e} val={val_loss:.6e}"
                )
                break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{name}] ep{epoch + 1}: train={train_loss:.6e} val={val_loss:.6e} "
                f"pat={patience}/{PATIENCE}"
            )

    assert best_state is not None
    model_cpu = _create_model(config)
    model_cpu.load_state_dict(best_state)
    path = MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path / "weights.pt")
    (path / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "loss_mode": loss_mode,
                "best_epoch": best_epoch,
                "best_val": best_val,
                "train_end": TRAIN_END.isoformat(),
                "symbols": SYMBOLS,
                "seed": SEED,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[{name}] saved {path} best_epoch={best_epoch} best_val={best_val:.6e}")
    return model_cpu


def _predict_weekly_close_pct(
    model: PatchTSTForPrediction,
    context: np.ndarray,
    device: torch.device,
) -> float:
    """context: (60, 5) actual OHLCV log rets; return compounded close weekly %."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(context[None, ...]).float().to(device)
        preds = model(past_values=x).prediction_outputs  # (1, 5, 5)
        daily = preds[0, :, CLOSE_IDX].cpu().numpy()
    return float((np.exp(np.sum(daily)) - 1.0) * 100.0)


def _iter_2026_weeks(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]]:
    """Return list of (week_start, week_end, list of 5 trading days) for EVAL_YEAR.

    Uses the intersection calendar of all symbols. Requires exactly 5 sessions
    in the ISO week (skip short weeks).
    """
    # Use AAPL calendar as reference (liquid)
    ref = prices[SYMBOLS[0]]
    year_mask = ref.index.year == EVAL_YEAR
    days = list(ref.index[year_mask])
    if not days:
        raise RuntimeError("no 2026 trading days in downloaded data")

    weeks: list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]] = []
    by_period: dict = {}
    for ts in days:
        p = ts.to_period("W")
        by_period.setdefault(p, []).append(ts)

    for _p, sessions in sorted(by_period.items(), key=lambda kv: kv[1][0]):
        if len(sessions) < 5:
            continue
        # Take first 5 trading days of the week as the prediction horizon
        five = sessions[:5]
        # Need full context before week start and 5 actual days available
        week_start = five[0]
        # Ensure every symbol has these 5 days
        ok = True
        for sym, pdf in prices.items():
            for d in five:
                if d not in pdf.index:
                    ok = False
                    break
            if not ok:
                break
            hist = pdf[pdf.index < week_start]
            if len(hist) < config.context_length + 1:
                ok = False
                break
        if not ok:
            continue
        weeks.append((five[0], five[-1], five))
    print(f"2026 evaluable weeks (5 sessions): {len(weeks)}")
    return weeks


def _context_and_actual(
    pdf: pd.DataFrame, five: list[pd.Timestamp], config: PatchTSTConfig
) -> tuple[np.ndarray, float, str]:
    """Build actual context ending day before week; actual weekly close %.

    Context uses *actual* history only (< week_start), so prior weeks' true
    OHLCV returns are visible to the model (teacher-forced walk-forward).
    """
    week_start = five[0]
    hist = pdf[pdf.index < week_start]
    rets = _ohlcv_rets(hist)
    if len(rets) < config.context_length:
        raise RuntimeError("insufficient context")
    context = rets.iloc[-config.context_length :].values.astype(np.float32)
    # Actual: compound close log rets over the 5 target sessions
    full_rets = _ohlcv_rets(pdf)
    daily = []
    for d in five:
        daily.append(float(full_rets.loc[d, "close_ret"]))
    actual_pct = float((np.exp(np.sum(daily)) - 1.0) * 100.0)
    ctx_end = str(rets.index[-1].date())
    return context, actual_pct, ctx_end


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== download ===")
    prices = _download_prices()
    config = PatchTSTConfig()
    config.batch_size = BATCH
    config.learning_rate = PROD_LR
    config.epochs = MAX_EPOCHS
    config.early_stopping_patience = PATIENCE

    print("=== build train set (targets <= 2025-12-31) ===")
    X, y = _build_train_xy(prices, config)

    print("=== train close_only (Phase A) ===")
    model_close = _train("close_only", "close_only", X, y, config)
    print("=== train multitask (337a8af) ===")
    model_mt = _train("multitask", "multitask", X, y, config)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model_close = model_close.to(device)
    model_mt = model_mt.to(device)

    print("=== walk-forward 2026 (actual prior weeks in context) ===")
    weeks = _iter_2026_weeks(prices, config)
    rows: list[WeekScore] = []

    for week_start, week_end, five in weeks:
        for sym in SYMBOLS:
            pdf = prices[sym]
            context, actual_pct, ctx_end = _context_and_actual(pdf, five, config)
            pred_c = _predict_weekly_close_pct(model_close, context, device)
            pred_m = _predict_weekly_close_pct(model_mt, context, device)
            rows.append(
                WeekScore(
                    week_start=str(week_start.date()),
                    week_end=str(week_end.date()),
                    symbol=sym,
                    actual_weekly_pct=actual_pct,
                    pred_close_only_pct=pred_c,
                    pred_multitask_pct=pred_m,
                    err_close_only=pred_c - actual_pct,
                    err_multitask=pred_m - actual_pct,
                    abs_err_close_only=abs(pred_c - actual_pct),
                    abs_err_multitask=abs(pred_m - actual_pct),
                    context_end=ctx_end,
                    n_context_days=int(context.shape[0]),
                )
            )
        print(
            f"  week {week_start.date()} .. {week_end.date()} "
            f"ctx_ends~{rows[-1].context_end} n={len(SYMBOLS)}"
        )

    # Summaries
    mae_c = float(np.mean([r.abs_err_close_only for r in rows]))
    mae_m = float(np.mean([r.abs_err_multitask for r in rows]))

    # Direction accuracy
    def dir_acc(pred_attr: str) -> float:
        ok = 0
        for r in rows:
            pred = getattr(r, pred_attr)
            if (pred >= 0 and r.actual_weekly_pct >= 0) or (
                pred < 0 and r.actual_weekly_pct < 0
            ):
                ok += 1
        return ok / max(len(rows), 1)

    # Rank corr proxy: per-week which symbol ranked higher — skip for brevity; MAE + dir
    summary = {
        "symbols": SYMBOLS,
        "train_end": TRAIN_END.isoformat(),
        "n_weeks": len(weeks),
        "n_rows": len(rows),
        "mae_close_only_pct": mae_c,
        "mae_multitask_pct": mae_m,
        "dir_acc_close_only": dir_acc("pred_close_only_pct"),
        "dir_acc_multitask": dir_acc("pred_multitask_pct"),
        "winner_mae": "close_only"
        if mae_c < mae_m
        else ("multitask" if mae_m < mae_c else "tie"),
        "note": (
            "Each week uses actual OHLCV log returns in the 60-day context ending "
            "the trading day before week_start (prior weeks teacher-forced)."
        ),
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "weekly_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n"
    )
    # CSV-ish
    import csv

    with (RESULTS_DIR / "weekly_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    # Ensure brain_api imports resolve
    sys.path.insert(0, str(ROOT / "brain_api"))
    raise SystemExit(main())
