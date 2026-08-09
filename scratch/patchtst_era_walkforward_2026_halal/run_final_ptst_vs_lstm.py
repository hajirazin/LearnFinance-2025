#!/usr/bin/env python3
"""Final bake-off: best PatchTST (from E1–E10) vs current LSTM (unchanged).

Best PatchTST config (experiment-driven only):
  - Prod close-only OHLCV log-return MSE + RevIN (unchanged architecture)
  - weight_decay=0          ← E8 only proven train fix
  - lr=3e-4, clip=1.0, dropout=0.2, patience=15
  - NOT taken: RevIN-loss, gated ES, daily-frac, price-z, pct-z, XS-z, ListNet
    (none beat naive-0 / rank gates OOS)

LSTM: brain_api.core.lstm build_dataset + train_model_pytorch + run_inference
as-is (no hyperparameter edits).

Train window: 2015-01-01 .. 2025-12-31 on legacy 12-name halal cache.
Eval: 2026 weeks, week-by-week pred vs actual (both models).

Artifacts: scratch/.../exp_final_ptst_vs_lstm/
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

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import WeekBoundaries
from brain_api.core.lstm.config import LSTMConfig
from brain_api.core.lstm.dataset import build_dataset
from brain_api.core.lstm.inference import build_inference_features, run_inference
from brain_api.core.lstm.training import train_model_pytorch
from brain_api.core.patchtst.config import PatchTSTConfig

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_final_ptst_vs_lstm"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
PRICE_CACHE = BASE_DIR / "exp_e123" / "cache" / "prices.pkl"

TRAIN_START = date(2015, 1, 1)
TRAIN_END = date(2025, 12, 31)
EVAL_YEAR = 2026
SEED = 20260809
CLOSE_IDX = 3
PATIENCE_PTST = 15
MAX_EPOCHS = 100
# Best from E8
PTST_LR = 3e-4
PTST_WD = 0.0
PTST_CLIP = 1.0


@dataclass
class WeekRow:
    week_start: str
    week_end: str
    symbol: str
    actual_pct: float
    pred_ptst_pct: float
    pred_lstm_pct: float
    err_ptst: float
    err_lstm: float
    abs_err_ptst: float
    abs_err_lstm: float
    dir_ptst_ok: bool
    dir_lstm_ok: bool
    closer: str  # ptst | lstm | tie


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _ohlcv_rets(pdf: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(pdf, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


def _truncate_prices_end(
    prices: dict[str, pd.DataFrame], end: date
) -> dict[str, pd.DataFrame]:
    hi = pd.Timestamp(end)
    return {s: df[df.index <= hi].copy() for s, df in prices.items()}


def _build_ptst_xy(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    lo, hi = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    for _sym, pdf in prices.items():
        pdf = pdf[(pdf.index >= lo) & (pdf.index <= hi)]
        rets = _ohlcv_rets(pdf)
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
                    if seq.shape == (config.context_length, 5) and tgt.shape == (5, 5):
                        if not (
                            np.isnan(seq).any()
                            or np.isinf(seq).any()
                            or np.isnan(tgt).any()
                            or np.isinf(tgt).any()
                        ):
                            xs.append(seq.astype(np.float32))
                            ys.append(tgt.astype(np.float32))
            i = j
    return np.stack(xs), np.stack(ys)


def _create_ptst(config: PatchTSTConfig) -> PatchTSTForPrediction:
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


def _train_ptst(
    X: np.ndarray, y: np.ndarray, config: PatchTSTConfig
) -> tuple[PatchTSTForPrediction, dict]:
    _set_seeds()
    split = int(len(X) * (1 - config.validation_split))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    close_va = y_va[:, :, CLOSE_IDX]
    baseline = float(
        np.mean((close_va - np.mean(close_va, axis=0, keepdims=True)) ** 2)
    )

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

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

    model = _create_ptst(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=PTST_LR, weight_decay=PTST_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val, best_epoch, best_state, patience = float("inf"), 0, None, 0
    history = []
    print(
        f"[PTST] wd={PTST_WD} lr={PTST_LR} n_train={len(X_tr)} n_val={len(X_va)} "
        f"baseline={baseline:.6e} device={device}"
    )

    for epoch in range(config.epochs):
        model.train()
        tot, n = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(past_values=bx).prediction_outputs
            loss = F.mse_loss(preds[:, :, CLOSE_IDX], by[:, :, CLOSE_IDX])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PTST_CLIP)
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
                vtot += float(
                    F.mse_loss(preds[:, :, CLOSE_IDX], vy[:, :, CLOSE_IDX]).detach()
                )
                vn += 1
        val_loss = vtot / max(vn, 1)
        sched.step(val_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_close_mse": train_loss,
                "val_close_mse": val_loss,
                "beats_baseline": val_loss < baseline,
            }
        )
        if val_loss < best_val:
            best_val, best_epoch, patience = val_loss, epoch + 1, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[PTST] ep{epoch + 1}: train={train_loss:.6e} val={val_loss:.6e} "
                f"best={best_epoch} pat={patience}/{PATIENCE_PTST}"
            )
        if patience >= PATIENCE_PTST:
            print(f"[PTST] early-stop ep{epoch + 1} best_ep={best_epoch}")
            break

    assert best_state is not None
    model_cpu = _create_ptst(config)
    model_cpu.load_state_dict(best_state)
    meta = {
        "name": "PTST_best_from_E1_E10",
        "changes_from_prod": {
            "weight_decay": "1e-4 -> 0 (E8)",
            "unchanged": [
                "close-only MSE",
                "RevIN",
                "lr=3e-4",
                "clip=1.0",
                "dropout=0.2",
                "patience=15",
            ],
            "rejected": [
                "RevIN-normalized loss",
                "gated ES",
                "lr=1e-3",
                "daily frac",
                "z-score price",
                "z-score pct",
                "XS z-score",
                "ListNet",
            ],
        },
        "best_epoch": best_epoch,
        "best_val_close_mse": best_val,
        "baseline_mse": baseline,
        "beats_baseline": best_val < baseline,
        "stopped_epoch": history[-1]["epoch"],
        "weight_decay": PTST_WD,
        "lr": PTST_LR,
    }
    path = MODELS_DIR / "patchtst"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path / "weights.pt")
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model_cpu, meta


def _predict_ptst_weekly(
    model: PatchTSTForPrediction, context: np.ndarray, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(context[None, ...]).float().to(device)
        daily = model(past_values=x).prediction_outputs[0, :, CLOSE_IDX].cpu().numpy()
    return float((np.exp(np.sum(daily)) - 1.0) * 100.0)


def _iter_weeks(prices, symbols, ctx_len):
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
            and len(prices[s][prices[s].index < five[0]]) >= ctx_len + 1
            for s in symbols
        )
        if ok:
            weeks.append(five)
    return weeks


def _actual_weekly_pct(pdf: pd.DataFrame, five: list[pd.Timestamp]) -> float:
    full = _ohlcv_rets(pdf)
    daily = [float(full.loc[d, "close_ret"]) for d in five]
    return float((np.exp(np.sum(daily)) - 1.0) * 100.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prices_full = pd.read_pickle(PRICE_CACHE)
    symbols = sorted(prices_full.keys())
    print(f"symbols ({len(symbols)}): {symbols}")
    prices_train = _truncate_prices_end(prices_full, TRAIN_END)

    # --- PatchTST (best config) ---
    print("\n=== Train PatchTST (best from E1–E10: wd=0 close-only) ===")
    ptst_cfg = PatchTSTConfig()
    ptst_cfg.epochs = MAX_EPOCHS
    ptst_cfg.early_stopping_patience = PATIENCE_PTST
    ptst_cfg.learning_rate = PTST_LR
    ptst_cfg.weight_decay = PTST_WD
    X, y = _build_ptst_xy(prices_train, ptst_cfg)
    print(f"PTST train X={X.shape}")
    ptst_model, ptst_meta = _train_ptst(X, y, ptst_cfg)

    # --- LSTM (unchanged current code) ---
    print("\n=== Train LSTM (current code, no hyperparameter changes) ===")
    lstm_cfg = LSTMConfig()  # defaults as in repo
    ds = build_dataset(prices_train, lstm_cfg)
    print(f"LSTM train X={ds.X.shape} y={ds.y.shape}")
    lstm_result = train_model_pytorch(ds.X, ds.y, ds.feature_scaler, lstm_cfg)
    lstm_model = lstm_result.model
    lstm_scaler = lstm_result.feature_scaler
    lstm_path = MODELS_DIR / "lstm"
    lstm_path.mkdir(parents=True, exist_ok=True)
    torch.save(lstm_model.state_dict(), lstm_path / "weights.pt")
    lstm_meta = {
        "name": "LSTM_current_defaults",
        "config": lstm_cfg.to_dict(),
        "train_loss": lstm_result.train_loss,
        "val_loss": lstm_result.val_loss,
        "baseline_loss": lstm_result.baseline_loss,
        "note": "train_model_pytorch / build_dataset used as-is",
    }
    (lstm_path / "meta.json").write_text(json.dumps(lstm_meta, indent=2) + "\n")

    # --- 2026 walk-forward ---
    print("\n=== 2026 week-by-week compare ===")
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    ptst_model = ptst_model.to(device)
    # run_inference builds CPU tensors; keep LSTM on CPU (unchanged API)
    lstm_model = lstm_model.to("cpu")
    lstm_model.eval()

    weeks = _iter_weeks(prices_full, symbols, ptst_cfg.context_length)
    print(f"weeks={len(weeks)}")
    rows: list[WeekRow] = []

    for five in weeks:
        ws = str(five[0].date())
        we = str(five[-1].date())
        wb = WeekBoundaries(
            target_week_start=five[0].date(),
            target_week_end=five[-1].date(),
            calendar_monday=five[0].date(),
            calendar_friday=five[-1].date(),
        )
        # LSTM batch for this week
        feats = [
            build_inference_features(s, prices_full[s], lstm_cfg, five[0].date())
            for s in symbols
        ]
        lstm_preds = {
            p.symbol: p.predicted_weekly_return_pct
            for p in run_inference(lstm_model, lstm_scaler, feats, wb)
        }

        for sym in symbols:
            hist = prices_full[sym][prices_full[sym].index < five[0]]
            rets = _ohlcv_rets(hist)
            ctx = rets.iloc[-ptst_cfg.context_length :].values.astype(np.float32)
            actual = _actual_weekly_pct(prices_full[sym], five)
            ptst_p = _predict_ptst_weekly(ptst_model, ctx, device)
            lstm_p = lstm_preds.get(sym)
            if lstm_p is None:
                continue
            ae_p, ae_l = abs(ptst_p - actual), abs(lstm_p - actual)
            if ae_p < ae_l - 1e-12:
                closer = "ptst"
            elif ae_l < ae_p - 1e-12:
                closer = "lstm"
            else:
                closer = "tie"
            rows.append(
                WeekRow(
                    week_start=ws,
                    week_end=we,
                    symbol=sym,
                    actual_pct=actual,
                    pred_ptst_pct=ptst_p,
                    pred_lstm_pct=float(lstm_p),
                    err_ptst=ptst_p - actual,
                    err_lstm=float(lstm_p) - actual,
                    abs_err_ptst=ae_p,
                    abs_err_lstm=ae_l,
                    dir_ptst_ok=bool(
                        (np.sign(ptst_p) == np.sign(actual))
                        if actual != 0
                        else (ptst_p == 0)
                    ),
                    dir_lstm_ok=bool(
                        (np.sign(lstm_p) == np.sign(actual))
                        if actual != 0
                        else (lstm_p == 0)
                    ),
                    closer=closer,
                )
            )

    actual = np.array([r.actual_pct for r in rows])
    pp = np.array([r.pred_ptst_pct for r in rows])
    pl = np.array([r.pred_lstm_pct for r in rows])
    naive0 = float(np.mean(np.abs(actual)))
    mae_p = float(np.mean([r.abs_err_ptst for r in rows]))
    mae_l = float(np.mean([r.abs_err_lstm for r in rows]))
    nonzero = actual != 0
    dir_p = float(np.mean(np.sign(pp[nonzero]) == np.sign(actual[nonzero])))
    dir_l = float(np.mean(np.sign(pl[nonzero]) == np.sign(actual[nonzero])))
    n_closer = {
        "ptst": sum(1 for r in rows if r.closer == "ptst"),
        "lstm": sum(1 for r in rows if r.closer == "lstm"),
        "tie": sum(1 for r in rows if r.closer == "tie"),
    }

    # week aggregates
    by_week: dict[str, list[WeekRow]] = {}
    for r in rows:
        by_week.setdefault(r.week_start, []).append(r)
    week_summaries = []
    for ws in sorted(by_week):
        sub = by_week[ws]
        week_summaries.append(
            {
                "week_start": ws,
                "week_end": sub[0].week_end,
                "mae_ptst": round(float(np.mean([r.abs_err_ptst for r in sub])), 4),
                "mae_lstm": round(float(np.mean([r.abs_err_lstm for r in sub])), 4),
                "dir_ptst": round(float(np.mean([r.dir_ptst_ok for r in sub])), 4),
                "dir_lstm": round(float(np.mean([r.dir_lstm_ok for r in sub])), 4),
                "winner_mae": (
                    "ptst"
                    if np.mean([r.abs_err_ptst for r in sub])
                    < np.mean([r.abs_err_lstm for r in sub])
                    else "lstm"
                ),
                "symbols": [
                    {
                        "symbol": r.symbol,
                        "actual_pct": round(r.actual_pct, 4),
                        "pred_ptst_pct": round(r.pred_ptst_pct, 4),
                        "pred_lstm_pct": round(r.pred_lstm_pct, 4),
                        "abs_err_ptst": round(r.abs_err_ptst, 4),
                        "abs_err_lstm": round(r.abs_err_lstm, 4),
                        "closer": r.closer,
                    }
                    for r in sorted(sub, key=lambda x: x.symbol)
                ],
            }
        )

    overall = {
        "n_rows": len(rows),
        "n_weeks": len(by_week),
        "naive0_mae_pp": naive0,
        "ptst": {
            "mae_pp": mae_p,
            "beats_naive0": mae_p < naive0,
            "dir_acc": dir_p,
            "corr": float(np.corrcoef(pp, actual)[0, 1]),
            "std_pred": float(np.std(pp)),
            "meta": ptst_meta,
        },
        "lstm": {
            "mae_pp": mae_l,
            "beats_naive0": mae_l < naive0,
            "dir_acc": dir_l,
            "corr": float(np.corrcoef(pl, actual)[0, 1]),
            "std_pred": float(np.std(pl)),
            "meta": lstm_meta,
        },
        "head_to_head": {
            "closer_counts": n_closer,
            "ptst_win_rate": n_closer["ptst"] / max(len(rows), 1),
            "lstm_win_rate": n_closer["lstm"] / max(len(rows), 1),
            "lower_mae": "ptst" if mae_p < mae_l else "lstm",
            "weeks_ptst_better_mae": sum(
                1 for w in week_summaries if w["winner_mae"] == "ptst"
            ),
            "weeks_lstm_better_mae": sum(
                1 for w in week_summaries if w["winner_mae"] == "lstm"
            ),
        },
        "caveat": (
            "Neither is expected to beat research bar / naive-0; this is a "
            "fair same-slate same-window compare after applying only the E8 wd=0 fix."
        ),
    }

    out = {"overall": overall, "weeks": week_summaries}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS_DIR / "weekly_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n"
    )
    (RESULTS_DIR / "weekly_by_week.json").write_text(
        json.dumps(week_summaries, indent=2) + "\n"
    )

    print("\n======== FINAL PTST vs LSTM ========")
    print(json.dumps(overall, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
