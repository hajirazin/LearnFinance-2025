#!/usr/bin/env python3
"""E8: Weight-decay sweep on prod close-only (Claude #2), OOS-gated.

Sweep wd ∈ {0, 1e-5, 3e-5, 1e-4} with otherwise prod close-only hypers
(lr=3e-4, clip=1.0, dropout=0.2, RevIN on, OHLCV log-ret, close-MSE).

Per epoch logs:
  train/val close-MSE, grad_norm_before_clip, wd_param_norm=wd*||θ||,
  ratio = wd_param_norm / grad_norm (decay vs signal).

OOS gates (2026 halal walk-forward) — ALL required for PASS:
  1) best_epoch >> 2  (best_epoch >= 10)
  2) beats mean-close baseline on val (best_val < baseline)
  3) dir_acc materially > 50%  (dir_acc >= 0.55)
  4) beats naive predict-0 on weekly MAE

If no arm PASSes gate 4, script exits with code 3 and writes
  ranking_loss_go=true so #3 can proceed.
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
from brain_api.core.patchtst.config import PatchTSTConfig

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_e8_wd_sweep"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
PRICE_CACHE = BASE_DIR / "exp_e123" / "cache" / "prices.pkl"

TRAIN_START = date(2015, 1, 1)
TRAIN_END = date(2025, 12, 31)
EVAL_YEAR = 2026
SEED = 20260809
CLOSE_IDX = 3
PATIENCE = 15
MAX_EPOCHS = 100
LR = 3e-4
MAX_GRAD_NORM = 1.0
WD_SWEEP = [0.0, 1e-5, 3e-5, 1e-4]
BEST_EPOCH_MIN = 10
DIR_ACC_MIN = 0.55


@dataclass
class WeekRow:
    week_start: str
    symbol: str
    actual_pct: float
    pred_pct: float
    abs_err: float
    dir_correct: bool


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _ohlcv_rets(pdf: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(pdf, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


def _build_train_xy(
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
    X, y = np.stack(xs), np.stack(ys)
    print(f"train set X={X.shape}")
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


def _param_l2(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += float(p.data.norm(2).item() ** 2)
    return total**0.5


def _grad_l2(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item() ** 2)
    return total**0.5


def _mean_close_baseline(y_va: np.ndarray) -> float:
    close = y_va[:, :, CLOSE_IDX]
    mean = np.mean(close, axis=0, keepdims=True)
    return float(np.mean((close - mean) ** 2))


def _train_one(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    config: PatchTSTConfig,
    weight_decay: float,
) -> tuple[PatchTSTForPrediction, dict, list[dict]]:
    _set_seeds()
    split = int(len(X) * (1 - config.validation_split))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    baseline = _mean_close_baseline(y_va)

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
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history: list[dict] = []

    print(
        f"[{name}] wd={weight_decay} n_train={len(X_tr)} n_val={len(X_va)} "
        f"baseline={baseline:.6e} device={device}"
    )

    for epoch in range(config.epochs):
        model.train()
        tot, n = 0.0, 0
        g_sum, ratio_sum, wd_sum = 0.0, 0.0, 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(past_values=bx).prediction_outputs
            loss = F.mse_loss(preds[:, :, CLOSE_IDX], by[:, :, CLOSE_IDX])
            loss.backward()
            g_before = _grad_l2(model)
            p_norm = _param_l2(model)
            wd_term = weight_decay * p_norm
            ratio = (wd_term / g_before) if g_before > 1e-20 else float("inf")
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            tot += float(loss.detach())
            n += 1
            g_sum += g_before
            wd_sum += wd_term
            ratio_sum += ratio
        train_loss = tot / max(n, 1)
        mean_g = g_sum / max(n, 1)
        mean_wd = wd_sum / max(n, 1)
        mean_ratio = ratio_sum / max(n, 1)

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                preds = model(past_values=vx).prediction_outputs
                vloss = F.mse_loss(preds[:, :, CLOSE_IDX], vy[:, :, CLOSE_IDX])
                vtot += float(vloss.detach())
                vn += 1
        val_loss = vtot / max(vn, 1)
        sched.step(val_loss)

        row = {
            "epoch": epoch + 1,
            "train_close_mse": train_loss,
            "val_close_mse": val_loss,
            "beats_baseline": val_loss < baseline,
            "grad_norm_before_clip": mean_g,
            "wd_param_norm": mean_wd,
            "wd_over_grad_ratio": mean_ratio,
        }
        history.append(row)

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
                f"beat={val_loss < baseline} g={mean_g:.3e} "
                f"wd||θ||={mean_wd:.3e} ratio={mean_ratio:.3f} "
                f"pat={patience}/{PATIENCE} best={best_epoch}"
            )

        if patience >= PATIENCE:
            print(f"[{name}] early-stop ep{epoch + 1} best_ep={best_epoch}")
            break

    assert best_state is not None
    model_cpu = _create_model(config)
    model_cpu.load_state_dict(best_state)
    path = MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path / "weights.pt")
    meta = {
        "name": name,
        "weight_decay": weight_decay,
        "lr": LR,
        "max_grad_norm": MAX_GRAD_NORM,
        "best_epoch": best_epoch,
        "best_val_close_mse": best_val,
        "baseline_mse": baseline,
        "beats_baseline_best": best_val < baseline,
        "stopped_epoch": history[-1]["epoch"],
        "barely_trained": best_epoch <= 2,
        "mean_wd_over_grad_ratio_best_ep": next(
            (h["wd_over_grad_ratio"] for h in history if h["epoch"] == best_epoch),
            None,
        ),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[{name}] meta={json.dumps(meta)}")
    return model_cpu, meta, history


def _predict_weekly(
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


def _eval_oos(
    model: PatchTSTForPrediction,
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
    symbols: list[str],
) -> tuple[list[WeekRow], dict]:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)
    rows: list[WeekRow] = []
    for five in _iter_weeks(prices, symbols, config.context_length):
        ws = str(five[0].date())
        for sym in symbols:
            hist = prices[sym][prices[sym].index < five[0]]
            rets = _ohlcv_rets(hist)
            ctx = rets.iloc[-config.context_length :].values.astype(np.float32)
            full = _ohlcv_rets(prices[sym])
            daily = [float(full.loc[d, "close_ret"]) for d in five]
            actual = float((np.exp(np.sum(daily)) - 1.0) * 100.0)
            pred = _predict_weekly(model, ctx, device)
            rows.append(
                WeekRow(
                    week_start=ws,
                    symbol=sym,
                    actual_pct=actual,
                    pred_pct=pred,
                    abs_err=abs(pred - actual),
                    dir_correct=bool(
                        (np.sign(pred) == np.sign(actual))
                        if actual != 0
                        else (pred == 0)
                    ),
                )
            )
    actual = np.array([r.actual_pct for r in rows])
    pred = np.array([r.pred_pct for r in rows])
    mae = float(np.mean([r.abs_err for r in rows]))
    naive0 = float(np.mean(np.abs(actual)))
    nonzero = actual != 0
    dir_acc = float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero])))
    return rows, {
        "n_rows": len(rows),
        "mae_pp": mae,
        "naive0_mae_pp": naive0,
        "beats_naive0": mae < naive0,
        "dir_acc": dir_acc,
        "mean_pred_pp": float(np.mean(pred)),
        "std_pred_pp": float(np.std(pred)),
        "std_actual_pp": float(np.std(actual)),
        "corr": float(np.corrcoef(pred, actual)[0, 1]) if len(rows) > 1 else None,
    }


def _gate(meta: dict, oos: dict) -> dict:
    g1 = meta["best_epoch"] >= BEST_EPOCH_MIN
    g2 = bool(meta["beats_baseline_best"])
    g3 = oos["dir_acc"] >= DIR_ACC_MIN
    g4 = bool(oos["beats_naive0"])
    return {
        "best_epoch_ge_10": g1,
        "beats_val_mean_close_baseline": g2,
        "dir_acc_ge_0_55": g3,
        "beats_naive0_oos": g4,
        "pass_all": g1 and g2 and g3 and g4,
        "pass_train_dynamics": g1 and g2,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_pickle(PRICE_CACHE)
    symbols = sorted(prices.keys())
    print(f"symbols ({len(symbols)}): {symbols}")

    config = PatchTSTConfig()
    config.epochs = MAX_EPOCHS
    config.early_stopping_patience = PATIENCE
    config.learning_rate = LR
    config.max_grad_norm = MAX_GRAD_NORM

    X, y = _build_train_xy(prices, config)

    arm_summaries = []
    any_pass_all = False
    any_beats_naive0 = False

    for wd in WD_SWEEP:
        name = f"E8_wd_{wd:g}".replace(".", "p")
        print(f"\n======== {name} ========")
        model, meta, history = _train_one(name, X, y, config, wd)
        rows, oos = _eval_oos(model, prices, config, symbols)
        gates = _gate(meta, oos)
        any_pass_all = any_pass_all or gates["pass_all"]
        any_beats_naive0 = any_beats_naive0 or gates["beats_naive0_oos"]

        # ratio at late train
        late = history[-min(5, len(history)) :]
        summary = {
            "weight_decay": wd,
            "train": meta,
            "oos": oos,
            "gates": gates,
            "late_mean_wd_over_grad_ratio": float(
                np.mean([h["wd_over_grad_ratio"] for h in late])
            ),
            "early_mean_wd_over_grad_ratio": float(
                np.mean(
                    [h["wd_over_grad_ratio"] for h in history[: min(5, len(history))]]
                )
            ),
        }
        arm_summaries.append(summary)
        (RESULTS_DIR / f"{name}_oos_rows.json").write_text(
            json.dumps([asdict(r) for r in rows], indent=2) + "\n"
        )
        print(
            f"[{name}] OOS mae={oos['mae_pp']:.4f} naive0={oos['naive0_mae_pp']:.4f} "
            f"dir={oos['dir_acc']:.3f} gates={gates}"
        )

    # pick best by OOS mae among those with train dynamics pass, else best mae
    dyn = [a for a in arm_summaries if a["gates"]["pass_train_dynamics"]]
    pool = dyn if dyn else arm_summaries
    best = min(pool, key=lambda a: a["oos"]["mae_pp"])

    ranking_loss_go = not any_beats_naive0
    out = {
        "gates_definition": {
            "best_epoch_ge": BEST_EPOCH_MIN,
            "dir_acc_ge": DIR_ACC_MIN,
            "beats_val_baseline": True,
            "beats_naive0_oos": True,
        },
        "arms": arm_summaries,
        "best_arm": best["train"]["name"],
        "any_pass_all": any_pass_all,
        "any_beats_naive0": any_beats_naive0,
        "ranking_loss_go": ranking_loss_go,
        "recommendation": (
            "STOP — at least one wd arm PASSed all OOS gates; use that wd."
            if any_pass_all
            else (
                "GO #3 ranking loss — train dynamics may improve with lower wd, "
                "but no arm beats naive-0 OOS (pointwise MSE mean-regression remains)."
                if ranking_loss_go
                else "Partial — beats naive-0 but other gates failed; inspect arms."
            )
        ),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n======== E8 SUMMARY ========")
    print(json.dumps(out, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0 if any_pass_all else (3 if ranking_loss_go else 1)


if __name__ == "__main__":
    sys.exit(main())
