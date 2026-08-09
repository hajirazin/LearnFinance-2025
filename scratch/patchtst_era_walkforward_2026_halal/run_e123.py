#!/usr/bin/env python3
"""Close-only experiments E1/E2/E3 on same halal 10y + 2026 WF harness.

Baseline JSON left untouched:
  scratch/patchtst_era_walkforward_2026_halal/results/weekly_pp_*.json

E1) lr=3e-4; early-stop only if patience>=15 AND val close MSE has beaten
    mean-close baseline at least once; else run all 100 epochs.
E2) lr=1e-3; original early-stop (patience>=15 regardless of baseline).
E3) E1 + E2 (lr=1e-3 + baseline-gated early-stop).

Artifacts under scratch/patchtst_era_walkforward_2026_halal/exp_e123/
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
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_e123"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
CACHE_DIR = OUT_DIR / "cache"

TRAIN_START = date(2015, 1, 1)
TRAIN_END = date(2025, 12, 31)
EVAL_YEAR = 2026
SEED = 20260809
CLOSE_IDX = 3
PATIENCE = 15
MAX_EPOCHS = 100


@dataclass
class WeekScore:
    week_start: str
    week_end: str
    symbol: str
    actual_weekly_pct: float
    pred_pct: float
    pp_error_actual_minus_pred: float
    abs_pp: float
    context_end: str


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _ohlcv_rets(price_df: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(price_df, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


def _download_or_load(symbols: list[str]) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "prices.pkl"
    if cache_path.exists():
        print(f"loading cached prices {cache_path}")
        obj = pd.read_pickle(cache_path)
        return {k: v for k, v in obj.items() if k in symbols}

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
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
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


def _mean_close_baseline(y_va: np.ndarray) -> float:
    close = y_va[:, :, CLOSE_IDX]
    mean = np.mean(close, axis=0, keepdims=True)
    return float(np.mean((close - mean) ** 2))


def _train_close_only(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    config: PatchTSTConfig,
    lr: float,
    require_beat_baseline_for_early_stop: bool,
) -> tuple[PatchTSTForPrediction, dict]:
    """Close-only denorm MSE.

    Early-stop rules:
    - baseline gated (E1/E3): stop only if patience>=15 AND val has beaten
      mean-close baseline at least once; else run all MAX_EPOCHS.
    - classic (E2): stop when patience>=15 (production rule).
    """
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
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    ever_beat_baseline = False
    history: list[dict] = []

    print(
        f"[{name}] lr={lr} gated_es={require_beat_baseline_for_early_stop} "
        f"n_train={len(X_tr)} n_val={len(X_va)} baseline={baseline:.6e} device={device}"
    )

    for epoch in range(config.epochs):
        model.train()
        tot = 0.0
        n = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(past_values=bx).prediction_outputs
            loss = F.mse_loss(preds[:, :, CLOSE_IDX], by[:, :, CLOSE_IDX])
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
                vloss = F.mse_loss(preds[:, :, CLOSE_IDX], vy[:, :, CLOSE_IDX])
                vtot += float(vloss.detach())
                vn += 1
        val_loss = vtot / max(vn, 1)
        sched.step(val_loss)

        beat_now = val_loss < baseline
        if beat_now:
            ever_beat_baseline = True

        history.append(
            {
                "epoch": epoch + 1,
                "train_close_mse": train_loss,
                "val_close_mse": val_loss,
                "beats_baseline": beat_now,
                "ever_beat_baseline": ever_beat_baseline,
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

        if (epoch + 1) % 5 == 0 or epoch == 0 or beat_now:
            print(
                f"[{name}] ep{epoch + 1}: train={train_loss:.6e} val={val_loss:.6e} "
                f"beat={beat_now} ever={ever_beat_baseline} "
                f"pat={patience}/{PATIENCE} best_ep={best_epoch}"
            )

        if patience >= PATIENCE:
            if require_beat_baseline_for_early_stop:
                if ever_beat_baseline:
                    print(
                        f"[{name}] early-stop ep{epoch + 1} "
                        f"(patience + ever beat baseline) best_ep={best_epoch}"
                    )
                    break
                # else: keep going toward 100
            else:
                print(
                    f"[{name}] early-stop ep{epoch + 1} (classic patience) "
                    f"best_ep={best_epoch}"
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
        "lr": lr,
        "require_beat_baseline_for_early_stop": require_beat_baseline_for_early_stop,
        "best_epoch": best_epoch,
        "best_val_close_mse": best_val,
        "baseline_mse": baseline,
        "beats_baseline_best": best_val < baseline,
        "ever_beat_baseline": ever_beat_baseline,
        "stopped_epoch": history[-1]["epoch"],
        "ran_full_100": history[-1]["epoch"] == MAX_EPOCHS,
        "barely_trained": best_epoch <= 2,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[{name}] meta={json.dumps(meta)}")
    return model_cpu, meta


def _predict_weekly(
    model: PatchTSTForPrediction, context: np.ndarray, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(context[None, ...]).float().to(device)
        daily = model(past_values=x).prediction_outputs[0, :, CLOSE_IDX].cpu().numpy()
    return float((np.exp(np.sum(daily)) - 1.0) * 100.0)


def _iter_weeks(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig, symbols: list[str]
):
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
            if len(pdf[pdf.index < week_start]) < config.context_length + 1:
                ok = False
                break
        if ok:
            weeks.append((five[0], five[-1], five))
    return weeks


def _context_actual(pdf, five, config):
    week_start = five[0]
    rets = _ohlcv_rets(pdf[pdf.index < week_start])
    ctx = rets.iloc[-config.context_length :].values.astype(np.float32)
    full = _ohlcv_rets(pdf)
    daily = [float(full.loc[d, "close_ret"]) for d in five]
    actual = float((np.exp(np.sum(daily)) - 1.0) * 100.0)
    return ctx, actual, str(rets.index[-1].date())


def _walkforward(name, model, prices, symbols, config, device):
    weeks = _iter_weeks(prices, config, symbols)
    rows: list[WeekScore] = []
    for ws, we, five in weeks:
        for sym in symbols:
            ctx, actual, cend = _context_actual(prices[sym], five, config)
            pred = _predict_weekly(model, ctx, device)
            pp = actual - pred
            rows.append(
                WeekScore(
                    week_start=str(ws.date()),
                    week_end=str(we.date()),
                    symbol=sym,
                    actual_weekly_pct=actual,
                    pred_pct=pred,
                    pp_error_actual_minus_pred=pp,
                    abs_pp=abs(pp),
                    context_end=cend,
                )
            )
    mae = float(np.mean([r.abs_pp for r in rows]))
    mean_signed = float(np.mean([r.pp_error_actual_minus_pred for r in rows]))
    acts = [r.actual_weekly_pct for r in rows]
    naive0 = float(np.mean([abs(a) for a in acts]))
    return rows, {
        "name": name,
        "n_weeks": len(weeks),
        "n_rows": len(rows),
        "mae_pp": mae,
        "mean_signed_pp_actual_minus_pred": mean_signed,
        "naive_predict0_mae_pp": naive0,
        "beats_naive0": mae < naive0,
    }


def _write_week_json(
    name: str, rows: list[WeekScore], summary: dict, meta: dict
) -> None:
    by_week: dict = {}
    for r in rows:
        by_week.setdefault((r.week_start, r.week_end), []).append(r)
    weeks_out = []
    for (ws, we), items in sorted(by_week.items()):
        weeks_out.append(
            {
                "week_start": ws,
                "week_end": we,
                "week_mae_pp": round(sum(i.abs_pp for i in items) / len(items), 4),
                "week_mean_signed_pp_actual_minus_pred": round(
                    sum(i.pp_error_actual_minus_pred for i in items) / len(items), 4
                ),
                "per_stock": {
                    i.symbol: {
                        "actual_pct": round(i.actual_weekly_pct, 4),
                        "pred_pct": round(i.pred_pct, 4),
                        "pp_actual_minus_pred": round(i.pp_error_actual_minus_pred, 4),
                        "abs_pp": round(i.abs_pp, 4),
                    }
                    for i in items
                },
            }
        )
    payload = {
        "experiment": name,
        "train_meta": meta,
        "overall": summary,
        "metric": "pp = actual_weekly_pct - pred_weekly_pct",
        "weeks": weeks_out,
    }
    path = RESULTS_DIR / f"{name}_weekly_pp.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with (RESULTS_DIR / f"{name}_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"wrote {path}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = get_halal_symbols()
    print("symbols", symbols)
    prices = _download_or_load(symbols)
    symbols = sorted(prices.keys())
    config = PatchTSTConfig()
    config.epochs = MAX_EPOCHS
    config.early_stopping_patience = PATIENCE

    X, y = _build_train_xy(prices, config)

    experiments = [
        ("E1_close_lr3e4_gatedES", 3e-4, True),
        ("E2_close_lr1e3_classicES", 1e-3, False),
        ("E3_close_lr1e3_gatedES", 1e-3, True),
    ]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    compare = {
        "baseline_ref": str(BASE_DIR / "results" / "weekly_pp_readable.json"),
        "baseline_note": "Prior close_only best_epoch=2, mae~5.08 vs naive0~4.86",
        "experiments": {},
    }

    for name, lr, gated in experiments:
        print(f"\n======== {name} ========")
        model, meta = _train_close_only(name, X, y, config, lr, gated)
        model = model.to(device)
        rows, summary = _walkforward(name, model, prices, symbols, config, device)
        _write_week_json(name, rows, summary, meta)
        compare["experiments"][name] = {"train": meta, "walkforward": summary}

    # Attach prior baseline summary numbers if present
    base_sum = BASE_DIR / "results" / "summary.json"
    if base_sum.exists():
        compare["prior_baseline_summary"] = json.loads(base_sum.read_text())

    out = RESULTS_DIR / "compare_e123_vs_baseline.json"
    out.write_text(json.dumps(compare, indent=2) + "\n")
    print("=== COMPARE ===")
    print(json.dumps(compare["experiments"], indent=2))
    print(f"wrote {out}")
    print(f"baseline JSON kept at {BASE_DIR / 'results'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "brain_api"))
    raise SystemExit(main())
