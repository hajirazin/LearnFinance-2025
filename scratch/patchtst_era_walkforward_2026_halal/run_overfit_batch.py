#!/usr/bin/env python3
"""Sanity: overfit a tiny fixed batch (Claude suggestion #1).

Settles pipeline bug vs unlearnable-task before more loss-scale work.

Arms (same 16 fixed OHLCV close-only samples from the halal cache):
  A) prod-like: lr=3e-4, weight_decay=1e-4, max_grad_norm=1.0, dropout=0.2
  B) memorize:  lr=1e-3, weight_decay=0,   no grad clip,       dropout=0
  C) memorize + wd=1e-4 (isolate decay)
  D) memorize + clip=1.0 (isolate clip)

Pass rule (B): train close-MSE drops to ~0 (final < 1e-8 or < 1e-3 * initial).
Also logs grad-norm before clip each step.

Artifacts: scratch/.../exp_overfit_batch/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.patchtst.config import PatchTSTConfig

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_overfit_batch"
RESULTS_DIR = OUT_DIR / "results"
PRICE_CACHE = BASE_DIR / "exp_e123" / "cache" / "prices.pkl"

SEED = 20260809
N_SAMPLES = 16
STEPS = 400
CLOSE_IDX = 3


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _ohlcv_rets(pdf: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(pdf, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


def _build_fixed_batch(
    prices: dict[str, pd.DataFrame], config: PatchTSTConfig, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """First n valid week-aligned OHLCV samples (deterministic order)."""
    xs, ys = [], []
    for sym in sorted(prices.keys()):
        rets = _ohlcv_rets(prices[sym])
        if len(rets) < config.context_length + 5:
            continue
        periods = rets.index.to_period("W")
        i, m = 0, len(rets)
        while i < m and len(xs) < n:
            p = periods[i]
            j = i + 1
            while j < m and periods[j] == p:
                j += 1
            if j - i >= config.min_week_days:
                t = j - 1
                if t >= config.context_length - 1 and t + 5 < m:
                    seq = rets.iloc[t - config.context_length + 1 : t + 1].values
                    tgt = rets.iloc[t + 1 : t + 6].values
                    if (
                        seq.shape == (config.context_length, 5)
                        and tgt.shape == (5, 5)
                        and not (
                            np.isnan(seq).any()
                            or np.isinf(seq).any()
                            or np.isnan(tgt).any()
                            or np.isinf(tgt).any()
                        )
                    ):
                        xs.append(seq.astype(np.float32))
                        ys.append(tgt.astype(np.float32))
            i = j
        if len(xs) >= n:
            break
    if len(xs) < n:
        raise RuntimeError(f"only got {len(xs)} samples, need {n}")
    X = torch.from_numpy(np.stack(xs[:n]))
    y = torch.from_numpy(np.stack(ys[:n]))
    return X, y


def _create_model(config: PatchTSTConfig, dropout: float) -> PatchTSTForPrediction:
    hf = HFPatchTSTConfig(
        num_input_channels=config.num_input_channels,
        context_length=config.context_length,
        patch_length=config.patch_length,
        stride=config.stride,
        d_model=config.d_model,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        ffn_dim=config.ffn_dim,
        dropout=dropout,
        prediction_length=config.prediction_length,
        attention_dropout=dropout,
        positional_dropout=dropout,
        use_cls_token=False,
        pooling_type="mean",
        # keep default RevIN (prod)
    )
    return PatchTSTForPrediction(hf)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item() ** 2)
    return total**0.5


def _run_arm(
    name: str,
    X: torch.Tensor,
    y: torch.Tensor,
    config: PatchTSTConfig,
    *,
    lr: float,
    weight_decay: float,
    max_grad_norm: float | None,
    dropout: float,
    steps: int,
    device: torch.device,
) -> dict:
    _set_seeds()
    model = _create_model(config, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xb, yb = X.to(device), y.to(device)

    history = []
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        preds = model(past_values=Xb).prediction_outputs
        loss = F.mse_loss(preds[:, :, CLOSE_IDX], yb[:, :, CLOSE_IDX])
        loss.backward()
        g_before = _grad_norm(model)
        g_after = g_before
        if max_grad_norm is not None:
            g_after = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            )
        opt.step()
        loss_f = float(loss.detach())
        if step == 1 or step % 25 == 0 or step == steps:
            with torch.no_grad():
                pred_std = float(preds[:, :, CLOSE_IDX].std().cpu())
                tgt_std = float(yb[:, :, CLOSE_IDX].std().cpu())
            history.append(
                {
                    "step": step,
                    "train_close_mse": loss_f,
                    "grad_norm_before_clip": g_before,
                    "grad_norm_after_clip": g_after,
                    "pred_close_std": pred_std,
                    "tgt_close_std": tgt_std,
                }
            )
            print(
                f"[{name}] step{step}: loss={loss_f:.6e} "
                f"grad={g_before:.6e}->{g_after:.6e} "
                f"pred_std={pred_std:.6e} tgt_std={tgt_std:.6e}"
            )

    init = history[0]["train_close_mse"]
    final = history[-1]["train_close_mse"]
    pass_abs = final < 1e-8
    pass_rel = final < init * 1e-3
    return {
        "name": name,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "dropout": dropout,
        "steps": steps,
        "n_samples": int(X.shape[0]),
        "initial_loss": init,
        "final_loss": final,
        "loss_ratio_final_over_init": final / init if init > 0 else None,
        "pass_abs_lt_1e-8": pass_abs,
        "pass_rel_lt_1e-3_of_init": pass_rel,
        "memorize_pass": pass_abs or pass_rel,
        "history": history,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_pickle(PRICE_CACHE)
    config = PatchTSTConfig()
    X, y = _build_fixed_batch(prices, config, N_SAMPLES)
    print(
        f"fixed batch X={tuple(X.shape)} y={tuple(y.shape)} "
        f"close_tgt mean={y[:, :, CLOSE_IDX].mean():.6e} "
        f"std={y[:, :, CLOSE_IDX].std():.6e}"
    )

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    arms = [
        dict(
            name="A_prod_like",
            lr=3e-4,
            weight_decay=1e-4,
            max_grad_norm=1.0,
            dropout=0.2,
        ),
        dict(
            name="B_memorize",
            lr=1e-3,
            weight_decay=0.0,
            max_grad_norm=None,
            dropout=0.0,
        ),
        dict(
            name="C_memorize_plus_wd",
            lr=1e-3,
            weight_decay=1e-4,
            max_grad_norm=None,
            dropout=0.0,
        ),
        dict(
            name="D_memorize_plus_clip",
            lr=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            dropout=0.0,
        ),
    ]

    results = []
    for a in arms:
        print(f"\n=== {a['name']} ===")
        results.append(
            _run_arm(
                a["name"],
                X,
                y,
                config,
                lr=a["lr"],
                weight_decay=a["weight_decay"],
                max_grad_norm=a["max_grad_norm"],
                dropout=a["dropout"],
                steps=STEPS,
                device=device,
            )
        )

    b = next(r for r in results if r["name"] == "B_memorize")
    verdict = {
        "pipeline_ok_can_memorize": b["memorize_pass"],
        "interpretation": (
            "PASS: model can drive tiny-batch train loss ~0 → plumbing OK; "
            "E1–E7 failures are optimization/generalization/SNR, not a broken loss wire."
            if b["memorize_pass"]
            else "FAIL: cannot memorize 16 samples → investigate grad flow / loss wiring / LR "
            "before more walk-forward ablations."
        ),
        "next_if_pass": [
            "weight_decay=0 + looser clip on full close-only train (Claude #2)",
            "listwise/pairwise ranking loss scored by rank IC (Claude #3)",
        ],
        "config_defaults_noted": {
            "weight_decay": 1e-4,
            "max_grad_norm": 1.0,
            "note": "decay is 1e-4 not 0.01; still worth ablating to 0 on full train",
        },
    }

    out = {"verdict": verdict, "arms": results}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n======== OVERFIT VERDICT ========")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0 if b["memorize_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
