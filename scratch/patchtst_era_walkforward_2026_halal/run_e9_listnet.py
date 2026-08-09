#!/usr/bin/env python3
"""E9: ListNet ranking loss (Claude #3) — Alpha-HRP objective alignment.

Triggered by E8: wd=0 fixes train dynamics but pointwise MSE still loses
to naive-0 OOS. Here we optimize cross-sectional *ordering*, not magnitudes.

Per week panel (all names with valid context):
  score_i = compounded 5-day predicted close log-return
  y_i     = compounded 5-day actual close log-return
  ListNet: KL(softmax(y/temp) || softmax(score/temp))

wd=0 (from E8), lr=3e-4, clip=1.0, dropout=0.2, RevIN on.

OOS metrics (gates):
  - mean Spearman IC > 0.05
  - top-3 hit rate > random (~3/12 choose = need > ~0.25 pairwise?);
    use: fraction of weeks where |pred_top3 ∩ actual_top3| / 3 > 1/3
    i.e. mean overlap/3 > random baseline 3*(3/12)=0.75 → hit@3 mean > 0.75
    simpler: mean IC and top6→HRP vs HRP sharpe from prior bake-off.
  - dir_acc >= 0.55 on weekly sign (secondary)
  - ranking_pass if mean IC > 0.05 AND beats prior close-only IC (~0.02)

Artifacts: scratch/.../exp_e9_listnet/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.hrp import compute_hrp_allocation
from brain_api.core.patchtst.config import PatchTSTConfig

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "scratch" / "patchtst_era_walkforward_2026_halal"
OUT_DIR = BASE_DIR / "exp_e9_listnet"
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
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
LISTNET_TEMP = 0.02  # scale on return fractions (~2% typical move)
TOP_K = 6
HRP_LOOKBACK = 252
IC_PASS = 0.05


@dataclass
class WeekPanel:
    week_start: pd.Timestamp
    five: list[pd.Timestamp]
    symbols: list[str]
    contexts: np.ndarray  # (n, 60, 5)
    actual_weekly_frac: np.ndarray  # (n,)


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _ohlcv_rets(pdf: pd.DataFrame) -> pd.DataFrame:
    return compute_ohlcv_log_returns(pdf, use_returns=True)[
        ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ]


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


def _compound_close_log(daily_close_log: np.ndarray) -> float:
    return float(np.exp(np.sum(daily_close_log)) - 1.0)


def _build_week_panels(
    prices: dict[str, pd.DataFrame],
    symbols: list[str],
    config: PatchTSTConfig,
    start: date,
    end: date | None,
) -> list[WeekPanel]:
    ref = prices[symbols[0]]
    if end is None:
        days = list(ref.index[ref.index >= pd.Timestamp(start)])
    else:
        days = list(
            ref.index[
                (ref.index >= pd.Timestamp(start)) & (ref.index <= pd.Timestamp(end))
            ]
        )
    by_period: dict = {}
    for ts in days:
        by_period.setdefault(ts.to_period("W"), []).append(ts)

    panels: list[WeekPanel] = []
    for _p, sessions in sorted(by_period.items(), key=lambda kv: kv[1][0]):
        if len(sessions) < 5:
            continue
        five = sessions[:5]
        week_start = five[0]
        ctxs, acts, syms = [], [], []
        for sym in symbols:
            pdf = prices[sym]
            if any(d not in pdf.index for d in five):
                continue
            hist = pdf[pdf.index < week_start]
            if len(hist) < config.context_length + 1:
                continue
            rets = _ohlcv_rets(hist)
            if len(rets) < config.context_length:
                continue
            ctx = rets.iloc[-config.context_length :].values.astype(np.float32)
            full = _ohlcv_rets(pdf)
            daily = np.array(
                [float(full.loc[d, "close_ret"]) for d in five], dtype=np.float64
            )
            if np.isnan(ctx).any() or np.isnan(daily).any():
                continue
            ctxs.append(ctx)
            acts.append(_compound_close_log(daily))
            syms.append(sym)
        if len(syms) < 4:
            continue
        panels.append(
            WeekPanel(
                week_start=week_start,
                five=five,
                symbols=syms,
                contexts=np.stack(ctxs),
                actual_weekly_frac=np.array(acts, dtype=np.float64),
            )
        )
    return panels


def _listnet_loss(
    scores: torch.Tensor, targets: torch.Tensor, temp: float
) -> torch.Tensor:
    """scores, targets: (n,) — ListNet cross-entropy with temperature."""
    s = scores / temp
    t = targets / temp
    log_p = F.log_softmax(s, dim=0)
    q = F.softmax(t, dim=0)
    return -(q * log_p).sum()


def _scores_from_model(
    model: PatchTSTForPrediction, contexts: torch.Tensor
) -> torch.Tensor:
    """contexts (n, 60, 5) -> weekly compound frac scores (n,)."""
    preds = model(past_values=contexts).prediction_outputs  # (n, 5, 5)
    daily = preds[:, :, CLOSE_IDX]  # (n, 5)
    return torch.exp(daily.sum(dim=1)) - 1.0


def _train(
    name: str,
    train_panels: list[WeekPanel],
    val_panels: list[WeekPanel],
    config: PatchTSTConfig,
) -> tuple[PatchTSTForPrediction, dict]:
    _set_seeds()
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model = _create_model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history: list[dict] = []

    print(
        f"[{name}] n_train_weeks={len(train_panels)} n_val_weeks={len(val_panels)} "
        f"wd={WEIGHT_DECAY} device={device}"
    )

    for epoch in range(config.epochs):
        model.train()
        order = np.random.permutation(len(train_panels))
        tot, n = 0.0, 0
        for idx in order:
            panel = train_panels[int(idx)]
            x = torch.from_numpy(panel.contexts).to(device)
            y = torch.from_numpy(panel.actual_weekly_frac.astype(np.float32)).to(device)
            opt.zero_grad()
            scores = _scores_from_model(model, x)
            loss = _listnet_loss(scores, y, LISTNET_TEMP)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            tot += float(loss.detach())
            n += 1
        train_loss = tot / max(n, 1)

        model.eval()
        vtot, vn = 0.0, 0
        ics = []
        with torch.no_grad():
            for panel in val_panels:
                x = torch.from_numpy(panel.contexts).to(device)
                y = torch.from_numpy(panel.actual_weekly_frac.astype(np.float32)).to(
                    device
                )
                scores = _scores_from_model(model, x)
                vtot += float(_listnet_loss(scores, y, LISTNET_TEMP).detach())
                vn += 1
                sc = scores.cpu().numpy()
                yt = panel.actual_weekly_frac
                rho, _ = spearmanr(sc, yt)
                if np.isfinite(rho):
                    ics.append(float(rho))
        val_loss = vtot / max(vn, 1)
        val_ic = float(np.mean(ics)) if ics else 0.0
        sched.step(val_loss)

        history.append(
            {
                "epoch": epoch + 1,
                "train_listnet": train_loss,
                "val_listnet": val_loss,
                "val_spearman_ic": val_ic,
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
                f"[{name}] ep{epoch + 1}: train={train_loss:.4f} val={val_loss:.4f} "
                f"val_ic={val_ic:+.3f} pat={patience}/{PATIENCE} best={best_epoch}"
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
        "loss": "listnet_weekly_close",
        "weight_decay": WEIGHT_DECAY,
        "listnet_temp": LISTNET_TEMP,
        "best_epoch": best_epoch,
        "best_val_listnet": best_val,
        "stopped_epoch": history[-1]["epoch"],
        "barely_trained": best_epoch <= 2,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[{name}] meta={json.dumps(meta)}")
    return model_cpu, meta


def _truncate_prices(prices, as_of):
    return {
        s: df[df.index < as_of] for s, df in prices.items() if len(df[df.index < as_of])
    }


def _asset_week_ret(pdf, five):
    prior = pdf[pdf.index < five[0]]["close"]
    start = float(prior.iloc[-1]) if len(prior) else float(pdf.loc[five[0], "close"])
    return float(pdf.loc[five[-1], "close"]) / start - 1.0


def _hrp_weights(prices_asof, symbols, as_of: date):
    subset = {s: prices_asof[s] for s in symbols if s in prices_asof}
    res = compute_hrp_allocation(
        subset, lookback_days=HRP_LOOKBACK, min_data_days=60, as_of_date=as_of
    )
    w = {s: v / 100.0 for s, v in res.percentage_weights.items()}
    if not w:
        return {s: 1.0 / len(symbols) for s in symbols}
    ssum = sum(w.values())
    return {k: v / ssum for k, v in w.items()}


def _metrics(weekly_rets: list[float]) -> dict:
    r = np.array(weekly_rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(52) if sd > 1e-12 else float("nan")
    wealth = np.cumprod(1.0 + r)
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "cum_return_pct": cum * 100,
        "sharpe_ann_approx": float(sharpe),
        "hit_rate": float(np.mean(r > 0)),
        "max_drawdown_pct": float(np.min(dd) * 100),
    }


def _eval(
    model: PatchTSTForPrediction,
    panels: list[WeekPanel],
    prices: dict[str, pd.DataFrame],
) -> dict:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)
    model.eval()

    ics, top3_overlaps, top6_overlaps = [], [], []
    score_rows = []
    arm_rets = {"HRP-252": [], "E9_top6_HRP": [], "E9_top6_EW": [], "EW-all": []}

    with torch.no_grad():
        for panel in panels:
            x = torch.from_numpy(panel.contexts).to(device)
            scores = _scores_from_model(model, x).cpu().numpy()
            actual = panel.actual_weekly_frac
            rho, _ = spearmanr(scores, actual)
            if np.isfinite(rho):
                ics.append(float(rho))

            order_p = np.argsort(-scores)
            order_a = np.argsort(-actual)
            top3_p = set(np.array(panel.symbols)[order_p[:3]])
            top3_a = set(np.array(panel.symbols)[order_a[:3]])
            top6_p = set(np.array(panel.symbols)[order_p[:TOP_K]])
            top6_a = set(np.array(panel.symbols)[order_a[:TOP_K]])
            top3_overlaps.append(len(top3_p & top3_a) / 3.0)
            top6_overlaps.append(len(top6_p & top6_a) / float(TOP_K))

            ws = str(panel.week_start.date())
            for i, sym in enumerate(panel.symbols):
                score_rows.append(
                    {
                        "week_start": ws,
                        "symbol": sym,
                        "pred_weekly_frac": float(scores[i]),
                        "actual_weekly_frac": float(actual[i]),
                    }
                )

            # portfolio
            asset_rets = {
                s: _asset_week_ret(prices[s], panel.five) for s in panel.symbols
            }
            prices_asof = _truncate_prices(prices, panel.week_start)
            as_of = panel.week_start.date()
            top = [panel.symbols[i] for i in order_p[:TOP_K]]
            all_syms = panel.symbols
            arms = {
                "EW-all": {s: 1.0 / len(all_syms) for s in all_syms},
                "HRP-252": _hrp_weights(prices_asof, all_syms, as_of),
                "E9_top6_HRP": _hrp_weights(prices_asof, top, as_of),
                "E9_top6_EW": {s: 1.0 / TOP_K for s in top},
            }
            for arm, w in arms.items():
                arm_rets[arm].append(
                    float(sum(w.get(s, 0.0) * asset_rets[s] for s in w))
                )

    # random top3 overlap expectation ≈ 3*(3/n)/3 = 3/n
    n_avg = float(np.mean([len(p.symbols) for p in panels]))
    random_top3 = 3.0 / n_avg
    random_top6 = TOP_K / n_avg

    mean_ic = float(np.mean(ics)) if ics else 0.0
    arms_m = {a: _metrics(rs) for a, rs in arm_rets.items()}
    hrp_s = arms_m["HRP-252"]["sharpe_ann_approx"]

    gates = {
        "mean_ic_gt_0_05": mean_ic > IC_PASS,
        "top3_overlap_gt_random": float(np.mean(top3_overlaps)) > random_top3,
        "beats_hrp_sharpe": arms_m["E9_top6_HRP"]["sharpe_ann_approx"] > hrp_s,
        "pass_ranking": mean_ic > IC_PASS
        and float(np.mean(top3_overlaps)) > random_top3,
    }

    return {
        "n_weeks": len(panels),
        "mean_spearman_ic": mean_ic,
        "std_spearman_ic": float(np.std(ics)) if ics else None,
        "mean_top3_overlap": float(np.mean(top3_overlaps)),
        "mean_top6_overlap": float(np.mean(top6_overlaps)),
        "random_top3_overlap": random_top3,
        "random_top6_overlap": random_top6,
        "arms": arms_m,
        "gates": gates,
        "score_rows": score_rows,
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
    config.weight_decay = WEIGHT_DECAY

    print("=== build week panels ===")
    train_panels = _build_week_panels(prices, symbols, config, TRAIN_START, TRAIN_END)
    # val = last 20% of train panels by time
    split = int(len(train_panels) * 0.8)
    tr, va = train_panels[:split], train_panels[split:]
    oos_panels = _build_week_panels(
        prices, symbols, config, date(EVAL_YEAR, 1, 1), None
    )
    # only 2026
    oos_panels = [p for p in oos_panels if p.week_start.year == EVAL_YEAR]
    print(f"train weeks={len(tr)} val weeks={len(va)} oos weeks={len(oos_panels)}")

    name = "E9_listnet_wd0"
    model, meta = _train(name, tr, va, config)
    oos = _eval(model, oos_panels, prices)

    # compare to E8 / prior IC
    prior_ic = None
    e8 = BASE_DIR / "exp_ptst_vs_hrp" / "results" / "summary.json"
    if e8.exists():
        prior_ic = json.loads(e8.read_text())["rank_ic"].get("close_only_spearman_mean")

    score_rows = oos.pop("score_rows")
    out = {
        "train": meta,
        "oos": oos,
        "prior_close_only_ic": prior_ic,
        "ic_lift_vs_prior": (
            None if prior_ic is None else oos["mean_spearman_ic"] - prior_ic
        ),
        "note": (
            "ListNet on weekly compounded close log-returns; wd=0 from E8. "
            "Gates: mean IC>0.05 and top3 overlap > random."
        ),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS_DIR / "weekly_scores.json").write_text(
        json.dumps(score_rows, indent=2) + "\n"
    )

    print("\n======== E9 SUMMARY ========")
    print(json.dumps(out, indent=2))
    print(f"wrote {RESULTS_DIR}")
    return 0 if oos["gates"]["pass_ranking"] else 3


if __name__ == "__main__":
    sys.exit(main())
