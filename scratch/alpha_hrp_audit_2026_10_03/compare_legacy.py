"""Counterfactual prior checkpoint; multiple architecture/training changes confound channel causality."""

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import PatchTSTConfig as HFConfig
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
torch.set_num_threads(4)
prices = pickle.loads((OUT / "prices.pkl").read_bytes())
modeldir = ROOT / "brain_api/data/models/patchtst_halal_new/v2026-08-14-e7b14b211a54"
raw = json.loads((modeldir / "config.json").read_text())
hf = HFConfig(
    **{
        k: raw[k]
        for k in [
            "num_input_channels",
            "context_length",
            "patch_length",
            "stride",
            "d_model",
            "num_attention_heads",
            "num_hidden_layers",
            "ffn_dim",
            "dropout",
            "prediction_length",
        ]
    },
    attention_dropout=raw["dropout"],
    positional_dropout=raw["dropout"],
    use_cls_token=False,
    pooling_type="mean",
)
model = PatchTSTForPrediction(hf)
model.load_state_dict(
    torch.load(modeldir / "weights.pt", weights_only=True, map_location="cpu")
)
model.eval()
audit = json.loads((OUT / "analysis.json").read_text())
rows = []
for row, w in zip(audit["rows"], audit["weeks"]):
    monday = date.fromisoformat(row["week"])
    syms = []
    x = []
    ys = []
    for s in w["full_recomputed_scores"]:
        df = prices[s]
        before = df[df.index.date < monday]
        f = compute_ohlcv_log_returns(before)[raw["feature_names"]].tail(60).values
        if f.shape != (60, 5) or not np.isfinite(f).all():
            continue
        syms.append(s)
        x.append(f)
        ys.append(w["full_actual_next5"][s])
    tensor = torch.tensor(np.array(x), dtype=torch.float32)
    with torch.no_grad():
        pred = model(past_values=tensor).prediction_outputs[:, :, 3].numpy()
        scores = np.expm1(pred.sum(axis=1)) * 100
    top = np.argsort(-scores)[:15]
    rows.append(
        {
            "week": str(monday),
            "old5channel_rank_ic_next5": float(spearmanr(scores, ys).statistic),
            "old5channel_top15_return_next5": float(np.mean(np.array(ys)[top])),
            "old5channel_top15_symbols": [syms[i] for i in top],
        }
    )
    if not rows[:-1]:
        changed = tensor.clone()
        changed[:, :, [0, 1, 2, 4]] *= 7
        with torch.no_grad():
            alt = model(past_values=changed).prediction_outputs[:, :, 3].numpy()
        zero = float(abs(pred - alt).max())
out = {
    "old_model": "v2026-08-14-e7b14b211a54",
    "effective_patch_stride": hf.patch_stride,
    "nonclose_input_perturbation_max_close_output_delta": zero,
    "rows": rows,
    "mean_rank_ic": float(np.mean([r["old5channel_rank_ic_next5"] for r in rows])),
}
(OUT / "legacy_comparison.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
