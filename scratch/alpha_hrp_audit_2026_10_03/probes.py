"""Reproduce inference safeguards, legacy load, and August 31 discrepancy."""

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from transformers import PatchTSTConfig as HFConfig
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst.inference import build_inference_features
from brain_api.storage.patchtst.local import PatchTSTModelStorage

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
torch.set_num_threads(4)
prices = pickle.loads((OUT / "prices.pkl").read_bytes())
storage = PatchTSTModelStorage(ROOT / "brain_api/data")
art = storage.load_current_artifacts()
e = json.loads((OUT / "analysis.json").read_text())["weeks"][1]["email_top30_scores"]
probe = []
for days in range(-10, 11):
    cutoff = date(2026, 8, 31) + timedelta(days=days)
    features = [
        build_inference_features(s, prices[s], art.config, cutoff).features for s in e
    ]
    with torch.no_grad():
        y = (
            art.model(past_values=torch.tensor(np.array(features), dtype=torch.float32))
            .prediction_outputs[:, :, 0]
            .numpy()
        )
    scores = np.expm1(y.sum(axis=1)) * 100
    probe.append(
        {
            "cutoff": str(cutoff),
            "median_error_pp": float(
                np.median(abs(scores - np.array(list(e.values()))))
            ),
        }
    )
print("Cutoff probes", sorted(probe, key=lambda r: r["median_error_pp"])[:7])
out = {"cutoff_probes": probe}
try:
    storage.load_model("v2026-08-14-e7b14b211a54")
    out["legacy_current_loader"] = "loaded"
except Exception as exc:
    out["legacy_current_loader"] = str(exc)
print("Legacy loader", out["legacy_current_loader"][:1500])
raw = json.loads(
    (
        ROOT
        / "brain_api/data/models/patchtst_halal_new/v2026-08-14-e7b14b211a54/config.json"
    ).read_text()
)
legacy = HFConfig(
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
old = PatchTSTForPrediction(legacy)
old.load_state_dict(
    torch.load(
        ROOT
        / "brain_api/data/models/patchtst_halal_new/v2026-08-14-e7b14b211a54/weights.pt",
        weights_only=True,
        map_location="cpu",
    )
)
old.eval()
out["legacy_effective_stride"] = legacy.patch_stride
out["legacy_reconstructed"] = "loaded strict state_dict successfully"
out["current_effective_stride"] = art.model.config.patch_stride
df = prices["IBM"]
stale = df[df.index.date < date(2026, 9, 18)]
f = build_inference_features("IBM", stale, art.config, date(2026, 9, 28))
out["stale_input_accepted"] = {
    "has_enough_history": f.has_enough_history,
    "data_end_date": str(f.data_end_date),
    "requested_cutoff": "2026-09-28",
}
gapped = df.drop(df.index[-20:-15])
f = build_inference_features("IBM", gapped, art.config, date(2026, 10, 3))
out["missing5_sessions_accepted"] = {
    "has_enough_history": f.has_enough_history,
    "shape": list(f.features.shape),
}
out["same_version_seed"] = None
(OUT / "probes.json").write_text(json.dumps(out, indent=2))
print(
    json.dumps(
        {
            k: v
            for k, v in out.items()
            if k not in ["cutoff_probes", "legacy_current_loader"]
        },
        indent=2,
    )
)
