# Why PatchTST forecast a positive return for MSTR

Frozen model: `v2026-11-27-611c30dfacf0`. Input anchor: 2026-11-27.
The cached selection score was +0.70690745%;
the reconstructed score is +0.70689517%, a difference
of -0.00001228 percentage points.

## Method

Use the production feature builder with point-in-time news and the frozen
artifact's fitted target scale. Run inference on all 128 combinations of the
seven feature channels. A neutralized channel is replaced by zeros after the
original within-window normalization: a flat channel at its own context mean.
The original normalization and all other channels are held fixed. Exact group
Shapley values average marginal forecast differences over every channel order.
No model is retrained. These are channel-history attributions, not attributions
to just the final day's value.

| Feature history | Shapley contribution, percentage points | Forecast when this channel alone is neutralized |
| --- | ---: | ---: |
| momentum_20d | +0.224438 | +0.482457% |
| close | +0.103368 | +0.603527% |
| volume | +0.054534 | +0.652361% |
| relative_volume_20d | +0.040549 | +0.666346% |
| news_sentiment | +0.007466 | +0.699430% |
| momentum_12_1 | -0.026719 | +0.733615% |
| volatility_20d | -0.058408 | +0.765303% |

All-neutral input baseline: +0.361668%.
The seven contributions sum to the actual forecast minus that baseline.
The latest 20-session momentum was -42.37%, yet its full history contributes
positively to this prediction. This is consistent with a rebound-style response;
it does not establish that a fall itself causes the positive prediction.

News contributes only about +0.0075 percentage points; neutralizing its history
leaves the forecast at approximately +0.6994%.

## Limits and evidence

The zero-channel baselines are synthetic and can be outside the training
input distribution. Price-derived features are correlated; attribution depends
on the chosen baseline and is not a causal explanation or a statement of
accuracy. A positive forecast is not confidence that a rebound will occur.
The reconstruction uses currently available provider evidence, not a frozen
copy of the original request's feature tensor; the small score difference is
reported above. Price data and the exact input matrix are saved under `data/`;
`result.json` records price and model-weight hashes.

Two offline synthetic tests passed: exact recovery of additive effects and
equal sharing of a joint interaction. Ruff lint and formatting pass. This run
read model artifacts and news evidence without changing them. It did not modify
production code, start/restart training, or resume monitoring of the SAC job.
