# E2: weekly PatchTST, MSE versus Huber

Frozen November 28, 2026, before fitting either experimental arm.
The user authorized starting E2 directly. This bounded screening study performs
the evidence and implementation checks needed here, without claiming completion
of the roadmap's broader A0–A4 studies.

## Question and controlled change

Does limiting large residuals' gradient contribution improve later weekly
cross-sectional ranking? Control: the current US seven-channel weekly trainer's
MSE objective. Challenger: twice PyTorch Huber with raw-return delta 0.05,
converted to standardized units using fitting-fold target standard deviation.
The factor two matches MSE's value and gradient inside the quadratic region;
outside it the gradient is capped. This is a residual cutoff, not clipping of
returns or inputs. Source: [PyTorch HuberLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.HuberLoss.html).

Both arms use the same seed 42, initial parameters, shuffled batches, features,
unmodified labels, eligibility, train/validation dates, Adam at 0.0003, batch
256, gradient clipping 1, 60-epoch cap, patience 15, and plateau scheduler
(factor 0.5, patience 5, driven by validation MSE). Checkpoint selection and early
stopping use **validation MSE in both arms**, preserving the actual weekly
production contract. Seven normalized input channels, 365-calendar-day context
padded to 255 sessions, 10-session patches and stride 5 remain unchanged.
This newly fitted patience-15 control is not a reproduction of the old
patience-8 weights. No annual snapshot retrains or production writes occur.

## Frozen evidence and chronology

Use only existing adjusted OHLCV CSVs and their SHA-256 manifest in
`scratch/patchtst_geometry_sweep_2026_08_23/data/`. Freeze the roster in
`brain_api/data/models/patchtst_halal_new/v2026-11-20-549d9adc9504/metadata.json`.
Use available roster members, excluding the 13 already documented ticker
identity collisions in `scratch/AlphaAllocationV2/ROLLING_MAD_EXPERIMENT.md`.
Record every exclusion. Do not introduce a listing-age or volatility screen.
The corrected common cohort applies identically to both newly fitted arms.

Read historical sentiment using the production reader and strictly read-only
DuckDB connections to `brain_api/data/news/news_v1.duckdb`. Use strict
`point_in_time` revision policy; missing evidence stays missing and verified
zero-news remains zero. Persist the extracted feature evidence, its digest,
database stat information and query policy in scratch. Abort on database
contention, changed sources during extraction, or inadequate coverage. Do not
copy/backup the database or fetch news. The current roster and retrospective
price adjustments entail survivorship and revision limitations.

Construct features and five-session simple-return labels with production
`WeeklyPatchTSTInputs`. Fitting/inner-validation panel: anchors from 2017-01-01
through 2024-12-31 with labels ending by 2024-12-31. Preserve proportional
`validation_split=0.2`, at least 13 validation weeks, and purge fitting labels
ending on/after the validation boundary using `weekly_fold_indices`.
Outer evaluation: anchors on/after 2025-01-01 whose labels end by 2025-12-31.
Log the actual first/last dates and row/week/symbol counts after eligibility.
Friday holidays use the exchange week's last session; labels always span five
exchange sessions, so this is not always calendar Friday-to-Friday.
2025 has been inspected historically: this is reused historical evaluation,
not an untouched final holdout. Do not inspect 2026 results in this study.

## Measures and decision

Primary: paired difference in outer-evaluation mean weekly Spearman Rank IC,
with all stocks in a week kept together. Minimum worthwhile screening effect:
Huber minus MSE of +0.005 IC. Report a paired circular-week block bootstrap
95% interval (20,000 draws, seed 42), block 4 with 2/8 sensitivity, plus weekly
IC autocorrelations to assess that block assumption. One primary comparison.
Call the screening result promising only if the primary interval excludes zero,
the point estimate exceeds +0.005 and weekly-weighted RMSE worsens no more than
2% relative. If the upper interval is below +0.005, evidence does not support
the prespecified useful effect here; otherwise classify inconclusive. Any
promising result requires later seed/era confirmation before production use.

Secondary: weekly-weighted raw-return RMSE/MAE, zero/training-mean baselines,
prediction bias and within-week dispersion, top-15 gross five-session mean
return minus the eligible equal-weight universe, top-minus-bottom-15 spread,
and membership turnover. Select by descending forecast with symbol tie-break,
no stickiness, equal starting weights each week. These are descriptive gross
screening marks, not executable NAV: no broker, HRP, fees or slippage simulation.
Keep all original valid extremes in evaluation. Report the largest 1% of
baseline squared residuals' error share as a tail diagnostic.

Save every epoch's objective, raw MSE, validation MSE/IC/dispersion, gradient
norm and clipping frequency, learning rate, elapsed time, best/stopping epoch,
and checkpoint hashes. More epochs or lower fitting loss alone are not success.
Budget: two serial full-cohort fits, seed 42 only, up to 60 epochs each with
patience 15; no parameter search. Retain failed attempts; never silently reuse
outputs with changed input/code/config fingerprints.

## Verification and execution

Synthetic offline tests must verify Huber's scaled cutoff, quadratic gradient
parity and capped tails, chronological label purging, and numerical reproduction
of the actual production MSE trainer using a tiny model and temporary outputs.
Also exercise Huber through the same real scratch fitting path. Use the shared
offline I/O guard. Record code revision, source/code/protocol hashes, dependency
versions, device and exact commands in each attempt.

From the repository root:

```sh
PYTHONPATH=brain_api:scratch/patchtst_weekly_hypotheses_2026_11_28/experiments/E2 brain_api/.venv/bin/python -m pytest scratch/patchtst_weekly_hypotheses_2026_11_28/experiments/E2 -q
PYTHONPATH=brain_api brain_api/.venv/bin/python scratch/patchtst_weekly_hypotheses_2026_11_28/experiments/E2/run_experiment.py --device mps
```

The runner prepares a frozen scratch panel, trains MSE then Huber, evaluates
both only after checkpoint lock, and writes `REPORT.md` plus machine-readable
evidence under `data/attempts/`. On resumption it validates a completed arm
before reuse; incomplete fits get a new attempt. No production endpoint,
provider request, publication, current-pointer change, schedule or email.
