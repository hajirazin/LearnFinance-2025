# E2 result: MSE versus Huber

**Huber did not change early checkpoint selection. Its later ranking gain is
inconclusive.** Both models selected epoch 1 and stopped at epoch 16 under the
same current patience-15 policy. The Huber point estimate exceeded the frozen
+0.005 Rank IC screening threshold, but every reported interval includes zero.

Attempt: `20261128T102217556216Z`. 51 paired weeks, 19,157 stock outcomes.

| Arm | Best epoch | Stopped epoch | Validation MSE | Evaluation Rank IC | Evaluation RMSE (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mse | 1 | 16 | 0.00200244 | 0.01692 | 4.9920 |
| huber | 1 | 16 | 0.00200080 | 0.02402 | 4.9926 |

Huber minus MSE mean weekly Rank IC, paired circular-week bootstrap:

- Block 4: +0.00710, 95% interval [-0.01147, +0.02670].
- Block 2: +0.00710, 95% interval [-0.01705, +0.03056].
- Block 8: +0.00710, 95% interval [-0.00817, +0.02376].

## Did the models learn useful signal?

Both objectives can learn a strong known synthetic relationship in the offline
implementation tests. In real fitting, MSE fell from 0.0023644 to 0.0021271
for the control and from 0.0023647 to 0.0021334 for Huber, while neither improved
validation MSE after epoch 1. This demonstrates improving fitting error, not
that later epochs generalized.

Both selected checkpoints were worse than the training-mean validation
baseline (MSE 0.00199530). On reused 2025 evaluation, both narrowly beat zero
and training-mean prediction RMSE, but the forecast improvement is small.
Positive point-estimate ranking correlation alone does not establish a stable
signal. This result does not settle whether the remaining limitation is
optimization, architecture, data, validation policy, or weak financial signal.

| Evaluation measure | MSE | Huber |
| --- | ---: | ---: |
| Zero-return baseline RMSE (pp) | 5.0076 | 5.0076 |
| Training-mean baseline RMSE (pp) | 4.9962 | 4.9962 |
| Model MAE (pp) | 3.5152 | 3.5162 |
| Mean within-week forecast standard deviation (pp) | 0.2287 | 0.2011 |
| Top-15 return minus eligible-universe mean (pp/week, gross) | 0.2201 | 0.2824 |
| Mean fraction of top-15 names replaced | 0.2920 | 0.4347 |

Huber's higher gross basket excess comes with more name replacement; costs
were not simulated, and these secondary comparisons have no confirmatory
inference here. Do not select a production loss from these point estimates.
The largest 1% of fitting residuals from the training-mean predictor accounted
for 22.8% of squared error. This diagnostic motivates the question but does not
classify those observations as bad prices.

## Cohort and folds

The frozen 417-symbol roster yielded 388 archived candidates after excluding
11 present ticker collisions from the previously verified 13-symbol list and
18 names absent from this price archive. No IPO-age, ETF, share-class or
volatility filter was introduced. Window eligibility applies identically in
both arms.

| Fold | Examples | Anchor weeks | Symbols | Anchor dates | Latest label |
| --- | ---: | ---: | ---: | --- | --- |
| Fitting | 112,186 | 331 | 371 | 2017-01-06 to 2023-05-05 | 2023-05-12 |
| Inner validation | 31,234 | 84 | 376 | 2023-05-19 to 2024-12-20 | 2024-12-30 |
| Outer evaluation | 19,157 | 51 | 379 | 2025-01-03 to 2025-12-19 | 2025-12-29 |

One seed, reused 2025 evaluation, current-roster survivorship bias and incomplete independent ticker/corporate-action audit limit the conclusion. Strict point-in-time news was used; missing cells were not filled with neutral scores. Both arms retain unmodified returns and use the same validation-MSE checkpoint policy. Top-15 marks are descriptive gross five-session returns, without transaction costs or HRP. More epochs alone do not establish useful learning. No production model was changed.

## Verification and disposition

All 12 synthetic offline tests passed, including exact checkpoint parity with
the actual production MSE trainer, scaled Huber gradients, label purging,
known-mapping learnability, paired evaluation and checkpoint-tamper rejection.
Ruff and whitespace checks passed. Initial parameters matched exactly across
the GPU fits. All frozen panel and source-code hashes still matched after
training/evaluation. The exact run sources are archived under the attempt's
`source/` directory. Subsequent repository-configured lint cleanup only sorted
imports and rewrote equivalent literal/list construction. Test collection now
excludes generated `data/` archives; all 12 current offline tests were rerun.
The strict identity checks intentionally reject silently reusing the old panel
under changed source hashes; restore the archived sources to reproduce that
exact run rather than rewriting its evidence manifest.
The prediction helper emitted a read-only NumPy mapping
warning; inputs were transferred to MPS and their post-run hashes were unchanged.

Retain this inconclusive result. Review diagnostics or preregister additional
seeds and eras before adopting Huber; no further fits are running as part of this
completed attempt. Operational limitation: the runner reuses a
fully completed compatible pair, but an interrupted partial pair starts a new
attempt rather than resuming optimizer state or reusing its completed arm.

[Protocol](PROTOCOL.md) · [Full metrics](data/attempts/20261128T102217556216Z/evaluation.json) · [MSE epochs](data/attempts/20261128T102217556216Z/mse/progress.json) · [Huber epochs](data/attempts/20261128T102217556216Z/huber/progress.json)
