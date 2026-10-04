# Alpha-HRP / PatchTST audit — October 3, 2026

**Expanded original-five-task math audit:** see [MATH_AUDIT.md](MATH_AUDIT.md). The August 14 artifact compared below has five inputs but already uses close-only loss. Original May/July five-task artifacts are now separately tested, along with train-mode BatchNorm dependence and the forced RevIN momentum term. The dates and artifact-specific results below remain historical audit evidence.

Follow-up: [CORRECTION.md](CORRECTION.md) documents subsequent local code fixes, actual broker NAV and the downloaded market conditions. The performance table below remains the original target-weight diagnostic, not actual account returns.

The sampled top-15 selector is correct. There are confirmed input-validation and legacy-checkpoint loading defects, and a substantial model-quality problem. The evidence does **not** establish that removing OHLCV channels caused the deterioration or that restoring five channels will solve it.

This was a read-only audit of production code, saved checkpoints, Chrome Alpaca orders, and Gmail reports. Only diagnostic files in this directory were created. No production pointer, allocation, schedule, order, or account setting was changed.

## Scope and evidence

The close-only migration landed August 23 (`4e91cd1`, followed by `6618e2f`). There are six completed weekly Alpha-HRP reports after it through October 2. All six were examined, providing complete coverage rather than a random subset. September 7 was a US holiday; those orders filled September 8.

The six reports identify the same model: `v2026-08-21-0af698826abd`, trained August 23 on data through August 21. They score a 432-name universe and report 430–431 valid predictions. The current local checkpoint has the same identity. The local sticky database stops in July and was not treated as September production evidence.

Sources read in Chrome:

- [Alpaca HRP paper-account orders](https://app.alpaca.markets/account/orders). Account selection was verified explicitly. All 112 sampled order legs show `filled`.
- [August 24 Alpha-HRP email](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Alpha-HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWBjMMDZLFJgSKccsCQLRZsfk).
- [August 31 Alpha-HRP email](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Alpha-HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWLFnHsKxQBfXzWGTWCmmDzMN).
- [September 7 Alpha-HRP email](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Alpha-HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWLQCNXvgQLFRfKTWmSdpmzTj).
- [September 14 Alpha-HRP email](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Alpha-HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWTmbbrJcZnQtCkhtKqNclfMP).
- [September 21 Alpha-HRP email](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Alpha-HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWTvgVnVVMPrxppmcVhPVlSfn).
- [September 28 Alpha-HRP email](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Alpha-HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWfSHbhnFXBncQfmZPNDPRqzl).
- [August 17, last pre-migration report](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Alpha-HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F01+before%3A2026%2F08%2F23/FMfcgzQhVrKdmzhqmXgFfWBlxjzwpHJD).
- Six Double HRP reports and five SAC halal_filtered reports covering the same dates were also read. No August 31 SAC report appeared in the scoped search. [September 28 Double HRP](https://mail.google.com/mail/u/0/#search/subject%3A%22US+Double+HRP+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWfSHbhmvKcGwxwgMSKfKJWJF) and [September 28 SAC](https://mail.google.com/mail/u/0/#search/subject%3A%22US+SAC+(halal_filtered)+Weekly+Portfolio+Analysis%22+after%3A2026%2F08%2F22+before%3A2026%2F10%2F04/FMfcgzQhWfSHbhnFZHprSvLKGQqwWQSL).

Public price evidence was fetched October 3 with the repository's Yahoo loader (`auto_adjust=True`), April 1–October 2, for all 432 September-universe names plus SPY/QQQ. See `price_manifest.json` and `prices.pkl`. These are retrospective downloads, not original point-in-time run snapshots.

## Six-week pattern

Returns below are **diagnostic target-portfolio returns**, using the emailed rounded HRP weights and prior completed close to that calendar week's Friday close. They are not broker NAV returns or actual execution returns. They omit intraday fills, costs, cash drift, and allocation-threshold effects.

| Week | New / retained names | Filled buys / sells | Largest HRP weight | Target portfolio | SPY | QQQ |
|---|---:|---:|---|---:|---:|---:|
| Aug 24 | 6 / 9 | 7 / 17 | PSX 17.82% | +1.45% | +0.47% | +0.42% |
| Aug 31 | 6 / 9 | 9 / 7 | INCY 16.71% | −2.76% | +0.11% | +0.35% |
| Sep 7, trades Sep 8 | 8 / 7 | 8 / 14 | CTVA 26.91% | −2.38% | −0.77% | −0.57% |
| Sep 14 | 7 / 8 | 11 / 8 | PSX 19.15% | +3.47% | −0.09% | +0.92% |
| Sep 21 | 6 / 9 | 7 / 8 | PSX 15.13% | −2.74% | +1.27% | +3.30% |
| Sep 28 | 4 / 11 | 5 / 11 | COP 21.00% | +1.37% | −0.22% | +0.68% |

Across the six weeks, 37 additions replaced 41% of the 15-name roster per week on average. Actual buy/sell counts are higher because retained positions are resized; August 24 also flattened five tiny residual positions. Gross buy plus sell notionals total approximately $595k on an account around $100k.

SPY rose 0.76% and QQQ 5.18% from August 21 to October 2 adjusted closes. The target-portfolio calculation compounds to −1.75%; equal weighting the same selected names compounds to approximately +0.07%. This small, retrospective comparison suggests concentration amplified losses, but it is not a validated recommendation to change the allocator.

Concrete equal-quantity filled round trips:

| Symbol | Bought → sold | Fill-price return | Approximate price P&L |
|---|---|---:|---:|
| DASH | Aug 31 → Sep 8 | −13.77% | −$712 |
| MMYT | Aug 31 → Sep 8 | −10.57% | −$397 |
| CTVA | Sep 8 → Sep 14 | −3.68% | −$978 |
| EXPGY | Sep 14 → Sep 28 | −13.53% | −$1,447 |
| CRM | Sep 14 → Sep 28 | −11.56% | −$303 |
| OKTA | Aug 24 → Aug 31 | +26.70% | +$758 |

The model does produce winners; the evidence does not support an inverted ranking. CTVA's comparatively modest percentage loss mattered because it carried 26.91% of the portfolio. In the September 21 week, PSX alone contributed about −0.96 percentage points to the target portfolio while QQQ rose 3.30%.

## Findings

### 1. Selection math is correct in all six sampled reports

`core/sticky_selection.py:215` sorts descending score, with deterministic symbol tie-breaking. `select_with_rank_band` at line 360 keeps previous selected names within rank 20 and fills remaining slots from highest-ranked non-retained names. The workflow supplies the full-universe scores, partition `halal_new_alpha`, K_in=15/K_hold=20, then computes HRP252 only over those selected names.

All six email selections match this rule, including the August 17 carry-set for August 24. For September 28, retained ACN at rank 18 and MSTR at rank 20 occupy slots that otherwise could go to UI at rank 13 and NTOIY at rank 15. This is the intended sticky rule, not an off-by-one error. Stage 2 weights are risk-based; the highest forecast need not receive the highest weight.

### 2. Confirmed defect: stale/missing sessions are accepted at inference

`core/patchtst/inference.py:101–158` filters before the cutoff, checks row count, and takes the last 60 observed returns. It does not require the last expected completed XNYS session or a contiguous expected session history. A missing session changes a multi-day close move into one supposed daily return and shifts every subsequent patch position.

Reproduction in `probes.py`: IBM ending September 17 is accepted as `has_enough_history=True` for a September 28 decision. A frame with five interior sessions removed is also accepted with shape `(60,1)`. The training dataset similarly uses observed-row ISO-week anchors rather than verified exchange-session anchors.

**Evidence this mattered in a real sample:** five reports reproduce from the saved model with fresh prices to within 0.005 percentage points, the email's rounding precision. August 31 does not: ATEYY is reported +4.53%, versus −0.3423% using history including Friday August 28; SMCI is reported +3.98%, versus −1.0724% with Friday included. Omitting Friday makes *all 30 displayed scores* reproduce within 0.0046 percentage points. The run therefore has a strong signature of Thursday-ended inputs, one completed session behind.

Original run input snapshots were unavailable, so the precise upstream reason remains unproved: provider omission, a running older loader, or another cutoff path. The Yahoo inclusive-end fix landed August 30 (`d552876`); this alone does not prove it caused August 31's issue, since current batch inference requests a Sunday end for normal Monday decisions. [Yahoo's documented `end` semantics are exclusive](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

### 3. Confirmed defect: the current loader cannot load the prior checkpoint

`storage/patchtst/local.py:65–73` rebuilds an artifact using today's `config.to_hf_config()`. The previous adapter passed `stride=8`, but the Hugging Face field is `patch_stride`, default 1. Thus the August 14 checkpoint actually used 45 patches, not the intended six.

The current adapter correctly maps `patch_stride=self.stride`. Loading `v2026-08-14-e7b14b211a54` with it fails strictly: saved positional encoding `(45,64)`, rebuilt encoding `(6,64)`. This defect would break a pointer-only rollback, not silently corrupt the current one-channel model. Reconstructing the historical effective architecture loads the old checkpoint successfully, and its top 15 August 17 email scores reproduce within 0.0046 percentage points.

Model artifacts need an effective HF architecture/schema identity; old versions require a version-aware loader. Do not infer legacy architecture from today's adapter.

### 4. Material model-quality weakness: promotion verifies health, not skill

Current checkpoint metadata:

- Validation weekly rank IC: **0.02338**, close to zero.
- Validation daily-close MSE: **0.000647687**.
- Reported constant baseline MSE: **0.000632860**; model error is about 2.34% higher.
- Best epoch 11, stopped epoch 19; promoted with no health failures.

`core/training_utils.py:69–126` explicitly implements "always promote when guardrails pass": positive finite losses and nonempty artifact files. It never requires positive rank IC, a baseline improvement, or a fair incumbent comparison. This is a policy weakness, not a failure to implement the existing policy.

The baseline in `core/patchtst/training.py:443–446` uses the validation targets' own mean. It is a hindsight diagnostic rather than a deployable causal baseline. It should not become the sole promotion criterion unchanged. Training minimizes daily-close MSE while checkpoint selection uses weekly cross-sectional rank IC; these objectives are not identical. Raw MSE also gives larger-return observations more influence.

Reconstructed production scores have average forward next-five-session rank IC **−0.0161** over these six weeks. Three weeks are positive and three negative. Calendar-week average IC is −0.0735. Forecast ranks correlate **0.84–0.88** with trailing 60-return mean ranks, suggesting much of the signal acts like recent-trend extrapolation. This is descriptive correlation, not proof of the model's causal mechanism.

### 5. The migration changed much more than channel count

Original five-task setup: 5 inputs, 16-bar patches, **effective stride 1**, 45 patches, weight decay 1e−4, MSE-based checkpointing. From August 8 the objective was already close-only even though inputs still had five channels. Weight decay changed to zero on August 9, so the August 14 artifact already has zero weight decay.

After August 23: 1 input, 10-bar patches, stride 5, 11 patches, no weight decay, altered epoch/patience settings, rank-IC checkpointing, and fresh stochastic training. Those are confounded changes.

The old model is channel-independent with channel attention disabled. Multiplying its four non-close input channels by seven changes the close output by **exactly zero** in an eval-mode perturbation test. This does not imply training with extra channels is identical: shared weights and training normalization/objectives can differ. [HF documents channel independence and the optional channel-attention mechanism](https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/patchtst).

The old August 14 checkpoint, reconstructed correctly and applied with complete pre-decision history to these same six forward weeks, averages **−0.0410** next-five-session rank IC. It also struggles in the August 31 and September 21 weeks. This is a checkpoint counterfactual, not a controlled retraining ablation, but it does not support assuming five channels would restore the prior performance.

### 6. Secondary issues and limits

- **Holiday horizon:** forecast output is always five sessions. In the September 7 holiday week, it extends through September 14 close, beyond the next rebalance at September 14 open. The same scores have negative calendar-week IC but positive next-five-session IC. This contract is documented, but it mismatches a strict weekly rebalance outcome in holiday weeks.
- **Cash drift:** Alpha-HRP reports prior cash of −0.77%, −0.88%, −1.05%, and −1.25% in four sampled weeks. `core/orders.py:554` caps buys using snapshot-price notionals, while submission uses market quantities and targets sum to 100%. That cap does not guarantee cash remains nonnegative after fills. This is an execution-risk observation; it is not demonstrated to be the principal loss source.
- **Reproducibility:** training shuffles data and initializes randomly without persisting a seed. Training endpoints skip existing versions, so this is not proof that routine reruns overwrite current weights. Fresh reconstructions of the same version inputs can nevertheless differ. A config/date version alone is insufficient for exact reconstruction.
- **Audit lineage:** sticky score rows do not persist the model version, per-symbol data-end dates, or immutable input/model hashes. The emails supplied the model identity here; explaining the stale-Friday event requires stronger saved evidence.
- **Comparators:** Double HRP selected the same 15 names in all six emails, unlike Alpha's 37 additions. SAC changed most of its monthly slate on September 8 and used a different allocator checkpoint later in September, so it is not a clean test of one-vs-five-channel PatchTST. Current account balances ($94,734.97 Alpha, $99,829.42 Double HRP, $98,738.70 SAC) are snapshots, not equal-period returns. Comparator target calculations in JSON have the same price/horizon limitations as Alpha and are not reconciled broker performance.
- Six forward weeks are too few to establish stable skill or isolate channel causality. The 430-name recomputation excludes two nonfinite predictions; live reports sometimes counted 431. The missing/nonfinite-name discrepancy does not affect the displayed top-30 matches but prevents claiming exact full-universe point-in-time reproduction.

## Recommended repair/research order

1. Reject or explicitly exclude stale/gapped input using expected completed exchange sessions; preserve finite-score/minimum-count failure behavior. Enforce the same session contract in training.
2. Persist original inputs, actual data-end dates, effective HF config, model/weight hashes, selected-set reasons, and final fills. Repair version-aware checkpoint loading before attempting rollback.
3. Evaluate candidates and incumbents over identical untouched chronological weeks with causal baselines, weekly rank IC, top-15 selection outcomes and execution-aware returns. Keep artifact-health checks as a separate requirement.
4. Run a controlled channel ablation with fixed data splits, geometry, loss, checkpoint rule and seed sets. Test concentration/turnover changes as separate research experiments, without silently changing the strategy's mandated ranking.

## Verification and files

73 existing relevant tests passed: rank-band selection, PatchTST config/audit fixes, training rank IC, weekly rank IC, and inference route tests. The fresh-history safeguard and legacy loading probes above expose gaps those tests do not cover.

- `analyze.py` / `analysis.json`: all six transcribed targets/top-30 scores, full recomputed scores and realized returns.
- `probes.py` / `probes.json`: cutoff sweep, stale/gapped input acceptance, and legacy-loader failure/reconstruction.
- `compare_legacy.py` / `legacy_comparison.json`: forward prior-checkpoint comparison and channel perturbation.
- `orders_evidence.json`: weekly filled-order summaries and exact-quantity round-trip evidence.
- `comparators.py` / `comparator_analysis.json`: six Double HRP targets and five SAC targets.
- `fetch_prices.py`, `price_manifest.json`, `prices.pkl`: provider/time-stamped public price evidence.

Reproduce from repo root with `brain_api/.venv/bin/python scratch/alpha_hrp_audit_2026_10_03/analyze.py`, followed by the other three diagnostic scripts. No network is required once `prices.pkl` exists.
