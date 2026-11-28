# PatchTST research roadmap: diagnostics before tuning

Created: 2026-11-28. Updated: 2026-11-28. Status: E2 screening run completed;
the remaining experiment queue has not started.

## Purpose and working cadence

Determine whether weak US weekly PatchTST results arise from data errors,
limited predictive signal, optimization, validation policy, or model design.
Combine data diagnostics, controlled experiments, statistical inference, and
adaptive search. Do not assume that early checkpoint selection proves an
architecture failure or that a later best epoch is an improvement.

This is a multi-session research roadmap, not an instruction to implement or
run every experiment. Work on one experiment at a time over multiple days.
Each session plans the next experiment precisely, runs only that experiment
when authorized, records its outcome, and identifies the next useful question.
An experiment may take several sessions. Dates are recorded when work occurs;
there are no promised runtimes or automatic daily jobs.

**Latest result: E2, matched MSE versus Huber, inconclusive.** Both arms selected
epoch 1 and stopped at epoch 16. Huber's reused-2025 mean weekly Rank IC was
0.02402 versus 0.01692 for MSE; the paired difference was +0.00710 with a
four-week-block 95% interval of −0.01147 to +0.02670. RMSE was essentially
unchanged. See the [report](experiments/E2/REPORT.md) and
[protocol](experiments/E2/PROTOCOL.md). The user authorized starting E2 directly;
its bounded prerequisite checks do not complete the broader A0–A4 program.
Next: review diagnostics or preregister seed/era replication before choosing a
production loss. No additional experiment is running.

## Scope and isolation

- Initial scope: US seven-feature weekly PatchTST on `halal_new`. India is a
  separate future study; success in the US does not establish India parity.
- Put research code, manifests, logs, checkpoints, and results inside this
  directory. Reuse suitable repository calendar, feature, metric, and fitting
  functions; keep experimental variants in scratch.
- Production evidence may be inspected read-only. Record provenance before
  using it. Never move, overwrite, delete, or promote production artifacts.
- Do not invoke production training endpoints, change `current`, publish to
  HF, modify production configuration, or trigger Temporal, brokers, or email.
- Inventory existing local/HF evidence before proposing downloads. Run research
  against explicit frozen local inputs; record missing inputs and plan any
  acquisition separately. Avoid implicit provider calls or automatic HF writes.
- Do not make DuckDB backups or rearrange its live files. Plan a read-only,
  consistent extraction if the experiment needs news evidence from an active
  store; stop on contention rather than interfering with its owner.
- Ordinary tests for research code use synthetic data and temporary storage,
  with network and production artifact access blocked. Real-data research runs
  are explicit commands outside the ordinary test suite.
- A successful experiment supports a later production-change proposal; it
  neither changes production nor creates additional promotion requirements.

## Starting evidence and corrected assumptions

These facts were inspected on 2026-11-28; A0 must revalidate them against the
checkout and artifact selected for the study.

| Observation | Meaning for the study |
| --- | --- |
| Context is 365 calendar days, padded to 255 session slots; seven US features | One year is the input context, not the full fitting-history length. |
| Input channels are already normalized within each causal window | HF `scaling: null` does not establish an unscaled channel-magnitude problem. |
| Targets are divided by training-fold return standard deviation | Loss thresholds must account for this scale. |
| Split uses 20% of eligible anchor weeks, with at least 13 validation weeks | The main artifact has 114 validation weeks; fixed 52 weeks is a hypothesis. |
| Training requires both anchor and target dates before the validation boundary | Label purging already exists. Shared historical input context alone is not future leakage. |
| Weekly loop constructs Adam without passing `config.weight_decay` | Changing that config field alone will not test regularization. |
| Weekly loop explicitly computes MSE | Changing the HF loss field alone will not test Huber. |
| Annual fit results compute baseline loss and Rank IC, but snapshot metadata omits them | Collect these in scratch evidence; absent metadata is not a zero metric. |
| Main artifact and inspected 2020–2025 snapshots contain the same 417-symbol roster | The 360–408 counts describe latest-label eligibility, not historical membership. |

Saved reference: `v2026-11-20-549d9adc9504`, fitted with patience 8.
Its validation MSE is `0.003024189992251299`, training-mean baseline MSE is
`0.0030309744667191447`, and mean weekly Rank IC is `0.015852980561863204`.
The relative MSE improvement is approximately 0.224%; statistical uncertainty
has not been established. These metrics alone do not prove flat predictions.

The current default patience is 15. Distinguish reproduction of the saved
patience-8 artifact from comparisons using a newly frozen patience-15 control.
Never change patience silently between experimental arms.

Current-roster historical analysis has survivorship bias. Temporal splits and
data cleaning do not repair membership bias. Disclose it in every report;
obtain point-in-time membership before claiming an unbiased strategy backtest.

## Experiment queue

E2's first screening run is complete. Other entries remain pending;
later experiments are conditional on evidence, not a commitment to execute the
entire queue.

| ID | Question | Type | Dependency / stopping point |
| --- | --- | --- | --- |
| A0 | What exact evidence, code, and evaluation protocol can we reproduce? | Inventory and preregistration | First; stop if provenance or isolation cannot be established. |
| A1 | Are extreme observations errors or genuine market events? | Data-integrity audit | A0; resolve material integrity defects before comparative fitting. |
| A2 | Do tails dominate error, and which causal features show stable associations? | Statistical diagnostic | A1; quantify uncertainty rather than declaring features useless. |
| A3 | How strong are simple causal baselines on the same examples? | Baseline study | A2; establish the comparison to beat. |
| A4 | Can the implementation learn, and are its updates sensible? | Small implementation diagnostic | A1; repair demonstrated research-path defects before tuning. |
| E1 | Does a fixed 52-week validation window improve later predictions? | Controlled comparison | A3/A4; compare on common later evaluation dates. |
| E2 | Does robust loss improve outcomes without altering observed returns? | MSE versus Huber | A2/A4; one loss change, all other settings matched. |
| E3 | Which optimization factors matter and interact? | Designed screening | E1/E2 evidence; bounded factorial design, not a large grid. |
| E4 | Does target winsorization add value beyond robust loss? | Optional target transformation | Only if tail evidence warrants it. |
| E5 | Does a volatility-adjusted target help? | Optional target transformation | Only with causal, horizon-consistent volatility and inverse conversion. |
| E6 | Do stationary feature representations help beyond existing normalization? | Optional input ablation | Only if A2 identifies a reason; preserve inference reproducibility. |
| E7 | Can a representative smaller training roster accelerate research reliably? | Optional breadth study | Compare trained-on and excluded stocks; control update budget. |
| E8 | Can adaptive search refine the important settings efficiently? | Bayesian optimization | Only after factors and evaluation protocol are credible. |
| C1 | Do frozen finalists improve repeatably outside tuning? | Historical confirmation and future observation | Last; no automatic production change. |

## A0 — Evidence inventory and protocol

Inspect existing scratch studies before recreating loaders, panels, statistics,
or baseline implementations. Older close-only experiments are useful references,
but are not evidence for the current seven-feature recipe without parity checks.

Record code revision and working-tree differences, resolved configuration,
library versions, device, seeds, source files/commits and checksums, roster,
calendar, price adjustment policy, news timing/revision policy, and data limits.
Record exact anchor dates, target-end dates, fold boundaries and exclusions.
Do not assume that an artifact's data-window end is its latest fitting anchor.

Create a session-specific protocol before inspecting candidate evaluation
results. It must freeze:

- Hypothesis, mechanism, control, changed factors, primary metric, and secondary
  constraints. Proposed primary metric: mean weekly cross-sectional Rank IC.
- Development periods, fitting/validation/evaluation boundaries, label purge,
  shared eligibility rules, seeds, budgets, and stopping/pruning rules.
- Minimum practically worthwhile improvement and inference method. Choose these
  before candidate comparisons; do not invent a universal IC or p-value cutoff.
- What evidence permits proceeding, what rejects the hypothesis, and what counts
  as inconclusive. Record all tried configurations, including failures.

Use 2020, 2022, and 2024 cutoffs as candidate development blocks, subject to
verified input coverage and causal evaluation construction. Reserve remaining
historical blocks from further tuning where possible. Previously inspected
historical periods are historical confirmation, not genuinely untouched tests.
Future outcomes after recipe lock provide genuinely unseen evidence.

Deliverable: evidence manifest, frozen protocol, and a short readiness report.
No neural fitting is needed for this step.

## A1 — Data integrity before outlier treatment

Audit duplicate symbol/session rows, missing exact sessions, invalid/nonfinite
prices or volumes, stale bars, adjustment discontinuities, corporate actions,
timestamp alignment, and causal news availability. Treat suspicious jumps as
investigation candidates rather than automatically labelling them errors.

Separate confirmed errors, unresolved observations, and verified genuine events.
Keep an exclusion/repair ledger with evidence and affected symbol-weeks. Preserve
the session grid: an unusable example may be excluded, but deleting dates must
not compress a 255-slot context or a five-session target.

Do not automatically remove ETFs, multiple share classes, or legitimate volatile
stocks. Universe changes require their own hypothesis. Evaluate genuine crashes
and earnings jumps rather than clipping them out of the reported outcome.

Deliverable: integrity report and frozen eligible panel. If major defects change
the panel, establish a corrected control before comparing model variants.

## A2 — Tail influence and statistical signal audit

Measure return and residual distributions by date, stock, and volatility group.
Report the share of squared error contributed by the largest 1% of residuals,
prediction bias/dispersion, within-week forecast dispersion, and calibration.
Residual diagnostics need predictions; if saved predictions are unavailable,
record the reconstruction requirement instead of claiming they were measured.

Compute weekly Spearman IC, quantile spreads, and their era stability for causal
feature summaries and latest values. Check Pearson/Spearman redundancy. Latest
raw close or volume alone is not a complete test of the normalized input sequence.

Low univariate IC does not establish irrelevance in a nonlinear interaction.
Regression coefficients describe conditional associations, not causal effects.
ICIR is descriptive unless dependence and estimation uncertainty are addressed.
Do not discard features using a universal `abs(t) < 1.5` rule.

Deliverable: diagnostic report with uncertainty and a prioritized hypothesis list.
Avoid claiming that a CPU audit proves that no learnable signal exists.

## A3 / A4 — Baselines and implementation checks

For A3, compare zero return, training-only historical mean, and ridge using
causal summaries of the same seven channels. Fit transforms and hyperparameters
only within development folds. A shallow boosted-tree baseline is a later,
separate extension, not a prerequisite for the first comparison.

For A4, use tiny synthetic data to test whether fitting can learn a known mapping
and overfit a small batch. Inspect gradient norms, clipping frequency, parameter
updates, target scaling and inference inverse scaling. A bounded learning-rate
range diagnostic may locate unstable rates; it does not prove generalization.

Ridge beating PatchTST demonstrates a performance gap, not a unique diagnosis.
Ridge failing does not rule out nonlinear signal. Keep synthetic implementation
tests separate from real-data research and avoid pretrained downloads.

Deliverables: matched baseline predictions and an implementation-check report.

## E1 / E2 — First controlled model comparisons

Plan E1 and E2 separately. Start with the hypothesis best supported by A1–A4.

E1 compares the proportional holdout with 52 anchor weeks. Preserve causal
label purging in both arms. Each arm may choose checkpoints on its own inner
validation fold, but compare final performance on the same later outer dates.
Comparing their different validation losses directly is not a fair estimate of
the validation-policy effect. Do not refit across held-out outcomes implicitly.

E2 compares MSE and Huber on identical inputs, targets, folds, and seeds. Huber
acts on residuals, not directly on return magnitude. If testing a five-percentage-
point residual threshold, convert `0.05` to standardized units using only the
fitting-fold target scale. Record the loss convention and gradient scaling;
Huber can alter effective update magnitudes as well as tail influence.

Keep annual snapshot orchestration out of an individual research fit. Later
multi-cutoff studies deliberately schedule their own scratch fits and evidence.

## E3 / E8 — Designed experiments and adaptive search

Use controlled comparisons to understand mechanisms, then factorial designs to
test interactions. Four binary factors require 16 full combinations. A suitable
8-run fractional factorial can screen main effects, but confounds some
interactions; record the generator and alias structure and schedule follow-ups.
Do not treat a generic eight-trial sample as a valid fractional design.

Choose factors and ranges after diagnostics. Candidate factors include learning
rate, actual optimizer regularization, loss, and validation policy. Keep optimizer
family fixed when estimating decay effects; Adam versus AdamW is a separate
factor. Feed-forward and head dropout are separate factors unless a combined
regularization package is explicitly the question being tested.

Use matched cutoffs and seeds as experimental blocks. A single seed can screen
for gross effects but cannot establish stability. Do not assume one-factor
screening detects improvements that exist only through interactions.

After screening, Bayesian optimization can propose informative settings within
frozen bounds and a fixed trial budget. Functional ANOVA can estimate factor and
interaction importance from those trials; it cannot do so before trials exist,
and its estimates depend on the explored space and surrogate quality.

Use Hyperband-style pruning only after a pilot shows that short-run rankings
predict completed-run rankings. Lower learning rates may improve later, so a
two-epoch rejection rule is not justified by current best-epoch observations.

## Optional studies E4–E7

- E4: winsorize fitting targets only, using a preregistered causal rule. Preserve
  original evaluation targets; record clipping frequency and tail performance.
  Within-week thresholds use outcomes only from the fitting fold, never later
  validation/evaluation outcomes. Treat this as a changed learning objective.
- E5: predict return divided by anchor-known volatility. Current 20-day volatility
  is annualized; a five-session scale may use `sigma_annual * sqrt(5/252)` as an
  approximation, not an exact distributional guarantee. Specify a training-only
  floor, missing-value policy and inverse conversion. Evaluate raw-return
  predictions and rankings after conversion, not only risk-adjusted targets.
- E6: compare stationary price/volume representations with existing normalized
  channels. Do not add HF scaling as a presumed fix. Cross-sectional transforms
  require a defined reference roster at inference; arbitrary request batches
  must not silently change the feature definition.
- E7: select representative smaller rosters using information available at each
  cutoff, covering sectors, volatility, liquidity and history lengths. Compare
  against full breadth on a common evaluation roster and report results for
  included versus excluded stocks. Separate fixed-update-budget comparisons
  from fixed-epoch comparisons; fewer stocks otherwise imply fewer updates.

Student-t NLL, ranking-specific objectives, warmup, and architecture changes
remain later hypotheses. NLL needs a probabilistic output contract. Do not mix
all these changes into the first robust-loss or data-cleaning experiment.

## Statistical evaluation and confirmation

- Select checkpoints and tune parameters using inner chronological folds only.
  Evaluate frozen candidates on later outer periods with matured five-session
  labels. Use comparable information availability for news and prices.
- Report mean weekly Rank IC, RMSE/MAE in raw return units, prediction dispersion,
  top-15 excess return, spread and turnover. Define selection, weighting, return
  timing, benchmark and transaction-cost assumptions before portfolio analysis.
- Compute paired candidate-minus-control weekly metrics. Resample contiguous
  weeks with all same-week stocks kept together. Choose block length using
  dependence diagnostics and report sensitivity; do not assume all stock-weeks
  or overlapping annual snapshots are independent observations.
- Report effect sizes, confidence intervals, per-era results and seed variation.
  Absence of statistical significance is not proof of equivalence; a small
  sample may be inconclusive. Repeatedly inspected p-values are not valid
  stopping rules without a prespecified sequential procedure.
- Keep tuning evidence separate from confirmation. Record the number of trials,
  metrics inspected and protocol amendments. A confidence interval on the best
  tuning score alone does not correct selection bias or multiple comparisons.
- Freeze finalists before C1, repeat with multiple seeds, compare against both
  PatchTST control and simple baselines, and record negative results. Later best
  epochs or reduced training loss are not acceptance criteria.
- Historical confirmation retains current-roster survivorship and source-revision
  limitations. No scratch result automatically promotes a model or changes SAC.

## Resuming work and recording results

At each session, read this plan, the ledger and the last experiment report.
Validate existing manifests and resume compatible work instead of repeating it.
Run experiments serially unless a later protocol explicitly budgets concurrency.

Before execution, create `experiments/<ID>/PROTOCOL.md` with the hypothesis,
changed/preserved settings, exact commands, sources, fold dates, seeds, compute
budget, metrics, inference method, acceptance/rejection/inconclusive rules, and
synthetic verification. These details are intentionally planned per experiment;
this roadmap is not a blanket execution instruction.

During execution, append logs and state within `experiments/<ID>/`; use unique
attempt directories and retain failed attempts. Save configurations, predictions,
per-week metrics and checksums so conclusions do not depend on console logs.

After execution, write `experiments/<ID>/REPORT.md`: what ran, evidence, effect
sizes/uncertainty, limitations, disposition, and the next question. Update this
ledger before switching experiments. Do not check boxes for planned work.

| ID | Status | Protocol / report | Conclusion and next action |
| --- | --- | --- | --- |
| A0 | E2-scoped inventory completed; broader inventory pending | [E2 protocol](experiments/E2/PROTOCOL.md) | Frozen local prices and roster; strict read-only news extraction; manifests saved in E2 scratch data. |
| A1–A4 | E2-scoped checks completed; broader studies pending | [E2 checks](experiments/E2/test_training.py) | Known ticker collisions excluded in both arms; chronology, Huber gradients and exact synthetic production-MSE parity verified. |
| E2 | Screening completed 2026-11-28; inconclusive | [Protocol](experiments/E2/PROTOCOL.md), [report](experiments/E2/REPORT.md) | Both best/stopped 1/16. Huber − MSE IC +0.00710, 95% interval [−0.01147, +0.02670]; unchanged RMSE. No production change; replication remains a separate experiment. |
| E1, E3–E8 | Conditional | — | No other experiment has started. |
| C1 | Pending frozen finalists | — | Historical confirmation, then future observation. |

## Sources and repository entry points

Repository paths are relative to the repository root:

- `brain_api/brain_api/core/patchtst/weekly_data.py`: causal features and labels.
- `brain_api/brain_api/core/patchtst/weekly_training.py`: fold purge, target scale,
  optimizer, loss, checkpoint selection and Rank IC.
- `brain_api/brain_api/core/patchtst/config.py`: current weekly configurations.
- `brain_api/brain_api/services/patchtst_snapshots.py`: annual orchestration and
  current metadata omission; reference only, not the research execution path.
- `scratch/patchtst_corrected_experiments_2026_08_23/` and
  `scratch/patchtst_geometry_sweep_2026_08_23/`: prior controls and research
  protocols; verify compatibility before reuse.

Method references:

- [NIST: fractional factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri334.htm).
- [Bayesian optimization of machine-learning algorithms](https://arxiv.org/abs/1206.2944).
- [Hyperband and adaptive resource allocation](https://jmlr.org/papers/v18/16-558.html).
- [Functional ANOVA for hyperparameter importance](https://proceedings.mlr.press/v32/hutter14.html).
- [Model-selection overfitting and evaluation bias](https://jmlr.csail.mit.edu/papers/v11/cawley10a.html).
- [NIST: Kruskal–Wallis assumptions](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/kruskwal.htm).
- [Huber regression and robust residual loss](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html).
