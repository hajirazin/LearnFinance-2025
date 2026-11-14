# AlphaAllocationV1

Improve the US Alpha-HRP V1 strategy before resuming [AlphaAllocationV2](../AlphaAllocationV2/README.md). The goal is to find a V1 candidate with a repeatable net return advantage over dividend-adjusted SPY and the incumbent, with acceptable drawdown and turnover. This folder holds offline research. Any production adoption is a separate task.

[Saved research recommendation](V1_RESEARCH_RECOMMENDATION.md) · [Agent instructions](AGENTS.md)

## Starting evidence

The Alpaca HRP paper account returned **−9.90%** versus **−1.24%** for SPY from the August 21 to November 13, 2026 closes. It beat SPY in 5 of 12 measured weeks. The August 24 rebalance first used close-only checkpoint `v2026-08-21-0af698826abd`.

The baseline labeled “V1” in the earlier hypothesis tables was a **research refit**, trained on 2021–2024 and selected on 2025 validation. On the overlapping August 21 to November 6 window, the refit with HRP returned **−10.01%**, the paper account **−10.75%**, and SPY **−3.74%**. Across six archived production reports, the replay and production selected an average of 6.67 of the same 15 names. The realized shortfall motivates the investigation. The before-and-after comparison leaves the effects of model design, training data, market conditions, and execution unresolved.

[Account and checkpoint comparison](../AlphaAllocationV2/README.md#research-replay-versus-the-deployed-model) · [October production audit](../alpha_hrp_audit_2026_10_03/AUDIT.md) · [Input timing correction](../alpha_hrp_audit_2026_10_03/CORRECTION.md)

## Research sequence

1. **Reconcile deployed V1.** For several archived weeks, use the checkpoint, timestamped inputs, eligible universe, and prior holdings available at the decision. Compare scores, sticky selections, HRP weights, orders, fills, cash, and NAV with saved production evidence. Record discrepancies by stage and date. If a checkpoint or historical input is missing, report what can be verified from archived targets and fills, and label the rest as a reconstruction.
2. **Audit historical inputs.** Check when each symbol and observation became available. Use point-in-time membership where available. Disclose any frozen-roster approximation and its survivorship bias. Apply history, quality, and volatility exclusions using only data available at the relevant training or decision date. Distinguish verified zero news from missing coverage, and document delistings, corporate actions, session gaps, and excluded weeks.
3. **Run a compact walk-forward comparison.** Refit at the intended production cadence using completed labels. Keep training, checkpoint selection, and subsequent evaluation in chronological order. Use the same decision dates, eligible universe rules, `15/20` rank stickiness, 252-session HRP, execution conventions, and costs across candidates. Carry each candidate’s holdings and cash from its own simulated trades. Use a common evaluation universe where possible and report any differences in scoring coverage or eligibility. Prespecify seeds and report their spread.
4. **Confirm prospectively.** Use the earlier hypotheses and periods already inspected to choose a candidate and estimate plausible effect sizes. Freeze the candidate, incumbent comparison, SPY benchmark, and analysis before an untouched evaluation period or a prospective paper shadow run. Log each decision before the outcome is known. Base any production proposal on the confirmed results and the reconciliation evidence.

| Initial arm | Question |
| --- | --- |
| Close-only PatchTST with the production training recipe | What would a causally retrained incumbent have earned on the same weeks? |
| Close-only PatchTST with the research recipe | Does changing the training recipe improve results? |
| Seven-feature PatchTST with the research recipe | Do the seven channels improve results under that recipe? |
| Ridge with the same seven features | Does a simpler model deliver comparable portfolio results? |

Use the incumbent’s 60-session context and MSE on five daily close log returns for the PatchTST arms. Define the causal version of the research recipe, including its fit and validation windows, before running it. The exact deployed checkpoints are for historical reconciliation. The production-recipe arm is a separate walk-forward baseline. If the research recipe helps, isolate training-window length, validation design, and missing-session handling before attributing the improvement to any one change. Use the [V2 feature definitions](../AlphaAllocationV2/README.md#ridge-and-patchtst-on-matched-2025-fridays) to identify the seven channels. Align the target and input history with Ridge where practical and document any differences that prevent attributing the result to model choice alone.

## Statistical evaluation

Use matched weekly portfolio returns as the unit of analysis. Mark the strategy and SPY at the same times, include distributions consistently, and measure strategy returns after execution costs. The Friday-close prediction label can still be useful for ranking. Score actual strategy performance using orders placed after the decision cutoff and fills the strategy could have received. Report cumulative net return, maximum drawdown, turnover, costs, and the fraction of weeks ahead of SPY alongside the primary statistic.

For each matched week, calculate two differences in percentage points:

- **Market excess:** candidate net return minus SPY total return.
- **Incumbent improvement:** candidate net return minus incumbent net return.

For each comparison, test the one-sided null that the mean weekly difference is zero or negative. Resample blocks of consecutive weeks jointly across strategies and SPY to preserve pairing and account for serial dependence. Define the primary block length and sensitivity checks before looking at confirmatory outcomes. Report the estimated differences, confidence intervals, and p-values. Use **Holm correction at 5%** for the family of prespecified confirmatory comparisons. Stock rows and random seeds do not increase the number of observed market weeks. ANOVA may help answer a separate group comparison with suitable assumptions. Use the paired weekly tests for the portfolio question.

Before confirmation, agree on a minimum economically worthwhile improvement, drawdown and turnover tolerances, and an evaluation horizon informed by the variability of weekly differences. Evidence for promotion should support both positive SPY excess and improvement on the incumbent, with an effect large enough to matter and acceptable risk. If the evidence is too imprecise to meet the prespecified decision rule, report the result as inconclusive. Review performance across market periods, concentration in a few weeks or names, and sensitivity to costs before interpreting any apparent win.
