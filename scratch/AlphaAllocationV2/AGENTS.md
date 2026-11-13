# AlphaAllocationV2 research instructions

Follow the [repository-wide instructions](../../AGENTS.md). This folder contains offline AlphaAllocationV2 research. Keep notebooks, code, trial configs, caches, and outputs under `scratch/AlphaAllocationV2/`. Reuse production math and read-only snapshots where appropriate. Keep experiment writes away from production data, sticky histories, model storage and `current` pointers, Temporal, and broker accounts. Treat any production adoption as a separate task.

## Research order

- Work on screening first. Rank by the predicted simple return from Friday’s adjusted close to the following Friday’s adjusted close. Compare screens by the arithmetic mean of those realized returns for the 15 selections with equal starting weights. State how the experiment handles Friday holidays. Change one component at a time when measuring the effect of stickiness.
- Freeze the dated selections before comparing allocators. Give every allocator the same selected names, initial state, constraints, execution assumptions, and cost model. Allow each policy’s holdings to evolve from its own simulated trades.
- Keep the incumbent reproducible. Use the Alpha-HRP PatchTST score, rank-band sticky selection (`K_in=15`, `K_hold=20`), and HRP with its 252-session lookback for the end-to-end baseline. Use forecast artifacts available by each decision date or explicitly label a causally retrained approximation. Pair comparisons on the same weeks and disclose any coverage differences.

## Data and evaluation

- Before a confirmatory run, record the question, hypotheses, sample and units of analysis, metric, baselines, and comparison method. Define each feature’s formula, units, source, lookback, availability time, and missing-data policy. Record the Golden Cross moving-average windows and crossover rule. Resolve the `p*` features and black line from their sources.
- Use point-in-time universe membership, timestamps, and completed sessions. If only a current roster is available for historical work, freeze it and disclose survivorship bias. Document ticker changes, delistings, and corporate-action handling. Separate verified zero-news observations from missing news coverage. Report data gaps and any exclusions.
- Rebuild Friday news features from the archived events that were created and last updated by the selection cutoff. Treat unavailable prior article revisions as a coverage limitation. Keep the research return benchmark and the feasible order cutoff explicit when evaluating execution.
- Use chronological walk-forward development and reserve a final holdout for the frozen pipeline. Keep label windows from crossing split boundaries. Fit imputation, scaling, outlier thresholds, PCA, and model parameters on each training fold. Use validation results to select features and tune models. Compute contemporaneous ranks only from the eligible information available at that decision time.
- Match the test to the question and dependence in the data. For example, use within-week rank statistics for ranking and dependence-aware intervals over weeks. Use ANOVA for suitable group comparisons, check its assumptions, and choose a robust alternative when needed. Account for multiple comparisons when screening many features. Report effect sizes and uncertainty, then use out-of-sample comparisons or ablations to assess incremental predictive value.
- Inspect missingness and extremes before cleaning. Keep an audit trail of invalid records, dropped samples, and transformations. Compare proposed cleaning against the original valid data. Missing-feature imputation and treatment of genuine extremes are distinct decisions. Keep realized return labels and execution prices tied to observed evidence.
- Simulate fills after the decision cutoff and carry holdings and cash through the backtest. Account for fees, spread or slippage, and turnover with documented assumptions. Use the same assumptions for paired comparisons and test sensitivity where costs are uncertain.

## Experiment records

For each experiment, record the code revision, data snapshot or manifest, feature definitions, date ranges and folds, parameters, seeds where relevant, and exact run instructions. Keep the EDA and result plots with the run or report. Include baseline results, per-week metrics where applicable, limitations, and a short decision tied to the evidence. Retain null and unfavorable results alongside favorable ones.

Check the [learning resources](README.md#learning-resources) and other relevant sibling projects when choosing a method. Include new coursework and projects as they become available. Cite the specific material and explain the method, its assumptions, and any adaptation to weekly financial data in plain language so the experiment can be reviewed independently.
