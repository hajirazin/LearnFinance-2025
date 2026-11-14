# Saved V1 research recommendation

This is the original chat recommendation that led to this folder. The [README](README.md) contains the working research protocol.

**I would not deploy the research refit yet, and I would not repeat all 27 experiments now. First establish a trustworthy production baseline, then run a small, controlled comparison. ANOVA would not resolve our main problem.**

We have established that production underperformed SPY over the measured period, particularly after August. We have **not** established that PatchTST itself is unsuitable, or that the research training recipe fixes the problem. The research refit also struggled after August.

My earlier labeling of that refit as “V1” made the evidence misleading. Going forward, every result needs an exact checkpoint, training cutoff, evaluation period, and execution method.

**1. First prove that our simulator can explain production.**

For several completed production weeks, trace this entire chain:

`available data → model predictions → sticky top 15 → HRP weights → orders → fills → account NAV`

Use the checkpoint actually deployed that week, its available inputs, and the actual previous holdings. Reconcile the simulated portfolio with Alpaca, explaining differences from cash, execution timing, incomplete fills, dividends, and costs.

This separates three possible problems:

| Finding | What we should improve |
|---|---|
| Reconstructed predictions differ from archived production predictions | Input preparation, checkpoint loading, or inference |
| Predictions match, but selections or weights differ | Stickiness, eligibility, or allocation |
| Target portfolios match, but realized performance differs | Execution, cash management, and accounting |

**Until these differences are understood, a higher backtest return is insufficient evidence for deployment.**

**2. Audit the historical experiment data before comparing more models.**

Two earlier research choices deserve particular attention:

- Using today’s stock roster historically introduces survivorship bias.
- Removing stocks because of missing history or extreme returns observed over the entire dataset uses future information if applied to earlier decisions.

The 356-stock cohort and subsequent exclusions can support descriptive EDA. They are not automatically valid historical trading or training eligibility rules. For a predictive test, eligibility and exclusions must use information available at the relevant training or decision date.

Also distinguish verified “no news,” which you reasonably treat as neutral, from missing news coverage.

**3. Compare a small number of clearly defined candidates.**

My proposed first comparison is:

| Candidate | Purpose |
|---|---|
| Production training recipe, close-only | Establish the reproducible baseline |
| Research refit recipe, close-only | Test whether the training procedure helps |
| Research recipe with seven features | Test whether those features add value |
| Ridge regression with the same seven features | Check whether PatchTST’s complexity adds value |

For the two research PatchTST candidates, keep the 60-session lookback, daily-close target, loss, and architecture settings fixed wherever possible. Keep inference eligibility, stickiness, HRP, execution, and costs common across candidates.

The research recipe changes several things together. If it wins consistently, then isolate training-window length, validation design, and missing-session handling individually. That is more informative than immediately expanding the experiment list.

**4. Evaluate through time, with genuinely unseen outcomes.**

At each historical training date:

1. Train using only information available then.
2. Select the checkpoint using a subsequent validation window whose outcomes are already known by that date.
3. Freeze it and evaluate the following period.
4. Advance using the intended production retraining schedule.

Keep target periods from crossing training/validation boundaries. Learn preprocessing and feature selection within the training process. This follows rolling-origin evaluation, where future observations cannot enter earlier model fits. [Hyndman’s explanation](https://otext.robjhyndman.com/hyndsight/tscv/)

Run several predetermined random seeds and report their spread; do not select the seed with the best test return.

**The periods we repeatedly inspected across 27 experiments are now research data.** Historical walk-forward results remain useful, but those periods cannot become a fresh holdout merely by renaming them. Final confirmation needs untouched data, potentially collected prospectively in paper trading. Repeated strategy selection against the same history creates backtest-overfitting risk. [Bailey and colleagues](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

**5. Use statistical tests that match the portfolio question.**

For each matched week, calculate:

\[
d_t = r_{\text{challenger, net},t}-r_{\text{baseline, net},t}
\]

Also calculate each strategy’s net excess over SPY.

Our questions become:

- Does the challenger improve on production?
- Does it outperform SPY?
- Is the improvement large enough to matter after costs and risk?

I would use a **paired block bootstrap across weeks**, preserving common market periods and some serial dependence, to estimate uncertainty. Apply a multiple-comparison correction such as Holm at your chosen **5% significance level** for the predefined comparisons.

Ordinary ANOVA is not my first choice: these strategies share market weeks, and weekly returns can be dependent and heavy-tailed. Hundreds of stocks in one week do not supply hundreds of independent observations of portfolio performance. A nonsignificant result means insufficient evidence, not proof that the strategies are equivalent.

Alongside returns, report drawdown, turnover, cost sensitivity, performance across periods, and dependence on a few winning stocks. Ranking metrics help diagnose the filter; they do not replace portfolio results.

**My recommendation is to reconcile production first, run this compact comparison, then shadow the strongest challenger alongside production.** Promote only when the improvement survives realistic execution, multiple periods, seed variation, and prospective observation. Beating SPY every week is unrealistic; the objective is a repeatable net advantage over a predefined horizon with acceptable risk.
