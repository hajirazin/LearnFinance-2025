# Ridge regression with one year of inputs

Compare two pooled ridge regressions on the one-year question from H2 and H4. Fit both on the same historical stock-week examples so the training roster does not confound the result.

| Model | Input at each decision |
| --- | --- |
| One-channel ridge | Daily adjusted-close log returns in the trailing 365 calendar days |
| Seven-channel ridge | The archived seven daily channels over the same days: 20-day momentum, 12–1 momentum, 20-day volatility, 20-day relative volume, trailing news sentiment, adjusted-close price, and reported volume |

Use the H2/H4 window builder and common seven-feature eligibility for both arms, with the 255-position right-aligned windows. Require all adjusted-close log returns, the latest value of each feature, and at least 80% coverage per feature over the trading sessions in the window. Verified no-news windows have neutral sentiment. Normalize each feature history within its observed window and set masked positions to normalized zero. Flatten the window so ridge can learn a separate linear coefficient at each lag. Both arms predict the same five-session simple return directly.

Fit on stock-weeks with labels that finish by the end of 2024. Choose each ridge penalty by mean weekly stock MAE on 2025 validation. Both arms preferred the largest penalty in the initial grid (`0.0001` through `10`), so extend the grid through `10,000` to check stronger shrinkage. Selection uses only 2025 validation errors. Use the frozen broader research roster for both fits. Reuse the 33 exact five-session Friday pairs, stock labels, and fixed equal-weight 15-name baskets from the [forecast accuracy archive](V1_FORECAST_ACCURACY.md).

The primary metric is absolute return error on the same 15 names, averaged over Fridays. Report the seven-minus-one difference in percentage points, a 95% circular block-bootstrap interval, and a one-sided p-value for improvement (four adjacent observed weeks, 20,000 draws, seed 42). Report all-stock MAE and rank correlation, compare with a zero forecast, and show realized SPY for context. The seven-channel recipe uses adjusted-close price where the one-channel recipe uses a daily return. The comparison measures the two recipes as a whole. The frozen historical roster introduces survivorship bias, and the evaluation dates have informed earlier experiments, so treat this as exploratory.

## Results

Both models fit on 82,399 stock-weeks from 413 stocks and used 20,938 validation stock-weeks. The evaluation covers 13,712 observed stock returns and 33 matched Fridays from January 9 through October 30, 2026. The 2025 validation selected penalties of 10 for one channel and 100 for seven.

| Forecast | 15-stock basket MAE (pp) | Individual-stock MAE (pp) | Mean weekly Spearman |
| --- | ---: | ---: | ---: |
| One channel | 3.783 | 4.190 | −0.0222 |
| Seven channels | 3.781 | 4.184 | +0.0056 |
| Zero return | 3.863 | 4.181 | — |

Seven channels reduced mean basket MAE by 0.0013 percentage points, with lower error on 16 of 33 weeks. The paired difference (seven minus one) was −0.0013 pp, with a 95% interval of −0.037 to +0.034 pp and a one-sided p-value of 0.470. The two feature sets performed similarly on the primary outcome.

Seven channels reduced individual-stock MAE by 0.0059 pp (exploratory 95% interval −0.0103 to −0.0013), and forecasting zero gave the lowest individual-stock MAE. Both ridge forecasts stayed close to zero, and both average rank correlations were close to zero. This comparison gives little reason to prefer the seven-channel recipe for forecasting these baskets.

![Same-basket ridge forecasts, actual returns, SPY, and the error difference versus one channel](figures/ridge/one_year_feature_comparison.png)

[Run manifest and validation grid](data/ridge_lookback/summary.json) · [Weekly basket forecasts](data/ridge_lookback/weekly_forecasts.csv) · [Stock forecasts and actual returns](data/ridge_lookback/stock_forecasts.parquet)

## R-squared, adjusted R-squared, and ANOVA

The saved predictions give the following out-of-sample R² values. Individual-stock scores pool all 13,712 observed stock-week outcomes. Basket scores use the 33 weekly outcomes for the same fixed 15-stock portfolios.

| Forecast | Individual-stock R² | Basket R² | Individual-stock RMSE (pp) | Basket RMSE (pp) |
| --- | ---: | ---: | ---: | ---: |
| One channel | −0.002966 | −0.010707 | 5.8755 | 4.7685 |
| Seven channels | −0.000306 | −0.007543 | 5.8677 | 4.7611 |
| Zero return | −0.000349 | −0.025792 | 5.8679 | 4.8040 |

Compute R² directly from predictions and observations using

\[
R^2 = 1 - \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
\]

One is perfect and zero matches the squared error of predicting the evaluation sample's average actual return. A negative value means larger squared error than that constant. The sample mean is known retrospectively. This benchmark describes model fit. The zero-return forecast remains a feasible forecasting baseline. R² uses squared errors, so its ordering can differ from MAE. [Scikit-learn documentation](https://sklearn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

The classical adjusted statistic with an intercept is

\[
\bar R^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}.
\]

The ordinary regression version accounts for the number of estimated coefficients. The one-channel fit has 255 lag coefficients plus an intercept and the seven-channel fit has 1,785 plus an intercept. Ridge shrinkage changes effective model complexity. For a ridge smoother, effective degrees of freedom can be calculated from the training design and penalty through the trace of the fitted-value operator. The saved test forecasts alone do not determine it. A further adjustment using the 33 aggregated basket outcomes would also mix a stock-level fitted model with basket-level evaluation. Classical adjusted R² is therefore left blank in the metric export. An effective-degrees-of-freedom training diagnostic would be a separate calculation. [Statsmodels adjusted R² definition](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.rsquared_adj.html) · [Ridge degrees of freedom](https://online.stat.psu.edu/stat857/node/155/)

The classical nested-regression ANOVA F test compares a restricted and unrestricted OLS fit under its error assumptions. These ridge fits have separately selected penalties, the one-channel return input is absent from the seven-channel design, and observations share market weeks. A plain nested OLS F test does not establish predictive superiority for these recipes. [Statsmodels assumptions](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.compare_f_test.html)

Compare squared forecast errors using paired weekly circular block resampling. Use the same 20,000 draws, seed 42, primary four-observed-week blocks, and a two-sided null of equal mean squared error. Aggregate individual-stock squared errors within each week. Apply Holm adjustment to the two comparisons, stock and basket. The bootstrap handles the observed sequence of eligible weeks, with holidays and excluded weeks leaving calendar gaps. Two- and eight-observed-week blocks are sensitivity diagnostics.

| Squared-error comparison | Mean seven minus one (pp²) | 95% block interval (pp²) | Two-sided p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| Individual-stock weekly mean | −0.09170 | −0.15734 to −0.02467 | 0.0066 | 0.0132 |
| Fixed 15-stock basket | −0.07119 | −0.38106 to +0.26612 | 0.6692 | 0.6692 |

Seven channels show a small individual-stock squared-error improvement over one channel in this exploratory sample. Basket improvement remains inconclusive. Both models' R² values remain below zero. The adjustment covers the two comparisons in this follow-up, and prior use of these evaluation dates still limits confirmatory interpretation.

Regenerate this follow-up from the saved forecasts with `python scratch/AlphaAllocationV1/analyze_ridge_r_squared.py`. It requires NumPy, pandas, DuckDB, and scikit-learn. [R² table](data/ridge_lookback/r_squared.csv) · [Paired weekly squared-error differences](data/ridge_lookback/squared_error_weekly.csv) · [Summary, source hashes, and block sensitivity](data/ridge_lookback/r_squared_summary.json)

## R-squared against SPY

Use the same 33 matched Fridays and their archived dividend-adjusted SPY simple returns over the exact five-session horizon. Fit a descriptive market regression separately for each forecast series and the actual fixed basket

\[
y_t = a + b\,r_{\mathrm{SPY},t} + e_t.
\]

Here \(y_t\) is the predicted weekly return of the fixed basket or its actual return. Both ridge arms use identical basket holdings for error measurement, so there is one actual-basket series. The forecast rows are model outputs. This analysis uses the weekly series from the forecast archive rather than broker NAV. The intercept uses raw weekly returns, with no risk-free-rate subtraction.

| Series | Market regression R² | Adjusted R² | SPY beta |
| --- | ---: | ---: | ---: |
| One-channel basket forecast | 0.001390 | −0.030823 | −0.001610 |
| Seven-channel basket forecast | 0.022363 | −0.009173 | 0.004314 |
| Actual fixed basket return | 0.353085 | 0.332217 | 1.766494 |

The adjusted statistic applies to this descriptive OLS regression, with one explanatory SPY return and an intercept. Its adjustment is \(1-(1-R^2)32/31\), using 33 observations. The original ridge channels and lag coefficients are upstream of this regression. A negative adjusted value means the fit explains little variation relative to this adjustment. SPY explains 0.14% and 2.24% of the two forecast series' variation, compared with 35.31% of the actual basket's variation. These estimates describe co-movement on the evaluation dates and reuse those dates for fitting the explanatory regression. [Statsmodels R²](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.rsquared.html) · [Adjusted R²](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.rsquared_adj.html)

Direct tracking accuracy asks a different question. Treat realized SPY returns as the target and use the unmodified forecast or actual-basket series as its estimate

\[
R^2_{\mathrm{tracking}} = 1 -
\frac{\sum_t(r_{\mathrm{SPY},t}-y_t)^2}
{\sum_t(r_{\mathrm{SPY},t}-\bar r_{\mathrm{SPY}})^2}.
\]

This gives −0.061871 for the one-channel forecast, −0.041248 for seven channels, and −5.598764 for the actual basket. Tracking R² penalizes level and scale differences, while the fitted market regression estimates an intercept and slope. The ridge targets are stock returns, and these tracking scores measure agreement with SPY. Higher market regression R² indicates stronger co-movement. Performance relative to SPY requires examining excess returns. [Scikit-learn scoring definition](https://sklearn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

The same follow-up command regenerates these calculations from the archived forecasts. [SPY comparison table](data/ridge_lookback/spy_r_squared.csv) · [Method and source hashes](data/ridge_lookback/r_squared_summary.json)

## Reproduce

Run from the repository root with the PatchTST scratch dependencies and a DuckDB version that can read the archived news files. The experiment used Python 3.12.16, DuckDB 2.0.0, NumPy 1.26.4, pandas 2.3.3, and SciPy 1.16.3.

```sh
PYTHONPATH=brain_api:scratch/AlphaAllocationV2 OPENBLAS_NUM_THREADS=4 \
  VECLIB_MAXIMUM_THREADS=4 python scratch/AlphaAllocationV1/compare_ridge_lookback.py
```
