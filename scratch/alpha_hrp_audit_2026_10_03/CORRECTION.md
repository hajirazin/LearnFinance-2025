# Alpha-HRP correction and actual performance — October 3, 2026

**Follow-up:** [MATH_AUDIT.md](MATH_AUDIT.md) compares the original five-task May/July checkpoints, not only the August 14 five-input close-loss checkpoint. It confirms earlier May/June outperformance and documents the loss-scaling, RevIN, BatchNorm and geometry findings with an expanded 1,404-order / 435-price-series archive.

The 15-name rank-band selector is correct in all six post-migration reports. The important coding defect was accepting stale or incomplete price histories as a valid daily sequence. Close-only inputs also unnecessarily depended on complete OHLCV fields. Those paths are repaired locally, along with historical checkpoint/version loading. This does not demonstrate that the current model has a profitable forecasting edge or that its losses will disappear.

This follow-up supersedes the original audit's **target-weight performance estimate** with actual Alpaca paper-account NAV. The original [AUDIT.md](AUDIT.md) retains its read-only findings and email/order evidence. The production code changes described here were made afterward. No account orders, model pointer, allocation history, schedule, or deployed service was changed. No retraining or deployment was performed.

## Actual performance

Period: August 21 close through October 2 close, six completed weeks after the August 23 migration. Source: read-only Alpaca `/v2/account/portfolio/history`, daily timestamps converted to New York dates; benchmark adjusted closes from Yahoo. No deposit/withdrawal (`CSD`/`CSW`) entries were returned for these accounts. Download timestamps and original payloads are in `broker_history.json` and `market_manifest.json`.

| Series | Six-week return |
|---|---:|
| Alpha-HRP actual NAV: $99,715.76 → $94,734.97 | **−4.995%** |
| SAC halal_filtered actual NAV | −5.296% |
| Double HRP reported NAV | −4.162%* |
| SPY | +0.762% |
| QQQ | +5.175% |
| SPUS | +2.963% |
| HLAL | +4.641% |

Alpha-HRP lagged QQQ by **10.170 percentage points** and SPY by **5.757 points**. The original −1.75% diagnostic calculation used newly targeted weights at each prior Friday close; it omitted actual held positions over the weekend, market fills, sizing thresholds, cash, and broker marks. It must not be presented as actual account performance, and the difference must not simply be called trading costs.

| Week ending | Alpha actual NAV | SPY | QQQ |
|---|---:|---:|---:|
| Aug 28 | +1.597% | +0.474% | +0.419% |
| Sep 4 | −1.921% | +0.109% | +0.353% |
| Sep 11 | −3.980% | −0.766% | −0.567% |
| Sep 18 | +3.197% | −0.093% | +0.919% |
| Sep 25 | −4.322% | +1.268% | +3.302% |
| Oct 2 | +0.566% | −0.222% | +0.682% |

The decline was already underway before the channel migration: the HRP account lost **15.461% between July 1 and August 21**, while SPY gained 2.676% and QQQ lost 1.618%. The account may include older strategy variants; this is evidence about the account's timeline, not a controlled five-channel versus one-channel model experiment. It contradicts attributing the entire deterioration to August 23.

## Market conditions downloaded

25 series, January 2025 through October 2, 2026: SPY, QQQ, RSP, IWM, VIX, 10-year yield proxy, TLT, all 11 SPDR sectors, SMH, GLD, and five halal ETFs. All 25 downloaded successfully. They are retrospective, auto-adjusted daily OHLCV. VIX and TNX are index levels, not investable portfolio returns.

The six-week market was narrow:

| Series | Change |
|---|---:|
| Technology (XLK) | +9.128% |
| Semiconductors (SMH) | +12.523% |
| Equal-weight S&P 500 (RSP) | −5.031% |
| Small caps (IWM) | −5.902% |
| Healthcare (XLV) | −4.469% |
| Industrials (XLI) | −5.461% |
| Materials (XLB) | −8.321% |
| Energy (XLE) | −0.702% |
| Long Treasury ETF (TLT) | −4.826% |

VIX moved from 15.13 to 15.31. The TNX level moved from 4.738 to 5.277. This was not a broad market rally: the strong QQQ comparison reflects concentrated technology leadership. A basket concentrated in refiners, pharmaceuticals, software laggards and international ADRs could trail it materially. This is an inference from the observed prices, not proof of why individual selections failed.

Exports: `market_ohlcv.csv`, `market_adjusted_closes.csv`, `market_conditions.csv`, `broker_daily_nav.csv`, `weekly_actual_returns.csv`, and machine-readable `market_and_nav.json`. `market_and_nav.py` regenerates these offline from the downloaded payloads.

## What was corrected

1. **Close-only prices no longer require other fields.** The PatchTST close-only loader retains a valid close when open/high/low/volume is missing, including absent provider columns. Both batch and individual-fetch paths are covered. Other OHLCV consumers retain their full-field validity contract. Close-return feature computation reads close alone.
2. **US inference requires the actual completed exchange sessions.** For a 60-return context it requires the last 61 completed XNYS closes before the target week. Missing or stale sessions, duplicate/unordered daily rows and non-finite/nonpositive return inputs cannot produce a score. Actual holidays remain valid; no prices are interpolated or filled. Below the existing `min_predictions` floor, score-batch still fails rather than selecting a compromised basket. India does not inherit the US calendar.
3. **US training no longer compresses gaps.** Price sessions are aligned before differencing; absent sessions remain NaN. Training skips affected input and target windows, and anchors only at the verified last trading session of a week. Good Friday correctly permits Thursday; a missing ordinary Friday does not become a Thursday anchor.
4. **Artifact architecture is preserved.** New local, HF and snapshot artifacts serialize the model's effective HF architecture. Existing five-channel artifacts without that field reconstruct the known historical adapter's effective `patch_stride=1`. Checkpoint loading stays strict; no weights are reshaped. Current training defaults and config hashes are unchanged.
5. **Cached version requests load that version.** Previously HF download of an explicitly requested, already cached version returned the local `current` model instead. It now loads the requested version directly. An exact local-version read also supports read-only historical comparisons.

The rank-band 15/20 math, HRP weights, trade sizing, order types, model promotion health policy and hyperparameters were not altered.

## August 31: strong evidence of a real input problem

All 30 displayed August 31 email scores reproduce when Friday August 28 is omitted. Five other weekly reports reproduce with fresh histories. Exact original run snapshots are unavailable, so the upstream source of the omission is unresolved; a provider omission or older running code cannot be distinguished conclusively.

| Symbol | Email score | With Friday included |
|---|---:|---:|
| ATEYY | +4.53% | −0.3423% |
| SMCI | +3.98% | −1.0724% |

Using the current checkpoint, downloaded complete Friday history and the actual August 24 carry-set, the same canonical selector changes **six of 15** names:

- Reconstructed fresh-input additions: CTSH, GDDY, ILMN, LH, VLO, WDAY.
- Reconstructed removals versus the email: ATEYY, AXON, INCY, MMYT, ONC, SMCI.

This demonstrates material selection sensitivity to one missing session. It is a retrospective reconstruction, not a production rerun or proof that the replacement basket would have earned superior returns. See `correction_verification.json`.

## What remains unresolved

The model's validation rank IC is 0.02338, and its MSE is 2.34% above the reported hindsight constant-validation-target baseline. On the six reconstructed forward weeks, current-model mean rank IC is negative (−0.0161). This is weak predictive evidence. The earlier five-channel checkpoint also has negative forward rank IC (−0.0410) on those weeks. Restoring five channels is therefore not an established cure. The migration also changed patch geometry, weight decay and checkpoint selection; it was not a controlled channel-only change.

The account changed 37 names over six runs (about 41% of the basket per week), with roughly $595k gross filled turnover. HRP sometimes put 17–27% into one selected stock. Real losses include DASH −13.77%, MMYT −10.57%, EXPGY −13.53% and CRM −11.56% on matched-quantity round trips. A noisy ranking combined with concentration and turnover is a plausible performance explanation. Fixing input integrity does not establish a forecasting edge.

**Double-HRP caveat:** [Corteva's SEC-filed distribution announcement](https://www.sec.gov/Archives/edgar/data/30554/000119312526391369/d71834dex991.htm) specifies one Vylor share per Corteva share, distribution October 1 and ticker VYLR. The account holds 45.3909 CTVA shares but no VYLR position. The downloaded September 23–October 2 activities contain only fills, with no distribution entry. CTVA is marked at $11.92 after the separation. Thus the reported Double-HRP NAV may be missing a corporate-action credit; its economic return should not be compared confidently until reconciled. This does not explain Alpha-HRP's decline: it sold CTVA September 14. No corporate-action credit or manual adjustment was invented or submitted.

Negative cash in several reports and small OTC broker/Yahoo mark differences remain execution/accounting concerns. The current HRP cash is −$235.19. Market-order fill prices can differ from the sizing snapshot, so fixing model history does not impose a hard notional budget. No evidence established a reversed sort, price unit/sign error or current one-channel checkpoint shape mismatch.

## Verification and rollout state

**200 relevant tests passed**, including real model save/reload, US and India inference/training routes, price loaders, rank-IC training tests, model storage policies, HF, forecaster snapshots and SAC storage. Ruff checks passed for changed production files and the new regression test module.

Offline verification also loaded both actual saved US checkpoints without altering `current`: the legacy model's prediction difference versus the original adapter was exactly zero, and all 430 common fresh-input scores on August 24 and September 28 were unchanged. August 31 correctly reflects the complete Friday context.

These are **local source fixes**. The deployed Pi brain_api image has not been rebuilt/restarted, and an already running Mac trainer needs to reload the changed modules before its next train. There is no new model or rollback promotion. Production recovery still needs deployment of these code fixes and evaluation of a normally trained candidate; a return to market outperformance has not been demonstrated.
