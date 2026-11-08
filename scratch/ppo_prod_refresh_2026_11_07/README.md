# Production PPO refresh and local inference diagnosis — 7 November 2026

The normal production training pipeline produced
`v2026-11-07-91ba085dd8c6`. It is now **promoted locally and on HF `main` by
explicit one-time user override**, replacing `v2026-09-12-13a430da1914-seed42`.
The candidate failed the ordinary CAGR/Sharpe comparison. After being told the
exact metrics and the automatic approval-review rejection, the user explicitly
confirmed the production override. The existing post-validation commit function
performed promotion; production validation source and thresholds are unchanged.

HF main promotion commit: `eb21511405b910e7a24e48597df0c61addf7c892`.
`manual_promotion_authorization.json` and the artifact's `promotion_override`
metadata preserve the failed gates and authorization. HF metadata, local
pointer, and remote policy checksum were verified. Active-model `hf_first`
inference returned the same weights as the candidate test, so promotion does
not change the numerical performance graphs.

## Training and publication

- Frozen current universe: 417 symbols, snapshot
  `sha256:a53e9f88030417c93bd616f0121ea02087db0df12cd6b11ea7553ebcef143925`.
- Normal `/train/ppo-discovery/full` endpoint, default configuration, seed 42,
  MPS, end date 2026-11-07. No production source or parameter edits.
- Mandatory news gap fill completed all 148,869 requested symbol/window cells;
  7,789 were initially missing. It reported 23,179 scored events.
- News completed at 02:07 Singapore time. A scratch monitor initially expected
  `completed` instead of the ETL's `complete` status and did not launch training.
  The monitor was corrected and its regression test was verified failing before
  the correction and passing afterward.
- Training actually started at 06:49 and reached candidate publication at
  12:07 Singapore time. There was a 3h17 gap in monitoring timestamps; its cause
  was not established. Locking alone is not evidence of machine sleep.
- The initial job ended with HTTP 401 at upload after saving the full local
  candidate. After the user refreshed credentials, the existing production
  storage uploader published the saved candidate without retraining.
- Candidate checksums all verified. Upload commit:
  `9dd39853dd8164d451d19eef719a6e45bbe505b6`.
- HF `main` before and after upload:
  `caf4377eb4ab84779640d243c5e8d78a97f10f3d`.

| Stored test metric | New candidate | Incumbent |
| --- | ---: | ---: |
| CAGR | 47.9698% | 57.6587% |
| Sharpe | 1.39557 | 1.64759 |
| Maximum drawdown | 28.2828% | 15.2363% |

The candidate clears the 12% CAGR floor but does not beat either incumbent
metric, so the normal promotion API rejected it with HTTP 422. Promotion later
occurred through the separately recorded explicit one-time user override.
These are each artifact's stored test metrics, with different evaluation
windows; this is the prescribed promotion comparison, not a paired experiment.

## Local inference

An unchanged Brain API ran on Mac port 8001 against an isolated scratch data
root. It pinned the candidate through a scratch-only pointer using the normal
`local_first` storage policy. During that initial test, candidate metadata
remained `promoted=false` and the production pointer was unchanged. Its news database was an
independent APFS clone taken under the source news-store lock, with no WAL.

A separate Mac Temporal server ran at `127.0.0.1:17233`, with no production
schedules. The diagnostic worker registered only calendar resolution, PPO
portfolio read, PPO state-building, and PPO inference activities. It never
registered order, execution, experience, reporting or email activities.

Workflow `ppo-local-inference-20261107T043906Z`, decision 2026-11-02, reproduced
the production inference leg. Calendar and portfolio activities succeeded.
State-building reused 4,491 existing news events for all 417 symbols, without
fetching news or running FinBERT, and returned HTTP 200 with 412 eligible assets.

**The Temporal workflow failed before the inference activity:**

```text
grpc: received message larger than max (9103564 vs. 4194304)
```

The state alone is 9,103,285 JSON bytes, or 9,103,314 bytes as a Temporal
protobuf payload. The worker cannot submit this activity result through the
4 MiB gRPC boundary. Once the state-building heartbeat sidecar stops, the
workflow ultimately records an activity heartbeat timeout. The timeout is a
consequence of the rejected result, not evidence of missing news.

Diagnostic gzip size is still 3,880,088 bytes, above Temporal's usual 2 MiB
payload limit. Plain gzip is therefore not sufficient under default limits.
See Temporal's [documented limits](https://docs.temporal.io/troubleshooting).

A separate direct HTTP call, using the exact saved state and digest, passed
candidate inference: 14 selected stocks, cash weight 0.21016160317849297,
weights finite/nonnegative and summing to 1, selected symbols unique and eligible,
no warnings. No trades or emails were submitted. This establishes that the
candidate API works, while the current Temporal transport contract does not.

The production fix should keep the large state inside Brain/storage and send
a small durable reference through Temporal. State, inference and later
experience consumers must use that reference consistently. No such production
change was made in this operation.

## Pi boundary

An earlier isolation attempt wrongly paused five unrelated Pi schedules and
stopped the shared worker. That was acknowledged and reversed at 01:03
Singapore time: all 12 schedule paused states were checked against the original
snapshot and the originally running shared worker was restarted.

**No further Pi access or changes occurred during training, upload, or local
inference testing.** The Pi inference test was not run: the local Temporal
path has not passed, and the user also explicitly requested no Pi access while
discussing news reuse. The later authorized promotion updated HF `main` and the
Mac pointer without accessing the Pi. Pi inference will see the new HF model
when it next resolves the active artifact; no Pi workflow was triggered here.

## Existing news repository

`hajirazin/learnfinance-news-sentiments` already exists and its URL remains in
the Mac configuration. Commit `08b6a41` on 27 August 2026 removed the news HF
configuration getter and dataset push/pull methods when replacing the old
daily sentiment path with canonical DuckDB news windows. Current ETL has no
HF synchronization. Restoring shared reuse requires article events and exact
coverage/version records, rather than importing old daily aggregates as if
they satisfied today's contract. No sync changes were made here.

## Evidence

### Performance charts and retrospective cutoff issue

The later **old-versus-new replay** in `old_new_week_comparison.png` corrects
the decision-date price leak for both models. It slices every asset and market
frame to sessions strictly before Monday 2 November, leaving Friday 30 October
as the final available session. Both models receive the same saved Monday news
features and 417-symbol roster, but use their own artifact's channel/count
scalers and continued HMM posterior. Both start with $10,000 entirely in cash.
They have identical 412-asset eligibility masks; all outcomes are evaluated
after the allocations are frozen. No artifact or production pointer is changed.

| Corrected 2–6 November replay | Return | Ending value | Stock count | Cash |
| --- | ---: | ---: | ---: | ---: |
| Old PPO, 12 September | -11.6084% | $8,839.16 | 10 | 14.3985% |
| New PPO, 7 November | -13.8043% | $8,619.57 | 14 | 20.8097% |
| SPY | -3.1848% | $9,681.52 | 1 | 0% |
| QQQ | -5.6441% | $9,435.59 | 1 | 0% |

This is the relevant old/new comparison for the week. It supersedes the
Monday-close-contaminated illustration for evaluating pre-open allocations.
It remains retrospective: the roster was resolved on 7 November, not recovered
from Monday's historical cache, and these portfolios were not executed. The new
model remains promoted; the old artifact was loaded by explicit version only.
`old_new_week_summary.json`, `old_new_preopen_inference.json`, and
`old_new_week_values.csv` contain results, weights and plotted values.

A separate mechanical diagnostic also ran both policies on the **exact earlier
saved state**, yielding old -17.6951% versus new -10.0181%. That state includes
Monday close and uses the new model's preprocessing, so this is a transport/
policy comparison, not the fair model-specific pre-open evaluation above.

`weekly_performance.png` shows what the exact saved weights would have done
from Monday 2 November's open through Friday 6 November's close. On a $10,000
all-cash start with the normal IBKR Singapore Tiered entry-cost model, the
portfolio returned -10.0181% versus SPY -3.1848% and QQQ -5.6441%. Cash earns
zero, positions use fractional notionals, and Friday is a valuation rather
than an exit trade. This does not describe an executed portfolio.

Crucially, the allocation was generated on 7 November. The historical state
route requested prices through the decision date inclusive. Its 09:00 Monday
state matches the encoder history through **Monday's close**, not the prior
Friday: maximum absolute BE history error is about 2.18e-14 versus Monday and
17.1564 versus Friday. The figure is explicitly a retrospective, look-ahead-
contaminated illustration, not a valid Monday forecast test. Evidence is in
`weekly_replay_cutoff_check.json`. No production source correction was made.

`historical_performance.png` separately reconstructs the candidate's saved
71-week test from 23 June 2025 to 2 November 2026, using the native holiday-aware
open-to-open trade clock for SPY/QQQ. PPO gained 70.7464% versus SPY 29.3647% and
QQQ 32.7607%; maximum drawdowns were 28.2828%, 7.0687%, and 10.4364%. PPO returns
are after saved modeled costs; historical benchmark buy-and-hold curves are
before trading costs. This test retains current-roster survivorship bias.

Both figures also have SVG exports. CSVs contain the plotted portfolio values
and per-stock weekly contributions. Summary JSONs and `chart_price_manifest.json`
record assumptions and the Yahoo adjusted-price source. Raw prices, Matplotlib
dependencies, fonts, and font caches are isolated under ignored `data/`.

- `preflight.json`, `news_gap_status.json`, `training_job.json`,
  `training_final.json`: native endpoint evidence. The original job remains
  marked failed because publication was repaired separately after training.
- `hf_upload_result.json`, `candidate_assessment.json`: publication and gates.
- `promotion_attempt.json`: normal API rejection; `manual_promotion_result.json`:
  subsequent authorized override. `promoted_inference_verification.json` and
  `promoted_http_inference.json`: successful active-model inference afterward.
- `local_inference_result.json`, `local_state_size.json`: Temporal failure.
- `local_candidate_http_inference.json`, `local_candidate_http_summary.json`:
  successful direct candidate inference.
- `data/local_workflow_history.json`, `data/local_state_response.json`: ignored
  detailed diagnostic evidence. Logs, database clones and model files are also
  ignored. The operations scripts live only in this scratch folder.
