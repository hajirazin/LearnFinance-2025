# Legacy halal: historical PatchTST news readiness

Audit date: November 28, 2026 (Singapore).

## Finding

The 13-symbol cached `halal` roster matches the failed backfill's symbol hash
`15b513161d3fe4d33864feaf2ac19133ae1376cde2ce7e941ebee6f1d2e72beb`.
The job processed all 18,590 requested symbol/session cells and published
its evidence to news commit `1f80bc56523057fbc587d3e00f47b57bc8cebe55`,
but the production causal sentiment reader still rejects 909 cells.

All 909 already have exact coverage records with zero recorded future-revision
exclusions. Independently querying the article revision history shows that each
of those cells includes at least one article whose only stored revisions are
after that cell's historical cutoff. Coverage records alone therefore do not
establish usable historical sentiment. The original text/score as of the cutoff
is not available in the current news event store for those articles.

For example, TSLA article `20281589` was created March 22, 2021 at
17:30:17 UTC, but its earliest stored update is March 24 at 04:55:59 UTC.
It cannot supply its sentiment as of March 22 at 20:00 UTC.

## Actual training failure

Job `sac_halal:v2026-11-27_60b88588` is failed in
`walk_forward_forecasts`, with this error:

> Only 9 PatchTST inputs available for SAC at 2022-10-28

The news-only audit uses the production calendar, causal session sentiment
reader, and the existing requirements of a finite latest observation and at
least 80% observed news values over the 365-calendar-day input window. It
checks all 465 decision samples in the current January 2018 to November 2026
training interval, including the terminal sample used for forecast alignment.
Eight weekly decisions cannot meet the required 10 eligible stocks even before
considering price or other feature failures. News-only counts range from 6 to 13.

| Weekly decision | Forecast input anchor | News-eligible stocks | News-ineligible stocks |
| --- | --- | ---: | --- |
| 2022-10-31 | 2022-10-28 | 9 | AAPL, MSFT, META, TSLA |
| 2022-11-21 | 2022-11-18 | 9 | NVDA, TSM, TSLA, MU |
| 2023-10-30 | 2023-10-27 | 9 | MSFT, GOOGL, META, TSLA |
| 2024-01-02 | 2023-12-29 | 6 | AAPL, NVDA, MSFT, GOOGL, META, TSLA, AMD |
| 2024-01-29 | 2024-01-26 | 9 | NVDA, TSM, TSLA, AMD |
| 2025-06-23 | 2025-06-20 | 7 | AAPL, NVDA, MSFT, GOOGL, META, TSLA |
| 2025-09-22 | 2025-09-19 | 9 | NVDA, MSFT, GOOGL, META |
| 2026-02-09 | 2026-02-06 | 7 | AAPL, NVDA, MSFT, META, TSLA, LLY |

## Consequences

This is a historical revision availability problem in the current evidence,
not a failed upload or simply absent exact coverage records. The existing
materialization service reuses exact covered windows; blindly retrying the same
job will not recover the missing original article revisions.

The evidence already fetched and published is retained. Making this job appear
successful or replacing missing values with zero would not make the blocked
SAC decisions valid. Training still requires at least 10 eligible stocks.
The viable paths are to obtain verifiable article revisions available at those
historical cutoffs, or explicitly approve and evaluate a change to the historical
missing-news/training-data policy. Such a change must not silently alter the
current production contract or use future text at a past cutoff.

## Files and scope

`unusable_sessions.csv` contains all 909 rejected symbol/session cells with
coverage status and counts. `weekly_news_eligibility.json` contains per-decision
news eligibility and the eight blocked cases. All source database reads used
read-only DuckDB connections. This audit did not change production code, news
records, model artifacts, current pointers, or training jobs. Existing uncommitted
work and the user's training-chain script were preserved.


## Approved revision-policy update: November 28, 2026

After the initial strict-policy audit above, the user explicitly approved use
of later-edited article versions for these historical inputs. The shared news
reader now supports this historical policy. It prefers an existing pre-cutoff
revision and allows a stored later revision when an original version is absent.
Article creation-date membership and complete-coverage requirements remain in
force. Live news reads retain their point-in-time behavior.

SAC's historical PatchTST input construction selects the approved policy and
records the revision policy, affected-cell count, and look-ahead-bias flag in
training evidence. The SAC training version identity includes the data policy;
old strict-policy SAC artifacts are not silently reused for a new training run.
The PatchTST model recipe used by live inference is unchanged, and existing
annual snapshot weights and current pointers were preserved.

The original backfill was revalidated and is now complete: 18,590/18,590 cells,
zero unresolved cells, zero newly scored events. Previously stored evidence was
sufficient under the approved policy.

The actual historical forecast-generation function was then run using the
existing annual snapshots, real historical prices, and read-only news evidence.
All 464 training decision steps produced valid forecasts for all 13 stocks;
there were zero blocked decisions and no skipped weeks. The loaded feature grid
used later revisions in 1,018 cells, so the recorded look-ahead-bias flag is true.
See `approved_revision_forecast_check.json`.

The live API's halal SAC preflight returned `ready: true`, `missing: []`, and
`errors: []`; see `approved_revision_preflight.json`. The full offline suite
passed with 1,440 tests passed and four skipped. Ruff lint and formatting passed.
No SAC optimization run was started as part of this verification. Code changes
remain uncommitted.
