# Plan: Full-universe US PatchTST patch-geometry sweep

## 1. Goal

Run an auditable, non-production research experiment that isolates PatchTST patch length and stride on the current eligible `halal_new` universe. Compare 8/4, 10/5, and 16/8 geometries with every other data, architecture, optimization, seed, and evaluation choice frozen; treat 2024–2025 as development evidence and preserve available 2026 labels as confirmatory evidence until all confirmatory checkpoints are frozen.

## 2. Domain glossary (ubiquitous language)

- **Patch geometry**: the `(patch_length, patch_stride)` pair and its derived unpadded token count for a 60-session context.
- **Eligible symbol-week**: one current `halal_new` symbol with every exact XNYS close session required for its 60-session context and five-session target, with all required closes finite and positive.
- **Development fold**: an annual expanding-window evaluation whose 2024 or 2025 labels were already opened by the committed audit.
- **Confirmatory fold**: the 2026 annual expanding-window evaluation whose labels remain unread until all nine arm/seed checkpoints are frozen.
- **Causal mean gate**: the per-symbol historical mean weekly log return, initialized only from train and validation rows and updated after each evaluation week.
- **Ridge gate**: `Ridge(alpha=1.0)` fitted only on fold train plus validation rows using close-only lag/volatility features.
- **Paired block uncertainty**: a deterministic 2,000-repetition, four-decision-week moving-block bootstrap of paired weekly metric differences.
- **Research clearance**: a reporting verdict only; it never promotes an artifact or changes a production pointer.

## 3. Investigation findings (facts, with sources)

- Existing reusable code: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_full_universe_audit_2026_08_23/experiment_data.py:load_or_download_prices` records requested symbols, dates, provider/version, missing symbols, per-symbol CSV hashes, and adjusted-data provenance. Its production dependency is `/Users/razin/personal/LearnFinance-2025/brain_api/brain_api/core/prices.py:load_prices_yfinance`.
- Existing reusable code: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_full_universe_audit_2026_08_23/experiment_metrics.py` establishes the rank-IC, top-15 excess, top15-bottom15 spread, MAE, direction, overlap, turnover, and moving-block-bootstrap formulas. The new study preserves those formulas and extends paired uncertainty to turnover/stability transitions without mutating the committed evidence code or invalidating its hashes.
- Existing evidence: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_full_universe_audit_2026_08_23/REPORT.md` shows the earlier 10/5 versus 16/8 comparison changed pooling, dropout, channels/objective, and therefore cannot identify geometry. It also shows the causal historical mean and ridge controls beat both prior ensembles.
- Existing evidence: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_corrected_experiments_2026_08_23/results/selection.json` selected the close-only architecture over the two five-channel architectures on validation rank IC. No repository evidence establishes a stronger unconfounded channel choice, so this geometry study uses one close-return channel.
- Existing evidence: commit `a5843420524285cf44f3fb900f605d93e8337ec4` is patch-equivalent to current `main` commit `c4725b7f0f49d72f80918a34c8dc3bfa50f9dc3e`; `git cherry` reports `- a584342...` and the two trees are identical. No cherry-pick is required.
- Universe evidence: `/Users/razin/personal/LearnFinance-2025/brain_api/data/cache/universe/halal_new_2026-08.json` contains 430 unique Alpaca-tradable current symbols, fetched at `2026-08-08T14:37:54.382306+00:00`, SHA-256 `e2ed929ce96e21e61b46b135a36b5d6cd36874e016d38fccf6fdbbc31c2308b2`.
- Confirmatory availability: prior raw-price requests ended at `2026-01-06` exclusive and prior evaluation ended on decision date `2025-12-22`. No committed result reads 2026 decision labels, so 2026 can be genuinely confirmatory if configuration is frozen before the new download/panel unlock.
- Runtime convention: Python 3.12.13, PyTorch 2.9.1, Transformers 4.57.3, exchange-calendars 4.11.3, yfinance 1.0; Apple MPS is built and available outside the filesystem sandbox on the Apple M5 Pro host.
- Library fact: Transformers 4.57.3 names the stride field `patch_stride`, and the unpadded count is `floor((context_length - patch_length) / patch_stride) + 1` ([version-pinned source](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/patchtst/configuration_patchtst.py)).
- Calendar fact: exchange-calendars exposes `get_calendar("XNYS")` and `sessions_in_range` for regular-trading session labels ([official repository documentation](https://github.com/gerrymanoim/exchange_calendars/blob/master/README.md)).
- Determinism fact: PyTorch requires all RNGs to be seeded and `torch.use_deterministic_algorithms(True)` to reject known nondeterministic operations; exact reproducibility is not guaranteed across releases or platforms ([PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)).
- MPS fact: PyTorch documents moving models and tensors to `torch.device("mps")` for Metal acceleration ([PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)).
- Community gotcha: deterministic algorithms can materially slow MPS workloads, so runtime must be measured rather than inferred ([PyTorch issue #122394](https://github.com/pytorch/pytorch/issues/122394)).
- Provider gotcha: yfinance adjusted-price semantics have historically been ambiguous, so every raw extract is hashed and the report remains explicit that Yahoo is revision-prone ([yfinance issue #687](https://github.com/ranaroussi/yfinance/issues/687)).
- Conventions observed: pytest tests pure research functions; Ruff is configured in `/Users/razin/personal/LearnFinance-2025/brain_api/pyproject.toml`; imports are absolute within these standalone research scripts; no schema tests are permitted; every Python source file must remain under 600 lines.

## 4. Design decisions (each stated as fact, no options)

- Decision: create a new sibling research directory and never modify the committed audit scripts or artifacts. Rationale: their source/artifact hashes are evidence. Layer: research application.
- Decision: use adjusted close log returns only. A window requires one pre-context close, 60 context closes, and five target closes on exact XNYS sessions; non-close OHLCV fields do not affect eligibility. Rationale: this is the cleanest one-channel geometry study. Layer: research domain.
- Decision: freeze `context_length=60`, `prediction_length=5`, `num_input_channels=1`, two layers, `d_model=64`, four heads, FFN 128, shared embedding/projection, channel attention false, batch norm (`eps=1e-5`), pre-norm, GELU, bias true, fixed sin/cos positions, no CLS token, per-series `scaling="std"`, masking disabled, `pooling_type="mean"`, attention dropout 0.20, positional dropout 0.20, path/FFN/head dropout 0, init std 0.02, point MSE head, and daily close MSE. Rationale: this is the corrected-control recipe, with one channel; mean pooling keeps head width constant across token counts. Layer: research domain.
- Decision: the three arms are `patch_8_stride_4` (14 unpadded patches), `patch_10_stride_5` (11), and `patch_16_stride_8` (6). No stride-1 arm exists. Layer: research domain.
- Decision: freeze Adam (`lr=3e-4`, betas 0.9/0.999, eps `1e-8`, weight decay 0), batch 256, gradient clip 1.0, epoch cap 60, `ReduceLROnPlateau(mode="min", factor=0.5, patience=5)`, early-stop patience 8, and checkpoint choice by maximum validation weekly rank IC with validation MSE tie-break. Layer: research application.
- Decision: declare seeds `20260823`, `20260824`, `20260825`; average the three seed predictions for each arm/fold ensemble while reporting each seed separately. Layer: research application.
- Decision: use annual expanding folds: `development_2024` train `2015-05-04..2022-12-19`, validation `2023-01-09..2023-12-18`, evaluation `2024-01-08..2024-12-23`; `development_2025` train `2015-05-04..2023-12-18`, validation `2024-01-08..2024-12-16`, evaluation `2025-01-06..2025-12-22`; `confirmatory_2026` train `2015-05-04..2024-12-16`, validation `2025-01-06..2025-12-22`, evaluation `2026-01-12..2026-08-17`. Each adjacent block has a 21-calendar-day decision-date embargo. Layer: research domain.
- Decision: request prices from `2015-01-01` through `2026-08-22` exclusive, making `2026-08-17` the final decision whose five XNYS target sessions can end on `2026-08-21`. Layer: research infrastructure.
- Decision: block confirmatory label access until all nine 2026 arm/seed checkpoint files and metadata hashes exist; write `results/confirmatory_unlock.json` before rebuilding the panel with labels. Layer: research application.
- Decision: the causal mean gate uses train+validation symbol histories and updates only after predicting each evaluation week. The ridge gate uses close-only features `past_week_log_return`, `momentum_4w_log_return`, `context_log_return`, and `volatility_4w`, fits once per fold on train+validation, and never sees evaluation labels. Layer: research domain.
- Decision: report MAE, RMSE, direction accuracy, balanced direction, weekly rank IC, rank-IC information ratio, top-15 excess, top15-bottom15 spread, top-15 overlap, and top-15 turnover. Run paired 2,000-repetition/four-week moving-block intervals for rank IC, excess, spread, MAE, overlap, and turnover between every arm ensemble, between each arm and both mandatory gates, and for pooled development versus confirmatory evidence. Layer: research domain.
- Decision: research clearance requires positive rank IC in both development years and 2026, positive paired rank-IC deltas versus both gates on pooled development and confirmatory evidence, confirmatory 95% lower bounds above zero versus both gates, and nonnegative confirmatory point deltas for top-15 excess and spread. Failure of any condition is a rejection; no artifact can be promoted by this runner. Layer: research policy.
- Decision: enforce a nonblocking filesystem lock for the entire runner and execute an explicit nested `for fold -> arm -> seed` loop. One model is moved to MPS, trained/evaluated, moved to CPU, deleted, synchronized, and cache-cleared before the next job. No multiprocessing, parallel tests, asynchronous training, or production code path is invoked. Layer: research application.
- Decision: run a deterministic one-step MPS smoke twice with identical seed/input, require identical CPU state hashes and predictions, and save the sanitized hardware/runtime result before full training. Layer: research infrastructure.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_spec.py` defines `PatchGeometry`, `EvaluationFold`, `PATCH_GEOMETRIES`, `EVALUATION_FOLDS`, `build_patchtst_model`, `patch_count`, seed/runtime/hash helpers.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_panel.py` defines exact-session close-only panel construction and label locking; no reusable production function has this exact research contract.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_training.py` defines deterministic single-model training, fingerprinted resume, prediction, and MPS smoke behavior; the prior trainer is five-channel/arm-objective coupled and is not reusable without changing the committed evidence.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_metrics.py` defines controls, stability transitions, pooled metrics, paired uncertainty, and clearance verdict while preserving prior metric formulas.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/experiment_data.py` retains the audited acquisition contract but is standalone so module resolution cannot bind to the prior audit's five-channel `experiment_spec`; it continues to call the production batched yfinance loader.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/run_experiment.py` orchestrates data locking, the sequential fold/arm/seed loop, controls, evaluation, and manifests without touching production.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/test_experiment.py` contains deterministic behavioral regression tests only, never schema tests.
- New artifacts: `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/REPORT.md` records configuration before results, actual data eligibility, all results/uncertainty, limitations, and the no-promotion verdict.
- File-size check: projected source sizes are `geometry_spec.py` <220, `geometry_panel.py` <240, `geometry_training.py` <360, `geometry_metrics.py` <360, `experiment_data.py` <230, `run_experiment.py` <590, and `test_experiment.py` <400 lines. No touched source file exceeds 600 lines.

## 5. Open questions

NONE — all resolved in investigation and the delegated constraints.

## 6. Step-by-step implementation (TDD-ordered, atomic)

### Step 1 — Lock geometry and walk-forward contracts

- Test first: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/test_experiment.py` with `test_patch_geometry_is_the_only_arm_difference`, `test_patch_counts_are_14_11_6`, and `test_expanding_folds_are_chronological_and_embargoed`. Run `/Users/razin/personal/LearnFinance-2025/brain_api/.venv/bin/python -m pytest scratch/patchtst_geometry_sweep_2026_08_23/test_experiment.py -q` from the repository root; expect RED because the module does not exist.
- Implement: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_spec.py` with immutable `PatchGeometry` and `EvaluationFold`, the exact constants in section 4, `patch_count(context_length: int, patch_length: int, patch_stride: int) -> int`, `hf_config_for_geometry(geometry: PatchGeometry) -> PatchTSTConfig`, and `build_patchtst_model(geometry: PatchGeometry) -> PatchTSTForPrediction`.
- Verify: rerun the targeted tests; expect GREEN.
- Deployable-state note: production is untouched; the research module is independently importable.

### Step 2 — Enforce exact-session close-only eligibility and label locking

- Test first: add `test_close_only_panel_requires_exact_xnys_sessions`, `test_non_close_fields_cannot_change_close_only_eligibility`, `test_confirmatory_labels_remain_locked`, and `test_panel_rejects_nonpositive_or_nonfinite_close`. Run the targeted file; expect RED.
- Implement: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_panel.py` with `build_fold_panel(prices: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex, fold: EvaluationFold, include_evaluation_labels: bool) -> pd.DataFrame`, `panel_arrays(panel: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]`, exact-session validation, close-only log returns shaped `[sample, 60, 1]` and `[sample, 5, 1]`, exclusion counters, per-symbol counts, and a hard error when locked evaluation arrays are requested.
- Verify: rerun the targeted tests; expect GREEN.
- Deployable-state note: no external data is read and production remains untouched.

### Step 3 — Add auditable acquisition without production cache writes

- Test first: add `test_universe_cache_loader_preserves_all_symbols_and_hash`, `test_verified_price_cache_rejects_request_mismatch`, and `test_price_manifest_records_missing_symbols_and_file_hashes` using temporary files and a stubbed downloader. Run the targeted file; expect RED.
- Implement: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/experiment_data.py` with `load_halal_new_universe_cache`, `load_or_download_prices`, verified cache reads, atomic strict JSON, collision-safe filenames, and complete provider/request/file provenance. Write only inside the new scratch directory.
- Verify: rerun the targeted tests; expect GREEN.
- Deployable-state note: tests use temporary paths; the current production universe cache is read-only.

### Step 4 — Add deterministic single-model training and exclusive execution

- Test first: add `test_training_fingerprint_changes_for_fold_geometry_seed_or_data`, `test_training_lock_rejects_a_second_runner`, `test_training_job_order_is_strictly_fold_arm_seed`, and `test_checkpoint_resume_requires_matching_weight_hash`. Run the targeted file; expect RED.
- Implement: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_training.py` with `TrainingRunLock`, `training_jobs()`, `training_fingerprint(...)`, `train_geometry_seed(...)`, `predict_weekly_log_returns(...)`, checkpoint hash validation, deterministic DataLoader generators, configured optimizer/scheduler/early stopping, per-job runtime/hardware metadata, and explicit MPS cleanup.
- Verify: rerun the targeted tests on CPU; expect GREEN.
- Deployable-state note: tests train only tiny fixtures and never create production model artifacts.

### Step 5 — Add causal gates, stability, uncertainty, and clearance

- Test first: add `test_causal_mean_never_reads_future_evaluation_rows`, `test_ridge_uses_only_close_features_and_pre_evaluation_labels`, `test_paired_block_uncertainty_is_deterministic`, `test_turnover_uncertainty_uses_aligned_transitions`, and `test_clearance_fails_when_either_gate_is_not_credibly_beaten`. Run the targeted file; expect RED.
- Implement: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_metrics.py` with `build_causal_control_frames`, `weekly_metrics`, `aggregate_metrics`, `paired_block_bootstrap`, `pool_fold_predictions`, and `research_clearance`; use average-tie ranks and symbol tie-breaks exactly as in the committed evaluator.
- Verify: rerun the targeted tests; expect GREEN.
- Deployable-state note: all evaluation behavior is deterministic pure code.

### Step 6 — Orchestrate the label lock and sequential experiment

- Test first: add `test_confirmatory_unlock_is_written_after_all_nine_checkpoint_hashes`, `test_runner_never_has_more_than_one_live_model`, and `test_manifest_declares_no_pointer_or_production_side_effects` with monkeypatched lightweight trainers/data. Run the targeted file; expect RED.
- Implement: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/run_experiment.py` with `run(...)` and CLI arguments `--device`, `--universe-cache`, `--max-epochs`, and `--patience`; load the frozen 430-symbol cache, download/verify through `2026-08-22` exclusive, execute development folds, train all nine confirmatory checkpoints against locked labels, atomically write `confirmatory_unlock.json`, unlock/verify identity, evaluate controls and ensembles, bootstrap comparisons, and write strict hashed manifests/predictions/weekly metrics.
- Verify: rerun the targeted tests; expect GREEN.
- Deployable-state note: the runner contains no import of model storage, Temporal, Alpaca, order, email, or promotion modules.

### Step 7 — Verify deterministic MPS before full training

- Test first: add `test_mps_smoke_result_schema_is_sanitized` around the smoke result function; run on CPU fixtures and expect RED before implementation, GREEN after implementation.
- Implement: add `run_mps_determinism_smoke() -> dict[str, object]` to `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/geometry_training.py`; run two identical one-step 8/4 jobs, synchronize after each blocking transfer, compare exact state/prediction SHA-256 values, exclude serial/UUID identifiers, and write `results/mps_smoke.json`.
- Verify: execute the smoke through `/Users/razin/personal/LearnFinance-2025/brain_api/.venv/bin/python` outside the sandbox; abort the full run unless it passes.
- Deployable-state note: only a tiny scratch model reaches MPS.

### Step 8 — Run the full sweep and generate the report

- Execute: run `/Users/razin/personal/LearnFinance-2025/brain_api/.venv/bin/python scratch/patchtst_geometry_sweep_2026_08_23/run_experiment.py --device mps --universe-cache brain_api/data/cache/universe/halal_new_2026-08.json` outside the sandbox. Do not run tests or other heavy jobs while it runs. Poll progress without launching parallel work.
- Verify: require status complete; 27 unique model/fold/seed metadata records; 27 valid weight hashes; 430 requested symbols; explicit downloaded/missing counts; three fold counts and exclusions; 2026 unlock timestamp after all nine confirmatory checkpoint timestamps; all prediction/metric/source hashes valid; `production_current_pointers_touched=false`; and no production data/model pointer modification in `git status`.
- Implement: create `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/REPORT.md` from the frozen plan and actual manifest, including all requested metrics, paired intervals, seed dispersion, folds/samples/exclusions/runtime/hardware/hashes, gate verdicts, and limitations.
- Deployable-state note: results are research-only and no promotion, trade, or Temporal operation occurs.

### Step 9 — Run targeted regression and integrity checks

- Verify: run the new research test file serially, verify all manifest hashes with a read-only integrity command, run Ruff check/format-check on the new Python files, and confirm every new Python source remains under 600 lines.
- Deployable-state note: the research artifact is reproducible and internally consistent.

## 7. Final mandatory TODOs

- [ ] TODO (second-to-last): Fix ALL failing tests in the repository — related or unrelated to this change. The suite must be fully green before completion. Run the full test suite, list every failure, and fix each one.
- [ ] TODO (last): Fix ALL ruff issues in the repository — related or unrelated to this change. Run `ruff check .` and `ruff format .`, then resolve every remaining lint error and warning until `ruff check .` reports zero issues.
