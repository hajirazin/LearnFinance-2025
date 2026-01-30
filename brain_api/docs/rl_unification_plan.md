# Unify RL Endpoints: 12 to 6

## Goal

Reduce RL endpoints from 12 to 6 by having each RL agent (PPO, SAC) receive **both** LSTM and PatchTST forecasts as input features.

## Key Principle: Rename + Add, Don't Rewrite

The existing `ppo_patchtst/` and `sac_patchtst/` are already thin wrappers that reuse code from `ppo_lstm/` and `sac_lstm/`. The correct approach is:

1. **Rename** `ppo_lstm/` → `ppo/` (keep all implementation)
2. **Add** dual forecast support to `ppo/`
3. **Delete** `ppo_patchtst/` (no longer needed)
4. Same for SAC

## State Vector Change

- **Current**: 9 features per stock (7 signals + 1 forecast + 1 weight)
- **Proposed**: 10 features per stock (7 signals + 2 forecasts + 1 weight)

```
state_dim = n_stocks * 10 + 1  # (151 for 15 stocks)
```

---

## TODOs (38 total)

### Phase 0: Setup
- [ ] `0-copy-plan`: Copy this plan to brain_api/docs/rl_unification_plan.md

### Phase 1: Core - State Schema
- [ ] `1-state-schema`: Update StateSchema in state.py: add n_forecasts_per_stock=2, update state_dim calculation
- [ ] `2-state-build`: Update build_state_vector() to accept lstm_forecasts and patchtst_forecasts dicts

### Phase 2: Core - PPO Rename
- [ ] `3-ppo-git-mv`: git mv brain_api/core/ppo_lstm to brain_api/core/ppo
- [ ] `4-ppo-init`: Update ppo/__init__.py exports and imports
- [ ] `5-ppo-config`: Update ppo/config.py: rename PPOLSTMConfig to PPOConfig
- [ ] `6-ppo-inference`: Update ppo/inference.py: dual forecast params, update build_state_vector call
- [ ] `7-ppo-data`: Update ppo/data.py: TrainingData accepts both forecast arrays
- [ ] `8-ppo-train`: Update ppo/train.py: rename train_ppo_lstm to train_ppo, use dual forecasts
- [ ] `9-ppo-finetune`: Update ppo/finetune.py: rename finetune_ppo_lstm to finetune_ppo
- [ ] `10-ppo-patchtst-delete`: Delete brain_api/core/ppo_patchtst/ directory

### Phase 3: Core - SAC Rename
- [ ] `11-sac-git-mv`: git mv brain_api/core/sac_lstm to brain_api/core/sac
- [ ] `12-sac-init`: Update sac/__init__.py exports and imports
- [ ] `13-sac-config`: Update sac/config.py: rename SACLSTMConfig to SACConfig
- [ ] `14-sac-inference`: Update sac/inference.py: dual forecast params
- [ ] `15-sac-training`: Update sac/training.py: rename functions, use dual forecasts
- [ ] `16-sac-patchtst-delete`: Delete brain_api/core/sac_patchtst/ directory

### Phase 4: Walkforward
- [ ] `17-walkforward`: Add build_dual_forecast_features() function to walkforward.py

### Phase 5: Routes - Inference
- [ ] `18-route-inf-ppo`: Rename routes/inference/ppo_lstm.py to ppo.py, update endpoint and schema
- [ ] `19-route-inf-sac`: Rename routes/inference/sac_lstm.py to sac.py, update endpoint and schema
- [ ] `20-route-inf-delete`: Delete routes/inference/ppo_patchtst.py and sac_patchtst.py

### Phase 6: Routes - Training
- [ ] `21-route-train-ppo`: Rename routes/training/ppo_lstm.py to ppo.py, update endpoints
- [ ] `22-route-train-sac`: Rename routes/training/sac_lstm.py to sac.py, update endpoints
- [ ] `23-route-train-delete`: Delete routes/training/ppo_patchtst.py and sac_patchtst.py
- [ ] `24-main-routers`: Update main.py router registrations (remove old, add new)

### Phase 7: Storage
- [ ] `25-storage-ppo`: Rename storage/ppo_lstm/ to ppo/, update class names
- [ ] `26-storage-sac`: Rename storage/sac_lstm/ to sac/, update class names
- [ ] `27-storage-delete`: Delete storage/ppo_patchtst/ and sac_patchtst/

### Phase 8: Tests
- [ ] `28-test-ppo`: Rename test_ppo_lstm.py to test_ppo.py, update for dual forecasts
- [ ] `29-test-sac`: Rename test_sac_lstm.py to test_sac.py, update for dual forecasts
- [ ] `30-test-delete`: Delete test_ppo_patchtst.py and test_sac_patchtst.py

### Phase 9: n8n & Docs
- [ ] `31-n8n-endpoints`: Update n8n workflow endpoint URLs (/ppo_lstm to /ppo, etc.)
- [ ] `32-n8n-delete-nodes`: Delete POST SAC+LSTM and POST PPO+PatchTST nodes from workflow
- [ ] `33-n8n-code-nodes`: Update Prepare Phase 2, OpenAI Prompt, Email Body code nodes
- [ ] `34-n8n-rename`: Rename account/node references (PPO_LSTM to PPO, SAC_PatchTST to SAC)
- [ ] `35-docs`: Update CLAUDE.md endpoint tables and code structure section

### Phase 10: Final Cleanup
- [ ] `36-fix-ruff`: Run ruff check and fix all linting errors (related or unrelated)
- [ ] `37-fix-tests`: Run pytest and fix all failing tests (related or unrelated)

---

## Phase 1: Core Layer Changes

### Step 1.1: Update StateSchema for dual forecasts

File: `brain_api/core/portfolio_rl/state.py`

Changes:

- Add `n_forecasts_per_stock: int = 2` property
- Update `n_forecast_features` to return `n_stocks * 2`
- Update `build_state_vector()` to accept `lstm_forecasts` and `patchtst_forecasts` dicts
- Add `get_lstm_forecast_indices()` and `get_patchtst_forecast_indices()` methods

### Step 1.2: Rename ppo_lstm/ to ppo/

```bash
git mv brain_api/brain_api/core/ppo_lstm brain_api/brain_api/core/ppo
```

Update all internal imports within `ppo/` files:

- `from brain_api.core.ppo_lstm` → `from brain_api.core.ppo`
- Rename `PPOLSTMConfig` → `PPOConfig`
- Rename `train_ppo_lstm()` → `train_ppo()`
- Rename `finetune_ppo_lstm()` → `finetune_ppo()`

### Step 1.3: Update ppo/inference.py for dual forecasts

Change function signature:

```python
def run_ppo_inference(
    ...
    lstm_forecasts: dict[str, float],      # NEW
    patchtst_forecasts: dict[str, float],  # NEW (replaces forecast_features)
    ...
)
```

### Step 1.4: Update ppo/data.py and ppo/train.py for dual forecasts

- `build_training_data()` accepts both forecast arrays
- `train_ppo()` generates both LSTM and PatchTST walk-forward forecasts

### Step 1.5: Delete ppo_patchtst/

```bash
rm -rf brain_api/brain_api/core/ppo_patchtst
```

### Step 1.6: Repeat for SAC (Steps 1.2-1.5)

- Rename `sac_lstm/` → `sac/`
- Update `sac/inference.py` for dual forecasts
- Update `sac/training.py` for dual forecasts
- Delete `sac_patchtst/`

---

## Phase 2: Walkforward Changes

### Step 2.1: Add build_dual_forecast_features() function

File: `brain_api/core/portfolio_rl/walkforward.py`

Add new function that calls existing `build_forecast_features()` twice:

```python
def build_dual_forecast_features(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    use_lstm_snapshots: bool = False,
    use_patchtst_snapshots: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build both LSTM and PatchTST forecast features."""
    lstm = build_forecast_features(..., forecaster_type="lstm", ...)
    patchtst = build_forecast_features(..., forecaster_type="patchtst", ...)
    return lstm, patchtst
```

---

## Phase 3: Routes Layer Changes

### Step 3.1: Rename inference routes

```bash
git mv brain_api/brain_api/routes/inference/ppo_lstm.py brain_api/brain_api/routes/inference/ppo.py
git mv brain_api/brain_api/routes/inference/sac_lstm.py brain_api/brain_api/routes/inference/sac.py
rm brain_api/brain_api/routes/inference/ppo_patchtst.py
rm brain_api/brain_api/routes/inference/sac_patchtst.py
```

### Step 3.2: Update inference/ppo.py

- Change endpoint path: `/ppo_lstm` → `/ppo`
- Update request schema: add `patchtst_forecasts` field (both forecasts provided by n8n)
- Call `run_ppo_inference()` with both forecasts

### Step 3.3: Update inference/sac.py

Same pattern as PPO.

### Step 3.4: Rename training routes

```bash
git mv brain_api/brain_api/routes/training/ppo_lstm.py brain_api/brain_api/routes/training/ppo.py
git mv brain_api/brain_api/routes/training/sac_lstm.py brain_api/brain_api/routes/training/sac.py
rm brain_api/brain_api/routes/training/ppo_patchtst.py
rm brain_api/brain_api/routes/training/sac_patchtst.py
```

### Step 3.5: Update training/ppo.py and training/sac.py

- Change endpoint paths: `/ppo_lstm/full` → `/ppo/full`, etc.
- Call `build_dual_forecast_features()` instead of single forecast

### Step 3.6: Update main.py router registrations

Remove old routers, add new unified routers.

---

## Phase 4: Storage Layer Changes

### Step 4.1: Rename storage directories

```bash
git mv brain_api/brain_api/storage/ppo_lstm brain_api/brain_api/storage/ppo
git mv brain_api/brain_api/storage/sac_lstm brain_api/brain_api/storage/sac
rm -rf brain_api/brain_api/storage/ppo_patchtst
rm -rf brain_api/brain_api/storage/sac_patchtst
```

### Step 4.2: Update storage class names

- `PPOLSTMStorage` → `PPOStorage`
- `SACLSTMStorage` → `SACStorage`

---

## Phase 5: Test Updates

### Step 5.1: Update test files

- Rename `test_ppo_lstm.py` → `test_ppo.py`
- Rename `test_sac_lstm.py` → `test_sac.py`
- Delete `test_ppo_patchtst.py`, `test_sac_patchtst.py`
- Update test assertions for dual forecast state dimension (151 instead of 136)

---

## Phase 6: n8n Workflow Updates

### Step 6.1: Update API endpoint URLs

- `/inference/ppo_lstm` → `/inference/ppo`
- `/inference/sac_patchtst` → `/inference/sac`

### Step 6.2: Simplify workflow (delete 2 RL inference nodes)

Delete nodes:

- `POST SAC+LSTM`
- `POST PPO+PatchTST`

Keep and update:

- `POST PPO+LSTM` → `POST PPO` (URL: `/inference/ppo`)
- `POST SAC+PatchTST` → `POST SAC` (URL: `/inference/sac`)

### Step 6.3: Rename account references

- `PPO_LSTM` → `PPO` (credential ID unchanged: `YVMuh5FJ60TmYEVe`)
- `SAC_PatchTST` → `SAC` (credential ID unchanged: `LHsT9blDwa8KyKdR`)

---

## Files Changed Summary

| Action | File/Directory |
| ------ | --------------------------------------------------- |
| UPDATE | `brain_api/core/portfolio_rl/state.py` |
| UPDATE | `brain_api/core/portfolio_rl/walkforward.py` |
| RENAME | `brain_api/core/ppo_lstm/` → `brain_api/core/ppo/` |
| RENAME | `brain_api/core/sac_lstm/` → `brain_api/core/sac/` |
| DELETE | `brain_api/core/ppo_patchtst/` |
| DELETE | `brain_api/core/sac_patchtst/` |
| RENAME | `brain_api/routes/inference/ppo_lstm.py` → `ppo.py` |
| RENAME | `brain_api/routes/inference/sac_lstm.py` → `sac.py` |
| DELETE | `brain_api/routes/inference/ppo_patchtst.py` |
| DELETE | `brain_api/routes/inference/sac_patchtst.py` |
| RENAME | `brain_api/routes/training/ppo_lstm.py` → `ppo.py` |
| RENAME | `brain_api/routes/training/sac_lstm.py` → `sac.py` |
| DELETE | `brain_api/routes/training/ppo_patchtst.py` |
| DELETE | `brain_api/routes/training/sac_patchtst.py` |
| RENAME | `brain_api/storage/ppo_lstm/` → `ppo/` |
| RENAME | `brain_api/storage/sac_lstm/` → `sac/` |
| DELETE | `brain_api/storage/ppo_patchtst/` |
| DELETE | `brain_api/storage/sac_patchtst/` |
| RENAME | `tests/test_ppo_lstm.py` → `test_ppo.py` |
| RENAME | `tests/test_sac_lstm.py` → `test_sac.py` |
| DELETE | `tests/test_ppo_patchtst.py` |
| DELETE | `tests/test_sac_patchtst.py` |
| UPDATE | `n8n/workflows/weekly-lstm-forecast-email.json` |
| UPDATE | `CLAUDE.md` (endpoint tables) |

---

## Verification

After each phase, run:

```bash
cd brain_api
uv run ruff check brain_api/
uv run pytest tests/ -v
```
