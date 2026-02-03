# Plan 01: OpenAI Summary Endpoint

## Goal

Create `POST /summary/generate` endpoint that uses Jinja2 to build a prompt, calls OpenAI GPT-4o-mini, and returns a parsed JSON summary.

## TODOs

- [ ] Create `brain_api/brain_api/routes/summary.py` with POST /summary/generate endpoint
- [ ] Create `brain_api/brain_api/templates/` directory
- [ ] Create `brain_api/brain_api/templates/weekly_report_prompt.j2` Jinja2 template
- [ ] Add `OPENAI_API_KEY` env var to `.env.example`
- [ ] Add `jinja2` and `openai` to dependencies in `pyproject.toml` (if not already present)
- [ ] Wire router in `main.py`
- [ ] Write tests for the endpoint in `tests/test_summary.py`
- [ ] Run ruff and fix all lint issues
- [ ] Run all tests and fix any failures

## Endpoint Specification

### POST /summary/generate

**Request**:
```json
{
  "lstm": {
    "predictions": [{"symbol": "AAPL", "predicted_return": 0.025}],
    "model_version": "v2026-01-09-a4fecab1bdcc",
    "target_week_start": "2026-02-03",
    "target_week_end": "2026-02-10"
  },
  "patchtst": {
    "predictions": [{"symbol": "AAPL", "predicted_return": 0.018}],
    "model_version": "v2026-01-09-b5fecab1bdcc"
  },
  "news": {
    "per_symbol": [{"symbol": "AAPL", "sentiment_score": 0.65}]
  },
  "fundamentals": {
    "per_symbol": [{"symbol": "AAPL", "ratios": {"gross_margin": 0.43}}]
  },
  "hrp": {
    "percentage_weights": {"AAPL": 10.5, "MSFT": 8.2},
    "symbols_used": 15
  },
  "sac": {
    "target_weights": {"AAPL": 0.105, "MSFT": 0.082},
    "turnover": 0.12,
    "model_version": "v1"
  },
  "ppo": {
    "target_weights": {"AAPL": 0.095, "MSFT": 0.088},
    "turnover": 0.08,
    "model_version": "v1"
  },
  "order_results": {
    "ppo": {"orders_submitted": 3, "orders_failed": 0, "skipped": false},
    "sac": {"orders_submitted": 0, "orders_failed": 0, "skipped": true},
    "hrp": {"orders_submitted": 2, "orders_failed": 0, "skipped": false}
  },
  "skipped_algorithms": ["SAC"]
}
```

**Response**:
```json
{
  "summary": {
    "para_1_overall_summary": "The portfolio shows balanced allocation across technology and consumer sectors.",
    "para_2_sac": "SAC was skipped due to open orders.",
    "para_3_ppo": "PPO favors AAPL and MSFT with moderate turnover.",
    "para_4_hrp_summary": "HRP maintains diversified risk parity allocation.",
    "para_5_patchtst_forecast": "PatchTST predicts positive returns for tech sector.",
    "para_6_lstm_forecast": "LSTM shows bullish signals for AAPL.",
    "para_7_news_sentiment": "News sentiment is neutral to positive.",
    "para_8_fundamentals": "Strong fundamentals in selected stocks."
  }
}
```

## Implementation Details

### Jinja2 Template (`weekly_report_prompt.j2`)

The template receives the full request data and renders a prompt string for OpenAI. The prompt instructs GPT-4o-mini to return JSON with 8 paragraph fields.

### OpenAI Call

- Model: `gpt-4o-mini`
- Max tokens: 2500
- Temperature: 0.3
- Parse response as JSON

### Error Handling

- If OpenAI call fails, return 503 with error message
- If JSON parsing fails, return summary with `para_1_overall_summary: "Unable to generate AI summary"`

## Files to Create/Modify

| File | Action |
|------|--------|
| `brain_api/brain_api/routes/summary.py` | Create |
| `brain_api/brain_api/templates/weekly_report_prompt.j2` | Create |
| `brain_api/brain_api/main.py` | Add router |
| `brain_api/.env.example` | Add OPENAI_API_KEY |
| `brain_api/pyproject.toml` | Add dependencies |
| `brain_api/tests/test_summary.py` | Create |

## Test Requirements

1. Test successful summary generation (mock OpenAI response)
2. Test with skipped algorithms in request
3. Test OpenAI failure returns 503
4. Test invalid request returns 422
5. Test empty predictions handled gracefully
