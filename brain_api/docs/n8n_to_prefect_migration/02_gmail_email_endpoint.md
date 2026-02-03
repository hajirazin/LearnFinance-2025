# Plan 02: Gmail Email Endpoint

## Goal

Create `POST /email/send-weekly-report` endpoint that uses Jinja2 to build an HTML email body, then sends it via Gmail SMTP.

## TODOs

- [ ] Create `brain_api/brain_api/routes/email.py` with POST /email/send-weekly-report endpoint
- [ ] Create `brain_api/brain_api/templates/weekly_report_email.html.j2` Jinja2 template
- [ ] Add `GMAIL_USER` and `GMAIL_APP_PASSWORD` env vars to `.env.example`
- [ ] Wire router in `main.py`
- [ ] Write tests for the endpoint in `tests/test_email.py`
- [ ] Run ruff and fix all lint issues
- [ ] Run all tests and fix any failures

## Endpoint Specification

### POST /email/send-weekly-report

**Request**:
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
  },
  "order_results": {
    "ppo": {"orders_submitted": 3, "orders_failed": 0, "skipped": false},
    "sac": {"orders_submitted": 0, "orders_failed": 0, "skipped": true},
    "hrp": {"orders_submitted": 2, "orders_failed": 0, "skipped": false}
  },
  "skipped_algorithms": ["SAC"],
  "target_week_start": "2026-02-03",
  "target_week_end": "2026-02-10",
  "as_of_date": "2026-02-03",
  "to": "hajirazin@gmail.com"
}
```

**Response**:
```json
{
  "sent": true,
  "subject": "Weekly Portfolio Analysis (2026-02-03 -> 2026-02-10)",
  "to": "hajirazin@gmail.com"
}
```

## Implementation Details

### Jinja2 Template (`weekly_report_email.html.j2`)

The template receives the request data and renders an HTML email body with:
- Header with target week dates
- Skipped algorithms warning section (if any)
- Order execution summary table
- AI analysis section with summary paragraphs
- Footer with disclaimer

### Gmail SMTP

- Use `smtplib` with `SMTP_SSL` on port 465
- Server: `smtp.gmail.com`
- Auth: `GMAIL_USER` and `GMAIL_APP_PASSWORD` (app password, not regular password)
- Send as HTML email with proper Content-Type

### Subject Line Format

`Weekly Portfolio Analysis ({target_week_start} -> {target_week_end})`

### Error Handling

- If SMTP connection fails, return 503 with error message
- If authentication fails, return 503 with "Gmail authentication failed"

## Files to Create/Modify

| File | Action |
|------|--------|
| `brain_api/brain_api/routes/email.py` | Create |
| `brain_api/brain_api/templates/weekly_report_email.html.j2` | Create |
| `brain_api/brain_api/main.py` | Add router |
| `brain_api/.env.example` | Add GMAIL_USER, GMAIL_APP_PASSWORD |
| `brain_api/tests/test_email.py` | Create |

## Test Requirements

1. Test successful email send (mock SMTP)
2. Test with skipped algorithms in request
3. Test SMTP failure returns 503
4. Test invalid request returns 422
5. Test email subject format is correct
6. Test HTML body contains expected sections
