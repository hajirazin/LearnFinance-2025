"""``POST /experience/label/ppo-discovery``.

Kept out of ``routes/experience.py`` so that file stays under the 600-line cap.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from brain_api.core.alpaca_client import get_alpaca_client, resolve_alpaca_account
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.ppo_discovery.config import (
    MODEL_TYPE,
    UNIVERSE_NAME,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.rewards import ppo_discovery_reward
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.sac.trade_clock import experience_open_transition
from brain_api.routes.experience import _extract_prior_weights, get_experience_storage
from brain_api.routes.experience_models import LabelExperienceResponse
from brain_api.storage.experience import ExperienceStorage

router = APIRouter()


class PPOLabelRequest(BaseModel):
    run_id: str | None = None


@router.post("/label/ppo-discovery", response_model=LabelExperienceResponse)
def label_ppo_discovery_experience(
    request: PPOLabelRequest,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> LabelExperienceResponse:
    """Label Alpaca-executed records using actual NAV and IBKR costs."""
    today = date.today()
    labeled = skipped = 0
    errors: list[str] = []
    model_type = MODEL_TYPE
    if request.run_id:
        run_id = request.run_id
        if not run_id.endswith(f":{model_type}"):
            run_id = f"{run_id}:{model_type}"
        record = storage.load(run_id)
        records = [record] if record else []
    else:
        records = [r for r in storage.list_unlabeled() if r.model_type == model_type]

    for record in records:
        try:
            if record.universe != UNIVERSE_NAME:
                raise PPODiscoveryError(
                    f"{record.run_id} must set universe={UNIVERSE_NAME!r} explicitly"
                )
            week_end = date.fromisoformat(record.week_end)
            if week_end >= today:
                skipped += 1
                continue
            actual = record.actual_weights
            if not actual:
                account = resolve_alpaca_account(record.model_type, record.universe)
                actual = get_alpaca_client(account).get_portfolio_weights()
            symbols = [
                s for s in actual if s != "CASH" and abs(float(actual[s])) > 1e-12
            ]
            week_start = date.fromisoformat(record.week_start)
            symbol_returns: dict[str, float] = {}
            symbol_prices: dict[str, float] = {}
            if symbols:
                prices = load_prices_yfinance(
                    symbols,
                    week_start - timedelta(days=7),
                    week_end + timedelta(days=7),
                )
                for symbol in symbols:
                    frame = prices.get(symbol)
                    if frame is None or frame.empty:
                        raise PPODiscoveryError(
                            f"missing next-open prices for {symbol}; last-price fill is forbidden"
                        )
                    trade_price, weekly_return = experience_open_transition(
                        frame, week_start, symbol=symbol
                    )
                    symbol_returns[symbol] = weekly_return
                    symbol_prices[symbol] = trade_price
            prior = _extract_prior_weights(record)
            if record.nav_usd is None:
                raise PPODiscoveryError(f"{record.run_id} missing nav_usd")
            reward, _gross, _cost, economic_net_log = ppo_discovery_reward(
                prior_weights=prior,
                target_weights=actual,
                symbol_returns=symbol_returns,
                symbol_prices=symbol_prices,
                nav_usd=record.nav_usd,
                config=PPODiscoveryConfig(),
            )
            record.reward = reward
            record.realized_return = economic_net_log
            record.actual_weights = actual
            record.labeled_at = datetime.now(UTC).isoformat()
            storage.update(record)
            labeled += 1
        except Exception as exc:
            errors.append(f"{record.run_id}: {exc}")
    return LabelExperienceResponse(
        records_labeled=labeled,
        records_skipped=skipped,
        errors=errors,
    )
