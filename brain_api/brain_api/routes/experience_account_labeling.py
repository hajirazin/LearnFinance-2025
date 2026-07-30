"""Account-specific realized SAC experience labeling."""

import logging
from datetime import UTC, date, datetime

from brain_api.core.sac.experience_accounting import compute_realized_sac_reward
from brain_api.core.sac.trade_clock import experience_open_transition
from brain_api.routes.experience_models import (
    ExperienceRecord,
    LabelExperienceResponse,
)
from brain_api.storage.experience import ExperienceStorage

logger = logging.getLogger(__name__)


def compute_reward_from_actual_weights(
    actual_weights: dict[str, float],
    symbol_returns: dict[str, float],
    *,
    prior_weights: dict[str, float] | None = None,
    symbol_prices: dict[str, float] | None = None,
    nav_usd: float | None = None,
    reward_scale: float = 100.0,
) -> tuple[float, float]:
    """Compute reward based on ACTUAL portfolio weights using IBKR-SG costs.

    Differs from the simulator (env.step) in that the rebalance is
    measured against what actually executed (``actual_weights``) vs
    what the policy intended; the cost formula itself is identical
    -- per-symbol per-leg IBKR Singapore Tiered fees in
    :mod:`brain_api.core.portfolio_rl.broker_costs`.

    The legacy "estimated_turnover = 0.1" hack is gone -- the cost is
    now derived from the **actual** per-symbol weight deltas
    (``actual_weights`` - ``prior_weights``) and the **actual**
    per-symbol prices we have on record. If either side is missing
    for a symbol that traded, we raise per AGENTS.md rule #1.

    Args:
        actual_weights: Actual portfolio weights after orders settled.
        symbol_returns: Realized weekly returns for each symbol.
        prior_weights: Pre-rebalance weights (defaults to all-cash).
        symbol_prices: Per-symbol close prices for the rebalance week.
        nav_usd: Total portfolio equity in USD; defaults to the IBKR
            cost config's USD 10k anchor when None.
        reward_scale: Reward scaling factor.

    Returns:
        Tuple of (reward, portfolio_return).
    """
    return compute_realized_sac_reward(
        actual_weights,
        symbol_returns,
        prior_weights=prior_weights,
        symbol_prices=symbol_prices,
        nav_usd=nav_usd,
        reward_scale=reward_scale,
    )


def infer_universe_from_run_id(run_id: str) -> str:
    """Infer the SAC universe from a legacy run_id (no ``universe`` field).

    Used as a one-shot migration aid for experience records written
    before the ``universe`` field existed. The two parallel SAC A/B
    workflows have disjoint run_id prefixes by design (per AGENTS.md
    "Run identity & rerun semantics"):

    - ``paper:halal:YYYY-MM-DD[:sac]`` -> ``halal`` (IBKR-routed; the
      Alpaca labeller has no account for this universe and will
      surface an error on resolve_alpaca_account -- see AGENTS.md
      rule #1)
    - everything else                 -> ``halal_filtered`` (sac account)

    Per AGENTS.md rule #1 the inference is intentionally bounded to the
    two known SAC universes -- a future third bucket would need to land
    a ``universe`` field on the record before its experience is
    written, NOT a silent fallback here.
    """
    if run_id.startswith("paper:halal:"):
        return "halal"
    return "halal_filtered"


def label_experience_for_account(
    model_type: str,
    run_id: str | None,
    storage: ExperienceStorage,
) -> LabelExperienceResponse:
    """Label experience records for a model type using actual weights.

    Routes each record to the correct Alpaca account via
    :func:`resolve_alpaca_account` driven by ``record.universe``. Only
    ``halal_filtered`` is currently Alpaca-routable; ``halal`` records
    are IBKR-routed and MUST carry ``actual_weights`` plumbed in from
    the post-trade IBKR snapshot at write time -- the Alpaca fallback
    cannot serve them and will fail-loud per AGENTS.md rule #1.

    Args:
        model_type: ``"sac"`` (currently the only labeller-supported
            model type; see :func:`resolve_alpaca_account`).
        run_id: Specific run to label, or ``None`` to label every
            unlabeled record for this ``model_type``.
        storage: Experience storage instance.

    Returns:
        LabelExperienceResponse with labeling results.
    """
    from datetime import timedelta

    from brain_api.core.alpaca_client import (
        AlpacaClient,
        get_alpaca_client,
        resolve_alpaca_account,
    )
    from brain_api.core.lstm import load_prices_yfinance
    from brain_api.routes.experience import _extract_prior_weights

    today = date.today()
    records_labeled = 0
    records_skipped = 0
    errors = []

    # Get records to label
    if run_id:
        # Add model_type suffix if not present
        if not run_id.endswith(f":{model_type}"):
            run_id = f"{run_id}:{model_type}"
        record = storage.load(run_id)
        records = [record] if record else []
    else:
        # Get all unlabeled records for this model_type
        all_unlabeled = storage.list_unlabeled()
        records = [r for r in all_unlabeled if r.model_type == model_type]

    logger.info(
        f"[Experience] Found {len(records)} {model_type.upper()} records to potentially label"
    )

    # Cache one client per resolved account so a mixed-universe batch
    # only constructs each Alpaca client once (and so the labeller does
    # not re-read env vars per record).
    client_cache: dict[str, AlpacaClient] = {}

    def _get_client_for_record(rec: ExperienceRecord) -> AlpacaClient:
        universe = rec.universe
        if universe is None:
            universe = infer_universe_from_run_id(rec.run_id)
            logger.warning(
                f"[Experience] Record {rec.run_id} has no universe field; "
                f"inferred universe={universe!r} from run_id prefix. "
                f"This path is for legacy records only -- new SAC writes "
                f"set universe explicitly."
            )
        account = resolve_alpaca_account(rec.model_type, universe)
        cached = client_cache.get(account.value)
        if cached is None:
            cached = get_alpaca_client(account)
            client_cache[account.value] = cached
        return cached

    for record in records:
        try:
            # Check if week has ended
            week_end = date.fromisoformat(record.week_end)
            if week_end >= today:
                logger.info(
                    f"[Experience] Skipping {record.run_id}: week not ended yet"
                )
                records_skipped += 1
                continue

            # Get ACTUAL weights from Alpaca account
            # If we have actual_weights from update-execution, use those
            # Otherwise, fetch current positions (less accurate but fallback)
            if record.actual_weights:
                actual_weights = record.actual_weights
                logger.info(
                    f"[Experience] Using stored actual_weights for {record.run_id}"
                )
            else:
                try:
                    alpaca_client = _get_client_for_record(record)
                    actual_weights = alpaca_client.get_portfolio_weights()
                    logger.info(
                        f"[Experience] Fetched current weights from Alpaca "
                        f"({alpaca_client.account.value}) for {record.run_id}"
                    )
                except ValueError as e:
                    # Unknown (model_type, universe) -> we cannot pick an
                    # account. Per AGENTS.md rule #1, surface as an error
                    # rather than silently labelling against the wrong
                    # account.
                    error_msg = (
                        f"Cannot route {record.run_id} to an Alpaca "
                        f"account (model_type={record.model_type!r}, "
                        f"universe={record.universe!r}): {e}"
                    )
                    logger.error(f"[Experience] {error_msg}")
                    errors.append(error_msg)
                    continue
                except Exception as e:
                    logger.warning(
                        f"[Experience] Failed to fetch Alpaca weights: {e}. "
                        f"Falling back to intended action."
                    )
                    # Fallback to intended action if we can't get actual
                    actual_weights = record.intended_action or record.action

            # Get symbols from actual weights
            symbols = [s for s in actual_weights if s != "CASH"]

            if not symbols:
                logger.warning(
                    f"[Experience] No symbols in actual_weights for {record.run_id}"
                )
                records_skipped += 1
                continue

            # Fetch realized returns
            week_start = date.fromisoformat(record.week_start)
            data_start = week_start - timedelta(days=7)
            data_end = week_end + timedelta(days=7)

            prices = load_prices_yfinance(symbols, data_start, data_end)

            # Compute the holiday-aware first-session-open to next week's
            # first-session-open reward. Transaction costs use the starting
            # rebalance open, not a later close.
            symbol_returns = {}
            symbol_prices: dict[str, float] = {}
            for symbol in symbols:
                df = prices.get(symbol)
                if df is None or df.empty:
                    raise ValueError(
                        f"Missing realized prices for SAC experience symbol {symbol}"
                    )

                trade_price, weekly_return = experience_open_transition(
                    df,
                    week_start,
                    symbol=symbol,
                )
                symbol_returns[symbol] = weekly_return
                symbol_prices[symbol] = trade_price

            prior_weights = _extract_prior_weights(record)
            if record.nav_usd is None:
                logger.warning(
                    f"[Experience] Record {record.run_id} has no nav_usd; "
                    f"falling back to IBKRSingaporeCostConfig default NAV anchor"
                )

            # Compute reward using ACTUAL weights
            reward, realized_return = compute_reward_from_actual_weights(
                actual_weights=actual_weights,
                symbol_returns=symbol_returns,
                prior_weights=prior_weights,
                symbol_prices=symbol_prices,
                nav_usd=record.nav_usd,
            )

            # Update record
            record.reward = reward
            record.realized_return = realized_return
            record.actual_weights = actual_weights
            record.labeled_at = datetime.now(UTC).isoformat()

            storage.update(record)
            records_labeled += 1

            logger.info(
                f"[Experience] Labeled {model_type.upper()} {record.run_id}: "
                f"reward={reward:.4f}, return={realized_return:.4f}"
            )

        except Exception as e:
            error_msg = f"Error labeling {record.run_id}: {e}"
            logger.error(f"[Experience] {error_msg}")
            errors.append(error_msg)

    logger.info(
        f"[Experience] {model_type.upper()} labeling complete: "
        f"{records_labeled} labeled, {records_skipped} skipped, {len(errors)} errors"
    )

    return LabelExperienceResponse(
        records_labeled=records_labeled,
        records_skipped=records_skipped,
        errors=errors,
    )
