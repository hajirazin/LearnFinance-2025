"""Weekly summary endpoint using LLM."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .models import (
    DoubleHRPSummaryRequest,
    IndiaAlphaHRPSummaryRequest,
    SACWeeklySummaryRequest,
    USAlphaHRPSummaryRequest,
    USDoubleHRPSummaryRequest,
    WeeklySummaryResponse,
)
from .providers import LLMProvider, get_llm_provider, parse_json_response

logger = logging.getLogger(__name__)

router = APIRouter()

# Template directory (relative to brain_api package)
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


def get_jinja_env() -> Environment:
    """Get Jinja2 environment for loading templates."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,  # We're generating prompts, not HTML
    )


@router.post("/sac-weekly-summary", response_model=WeeklySummaryResponse)
def generate_sac_weekly_summary(
    request: SACWeeklySummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    """Generate an LLM summary of the SAC-only weekly run.

    Takes forecaster predictions (LSTM, PatchTST), the SAC allocator
    result, and signals (news, fundamentals), and generates a summary
    using the configured LLM provider (OpenAI or OLLAMA). HRP weekly
    reporting lives in ``/llm/us-alpha-hrp-summary``.

    Does NOT include Alpaca order results - that data is only used in
    the email endpoint (/email/sac-weekly-report).

    Args:
        request: SAC weekly forecast and allocation data.
        provider: LLM provider (injected via dependency).

    Returns:
        Summary with 6 paragraph fields and metadata.

    Raises:
        HTTPException: If template loading or LLM call fails.
    """
    logger.info(f"Generating SAC weekly summary using provider={provider.name}")

    try:
        env = get_jinja_env()
        template = env.get_template("sac_weekly_summary_prompt.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: sac_weekly_summary_prompt.j2",
        ) from e

    prompt = template.render(
        lstm=request.lstm.model_dump(),
        patchtst=request.patchtst.model_dump(),
        news=request.news.model_dump(),
        fundamentals=request.fundamentals.model_dump(),
        sac=request.sac.model_dump(),
    )

    logger.debug(f"Generated prompt length: {len(prompt)} chars")

    try:
        llm_response = provider.generate(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {e}",
        ) from e

    try:
        summary = parse_json_response(llm_response.content)
    except ValueError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        summary = {
            "para_1_overall_summary": "Unable to generate AI summary. Please check the logs for details.",
            "raw_response": llm_response.content[:500],
        }

    return WeeklySummaryResponse(
        summary=summary,
        provider=provider.name,
        model_used=llm_response.model,
        tokens_used=llm_response.tokens_used,
    )


@router.post("/india-alpha-hrp-summary", response_model=WeeklySummaryResponse)
def generate_india_alpha_hrp_summary(
    request: IndiaAlphaHRPSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    """Generate an LLM summary of India Alpha-HRP allocation results.

    India weekly allocation is structurally "PatchTST top-15 alpha screen on
    Nifty Shariah 500 (the ``halal_india`` universe) -> HRP", the India
    counterpart of the US Alpha-HRP path. Analyzes HRP concentration,
    diversification, and risk observations across the alpha-screened picks.

    Args:
        request: HRP allocation data from POST /allocation/hrp.
        provider: LLM provider (injected via dependency).

    Returns:
        Summary with 3 paragraph fields and metadata.

    Raises:
        HTTPException: If template loading or LLM call fails.
    """
    logger.info(f"Generating India Alpha-HRP summary using provider={provider.name}")

    try:
        env = get_jinja_env()
        template = env.get_template("india_alpha_hrp_summary_prompt.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: india_alpha_hrp_summary_prompt.j2",
        ) from e

    prompt = template.render(
        hrp=request.hrp.model_dump(),
        universe=request.universe,
    )

    logger.debug(f"Generated India Alpha-HRP prompt length: {len(prompt)} chars")

    try:
        llm_response = provider.generate(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {e}",
        ) from e

    try:
        summary = parse_json_response(llm_response.content)
    except ValueError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        summary = {
            "para_1_portfolio_overview": "Unable to generate AI summary. Please check the logs for details.",
            "raw_response": llm_response.content[:500],
        }

    return WeeklySummaryResponse(
        summary=summary,
        provider=provider.name,
        model_used=llm_response.model,
        tokens_used=llm_response.tokens_used,
    )


@router.post("/us-double-hrp-summary", response_model=WeeklySummaryResponse)
def generate_us_double_hrp_summary(
    request: USDoubleHRPSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    """Generate an LLM summary of US Double HRP with sticky selection.

    Stage 1 screens the full halal_new universe; sticky selection keeps
    week-over-week stable picks; Stage 2 re-allocates the resulting 15
    stocks. The summary frames the choice in terms of US paper trading
    via the dhrp Alpaca account.

    Args:
        request: Both stage results, universe label, and top_n.
        provider: LLM provider (injected via dependency).

    Returns:
        Summary with paragraph fields and metadata.

    Raises:
        HTTPException: If template loading or LLM call fails.
    """
    logger.info(f"Generating US Double HRP summary using provider={provider.name}")

    try:
        env = get_jinja_env()
        template = env.get_template("us_double_hrp_summary_prompt.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: us_double_hrp_summary_prompt.j2",
        ) from e

    prompt = template.render(
        stage1=request.stage1.model_dump(),
        stage2=request.stage2.model_dump(),
        universe=request.universe,
        top_n=request.top_n,
    )

    logger.debug(f"Generated US Double HRP prompt length: {len(prompt)} chars")

    try:
        llm_response = provider.generate(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {e}",
        ) from e

    try:
        summary = parse_json_response(llm_response.content)
    except ValueError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        summary = {
            "para_1_screening_overview": "Unable to generate AI summary. Please check the logs for details.",
            "raw_response": llm_response.content[:500],
        }

    return WeeklySummaryResponse(
        summary=summary,
        provider=provider.name,
        model_used=llm_response.model,
        tokens_used=llm_response.tokens_used,
    )


@router.post("/us-alpha-hrp-summary", response_model=WeeklySummaryResponse)
def generate_us_alpha_hrp_summary(
    request: USAlphaHRPSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    """Generate an LLM summary of US Alpha-HRP weekly results.

    Stage 1 is PatchTST predicted weekly returns over halal_new (alpha
    screen); rank-band sticky selection picks 15 with K_hold=30; Stage 2
    HRP risk-parity sizes the chosen names. The summary frames the
    alpha-then-risk pipeline for the human reviewer of the ``hrp``
    Alpaca paper account.
    """
    logger.info(f"Generating US Alpha-HRP summary using provider={provider.name}")

    try:
        env = get_jinja_env()
        template = env.get_template("us_alpha_hrp_summary_prompt.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: us_alpha_hrp_summary_prompt.j2",
        ) from e

    prompt = template.render(
        stage1_top_scores=[item.model_dump() for item in request.stage1_top_scores],
        model_version=request.model_version,
        predicted_count=request.predicted_count,
        requested_count=request.requested_count,
        selected_symbols=request.selected_symbols,
        kept_count=request.kept_count,
        fillers_count=request.fillers_count,
        evicted_from_previous=request.evicted_from_previous,
        previous_year_week_used=request.previous_year_week_used,
        stage2=request.stage2.model_dump(),
        universe=request.universe,
        top_n=request.top_n,
        hold_threshold=request.hold_threshold,
    )

    logger.debug(f"Generated US Alpha-HRP prompt length: {len(prompt)} chars")

    try:
        llm_response = provider.generate(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {e}",
        ) from e

    try:
        summary = parse_json_response(llm_response.content)
    except ValueError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        summary = {
            "para_1_market_outlook": "Unable to generate AI summary. Please check the logs for details.",
            "raw_response": llm_response.content[:500],
        }

    return WeeklySummaryResponse(
        summary=summary,
        provider=provider.name,
        model_used=llm_response.model,
        tokens_used=llm_response.tokens_used,
    )


@router.post("/india-double-hrp-summary", response_model=WeeklySummaryResponse)
def generate_india_double_hrp_summary(
    request: DoubleHRPSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    """Generate an LLM summary of two-stage Double HRP allocation.

    Stage 1 screens the full universe with a long lookback, then the
    top-N stocks are re-allocated with a shorter lookback in Stage 2.

    Args:
        request: Both stage results, universe label, and top_n.
        provider: LLM provider (injected via dependency).

    Returns:
        Summary with paragraph fields and metadata.

    Raises:
        HTTPException: If template loading or LLM call fails.
    """
    logger.info(f"Generating Double HRP summary using provider={provider.name}")

    try:
        env = get_jinja_env()
        template = env.get_template("india_double_hrp_summary_prompt.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: india_double_hrp_summary_prompt.j2",
        ) from e

    prompt = template.render(
        stage1=request.stage1.model_dump(),
        stage2=request.stage2.model_dump(),
        universe=request.universe,
        top_n=request.top_n,
    )

    logger.debug(f"Generated Double HRP prompt length: {len(prompt)} chars")

    try:
        llm_response = provider.generate(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {e}",
        ) from e

    try:
        summary = parse_json_response(llm_response.content)
    except ValueError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        summary = {
            "para_1_screening_overview": "Unable to generate AI summary. Please check the logs for details.",
            "raw_response": llm_response.content[:500],
        }

    return WeeklySummaryResponse(
        summary=summary,
        provider=provider.name,
        model_used=llm_response.model,
        tokens_used=llm_response.tokens_used,
    )
