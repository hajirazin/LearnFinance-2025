"""Weekly summary endpoint using LLM."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .models import (
    DoubleHRPSummaryRequest,
    IndiaWeeklySummaryRequest,
    WeeklySummaryRequest,
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


@router.post("/weekly-summary", response_model=WeeklySummaryResponse)
def generate_weekly_summary(
    request: WeeklySummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    """Generate an LLM summary of weekly forecast and allocation results.

    Takes all forecaster predictions (LSTM, PatchTST), allocator results
    (SAC, HRP), signals (news, fundamentals), and generates a summary
    using the configured LLM provider (OpenAI or OLLAMA).

    Does NOT include Alpaca order results - that data is only used in
    the email endpoint (/email/weekly-report).

    Args:
        request: Weekly forecast and allocation data from all endpoints.
        provider: LLM provider (injected via dependency).

    Returns:
        Summary with 8 paragraph fields and metadata.

    Raises:
        HTTPException: If template loading or LLM call fails.
    """
    logger.info(f"Generating weekly summary using provider={provider.name}")

    # Load and render the Jinja2 template
    try:
        env = get_jinja_env()
        template = env.get_template("weekly_summary_prompt.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: weekly_summary_prompt.j2",
        ) from e

    # Render prompt with all data
    prompt = template.render(
        lstm=request.lstm.model_dump(),
        patchtst=request.patchtst.model_dump(),
        news=request.news.model_dump(),
        fundamentals=request.fundamentals.model_dump(),
        hrp=request.hrp.model_dump(),
        sac=request.sac.model_dump(),
    )

    logger.debug(f"Generated prompt length: {len(prompt)} chars")

    # Call LLM provider
    try:
        llm_response = provider.generate(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {e}",
        ) from e

    # Parse JSON response
    try:
        summary = parse_json_response(llm_response.content)
    except ValueError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        # Return a fallback summary
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


@router.post("/india-weekly-summary", response_model=WeeklySummaryResponse)
def generate_india_weekly_summary(
    request: IndiaWeeklySummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    """Generate an LLM summary of India HRP allocation results.

    India pipeline is HRP-only (no SAC/news/fundamentals).
    Analyzes HRP concentration, diversification, and risk observations
    for NSE India halal stocks (Nifty 500 Shariah).

    Args:
        request: HRP allocation data from POST /allocation/hrp.
        provider: LLM provider (injected via dependency).

    Returns:
        Summary with 3 paragraph fields and metadata.

    Raises:
        HTTPException: If template loading or LLM call fails.
    """
    logger.info(f"Generating India weekly summary using provider={provider.name}")

    try:
        env = get_jinja_env()
        template = env.get_template("india_weekly_summary_prompt.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: india_weekly_summary_prompt.j2",
        ) from e

    prompt = template.render(
        hrp=request.hrp.model_dump(),
        universe=request.universe,
    )

    logger.debug(f"Generated India prompt length: {len(prompt)} chars")

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
