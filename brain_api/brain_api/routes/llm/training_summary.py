"""Training summary endpoints using LLM.

Three endpoints share the same prompt-render -> LLM-call -> JSON-parse
pipeline (see :func:`_run_training_summary`). Each endpoint differs only
in the Jinja template it loads, the request DTO it accepts, and the
fallback paragraph key used when the LLM returns un-parseable text:

* ``/llm/forecasters-training-summary`` -- US LSTM + PatchTST (called by
  the US Forecasters Temporal workflow).
* ``/llm/sac-training-summary`` -- US SAC (called by the US SAC Temporal
  workflow, which runs 12+ hours after forecasters and consumes whatever
  PatchTST ``current`` pointer is live at trigger time).
* ``/llm/india-training-summary`` -- India PatchTST (single India
  forecaster).
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .models import (
    ForecastersTrainingSummaryRequest,
    IndiaTrainingSummaryRequest,
    SACTrainingSummaryRequest,
    TrainingSummaryResponse,
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


def _run_training_summary(
    *,
    template_name: str,
    template_context: dict[str, Any],
    fallback_key: str,
    provider: LLMProvider,
    log_label: str,
) -> TrainingSummaryResponse:
    """Render a Jinja prompt, call the LLM, and parse the JSON response.

    Shared by every training-summary endpoint in this module so that
    template/LLM/JSON-parse error handling stays identical across
    forecasters, SAC, and India variants. Each caller only owns the
    template name, request payload, and fallback paragraph key.

    Args:
        template_name: Jinja template filename in ``brain_api/templates``.
        template_context: Variables to render into the prompt.
        fallback_key: Paragraph key to populate when the LLM returns
            text the JSON parser cannot consume (keeps the response
            schema stable for downstream email rendering).
        provider: LLM provider (injected via dependency).
        log_label: Short tag used in log lines (``forecasters``, ``sac``,
            ``india``).

    Returns:
        ``TrainingSummaryResponse`` with the parsed (or fallback)
        summary plus provider metadata.

    Raises:
        HTTPException: 500 if the template is missing, 503 if the LLM
            call fails.
    """
    logger.info(
        f"Generating {log_label} training summary using provider={provider.name}"
    )

    try:
        env = get_jinja_env()
        template = env.get_template(template_name)
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template not found: {template_name}",
        ) from e

    prompt = template.render(**template_context)
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
            fallback_key: "Unable to generate AI summary. Please check the logs for details.",
            "raw_response": llm_response.content[:500],
        }

    return TrainingSummaryResponse(
        summary=summary,
        provider=provider.name,
        model_used=llm_response.model,
        tokens_used=llm_response.tokens_used,
    )


@router.post("/forecasters-training-summary", response_model=TrainingSummaryResponse)
def generate_forecasters_training_summary(
    request: ForecastersTrainingSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> TrainingSummaryResponse:
    """Generate an LLM summary of US forecaster (LSTM + PatchTST) training.

    Called by the US Forecasters Temporal workflow after both forecasters
    have finished training serially. SAC is summarised independently by
    :func:`generate_sac_training_summary` so the two workflows email
    independent reports on different days.
    """
    return _run_training_summary(
        template_name="forecasters_training_summary_prompt.j2",
        template_context={
            "lstm": request.lstm.model_dump(),
            "patchtst": request.patchtst.model_dump(),
        },
        fallback_key="para_1_overall",
        provider=provider,
        log_label="forecasters",
    )


@router.post("/sac-training-summary", response_model=TrainingSummaryResponse)
def generate_sac_training_summary(
    request: SACTrainingSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> TrainingSummaryResponse:
    """Generate an LLM summary of US SAC allocator training.

    Called by either US SAC Temporal workflow (``halal_filtered`` at
    Sunday 02:00 UTC, ``halal`` at Sunday 13:00 UTC) after SAC has
    finished training. The SAC workflows read whatever PatchTST
    ``current`` pointer is live at trigger time, so forecaster
    metrics are not part of this payload. The ``request.universe``
    field is rendered into the prompt so the resulting summary is
    explicit about which bucket it describes.
    """
    return _run_training_summary(
        template_name="sac_training_summary_prompt.j2",
        template_context={
            "sac": request.sac.model_dump(),
            "universe": request.universe,
        },
        fallback_key="para_1_overall",
        provider=provider,
        log_label=f"sac-{request.universe}",
    )


@router.post("/india-training-summary", response_model=TrainingSummaryResponse)
def generate_india_training_summary(
    request: IndiaTrainingSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> TrainingSummaryResponse:
    """Generate an LLM summary of India PatchTST training results.

    India trains PatchTST only (no LSTM, SAC).
    """
    return _run_training_summary(
        template_name="india_training_summary_prompt.j2",
        template_context={"patchtst": request.patchtst.model_dump()},
        fallback_key="para_1_overall",
        provider=provider,
        log_label="india",
    )
