"""News window materialize/query routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from brain_api.news.errors import NewsError
from brain_api.news.models import NewsEvent, NewsWindow
from brain_api.news.service import NewsService, raise_http_status
from brain_api.news.store import NewsStore
from brain_api.routes.news.models import (
    NewsCoverageItem,
    NewsEventDTO,
    NewsWindowRequest,
    NewsWindowResult,
)
from brain_api.storage.base import DEFAULT_DATA_PATH

router = APIRouter()


def get_news_service() -> NewsService:
    return NewsService(NewsStore(DEFAULT_DATA_PATH))


def _to_result(
    window: NewsWindow, coverage, events: list[NewsEvent]
) -> NewsWindowResult:
    return NewsWindowResult(
        start_exclusive=window.start_exclusive,
        end_inclusive=window.end_inclusive,
        coverage=[
            NewsCoverageItem(
                symbol=row.symbol,
                status=row.status,
                event_count=row.event_count,
                future_revision_excluded_count=row.future_revision_excluded_count,
                sentiment_model_revision=row.sentiment_model_revision,
            )
            for row in coverage
        ],
        events=[
            NewsEventDTO(
                provider=event.provider,
                provider_article_id=event.provider_article_id,
                symbol=event.symbol,
                created_at=event.created_at,
                updated_at=event.updated_at,
                source=event.source,
                sentiment_score=event.sentiment_score,
                p_positive=event.p_positive,
                p_negative=event.p_negative,
                p_neutral=event.p_neutral,
                confidence=event.confidence,
                scored_text_sha256=event.scored_text_sha256,
                sentiment_model=event.sentiment_model,
                sentiment_model_revision=event.sentiment_model_revision,
                schema_version=event.schema_version,
            )
            for event in events
        ],
    )


def _window(request: NewsWindowRequest) -> NewsWindow:
    try:
        return NewsWindow(
            start_exclusive=request.start_exclusive,
            end_inclusive=request.end_inclusive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/windows/materialize", response_model=NewsWindowResult)
def materialize_news_window(request: NewsWindowRequest) -> NewsWindowResult:
    window = _window(request)
    try:
        coverage, events = get_news_service().materialize(request.symbols, window)
    except NewsError as exc:
        raise HTTPException(
            status_code=raise_http_status(exc), detail=str(exc)
        ) from exc
    return _to_result(window, coverage, events)


@router.post("/windows/query", response_model=NewsWindowResult)
def query_news_window(request: NewsWindowRequest) -> NewsWindowResult:
    window = _window(request)
    try:
        coverage, events = get_news_service().query(request.symbols, window)
    except NewsError as exc:
        raise HTTPException(
            status_code=raise_http_status(exc), detail=str(exc)
        ) from exc
    return _to_result(window, coverage, events)
