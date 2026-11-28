"""US session-close sentiment features from previously materialized news."""

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from brain_api.news.models import (
    CONFIDENCE_RECENCY_TAU_HOURS,
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsRevisionPolicy,
    NewsWindow,
)
from brain_api.news.store import NewsStore


def session_news_window(close: datetime) -> NewsWindow:
    local = pd.Timestamp(close).tz_convert("America/New_York")
    return NewsWindow(
        (local - pd.DateOffset(days=7)).to_pydatetime(), local.to_pydatetime()
    )


def load_session_sentiment(
    store: NewsStore,
    symbols: Sequence[str],
    closes: pd.Series,
    *,
    revision_policy: NewsRevisionPolicy = "point_in_time",
) -> pd.DataFrame:
    """Require coverage; optionally permit later edits of pre-cutoff articles.

    Historical use of later revisions is explicitly biased. Live callers retain
    point-in-time behavior. Publication membership and coverage are unchanged.
    """
    if revision_policy not in ("point_in_time", "allow_later_revisions"):
        raise ValueError(f"Unknown news revision policy: {revision_policy}")
    if closes.empty or not symbols:
        return pd.DataFrame(index=closes.index, columns=list(symbols), dtype=float)
    windows = [session_news_window(close) for close in closes]
    grid = pd.DataFrame(
        {
            "session": closes.index,
            "start_at": [
                pd.Timestamp(window.start_exclusive).tz_convert("UTC")
                for window in windows
            ],
            "end_at": [
                pd.Timestamp(window.end_inclusive).tz_convert("UTC")
                for window in windows
            ],
        }
    )
    roster = pd.DataFrame({"symbol": list(dict.fromkeys(symbols))})
    with store._connect() as con:
        con.register("requested_windows", grid)
        con.register("requested_symbols", roster)
        rows = con.execute(
            """
            WITH compatible_coverage AS (
                SELECT c.* FROM news_coverage c JOIN requested_symbols r USING (symbol)
                WHERE c.provider = $provider AND c.schema_version = $schema
                  AND c.sentiment_model = $model AND c.sentiment_model_revision = $revision
                  AND c.status IN ('complete', 'verified_empty')
                  AND c.future_revision_excluded_count = 0
            ), intervals AS (
                SELECT w.session, w.start_at, w.end_at, c.symbol,
                       greatest(w.start_at, c.window_start_exclusive) AS lo,
                       least(w.end_at, c.window_end_inclusive) AS hi
                FROM requested_windows w JOIN compatible_coverage c
                    ON c.window_end_inclusive > w.start_at AND c.window_start_exclusive < w.end_at
            ), ordered AS (
                SELECT *, max(hi) OVER (
                    PARTITION BY session, symbol ORDER BY lo, hi
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_hi
                FROM intervals
            ), covered AS (
                SELECT session, symbol, start_at, end_at FROM ordered
                GROUP BY session, symbol, start_at, end_at
                HAVING min(lo) = start_at AND max(hi) = end_at
                    AND NOT bool_or(lo > coalesce(prior_hi, start_at))
            ), revisions AS (
                SELECT c.session, c.symbol, c.end_at, e.created_at, e.updated_at,
                       e.sentiment_score, e.confidence,
                       e.confidence * exp((epoch(e.created_at) - epoch(c.end_at)) / $tau_seconds) AS weight
                FROM covered c JOIN news_events e ON c.symbol = e.symbol
                    AND e.created_at > c.start_at AND e.created_at <= c.end_at
                WHERE e.provider = $provider AND e.schema_version = $schema
                  AND e.sentiment_model = $model AND e.sentiment_model_revision = $revision
                QUALIFY row_number() OVER (
                  PARTITION BY c.session, c.symbol, e.provider_article_id
                  ORDER BY (e.updated_at <= c.end_at) DESC, e.updated_at DESC
                ) = 1
            ), sentiments AS (
                SELECT session, symbol, count(*) AS n,
                       bool_and(updated_at <= end_at) AS usable,
                       count_if(updated_at > end_at) AS later_revisions,
                       sum(weight * sentiment_score) / nullif(sum(weight), 0) AS score
                FROM revisions GROUP BY session, symbol
            )
            SELECT c.session, c.symbol,
                   CASE WHEN s.n IS NULL THEN 0.0
                        WHEN s.usable OR $allow_later_revisions THEN s.score END AS score,
                   coalesce(s.later_revisions, 0) AS later_revisions
            FROM covered c LEFT JOIN sentiments s USING (session, symbol)
            """,
            {
                "provider": NEWS_PROVIDER,
                "schema": NEWS_SCHEMA_VERSION,
                "model": NEWS_SENTIMENT_MODEL,
                "revision": NEWS_SENTIMENT_REVISION,
                "tau_seconds": CONFIDENCE_RECENCY_TAU_HOURS * 3600,
                "allow_later_revisions": revision_policy == "allow_later_revisions",
            },
        ).df()
    result = rows.pivot(index="session", columns="symbol", values="score").reindex(
        index=closes.index, columns=list(symbols)
    )
    result.attrs.update(
        revision_policy=revision_policy,
        later_revision_cells=int(
            (rows["score"].notna() & rows["later_revisions"].gt(0)).sum()
        ),
    )
    return result
