"""Schema for local news evidence and synchronization bookkeeping."""

NEWS_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_events (
    provider VARCHAR NOT NULL,
    provider_article_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    sentiment_score DOUBLE NOT NULL,
    p_positive DOUBLE NOT NULL,
    p_negative DOUBLE NOT NULL,
    p_neutral DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    scored_text_sha256 VARCHAR NOT NULL,
    sentiment_model VARCHAR NOT NULL,
    sentiment_model_revision VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider, provider_article_id, symbol, updated_at)
);
CREATE TABLE IF NOT EXISTS news_coverage (
    provider VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    window_start_exclusive TIMESTAMPTZ NOT NULL,
    window_end_inclusive TIMESTAMPTZ NOT NULL,
    schema_version INTEGER NOT NULL,
    sentiment_model VARCHAR NOT NULL,
    sentiment_model_revision VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    page_count INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    future_revision_excluded_count INTEGER NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    request_manifest_hash VARCHAR NOT NULL,
    PRIMARY KEY (
        provider, symbol, window_start_exclusive, window_end_inclusive,
        schema_version, sentiment_model, sentiment_model_revision
    )
);
CREATE TABLE IF NOT EXISTS news_jobs (
    job_id VARCHAR PRIMARY KEY,
    requested_start TIMESTAMPTZ NOT NULL,
    requested_end TIMESTAMPTZ NOT NULL,
    symbols_hash VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    sentiment_revision VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    last_completed_symbol VARCHAR,
    last_completed_window_end TIMESTAMPTZ,
    windows_done INTEGER NOT NULL,
    windows_total INTEGER NOT NULL,
    events_scored INTEGER NOT NULL,
    error VARCHAR,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS news_score_cache (
    scored_text_sha256 VARCHAR NOT NULL,
    sentiment_model VARCHAR NOT NULL,
    sentiment_model_revision VARCHAR NOT NULL,
    scoring_schema_version INTEGER NOT NULL,
    sentiment_score DOUBLE NOT NULL,
    p_positive DOUBLE NOT NULL,
    p_negative DOUBLE NOT NULL,
    p_neutral DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    PRIMARY KEY (
        scored_text_sha256, sentiment_model,
        sentiment_model_revision, scoring_schema_version
    )
);
CREATE TABLE IF NOT EXISTS news_sync_dirty (
    partition VARCHAR PRIMARY KEY,
    generation BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_sync_imports (
    repo VARCHAR NOT NULL,
    partition VARCHAR NOT NULL,
    checksum VARCHAR NOT NULL,
    PRIMARY KEY (repo, partition)
);
CREATE TABLE IF NOT EXISTS news_sync_seeded (
    repo VARCHAR PRIMARY KEY
);
"""
