"""Local lifecycle records for market-news jobs."""

from brain_api.news.models import NewsJob


class NewsJobs:
    """Store methods for local job bookkeeping."""

    def upsert_job(self, job: NewsJob) -> None:
        def _write(con) -> None:
            con.execute(
                """
                INSERT OR REPLACE INTO news_jobs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    job.job_id,
                    job.requested_start,
                    job.requested_end,
                    job.symbols_hash,
                    job.schema_version,
                    job.sentiment_revision,
                    job.status,
                    job.last_completed_symbol,
                    job.last_completed_window_end,
                    job.windows_done,
                    job.windows_total,
                    job.events_scored,
                    job.error,
                    job.created_at,
                    job.updated_at,
                ],
            )

        self._with_write_lock(_write)

    def get_job(self, job_id: str) -> NewsJob | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM news_jobs WHERE job_id = ?", [job_id]
            ).fetchone()
        if row is None:
            return None
        return NewsJob(
            job_id=row[0],
            requested_start=row[1],
            requested_end=row[2],
            symbols_hash=row[3],
            schema_version=int(row[4]),
            sentiment_revision=row[5],
            status=row[6],
            last_completed_symbol=row[7],
            last_completed_window_end=row[8],
            windows_done=int(row[9]),
            windows_total=int(row[10]),
            events_scored=int(row[11]),
            error=row[12],
            created_at=row[13],
            updated_at=row[14],
        )
