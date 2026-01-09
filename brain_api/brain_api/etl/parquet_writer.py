"""Chunked Parquet and CSV output writer.

Moved from news_sentiment_etl/output/parquet_writer.py.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brain_api.etl.aggregation import DailySentiment

# Schema for the output Parquet file
OUTPUT_SCHEMA = pa.schema(
    [
        ("date", pa.date32()),
        ("symbol", pa.string()),
        ("sentiment_score", pa.float32()),
        ("article_count", pa.int32()),
        ("avg_confidence", pa.float32()),
        ("p_pos_avg", pa.float32()),
        ("p_neg_avg", pa.float32()),
        ("total_articles", pa.int32()),
    ]
)


class ParquetWriter:
    """Writes DailySentiment records to Parquet in chunks.

    Supports:
    - Chunked writes to manage memory
    - Appending to existing file
    - Final compaction into single sorted file
    """

    def __init__(self, output_dir: Path, chunk_size: int = 100_000):
        """Initialize writer.

        Args:
            output_dir: Directory for output files
            chunk_size: Number of records per chunk file
        """
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self._buffer: list[DailySentiment] = []
        self._chunk_count = 0

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_path(self, chunk_idx: int) -> Path:
        """Get path for a chunk file."""
        return self.output_dir / f"chunk_{chunk_idx:05d}.parquet"

    def _write_chunk(self, records: list[DailySentiment]) -> Path:
        """Write a chunk of records to Parquet.

        Args:
            records: List of DailySentiment to write

        Returns:
            Path to the written chunk file
        """
        if not records:
            return Path()

        # Convert to DataFrame
        data = [r.to_dict() for r in records]
        df = pd.DataFrame(data)

        # Convert date strings to date objects
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA)

        # Write chunk
        chunk_path = self._chunk_path(self._chunk_count)
        pq.write_table(table, chunk_path, compression="snappy")
        self._chunk_count += 1

        return chunk_path

    def write(self, records: list[DailySentiment]) -> None:
        """Write records, buffering until chunk size is reached.

        Args:
            records: List of DailySentiment to write
        """
        self._buffer.extend(records)

        # Flush complete chunks
        while len(self._buffer) >= self.chunk_size:
            chunk = self._buffer[: self.chunk_size]
            self._buffer = self._buffer[self.chunk_size :]
            self._write_chunk(chunk)

    def flush(self) -> None:
        """Flush any remaining buffered records."""
        if self._buffer:
            self._write_chunk(self._buffer)
            self._buffer = []

    def finalize(
        self,
        parquet_filename: str | None = "daily_sentiment.parquet",
        csv_filename: str | None = None,
    ) -> dict[str, Path]:
        """Merge all chunks into sorted output file(s).

        Args:
            parquet_filename: Name for Parquet file (None to skip)
            csv_filename: Name for CSV file (None to skip)

        Returns:
            Dict with 'parquet' and/or 'csv' keys mapping to output paths
        """
        # Flush any remaining buffer
        self.flush()

        # Find all chunk files
        chunk_files = sorted(self.output_dir.glob("chunk_*.parquet"))

        output_paths: dict[str, Path] = {}

        # Build DataFrame from chunks
        if not chunk_files:
            # No data - create empty DataFrame
            df = pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "sentiment_score",
                    "article_count",
                    "avg_confidence",
                    "p_pos_avg",
                    "p_neg_avg",
                    "total_articles",
                ]
            )
        else:
            # Read and merge all chunks
            tables = []
            for chunk_file in chunk_files:
                tables.append(pq.read_table(chunk_file))

            merged = pa.concat_tables(tables)
            df = merged.to_pandas()
            df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

            # Clean up chunk files
            for chunk_file in chunk_files:
                chunk_file.unlink()

        # Write Parquet if requested
        if parquet_filename:
            parquet_path = self.output_dir / parquet_filename
            final_table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA)
            pq.write_table(final_table, parquet_path, compression="snappy")
            output_paths["parquet"] = parquet_path

        # Write CSV if requested
        if csv_filename:
            csv_path = self.output_dir / csv_filename
            df.to_csv(csv_path, index=False)
            output_paths["csv"] = csv_path

        return output_paths

    @property
    def records_written(self) -> int:
        """Approximate number of records written (excluding buffer)."""
        return self._chunk_count * self.chunk_size

    @property
    def buffer_size(self) -> int:
        """Number of records currently buffered."""
        return len(self._buffer)


def read_parquet_stats(path: Path) -> dict:
    """Read basic statistics from a Parquet file.

    Args:
        path: Path to Parquet file

    Returns:
        Dict with row count, date range, symbol count
    """
    if not path.exists():
        return {"error": "File not found"}

    table = pq.read_table(path)
    df = table.to_pandas()

    return {
        "row_count": len(df),
        "date_min": str(df["date"].min()) if len(df) > 0 else None,
        "date_max": str(df["date"].max()) if len(df) > 0 else None,
        "symbol_count": df["symbol"].nunique() if len(df) > 0 else 0,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }
