"""Parse and build names for content-addressed forecaster snapshot dirs / HF branches."""

from __future__ import annotations

from datetime import date

# Length of hexadecimal snapshot suffix (matches :func:`brain_api.core.version.compute_model_hash`).
SNAPSHOT_DIGEST_LEN = 12


def snapshot_branch_basename(cutoff_date: date, snapshot_digest: str) -> str:
    """Basename for local dir and HF branch: ``snapshot-{date}-{digest}``."""
    return f"snapshot-{cutoff_date.isoformat()}-{snapshot_digest}"


def parse_hashed_snapshot_folder_name(dirname: str) -> tuple[date, str] | None:
    """Parse ``snapshot-{YYYY-MM-DD}-{digest}``. Legacy ``snapshot-{date}`` returns None."""

    prefix = "snapshot-"
    if not dirname.startswith(prefix):
        return None
    body = dirname[len(prefix) :]
    if len(body) < SNAPSHOT_DIGEST_LEN + 1 + 10:  # date + hyphen + digest
        return None
    if body[10] != "-":
        return None
    iso_date = body[:10]
    digest = body[11:]
    if len(digest) != SNAPSHOT_DIGEST_LEN:
        return None
    hex_ok = set("0123456789abcdef")
    if not all(ch in hex_ok for ch in digest):
        return None
    try:
        cutoff = date.fromisoformat(iso_date)
    except ValueError:
        return None
    return cutoff, digest
