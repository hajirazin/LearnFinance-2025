"""Filesystem persistence for SAC experience records."""

import json
from pathlib import Path

from brain_api.routes.experience_models import ExperienceRecord
from brain_api.storage.base import DEFAULT_DATA_PATH


class ExperienceStorage:
    """Storage for RL experience records."""

    def __init__(self, base_path: Path | str | None = None):
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self._experience_path = self.base_path / "experience"
        self._experience_path.mkdir(parents=True, exist_ok=True)

    def _record_path(self, run_id: str) -> Path:
        """Get path for a specific run's experience record."""
        # Sanitize run_id for filesystem
        safe_id = run_id.replace(":", "_").replace("/", "_")
        return self._experience_path / f"{safe_id}.json"

    def store(self, record: ExperienceRecord) -> str:
        """Store an experience record.

        Returns:
            Record ID (same as run_id).
        """
        path = self._record_path(record.run_id)
        with open(path, "w") as f:
            json.dump(record.model_dump(), f, indent=2, default=str)
        return record.run_id

    def load(self, run_id: str) -> ExperienceRecord | None:
        """Load an experience record by run_id."""
        path = self._record_path(run_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ExperienceRecord(**data)

    def list_unlabeled(self) -> list[ExperienceRecord]:
        """List all unlabeled experience records."""
        records = []
        for path in self._experience_path.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            record = ExperienceRecord(**data)
            if record.reward is None:
                records.append(record)
        return records

    def list_all(self) -> list[ExperienceRecord]:
        """List all experience records."""
        records = []
        for path in self._experience_path.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            records.append(ExperienceRecord(**data))
        return records

    def update(self, record: ExperienceRecord) -> None:
        """Update an existing experience record."""
        self.store(record)
