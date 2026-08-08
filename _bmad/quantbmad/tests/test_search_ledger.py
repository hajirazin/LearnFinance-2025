"""Unit tests for QuantBMAD ledger search."""

from __future__ import annotations

import sys
from pathlib import Path

_BMAD = Path(__file__).resolve().parents[2]
if str(_BMAD) not in sys.path:
    sys.path.insert(0, str(_BMAD))

from quantbmad.scripts.search_ledger import search_ledger  # noqa: E402


def test_search_empty_file(tmp_path: Path) -> None:
    ledger = tmp_path / "research-ledger.md"
    ledger.write_text("# empty\n", encoding="utf-8")
    assert search_ledger(ledger, "tanh") == []


def test_search_missing_file(tmp_path: Path) -> None:
    assert search_ledger(tmp_path / "nope.md", "tanh") == []


def test_search_finds_keyword(tmp_path: Path) -> None:
    ledger = tmp_path / "research-ledger.md"
    ledger.write_text(
        """# Ledger

| Date | Hypothesis | Track | Experiment path | Result | Skeptic verdict | Shipped? | Re-review by | Human audit? | Author model | Skeptic model |
|------|------------|-------|-----------------|--------|-----------------|----------|--------------|--------------|--------------|---------------|
| 2026-08-08 | Removing tanh unlocks concentration | Research | scratch/a.py | FAIL | FAIL | No | n/a | n/a | claude | grok |
| 2026-08-08 | Softmax temperature=10 | Research | scratch/b.py | PASS | PASS | Yes | 2026-11-08 | pending | claude | gpt |
""",
        encoding="utf-8",
    )
    hits = search_ledger(ledger, "tanh")
    assert len(hits) == 1
    assert "tanh" in hits[0].lower()

    hits2 = search_ledger(ledger, "softmax temperature")
    assert len(hits2) == 1

    assert search_ledger(ledger, "nonexistent-xyz") == []
