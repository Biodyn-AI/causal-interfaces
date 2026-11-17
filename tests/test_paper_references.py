"""Guard the manuscript bibliography and its internet-validation record."""

from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPOSITORY_ROOT / "paper"


def _bibliography_entries() -> dict[str, str]:
    bibliography = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    matches = list(re.finditer(r"@\w+\{([^,]+),", bibliography))
    entries: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(bibliography)
        entries[match.group(1)] = bibliography[match.start() : end]
    return entries


def _cited_keys() -> set[str]:
    manuscript = (PAPER_DIR / "manuscript.tex").read_text(encoding="utf-8")
    citations = re.findall(r"\\cite\w*\{([^}]+)\}", manuscript)
    return {key.strip() for group in citations for key in group.split(",")}


def test_every_bibliography_entry_is_cited_and_has_a_url() -> None:
    entries = _bibliography_entries()
    assert len(entries) >= 40
    assert _cited_keys() == set(entries)
    assert all(re.search(r"\burl\s*=", entry, flags=re.IGNORECASE) for entry in entries.values())


def test_every_reference_has_an_explicit_validation_record() -> None:
    entries = _bibliography_entries()
    validation = (PAPER_DIR / "REFERENCE_VALIDATION.md").read_text(encoding="utf-8")
    for key, entry in entries.items():
        url_match = re.search(r"\burl\s*=\s*\{([^}]+)\}", entry, flags=re.IGNORECASE)
        assert url_match is not None
        assert f"`{key}`" in validation
        assert url_match.group(1) in validation
