"""Release-level tests for the portable E29.14 publication replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.e29_14.replay_scores import (
    EXPERIMENT_IDS,
    _scientific_values_match,
    load_manifest,
    replay_all,
)
from repro.e29_14.reproduce_inference import verify_checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_all_scientific_scores_replay_with_declared_tolerance() -> None:
    replay = replay_all(REPOSITORY_ROOT)
    assert replay["all_replays_match_published_scientific_fields"] is True
    assert tuple(record["experiment"] for record in replay["experiments"]) == tuple(
        f"E29.14{experiment_id}" for experiment_id in EXPERIMENT_IDS
    )


def test_cross_platform_numeric_comparison_is_strict() -> None:
    expected = {
        "score": 0.25,
        "nested": [1, {"bound": 0.125}],
        "passed": True,
    }
    within_tolerance = {
        "score": 0.25 + 5e-14,
        "nested": [1, {"bound": 0.125 - 5e-14}],
        "passed": True,
    }
    outside_tolerance = {
        "score": 0.25 + 1e-8,
        "nested": [1, {"bound": 0.125}],
        "passed": True,
    }

    assert _scientific_values_match(expected, within_tolerance)
    assert not _scientific_values_match(expected, outside_tolerance)
    assert not _scientific_values_match(expected, {**expected, "passed": 1})


def test_manifest_uses_relative_publication_paths() -> None:
    manifest = load_manifest()
    for record in manifest["experiments"].values():
        for key in ("registration", "labels", "published_summary"):
            assert not Path(record[key]).is_absolute()
        if "protocol_artifact" in record:
            assert not Path(record["protocol_artifact"]).is_absolute()


def test_checkpoint_verification_rejects_changed_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_checkpoint(checkpoint, {"config.json": "0" * 64})


def test_manifest_is_plain_json() -> None:
    manifest_path = REPOSITORY_ROOT / "repro/e29_14/manifest.json"
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "e29_14_portable_reproduction_manifest_v1"

