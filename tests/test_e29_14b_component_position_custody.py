"""Focused tests for E29.14b hash, attempt, and write-once custody."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    ATTEMPT_REQUIRED_KEYS,
    PARENT_E29_14A_REGISTRATION_SHA256,
    PHYSICAL_QUERY_ACCOUNTING,
    PRIOR_PROTOCOL_MASKS_SHA256,
    SOURCE_FILES,
    build_acquisition_attempt_record,
    create_acquisition_attempt_marker,
    load_verified_registration,
    sha256,
    verify_source_closure,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    QUERY_ACCOUNTING,
)


def test_source_closure_is_exact_and_names_future_execution_files(
    tmp_path: Path,
) -> None:
    names = ("alpha.py", "beta.py")
    for name in names:
        (tmp_path / name).write_text(f"{name}\n", encoding="utf-8")

    hashes = verify_source_closure(tmp_path, names)

    assert tuple(hashes) == names
    assert hashes == {name: sha256(tmp_path / name) for name in names}
    assert {
        "acquire_e29_14b_component_position.py",
        "score_e29_14b_component_position.py",
        "e29_14b_component_position_runner.py",
    } <= set(SOURCE_FILES)
    assert all("labels" not in name and "summary" not in name for name in SOURCE_FILES)


def test_source_closure_rejects_duplicate_nested_and_missing_names(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha.py").write_text("alpha\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        verify_source_closure(tmp_path, ("alpha.py", "alpha.py"))
    with pytest.raises(ValueError, match="plain filenames"):
        verify_source_closure(tmp_path, ("nested/alpha.py",))
    with pytest.raises(FileNotFoundError):
        verify_source_closure(tmp_path, ("missing.py",))


def test_attempt_marker_binds_every_custody_input_and_is_write_once(
    tmp_path: Path,
) -> None:
    registration = {
        "source_code_sha256": {"source.py": "1" * 64},
        "checkpoint_sha256": {"model.safetensors": "2" * 64},
        "prior_protocol_masks_sha256": PRIOR_PROTOCOL_MASKS_SHA256,
        "prompt_panel_sha256": "3" * 64,
        "component_position_protocol_sha256": "4" * 64,
        "environment": {"python": "test"},
        "execution": {"batch_size": 32},
    }
    marker = build_acquisition_attempt_record(
        registration=registration,
        registration_path=tmp_path / "registration.json",
        observed_registration_sha256="5" * 64,
        expected_registration_sha256="5" * 64,
    )
    path = tmp_path / "attempt.json"

    digest = create_acquisition_attempt_marker(marker, path)

    assert set(marker) == ATTEMPT_REQUIRED_KEYS
    assert marker["status"] == "attempt_consumed_before_model_load"
    assert marker["query_accounting"] == QUERY_ACCOUNTING
    assert marker["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert digest == sha256(path)
    assert json.loads(path.read_text(encoding="utf-8")) == marker
    with pytest.raises(FileExistsError, match="overwrite"):
        create_acquisition_attempt_marker(marker, path)


def test_registration_load_is_hash_bound_and_exact_key_bound(tmp_path: Path) -> None:
    path = tmp_path / "registration.json"
    path.write_text('{"schema_version":"test","status":"sealed"}\n')
    expected_keys = {"schema_version", "status"}

    assert load_verified_registration(
        path, expected_sha256=sha256(path), required_keys=expected_keys
    )["status"] == "sealed"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_verified_registration(
            path, expected_sha256="0" * 64, required_keys=expected_keys
        )
    path.write_text(
        '{"schema_version":"test","status":"sealed","extra":true}\n'
    )
    with pytest.raises(RuntimeError, match="keys changed"):
        load_verified_registration(
            path, expected_sha256=sha256(path), required_keys=expected_keys
        )


def test_fixed_parent_hashes_are_the_reviewed_artifacts() -> None:
    assert PARENT_E29_14A_REGISTRATION_SHA256 == (
        "f6ef1755bd6a4c3345a78fe180f00d7dd5e98086319348f34d9a1d9592ee23df"
    )
    assert PRIOR_PROTOCOL_MASKS_SHA256 == (
        "014b87e0ddf2cf913ff745e41e12bb6444420716753d0eb712bd465eaa0c83fd"
    )
    assert PHYSICAL_QUERY_ACCOUNTING["maximum_logical_queries"] == 2_352
    assert PHYSICAL_QUERY_ACCOUNTING["maximum_physical_model_examples"] == 2_740
