"""Focused tests for E29.14c hash, parent-pass, and attempt custody."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    ATTEMPT_REQUIRED_KEYS,
    MASK_ARTIFACT_SCHEMA,
    PARENT_E29_14B_REGISTRATION_SHA256,
    PARENT_E29_14B_SCORE_SUMMARY_SHA256,
    PHYSICAL_QUERY_ACCOUNTING,
    SOURCE_FILES,
    SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256,
    SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256,
    build_acquisition_attempt_record,
    create_acquisition_attempt_marker,
    load_verified_registration,
    sha256,
    validate_parent_full_pass,
    verify_source_closure,
    verify_v1_preacquisition_state,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    QUERY_ACCOUNTING,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14c_scale_scope import (
    PARENT_SCORE_SUMMARY,
)


def _passing_parent_summary() -> dict[str, object]:
    return {
        "schema_version": "e29_14b_component_position_bridge_score_v1",
        "integrity_passed": True,
        "all_registered_integrity_gates_passed": True,
        "claims": {
            "singleton_transfer": {"passed": True},
            "all_position_anchor_by_scale": {},
            "hub_off_scope_bridge_by_scale": {},
            "hub_on_scope_control_by_scale": {},
            "joint_hub_off_scope_bridge": {"passed": True},
            "joint_hub_on_scope_control": {"passed": True},
            "hub_off_scale_promotion_authorized": {"passed": True},
            "full_component_position_bridge": {"passed": True},
        },
        "query_accounting": {},
        "physical_query_accounting": {},
        "clean_summary": {},
        "custody": {},
        "recommended_interpretation": (
            "fresh_context_c24_c32_component_position_bridge_confirmed"
        ),
    }


def test_parent_summary_requires_the_exact_full_pass_contract() -> None:
    summary = _passing_parent_summary()

    assert validate_parent_full_pass(summary) == {
        "integrity_passed": True,
        "all_registered_integrity_gates_passed": True,
        "hub_off_scale_promotion_authorized": True,
        "full_component_position_bridge": True,
        "recommended_interpretation": (
            "fresh_context_c24_c32_component_position_bridge_confirmed"
        ),
    }

    for path in (
        ("integrity_passed",),
        ("all_registered_integrity_gates_passed",),
        ("claims", "hub_off_scale_promotion_authorized", "passed"),
        ("claims", "full_component_position_bridge", "passed"),
    ):
        changed = copy.deepcopy(summary)
        target = changed
        for key in path[:-1]:
            target = target[key]  # type: ignore[index]
        target[path[-1]] = False  # type: ignore[index]
        with pytest.raises(RuntimeError, match="E29.14b.*pass"):
            validate_parent_full_pass(changed)


def test_actual_hash_pinned_parent_summary_satisfies_the_exact_schema() -> None:
    assert sha256(PARENT_SCORE_SUMMARY) == PARENT_E29_14B_SCORE_SUMMARY_SHA256
    summary = json.loads(PARENT_SCORE_SUMMARY.read_text(encoding="utf-8"))

    assert validate_parent_full_pass(summary)[
        "all_registered_integrity_gates_passed"
    ] is True


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
        "acquire_e29_14c_scale_scope.py",
        "score_e29_14c_scale_scope.py",
        "e29_14c_scale_scope_protocol.py",
        "e29_14b_component_position_runner.py",
    } <= set(SOURCE_FILES)
    assert all("labels" not in name and "summary" not in name for name in SOURCE_FILES)


def test_attempt_marker_binds_parent_mask_protocol_and_runtime_once(
    tmp_path: Path,
) -> None:
    registration = {
        "parent_e29_14b_registration_sha256": (
            PARENT_E29_14B_REGISTRATION_SHA256
        ),
        "parent_e29_14b_score_summary_sha256": (
            PARENT_E29_14B_SCORE_SUMMARY_SHA256
        ),
        "source_code_sha256": {"source.py": "1" * 64},
        "checkpoint_sha256": {"model.safetensors": "2" * 64},
        "protocol_masks_sha256": "3" * 64,
        "prompt_panel_sha256": "4" * 64,
        "scale_scope_protocol_sha256": "5" * 64,
        "supersedes_e29_14c_v1_registration_sha256": (
            SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
        ),
        "supersedes_e29_14c_v1_protocol_masks_sha256": (
            SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
        ),
        "supersession": {
            "no_v1_attempt_or_labels_or_custody_or_summary_before_v2": True
        },
        "clean_metadata_contract_sha256": "7" * 64,
        "environment": {"python": "test"},
        "execution": {"batch_size": 32},
    }
    marker = build_acquisition_attempt_record(
        registration=registration,
        registration_path=tmp_path / "registration.json",
        observed_registration_sha256="6" * 64,
        expected_registration_sha256="6" * 64,
    )
    path = tmp_path / "attempt.json"

    digest = create_acquisition_attempt_marker(marker, path)

    assert set(marker) == ATTEMPT_REQUIRED_KEYS
    assert marker["status"] == "attempt_consumed_before_model_load"
    assert marker["query_accounting"] == QUERY_ACCOUNTING
    assert marker["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert marker["protocol_masks_sha256_verified_before_attempt"] == "3" * 64
    assert marker[
        "superseded_e29_14c_v1_registration_sha256_verified_before_attempt"
    ] == SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
    assert marker[
        "superseded_e29_14c_v1_protocol_masks_sha256_verified_before_attempt"
    ] == SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
    assert marker["clean_metadata_contract_sha256"] == "7" * 64
    assert digest == sha256(path)
    assert json.loads(path.read_text(encoding="utf-8")) == marker
    with pytest.raises(FileExistsError, match="overwrite"):
        create_acquisition_attempt_marker(marker, path)


def test_registration_load_is_hash_and_exact_key_bound(tmp_path: Path) -> None:
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


def test_reviewed_parent_hashes_and_exact_query_budget_are_fixed() -> None:
    assert PARENT_E29_14B_REGISTRATION_SHA256 == (
        "74f9a32093e50dec829f9a4d93da93b5d1db5e5f4f5a1c54e8a507d3deee18ad"
    )
    assert PARENT_E29_14B_SCORE_SUMMARY_SHA256 == (
        "fa7ae4d61ae4dd520901f4863f8790f6d9d85d8a4372c43d9e1fd74b3e64157f"
    )
    assert QUERY_ACCOUNTING["maximum_logical_queries"] == 3_456
    assert PHYSICAL_QUERY_ACCOUNTING["maximum_physical_model_examples"] == 4_028
    assert MASK_ARTIFACT_SCHEMA == "e29_14c_scale_scope_protocol_artifact_v2"


@pytest.mark.original_checkout
def test_v2_binds_the_immutable_preacquisition_v1_state() -> None:
    audit = verify_v1_preacquisition_state()

    assert audit["superseded_e29_14c_v1_registration_sha256"] == (
        SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
    )
    assert audit["superseded_e29_14c_v1_protocol_masks_sha256"] == (
        SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
    )
    assert audit[
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2"
    ] is True
    assert audit["scientific_protocol_unchanged"] is True
