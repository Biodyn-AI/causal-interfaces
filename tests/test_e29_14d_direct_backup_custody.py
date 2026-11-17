"""Focused tests for the prospective E29.14d custody boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    PHYSICAL_QUERY_ACCOUNTING,
    QUERY_ACCOUNTING,
    build_causal_carrier_protocol,
    build_condition_masks,
    build_query_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_direct_backup_custody import (
    ATTEMPT_REQUIRED_KEYS,
    PARENT_E29_14C_REGISTRATION_SHA256,
    PARENT_E29_14C_SCORE_SUMMARY_SHA256,
    POSTHOC_NOMINATIONS,
    POSTHOC_TOP4_RANKING,
    PRIOR_MASK_ARTIFACT_SPECS,
    PROTOCOL_ARTIFACT_SCHEMA,
    SOURCE_FILES,
    atomic_torch_save,
    audit_prior_mask_nonoverlap,
    build_acquisition_attempt_record,
    build_protocol_artifact,
    create_acquisition_attempt_marker,
    load_verified_protocol_artifact,
    sha256,
    validate_parent_full_pass,
    validate_posthoc_candidate_provenance,
)


def _passing_parent_summary() -> dict[str, object]:
    return {
        "schema_version": "e29_14c_hub_off_scale_scope_ladder_score_v2",
        "scientific_score_schema_version": (
            "e29_14c_hub_off_scale_scope_ladder_score_v1"
        ),
        "integrity_passed": True,
        "all_registered_integrity_gates_passed": True,
        "claims": {
            "singleton_transfer": {},
            "scope_equivalence_by_scale": {},
            "joint_equal_scale_scope_equivalence": {},
            "high_scale_lexical_scope_authorized": {"passed": True},
            "informative_c36_nonsaturation": {},
            "c47_backup_endpoint": {},
            "complete_hub_off_scale_ladder": {"passed": True},
        },
        "scale_response_geometry": {},
        "transition_response_increment_diagnostics": {},
        "transition_scope_surrogate_failure_localization": {},
        "query_accounting": {},
        "physical_query_accounting": {},
        "clean_summary": {},
        "custody": {"registration_sha256": PARENT_E29_14C_REGISTRATION_SHA256},
        "recommended_interpretation": (
            "hub_off_c36_c47_scope_equivalence_with_informative_transition_and_"
            "backup_endpoint_confirmed"
        ),
    }


def test_parent_summary_requires_exact_complete_ladder_pass() -> None:
    summary = _passing_parent_summary()
    contract = validate_parent_full_pass(summary)

    assert contract["complete_hub_off_scale_ladder"] is True
    assert contract["high_scale_lexical_scope_authorized"] is True

    for path in (
        ("integrity_passed",),
        ("all_registered_integrity_gates_passed",),
        ("claims", "complete_hub_off_scale_ladder", "passed"),
        ("claims", "high_scale_lexical_scope_authorized", "passed"),
    ):
        changed = copy.deepcopy(summary)
        target = changed
        for key in path[:-1]:
            target = target[key]  # type: ignore[index]
        target[path[-1]] = False  # type: ignore[index]
        with pytest.raises(RuntimeError, match="E29.14c.*pass"):
            validate_parent_full_pass(changed)


@pytest.mark.original_checkout
def test_protocol_artifact_binds_protocol_masks_and_both_manifests(
    tmp_path: Path,
) -> None:
    masks = build_condition_masks()
    artifact = build_protocol_artifact(masks)
    path = tmp_path / "protocol_conditions.pt"
    atomic_torch_save(artifact, path)

    loaded = load_verified_protocol_artifact(path, expected_sha256=sha256(path))

    assert loaded.dtype == torch.float64
    assert loaded.device.type == "cpu"
    assert torch.equal(loaded, masks)
    assert artifact["schema_version"] == PROTOCOL_ARTIFACT_SCHEMA
    assert artifact["causal_carrier_protocol"] == build_causal_carrier_protocol()
    assert artifact["query_manifest"] == list(build_query_manifest())
    assert artifact["prior_mask_audit"]["prior_c32_c33_unique_rows"] == 144
    assert artifact["prior_mask_audit"]["overlap_count"] == 0
    assert artifact["posthoc_candidate_provenance"]["ranking_order"] == list(
        POSTHOC_TOP4_RANKING
    )

    changed = copy.deepcopy(artifact)
    changed["query_manifest"][0]["condition_index"] += 1
    changed_path = tmp_path / "changed.pt"
    atomic_torch_save(changed, changed_path)
    with pytest.raises(RuntimeError, match="query manifest"):
        load_verified_protocol_artifact(
            changed_path, expected_sha256=sha256(changed_path)
        )


def test_attempt_marker_binds_every_preacquisition_identity_once(
    tmp_path: Path,
) -> None:
    registration = {
        "parent_e29_14c_registration_sha256": PARENT_E29_14C_REGISTRATION_SHA256,
        "parent_e29_14c_score_summary_sha256": PARENT_E29_14C_SCORE_SUMMARY_SHA256,
        "posthoc_candidate_provenance_sha256": "b" * 64,
        "prior_mask_audit_sha256": "c" * 64,
        "source_code_sha256": {"source.py": "1" * 64},
        "checkpoint_sha256": {"model.safetensors": "2" * 64},
        "protocol_artifact_sha256": "3" * 64,
        "prompt_panel_sha256": "4" * 64,
        "prompt_records_sha256": "a" * 64,
        "clean_metadata_contract_sha256": "5" * 64,
        "causal_carrier_protocol_sha256": "6" * 64,
        "condition_manifest_sha256": "7" * 64,
        "query_manifest_sha256": "8" * 64,
        "environment": {"python": "test"},
        "execution": {"batch_size": 32},
    }
    marker = build_acquisition_attempt_record(
        registration=registration,
        registration_path=tmp_path / "registration.json",
        observed_registration_sha256="9" * 64,
        expected_registration_sha256="9" * 64,
    )
    path = tmp_path / "attempt.json"

    digest = create_acquisition_attempt_marker(marker, path)

    assert set(marker) == ATTEMPT_REQUIRED_KEYS
    assert marker["status"] == "attempt_consumed_before_model_load"
    assert marker["query_accounting"] == QUERY_ACCOUNTING
    assert marker["physical_query_accounting"] == json.loads(
        json.dumps(PHYSICAL_QUERY_ACCOUNTING)
    )
    assert marker["protocol_artifact_sha256_verified_before_attempt"] == "3" * 64
    assert (
        marker["posthoc_candidate_provenance_sha256_verified_before_attempt"]
        == "b" * 64
    )
    assert marker["prior_mask_audit_sha256_verified_before_attempt"] == "c" * 64
    assert digest == sha256(path)
    assert json.loads(path.read_text(encoding="utf-8")) == marker
    with pytest.raises(FileExistsError, match="overwrite"):
        create_acquisition_attempt_marker(marker, path)


def test_source_closure_names_the_entire_future_outcome_path() -> None:
    assert {
        "acquire_e29_14d_direct_backup.py",
        "e29_14d_direct_backup_custody.py",
        "e29_14d_causal_carrier_protocol.py",
        "score_e29_14d_direct_backup.py",
    } <= set(SOURCE_FILES)
    assert len(SOURCE_FILES) == len(set(SOURCE_FILES))
    assert all("labels" not in name and "summary" not in name for name in SOURCE_FILES)


@pytest.mark.original_checkout
def test_posthoc_top4_provenance_is_hash_source_and_ranking_bound(
    tmp_path: Path,
) -> None:
    provenance = validate_posthoc_candidate_provenance()

    assert provenance["ranking_order"] == [38, 42, 40, 36]
    assert provenance["protocol_sorted_core"] == [36, 38, 40, 42]
    assert provenance["top20_attention_count"] == 20
    assert provenance["class_hypothesis_role"] == (
        "motivated_prospective_attention_vs_mlp_primary_hypothesis"
    )
    assert provenance["source_score_summary_sha256"] == (
        PARENT_E29_14C_SCORE_SUMMARY_SHA256
    )

    changed = json.loads(POSTHOC_NOMINATIONS.read_text(encoding="utf-8"))
    changed["overall_consensus"]["component_ranking"][0]["component"] = 36
    changed_path = tmp_path / "changed_posthoc.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="top-four"):
        validate_posthoc_candidate_provenance(
            changed_path, expected_sha256=sha256(changed_path)
        )

    changed = json.loads(POSTHOC_NOMINATIONS.read_text(encoding="utf-8"))
    changed["overall_consensus"]["component_ranking"][19]["component"] = 3
    changed_path = tmp_path / "changed_attention_class_posthoc.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="top-twenty attention-class"):
        validate_posthoc_candidate_provenance(
            changed_path, expected_sha256=sha256(changed_path)
        )


@pytest.mark.original_checkout
def test_actual_prior_mask_closure_is_zero_overlap_and_fails_closed(
    tmp_path: Path,
) -> None:
    masks = build_condition_masks()
    audit = audit_prior_mask_nonoverlap(masks)

    assert audit["artifact_count"] == len(PRIOR_MASK_ARTIFACT_SPECS) == 6
    assert audit["current_c32_c33_unique_rows"] == 254
    assert audit["prior_c32_c33_unique_rows"] == 144
    assert audit["overlap_count"] == 0

    drifted = [dict(spec) for spec in PRIOR_MASK_ARTIFACT_SPECS]
    drifted[0] = {**drifted[0], "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="hash mismatch"):
        audit_prior_mask_nonoverlap(masks, artifact_specs=drifted)

    c32_index = int(torch.where(masks.sum(dim=1) == 32)[0][0])
    collision_path = tmp_path / "collision.pt"
    atomic_torch_save(
        {
            "schema_version": "synthetic_prior_v1",
            "masks": masks[c32_index : c32_index + 1].clone(),
        },
        collision_path,
    )
    collision_spec = {
        "label": "synthetic",
        "path": str(collision_path.resolve()),
        "sha256": sha256(collision_path),
        "schema_version": "synthetic_prior_v1",
        "required_bindings": {},
    }
    with pytest.raises(ValueError, match="overlap prior c32/c33"):
        audit_prior_mask_nonoverlap(
            masks,
            artifact_specs=(collision_spec,),
            expected_prior_unique_rows=1,
        )


def test_reviewed_parent_hashes_and_query_budget_are_fixed() -> None:
    assert PARENT_E29_14C_REGISTRATION_SHA256 == (
        "bc2a6766b8741f0d9427daebe480c261532a7b1832daa049e98a4e6f16005476"
    )
    assert PARENT_E29_14C_SCORE_SUMMARY_SHA256 == (
        "0da2fff7fb113c8cfc9ee50cac564b9eb85c01f20846019769fcdc556ff4d33a"
    )
    assert QUERY_ACCOUNTING["maximum_logical_queries"] == 7_560
    assert PHYSICAL_QUERY_ACCOUNTING["maximum_physical_model_examples"] == 8_816
