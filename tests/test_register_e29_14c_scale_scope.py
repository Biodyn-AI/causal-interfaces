"""Focused tests for prospective E29.14c registration assembly."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    BATCH_INVARIANCE,
    MASK_ARTIFACT_SCHEMA,
    PARENT_E29_14B_REGISTRATION_SHA256,
    PARENT_E29_14B_SCORE_SUMMARY_SHA256,
    PHYSICAL_QUERY_ACCOUNTING,
    SOURCE_FILES,
    SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256,
    SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256,
    atomic_torch_save,
    build_clean_metadata_contract,
    build_protocol_mask_artifact,
    validate_registration_payload,
    verify_v1_preacquisition_state,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    BOOTSTRAP,
    MASK_SPLITS,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    build_scale_masks,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14c_scale_scope import (
    DEFAULT_OUTPUT,
    DEFAULT_REGISTRATION,
    PARENT_REGISTRATION,
    PARENT_SCORE_SUMMARY,
    PROTOCOL_NOTE,
    SUPERSEDED_PROTOCOL_MASKS,
    SUPERSEDED_REGISTRATION,
    build_registration_record,
)


@lru_cache(maxsize=1)
def _clean_contract() -> tuple[dict[str, Any], ...]:
    from transformers import AutoTokenizer

    parent = json.loads(PARENT_REGISTRATION.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        parent["model_snapshot"], local_files_only=True
    )
    return tuple(build_clean_metadata_contract(parent["prompt_checks"], tokenizer))


def _record(tmp_path: Path) -> dict[str, Any]:
    parent = json.loads(PARENT_REGISTRATION.read_text(encoding="utf-8"))
    parent_summary = json.loads(PARENT_SCORE_SUMMARY.read_text(encoding="utf-8"))
    masks = build_scale_masks()
    masks_path = tmp_path / "protocol_masks.pt"
    atomic_torch_save(build_protocol_mask_artifact(masks), masks_path)
    return build_registration_record(
        parent_registration_path=PARENT_REGISTRATION,
        parent=parent,
        parent_score_summary_path=PARENT_SCORE_SUMMARY,
        parent_score_summary=parent_summary,
        protocol_note_path=PROTOCOL_NOTE,
        protocol_masks_path=masks_path,
        held_masks=masks,
        clean_metadata_contract=_clean_contract(),
        supersession=verify_v1_preacquisition_state(),
        source_hashes={name: "1" * 64 for name in SOURCE_FILES},
        environment={"python": "test"},
    )


@pytest.mark.original_checkout
def test_registration_seals_parent_pass_masks_gates_and_exact_accounting(
    tmp_path: Path,
) -> None:
    registration = _record(tmp_path)

    validate_registration_payload(registration, verify_external_files=False)

    assert registration["parent_e29_14b_registration_sha256"] == (
        PARENT_E29_14B_REGISTRATION_SHA256
    )
    assert registration["parent_e29_14b_score_summary_sha256"] == (
        PARENT_E29_14B_SCORE_SUMMARY_SHA256
    )
    assert registration["parent_e29_14b_pass_contract"] == {
        "integrity_passed": True,
        "all_registered_integrity_gates_passed": True,
        "hub_off_scale_promotion_authorized": True,
        "full_component_position_bridge": True,
        "recommended_interpretation": (
            "fresh_context_c24_c32_component_position_bridge_confirmed"
        ),
    }
    assert registration["prospective_scope"][
        "parent_numeric_responses_used_to_construct_masks_or_gates"
    ] is False
    assert registration["prospective_scope"][
        "parent_summary_used_only_for_exact_full_pass_contract"
    ] is True
    assert registration["bootstrap"] == BOOTSTRAP
    assert registration["scoring_gates"] == SCORING_GATES
    assert registration["query_accounting"] == QUERY_ACCOUNTING
    assert registration["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert registration["batch_invariance"] == BATCH_INVARIANCE
    assert registration["protocol_masks_schema"] == MASK_ARTIFACT_SCHEMA
    assert registration["scale_scope_protocol"]["mask_splits"] == list(MASK_SPLITS)
    assert registration["schema_version"] == "e29_14c_scale_scope_registration_v2"
    assert registration["supersedes_e29_14c_v1_registration_sha256"] == (
        SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
    )
    assert registration["supersedes_e29_14c_v1_protocol_masks_sha256"] == (
        SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
    )
    assert registration["supersession"][
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2"
    ] is True
    assert len(registration["clean_metadata_contract"]) == 12


@pytest.mark.original_checkout
def test_registration_rejects_parent_pass_protocol_query_and_checkpoint_drift(
    tmp_path: Path,
) -> None:
    registration = _record(tmp_path)

    changed_parent = copy.deepcopy(registration)
    changed_parent["parent_e29_14b_pass_contract"][
        "full_component_position_bridge"
    ] = False
    with pytest.raises(RuntimeError, match="parent.*pass"):
        validate_registration_payload(changed_parent, verify_external_files=False)

    changed_protocol = copy.deepcopy(registration)
    changed_protocol["scale_scope_protocol"]["mask_rows"]["scale36"][0][0] = 1
    with pytest.raises((RuntimeError, ValueError), match="mask|protocol|hub"):
        validate_registration_payload(changed_protocol, verify_external_files=False)

    changed_query = copy.deepcopy(registration)
    changed_query["physical_query_accounting"][
        "maximum_physical_model_examples"
    ] += 1
    with pytest.raises(RuntimeError, match="physical query accounting"):
        validate_registration_payload(changed_query, verify_external_files=False)

    changed_checkpoint = copy.deepcopy(registration)
    changed_checkpoint["checkpoint_sha256"] = {}
    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_registration_payload(changed_checkpoint, verify_external_files=False)


@pytest.mark.original_checkout
def test_registration_inherits_exact_prompt_and_execution_contract(
    tmp_path: Path,
) -> None:
    parent = json.loads(PARENT_REGISTRATION.read_text(encoding="utf-8"))
    registration = _record(tmp_path)

    for key in (
        "prompt_records",
        "prompt_checks",
        "prompt_panel_sha256",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "execution",
        "component_map",
        "numerical_gates",
    ):
        assert registration[key] == parent[key]


@pytest.mark.original_checkout
def test_registration_contains_only_parent_pass_not_parent_response_values(
    tmp_path: Path,
) -> None:
    registration = _record(tmp_path)
    serialized = json.dumps(registration, sort_keys=True)

    assert set(registration["parent_e29_14b_pass_contract"]) == {
        "integrity_passed",
        "all_registered_integrity_gates_passed",
        "hub_off_scale_promotion_authorized",
        "full_component_position_bridge",
        "recommended_interpretation",
    }
    assert "parent_e29_14b_score_summary_payload" not in registration
    assert "all_position_anchor_by_scale" not in serialized
    assert "hub_off_scope_bridge_by_scale" not in serialized
    assert "component_position_labels" not in serialized


def test_default_paths_are_new_and_do_not_overlap_e29_14b() -> None:
    assert DEFAULT_OUTPUT.name == "e29_14c_hub_off_scale_ladder_v2"
    assert DEFAULT_REGISTRATION.name == (
        "E29.14c-HUB-OFF-SCALE-LADDER-REGISTRATION-V2.json"
    )
    assert SUPERSEDED_REGISTRATION.name == (
        "E29.14c-HUB-OFF-SCALE-LADDER-REGISTRATION.json"
    )
    assert SUPERSEDED_PROTOCOL_MASKS.parent.name == (
        "e29_14c_hub_off_scale_ladder_v1"
    )
