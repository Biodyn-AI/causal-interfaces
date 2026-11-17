"""Focused tests for prospective E29.14d registration assembly."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    BOOTSTRAP,
    PHYSICAL_QUERY_ACCOUNTING,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    build_condition_masks,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_direct_backup_custody import (
    BATCH_INVARIANCE,
    OUTCOME_CUSTODY,
    PARENT_E29_14C_REGISTRATION_SHA256,
    PARENT_E29_14C_SCORE_SUMMARY_SHA256,
    PROTOCOL_ARTIFACT_SCHEMA,
    SOURCE_FILES,
    atomic_torch_save,
    build_clean_metadata_contract,
    build_protocol_artifact,
    validate_registration_payload,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_prompt_panel import (
    e29_14d_prompt_panel,
    validate_e29_14d_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14d_direct_backup import (
    DEFAULT_OUTPUT,
    DEFAULT_REGISTRATION,
    PARENT_REGISTRATION,
    build_registration_record,
)


def _passing_parent_summary() -> dict[str, Any]:
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


@lru_cache(maxsize=1)
def _prompt_material() -> tuple[list[dict[str, Any]], Any, Any]:
    from transformers import AutoTokenizer

    parent = json.loads(PARENT_REGISTRATION.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        parent["model_snapshot"], local_files_only=True
    )
    checks = validate_e29_14d_prompt_panel(tokenizer, e29_14d_prompt_panel())
    prompt_keys = (
        "template",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
    )
    records = [{key: str(check[key]) for key in prompt_keys} for check in checks]
    return records, checks, build_clean_metadata_contract(checks, tokenizer)


def _record(tmp_path: Path) -> dict[str, Any]:
    parent = json.loads(PARENT_REGISTRATION.read_text(encoding="utf-8"))
    records, checks, clean = _prompt_material()
    artifact_path = tmp_path / "protocol_conditions.pt"
    atomic_torch_save(build_protocol_artifact(build_condition_masks()), artifact_path)
    return build_registration_record(
        parent_registration_path=PARENT_REGISTRATION,
        parent=parent,
        parent_score_summary_path=tmp_path / "parent_summary.json",
        parent_score_summary=_passing_parent_summary(),
        protocol_artifact_path=artifact_path,
        prompt_records=records,
        prompt_checks=checks,
        clean_metadata_contract=clean,
        source_hashes={name: "1" * 64 for name in SOURCE_FILES},
        environment={"python": "test"},
    )


@pytest.mark.original_checkout
def test_registration_seals_fresh_prompts_protocol_gates_and_accounting(
    tmp_path: Path,
) -> None:
    registration = _record(tmp_path)

    validate_registration_payload(registration, verify_external_files=False)

    assert registration["parent_e29_14c_registration_sha256"] == (
        PARENT_E29_14C_REGISTRATION_SHA256
    )
    assert registration["parent_e29_14c_score_summary_sha256"] == (
        PARENT_E29_14C_SCORE_SUMMARY_SHA256
    )
    assert registration["protocol_artifact_schema"] == PROTOCOL_ARTIFACT_SCHEMA
    assert registration["bootstrap"] == BOOTSTRAP
    assert registration["scoring_gates"] == SCORING_GATES
    assert registration["query_accounting"] == QUERY_ACCOUNTING
    assert registration["physical_query_accounting"] == json.loads(
        json.dumps(PHYSICAL_QUERY_ACCOUNTING)
    )
    assert registration["batch_invariance"] == BATCH_INVARIANCE
    assert registration["outcome_custody"] == OUTCOME_CUSTODY
    assert registration["prospective_scope"]["class_balanced_backgrounds"] is True
    assert (
        registration["prospective_scope"][
            "parent_posthoc_responses_selected_top4_factorial_components"
        ]
        is True
    )
    assert (
        registration["prospective_scope"][
            "parent_posthoc_responses_motivated_attention_vs_mlp_primary_hypothesis"
        ]
        is True
    )
    assert (
        registration["prospective_scope"][
            "parent_responses_used_to_construct_class_balanced_scan_backgrounds"
        ]
        is False
    )
    assert (
        registration["prospective_scope"]["parent_responses_used_to_set_numeric_gates"]
        is False
    )
    assert (
        registration["prospective_scope"]["no_norm_adaptive_conditions_or_dosing"]
        is True
    )
    assert len(registration["prompt_records"]) == 12
    assert len(registration["condition_manifest"]) == 314
    assert len(registration["query_manifest"]) == 628
    assert registration["posthoc_candidate_provenance"]["ranking_order"] == [
        38,
        42,
        40,
        36,
    ]
    assert registration["prior_mask_audit"]["prior_c32_c33_unique_rows"] == 144
    assert registration["prior_mask_audit"]["overlap_count"] == 0


@pytest.mark.original_checkout
def test_registration_rejects_protocol_query_and_checkpoint_drift(
    tmp_path: Path,
) -> None:
    registration = _record(tmp_path)

    changed = copy.deepcopy(registration)
    changed["query_accounting"]["maximum_logical_queries"] += 1
    with pytest.raises(RuntimeError, match="query accounting"):
        validate_registration_payload(changed, verify_external_files=False)

    changed = copy.deepcopy(registration)
    changed["condition_manifest"][0]["condition_id"] = "changed"
    with pytest.raises(RuntimeError, match="condition manifest"):
        validate_registration_payload(changed, verify_external_files=False)

    changed = copy.deepcopy(registration)
    changed["checkpoint_sha256"] = {}
    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_registration_payload(changed, verify_external_files=False)

    changed = copy.deepcopy(registration)
    changed["posthoc_candidate_provenance"]["ranking_order"] = [36, 38, 40, 42]
    with pytest.raises(RuntimeError, match="posthoc candidate provenance"):
        validate_registration_payload(changed, verify_external_files=False)

    changed = copy.deepcopy(registration)
    changed["prior_mask_audit"]["overlap_count"] = 1
    with pytest.raises(RuntimeError, match="prior-mask audit"):
        validate_registration_payload(changed, verify_external_files=False)


@pytest.mark.original_checkout
def test_registration_retains_parent_pass_and_selection_not_numeric_effects(
    tmp_path: Path,
) -> None:
    registration = _record(tmp_path)
    serialized = json.dumps(registration, sort_keys=True)

    assert set(registration["parent_e29_14c_pass_contract"]) == {
        "integrity_passed",
        "all_registered_integrity_gates_passed",
        "high_scale_lexical_scope_authorized",
        "complete_hub_off_scale_ladder",
        "recommended_interpretation",
    }
    assert "scale_response_geometry" not in serialized
    assert "transition_response_increment_diagnostics" not in serialized
    assert "scale_scope_labels" not in serialized


def test_default_paths_are_new_and_write_once() -> None:
    assert DEFAULT_OUTPUT.name == "e29_14d_direct_backup_v1"
    assert DEFAULT_REGISTRATION.name == (
        "E29.14d-DIRECT-CAUSAL-CARRIER-REGISTRATION.json"
    )
