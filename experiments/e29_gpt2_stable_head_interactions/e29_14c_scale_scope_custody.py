"""Strict custody for the prospectively promoted E29.14c scale ladder."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    atomic_json_save,
    atomic_torch_save,
    canonical_sha256,
    load_verified_json,
    require_exact_keys,
    runtime_environment,
    sha256,
    verify_checkpoint_closure,
    verify_file_hash,
    verify_source_closure,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    validate_registration_payload as validate_e29_14b_registration,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    BOOTSTRAP,
    CONTEXT_COUNT,
    EXPECTED_MASK_CANONICAL_SHA256,
    HUB_COMPONENT,
    MASK_SPLITS,
    PHYSICAL_QUERY_ACCOUNTING,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    build_scale_masks,
    build_scale_scope_protocol,
    validate_scale_masks,
    validate_scale_scope_protocol,
)

PARENT_E29_14B_REGISTRATION_SHA256 = (
    "74f9a32093e50dec829f9a4d93da93b5d1db5e5f4f5a1c54e8a507d3deee18ad"
)
PARENT_E29_14B_SCORE_SUMMARY_SHA256 = (
    "fa7ae4d61ae4dd520901f4863f8790f6d9d85d8a4372c43d9e1fd74b3e64157f"
)
PARENT_E29_14B_PROMPT_PANEL_SHA256 = (
    "2411d821fdc3599beb812ed8104bc86b7f53058b570dbdff7b89be6f71e29ef6"
)
SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256 = (
    "64d8d934887855dbe0e0ec89744ebc5120c70b9a517b9fe3660b12d924727994"
)
SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256 = (
    "b0c54036e1998ebc54385130e94fd4108caf2c8767b619332b7e956dda91128b"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPERSEDED_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.14c-HUB-OFF-SCALE-LADDER-REGISTRATION.json"
)
SUPERSEDED_OUTPUT = REPOSITORY_ROOT / "runs/e29_14c_hub_off_scale_ladder_v1"
SUPERSEDED_PROTOCOL_MASKS = SUPERSEDED_OUTPUT / "protocol_masks.pt"
PROTOCOL_NOTE = (
    REPOSITORY_ROOT / "notes/E29.14c-HUB-OFF-SCALE-LADDER-PROTOCOL-DRAFT.md"
)
V1_FORBIDDEN_OUTCOME_FILENAMES = (
    "scale_scope_acquisition_attempt.json",
    "scale_scope_labels.pt",
    "scale_scope_acquisition_custody.json",
    "scale_scope_summary.json",
)
SUPERSESSION_REASON = (
    "v1_scorer_did_not_bind_clean_metadata_prompt_fields_and_position_scope_"
    "indices_to_a_registration_sealed_contract"
)
MASK_ARTIFACT_SCHEMA = "e29_14c_scale_scope_protocol_artifact_v2"
REGISTRATION_SCHEMA = "e29_14c_scale_scope_registration_v2"
REGISTRATION_STATUS = (
    "prospective_v2_superseding_preacquisition_v1_after_exact_e29_14b_full_"
    "pass_before_any_e29_14c_model_response"
)

# Acquisition and scoring deliberately appear before they exist. This prevents
# sealing a protocol whose executable outcome path is not yet hash-bound.
SOURCE_FILES = (
    "acquire_e29_14c_scale_scope.py",
    "e29_14c_scale_scope_custody.py",
    "e29_14c_scale_scope_protocol.py",
    "register_e29_14c_scale_scope.py",
    "score_e29_14c_scale_scope.py",
    "e29_14b_component_position_custody.py",
    "e29_14b_component_position_protocol.py",
    "e29_14b_component_position_runner.py",
    "e29_14b_prompt_panel.py",
    "e29_8_prompt_panel.py",
    "e29_9_seven_gate_protocol.py",
    "e29_10_float64_protocol.py",
    "e29_11_multiscale_protocol.py",
    "e29_12_prompt_panel.py",
    "e29_13_prompt_panel.py",
    "e29_14a_prompt_panel.py",
    "e29_14a_position_protocol.py",
    "generalized_sublayer_patch.py",
)

BATCH_INVARIANCE = {
    "context_index": 0,
    "condition_rows": QUERY_ACCOUNTING["patched_condition_rows_per_context"],
    "reference_batch_size": 1,
    "candidate_batch_size": 32,
    "absolute_tolerance": 1e-9,
    "relative_tolerance": 1e-9,
    "physical_condition_cells": PHYSICAL_QUERY_ACCOUNTING[
        "batch_smoke_physical_condition_cells"
    ],
    "smoke_outputs_enter_scientific_labels": False,
}
OUTCOME_CUSTODY = {
    "exact_e29_14b_registration_and_full_pass_required": True,
    "exact_preacquisition_e29_14c_v1_supersession_required": True,
    "registration_sha_required_before_acquisition": True,
    "source_checkpoint_masks_parent_and_environment_verified_before_attempt": True,
    "write_once_attempt_marker_created_before_model_load": True,
    "any_attempt_marker_blocks_silent_rerun": True,
    "all_286_conditions_acquired_nonadaptively_per_context": True,
    "batch_smoke_runs_before_scientific_label_write": True,
    "batch_smoke_outputs_excluded_from_scientific_labels": True,
    "clean_and_numerical_gates_required_before_label_write": True,
    "attempt_labels_and_custody_are_write_once": True,
}
PARENT_PASS_CONTRACT = {
    "integrity_passed": True,
    "all_registered_integrity_gates_passed": True,
    "hub_off_scale_promotion_authorized": True,
    "full_component_position_bridge": True,
    "recommended_interpretation": (
        "fresh_context_c24_c32_component_position_bridge_confirmed"
    ),
}

MASK_ARTIFACT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "parent_e29_14b_registration_sha256",
        "parent_e29_14b_score_summary_sha256",
        "supersedes_e29_14c_v1_registration_sha256",
        "supersedes_e29_14c_v1_protocol_masks_sha256",
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2",
        "scale_scope_protocol",
        "scale_scope_protocol_sha256",
        "held_masks",
    }
)
REGISTRATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "prospective_scope",
        "parent_e29_14b_registration",
        "parent_e29_14b_registration_sha256",
        "parent_e29_14b_score_summary",
        "parent_e29_14b_score_summary_sha256",
        "parent_e29_14b_pass_contract",
        "supersedes_e29_14c_v1_registration",
        "supersedes_e29_14c_v1_registration_sha256",
        "supersedes_e29_14c_v1_protocol_masks",
        "supersedes_e29_14c_v1_protocol_masks_sha256",
        "supersession",
        "protocol_note",
        "protocol_note_sha256",
        "protocol_masks",
        "protocol_masks_sha256",
        "protocol_masks_schema",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "component_map",
        "hub_component",
        "patch_semantics",
        "prompt_records",
        "prompt_checks",
        "prompt_panel_sha256",
        "clean_metadata_contract",
        "clean_metadata_contract_sha256",
        "scale_scope_protocol",
        "scale_scope_protocol_sha256",
        "mask_canonical_sha256",
        "bootstrap",
        "scoring_gates",
        "query_accounting",
        "physical_query_accounting",
        "batch_invariance",
        "numerical_gates",
        "outcome_custody",
        "source_code_sha256",
        "protocol_deviations",
    }
)
ATTEMPT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "registration",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "parent_e29_14b_registration_sha256_verified_before_attempt",
        "parent_e29_14b_score_summary_sha256_verified_before_attempt",
        "superseded_e29_14c_v1_registration_sha256_verified_before_attempt",
        "superseded_e29_14c_v1_protocol_masks_sha256_verified_before_attempt",
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2_verified_before_attempt",
        "source_code_sha256_verified_before_attempt",
        "checkpoint_sha256_verified_before_attempt",
        "protocol_masks_sha256_verified_before_attempt",
        "prompt_panel_sha256",
        "clean_metadata_contract_sha256",
        "scale_scope_protocol_sha256",
        "query_accounting",
        "physical_query_accounting",
        "environment",
        "execution",
    }
)

CLEAN_METADATA_CONTRACT_KEYS = frozenset(
    {
        "context_index",
        "template",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
        "base_target_token_id",
        "donor_target_token_id",
        "token_length",
        "position_scope_indices",
    }
)
SUPERSESSION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "reason",
        "superseded_e29_14c_v1_registration",
        "superseded_e29_14c_v1_registration_sha256",
        "superseded_e29_14c_v1_protocol_masks",
        "superseded_e29_14c_v1_protocol_masks_sha256",
        "original_protocol_note",
        "original_protocol_note_sha256",
        "forbidden_v1_outcome_paths",
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2",
        "scientific_protocol_unchanged",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_v1_protocol_mask_artifact(path: Path) -> None:
    """Validate the immutable V1 masks without treating V1 as an active seal."""
    verify_file_hash(path, SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    old_keys = {
        "schema_version",
        "parent_e29_14b_registration_sha256",
        "parent_e29_14b_score_summary_sha256",
        "scale_scope_protocol",
        "scale_scope_protocol_sha256",
        "held_masks",
    }
    if not isinstance(artifact, Mapping):
        raise TypeError("superseded E29.14c V1 mask artifact must be a mapping")
    require_exact_keys(artifact, old_keys, label="superseded E29.14c V1 masks")
    if (
        artifact["schema_version"] != "e29_14c_scale_scope_protocol_artifact_v1"
        or artifact["parent_e29_14b_registration_sha256"]
        != PARENT_E29_14B_REGISTRATION_SHA256
        or artifact["parent_e29_14b_score_summary_sha256"]
        != PARENT_E29_14B_SCORE_SUMMARY_SHA256
    ):
        raise RuntimeError("superseded E29.14c V1 mask parent binding changed")
    protocol = validate_scale_scope_protocol(artifact["scale_scope_protocol"])
    if artifact["scale_scope_protocol_sha256"] != canonical_sha256(protocol):
        raise RuntimeError("superseded E29.14c V1 protocol hash changed")
    masks = validate_scale_masks(artifact["held_masks"])
    expected = build_scale_masks()
    if any(not torch.equal(masks[split], expected[split]) for split in MASK_SPLITS):
        raise RuntimeError("superseded E29.14c V1 scientific masks changed")


def verify_v1_preacquisition_state(
    registration_path: Path = SUPERSEDED_REGISTRATION,
    protocol_masks_path: Path = SUPERSEDED_PROTOCOL_MASKS,
) -> dict[str, Any]:
    """Prove V1 was sealed but produced no attempt, labels, custody, or score."""
    registration = load_verified_json(
        registration_path, SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
    )
    if (
        registration.get("schema_version")
        != "e29_14c_scale_scope_registration_v1"
        or registration.get("status")
        != "prospective_sealed_after_exact_e29_14b_full_pass_before_any_"
        "e29_14c_model_response"
    ):
        raise RuntimeError("superseded E29.14c V1 registration identity changed")
    expected_science = {
        "prompt_panel_sha256": PARENT_E29_14B_PROMPT_PANEL_SHA256,
        "scale_scope_protocol": build_scale_scope_protocol(),
        "scale_scope_protocol_sha256": canonical_sha256(
            build_scale_scope_protocol()
        ),
        "mask_canonical_sha256": _jsonable(EXPECTED_MASK_CANONICAL_SHA256),
        "bootstrap": _jsonable(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "protocol_note": str(PROTOCOL_NOTE.resolve()),
        "protocol_note_sha256": sha256(PROTOCOL_NOTE),
        "protocol_masks": str(protocol_masks_path.resolve()),
        "protocol_masks_sha256": SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256,
    }
    if any(registration.get(key) != value for key, value in expected_science.items()):
        raise RuntimeError("superseded E29.14c V1 scientific protocol changed")
    _validate_v1_protocol_mask_artifact(protocol_masks_path)
    forbidden_paths = [
        SUPERSEDED_OUTPUT / filename for filename in V1_FORBIDDEN_OUTCOME_FILENAMES
    ]
    if any(path.exists() for path in forbidden_paths):
        raise RuntimeError("E29.14c V1 has an attempt or scientific outcome artifact")
    record = {
        "schema_version": "e29_14c_v1_preacquisition_supersession_audit_v1",
        "reason": SUPERSESSION_REASON,
        "superseded_e29_14c_v1_registration": str(registration_path.resolve()),
        "superseded_e29_14c_v1_registration_sha256": (
            SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
        ),
        "superseded_e29_14c_v1_protocol_masks": str(
            protocol_masks_path.resolve()
        ),
        "superseded_e29_14c_v1_protocol_masks_sha256": (
            SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
        ),
        "original_protocol_note": str(PROTOCOL_NOTE.resolve()),
        "original_protocol_note_sha256": sha256(PROTOCOL_NOTE),
        "forbidden_v1_outcome_paths": [
            str(path.resolve()) for path in forbidden_paths
        ],
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2": True,
        "scientific_protocol_unchanged": True,
    }
    require_exact_keys(
        record, SUPERSESSION_REQUIRED_KEYS, label="E29.14c V1 supersession"
    )
    return record


def build_clean_metadata_contract(
    prompt_checks: Any, tokenizer: Any
) -> list[dict[str, Any]]:
    """Seal response-blind prompt identities and exact position scopes."""
    if not isinstance(prompt_checks, (tuple, list)) or len(prompt_checks) != CONTEXT_COUNT:
        raise RuntimeError("E29.14c clean metadata requires twelve prompt checks")
    contract = []
    prompt_keys = (
        "template",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
    )
    for context_index, check in enumerate(prompt_checks):
        if not isinstance(check, Mapping) or check.get("context_index") != context_index:
            raise RuntimeError("E29.14c prompt-check order changed")
        base_ids = tokenizer.encode(
            str(check["base_prompt"]), add_special_tokens=False
        )
        donor_ids = tokenizer.encode(
            str(check["donor_prompt"]), add_special_tokens=False
        )
        if (
            not isinstance(base_ids, list)
            or not isinstance(donor_ids, list)
            or base_ids == donor_ids
            or len(base_ids) != len(donor_ids)
            or len(base_ids) != check["token_length"]
        ):
            raise RuntimeError("E29.14c response-blind token alignment changed")
        lexical = [
            index
            for index, (base_id, donor_id) in enumerate(
                zip(base_ids, donor_ids, strict=True)
            )
            if int(base_id) != int(donor_id)
        ]
        if not lexical:
            raise RuntimeError("E29.14c differing-lexical position scope is empty")
        record = {
            "context_index": context_index,
            **{key: str(check[key]) for key in prompt_keys},
            "base_target_token_id": int(check["sides"]["base"]["target_token_id"]),
            "donor_target_token_id": int(
                check["sides"]["donor"]["target_token_id"]
            ),
            "token_length": int(check["token_length"]),
            "position_scope_indices": {
                "all_positions": list(range(len(base_ids))),
                "differing_lexical_positions": lexical,
            },
        }
        require_exact_keys(
            record,
            CLEAN_METADATA_CONTRACT_KEYS,
            label="E29.14c clean metadata contract",
        )
        contract.append(record)
    return contract


def validate_clean_metadata_contract(
    contract: Any, prompt_checks: Any
) -> list[dict[str, Any]]:
    """Validate the sealed clean metadata contract against inherited prompts."""
    if (
        not isinstance(contract, (tuple, list))
        or not isinstance(prompt_checks, (tuple, list))
        or len(contract) != CONTEXT_COUNT
        or len(prompt_checks) != CONTEXT_COUNT
    ):
        raise RuntimeError("E29.14c clean metadata contract count changed")
    validated = []
    prompt_keys = (
        "template",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
    )
    for context_index, (record, check) in enumerate(
        zip(contract, prompt_checks, strict=True)
    ):
        if not isinstance(record, Mapping) or not isinstance(check, Mapping):
            raise TypeError("E29.14c clean metadata contract must contain mappings")
        require_exact_keys(
            record,
            CLEAN_METADATA_CONTRACT_KEYS,
            label="E29.14c clean metadata contract",
        )
        expected = {
            "context_index": context_index,
            **{key: str(check[key]) for key in prompt_keys},
            "base_target_token_id": int(check["sides"]["base"]["target_token_id"]),
            "donor_target_token_id": int(
                check["sides"]["donor"]["target_token_id"]
            ),
            "token_length": int(check["token_length"]),
        }
        if any(record[key] != value for key, value in expected.items()):
            raise RuntimeError("E29.14c clean metadata prompt contract changed")
        scopes = record["position_scope_indices"]
        token_length = expected["token_length"]
        if (
            not isinstance(scopes, Mapping)
            or tuple(scopes) != (
                "all_positions",
                "differing_lexical_positions",
            )
            or scopes["all_positions"] != list(range(token_length))
            or not isinstance(scopes["differing_lexical_positions"], list)
            or not scopes["differing_lexical_positions"]
            or scopes["differing_lexical_positions"]
            != sorted(set(scopes["differing_lexical_positions"]))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= token_length
                for index in scopes["differing_lexical_positions"]
            )
        ):
            raise RuntimeError("E29.14c clean metadata position scope changed")
        validated.append(_jsonable(record))
    return validated


def validate_parent_full_pass(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact E29.14b score schema and its two promotion passes."""
    if not isinstance(summary, Mapping):
        raise TypeError("E29.14b score summary must be a mapping")
    required = {
        "schema_version",
        "integrity_passed",
        "all_registered_integrity_gates_passed",
        "claims",
        "clean_summary",
        "physical_query_accounting",
        "query_accounting",
        "recommended_interpretation",
        "custody",
    }
    require_exact_keys(summary, required, label="E29.14b score summary")
    if summary["schema_version"] != "e29_14b_component_position_bridge_score_v1":
        raise RuntimeError("E29.14b full-pass score schema changed")
    claims = summary["claims"]
    claim_keys = {
        "singleton_transfer",
        "all_position_anchor_by_scale",
        "hub_off_scope_bridge_by_scale",
        "hub_on_scope_control_by_scale",
        "joint_hub_off_scope_bridge",
        "joint_hub_on_scope_control",
        "hub_off_scale_promotion_authorized",
        "full_component_position_bridge",
    }
    if not isinstance(claims, Mapping):
        raise TypeError("E29.14b score claims must be a mapping")
    require_exact_keys(claims, claim_keys, label="E29.14b score claims")
    observed = {
        "integrity_passed": summary["integrity_passed"],
        "all_registered_integrity_gates_passed": summary[
            "all_registered_integrity_gates_passed"
        ],
        "hub_off_scale_promotion_authorized": claims[
            "hub_off_scale_promotion_authorized"
        ].get("passed"),
        "full_component_position_bridge": claims[
            "full_component_position_bridge"
        ].get("passed"),
        "recommended_interpretation": summary["recommended_interpretation"],
    }
    if observed != PARENT_PASS_CONTRACT:
        raise RuntimeError("E29.14b full-pass promotion contract failed")
    return dict(PARENT_PASS_CONTRACT)


def build_protocol_mask_artifact(
    held_masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Build the exact response-blind Torch artifact sealed with registration."""
    masks = validate_scale_masks(held_masks)
    protocol = build_scale_scope_protocol()
    artifact = {
        "schema_version": MASK_ARTIFACT_SCHEMA,
        "parent_e29_14b_registration_sha256": (
            PARENT_E29_14B_REGISTRATION_SHA256
        ),
        "parent_e29_14b_score_summary_sha256": (
            PARENT_E29_14B_SCORE_SUMMARY_SHA256
        ),
        "supersedes_e29_14c_v1_registration_sha256": (
            SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
        ),
        "supersedes_e29_14c_v1_protocol_masks_sha256": (
            SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
        ),
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2": True,
        "scale_scope_protocol": protocol,
        "scale_scope_protocol_sha256": canonical_sha256(protocol),
        "held_masks": {split: masks[split].clone() for split in MASK_SPLITS},
    }
    require_exact_keys(
        artifact, MASK_ARTIFACT_REQUIRED_KEYS, label="E29.14c mask artifact"
    )
    return artifact


def validate_protocol_mask_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Validate parent binding, embedded protocol, and every exact mask byte."""
    if not isinstance(artifact, Mapping):
        raise TypeError("E29.14c protocol-mask artifact must be a mapping")
    require_exact_keys(
        artifact, MASK_ARTIFACT_REQUIRED_KEYS, label="E29.14c mask artifact"
    )
    if (
        artifact["schema_version"] != MASK_ARTIFACT_SCHEMA
        or artifact["parent_e29_14b_registration_sha256"]
        != PARENT_E29_14B_REGISTRATION_SHA256
        or artifact["parent_e29_14b_score_summary_sha256"]
        != PARENT_E29_14B_SCORE_SUMMARY_SHA256
        or artifact["supersedes_e29_14c_v1_registration_sha256"]
        != SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
        or artifact["supersedes_e29_14c_v1_protocol_masks_sha256"]
        != SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
        or artifact[
            "no_v1_attempt_or_labels_or_custody_or_summary_before_v2"
        ]
        is not True
    ):
        raise RuntimeError("E29.14c protocol-mask parent binding changed")
    protocol = validate_scale_scope_protocol(artifact["scale_scope_protocol"])
    if artifact["scale_scope_protocol_sha256"] != canonical_sha256(protocol):
        raise RuntimeError("E29.14c protocol-mask canonical hash changed")
    masks = validate_scale_masks(artifact["held_masks"])
    expected = validate_scale_masks(build_scale_masks())
    if any(not torch.equal(masks[split], expected[split]) for split in MASK_SPLITS):
        raise RuntimeError("E29.14c fixed scale masks changed")
    return masks


def load_verified_protocol_masks(
    path: Path, *, expected_sha256: str
) -> dict[str, torch.Tensor]:
    """Load only the exact response-blind protocol-mask artifact."""
    verify_file_hash(path, expected_sha256)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    return validate_protocol_mask_artifact(artifact)


def load_verified_registration(
    path: Path,
    *,
    expected_sha256: str,
    required_keys: Collection[str] | None = None,
) -> dict[str, Any]:
    """Load a hash-bound registration and optionally enforce its full schema."""
    registration = load_verified_json(path, expected_sha256)
    if required_keys is not None:
        require_exact_keys(registration, required_keys, label="E29.14c registration")
    return registration


def build_acquisition_attempt_record(
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    observed_registration_sha256: str,
    expected_registration_sha256: str,
) -> dict[str, Any]:
    """Build the marker that must be saved immediately before model loading."""
    if (
        observed_registration_sha256 != expected_registration_sha256
        or not _valid_sha256(observed_registration_sha256)
    ):
        raise RuntimeError("E29.14c registration hash differs before attempt")
    marker = {
        "schema_version": "e29_14c_scale_scope_acquisition_attempt_v2",
        "status": "attempt_consumed_before_model_load",
        "registration": str(registration_path.resolve()),
        "registration_sha256": observed_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "parent_e29_14b_registration_sha256_verified_before_attempt": registration[
            "parent_e29_14b_registration_sha256"
        ],
        "parent_e29_14b_score_summary_sha256_verified_before_attempt": registration[
            "parent_e29_14b_score_summary_sha256"
        ],
        "superseded_e29_14c_v1_registration_sha256_verified_before_attempt": registration[
            "supersedes_e29_14c_v1_registration_sha256"
        ],
        "superseded_e29_14c_v1_protocol_masks_sha256_verified_before_attempt": registration[
            "supersedes_e29_14c_v1_protocol_masks_sha256"
        ],
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2_verified_before_attempt": registration[
            "supersession"
        ]["no_v1_attempt_or_labels_or_custody_or_summary_before_v2"],
        "source_code_sha256_verified_before_attempt": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_attempt": dict(
            registration["checkpoint_sha256"]
        ),
        "protocol_masks_sha256_verified_before_attempt": registration[
            "protocol_masks_sha256"
        ],
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "clean_metadata_contract_sha256": registration[
            "clean_metadata_contract_sha256"
        ],
        "scale_scope_protocol_sha256": registration[
            "scale_scope_protocol_sha256"
        ],
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "environment": dict(registration["environment"]),
        "execution": dict(registration["execution"]),
    }
    require_exact_keys(marker, ATTEMPT_REQUIRED_KEYS, label="E29.14c attempt")
    return marker


def create_acquisition_attempt_marker(record: Mapping[str, Any], path: Path) -> str:
    """Create the write-once attempt marker; any marker consumes the attempt."""
    require_exact_keys(record, ATTEMPT_REQUIRED_KEYS, label="E29.14c attempt")
    if record["status"] != "attempt_consumed_before_model_load":
        raise RuntimeError("E29.14c attempt status changed")
    atomic_json_save(dict(record), path)
    return sha256(path)


def _validate_prompt_contract(registration: Mapping[str, Any]) -> None:
    records = registration["prompt_records"]
    checks = registration["prompt_checks"]
    if (
        not isinstance(records, list)
        or not isinstance(checks, list)
        or len(records) != CONTEXT_COUNT
        or len(checks) != CONTEXT_COUNT
    ):
        raise RuntimeError("E29.14c must inherit twelve E29.14b prompts")
    if registration["prompt_panel_sha256"] != PARENT_E29_14B_PROMPT_PANEL_SHA256:
        raise RuntimeError("E29.14c prompt-panel parent hash changed")
    if canonical_sha256(records) != registration["prompt_panel_sha256"]:
        raise RuntimeError("E29.14c prompt records differ from inherited hash")
    for context_index, check in enumerate(checks):
        if not isinstance(check, Mapping) or check.get("context_index") != context_index:
            raise RuntimeError("E29.14c inherited prompt-check order changed")


def validate_registration_payload(
    registration: Mapping[str, Any],
    *,
    source_directory: Path | None = None,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Validate the complete E29.14c scientific, parent, and custody seal."""
    if not isinstance(registration, Mapping):
        raise TypeError("E29.14c registration must be a mapping")
    require_exact_keys(
        registration, REGISTRATION_REQUIRED_KEYS, label="E29.14c registration"
    )
    if (
        registration["schema_version"] != REGISTRATION_SCHEMA
        or registration["status"] != REGISTRATION_STATUS
    ):
        raise RuntimeError("E29.14c registration status changed")
    if (
        registration["parent_e29_14b_registration_sha256"]
        != PARENT_E29_14B_REGISTRATION_SHA256
        or registration["parent_e29_14b_score_summary_sha256"]
        != PARENT_E29_14B_SCORE_SUMMARY_SHA256
    ):
        raise RuntimeError("E29.14c immutable parent hash changed")
    if registration["parent_e29_14b_pass_contract"] != PARENT_PASS_CONTRACT:
        raise RuntimeError("E29.14c parent full-pass contract changed")
    if (
        registration["supersedes_e29_14c_v1_registration_sha256"]
        != SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
        or registration["supersedes_e29_14c_v1_protocol_masks_sha256"]
        != SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
    ):
        raise RuntimeError("E29.14c V1 supersession hash changed")
    supersession = registration["supersession"]
    if not isinstance(supersession, Mapping):
        raise TypeError("E29.14c V1 supersession must be a mapping")
    require_exact_keys(
        supersession,
        SUPERSESSION_REQUIRED_KEYS,
        label="E29.14c V1 supersession",
    )
    if (
        supersession["reason"] != SUPERSESSION_REASON
        or supersession[
            "no_v1_attempt_or_labels_or_custody_or_summary_before_v2"
        ]
        is not True
        or supersession["scientific_protocol_unchanged"] is not True
    ):
        raise RuntimeError("E29.14c V1 supersession contract changed")
    if registration["protocol_masks_schema"] != MASK_ARTIFACT_SCHEMA:
        raise RuntimeError("E29.14c protocol-mask schema changed")

    protocol = validate_scale_scope_protocol(registration["scale_scope_protocol"])
    if registration["scale_scope_protocol_sha256"] != canonical_sha256(protocol):
        raise RuntimeError("E29.14c protocol canonical hash changed")
    expected_values = {
        "mask_canonical_sha256": _jsonable(EXPECTED_MASK_CANONICAL_SHA256),
        "bootstrap": _jsonable(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "batch_invariance": BATCH_INVARIANCE,
        "outcome_custody": OUTCOME_CUSTODY,
        "hub_component": HUB_COMPONENT,
    }
    for key, expected in expected_values.items():
        if registration[key] != expected:
            raise RuntimeError(f"E29.14c {key.replace('_', ' ')} changed")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("E29.14c deviations must be empty at seal")
    if not _valid_sha256(registration["protocol_masks_sha256"]):
        raise RuntimeError("E29.14c protocol-mask hash is invalid")
    checkpoint_hashes = registration["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_hashes, Mapping)
        or not checkpoint_hashes
        or any(Path(str(name)).name != str(name) for name in checkpoint_hashes)
        or any(not _valid_sha256(digest) for digest in checkpoint_hashes.values())
    ):
        raise RuntimeError("E29.14c checkpoint hashes are invalid")
    source_hashes = registration["source_code_sha256"]
    if not isinstance(source_hashes, Mapping):
        raise TypeError("E29.14c source-code hashes must be a mapping")
    require_exact_keys(source_hashes, SOURCE_FILES, label="E29.14c sources")
    if any(not _valid_sha256(digest) for digest in source_hashes.values()):
        raise RuntimeError("E29.14c source-code hash is invalid")
    _validate_prompt_contract(registration)
    clean_contract = validate_clean_metadata_contract(
        registration["clean_metadata_contract"], registration["prompt_checks"]
    )
    if registration["clean_metadata_contract_sha256"] != canonical_sha256(
        clean_contract
    ):
        raise RuntimeError("E29.14c clean metadata contract hash changed")

    execution = registration["execution"]
    if (
        not isinstance(execution, Mapping)
        or execution.get("device") != "cpu"
        or execution.get("model_dtype") != "float64"
        or execution.get("attention_implementation") != "eager"
        or execution.get("batch_size") != 32
    ):
        raise RuntimeError("E29.14c execution contract changed")

    if verify_external_files:
        observed_supersession = verify_v1_preacquisition_state(
            Path(registration["supersedes_e29_14c_v1_registration"]),
            Path(registration["supersedes_e29_14c_v1_protocol_masks"]),
        )
        if supersession != observed_supersession:
            raise RuntimeError("E29.14c V1 pre-acquisition state changed")
        parent_path = Path(registration["parent_e29_14b_registration"])
        parent = load_verified_json(
            parent_path, PARENT_E29_14B_REGISTRATION_SHA256
        )
        parent = validate_e29_14b_registration(
            parent,
            source_directory=Path(__file__).resolve().parent,
            verify_external_files=True,
        )
        summary = load_verified_json(
            Path(registration["parent_e29_14b_score_summary"]),
            PARENT_E29_14B_SCORE_SUMMARY_SHA256,
        )
        if registration["parent_e29_14b_pass_contract"] != validate_parent_full_pass(
            summary
        ):
            raise RuntimeError("E29.14c stored parent-pass contract changed")
        inherited = (
            "model_snapshot",
            "model_revision",
            "checkpoint_sha256",
            "execution",
            "component_map",
            "prompt_records",
            "prompt_checks",
            "prompt_panel_sha256",
            "numerical_gates",
        )
        if any(
            _jsonable(registration[key]) != _jsonable(parent[key])
            for key in inherited
        ):
            raise RuntimeError("E29.14c inherited E29.14b contract changed")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            Path(registration["model_snapshot"]), local_files_only=True
        )
        observed_clean_contract = build_clean_metadata_contract(
            registration["prompt_checks"], tokenizer
        )
        if clean_contract != observed_clean_contract:
            raise RuntimeError("E29.14c token-derived clean metadata contract changed")
        verify_file_hash(
            Path(registration["protocol_note"]),
            str(registration["protocol_note_sha256"]),
        )
        masks = load_verified_protocol_masks(
            Path(registration["protocol_masks"]),
            expected_sha256=str(registration["protocol_masks_sha256"]),
        )
        if canonical_sha256(build_scale_scope_protocol()) != registration[
            "scale_scope_protocol_sha256"
        ] or any(
            not torch.equal(masks[split], build_scale_masks()[split])
            for split in MASK_SPLITS
        ):
            raise RuntimeError("E29.14c external protocol closure changed")
        verify_checkpoint_closure(
            Path(registration["model_snapshot"]), registration["checkpoint_sha256"]
        )
        directory = source_directory or Path(__file__).resolve().parent
        if registration["source_code_sha256"] != verify_source_closure(
            directory, SOURCE_FILES
        ):
            raise RuntimeError("E29.14c source closure changed after registration")
        if registration["environment"] != runtime_environment():
            raise RuntimeError("E29.14c runtime environment differs from registration")
    return dict(registration)


__all__ = [
    "ATTEMPT_REQUIRED_KEYS",
    "BATCH_INVARIANCE",
    "CLEAN_METADATA_CONTRACT_KEYS",
    "MASK_ARTIFACT_REQUIRED_KEYS",
    "MASK_ARTIFACT_SCHEMA",
    "OUTCOME_CUSTODY",
    "PARENT_E29_14B_REGISTRATION_SHA256",
    "PARENT_E29_14B_SCORE_SUMMARY_SHA256",
    "PHYSICAL_QUERY_ACCOUNTING",
    "REGISTRATION_REQUIRED_KEYS",
    "REGISTRATION_SCHEMA",
    "REGISTRATION_STATUS",
    "SOURCE_FILES",
    "SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256",
    "SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256",
    "SUPERSEDED_OUTPUT",
    "SUPERSEDED_PROTOCOL_MASKS",
    "SUPERSEDED_REGISTRATION",
    "SUPERSESSION_REASON",
    "SUPERSESSION_REQUIRED_KEYS",
    "atomic_json_save",
    "atomic_torch_save",
    "build_acquisition_attempt_record",
    "build_clean_metadata_contract",
    "build_protocol_mask_artifact",
    "canonical_sha256",
    "create_acquisition_attempt_marker",
    "load_verified_json",
    "load_verified_protocol_masks",
    "load_verified_registration",
    "require_exact_keys",
    "runtime_environment",
    "sha256",
    "validate_clean_metadata_contract",
    "validate_parent_full_pass",
    "validate_protocol_mask_artifact",
    "validate_registration_payload",
    "verify_checkpoint_closure",
    "verify_file_hash",
    "verify_source_closure",
    "verify_v1_preacquisition_state",
]
