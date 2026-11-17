"""Strict pre-acquisition custody for E29.14d direct carrier testing.

This module contains no outcome scoring.  It binds the exact fresh prompts,
condition/query manifests, binary intervention panel, executable source closure,
checkpoint, environment, and one write-once acquisition attempt.
"""

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
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    build_clean_metadata_contract,
    validate_clean_metadata_contract,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    validate_registration_payload as validate_e29_14c_registration,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    BOOTSTRAP,
    CANDIDATE_GROUPS,
    CONTEXT_COUNT,
    HUB_COMPONENT,
    PHYSICAL_QUERY_ACCOUNTING,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    build_causal_carrier_protocol,
    build_condition_manifest,
    build_condition_masks,
    build_query_manifest,
    validate_causal_carrier_protocol,
    validate_condition_masks,
    validate_no_prior_c32_c33_overlap,
)

PARENT_E29_14C_REGISTRATION_SHA256 = (
    "bc2a6766b8741f0d9427daebe480c261532a7b1832daa049e98a4e6f16005476"
)
PARENT_E29_14C_SCORE_SUMMARY_SHA256 = (
    "0da2fff7fb113c8cfc9ee50cac564b9eb85c01f20846019769fcdc556ff4d33a"
)
EXPECTED_PROMPT_PANEL_SHA256 = (
    "99ca9954ba0921aa43f92b028c76794e8ee34c6e7916857747ee16a965083551"
)
EXPECTED_PROMPT_RECORDS_SHA256 = (
    "7f1a21347161cdb55a83471f8dd20b33e9eb5314ae185ebd3498088a85d4f09c"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.14c-HUB-OFF-SCALE-LADDER-REGISTRATION-V2.json"
)
PARENT_SCORE_SUMMARY = (
    REPOSITORY_ROOT / "runs/e29_14c_hub_off_scale_ladder_v2/scale_scope_summary.json"
)
POSTHOC_NOMINATIONS = (
    REPOSITORY_ROOT / "runs/e29_14c_hub_off_scale_ladder_v2/"
    "sparse_backup_nominations_posthoc.json"
)
POSTHOC_NOMINATIONS_SHA256 = (
    "74ff2e2680abf19197e35a8db620bedbf3d4945cecae410ecaf7aef4cb14359e"
)
POSTHOC_TOP4_RANKING = (38, 42, 40, 36)

PRIOR_MASK_ARTIFACT_SPECS = (
    {
        "label": "e29_8",
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_8_gpt2_medium_tensor_scaleup_v1/protocol_masks.pt"
            ).resolve()
        ),
        "sha256": "fb0720a00efeda09ca239b57baddb81a372d56fa9bd9a21c657a8a1d9018d4ce",
        "schema_version": "e29_8_stage_a_and_b_response_blind_masks_v1",
        "required_bindings": {"source_seed": 29801, "transfer_seed": 29808},
    },
    {
        "label": "e29_9",
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_9_gpt2_medium_adaptive_transfer_v1/protocol_masks.pt"
            ).resolve()
        ),
        "sha256": "3d9e2612a71741eb83697303577fd3846cc6cd99f0841000e9ac0a202428a95a",
        "schema_version": "e29_9_response_blind_mask_protocol_v1",
        "required_bindings": {
            "sealed_e29_8_masks_sha256": (
                "fb0720a00efeda09ca239b57baddb81a372d56fa9bd9a21c657a8a1d9018d4ce"
            )
        },
    },
    {
        "label": "e29_11",
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_11_gpt2_medium_scale_ladder_v1/protocol_masks.pt"
            ).resolve()
        ),
        "sha256": "c239215507b20ec55482ca90cba9dc4ecb015f42b58a2b399a7b72acd9b6e961",
        "schema_version": "e29_11_response_blind_mask_protocol_v1",
        "required_bindings": {
            "sealed_e29_8_masks_sha256": (
                "fb0720a00efeda09ca239b57baddb81a372d56fa9bd9a21c657a8a1d9018d4ce"
            ),
            "sealed_e29_9_masks_sha256": (
                "3d9e2612a71741eb83697303577fd3846cc6cd99f0841000e9ac0a202428a95a"
            ),
        },
    },
    {
        "label": "e29_12",
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_12_endpoint_boundary_confirmation_v1/protocol_masks.pt"
            ).resolve()
        ),
        "sha256": "ea4dd31f27735edfefd8f50d41110c2ced4d78a71f2830938b877f6b26eca6be",
        "schema_version": "e29_12_endpoint_boundary_masks_v1",
        "required_bindings": {
            "parent_e29_11_registration_sha256": (
                "96987847c9df63da967d78ad467ef306d7e63f3108d19fbff286768289e5fb79"
            ),
            "prior_mask_artifact_sha256": {
                "e29_8": "fb0720a00efeda09ca239b57baddb81a372d56fa9bd9a21c657a8a1d9018d4ce",
                "e29_9": "3d9e2612a71741eb83697303577fd3846cc6cd99f0841000e9ac0a202428a95a",
                "e29_11": "c239215507b20ec55482ca90cba9dc4ecb015f42b58a2b399a7b72acd9b6e961",
            },
        },
    },
    {
        "label": "e29_13",
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_13_absorbing_hub_confirmation_v1/protocol_masks.pt"
            ).resolve()
        ),
        "sha256": "014b87e0ddf2cf913ff745e41e12bb6444420716753d0eb712bd465eaa0c83fd",
        "schema_version": "e29_13_absorbing_hub_protocol_artifact_v1",
        "required_bindings": {
            "parent_e29_12_registration_sha256": (
                "1f7661cd643d37d1abd8b785a170acc0699216420f77f0e248fbb12b776dc078"
            ),
            "prior_mask_artifact_sha256": {
                "e29_8": "fb0720a00efeda09ca239b57baddb81a372d56fa9bd9a21c657a8a1d9018d4ce",
                "e29_9": "3d9e2612a71741eb83697303577fd3846cc6cd99f0841000e9ac0a202428a95a",
                "e29_11": "c239215507b20ec55482ca90cba9dc4ecb015f42b58a2b399a7b72acd9b6e961",
                "e29_12": "ea4dd31f27735edfefd8f50d41110c2ced4d78a71f2830938b877f6b26eca6be",
            },
        },
    },
    {
        "label": "e29_14c",
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_14c_hub_off_scale_ladder_v2/protocol_masks.pt"
            ).resolve()
        ),
        "sha256": "440bb944c2b35801bdb2f93ffa7adab616214574743e552d1ab7821f077a6c25",
        "schema_version": "e29_14c_scale_scope_protocol_artifact_v2",
        "required_bindings": {
            "parent_e29_14b_registration_sha256": (
                "74f9a32093e50dec829f9a4d93da93b5d1db5e5f4f5a1c54e8a507d3deee18ad"
            ),
            "parent_e29_14b_score_summary_sha256": (
                "fa7ae4d61ae4dd520901f4863f8790f6d9d85d8a4372c43d9e1fd74b3e64157f"
            ),
        },
    },
)

PROTOCOL_ARTIFACT_SCHEMA = "e29_14d_direct_backup_protocol_artifact_v1"
REGISTRATION_SCHEMA = "e29_14d_direct_backup_registration_v1"
REGISTRATION_STATUS = "prospective_sealed_before_any_e29_14d_model_response"

# The future scorer is intentionally included before registration.  A seal may
# not be created until the complete executable outcome path exists and hashes.
SOURCE_FILES = (
    "acquire_e29_14d_direct_backup.py",
    "e29_14d_direct_backup_custody.py",
    "e29_14d_causal_carrier_protocol.py",
    "register_e29_14d_direct_backup.py",
    "score_e29_14d_direct_backup.py",
    "e29_14d_prompt_panel.py",
    "e29_14b_component_position_runner.py",
    "e29_14b_component_position_custody.py",
    "e29_14c_scale_scope_custody.py",
    "e29_14c_scale_scope_protocol.py",
    "e29_14b_prompt_panel.py",
    "e29_14a_prompt_panel.py",
    "e29_14a_position_protocol.py",
    "e29_13_prompt_panel.py",
    "e29_12_prompt_panel.py",
    "e29_11_multiscale_protocol.py",
    "e29_10_float64_protocol.py",
    "e29_9_seven_gate_protocol.py",
    "e29_8_prompt_panel.py",
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
    "exact_e29_14c_registration_and_complete_ladder_pass_required": True,
    "registration_sha_required_before_acquisition": True,
    "source_checkpoint_protocol_parent_and_environment_verified_before_attempt": True,
    "write_once_attempt_marker_created_before_model_load": True,
    "any_attempt_marker_blocks_silent_rerun": True,
    "all_registered_conditions_acquired_nonadaptively": True,
    "batch_smoke_runs_before_scientific_label_write": True,
    "batch_smoke_outputs_excluded_from_scientific_labels": True,
    "clean_and_numerical_gates_required_before_label_write": True,
    "attempt_labels_and_custody_are_write_once": True,
    "component_update_norms_are_diagnostic_only": True,
}
PARENT_PASS_CONTRACT = {
    "integrity_passed": True,
    "all_registered_integrity_gates_passed": True,
    "high_scale_lexical_scope_authorized": True,
    "complete_hub_off_scale_ladder": True,
    "recommended_interpretation": (
        "hub_off_c36_c47_scope_equivalence_with_informative_transition_and_"
        "backup_endpoint_confirmed"
    ),
}

PROTOCOL_ARTIFACT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "parent_e29_14c_registration_sha256",
        "parent_e29_14c_score_summary_sha256",
        "posthoc_candidate_provenance",
        "posthoc_candidate_provenance_sha256",
        "prior_mask_audit",
        "prior_mask_audit_sha256",
        "causal_carrier_protocol",
        "causal_carrier_protocol_sha256",
        "condition_manifest",
        "condition_manifest_sha256",
        "query_manifest",
        "query_manifest_sha256",
        "condition_masks",
        "condition_masks_canonical_sha256",
    }
)
REGISTRATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "prospective_scope",
        "parent_e29_14c_registration",
        "parent_e29_14c_registration_sha256",
        "parent_e29_14c_score_summary",
        "parent_e29_14c_score_summary_sha256",
        "parent_e29_14c_pass_contract",
        "posthoc_candidate_provenance",
        "posthoc_candidate_provenance_sha256",
        "prior_mask_audit",
        "prior_mask_audit_sha256",
        "protocol_artifact",
        "protocol_artifact_sha256",
        "protocol_artifact_schema",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "component_map",
        "hub_component",
        "patch_semantics",
        "prompt_records",
        "prompt_records_sha256",
        "prompt_checks",
        "prompt_panel_sha256",
        "clean_metadata_contract",
        "clean_metadata_contract_sha256",
        "causal_carrier_protocol",
        "causal_carrier_protocol_sha256",
        "condition_masks_canonical_sha256",
        "condition_manifest",
        "condition_manifest_sha256",
        "query_manifest",
        "query_manifest_sha256",
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
        "parent_e29_14c_registration_sha256_verified_before_attempt",
        "parent_e29_14c_score_summary_sha256_verified_before_attempt",
        "posthoc_candidate_provenance_sha256_verified_before_attempt",
        "prior_mask_audit_sha256_verified_before_attempt",
        "source_code_sha256_verified_before_attempt",
        "checkpoint_sha256_verified_before_attempt",
        "protocol_artifact_sha256_verified_before_attempt",
        "prompt_panel_sha256",
        "prompt_records_sha256",
        "clean_metadata_contract_sha256",
        "causal_carrier_protocol_sha256",
        "condition_manifest_sha256",
        "query_manifest_sha256",
        "query_accounting",
        "physical_query_accounting",
        "environment",
        "execution",
    }
)

_PROMPT_RECORD_KEYS = frozenset(
    {"template", "base_prompt", "donor_prompt", "base_target", "donor_target"}
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


def _validated_masks(value: Any) -> torch.Tensor:
    validated = validate_condition_masks(value)
    masks = value if validated is None else validated
    if not isinstance(masks, torch.Tensor):
        raise TypeError("E29.14d condition-mask validator did not return a tensor")
    return masks.detach().cpu().to(torch.float64)


def validate_posthoc_candidate_provenance(
    path: Path = POSTHOC_NOMINATIONS,
    *,
    expected_sha256: str = POSTHOC_NOMINATIONS_SHA256,
) -> dict[str, Any]:
    """Bind the exploratory parent artifact that selected the top-four core."""
    artifact = load_verified_json(path, expected_sha256)
    required = {
        "schema_version",
        "status",
        "source_score_summary",
        "source_score_summary_sha256",
        "source_score_schema_version",
        "source_scientific_score_schema_version",
        "source_integrity_passed",
        "method",
        "transition_analyses",
        "overall_consensus",
        "recommended_next_test",
        "limitations",
    }
    require_exact_keys(artifact, required, label="E29.14c posthoc nominations")
    if (
        artifact["schema_version"] != "e29_14c_sparse_backup_nominations_posthoc_v1"
        or artifact["status"]
        != "exploratory_posthoc_candidate_generation_not_confirmatory"
        or artifact["source_score_summary_sha256"]
        != PARENT_E29_14C_SCORE_SUMMARY_SHA256
        or Path(artifact["source_score_summary"]).resolve()
        != PARENT_SCORE_SUMMARY.resolve()
        or artifact["source_score_schema_version"]
        != "e29_14c_hub_off_scale_scope_ladder_score_v2"
        or artifact["source_scientific_score_schema_version"]
        != "e29_14c_hub_off_scale_scope_ladder_score_v1"
        or artifact["source_integrity_passed"] is not True
    ):
        raise RuntimeError("E29.14c posthoc source binding changed")
    consensus = artifact["overall_consensus"]
    if not isinstance(consensus, Mapping):
        raise TypeError("E29.14c posthoc consensus must be a mapping")
    require_exact_keys(
        consensus,
        {"component_ranking", "panels", "top_candidates"},
        label="E29.14c posthoc consensus",
    )
    ranking = consensus["component_ranking"]
    top = consensus["top_candidates"]
    if (
        not isinstance(ranking, list)
        or not isinstance(top, list)
        or len(ranking) != 47
        or len(top) != 8
    ):
        raise RuntimeError("E29.14c posthoc ranking shape changed")
    top4 = tuple(int(record["component"]) for record in ranking[:4])
    top4_from_shortlist = tuple(int(record["component"]) for record in top[:4])
    ranks = tuple(int(record["rank"]) for record in ranking[:4])
    top20 = tuple(int(record["component"]) for record in ranking[:20])
    protocol_core = tuple(
        int(value) for value in CANDIDATE_GROUPS["late_attention_core"]
    )
    if (
        top4 != POSTHOC_TOP4_RANKING
        or top4_from_shortlist != POSTHOC_TOP4_RANKING
        or ranks != (1, 2, 3, 4)
        or tuple(sorted(top4)) != protocol_core
    ):
        raise RuntimeError("E29.14c posthoc top-four provenance changed")
    if any(component % 2 != 0 for component in top20):
        raise RuntimeError("E29.14c posthoc top-twenty attention-class result changed")
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "schema_version": artifact["schema_version"],
        "status": artifact["status"],
        "source_score_summary_sha256": artifact["source_score_summary_sha256"],
        "source_score_schema_version": artifact["source_score_schema_version"],
        "source_scientific_score_schema_version": artifact[
            "source_scientific_score_schema_version"
        ],
        "source_integrity_passed": True,
        "selection_role": (
            "motivated_attention_vs_mlp_primary_hypothesis_and_selected_top4_"
            "factorial_components"
        ),
        "class_hypothesis_role": (
            "motivated_prospective_attention_vs_mlp_primary_hypothesis"
        ),
        "top20_attention_count": len(top20),
        "ranking_order": list(top4),
        "protocol_sorted_core": list(protocol_core),
    }


def _collect_binary_component_panels(value: Any) -> list[torch.Tensor]:
    panels: list[torch.Tensor] = []
    if isinstance(value, Mapping):
        for item in value.values():
            panels.extend(_collect_binary_component_panels(item))
    elif (
        isinstance(value, torch.Tensor)
        and value.ndim == 2
        and value.shape[1] == 48
        and bool(torch.logical_or(value == 0, value == 1).all())
    ):
        panels.append(value.detach().cpu())
    return panels


def audit_prior_mask_nonoverlap(
    condition_masks: Any,
    *,
    artifact_specs: Any = PRIOR_MASK_ARTIFACT_SPECS,
    expected_prior_unique_rows: int = 144,
) -> dict[str, Any]:
    """Verify the exact prior mask closure and prove c32/c33 novelty."""
    if not isinstance(artifact_specs, (tuple, list)) or not artifact_specs:
        raise RuntimeError("E29.14d prior-mask artifact specs are invalid")
    if any(not isinstance(spec, Mapping) for spec in artifact_specs) or len(
        {spec.get("label") for spec in artifact_specs}
    ) != len(artifact_specs):
        raise RuntimeError("E29.14d prior-mask artifact specs are invalid")
    panels: list[torch.Tensor] = []
    verified_specs = []
    for spec in artifact_specs:
        if not isinstance(spec, Mapping):
            raise TypeError("E29.14d prior-mask spec must be a mapping")
        require_exact_keys(
            spec,
            {"label", "path", "sha256", "schema_version", "required_bindings"},
            label="E29.14d prior-mask spec",
        )
        path = Path(str(spec["path"]))
        verify_file_hash(path, str(spec["sha256"]))
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, Mapping):
            raise TypeError("E29.14d prior mask artifact must be a mapping")
        if artifact.get("schema_version") != spec["schema_version"]:
            raise RuntimeError("E29.14d prior-mask artifact schema changed")
        bindings = spec["required_bindings"]
        if not isinstance(bindings, Mapping) or any(
            artifact.get(key) != value for key, value in bindings.items()
        ):
            raise RuntimeError("E29.14d prior-mask source binding changed")
        artifact_panels = _collect_binary_component_panels(artifact)
        if not artifact_panels:
            raise RuntimeError("E29.14d prior artifact contains no component masks")
        panels.extend(artifact_panels)
        verified_specs.append(_jsonable(spec))
    overlap = validate_no_prior_c32_c33_overlap(
        _validated_masks(condition_masks), panels
    )
    if (
        overlap["current_c32_c33_unique_rows"] != 254
        or overlap["prior_c32_c33_unique_rows"] != expected_prior_unique_rows
        or overlap["overlap_count"] != 0
    ):
        raise RuntimeError("E29.14d prior-mask nonoverlap audit counts changed")
    return {
        "schema_version": "e29_14d_prior_c32_c33_nonoverlap_audit_v1",
        "artifact_specs": verified_specs,
        "artifact_count": len(verified_specs),
        "binary_component_panels_scanned": len(panels),
        **overlap,
    }


def validate_parent_full_pass(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the exact E29.14c complete-ladder promotion result."""
    if not isinstance(summary, Mapping):
        raise TypeError("E29.14c score summary must be a mapping")
    required = {
        "schema_version",
        "scientific_score_schema_version",
        "integrity_passed",
        "all_registered_integrity_gates_passed",
        "claims",
        "scale_response_geometry",
        "transition_response_increment_diagnostics",
        "transition_scope_surrogate_failure_localization",
        "query_accounting",
        "physical_query_accounting",
        "clean_summary",
        "custody",
        "recommended_interpretation",
    }
    require_exact_keys(summary, required, label="E29.14c score summary")
    if (
        summary["schema_version"] != "e29_14c_hub_off_scale_scope_ladder_score_v2"
        or summary["scientific_score_schema_version"]
        != "e29_14c_hub_off_scale_scope_ladder_score_v1"
    ):
        raise RuntimeError("E29.14c full-pass score schema changed")
    claims = summary["claims"]
    claim_keys = {
        "singleton_transfer",
        "scope_equivalence_by_scale",
        "joint_equal_scale_scope_equivalence",
        "high_scale_lexical_scope_authorized",
        "informative_c36_nonsaturation",
        "c47_backup_endpoint",
        "complete_hub_off_scale_ladder",
    }
    if not isinstance(claims, Mapping):
        raise TypeError("E29.14c score claims must be a mapping")
    require_exact_keys(claims, claim_keys, label="E29.14c score claims")
    observed = {
        "integrity_passed": summary["integrity_passed"],
        "all_registered_integrity_gates_passed": summary[
            "all_registered_integrity_gates_passed"
        ],
        "high_scale_lexical_scope_authorized": claims[
            "high_scale_lexical_scope_authorized"
        ].get("passed"),
        "complete_hub_off_scale_ladder": claims["complete_hub_off_scale_ladder"].get(
            "passed"
        ),
        "recommended_interpretation": summary["recommended_interpretation"],
    }
    if observed != PARENT_PASS_CONTRACT:
        raise RuntimeError("E29.14c complete-ladder pass contract failed")
    custody = summary["custody"]
    if (
        not isinstance(custody, Mapping)
        or custody.get("registration_sha256") != PARENT_E29_14C_REGISTRATION_SHA256
    ):
        raise RuntimeError("E29.14c score custody does not bind its registration")
    return dict(PARENT_PASS_CONTRACT)


def build_protocol_artifact(condition_masks: Any) -> dict[str, Any]:
    """Build the exact response-blind Torch artifact sealed by registration."""
    masks = _validated_masks(condition_masks)
    expected_masks = _validated_masks(build_condition_masks())
    if not torch.equal(masks, expected_masks):
        raise RuntimeError("E29.14d condition masks differ from the fixed builder")
    protocol = validate_causal_carrier_protocol(build_causal_carrier_protocol())
    conditions = _jsonable(build_condition_manifest())
    queries = _jsonable(build_query_manifest())
    posthoc = validate_posthoc_candidate_provenance()
    prior_mask_audit = audit_prior_mask_nonoverlap(masks)
    artifact = {
        "schema_version": PROTOCOL_ARTIFACT_SCHEMA,
        "parent_e29_14c_registration_sha256": PARENT_E29_14C_REGISTRATION_SHA256,
        "parent_e29_14c_score_summary_sha256": PARENT_E29_14C_SCORE_SUMMARY_SHA256,
        "posthoc_candidate_provenance": posthoc,
        "posthoc_candidate_provenance_sha256": canonical_sha256(posthoc),
        "prior_mask_audit": prior_mask_audit,
        "prior_mask_audit_sha256": canonical_sha256(prior_mask_audit),
        "causal_carrier_protocol": _jsonable(protocol),
        "causal_carrier_protocol_sha256": canonical_sha256(protocol),
        "condition_manifest": conditions,
        "condition_manifest_sha256": canonical_sha256(conditions),
        "query_manifest": queries,
        "query_manifest_sha256": canonical_sha256(queries),
        "condition_masks": masks.clone(),
        "condition_masks_canonical_sha256": canonical_sha256(masks),
    }
    require_exact_keys(
        artifact,
        PROTOCOL_ARTIFACT_REQUIRED_KEYS,
        label="E29.14d protocol artifact",
    )
    return artifact


def validate_protocol_artifact(artifact: Mapping[str, Any]) -> torch.Tensor:
    """Validate the embedded protocol, both manifests, and every mask value."""
    if not isinstance(artifact, Mapping):
        raise TypeError("E29.14d protocol artifact must be a mapping")
    require_exact_keys(
        artifact,
        PROTOCOL_ARTIFACT_REQUIRED_KEYS,
        label="E29.14d protocol artifact",
    )
    if (
        artifact["schema_version"] != PROTOCOL_ARTIFACT_SCHEMA
        or artifact["parent_e29_14c_registration_sha256"]
        != PARENT_E29_14C_REGISTRATION_SHA256
        or artifact["parent_e29_14c_score_summary_sha256"]
        != PARENT_E29_14C_SCORE_SUMMARY_SHA256
    ):
        raise RuntimeError("E29.14d protocol-artifact parent binding changed")
    posthoc = validate_posthoc_candidate_provenance()
    if artifact["posthoc_candidate_provenance"] != posthoc or artifact[
        "posthoc_candidate_provenance_sha256"
    ] != canonical_sha256(posthoc):
        raise RuntimeError("E29.14d posthoc candidate provenance changed")
    protocol = validate_causal_carrier_protocol(artifact["causal_carrier_protocol"])
    if artifact["causal_carrier_protocol_sha256"] != canonical_sha256(protocol):
        raise RuntimeError("E29.14d causal-carrier protocol hash changed")
    conditions = _jsonable(build_condition_manifest())
    queries = _jsonable(build_query_manifest())
    if artifact["condition_manifest"] != conditions or artifact[
        "condition_manifest_sha256"
    ] != canonical_sha256(conditions):
        raise RuntimeError("E29.14d condition manifest changed")
    if artifact["query_manifest"] != queries or artifact[
        "query_manifest_sha256"
    ] != canonical_sha256(queries):
        raise RuntimeError("E29.14d query manifest changed")
    masks = _validated_masks(artifact["condition_masks"])
    expected = _validated_masks(build_condition_masks())
    if not torch.equal(masks, expected) or artifact[
        "condition_masks_canonical_sha256"
    ] != canonical_sha256(masks):
        raise RuntimeError("E29.14d condition masks changed")
    prior_mask_audit = audit_prior_mask_nonoverlap(masks)
    if artifact["prior_mask_audit"] != prior_mask_audit or artifact[
        "prior_mask_audit_sha256"
    ] != canonical_sha256(prior_mask_audit):
        raise RuntimeError("E29.14d prior-mask audit changed")
    return masks


def load_verified_protocol_artifact(
    path: Path, *, expected_sha256: str
) -> torch.Tensor:
    verify_file_hash(path, expected_sha256)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    return validate_protocol_artifact(artifact)


def load_verified_registration(
    path: Path,
    *,
    expected_sha256: str,
    required_keys: Collection[str] | None = None,
) -> dict[str, Any]:
    registration = load_verified_json(path, expected_sha256)
    if required_keys is not None:
        require_exact_keys(registration, required_keys, label="E29.14d registration")
    return registration


def build_acquisition_attempt_record(
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    observed_registration_sha256: str,
    expected_registration_sha256: str,
) -> dict[str, Any]:
    """Build the record written immediately before any model is loaded."""
    if (
        observed_registration_sha256 != expected_registration_sha256
        or not _valid_sha256(observed_registration_sha256)
    ):
        raise RuntimeError("E29.14d registration hash differs before attempt")
    marker = {
        "schema_version": "e29_14d_direct_backup_acquisition_attempt_v1",
        "status": "attempt_consumed_before_model_load",
        "registration": str(registration_path.resolve()),
        "registration_sha256": observed_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "parent_e29_14c_registration_sha256_verified_before_attempt": registration[
            "parent_e29_14c_registration_sha256"
        ],
        "parent_e29_14c_score_summary_sha256_verified_before_attempt": registration[
            "parent_e29_14c_score_summary_sha256"
        ],
        "posthoc_candidate_provenance_sha256_verified_before_attempt": registration[
            "posthoc_candidate_provenance_sha256"
        ],
        "prior_mask_audit_sha256_verified_before_attempt": registration[
            "prior_mask_audit_sha256"
        ],
        "source_code_sha256_verified_before_attempt": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_attempt": dict(
            registration["checkpoint_sha256"]
        ),
        "protocol_artifact_sha256_verified_before_attempt": registration[
            "protocol_artifact_sha256"
        ],
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "prompt_records_sha256": registration["prompt_records_sha256"],
        "clean_metadata_contract_sha256": registration[
            "clean_metadata_contract_sha256"
        ],
        "causal_carrier_protocol_sha256": registration[
            "causal_carrier_protocol_sha256"
        ],
        "condition_manifest_sha256": registration["condition_manifest_sha256"],
        "query_manifest_sha256": registration["query_manifest_sha256"],
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "environment": dict(registration["environment"]),
        "execution": dict(registration["execution"]),
    }
    require_exact_keys(marker, ATTEMPT_REQUIRED_KEYS, label="E29.14d attempt")
    return marker


def create_acquisition_attempt_marker(record: Mapping[str, Any], path: Path) -> str:
    """Consume the single attempt using the shared atomic write-once helper."""
    require_exact_keys(record, ATTEMPT_REQUIRED_KEYS, label="E29.14d attempt")
    if record["status"] != "attempt_consumed_before_model_load":
        raise RuntimeError("E29.14d attempt status changed")
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
        raise RuntimeError("E29.14d requires twelve fresh prompt contexts")
    if registration["prompt_panel_sha256"] != EXPECTED_PROMPT_PANEL_SHA256:
        raise RuntimeError("E29.14d fixed prompt-panel hash changed")
    if (
        registration["prompt_records_sha256"] != EXPECTED_PROMPT_RECORDS_SHA256
        or canonical_sha256(records) != registration["prompt_records_sha256"]
    ):
        raise RuntimeError("E29.14d prompt records differ from their seal")
    for context_index, (record, check) in enumerate(zip(records, checks, strict=True)):
        if not isinstance(record, Mapping) or not isinstance(check, Mapping):
            raise TypeError("E29.14d prompt entries must be mappings")
        require_exact_keys(record, _PROMPT_RECORD_KEYS, label="E29.14d prompt record")
        if check.get("context_index") != context_index:
            raise RuntimeError("E29.14d prompt-check order changed")
        if any(str(record[key]) != str(check[key]) for key in _PROMPT_RECORD_KEYS):
            raise RuntimeError("E29.14d prompt record and tokenizer check differ")
        if check.get("target_positions_aligned") is not True:
            raise RuntimeError("E29.14d target positions are not aligned")


def validate_registration_payload(
    registration: Mapping[str, Any],
    *,
    source_directory: Path | None = None,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Validate the complete E29.14d prospective seal."""
    if not isinstance(registration, Mapping):
        raise TypeError("E29.14d registration must be a mapping")
    require_exact_keys(
        registration, REGISTRATION_REQUIRED_KEYS, label="E29.14d registration"
    )
    if (
        registration["schema_version"] != REGISTRATION_SCHEMA
        or registration["status"] != REGISTRATION_STATUS
    ):
        raise RuntimeError("E29.14d registration status changed")
    if (
        registration["parent_e29_14c_registration_sha256"]
        != PARENT_E29_14C_REGISTRATION_SHA256
        or registration["parent_e29_14c_score_summary_sha256"]
        != PARENT_E29_14C_SCORE_SUMMARY_SHA256
        or registration["parent_e29_14c_pass_contract"] != PARENT_PASS_CONTRACT
    ):
        raise RuntimeError("E29.14d parent pass binding changed")
    if registration["protocol_artifact_schema"] != PROTOCOL_ARTIFACT_SCHEMA:
        raise RuntimeError("E29.14d protocol-artifact schema changed")

    posthoc = validate_posthoc_candidate_provenance()
    prior_mask_audit = audit_prior_mask_nonoverlap(build_condition_masks())
    if registration["posthoc_candidate_provenance"] != posthoc or registration[
        "posthoc_candidate_provenance_sha256"
    ] != canonical_sha256(posthoc):
        raise RuntimeError("E29.14d posthoc candidate provenance changed")
    if registration["prior_mask_audit"] != prior_mask_audit or registration[
        "prior_mask_audit_sha256"
    ] != canonical_sha256(prior_mask_audit):
        raise RuntimeError("E29.14d prior-mask audit changed")

    protocol = validate_causal_carrier_protocol(registration["causal_carrier_protocol"])
    conditions = _jsonable(build_condition_manifest())
    queries = _jsonable(build_query_manifest())
    expected_values = {
        "causal_carrier_protocol_sha256": canonical_sha256(protocol),
        "condition_masks_canonical_sha256": canonical_sha256(
            _validated_masks(build_condition_masks())
        ),
        "condition_manifest": conditions,
        "condition_manifest_sha256": canonical_sha256(conditions),
        "query_manifest": queries,
        "query_manifest_sha256": canonical_sha256(queries),
        "bootstrap": _jsonable(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "batch_invariance": BATCH_INVARIANCE,
        "outcome_custody": OUTCOME_CUSTODY,
        "hub_component": HUB_COMPONENT,
    }
    if registration["causal_carrier_protocol"] != _jsonable(protocol):
        raise RuntimeError("E29.14d causal-carrier protocol changed")
    for key, expected in expected_values.items():
        if registration[key] != expected:
            raise RuntimeError(f"E29.14d {key.replace('_', ' ')} changed")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("E29.14d deviations must be empty at seal")
    if not _valid_sha256(registration["protocol_artifact_sha256"]):
        raise RuntimeError("E29.14d protocol-artifact hash is invalid")

    checkpoint_hashes = registration["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_hashes, Mapping)
        or not checkpoint_hashes
        or any(Path(str(name)).name != str(name) for name in checkpoint_hashes)
        or any(not _valid_sha256(digest) for digest in checkpoint_hashes.values())
    ):
        raise RuntimeError("E29.14d checkpoint hashes are invalid")
    source_hashes = registration["source_code_sha256"]
    if not isinstance(source_hashes, Mapping):
        raise TypeError("E29.14d source-code hashes must be a mapping")
    require_exact_keys(source_hashes, SOURCE_FILES, label="E29.14d sources")
    if any(not _valid_sha256(digest) for digest in source_hashes.values()):
        raise RuntimeError("E29.14d source-code hash is invalid")
    _validate_prompt_contract(registration)
    clean_contract = validate_clean_metadata_contract(
        registration["clean_metadata_contract"], registration["prompt_checks"]
    )
    if registration["clean_metadata_contract_sha256"] != canonical_sha256(
        clean_contract
    ):
        raise RuntimeError("E29.14d clean metadata contract hash changed")
    execution = registration["execution"]
    if (
        not isinstance(execution, Mapping)
        or execution.get("device") != "cpu"
        or execution.get("model_dtype") != "float64"
        or execution.get("attention_implementation") != "eager"
        or execution.get("batch_size") != 32
    ):
        raise RuntimeError("E29.14d execution contract changed")

    if verify_external_files:
        parent = load_verified_json(
            Path(registration["parent_e29_14c_registration"]),
            PARENT_E29_14C_REGISTRATION_SHA256,
        )
        parent = validate_e29_14c_registration(
            parent,
            source_directory=Path(__file__).resolve().parent,
            verify_external_files=True,
        )
        summary = load_verified_json(
            Path(registration["parent_e29_14c_score_summary"]),
            PARENT_E29_14C_SCORE_SUMMARY_SHA256,
        )
        if registration["parent_e29_14c_pass_contract"] != validate_parent_full_pass(
            summary
        ):
            raise RuntimeError("E29.14d stored parent-pass contract changed")
        inherited = (
            "model_snapshot",
            "model_revision",
            "checkpoint_sha256",
            "execution",
            "component_map",
            "numerical_gates",
        )
        if any(
            _jsonable(registration[key]) != _jsonable(parent[key]) for key in inherited
        ):
            raise RuntimeError("E29.14d inherited model contract changed")

        from transformers import AutoTokenizer

        from experiments.e29_gpt2_stable_head_interactions.e29_14d_prompt_panel import (
            e29_14d_prompt_panel,
            validate_e29_14d_prompt_panel,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            Path(registration["model_snapshot"]), local_files_only=True
        )
        panel = e29_14d_prompt_panel()
        if canonical_sha256(panel) != EXPECTED_PROMPT_PANEL_SHA256:
            raise RuntimeError("E29.14d fresh prompt-panel source changed")
        checks = validate_e29_14d_prompt_panel(tokenizer, panel)
        prompt_records = [
            {key: str(check[key]) for key in _PROMPT_RECORD_KEYS} for check in checks
        ]
        if (
            registration["prompt_checks"] != _jsonable(checks)
            or registration["prompt_records"] != prompt_records
        ):
            raise RuntimeError("E29.14d fresh prompt panel changed")
        observed_clean = build_clean_metadata_contract(checks, tokenizer)
        if clean_contract != observed_clean:
            raise RuntimeError("E29.14d token-derived clean metadata changed")

        masks = load_verified_protocol_artifact(
            Path(registration["protocol_artifact"]),
            expected_sha256=registration["protocol_artifact_sha256"],
        )
        if not torch.equal(masks, _validated_masks(build_condition_masks())):
            raise RuntimeError("E29.14d external condition masks changed")
        verify_checkpoint_closure(
            Path(registration["model_snapshot"]), registration["checkpoint_sha256"]
        )
        directory = source_directory or Path(__file__).resolve().parent
        if registration["source_code_sha256"] != verify_source_closure(
            directory, SOURCE_FILES
        ):
            raise RuntimeError("E29.14d source closure changed after registration")
        if registration["environment"] != runtime_environment():
            raise RuntimeError("E29.14d runtime environment differs from registration")
    return dict(registration)


__all__ = [
    "ATTEMPT_REQUIRED_KEYS",
    "BATCH_INVARIANCE",
    "EXPECTED_PROMPT_PANEL_SHA256",
    "EXPECTED_PROMPT_RECORDS_SHA256",
    "OUTCOME_CUSTODY",
    "PARENT_E29_14C_REGISTRATION_SHA256",
    "PARENT_E29_14C_SCORE_SUMMARY_SHA256",
    "PARENT_PASS_CONTRACT",
    "PARENT_REGISTRATION",
    "PARENT_SCORE_SUMMARY",
    "PROTOCOL_ARTIFACT_REQUIRED_KEYS",
    "PROTOCOL_ARTIFACT_SCHEMA",
    "REGISTRATION_REQUIRED_KEYS",
    "REGISTRATION_SCHEMA",
    "REGISTRATION_STATUS",
    "SOURCE_FILES",
    "atomic_json_save",
    "atomic_torch_save",
    "build_acquisition_attempt_record",
    "build_clean_metadata_contract",
    "build_protocol_artifact",
    "canonical_sha256",
    "create_acquisition_attempt_marker",
    "load_verified_protocol_artifact",
    "load_verified_registration",
    "runtime_environment",
    "sha256",
    "validate_clean_metadata_contract",
    "validate_parent_full_pass",
    "validate_protocol_artifact",
    "validate_registration_payload",
    "verify_checkpoint_closure",
    "verify_source_closure",
]
