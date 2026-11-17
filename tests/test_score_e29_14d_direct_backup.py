"""Focused tests for the prospective E29.14d pure statistical scorer."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14d_direct_backup import (
    CLEAN_METADATA_KEYS,
    CUSTODY_KEYS,
    LABEL_KEYS,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    build_condition_manifest,
    build_condition_masks,
    build_query_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.score_e29_14d_direct_backup import (
    _expected_label_numerical_summary,
    _interpretation_branch,
    _registered_scientific_arguments,
    _score_add_one_class,
    _score_dropout,
    _score_factorial,
    _validate_custody_summary,
    score_direct_backup_response_tensors,
    score_from_custody,
    score_label_artifact,
)

COMPONENTS = tuple(component for component in range(48) if component != 1)
ATTENTION = tuple(range(0, 48, 2))
MLP = tuple(range(3, 48, 2))
MATCHED = tuple((2 * layer, 2 * layer + 1) for layer in range(1, 24))
TOP = (36, 38, 40, 42)
FLANK = (32, 34, 44, 46)
BOOTSTRAP = {
    "seed": 29_144,
    "replicates": 10_000,
    "template_strata": ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)),
}
ADD_GATES = {
    "attention_mean_lower_95_min": 0.005,
    "matched_attention_minus_mlp_lower_95_min": 0.005,
    "auc_lower_95_min": 0.75,
    "minimum_positive_layer_contrasts": 18,
    "top_minus_flanking_lower_95_min": 0.005,
}
DROP_GATES = {
    "attention_mean_lower_95_min": 0.002,
    "matched_attention_minus_mlp_lower_95_min": 0.002,
    "individual_top4_simultaneous_lower_95_min": 0.002,
    "matched_top4_simultaneous_lower_95_min": 0.002,
}


def _component_effect(component: int) -> float:
    if component in TOP:
        return 0.030
    if component in FLANK:
        return 0.015
    if component % 2 == 0:
        return 0.012
    return 0.001


def test_class_and_flanking_gates_pass_for_stable_attention_carriers() -> None:
    effects = (
        torch.tensor(
            [_component_effect(component) for component in COMPONENTS],
            dtype=torch.float64,
        )[:, None, None]
        .expand(-1, 3, 12)
        .clone()
    )

    scored = _score_add_one_class(
        effects,
        components=COMPONENTS,
        attention_components=ATTENTION,
        mlp_components=MLP,
        matched_pairs=MATCHED,
        top_components=TOP,
        flanking_components=FLANK,
        bootstrap=BOOTSTRAP,
        gates=ADD_GATES,
    )

    assert scored["class_confirmation_passed"] is True
    assert scored["top4_vs_flanking_secondary_passed"] is True
    assert scored["attention_vs_mlp_auc"]["lower_95"] == 1.0
    assert scored["positive_matched_layer_contrasts"] == 23


def test_class_gate_rejects_no_attention_mlp_separation() -> None:
    effects = torch.full((47, 3, 12), 0.01, dtype=torch.float64)

    scored = _score_add_one_class(
        effects,
        components=COMPONENTS,
        attention_components=ATTENTION,
        mlp_components=MLP,
        matched_pairs=MATCHED,
        top_components=TOP,
        flanking_components=FLANK,
        bootstrap=BOOTSTRAP,
        gates=ADD_GATES,
    )

    assert scored["class_confirmation_passed"] is False
    assert scored["attention_vs_mlp_auc"]["point_estimate"] == 0.5
    assert scored["positive_matched_layer_contrasts"] == 0


def test_factorial_uses_joint_max_error_bound_over_all_twelve_claims() -> None:
    # Cell order is B, B+A, B+M, B+A+M.  A has a 0.03 marginal in
    # both M states while M has a 0.001 marginal in both A states.
    cells = torch.tensor((0.40, 0.43, 0.401, 0.431), dtype=torch.float64)
    cells = cells.expand(4, 3, 12, 4).clone()

    scored = _score_factorial(
        cells,
        top_components=TOP,
        bootstrap=BOOTSTRAP,
        threshold=0.005,
    )

    assert scored["familywise_claim_count"] == 12
    assert scored["all_simultaneous_lower_bounds_passed"] is True
    assert scored["by_attention_component"]["36"]["A0"]["simultaneous_lower_95"] > 0.005


def test_dropout_separates_class_and_simultaneous_top4_necessity() -> None:
    effects = (
        torch.tensor(
            [_component_effect(component) for component in COMPONENTS],
            dtype=torch.float64,
        )[:, None]
        .expand(-1, 12)
        .clone()
    )

    scored = _score_dropout(
        effects,
        components=COMPONENTS,
        attention_components=ATTENTION,
        mlp_components=MLP,
        matched_pairs=MATCHED,
        top_components=TOP,
        bootstrap=BOOTSTRAP,
        gates=DROP_GATES,
    )

    assert scored["class_necessity_passed"] is True
    assert scored["top4_individual_necessity_passed"] is True
    assert len(scored["top4_individual_and_matched_contrasts"]) == 8

    stricter_matched = {**DROP_GATES, "matched_top4_simultaneous_lower_95_min": 0.05}
    rescored = _score_dropout(
        effects,
        components=COMPONENTS,
        attention_components=ATTENTION,
        mlp_components=MLP,
        matched_pairs=MATCHED,
        top_components=TOP,
        bootstrap=BOOTSTRAP,
        gates=stricter_matched,
    )
    assert rescored["top4_individual_loss_threshold"] == 0.002
    assert rescored["top4_attention_minus_matched_mlp_threshold"] == 0.05
    assert rescored["top4_individual_necessity_passed"] is False


def test_public_pure_scorer_normalizes_and_keeps_scope_and_norms_descriptive() -> None:
    normalized_rows: list[torch.Tensor] = []
    add_records = []
    for component in COMPONENTS:
        for family in range(3):
            base = len(normalized_rows)
            normalized_rows.append(torch.full((12,), 0.30, dtype=torch.float64))
            plus = len(normalized_rows)
            normalized_rows.append(
                torch.full(
                    (12,), 0.30 + _component_effect(component), dtype=torch.float64
                )
            )
            add_records.append(
                {
                    "pair_id": f"pair-{component}-{family}",
                    "target_component": component,
                    "target_class": "attention" if component % 2 == 0 else "mlp",
                    "family_id": f"family_{family}",
                    "family_index": family,
                    "base_condition_index": base,
                    "plus_condition_index": plus,
                    "base_condition_id": f"base-{base}",
                    "plus_condition_id": f"plus-{plus}",
                }
            )
    factorial_records = []
    for attention in TOP:
        for background in range(3):
            indices = []
            for value in (0.40, 0.43, 0.401, 0.431):
                indices.append(len(normalized_rows))
                normalized_rows.append(torch.full((12,), value, dtype=torch.float64))
            factorial_records.append(
                {
                    "factorial_id": f"factorial-{attention}-{background}",
                    "attention_component": attention,
                    "mlp_component": attention + 1,
                    "layer": attention // 2,
                    "background_index": background,
                    "family_id": f"family_{background}",
                    "condition_indices": {
                        "neither": indices[0],
                        "attention_only": indices[1],
                        "mlp_only": indices[2],
                        "both": indices[3],
                    },
                    "condition_ids": {
                        "neither": f"condition-{indices[0]}",
                        "attention_only": f"condition-{indices[1]}",
                        "mlp_only": f"condition-{indices[2]}",
                        "both": f"condition-{indices[3]}",
                    },
                }
            )
    full = len(normalized_rows)
    normalized_rows.append(torch.full((12,), 0.99, dtype=torch.float64))
    dropout_records = []
    for component in COMPONENTS:
        dropout = len(normalized_rows)
        normalized_rows.append(
            torch.full((12,), 0.99 - _component_effect(component), dtype=torch.float64)
        )
        dropout_records.append(
            {
                "dropout_id": f"drop-{component}",
                "dropped_component": component,
                "dropped_class": "attention" if component % 2 == 0 else "mlp",
                "full_condition_index": full,
                "dropout_condition_index": dropout,
                "full_condition_id": "full",
                "dropout_condition_id": f"drop-{component}",
            }
        )
    hub_singleton = len(normalized_rows)
    normalized_rows.append(torch.full((12,), 0.995, dtype=torch.float64))
    normalized = torch.stack(normalized_rows)
    endpoints = torch.linspace(8.0, 10.0, 12, dtype=torch.float64)
    raw = normalized * endpoints[None]
    norms = {
        scope: torch.ones((48, 12), dtype=torch.float64)
        for scope in ("all_positions", "differing_lexical_positions")
    }
    gates = {
        "integrity_anchors": {
            "hub_singleton_normalized_mean_min": 0.985,
            "hub_singleton_normalized_mean_max": 1.01,
            "hub_singleton_absolute_endpoint_gap_bootstrap_upper_max": 0.03,
            "c47_full_normalized_mean_min": 0.95,
            "c47_full_normalized_mean_max": 1.01,
            "c47_full_absolute_endpoint_gap_bootstrap_upper_max": 0.05,
        },
        "scope_secondary": {
            "condition_response_mean_absolute_gap_bootstrap_upper_max": 0.03,
            "add_one_effect_mean_absolute_gap_bootstrap_upper_max": 0.03,
            "factorial_contrast_mean_absolute_gap_bootstrap_upper_max": 0.03,
            "dropout_effect_mean_absolute_gap_bootstrap_upper_max": 0.03,
            "context_mean_absolute_gap_max": 0.05,
            "minimum_passing_contexts": 11,
        },
        "balanced_add_one": {
            "attention_class_mean_effect_bootstrap_lower_strict_min": 0.005,
            "matched_layer_attention_minus_mlp_bootstrap_lower_strict_min": 0.005,
            "attention_vs_mlp_component_auc_bootstrap_lower_strict_min": 0.75,
            "minimum_positive_matched_layer_contrasts": 18,
        },
        "top4_factorial": {"simultaneous_lower_band_strict_min": 0.005},
        "regional_flanking_secondary": {
            "late_minus_regional_flanking_bootstrap_lower_strict_min": 0.005
        },
        "single_dropout_secondary": {
            "attention_class_mean_loss_bootstrap_lower_strict_min": 0.002,
            "matched_layer_attention_minus_mlp_loss_lower_strict_min": 0.002,
            "top4_individual_loss_simultaneous_lower_strict_min": 0.002,
            "top4_attention_minus_matched_mlp_loss_simultaneous_lower_strict_min": 0.002,
        },
    }

    scientific_arguments = {
        "condition_count": len(normalized),
        "context_count": 12,
        "position_scopes": ("all_positions", "differing_lexical_positions"),
        "primary_scope": "all_positions",
        "components": COMPONENTS,
        "attention_components": ATTENTION,
        "mlp_components": MLP,
        "matched_pairs": MATCHED,
        "top_components": TOP,
        "flanking_components": FLANK,
        "add_one_records": add_records,
        "factorial_records": factorial_records,
        "dropout_records": dropout_records,
        "family_count": 3,
        "factorial_background_count": 3,
        "hub_singleton_condition_index": hub_singleton,
        "full_condition_index": full,
        "bootstrap": BOOTSTRAP,
        "scoring_gates": gates,
        "integrity_passed": True,
    }
    scored = score_direct_backup_response_tensors(
        {"all_positions": raw, "differing_lexical_positions": raw.clone()},
        endpoints,
        norms,
        norms,
        **scientific_arguments,
    )

    assert scored["primary_claims"]["causal_add_one_program_passed"] is True
    assert scored["all_claims_passed_descriptive_conjunction"] is True
    assert scored["all_claims_descriptive_conjunction_is_a_promotion_gate"] is False
    assert scored["scope_concordance_descriptive"]["add_one_effect_cells"][
        "pearson"
    ] == pytest.approx(1.0)
    assert (
        "not actual multi-component injected norms"
        in scored["clean_reference_update_norm_diagnostics"]["rms"]["all_positions"][
            "warning"
        ]
    )
    assert scored["claim_boundaries"]["norm_matched_causal_lane_was_run"] is False
    assert scored["claim_boundaries"]["identity_only_carrier_claim_authorized"] is False

    failed_anchor_raw = raw.clone()
    failed_anchor_raw[hub_singleton] = 0.80 * endpoints
    failed_anchor = score_direct_backup_response_tensors(
        {
            "all_positions": failed_anchor_raw,
            "differing_lexical_positions": failed_anchor_raw.clone(),
        },
        endpoints,
        norms,
        norms,
        **scientific_arguments,
    )
    assert (
        failed_anchor["analyses_by_scope"]["all_positions"]["add_one"][
            "class_confirmation_passed"
        ]
        is True
    )
    assert failed_anchor["integrity_anchor_claims"]["passed"] is False
    assert not any(failed_anchor["primary_claims"].values())
    assert not any(failed_anchor["independent_dropout_claims"].values())


def test_add_one_confirmation_survives_independent_top4_dropout_failure() -> None:
    branch = _interpretation_branch(
        class_passed=True,
        factorial_passed=True,
        dropout_class_passed=True,
        dropout_top4_passed=False,
    )

    assert branch.startswith("broad_attention_and_late_top4_causal_add_one_confirmed")
    assert branch.endswith(
        "c47_class_necessity_supported_but_top4_necessity_not_supported"
    )


def _registered_label_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    masks = build_condition_masks()
    weights = torch.tensor(
        [_component_effect(component) for component in range(48)],
        dtype=torch.float64,
    )
    normalized = 0.20 + masks @ weights
    manifest = list(build_condition_manifest())
    hub_index = next(
        record["condition_index"]
        for record in manifest
        if tuple(record["selected_components"]) == (1,)
    )
    full_index = next(
        record["condition_index"] for record in manifest if record["cardinality"] == 47
    )
    normalized[hub_index] = 0.995
    normalized[full_index] = 0.99
    endpoints = torch.full((12,), 10.0, dtype=torch.float64)
    raw = normalized[:, None] * endpoints[None]
    responses = {scope: raw.clone() for scope in POSITION_SCOPE_ORDER}
    norms = {
        scope: torch.ones((48, 12), dtype=torch.float64)
        for scope in POSITION_SCOPE_ORDER
    }
    clean_contract: list[dict[str, object]] = []
    clean_metadata = []
    for context_index in range(12):
        contract = {
            "context_index": context_index,
            "template": f"template-{context_index // 3}",
            "base_prompt": f"base-{context_index}",
            "donor_prompt": f"donor-{context_index}",
            "base_target": f"base-target-{context_index}",
            "donor_target": f"donor-target-{context_index}",
            "base_target_token_id": 100 + 2 * context_index,
            "donor_target_token_id": 101 + 2 * context_index,
            "token_length": 4,
            "position_scope_indices": {
                "all_positions": [0, 1, 2, 3],
                "differing_lexical_positions": [0, 1, 3],
            },
        }
        clean_contract.append(contract)
        clean_metadata.append(
            {
                key: contract[key]
                for key in (
                    "template",
                    "base_prompt",
                    "donor_prompt",
                    "base_target",
                    "donor_target",
                    "base_target_token_id",
                    "donor_target_token_id",
                    "token_length",
                    "position_scope_indices",
                )
            }
            | {
                "model_layer_count": 24,
                "component_count": 48,
                "component_semantics": "attention_and_mlp_residual_updates",
                "patch_direction": "donor_to_base_only",
                "position_scope_order": list(POSITION_SCOPE_ORDER),
                "clean_base_correct_minus_donor_margin": 5.0,
                "clean_donor_correct_minus_base_margin": 5.0,
                "clean_endpoint": 10.0,
                "clean_base_logit_absolute_max": 8.0,
                "clean_donor_logit_absolute_max": 8.0,
                "clean_logit_absolute_max": 8.0,
                "clean_logits_finite": True,
                "device": "cpu",
                "model_dtype": "float64",
            }
        )
    smoke_records = {
        scope: {
            "mask_rows": 314,
            "reference_batch_size": 1,
            "candidate_batch_size": 32,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "atol": 1e-9,
            "rtol": 1e-9,
            "passed": True,
        }
        for scope in POSITION_SCOPE_ORDER
    }
    labels: dict[str, object] = {
        "schema_version": "e29_14d_direct_backup_labels_v1",
        "registration_sha256": "a" * 64,
        "expected_registration_sha256_supplied": "a" * 64,
        "condition_manifest": manifest,
        "query_manifest": list(build_query_manifest()),
        "responses_by_scope": responses,
        "clean_metadata": clean_metadata,
        "clean_base_correct_minus_donor_margins": torch.full(
            (12,), 5.0, dtype=torch.float64
        ),
        "clean_donor_correct_minus_base_margins": torch.full(
            (12,), 5.0, dtype=torch.float64
        ),
        "clean_endpoints": endpoints,
        "clean_reference_update_frobenius_norms": norms,
        "clean_reference_update_rms_norms": norms,
        "clean_reference_norm_semantics": {
            "quantity": "clean_donor_update_minus_clean_base_update",
            "scopes": list(POSITION_SCOPE_ORDER),
            "frobenius": "l2_norm_over_selected_positions_and_hidden_coordinates",
            "rms": "root_mean_square_over_selected_positions_and_hidden_coordinates",
            "not_actual_multi_mask_injected_norms": True,
            "not_used_for_condition_selection_or_dosing": True,
            "descriptive_only": True,
        },
        "batch_invariance_smoke": {
            "records": smoke_records,
            "mask_rows": 314,
            "unique_condition_rows": 628,
            "physical_condition_cells": 1256,
            "scientific_labels_produced": False,
            "passed": True,
        },
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": _expected_label_numerical_summary(
            responses, endpoints, norms, norms, POSITION_SCOPE_ORDER
        ),
        "all_integrity_gates_passed": True,
    }
    assert set(labels) == LABEL_KEYS
    assert set(clean_metadata[0]) == CLEAN_METADATA_KEYS
    return labels, clean_contract


def _score_registered_labels(
    labels: dict[str, object], clean_contract: list[dict[str, object]]
) -> dict[str, object]:
    return score_label_artifact(
        labels,
        expected_registration_sha256="a" * 64,
        expected_label_keys=LABEL_KEYS,
        expected_condition_manifest=build_condition_manifest(),
        expected_query_manifest=build_query_manifest(),
        expected_query_accounting=QUERY_ACCOUNTING,
        expected_physical_query_accounting=PHYSICAL_QUERY_ACCOUNTING,
        clean_metadata_contract=clean_contract,
        clean_metadata_keys=CLEAN_METADATA_KEYS,
        scientific_arguments=_registered_scientific_arguments(),
    )


def test_label_scorer_binds_exact_manifests_order_endpoints_and_anchors() -> None:
    labels, clean_contract = _registered_label_fixture()

    scored = _score_registered_labels(labels, clean_contract)

    assert scored["integrity_anchor_claims"]["passed"] is True
    assert scored["primary_claims"]["causal_add_one_program_passed"] is True
    assert scored["scope_secondary_registered_claim"]["passed"] is True

    condition_drift = copy.deepcopy(labels)
    condition_drift["condition_manifest"][0]["condition_index"] = 9
    with pytest.raises(RuntimeError, match="condition manifest"):
        _score_registered_labels(condition_drift, clean_contract)

    query_drift = copy.deepcopy(labels)
    query_drift["query_manifest"][0], query_drift["query_manifest"][1] = (
        query_drift["query_manifest"][1],
        query_drift["query_manifest"][0],
    )
    with pytest.raises(RuntimeError, match="query manifest or order"):
        _score_registered_labels(query_drift, clean_contract)

    endpoint_drift = copy.deepcopy(labels)
    endpoint_drift["clean_endpoints"][0] += 1.0
    with pytest.raises(RuntimeError, match="endpoint arithmetic"):
        _score_registered_labels(endpoint_drift, clean_contract)

    anchor_failure = copy.deepcopy(labels)
    hub_index = next(
        record["condition_index"]
        for record in anchor_failure["condition_manifest"]
        if tuple(record["selected_components"]) == (1,)
    )
    for scope in POSITION_SCOPE_ORDER:
        anchor_failure["responses_by_scope"][scope][hub_index] = (
            0.80 * anchor_failure["clean_endpoints"]
        )
    anchor_failure["numerical_summary"] = _expected_label_numerical_summary(
        anchor_failure["responses_by_scope"],
        anchor_failure["clean_endpoints"],
        anchor_failure["clean_reference_update_frobenius_norms"],
        anchor_failure["clean_reference_update_rms_norms"],
        POSITION_SCOPE_ORDER,
    )
    anchor_scored = _score_registered_labels(anchor_failure, clean_contract)
    assert anchor_scored["integrity_anchor_claims"]["passed"] is False
    assert anchor_scored["primary_claims"]["causal_add_one_program_passed"] is False
    assert anchor_scored["recommended_interpretation"].startswith(
        "integrity_anchors_failed"
    )


def _custody_fixture() -> tuple[dict[str, object], dict[str, object]]:
    registration = {
        "source_code_sha256": {"source.py": "1" * 64},
        "checkpoint_sha256": {"model.safetensors": "2" * 64},
        "protocol_artifact_sha256": "3" * 64,
        "prompt_panel_sha256": "4" * 64,
        "prompt_records_sha256": "5" * 64,
        "posthoc_candidate_provenance_sha256": "d" * 64,
        "prior_mask_audit_sha256": "e" * 64,
        "clean_metadata_contract_sha256": "6" * 64,
        "causal_carrier_protocol_sha256": "7" * 64,
        "condition_manifest_sha256": "8" * 64,
        "query_manifest_sha256": "9" * 64,
        "execution": {"batch_size": 32},
    }
    custody: dict[str, object] = {
        "schema_version": "e29_14d_direct_backup_acquisition_custody_v1",
        "status": "all_registered_direct_backup_conditions_acquired_after_integrity_gates",
        "registration": "/tmp/registration.json",
        "registration_sha256": "a" * 64,
        "expected_registration_sha256_supplied": "a" * 64,
        "source_code_sha256_verified_before_acquisition": registration[
            "source_code_sha256"
        ],
        "checkpoint_sha256_verified_before_acquisition": registration[
            "checkpoint_sha256"
        ],
        "protocol_artifact": "/tmp/protocol.pt",
        "protocol_artifact_sha256_verified_before_acquisition": "3" * 64,
        "protocol_artifact_schema": "e29_14d_direct_backup_protocol_artifact_v1",
        "acquisition_attempt": "/tmp/attempt.json",
        "acquisition_attempt_sha256": "b" * 64,
        "direct_backup_labels": "/tmp/labels.pt",
        "direct_backup_labels_sha256": "c" * 64,
        "prompt_panel_sha256": "4" * 64,
        "prompt_records_sha256": "5" * 64,
        "posthoc_candidate_provenance_sha256": "d" * 64,
        "prior_mask_audit_sha256": "e" * 64,
        "clean_metadata_contract_sha256": "6" * 64,
        "causal_carrier_protocol_sha256": "7" * 64,
        "condition_manifest_sha256": "8" * 64,
        "query_manifest_sha256": "9" * 64,
        "contexts": 12,
        "condition_masks": 314,
        "position_scopes": 2,
        "patched_condition_rows_per_context": 628,
        "patched_context_condition_cells": 7536,
        "clean_forward_examples": 24,
        "batch_smoke_unique_query_rows": 628,
        "batch_smoke_physical_condition_cells": 1256,
        "maximum_logical_queries": 7560,
        "maximum_physical_model_examples": 8816,
        "batch_size": 32,
        "batch_invariance_passed": True,
        "all_clean_and_numerical_gates_passed": True,
        "clean_reference_norms_are_diagnostic_only": True,
        "device": "cpu",
        "dtype": "float64",
        "response_absolute_max": 10.0,
        "runtime_seconds": 1.0,
    }
    assert set(custody) == CUSTODY_KEYS
    return registration, custody


def test_custody_validation_is_exact_and_summary_is_write_once(tmp_path: Path) -> None:
    registration, custody = _custody_fixture()
    _validate_custody_summary(
        custody,
        registration=registration,
        expected_registration_sha256="a" * 64,
    )

    drift = copy.deepcopy(custody)
    drift["query_manifest_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="custody contract"):
        _validate_custody_summary(
            drift,
            registration=registration,
            expected_registration_sha256="a" * 64,
        )

    (tmp_path / "direct_backup_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        score_from_custody(
            output_directory=tmp_path,
            expected_registration_sha256="a" * 64,
            expected_acquisition_custody_sha256="b" * 64,
        )
