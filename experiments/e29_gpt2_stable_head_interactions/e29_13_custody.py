"""Write-once artifacts and strict hash custody for prospective E29.13."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_custody import (
    atomic_json_save,
    atomic_torch_save,
    load_verified_json,
    require_exact_keys,
    runtime_environment,
    sha256,
    verify_file_hash,
)
from experiments.e29_gpt2_stable_head_interactions.e29_13_absorbing_hub_fitting import (
    BRANCH_SHAPE_SCALES,
    METHOD_MANIFEST,
    PREDICTION_SPLITS,
    RIDGE,
)
from experiments.e29_gpt2_stable_head_interactions.e29_13_absorbing_hub_protocol import (
    BOOTSTRAP,
    COMPONENT_COUNT,
    CONTROL_COMPONENT,
    CONTROL_COMPONENT_SPEC,
    DIAGNOSTIC_MASK_SHAPES,
    HELD_MASK_SHAPES,
    HUB_COMPONENT,
    HUB_COMPONENT_SPEC,
    SATURATION_CONSTANTS,
    SCORING_GATES,
    SOURCE_INHERITED_SHAPES,
    SOURCE_STRATA,
    validate_absorbing_hub_protocol,
)

PARENT_E29_12_REGISTRATION_SHA256 = (
    "1f7661cd643d37d1abd8b785a170acc0699216420f77f0e248fbb12b776dc078"
)
PARENT_E29_12_RESULTS_SHA256 = (
    "d02848f4be09d61833234fe52a4c90f7013fe3d6b896d722d8405c216b34bfd9"
)
DEVELOPMENT_POSTHOC_SHA256 = (
    "fe354dad636617f4d37950cddd3565f89f5ee484789e525525c3d57b2e94e7b9"
)

SOURCE_FILES = (
    "e29_13_absorbing_hub_protocol.py",
    "e29_13_prompt_panel.py",
    "e29_13_absorbing_hub_fitting.py",
    "e29_13_custody.py",
    "register_e29_13_absorbing_hub.py",
    "acquire_e29_13_fit.py",
    "fit_e29_13_absorbing_hub.py",
    "acquire_e29_13_held.py",
    "score_e29_13_absorbing_hub.py",
)

MASK_ARTIFACT_SCHEMA = "e29_13_absorbing_hub_protocol_artifact_v1"
MASK_PANEL_ORDER = (
    "source_fit",
    *PREDICTION_SPLITS,
    "all_components",
    "all_minus_hub",
)
MASK_SHAPES = {
    "source_fit": (283, COMPONENT_COUNT),
    **HELD_MASK_SHAPES,
    **DIAGNOSTIC_MASK_SHAPES,
}

QUERY_SCHEDULE = {
    "fit_stage": {
        "single_combined_patched_call": True,
        "source_rows": 283,
        "all_components_c48_calibration_rows": 1,
        "all_minus_hub_quarantined_rows": 1,
        "total_patched_rows": 285,
        "patched_context_mask_cells": 3_420,
        "clean_forward_examples": 36,
    },
    "held_stage": {
        "joint_c24_c32_call": True,
        "confirmation_size24_rows": 48,
        "confirmation_size32_rows": 48,
        "total_patched_rows": 96,
        "patched_context_mask_cells": 1_152,
        "clean_forward_examples": 36,
    },
}

QUERY_BUDGET = {
    "contexts": 12,
    "source_fit_rows": 283,
    "held_rows": 96,
    "diagnostic_rows": 2,
    "total_mask_rows": 381,
    "fit_stage_patched_rows": 285,
    "held_stage_patched_rows": 96,
    "source_fit_patched_context_mask_cells": 3_396,
    "diagnostic_patched_context_mask_cells": 24,
    "fit_stage_patched_context_mask_cells": 3_420,
    "held_patched_context_mask_cells": 1_152,
    "total_patched_context_mask_cells": 4_572,
    "fit_clean_forward_examples": 36,
    "held_clean_forward_examples": 36,
    "total_clean_forward_examples": 72,
    "maximum_logical_queries": 4_644,
}

DIAGNOSTIC_ROLES = {
    "all_components": "registered_c48_fit_stage_scale_calibration",
    "all_minus_hub": "quarantined_report_only_never_fitter_or_gate_input",
}

OUTCOME_CUSTODY = {
    "registration_sha_required_for_fit_acquisition": True,
    "fit_acquisition_is_one_registered_285_row_call": True,
    "fit_acquisition_queries_only_source_plus_two_diagnostics": True,
    "fit_acquisition_custody_sha_required_for_fitting": True,
    "all_components_c48_is_registered_fitter_calibration": True,
    "all_minus_hub_acquired_with_fit_but_quarantined_from_fitter": True,
    "fitter_has_no_all_minus_hub_response_argument": True,
    "all_held_predictions_hashed_before_either_held_scale_response": True,
    "preparation_sha_required_for_joint_held_acquisition": True,
    "held_acquisition_queries_both_registered_scales_in_one_call": True,
    "held_acquisition_invokes_no_fitting_or_scoring_function": True,
    "held_acquisition_deserializes_no_raw_fit_or_source_response_artifact": True,
    "all_minus_hub_excluded_from_every_hard_gate": True,
    "every_acquisition_repeats_base_and_donor_clean_gates": True,
}

REGISTRATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "prospective_scope",
        "parent_e29_12_registration",
        "parent_e29_12_registration_sha256",
        "parent_e29_12_results",
        "parent_e29_12_results_sha256",
        "development_evidence",
        "prior_mask_artifacts",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "component_map",
        "patch_scope",
        "component_semantics",
        "protocol_note",
        "protocol_note_sha256",
        "protocol_masks",
        "protocol_masks_sha256",
        "protocol_masks_schema",
        "mask_panel_order",
        "mask_shapes",
        "source_strata",
        "prompt_checks",
        "prompt_panel_sha256",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
        "hub_component",
        "control_component",
        "saturation_constants",
        "method_manifest",
        "primary_method",
        "secondary_methods",
        "star_comparisons",
        "ridge",
        "branch_shape_scales",
        "prediction_splits",
        "bootstrap",
        "scoring_gates",
        "diagnostic_roles",
        "query_schedule",
        "query_budget",
        "numerical_gates",
        "outcome_custody",
        "source_code_sha256",
        "protocol_deviations",
    }
)


def _jsonable(value: Any) -> Any:
    """Mirror strict JSON's tuple-to-list conversion."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _collect_mask_tensors(value: Any) -> list[torch.Tensor]:
    """Collect binary, nonempty, two-dimensional 48-component panels."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().double()
        if (
            tensor.ndim == 2
            and tensor.shape[1] == COMPONENT_COUNT
            and len(tensor)
            and bool(torch.logical_or(tensor == 0.0, tensor == 1.0).all())
        ):
            return [tensor]
        return []
    if isinstance(value, Mapping):
        return [
            panel for item in value.values() for panel in _collect_mask_tensors(item)
        ]
    if isinstance(value, (tuple, list)):
        return [panel for item in value for panel in _collect_mask_tensors(item)]
    return []


def prior_mask_records(parent: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Return the exact E29.8--E29.12 mask-artifact provenance closure."""
    inherited = parent.get("prior_mask_artifacts")
    if not isinstance(inherited, Mapping):
        raise TypeError("E29.12 prior mask artifacts must be a mapping")
    require_exact_keys(
        inherited,
        {"e29_8", "e29_9", "e29_11"},
        label="E29.12 prior mask artifacts",
    )
    records: dict[str, dict[str, str]] = {}
    for key in ("e29_8", "e29_9", "e29_11"):
        record = inherited[key]
        require_exact_keys(record, {"path", "sha256"}, label=f"{key} mask record")
        records[key] = {
            "path": str(Path(record["path"]).resolve()),
            "sha256": str(record["sha256"]),
        }
    records["e29_12"] = {
        "path": str(Path(parent["protocol_masks"]).resolve()),
        "sha256": str(parent["protocol_masks_sha256"]),
    }
    return records


def load_prior_mask_closure(
    records: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Verify all prior mask files and recover inherited fit panels."""
    require_exact_keys(
        records,
        {"e29_8", "e29_9", "e29_11", "e29_12"},
        label="E29.13 prior mask artifacts",
    )
    prior: list[torch.Tensor] = []
    inherited_fit_masks: dict[str, torch.Tensor] | None = None
    for key in ("e29_8", "e29_9", "e29_11", "e29_12"):
        record = records[key]
        require_exact_keys(record, {"path", "sha256"}, label=f"{key} mask record")
        path = Path(record["path"])
        verify_file_hash(path, str(record["sha256"]))
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        panels = _collect_mask_tensors(artifact)
        if not panels:
            raise RuntimeError(f"{key} mask artifact contains no binary mask panel")
        prior.extend(panels)
        if key == "e29_12":
            masks = artifact.get("masks")
            if not isinstance(masks, Mapping):
                raise RuntimeError("E29.12 mask artifact is malformed")
            inherited_fit_masks = {}
            for split, shape in SOURCE_INHERITED_SHAPES.items():
                value = masks.get(split)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"E29.12 masks omit tensor split {split}")
                value = value.detach().cpu().double()
                if tuple(value.shape) != shape:
                    raise RuntimeError(f"E29.12 {split} shape changed")
                inherited_fit_masks[split] = value
    if inherited_fit_masks is None:
        raise RuntimeError("E29.12 inherited fit masks were not loaded")
    return inherited_fit_masks, prior


def load_registered_protocol(
    registration: Mapping[str, Any],
    *,
    inherited_fit_masks: Mapping[str, torch.Tensor],
    prior_masks: Sequence[torch.Tensor],
    expected_prior_mask_artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Load and purely validate the exact registered protocol artifact."""
    for key in ("protocol_masks", "protocol_masks_sha256", "protocol_masks_schema"):
        if key not in registration:
            raise RuntimeError(f"registration omits {key}")
    if registration["protocol_masks_schema"] != MASK_ARTIFACT_SCHEMA:
        raise RuntimeError("registered protocol-mask schema changed")
    path = Path(registration["protocol_masks"])
    verify_file_hash(path, str(registration["protocol_masks_sha256"]))
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "parent_e29_12_registration_sha256",
        "prior_mask_artifact_sha256",
        "method_manifest",
        "ridge",
        "branch_shape_scales",
        "prediction_splits",
        "mask_panel_order",
        "mask_shapes",
        "diagnostic_roles",
        "query_schedule",
        "protocol",
    }
    if not isinstance(artifact, Mapping):
        raise TypeError("E29.13 protocol artifact must be a mapping")
    require_exact_keys(artifact, required, label="E29.13 protocol artifact")
    if artifact["schema_version"] != MASK_ARTIFACT_SCHEMA:
        raise RuntimeError("unexpected E29.13 protocol artifact schema")
    if (
        artifact["parent_e29_12_registration_sha256"]
        != PARENT_E29_12_REGISTRATION_SHA256
    ):
        raise RuntimeError("E29.13 protocol artifact changed its parent")
    expected_prior_hashes = dict(expected_prior_mask_artifact_sha256)
    if dict(artifact["prior_mask_artifact_sha256"]) != expected_prior_hashes:
        raise RuntimeError("prior mask artifact provenance changed")
    if tuple(artifact["method_manifest"]) != METHOD_MANIFEST:
        raise RuntimeError("E29.13 method manifest changed")
    if artifact["ridge"] != RIDGE:
        raise RuntimeError("E29.13 ridge changed")
    if artifact["branch_shape_scales"] != BRANCH_SHAPE_SCALES:
        raise RuntimeError("E29.13 branch-shape scales changed")
    if tuple(artifact["prediction_splits"]) != PREDICTION_SPLITS:
        raise RuntimeError("E29.13 prediction splits changed")
    if tuple(artifact["mask_panel_order"]) != MASK_PANEL_ORDER:
        raise RuntimeError("E29.13 mask-panel order changed")
    if artifact["mask_shapes"] != {
        key: list(shape) for key, shape in MASK_SHAPES.items()
    }:
        raise RuntimeError("E29.13 mask shapes changed")
    if artifact["diagnostic_roles"] != DIAGNOSTIC_ROLES:
        raise RuntimeError("E29.13 diagnostic roles changed")
    if artifact["query_schedule"] != QUERY_SCHEDULE:
        raise RuntimeError("E29.13 query schedule changed")

    protocol = validate_absorbing_hub_protocol(
        artifact["protocol"],
        inherited_fit_masks=inherited_fit_masks,
        prior_masks=prior_masks,
    )
    if protocol["query_accounting"] != {
        key: QUERY_BUDGET[key]
        for key in (
            "contexts",
            "source_fit_rows",
            "held_rows",
            "diagnostic_rows",
            "total_mask_rows",
            "source_fit_patched_context_mask_cells",
            "held_patched_context_mask_cells",
            "diagnostic_patched_context_mask_cells",
            "total_patched_context_mask_cells",
        )
    }:
        raise RuntimeError("E29.13 protocol query accounting changed")
    return protocol


def load_protocol_for_verified_registration(
    registration: Mapping[str, Any],
    *,
    registration_path: Path | None = None,
    expected_registration_sha256: str | None = None,
) -> dict[str, Any]:
    """Reload the registered protocol and its complete prior-mask closure."""
    if (registration_path is None) != (expected_registration_sha256 is None):
        raise ValueError("registration path and expected SHA must be provided together")
    if registration_path is not None and expected_registration_sha256 is not None:
        sealed = load_verified_json(registration_path, expected_registration_sha256)
        if dict(registration) != sealed:
            raise RuntimeError("in-memory registration differs from its sealed file")

    from experiments.e29_gpt2_stable_head_interactions.e29_12_custody import (
        load_verified_registration as load_parent_registration,
    )

    parent = load_parent_registration(
        Path(registration["parent_e29_12_registration"]),
        expected_sha256=registration["parent_e29_12_registration_sha256"],
    )
    records = registration["prior_mask_artifacts"]
    expected_records = prior_mask_records(parent)
    if records != expected_records:
        raise RuntimeError("E29.13 prior mask artifact records changed")
    inherited_fit_masks, prior_masks = load_prior_mask_closure(records)
    return load_registered_protocol(
        registration,
        inherited_fit_masks=inherited_fit_masks,
        prior_masks=prior_masks,
        expected_prior_mask_artifact_sha256={
            key: record["sha256"] for key, record in records.items()
        },
    )


def _validate_development_evidence(registration: Mapping[str, Any]) -> None:
    development = registration["development_evidence"]
    require_exact_keys(
        development,
        {
            "absorbing_hub_posthoc",
            "absorbing_hub_posthoc_sha256",
            "schema_version",
            "status",
            "focal_component",
            "posthoc_choices_frozen_prospectively_here",
        },
        label="E29.13 development evidence",
    )
    verify_file_hash(
        Path(development["absorbing_hub_posthoc"]),
        development["absorbing_hub_posthoc_sha256"],
    )
    if development != {
        "absorbing_hub_posthoc": development["absorbing_hub_posthoc"],
        "absorbing_hub_posthoc_sha256": DEVELOPMENT_POSTHOC_SHA256,
        "schema_version": "e29_12_absorbing_hub_posthoc_v1",
        "status": "exploratory_posthoc_after_E29.12_completion_not_confirmatory",
        "focal_component": {"index": 1, "layer": 0, "kind": "mlp"},
        "posthoc_choices_frozen_prospectively_here": True,
    }:
        raise RuntimeError("E29.13 development-evidence contract changed")


def load_verified_registration(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Verify the E29.13 seal and every response-blind dependency."""
    registration = load_verified_json(path, expected_sha256)
    require_exact_keys(
        registration,
        REGISTRATION_REQUIRED_KEYS,
        label="E29.13 registration",
    )
    if registration["schema_version"] != "e29_13_absorbing_hub_registration_v1":
        raise RuntimeError("unexpected E29.13 registration schema")
    if registration["status"] != "sealed_before_any_e29_13_response":
        raise RuntimeError("E29.13 is not in its sealed state")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("registration contains a protocol deviation")
    if registration["environment"] != runtime_environment():
        raise RuntimeError("runtime environment differs from E29.13 seal")
    if registration["parent_e29_12_registration_sha256"] != (
        PARENT_E29_12_REGISTRATION_SHA256
    ):
        raise RuntimeError("E29.13 parent registration changed")
    if registration["parent_e29_12_results_sha256"] != PARENT_E29_12_RESULTS_SHA256:
        raise RuntimeError("E29.13 parent results changed")
    verify_file_hash(
        Path(registration["parent_e29_12_results"]),
        registration["parent_e29_12_results_sha256"],
    )
    _validate_development_evidence(registration)

    from experiments.e29_gpt2_stable_head_interactions.e29_12_custody import (
        load_verified_registration as load_parent_registration,
    )

    parent = load_parent_registration(
        Path(registration["parent_e29_12_registration"]),
        expected_sha256=registration["parent_e29_12_registration_sha256"],
    )
    for key in (
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "execution",
        "component_map",
        "patch_scope",
        "component_semantics",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
    ):
        if registration[key] != parent[key]:
            raise RuntimeError(f"E29.13 changed inherited parent field {key}")

    expected_science = {
        "protocol_masks_schema": MASK_ARTIFACT_SCHEMA,
        "mask_panel_order": list(MASK_PANEL_ORDER),
        "mask_shapes": {key: list(shape) for key, shape in MASK_SHAPES.items()},
        "source_strata": list(SOURCE_STRATA),
        "hub_component": {
            "index": HUB_COMPONENT,
            "spec": list(HUB_COMPONENT_SPEC),
        },
        "control_component": {
            "index": CONTROL_COMPONENT,
            "spec": list(CONTROL_COMPONENT_SPEC),
        },
        "saturation_constants": SATURATION_CONSTANTS,
        "method_manifest": list(METHOD_MANIFEST),
        "primary_method": "gate_shape_c48_calibrated",
        "secondary_methods": [
            "gate_shape_clean_proxy_calibrated",
            "gate_only_c48_calibrated",
        ],
        "star_comparisons": {
            "methods": ["analytic_hub_star_c48", "analytic_control_star_c48"],
            "role": "secondary_report_only_no_hard_pass",
        },
        "ridge": RIDGE,
        "branch_shape_scales": _jsonable(BRANCH_SHAPE_SCALES),
        "prediction_splits": list(PREDICTION_SPLITS),
        "bootstrap": BOOTSTRAP,
        "scoring_gates": _jsonable(SCORING_GATES),
        "diagnostic_roles": DIAGNOSTIC_ROLES,
        "query_schedule": QUERY_SCHEDULE,
        "query_budget": QUERY_BUDGET,
        "numerical_gates": NUMERICAL_GATES,
        "outcome_custody": OUTCOME_CUSTODY,
    }
    for key, expected in expected_science.items():
        if registration[key] != expected:
            raise RuntimeError(f"E29.13 registered field changed: {key}")
    if len(registration["prompt_checks"]) != 12:
        raise RuntimeError("E29.13 requires exactly twelve prompt checks")

    verify_file_hash(
        Path(registration["protocol_note"]), registration["protocol_note_sha256"]
    )
    require_exact_keys(
        registration["source_code_sha256"],
        SOURCE_FILES,
        label="E29.13 source hash manifest",
    )
    source_directory = Path(__file__).resolve().parent
    for filename in SOURCE_FILES:
        verify_file_hash(
            source_directory / filename,
            registration["source_code_sha256"][filename],
        )
    load_protocol_for_verified_registration(registration)
    return registration


__all__ = [
    "DEVELOPMENT_POSTHOC_SHA256",
    "DIAGNOSTIC_ROLES",
    "MASK_ARTIFACT_SCHEMA",
    "MASK_PANEL_ORDER",
    "MASK_SHAPES",
    "OUTCOME_CUSTODY",
    "PARENT_E29_12_REGISTRATION_SHA256",
    "PARENT_E29_12_RESULTS_SHA256",
    "QUERY_BUDGET",
    "QUERY_SCHEDULE",
    "REGISTRATION_REQUIRED_KEYS",
    "SOURCE_FILES",
    "atomic_json_save",
    "atomic_torch_save",
    "load_prior_mask_closure",
    "load_protocol_for_verified_registration",
    "load_registered_protocol",
    "load_verified_json",
    "load_verified_registration",
    "prior_mask_records",
    "require_exact_keys",
    "runtime_environment",
    "sha256",
    "verify_file_hash",
]
