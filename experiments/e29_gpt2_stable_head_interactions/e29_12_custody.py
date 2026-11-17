"""Write-once artifacts and strict hash custody for prospective E29.12."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    ATTENTION_IMPLEMENTATION,
    BATCH_SIZE,
    CHECKPOINT_FILES,
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_custody import (
    atomic_json_save,
    atomic_torch_save,
    load_verified_json,
    runtime_environment,
    sha256,
    verify_file_hash,
)
from experiments.e29_gpt2_stable_head_interactions.e29_12_endpoint_protocol import (
    validate_endpoint_masks,
)
from experiments.e29_gpt2_stable_head_interactions.e29_12_fitting import (
    DEVELOPMENT_SCALE_RIDGES,
    FIT_SHAPES,
    FIT_SPLITS,
    METHOD_MANIFEST,
    PREDICTION_SHAPES,
    PREDICTION_SPLITS,
)

SOURCE_FILES = (
    "e29_12_endpoint_protocol.py",
    "e29_12_prompt_panel.py",
    "e29_12_development_controls.py",
    "e29_12_fitting.py",
    "e29_12_custody.py",
    "register_e29_12_endpoint_boundary.py",
    "acquire_e29_12_fit.py",
    "fit_e29_12_endpoint_models.py",
    "acquire_e29_12_held.py",
    "score_e29_12_endpoint_confirmation.py",
)

MASK_SPLITS = (*FIT_SPLITS, *PREDICTION_SPLITS)
MASK_SHAPES = {**FIT_SHAPES, **PREDICTION_SHAPES}

QUERY_BUDGET = {
    "model_fit_rows": 192,
    "endpoint_diagnostic_rows": 1,
    "fit_acquired_rows_including_diagnostic": 193,
    "held_rows": 128,
    "contexts": 12,
    "fit_patched_context_mask_cells": 2_316,
    "held_patched_context_mask_cells": 1_536,
    "total_patched_context_mask_cells": 3_852,
    "fit_clean_forward_examples": 36,
    "held_clean_forward_examples": 36,
    "total_clean_forward_examples": 72,
    "maximum_logical_queries": 3_924,
}

OUTCOME_CUSTODY = {
    "registration_sha_required_for_fit_acquisition": True,
    "fit_acquisition_queries_only_registered_fit_masks_plus_all48_diagnostic": True,
    "fit_custody_sha_required_for_prediction_fitting": True,
    "all_eight_method_predictions_hashed_before_either_held_scale": True,
    "preparation_sha_required_for_joint_held_acquisition": True,
    "fit_label_artifact_not_loaded_by_held_acquisition": True,
    "every_acquisition_repeats_base_and_donor_clean_gates": True,
    "both_held_scales_acquired_in_one_concatenated_panel": True,
}

REGISTRATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "parent_e29_11_registration",
        "parent_e29_11_registration_sha256",
        "development_evidence",
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
        "prior_mask_artifacts",
        "mask_split_order",
        "mask_shapes",
        "prompt_checks",
        "prompt_panel_sha256",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
        "fit_splits",
        "prediction_splits",
        "method_manifest",
        "query_budget",
        "numerical_gates",
        "outcome_custody",
        "source_code_sha256",
        "protocol_deviations",
    }
)


def require_keys(
    record: Mapping[str, Any], required: Collection[str], *, label: str
) -> None:
    """Require a closed scientific core while tolerating provenance additions."""
    missing = set(required) - set(record)
    if missing:
        raise RuntimeError(f"{label} keys changed; missing={sorted(missing)}")


def _load_parent_masks(parent: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    path = Path(parent["protocol_masks"])
    verify_file_hash(path, parent["protocol_masks_sha256"])
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    masks = artifact.get("masks")
    if not isinstance(masks, Mapping):
        raise TypeError("parent E29.11 mask artifact is malformed")
    return {str(key): value.detach().cpu().double() for key, value in masks.items()}


def _collect_mask_tensors(value: Any) -> list[torch.Tensor]:
    """Recursively collect binary two-dimensional 48-column mask panels."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().double()
        if (
            tensor.ndim == 2
            and tensor.shape[1] == 48
            and len(tensor)
            and bool(torch.logical_or(tensor == 0.0, tensor == 1.0).all())
        ):
            return [tensor]
        return []
    if isinstance(value, Mapping):
        return [panel for item in value.values() for panel in _collect_mask_tensors(item)]
    if isinstance(value, (tuple, list)):
        return [panel for item in value for panel in _collect_mask_tensors(item)]
    return []


def load_prior_mask_closure(
    registration: Mapping[str, Any], parent: Mapping[str, Any]
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Verify and collect every earlier registered 48-component mask panel."""
    records = registration["prior_mask_artifacts"]
    if not isinstance(records, Mapping):
        raise TypeError("prior mask artifacts must be a mapping")
    require_keys(records, {"e29_8", "e29_9", "e29_11"}, label="prior masks")
    closure: list[torch.Tensor] = []
    parent_masks: dict[str, torch.Tensor] | None = None
    for name in ("e29_8", "e29_9", "e29_11"):
        record = records[name]
        require_keys(record, {"path", "sha256"}, label=f"{name} prior mask")
        artifact_path = Path(record["path"])
        verify_file_hash(artifact_path, record["sha256"])
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        panels = _collect_mask_tensors(artifact)
        if not panels:
            raise RuntimeError(f"{name} prior mask artifact contains no mask panel")
        closure.extend(panels)
        if name == "e29_11":
            if (
                artifact_path.resolve() != Path(parent["protocol_masks"]).resolve()
                or record["sha256"] != parent["protocol_masks_sha256"]
            ):
                raise RuntimeError("E29.12 identifies another E29.11 mask artifact")
            parent_masks = _load_parent_masks(parent)
    if parent_masks is None:
        raise RuntimeError("E29.11 parent masks were not loaded")
    return parent_masks, closure


def load_registered_masks(
    registration: Mapping[str, Any],
    *,
    parent_fit_masks: Mapping[str, torch.Tensor],
    prior_masks: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Load the sealed panel and reject mask, order, or parent-reuse drift."""
    if tuple(registration["mask_split_order"]) != MASK_SPLITS:
        raise RuntimeError("registered mask split order changed")
    expected_shapes = {key: list(shape) for key, shape in MASK_SHAPES.items()}
    if registration["mask_shapes"] != expected_shapes:
        raise RuntimeError("registered mask shapes changed")

    path = Path(registration["protocol_masks"])
    verify_file_hash(path, str(registration["protocol_masks_sha256"]))
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    require_keys(
        artifact,
        {"schema_version", "split_order", "masks"},
        label="E29.12 mask artifact",
    )
    if artifact["schema_version"] != "e29_12_endpoint_boundary_masks_v1":
        raise RuntimeError("unexpected E29.12 mask-artifact schema")
    if tuple(artifact["split_order"]) != MASK_SPLITS:
        raise RuntimeError("E29.12 mask-artifact split order changed")
    masks = artifact["masks"]
    if not isinstance(masks, Mapping) or tuple(masks) != MASK_SPLITS:
        raise RuntimeError("E29.12 mask splits changed")

    validated: dict[str, torch.Tensor] = {}
    for split in MASK_SPLITS:
        value = masks[split]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{split} masks are not a tensor")
        value = value.detach().cpu().double()
        if tuple(value.shape) != MASK_SHAPES[split]:
            raise RuntimeError(f"unexpected {split} mask shape")
        if not bool(torch.logical_or(value == 0.0, value == 1.0).all()):
            raise RuntimeError(f"{split} masks are not binary")
        validated[split] = value

    if tuple(parent_fit_masks) != FIT_SPLITS:
        raise RuntimeError("parent fit-mask keys changed")
    for split in FIT_SPLITS:
        if not torch.equal(
            validated[split], parent_fit_masks[split].detach().cpu().double()
        ):
            raise RuntimeError(f"E29.12 does not exactly reuse parent {split} masks")

    endpoint_masks = {split: validated[split] for split in PREDICTION_SPLITS}
    prior_closure = [
        *(value.detach().cpu().double() for value in parent_fit_masks.values()),
        *(value.detach().cpu().double() for value in prior_masks),
    ]
    try:
        validate_endpoint_masks(endpoint_masks, prior_masks=prior_closure)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    prior_rows: dict[int, set[tuple[int, ...]]] = {}
    for panel in prior_closure:
        for row in panel:
            indices = tuple(
                int(index)
                for index in torch.nonzero(row, as_tuple=False).flatten().tolist()
            )
            prior_rows.setdefault(len(indices), set()).add(indices)
    for split, complement_cardinality in (
        ("confirmation_size24", 24),
        ("confirmation_size32", 16),
    ):
        for row in endpoint_masks[split]:
            selected = {
                int(index)
                for index in torch.nonzero(row, as_tuple=False).flatten().tolist()
            }
            complement = tuple(
                component for component in range(48) if component not in selected
            )
            if complement in prior_rows.get(complement_cardinality, set()):
                raise RuntimeError(
                    f"{split} contains the complement of a prior mask"
                )
    explicit = registration.get("explicit_masks")
    if explicit is not None:
        if not isinstance(explicit, Mapping) or tuple(explicit) != MASK_SPLITS:
            raise RuntimeError("registered explicit mask keys changed")
        for split in MASK_SPLITS:
            if validated[split].tolist() != explicit[split]:
                raise RuntimeError(f"{split} masks differ from explicit registration")
    return validated


def load_masks_for_verified_registration(
    registration: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Reload masks after ``load_verified_registration`` established custody."""
    parent = load_verified_json(
        Path(registration["parent_e29_11_registration"]),
        registration["parent_e29_11_registration_sha256"],
    )
    parent_masks, prior_masks = load_prior_mask_closure(registration, parent)
    return load_registered_masks(
        registration,
        parent_fit_masks={split: parent_masks[split] for split in FIT_SPLITS},
        prior_masks=prior_masks,
    )


def load_verified_registration(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Verify the E29.12 seal and every dependency needed for acquisition."""
    registration = load_verified_json(path, expected_sha256)
    require_keys(
        registration,
        REGISTRATION_REQUIRED_KEYS,
        label="E29.12 registration",
    )
    if registration["schema_version"] != "e29_12_endpoint_boundary_registration_v1":
        raise RuntimeError("unexpected E29.12 registration schema")
    if registration["status"] != "sealed_before_any_e29_12_patched_outcome":
        raise RuntimeError("E29.12 is not in its sealed state")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("registration contains a protocol deviation")
    if registration["environment"] != runtime_environment():
        raise RuntimeError("runtime environment differs from E29.12 seal")

    expected_execution = {
        "device": "cpu",
        "model_dtype": "float64",
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "batch_size": BATCH_SIZE,
        "torch_num_threads_during_fitting": 1,
    }
    if registration["execution"] != expected_execution:
        raise RuntimeError("E29.12 execution recipe changed")
    if registration["numerical_gates"] != NUMERICAL_GATES:
        raise RuntimeError("E29.12 numerical gates changed")
    if tuple(registration["fit_splits"]) != FIT_SPLITS:
        raise RuntimeError("E29.12 fit splits changed")
    if tuple(registration["prediction_splits"]) != PREDICTION_SPLITS:
        raise RuntimeError("E29.12 prediction splits changed")
    if tuple(registration["method_manifest"]) != METHOD_MANIFEST:
        raise RuntimeError("E29.12 method manifest changed")
    if registration["query_budget"] != QUERY_BUDGET:
        raise RuntimeError("E29.12 query budget changed")
    if registration["outcome_custody"] != OUTCOME_CUSTODY:
        raise RuntimeError("E29.12 outcome custody changed")
    if len(registration["prompt_checks"]) != 12:
        raise RuntimeError("E29.12 requires exactly twelve prompt checks")

    from experiments.e29_gpt2_stable_head_interactions.e29_11_custody import (
        load_verified_registration as load_parent_registration,
    )

    parent = load_parent_registration(
        Path(registration["parent_e29_11_registration"]),
        expected_sha256=registration["parent_e29_11_registration_sha256"],
    )
    for key in (
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "component_map",
        "patch_scope",
        "component_semantics",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
    ):
        if registration[key] != parent[key]:
            raise RuntimeError(f"E29.12 changed parent field {key}")

    require_keys(
        registration["checkpoint_sha256"],
        CHECKPOINT_FILES,
        label="E29.12 checkpoint hash manifest",
    )
    snapshot = Path(registration["model_snapshot"])
    for filename in CHECKPOINT_FILES:
        verify_file_hash(snapshot / filename, registration["checkpoint_sha256"][filename])
    verify_file_hash(
        Path(registration["e29_8_feature_artifact"]),
        registration["e29_8_feature_artifact_sha256"],
    )
    verify_file_hash(
        Path(registration["protocol_note"]),
        registration["protocol_note_sha256"],
    )
    development = registration["development_evidence"]
    require_keys(
        development,
        {
            "parent_results",
            "parent_results_sha256",
            "additive_shape_grid",
            "additive_shape_grid_sha256",
            "additive_shape_grid_selected_ridges",
            "posthoc_choices_are_frozen_prospectively_here",
        },
        label="E29.12 development evidence",
    )
    verify_file_hash(
        Path(development["parent_results"]), development["parent_results_sha256"]
    )
    verify_file_hash(
        Path(development["additive_shape_grid"]),
        development["additive_shape_grid_sha256"],
    )
    if (
        development["additive_shape_grid_selected_ridges"]
        != DEVELOPMENT_SCALE_RIDGES
        or development["posthoc_choices_are_frozen_prospectively_here"] is not True
    ):
        raise RuntimeError("E29.12 development-control provenance changed")

    source_hashes = registration["source_code_sha256"]
    require_keys(source_hashes, SOURCE_FILES, label="E29.12 source hash manifest")
    source_directory = Path(__file__).resolve().parent
    for filename in SOURCE_FILES:
        verify_file_hash(source_directory / filename, source_hashes[filename])

    parent_masks, prior_masks = load_prior_mask_closure(registration, parent)
    parent_fit_masks = {split: parent_masks[split] for split in FIT_SPLITS}
    load_registered_masks(
        registration,
        parent_fit_masks=parent_fit_masks,
        prior_masks=prior_masks,
    )
    return registration


__all__ = [
    "MASK_SHAPES",
    "MASK_SPLITS",
    "OUTCOME_CUSTODY",
    "QUERY_BUDGET",
    "REGISTRATION_REQUIRED_KEYS",
    "SOURCE_FILES",
    "atomic_json_save",
    "atomic_torch_save",
    "load_masks_for_verified_registration",
    "load_prior_mask_closure",
    "load_registered_masks",
    "load_verified_json",
    "load_verified_registration",
    "require_keys",
    "sha256",
    "verify_file_hash",
]
