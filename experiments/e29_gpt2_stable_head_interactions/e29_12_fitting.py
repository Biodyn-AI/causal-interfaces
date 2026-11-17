"""Pure endpoint-aware fitting and response-blind prediction for E29.12.

The public API accepts labels only for the three fitting panels.  Both held
panels enter as masks alone, so all eight methods and their predictions can be
frozen before either held response table is acquired.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_9_transfer_fitting import (
    _feature_state,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_fitting import (
    apply_affine_readout,
    fit_fixed_affine_readout,
)

COMPONENT_COUNT = 48
OUTPUT_COUNT = 12
RIDGE = 10.0
DEVELOPMENT_EXPONENT = 2.4740
ADDITIVE_RIDGE_GRID = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0)
DEVELOPMENT_SCALE_RIDGES = {
    "confirmation_size24": 100.0,
    "confirmation_size32": 1_000.0,
}

FIT_SPLITS = ("calibration_low", "calibration_size8", "bridge_size16")
PREDICTION_SPLITS = ("confirmation_size24", "confirmation_size32")
FIT_SHAPES = {
    "calibration_low": (96, COMPONENT_COUNT),
    "calibration_size8": (32, COMPONENT_COUNT),
    "bridge_size16": (64, COMPONENT_COUNT),
}
PREDICTION_SHAPES = {
    "confirmation_size24": (64, COMPONENT_COUNT),
    "confirmation_size32": (64, COMPONENT_COUNT),
}
HELD_CARDINALITIES = {
    "confirmation_size24": 24,
    "confirmation_size32": 32,
}
METHOD_MANIFEST = (
    "mlp7_unbounded_ridge10",
    "additive_unbounded_fitcv",
    "mlp7_endpoint_cap_ridge10",
    "additive_endpoint_cap_fitcv",
    "mlp7_endpoint_reanchored_p2_4740_ridge10",
    "additive_endpoint_reanchored_p2_4740_fitcv",
    "density_conditioned_additive_unbounded_fitcv",
    "additive_shape_dev_scale_ridge",
)


def _as_finite_matrix(
    value: Any, *, name: str, expected_shape: tuple[int, int]
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    matrix = value.detach().cpu().double()
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not bool(matrix.isfinite().all()):
        raise ValueError(f"{name} must be finite")
    return matrix


def _validated_mlp7_spec(source_artifact: Mapping[str, Any]) -> dict[str, Any]:
    if source_artifact.get("format_version") != 1:
        raise ValueError("unsupported E29.8 source artifact")
    if source_artifact.get("component_count") != COMPONENT_COUNT:
        raise ValueError("E29.12 requires the frozen 48-component artifact")
    try:
        state = source_artifact["methods"]["mlp_hidden7"]["state"]
        input_weight = state["input.weight"]
        input_bias = state["input.bias"]
    except (KeyError, TypeError) as error:
        raise ValueError("source artifact omits the frozen MLP7 state") from error
    input_weight = _as_finite_matrix(
        input_weight, name="MLP7 input weight", expected_shape=(7, COMPONENT_COUNT)
    )
    if not isinstance(input_bias, torch.Tensor):
        raise TypeError("MLP7 input bias must be a tensor")
    input_bias = input_bias.detach().cpu().double()
    if tuple(input_bias.shape) != (7,):
        raise ValueError("MLP7 input bias must have shape (7,)")
    if not bool(input_bias.isfinite().all()):
        raise ValueError("MLP7 input bias must be finite")
    return {
        "family": "mlp7_features",
        # _feature_state intentionally follows the exact E29.11 float32 frozen
        # feature computation before returning a float64 design matrix.
        "input_weight": input_weight,
        "input_bias": input_bias,
        "permutation": torch.arange(COMPONENT_COUNT),
        "source": "frozen_e29_8_mlp_hidden7",
    }


def _validate_inputs(
    source_artifact: Mapping[str, Any],
    fit_masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
    held_masks: Mapping[str, torch.Tensor],
    clean_endpoints: torch.Tensor,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    mlp7_spec = _validated_mlp7_spec(source_artifact)
    if tuple(fit_masks) != FIT_SPLITS:
        raise ValueError(f"fit mask keys and order must be {FIT_SPLITS}")
    if tuple(fit_responses) != FIT_SPLITS:
        raise ValueError(f"fit response keys and order must be {FIT_SPLITS}")
    if tuple(held_masks) != PREDICTION_SPLITS:
        raise ValueError(f"held mask keys and order must be {PREDICTION_SPLITS}")

    clean_fit_masks: dict[str, torch.Tensor] = {}
    clean_fit_responses: dict[str, torch.Tensor] = {}
    for split in FIT_SPLITS:
        masks = _as_finite_matrix(
            fit_masks[split], name=f"{split} masks", expected_shape=FIT_SHAPES[split]
        )
        if not bool(torch.logical_or(masks == 0.0, masks == 1.0).all()):
            raise ValueError(f"{split} masks must be binary")
        responses = _as_finite_matrix(
            fit_responses[split],
            name=f"{split} responses",
            expected_shape=(FIT_SHAPES[split][0], OUTPUT_COUNT),
        )
        clean_fit_masks[split] = masks
        clean_fit_responses[split] = responses

    clean_held_masks: dict[str, torch.Tensor] = {}
    for split in PREDICTION_SPLITS:
        masks = _as_finite_matrix(
            held_masks[split],
            name=f"{split} masks",
            expected_shape=PREDICTION_SHAPES[split],
        )
        if not bool(torch.logical_or(masks == 0.0, masks == 1.0).all()):
            raise ValueError(f"{split} masks must be binary")
        cardinality = HELD_CARDINALITIES[split]
        if not bool((masks.sum(dim=1) == cardinality).all()):
            raise ValueError(f"{split} masks must select {cardinality} components")
        clean_held_masks[split] = masks

    if not isinstance(clean_endpoints, torch.Tensor):
        raise TypeError("clean endpoints must be a tensor")
    endpoints = clean_endpoints.detach().cpu().double()
    if tuple(endpoints.shape) != (OUTPUT_COUNT,):
        raise ValueError(f"clean endpoints must have shape ({OUTPUT_COUNT},)")
    if not bool(endpoints.isfinite().all()):
        raise ValueError("clean endpoints must be finite")
    if not bool((endpoints > 0.0).all()):
        raise ValueError("clean endpoints must be strictly positive")
    return (
        mlp7_spec,
        clean_fit_masks,
        clean_fit_responses,
        clean_held_masks,
        endpoints,
    )


def _fit_unbounded_method(
    *,
    family: str,
    feature_spec: Mapping[str, Any] | None,
    fit_masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
    held_masks: Mapping[str, torch.Tensor],
    ridge: float,
    fit_only_ridge_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def features(masks: torch.Tensor) -> torch.Tensor:
        if feature_spec is None:
            return masks
        return _feature_state(feature_spec, masks)

    readout = fit_fixed_affine_readout(
        torch.cat([features(fit_masks[split]) for split in FIT_SPLITS]),
        torch.cat([fit_responses[split] for split in FIT_SPLITS]),
        ridge=ridge,
    )
    if fit_only_ridge_selection is not None:
        readout["ridge_selected_from_current_outcomes"] = True
    record: dict[str, Any] = {
        "family": family,
        "readout": readout,
        "fit_splits": FIT_SPLITS,
        "fit_rows": sum(FIT_SHAPES[split][0] for split in FIT_SPLITS),
        "predictions": {
            split: apply_affine_readout(readout, features(held_masks[split]))
            for split in PREDICTION_SPLITS
        },
    }
    if feature_spec is not None:
        record["feature_spec"] = feature_spec
    if fit_only_ridge_selection is not None:
        record["fit_only_ridge_selection"] = dict(fit_only_ridge_selection)
    return record


def _select_additive_ridge(
    fit_masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
    clean_endpoints: torch.Tensor,
) -> dict[str, Any]:
    """Select one shared additive ridge without accessing either held panel.

    Low-cardinality and cardinality-8 rows train each candidate. The already
    designated cardinality-16 bridge is forward validation. Because the hard
    endpoint cap is the primary additive baseline, its capped c16 MSE is the
    registered selection objective.
    """
    training_splits = ("calibration_low", "calibration_size8")
    validation_split = "bridge_size16"
    training_masks = torch.cat([fit_masks[split] for split in training_splits])
    training_responses = torch.cat(
        [fit_responses[split] for split in training_splits]
    )
    validation_masks = fit_masks[validation_split]
    validation_responses = fit_responses[validation_split]
    candidates: list[dict[str, float]] = []
    for ridge in ADDITIVE_RIDGE_GRID:
        readout = fit_fixed_affine_readout(
            training_masks,
            training_responses,
            ridge=ridge,
        )
        predictions = apply_affine_readout(readout, validation_masks)
        capped = torch.minimum(predictions, clean_endpoints[None, :])
        candidates.append(
            {
                "ridge": ridge,
                "validation_capped_mse": float(
                    (capped - validation_responses).square().mean()
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["validation_capped_mse"],
            candidate["ridge"],
        ),
    )
    return {
        "protocol": "low_plus_c8_train_c16_forward_validation",
        "training_splits": training_splits,
        "validation_split": validation_split,
        "objective": "pooled_endpoint_capped_mse_over_64_by_12_cells",
        "tie_break": "smallest_ridge",
        "candidates": candidates,
        "selected_ridge": selected["ridge"],
    }


def _density_conditioned_additive_features(masks: torch.Tensor) -> torch.Tensor:
    """Allow each additive component effect to vary linearly with mask density."""
    masks = masks.detach().cpu().double()
    density = masks.sum(dim=1, keepdim=True) / COMPONENT_COUNT
    return torch.cat((masks, density * masks), dim=1)


def _select_density_additive_ridge(
    fit_masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Select the scale-conditioned control on held-forward centered shape."""
    training_splits = ("calibration_low", "calibration_size8")
    validation_split = "bridge_size16"
    training_features = torch.cat(
        [
            _density_conditioned_additive_features(fit_masks[split])
            for split in training_splits
        ]
    )
    training_responses = torch.cat(
        [fit_responses[split] for split in training_splits]
    )
    validation_features = _density_conditioned_additive_features(
        fit_masks[validation_split]
    )
    validation_responses = fit_responses[validation_split]
    centered_validation = validation_responses - validation_responses.mean(
        dim=0, keepdim=True
    )
    candidates: list[dict[str, float]] = []
    for ridge in ADDITIVE_RIDGE_GRID:
        readout = fit_fixed_affine_readout(
            training_features,
            training_responses,
            ridge=ridge,
        )
        predictions = apply_affine_readout(readout, validation_features)
        centered_predictions = predictions - predictions.mean(dim=0, keepdim=True)
        candidates.append(
            {
                "ridge": ridge,
                "validation_centered_shape_mse": float(
                    (centered_predictions - centered_validation).square().mean()
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["validation_centered_shape_mse"],
            candidate["ridge"],
        ),
    )
    return {
        "protocol": "low_plus_c8_train_c16_forward_validation",
        "training_splits": training_splits,
        "validation_split": validation_split,
        "objective": (
            "pooled_within_context_centered_shape_mse_over_64_by_12_cells"
        ),
        "tie_break": "smallest_ridge",
        "candidates": candidates,
        "selected_ridge": selected["ridge"],
    }


def _fit_density_conditioned_additive(
    *,
    fit_masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
    held_masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    selection = _select_density_additive_ridge(fit_masks, fit_responses)
    readout = fit_fixed_affine_readout(
        torch.cat(
            [_density_conditioned_additive_features(fit_masks[split]) for split in FIT_SPLITS]
        ),
        torch.cat([fit_responses[split] for split in FIT_SPLITS]),
        ridge=float(selection["selected_ridge"]),
    )
    readout["ridge_selected_from_current_outcomes"] = True
    return {
        "family": "density_conditioned_additive_masks_with_fixed_affine_readout",
        "feature_definition": "concat(mask, (cardinality/48)*mask)",
        "readout": readout,
        "fit_only_ridge_selection": selection,
        "fit_splits": FIT_SPLITS,
        "fit_rows": sum(FIT_SHAPES[split][0] for split in FIT_SPLITS),
        "predictions": {
            split: apply_affine_readout(
                readout,
                _density_conditioned_additive_features(held_masks[split]),
            )
            for split in PREDICTION_SPLITS
        },
    }


def _fit_development_scale_additive(
    *,
    fit_masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
    held_masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Fit the strongest scale-specific plain-additive E29.11 shape control."""
    fit_features = torch.cat([fit_masks[split] for split in FIT_SPLITS])
    fit_targets = torch.cat([fit_responses[split] for split in FIT_SPLITS])
    readouts = {
        split: fit_fixed_affine_readout(
            fit_features,
            fit_targets,
            ridge=DEVELOPMENT_SCALE_RIDGES[split],
        )
        for split in PREDICTION_SPLITS
    }
    return {
        "family": "additive_masks_with_development_scale_specific_ridge",
        "ridge_source": "frozen_e29_11_centered_shape_development",
        "ridge_grid_searched_in_development": ADDITIVE_RIDGE_GRID,
        "readouts_by_split": readouts,
        "fit_splits": FIT_SPLITS,
        "fit_rows": sum(FIT_SHAPES[split][0] for split in FIT_SPLITS),
        "predictions": {
            split: apply_affine_readout(readouts[split], held_masks[split])
            for split in PREDICTION_SPLITS
        },
    }


def _endpoint_cap_method(
    parent_name: str,
    parent: Mapping[str, Any],
    clean_endpoints: torch.Tensor,
) -> dict[str, Any]:
    return {
        "family": "elementwise_clean_endpoint_upper_cap",
        "parent_method": parent_name,
        "clean_endpoints": clean_endpoints.clone(),
        "predictions": {
            split: torch.minimum(
                parent["predictions"][split], clean_endpoints[None, :]
            )
            for split in PREDICTION_SPLITS
        },
    }


def _endpoint_reanchored_method(
    parent_name: str,
    parent: Mapping[str, Any],
    clean_endpoints: torch.Tensor,
) -> dict[str, Any]:
    predictions: dict[str, torch.Tensor] = {}
    target_means: dict[str, torch.Tensor] = {}
    for split in PREDICTION_SPLITS:
        unbounded = parent["predictions"][split]
        cardinality = HELD_CARDINALITIES[split]
        target_mean = clean_endpoints * (
            1.0
            - (1.0 - cardinality / COMPONENT_COUNT) ** DEVELOPMENT_EXPONENT
        )
        predictions[split] = (
            unbounded - unbounded.mean(dim=0, keepdim=True) + target_mean[None, :]
        )
        target_means[split] = target_mean
    return {
        "family": "smooth_clean_endpoint_reanchoring",
        "parent_method": parent_name,
        "development_exponent": DEVELOPMENT_EXPONENT,
        "target_panel_means": target_means,
        "predictions": predictions,
    }


def fit_endpoint_aware_models(
    source_artifact: Mapping[str, Any],
    fit_masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
    held_masks: Mapping[str, torch.Tensor],
    clean_endpoints: torch.Tensor,
) -> dict[str, Any]:
    """Fit the registered readouts and freeze all eight held predictions."""
    (
        mlp7_spec,
        fit_masks,
        fit_responses,
        held_masks,
        clean_endpoints,
    ) = _validate_inputs(
        source_artifact,
        fit_masks,
        fit_responses,
        held_masks,
        clean_endpoints,
    )
    mlp_name = "mlp7_unbounded_ridge10"
    additive_name = "additive_unbounded_fitcv"
    additive_ridge_selection = _select_additive_ridge(
        fit_masks,
        fit_responses,
        clean_endpoints,
    )
    mlp = _fit_unbounded_method(
        family="frozen_mlp7_features_with_fixed_affine_readout",
        feature_spec=mlp7_spec,
        fit_masks=fit_masks,
        fit_responses=fit_responses,
        held_masks=held_masks,
        ridge=RIDGE,
    )
    additive = _fit_unbounded_method(
        family="additive_masks_with_fixed_affine_readout",
        feature_spec=None,
        fit_masks=fit_masks,
        fit_responses=fit_responses,
        held_masks=held_masks,
        ridge=float(additive_ridge_selection["selected_ridge"]),
        fit_only_ridge_selection=additive_ridge_selection,
    )
    density_additive = _fit_density_conditioned_additive(
        fit_masks=fit_masks,
        fit_responses=fit_responses,
        held_masks=held_masks,
    )
    development_scale_additive = _fit_development_scale_additive(
        fit_masks=fit_masks,
        fit_responses=fit_responses,
        held_masks=held_masks,
    )
    methods = {
        mlp_name: mlp,
        additive_name: additive,
        "mlp7_endpoint_cap_ridge10": _endpoint_cap_method(
            mlp_name, mlp, clean_endpoints
        ),
        "additive_endpoint_cap_fitcv": _endpoint_cap_method(
            additive_name, additive, clean_endpoints
        ),
        "mlp7_endpoint_reanchored_p2_4740_ridge10": (
            _endpoint_reanchored_method(mlp_name, mlp, clean_endpoints)
        ),
        "additive_endpoint_reanchored_p2_4740_fitcv": (
            _endpoint_reanchored_method(additive_name, additive, clean_endpoints)
        ),
        "density_conditioned_additive_unbounded_fitcv": density_additive,
        "additive_shape_dev_scale_ridge": development_scale_additive,
    }
    if tuple(methods) != METHOD_MANIFEST:
        raise RuntimeError("E29.12 method order differs from the frozen manifest")
    for method in methods.values():
        for split in PREDICTION_SPLITS:
            prediction = method["predictions"][split]
            expected_shape = (PREDICTION_SHAPES[split][0], OUTPUT_COUNT)
            if tuple(prediction.shape) != expected_shape:
                raise RuntimeError(f"{split} prediction has the wrong shape")
            if not bool(prediction.isfinite().all()):
                raise RuntimeError(f"{split} prediction is non-finite")
    return {
        "schema_version": "e29_12_endpoint_aware_predictions_v1",
        "status": "all_eight_methods_frozen_before_held_labels",
        "component_count": COMPONENT_COUNT,
        "output_count": OUTPUT_COUNT,
        "ridge": RIDGE,
        "additive_ridge_grid": ADDITIVE_RIDGE_GRID,
        "development_exponent": DEVELOPMENT_EXPONENT,
        "fit_response_splits": FIT_SPLITS,
        "prediction_masks": {
            split: held_masks[split].clone() for split in PREDICTION_SPLITS
        },
        "clean_endpoints": clean_endpoints.clone(),
        "method_manifest": METHOD_MANIFEST,
        "methods": methods,
    }


__all__ = [
    "ADDITIVE_RIDGE_GRID",
    "DEVELOPMENT_EXPONENT",
    "DEVELOPMENT_SCALE_RIDGES",
    "FIT_SPLITS",
    "METHOD_MANIFEST",
    "PREDICTION_SPLITS",
    "RIDGE",
    "fit_endpoint_aware_models",
]
