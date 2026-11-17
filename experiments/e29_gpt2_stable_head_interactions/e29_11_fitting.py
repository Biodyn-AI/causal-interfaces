"""Pure fixed-feature fitting for prospective E29.11 scale-ladder transfer.

The public API deliberately accepts only registered masks and the four
acquired fit/control response tables. Held c24/c32 labels cannot enter model
fitting or prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_9_transfer_fitting import (
    _feature_state,
    _permuted_feature_spec,
    _random_mlp_feature_spec,
    _source_feature_specs,
    component_permutations,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    COMPONENT_PERMUTATION_SEEDS,
    COUNT_MATCHED_CONTROL_METHOD,
    FIT_ACQUISITION_SPLITS,
    FIT_SPLITS,
    MASK_SHAPES,
    METHOD_MANIFEST,
    PREDICTION_SPLITS,
    PRIMARY_METHOD,
    PRIMARY_RIDGE,
    RANDOM_FEATURE_METHODS,
    RANDOM_FEATURE_SEEDS,
    SECONDARY_METHOD,
    SECONDARY_RIDGE,
)


def _ridge_coefficients(
    features: torch.Tensor, targets: torch.Tensor, ridge: float
) -> torch.Tensor:
    gram = features.mT @ features
    return torch.linalg.solve(
        gram + float(ridge) * torch.eye(gram.shape[0], dtype=features.dtype),
        features.mT @ targets,
    )


def fit_fixed_affine_readout(
    features: torch.Tensor, targets: torch.Tensor, *, ridge: float
) -> dict[str, Any]:
    """Fit one standardized affine readout with a prespecified ridge."""
    features = features.detach().cpu().double()
    targets = targets.detach().cpu().double()
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned matrices")
    if not len(features) or ridge < 0:
        raise ValueError("fit rows must be nonempty and ridge nonnegative")
    if not bool(features.isfinite().all()) or not bool(targets.isfinite().all()):
        raise ValueError("fit inputs must be finite")
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0)
    active = feature_scale > 1e-12
    if not bool(active.any()):
        raise ValueError("all readout features are constant")
    response_mean = targets.mean(dim=0)
    response_scale = targets.std(dim=0).clamp_min(1e-6)
    standardized_features = (
        features[:, active] - feature_mean[active]
    ) / feature_scale[active]
    standardized_targets = (targets - response_mean) / response_scale
    coefficients = _ridge_coefficients(
        standardized_features, standardized_targets, ridge
    )
    if not bool(coefficients.isfinite().all()):
        raise RuntimeError("fixed readout produced non-finite coefficients")
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "active_features": active,
        "response_mean": response_mean,
        "response_scale": response_scale,
        "coefficients": coefficients,
        "selected_ridge": float(ridge),
        "fit_rows": len(features),
        "ridge_selected_from_current_outcomes": False,
    }


def apply_affine_readout(
    readout: Mapping[str, Any], features: torch.Tensor
) -> torch.Tensor:
    """Apply a fitted readout to a response-blind feature table."""
    features = features.detach().cpu().double()
    active = readout["active_features"]
    standardized = (
        features[:, active] - readout["feature_mean"][active]
    ) / readout["feature_scale"][active]
    predictions = (standardized @ readout["coefficients"]) * readout[
        "response_scale"
    ] + readout["response_mean"]
    if not bool(predictions.isfinite().all()):
        raise RuntimeError("readout prediction is non-finite")
    return predictions


def _validate_inputs(
    source_artifact: Mapping[str, Any],
    masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if source_artifact.get("format_version") != 1:
        raise ValueError("unsupported E29.8 source artifact")
    if source_artifact.get("component_count") != 48:
        raise ValueError("E29.11 requires the frozen 48-component artifact")
    if tuple(masks) != (*FIT_ACQUISITION_SPLITS, *PREDICTION_SPLITS):
        raise ValueError("mask split order differs from E29.11 protocol")
    if tuple(fit_responses) != FIT_ACQUISITION_SPLITS:
        raise ValueError("fit-response split order differs from E29.11 protocol")
    clean_masks: dict[str, torch.Tensor] = {}
    for split, expected_shape in MASK_SHAPES.items():
        value = masks[split].detach().cpu().double()
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{split} masks must have shape {expected_shape}")
        if not bool(torch.logical_or(value == 0, value == 1).all()):
            raise ValueError(f"{split} masks must be binary")
        clean_masks[split] = value
    clean_responses: dict[str, torch.Tensor] = {}
    for split in FIT_ACQUISITION_SPLITS:
        value = fit_responses[split].detach().cpu().double()
        expected_shape = (MASK_SHAPES[split][0], 12)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{split} responses must have shape {expected_shape}")
        if not bool(value.isfinite().all()):
            raise ValueError(f"{split} responses must be finite")
        clean_responses[split] = value
    return clean_masks, clean_responses


def _fit_feature_method(
    feature_spec: Mapping[str, Any],
    masks: Mapping[str, torch.Tensor],
    responses: Mapping[str, torch.Tensor],
    fit_splits: Sequence[str],
    *,
    ridge: float,
) -> dict[str, Any]:
    fit_features = torch.cat(
        [_feature_state(feature_spec, masks[split]) for split in fit_splits]
    )
    fit_targets = torch.cat([responses[split] for split in fit_splits])
    readout = fit_fixed_affine_readout(fit_features, fit_targets, ridge=ridge)
    return {
        "family": "frozen_features_with_fixed_affine_readout",
        "feature_spec": feature_spec,
        "readout": readout,
        "fit_splits": tuple(fit_splits),
        "fit_rows": int(sum(len(masks[split]) for split in fit_splits)),
        "predictions": {
            split: apply_affine_readout(
                readout, _feature_state(feature_spec, masks[split])
            )
            for split in PREDICTION_SPLITS
        },
    }


def _fit_additive(
    masks: Mapping[str, torch.Tensor],
    responses: Mapping[str, torch.Tensor],
    fit_splits: Sequence[str],
) -> dict[str, Any]:
    readout = fit_fixed_affine_readout(
        torch.cat([masks[split] for split in fit_splits]),
        torch.cat([responses[split] for split in fit_splits]),
        ridge=10.0,
    )
    return {
        "family": "additive_fixed_affine_readout",
        "readout": readout,
        "fit_splits": tuple(fit_splits),
        "fit_rows": int(sum(len(masks[split]) for split in fit_splits)),
        "predictions": {
            split: apply_affine_readout(readout, masks[split])
            for split in PREDICTION_SPLITS
        },
    }


def _fit_intercept_correction(
    parent: Mapping[str, Any],
    feature_spec: Mapping[str, Any],
    masks: Mapping[str, torch.Tensor],
    responses: Mapping[str, torch.Tensor],
    *,
    offset_splits: Sequence[str],
) -> dict[str, Any]:
    """Keep low-only slopes fixed and fit only one offset per context."""
    parent_readout = parent["readout"]
    offsets = torch.cat(
        [
            responses[split]
            - apply_affine_readout(
                parent_readout, _feature_state(feature_spec, masks[split])
            )
            for split in offset_splits
        ]
    ).mean(dim=0)
    return {
        "family": "frozen_low_slope_with_multiscale_context_intercept",
        "parent_fit_splits": parent["fit_splits"],
        "offset_fit_splits": tuple(offset_splits),
        "context_offsets": offsets,
        "additional_parameter_count": len(offsets),
        "predictions": {
            split: apply_affine_readout(
                parent_readout, _feature_state(feature_spec, masks[split])
            )
            + offsets
            for split in PREDICTION_SPLITS
        },
    }


def _primary_learning_curves(
    feature_spec: Mapping[str, Any],
    masks: Mapping[str, torch.Tensor],
    responses: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Freeze report-only MLP7 predictions at the registered nested budgets."""
    budgets = (
        ("low64", (("calibration_low", 64),)),
        ("low96", (("calibration_low", 96),)),
        (
            "low96_c8_16",
            (("calibration_low", 96), ("calibration_size8", 16)),
        ),
        (
            "low96_c8_32",
            (("calibration_low", 96), ("calibration_size8", 32)),
        ),
        (
            "low64_c8_32_c16_32",
            (
                ("calibration_low", 64),
                ("calibration_size8", 32),
                ("bridge_size16", 32),
            ),
        ),
        (
            "low96_c8_32_c16_16",
            (
                ("calibration_low", 96),
                ("calibration_size8", 32),
                ("bridge_size16", 16),
            ),
        ),
        (
            "low96_c8_32_c16_32",
            (
                ("calibration_low", 96),
                ("calibration_size8", 32),
                ("bridge_size16", 32),
            ),
        ),
        (
            "low96_c8_32_c16_64",
            (
                ("calibration_low", 96),
                ("calibration_size8", 32),
                ("bridge_size16", 64),
            ),
        ),
    )
    curves = {}
    for name, split_budgets in budgets:
        features = torch.cat(
            [
                _feature_state(feature_spec, masks[split])[:count]
                for split, count in split_budgets
            ]
        )
        targets = torch.cat(
            [responses[split][:count] for split, count in split_budgets]
        )
        readout = fit_fixed_affine_readout(features, targets, ridge=PRIMARY_RIDGE)
        curves[name] = {
            "split_budgets": tuple(split_budgets),
            "fit_rows": len(features),
            "readout": readout,
            "predictions": {
                split: apply_affine_readout(
                    readout, _feature_state(feature_spec, masks[split])
                )
                for split in PREDICTION_SPLITS
            },
        }
    return curves


def _feature_support_metrics(
    feature_spec: Mapping[str, Any],
    masks: Mapping[str, torch.Tensor],
    fit_splits: Sequence[str],
    prediction_split: str,
) -> dict[str, Any]:
    """Quantify response-blind extrapolation in one frozen feature space."""
    fit_features = torch.cat(
        [_feature_state(feature_spec, masks[split]) for split in fit_splits]
    ).double()
    prediction_features = _feature_state(
        feature_spec, masks[prediction_split]
    ).double()
    mean = fit_features.mean(dim=0)
    scale = fit_features.std(dim=0)
    active = scale > 1e-12
    if not bool(active.any()):
        raise RuntimeError("feature-support diagnostic has no active coordinate")
    standardized_fit = (fit_features[:, active] - mean[active]) / scale[active]
    standardized_prediction = (
        prediction_features[:, active] - mean[active]
    ) / scale[active]
    maximum_absolute_z = standardized_prediction.abs().max(dim=1).values
    fit_min = fit_features[:, active].min(dim=0).values
    fit_max = fit_features[:, active].max(dim=0).values
    outside_range = torch.logical_or(
        prediction_features[:, active] < fit_min,
        prediction_features[:, active] > fit_max,
    ).any(dim=1)
    covariance = standardized_fit.mT @ standardized_fit / max(
        len(standardized_fit) - 1, 1
    )
    covariance = covariance + 1e-6 * torch.eye(
        covariance.shape[0], dtype=torch.float64
    )
    mahalanobis_squared = (
        standardized_prediction
        * torch.linalg.solve(covariance, standardized_prediction.mT).mT
    ).sum(dim=1)
    mahalanobis = mahalanobis_squared.clamp_min(0).sqrt()
    return {
        "fit_rows": len(fit_features),
        "active_feature_dimensions": int(active.sum()),
        "median_maximum_absolute_z": float(maximum_absolute_z.median()),
        "maximum_absolute_z": float(maximum_absolute_z.max()),
        "fraction_rows_outside_componentwise_fit_range": float(
            outside_range.double().mean()
        ),
        "median_mahalanobis_distance": float(mahalanobis.median()),
        "maximum_mahalanobis_distance": float(mahalanobis.max()),
    }


def _feature_support_diagnostics(
    source_specs: Mapping[str, Mapping[str, Any]],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    fit_designs = {
        "low_size8": ("calibration_low", "calibration_size8"),
        "multiscale": FIT_SPLITS,
        "size8_count_matched": (
            "calibration_low",
            "calibration_size8",
            "count_control_size8",
        ),
    }
    return {
        representation: {
            design: {
                split: _feature_support_metrics(
                    feature_spec, masks, fit_splits, split
                )
                for split in PREDICTION_SPLITS
            }
            for design, fit_splits in fit_designs.items()
        }
        for representation, feature_spec in source_specs.items()
        if representation in {"mlp7", "tt2"}
    }


def fit_multiscale_models(
    source_artifact: Mapping[str, Any],
    masks: Mapping[str, torch.Tensor],
    fit_responses: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Fit the frozen E29.11 manifest and return held predictions."""
    masks, fit_responses = _validate_inputs(source_artifact, masks, fit_responses)
    source_specs = _source_feature_specs(source_artifact)
    all_fit = FIT_SPLITS
    low_only = ("calibration_low",)
    low_c8 = ("calibration_low", "calibration_size8")
    low_c8_count_matched = (
        "calibration_low",
        "calibration_size8",
        "count_control_size8",
    )
    methods: dict[str, dict[str, Any]] = {
        PRIMARY_METHOD: _fit_feature_method(
            source_specs["mlp7"], masks, fit_responses, all_fit, ridge=PRIMARY_RIDGE
        ),
        SECONDARY_METHOD: _fit_feature_method(
            source_specs["tt2"], masks, fit_responses, all_fit, ridge=SECONDARY_RIDGE
        ),
        "mlp7_low_only_ridge10": _fit_feature_method(
            source_specs["mlp7"], masks, fit_responses, low_only, ridge=PRIMARY_RIDGE
        ),
        "mlp7_low_size8_ridge10": _fit_feature_method(
            source_specs["mlp7"], masks, fit_responses, low_c8, ridge=PRIMARY_RIDGE
        ),
        COUNT_MATCHED_CONTROL_METHOD: _fit_feature_method(
            source_specs["mlp7"],
            masks,
            fit_responses,
            low_c8_count_matched,
            ridge=PRIMARY_RIDGE,
        ),
        "tt2_low_only_ridge1e-4": _fit_feature_method(
            source_specs["tt2"], masks, fit_responses, low_only, ridge=SECONDARY_RIDGE
        ),
        "tt2_low_size8_ridge1e-4": _fit_feature_method(
            source_specs["tt2"], masks, fit_responses, low_c8, ridge=SECONDARY_RIDGE
        ),
        "additive_multiscale_ridge10": _fit_additive(
            masks, fit_responses, all_fit
        ),
    }
    for seed, method_name in zip(
        RANDOM_FEATURE_SEEDS, RANDOM_FEATURE_METHODS, strict=True
    ):
        methods[method_name] = _fit_feature_method(
            _random_mlp_feature_spec(seed),
            masks,
            fit_responses,
            all_fit,
            ridge=PRIMARY_RIDGE,
        )
    permutations = component_permutations()
    for seed in COMPONENT_PERMUTATION_SEEDS:
        method_name = f"mlp7_permuted_seed{seed}_multiscale_ridge10"
        methods[method_name] = _fit_feature_method(
            _permuted_feature_spec(source_specs["mlp7"], permutations[f"seed{seed}"]),
            masks,
            fit_responses,
            all_fit,
            ridge=PRIMARY_RIDGE,
        )
    for name, permutation in (
        ("mlp7_permuted_reverse_multiscale_ridge10", permutations["reverse"]),
        (
            "mlp7_permuted_attn_mlp_swap_multiscale_ridge10",
            permutations["attn_mlp_swap"],
        ),
    ):
        methods[name] = _fit_feature_method(
            _permuted_feature_spec(source_specs["mlp7"], permutation),
            masks,
            fit_responses,
            all_fit,
            ridge=PRIMARY_RIDGE,
        )
    methods["mlp7_low_slope_multiscale_intercept"] = _fit_intercept_correction(
        methods["mlp7_low_only_ridge10"],
        source_specs["mlp7"],
        masks,
        fit_responses,
        offset_splits=("calibration_size8", "bridge_size16"),
    )
    methods["tt2_low_slope_multiscale_intercept"] = _fit_intercept_correction(
        methods["tt2_low_only_ridge1e-4"],
        source_specs["tt2"],
        masks,
        fit_responses,
        offset_splits=("calibration_size8", "bridge_size16"),
    )
    if tuple(methods) != METHOD_MANIFEST:
        raise RuntimeError("fitter method order differs from registered manifest")
    for method in methods.values():
        for split in PREDICTION_SPLITS:
            expected = (MASK_SHAPES[split][0], 12)
            if tuple(method["predictions"][split].shape) != expected:
                raise RuntimeError("frozen prediction has the wrong shape")
    return {
        "schema_version": "e29_11_multiscale_predictions_v1",
        "status": "all_size24_and_conditional_size32_predictions_frozen",
        "component_count": 48,
        "output_count": 12,
        "method_manifest": METHOD_MANIFEST,
        "primary_method": PRIMARY_METHOD,
        "secondary_method": SECONDARY_METHOD,
        "fit_response_splits": FIT_ACQUISITION_SPLITS,
        "prediction_masks": {
            split: masks[split].clone() for split in PREDICTION_SPLITS
        },
        "primary_learning_curves": _primary_learning_curves(
            source_specs["mlp7"], masks, fit_responses
        ),
        "feature_support_diagnostics": _feature_support_diagnostics(
            source_specs, masks
        ),
        "methods": methods,
    }


__all__ = [
    "apply_affine_readout",
    "fit_fixed_affine_readout",
    "fit_multiscale_models",
]
