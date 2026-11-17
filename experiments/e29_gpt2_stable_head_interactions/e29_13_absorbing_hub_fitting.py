"""Pure prospective fitting and held-mask prediction for E29.13.

The API accepts source responses and held masks, but it has no held-response
argument.  Every method is therefore frozen before either confirmation label
table can be acquired.
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
from experiments.e29_gpt2_stable_head_interactions.e29_12_fitting import (
    _validated_mlp7_spec,
)
from experiments.e29_gpt2_stable_head_interactions.e29_13_absorbing_hub_protocol import (
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    CONTROL_COMPONENT,
    HELD_CARDINALITIES,
    HELD_MASK_SHAPES,
    HUB_COMPONENT,
    REFERENCE_MASK_SEEDS,
    SATURATION_CONSTANTS,
    SOURCE_INHERITED_SHAPES,
    SOURCE_STRATA,
)

RIDGE = 10.0
BRANCH_SHAPE_SCALES = {
    "confirmation_size24": {"hub_off": 1.0, "hub_on": 0.0},
    "confirmation_size32": {"hub_off": 0.85, "hub_on": 0.0},
}
PREDICTION_SPLITS = tuple(HELD_MASK_SHAPES)
METHOD_MANIFEST = (
    "mlp7_unbounded_ridge10",
    "additive_unbounded_ridge10",
    "mlp7_endpoint_cap_ridge10",
    "additive_endpoint_cap_ridge10",
    "mlp7_saturation_reanchored_c48",
    "gate_only_c48_calibrated",
    "gate_shape_c48_calibrated",
    "gate_shape_clean_proxy_calibrated",
    "capped_mlp_additive_ensemble_c48",
    "analytic_hub_star_c48",
    "analytic_control_star_c48",
)


def _finite_matrix(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    tensor = value.detach().cpu().double()
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not bool(tensor.isfinite().all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def saturation_fraction(cardinality: int) -> float:
    """Return the frozen average of the two development saturation curves."""
    if isinstance(cardinality, bool) or not isinstance(cardinality, int):
        raise TypeError("cardinality must be an integer")
    if not 0 <= cardinality <= COMPONENT_COUNT:
        raise ValueError("cardinality must lie between zero and 48")
    if cardinality == 0:
        return 0.0
    if cardinality == COMPONENT_COUNT:
        return 1.0
    density = cardinality / COMPONENT_COUNT
    kumaraswamy = 1.0 - (1.0 - density ** float(SATURATION_CONSTANTS["a"])) ** float(
        SATURATION_CONSTANTS["b"]
    )
    power_survival = 1.0 - (1.0 - density) ** float(SATURATION_CONSTANTS["p"])
    return 0.5 * (kumaraswamy + power_survival)


def gate_shape_prediction(
    masks: torch.Tensor,
    score: torch.Tensor,
    *,
    hub_level: torch.Tensor,
    target_panel_mean: torch.Tensor,
    clean_endpoint: torch.Tensor,
    hub_component: int = HUB_COMPONENT,
    hub_off_scale: float,
    hub_on_scale: float,
) -> torch.Tensor:
    """Set analytic gate-branch levels and retain frozen within-branch shape."""
    masks = _finite_matrix(masks, label="gate masks")
    score = _finite_matrix(score, label="gate score")
    if masks.ndim != 2 or masks.shape[1] != COMPONENT_COUNT:
        raise ValueError("gate masks must have shape (rows, 48)")
    if score.ndim != 2 or score.shape != (len(masks), CONTEXT_COUNT):
        raise ValueError("gate score must have shape (rows, 12)")
    if not bool(torch.logical_or(masks == 0.0, masks == 1.0).all()):
        raise ValueError("gate masks must be binary")
    hub_level = _finite_matrix(hub_level, label="hub level", shape=(CONTEXT_COUNT,))
    target_panel_mean = _finite_matrix(
        target_panel_mean, label="target panel mean", shape=(CONTEXT_COUNT,)
    )
    clean_endpoint = _finite_matrix(
        clean_endpoint, label="clean endpoint", shape=(CONTEXT_COUNT,)
    )
    if not 0 <= hub_component < COMPONENT_COUNT:
        raise ValueError("hub component is out of range")
    if hub_off_scale < 0.0 or hub_on_scale < 0.0:
        raise ValueError("branch shape scales must be nonnegative")

    hub_on = masks[:, hub_component].bool()
    hub_fraction = float(hub_on.double().mean())
    if hub_fraction <= 0.0 or hub_fraction >= 1.0:
        raise ValueError("gate panel must contain both hub branches")
    off_level = (target_panel_mean - hub_fraction * hub_level) / (1.0 - hub_fraction)
    prediction = torch.empty_like(score)
    for state, level, scale in (
        (False, off_level, hub_off_scale),
        (True, hub_level, hub_on_scale),
    ):
        selected = hub_on == state
        centered_score = score[selected] - score[selected].mean(dim=0, keepdim=True)
        prediction[selected] = level[None, :] + scale * centered_score
    return torch.minimum(prediction, clean_endpoint[None, :])


def solve_capped_offset(
    reference_score: torch.Tensor,
    clean_endpoint: torch.Tensor,
    target_mean: torch.Tensor,
) -> torch.Tensor:
    """Solve one monotone offset per context for an upper-capped score."""
    reference_score = _finite_matrix(reference_score, label="reference score")
    if reference_score.ndim != 2 or reference_score.shape[1] != CONTEXT_COUNT:
        raise ValueError("reference score must have shape (rows, 12)")
    clean_endpoint = _finite_matrix(
        clean_endpoint, label="clean endpoint", shape=(CONTEXT_COUNT,)
    )
    target_mean = _finite_matrix(
        target_mean, label="target mean", shape=(CONTEXT_COUNT,)
    )
    if bool((target_mean > clean_endpoint).any()):
        raise ValueError("capped target mean cannot exceed the clean endpoint")

    lower = target_mean - reference_score.max(dim=0).values - clean_endpoint.abs()
    upper = clean_endpoint - reference_score.min(dim=0).values + clean_endpoint.abs()
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        achieved = torch.minimum(
            reference_score + midpoint[None, :], clean_endpoint[None, :]
        ).mean(dim=0)
        too_low = achieved < target_mean
        lower = torch.where(too_low, midpoint, lower)
        upper = torch.where(too_low, upper, midpoint)
    offset = 0.5 * (lower + upper)
    achieved = torch.minimum(
        reference_score + offset[None, :], clean_endpoint[None, :]
    ).mean(dim=0)
    if not torch.allclose(achieved, target_mean, atol=1e-10, rtol=0.0):
        raise RuntimeError("capped offset solver did not reach its target mean")
    return offset


def analytic_star_score(
    masks: torch.Tensor,
    *,
    singleton_responses: torch.Tensor,
    pair_responses: torch.Tensor,
    hub_component: int,
) -> torch.Tensor:
    """Evaluate singles plus every measured pair interaction around one hub."""
    masks = _finite_matrix(masks, label="star masks")
    if masks.ndim != 2 or masks.shape[1] != len(singleton_responses):
        raise ValueError("star masks and singleton count disagree")
    singleton_responses = _finite_matrix(
        singleton_responses,
        label="singleton responses",
        shape=(masks.shape[1], CONTEXT_COUNT),
    )
    if not 0 <= hub_component < masks.shape[1]:
        raise ValueError("star hub is out of range")
    spokes = [
        component for component in range(masks.shape[1]) if component != hub_component
    ]
    pair_responses = _finite_matrix(
        pair_responses,
        label="pair responses",
        shape=(len(spokes), CONTEXT_COUNT),
    )
    interactions = pair_responses - singleton_responses[hub_component][None, :]
    interactions -= singleton_responses[spokes]
    additive = masks @ singleton_responses
    pair_term = masks[:, spokes] @ interactions
    return additive + masks[:, hub_component : hub_component + 1] * pair_term


def _reference_masks(split: str) -> torch.Tensor:
    rows = int(SATURATION_CONSTANTS["reference_rows"])
    cardinality = HELD_CARDINALITIES[split]
    seed = int(REFERENCE_MASK_SEEDS[f"reference_size{cardinality}"])
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(rows, COMPONENT_COUNT, generator=generator, dtype=torch.float64)
    selected = scores.topk(cardinality, dim=1).indices
    masks = torch.zeros(rows, COMPONENT_COUNT, dtype=torch.float64)
    masks.scatter_(1, selected, 1.0)
    return masks


def _extract_source_inputs(
    source_fit: Mapping[str, Any],
    source_labels: Mapping[str, Any],
    inherited_fit_masks: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if tuple(source_fit) != ("masks", "strata"):
        raise ValueError("source-fit keys or order changed")
    masks = _finite_matrix(source_fit["masks"], label="source masks")
    if tuple(masks.shape) != (283, COMPONENT_COUNT):
        raise ValueError("source masks must have shape (283, 48)")
    strata = source_fit["strata"]
    if not isinstance(strata, Mapping) or tuple(strata) != SOURCE_STRATA:
        raise ValueError("source strata changed")
    clean_strata = {
        name: torch.as_tensor(strata[name], dtype=torch.long).detach().cpu()
        for name in SOURCE_STRATA
    }
    if "responses" not in source_labels:
        raise ValueError("source labels omit responses")
    responses = _finite_matrix(
        source_labels["responses"],
        label="source responses",
        shape=(len(masks), CONTEXT_COUNT),
    )
    for split, shape in SOURCE_INHERITED_SHAPES.items():
        expected = _finite_matrix(
            inherited_fit_masks[split], label=f"inherited {split}", shape=shape
        )
        if not torch.equal(masks[clean_strata[split]], expected):
            raise ValueError(f"source stratum {split} differs from inherited masks")
    return masks, responses, clean_strata


def _method(
    family: str,
    predictions: Mapping[str, torch.Tensor],
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "family": family,
        "held_labels_accessed": False,
        **metadata,
        "predictions": {
            split: predictions[split].clone() for split in PREDICTION_SPLITS
        },
    }


def fit_absorbing_hub_models(
    source_artifact: Mapping[str, Any],
    source_fit: Mapping[str, Any],
    source_labels: Mapping[str, Any],
    inherited_fit_masks: Mapping[str, torch.Tensor],
    held_masks: Mapping[str, torch.Tensor],
    clean_endpoints: torch.Tensor,
    all_components_responses: torch.Tensor,
) -> dict[str, Any]:
    """Fit source-only readouts and freeze every E29.13 held prediction."""
    mlp_spec = _validated_mlp7_spec(source_artifact)
    source_masks, source_responses, strata = _extract_source_inputs(
        source_fit, source_labels, inherited_fit_masks
    )
    clean_endpoints = _finite_matrix(
        clean_endpoints, label="clean endpoints", shape=(CONTEXT_COUNT,)
    )
    full_responses = _finite_matrix(
        all_components_responses,
        label="all-components responses",
        shape=(CONTEXT_COUNT,),
    )
    if bool((clean_endpoints <= 0.0).any()) or bool((full_responses <= 0.0).any()):
        raise ValueError("endpoint scales must be strictly positive")
    if tuple(held_masks) != PREDICTION_SPLITS:
        raise ValueError("held-mask keys or order changed")
    clean_held = {}
    for split, shape in HELD_MASK_SHAPES.items():
        panel = _finite_matrix(held_masks[split], label=split, shape=shape)
        if not bool(torch.logical_or(panel == 0.0, panel == 1.0).all()):
            raise ValueError(f"{split} masks must be binary")
        if set(panel.sum(dim=1).tolist()) != {float(HELD_CARDINALITIES[split])}:
            raise ValueError(f"{split} cardinality changed")
        clean_held[split] = panel

    fit_indices = torch.cat([strata[split] for split in SOURCE_INHERITED_SHAPES])
    fit_masks = source_masks[fit_indices]
    fit_targets = source_responses[fit_indices]
    mlp_readout = fit_fixed_affine_readout(
        _feature_state(mlp_spec, fit_masks), fit_targets, ridge=RIDGE
    )
    additive_readout = fit_fixed_affine_readout(fit_masks, fit_targets, ridge=RIDGE)

    singleton_responses = source_responses[strata["all_singletons"]]
    hub_pair_responses = source_responses[strata["hub_incident_pairs"]]
    control_pair_responses = source_responses[strata["control_incident_pairs"]]
    hub_level = singleton_responses[HUB_COMPONENT]

    raw_mlp: dict[str, torch.Tensor] = {}
    raw_additive: dict[str, torch.Tensor] = {}
    gate_only: dict[str, torch.Tensor] = {}
    gate_shape: dict[str, torch.Tensor] = {}
    gate_shape_fallback: dict[str, torch.Tensor] = {}
    smooth_mlp: dict[str, torch.Tensor] = {}
    calibrated_ensemble: dict[str, torch.Tensor] = {}
    hub_star: dict[str, torch.Tensor] = {}
    control_star: dict[str, torch.Tensor] = {}
    target_means: dict[str, torch.Tensor] = {}
    fallback_target_means: dict[str, torch.Tensor] = {}
    capped_offsets: dict[str, dict[str, torch.Tensor]] = {}

    for split in PREDICTION_SPLITS:
        masks = clean_held[split]
        cardinality = HELD_CARDINALITIES[split]
        fraction = saturation_fraction(cardinality)
        target_mean = fraction * full_responses
        fallback_mean = (
            fraction * float(SATURATION_CONSTANTS["D_over_A"]) * clean_endpoints
        )
        target_means[split] = target_mean
        fallback_target_means[split] = fallback_mean
        raw_mlp[split] = apply_affine_readout(
            mlp_readout, _feature_state(mlp_spec, masks)
        )
        raw_additive[split] = apply_affine_readout(additive_readout, masks)
        ensemble_score = 0.5 * (raw_mlp[split] + raw_additive[split])
        scales = BRANCH_SHAPE_SCALES[split]
        gate_only[split] = gate_shape_prediction(
            masks,
            ensemble_score,
            hub_level=hub_level,
            target_panel_mean=target_mean,
            clean_endpoint=clean_endpoints,
            hub_off_scale=0.0,
            hub_on_scale=0.0,
        )
        gate_shape[split] = gate_shape_prediction(
            masks,
            ensemble_score,
            hub_level=hub_level,
            target_panel_mean=target_mean,
            clean_endpoint=clean_endpoints,
            hub_off_scale=scales["hub_off"],
            hub_on_scale=scales["hub_on"],
        )
        gate_shape_fallback[split] = gate_shape_prediction(
            masks,
            ensemble_score,
            hub_level=hub_level,
            target_panel_mean=fallback_mean,
            clean_endpoint=clean_endpoints,
            hub_off_scale=scales["hub_off"],
            hub_on_scale=scales["hub_on"],
        )
        smooth_mlp[split] = (
            raw_mlp[split]
            - raw_mlp[split].mean(dim=0, keepdim=True)
            + target_mean[None, :]
        )

        reference = _reference_masks(split)
        reference_mlp = apply_affine_readout(
            mlp_readout, _feature_state(mlp_spec, reference)
        )
        reference_additive = apply_affine_readout(additive_readout, reference)
        reference_ensemble = 0.5 * (reference_mlp + reference_additive)
        ensemble_offset = solve_capped_offset(
            reference_ensemble, clean_endpoints, target_mean
        )
        calibrated_ensemble[split] = torch.minimum(
            ensemble_score + ensemble_offset[None, :], clean_endpoints[None, :]
        )

        held_hub_star = analytic_star_score(
            masks,
            singleton_responses=singleton_responses,
            pair_responses=hub_pair_responses,
            hub_component=HUB_COMPONENT,
        )
        held_control_star = analytic_star_score(
            masks,
            singleton_responses=singleton_responses,
            pair_responses=control_pair_responses,
            hub_component=CONTROL_COMPONENT,
        )
        reference_hub_star = analytic_star_score(
            reference,
            singleton_responses=singleton_responses,
            pair_responses=hub_pair_responses,
            hub_component=HUB_COMPONENT,
        )
        reference_control_star = analytic_star_score(
            reference,
            singleton_responses=singleton_responses,
            pair_responses=control_pair_responses,
            hub_component=CONTROL_COMPONENT,
        )
        hub_offset = solve_capped_offset(
            reference_hub_star, clean_endpoints, target_mean
        )
        control_offset = solve_capped_offset(
            reference_control_star, clean_endpoints, target_mean
        )
        hub_star[split] = torch.minimum(
            held_hub_star + hub_offset[None, :], clean_endpoints[None, :]
        )
        control_star[split] = torch.minimum(
            held_control_star + control_offset[None, :], clean_endpoints[None, :]
        )
        capped_offsets[split] = {
            "ensemble": ensemble_offset,
            "hub_star": hub_offset,
            "control_star": control_offset,
        }

    methods = {
        "mlp7_unbounded_ridge10": _method(
            "frozen_mlp7_features_affine_readout", raw_mlp, readout=mlp_readout
        ),
        "additive_unbounded_ridge10": _method(
            "additive_mask_affine_readout", raw_additive, readout=additive_readout
        ),
        "mlp7_endpoint_cap_ridge10": _method(
            "clean_endpoint_upper_cap",
            {
                split: torch.minimum(raw_mlp[split], clean_endpoints[None, :])
                for split in PREDICTION_SPLITS
            },
            parent="mlp7_unbounded_ridge10",
        ),
        "additive_endpoint_cap_ridge10": _method(
            "clean_endpoint_upper_cap",
            {
                split: torch.minimum(raw_additive[split], clean_endpoints[None, :])
                for split in PREDICTION_SPLITS
            },
            parent="additive_unbounded_ridge10",
        ),
        "mlp7_saturation_reanchored_c48": _method(
            "c48_scaled_saturation_mean_with_mlp_shape",
            smooth_mlp,
            target_panel_means=target_means,
        ),
        "gate_only_c48_calibrated": _method(
            "explicit_hub_gate_branch_levels",
            gate_only,
            hub_component=HUB_COMPONENT,
            target_panel_means=target_means,
        ),
        "gate_shape_c48_calibrated": _method(
            "explicit_hub_gate_plus_frozen_off_branch_shape",
            gate_shape,
            hub_component=HUB_COMPONENT,
            branch_shape_scales=BRANCH_SHAPE_SCALES,
            target_panel_means=target_means,
        ),
        "gate_shape_clean_proxy_calibrated": _method(
            "explicit_hub_gate_plus_clean_proxy_scale",
            gate_shape_fallback,
            hub_component=HUB_COMPONENT,
            branch_shape_scales=BRANCH_SHAPE_SCALES,
            target_panel_means=fallback_target_means,
            endpoint_ratio=float(SATURATION_CONSTANTS["D_over_A"]),
        ),
        "capped_mlp_additive_ensemble_c48": _method(
            "reference_mean_calibrated_capped_score_ensemble",
            calibrated_ensemble,
            reference_rows=int(SATURATION_CONSTANTS["reference_rows"]),
            reference_seeds=dict(REFERENCE_MASK_SEEDS),
            offsets={
                split: capped_offsets[split]["ensemble"] for split in PREDICTION_SPLITS
            },
        ),
        "analytic_hub_star_c48": _method(
            "measured_singletons_plus_hub_incident_pair_interactions",
            hub_star,
            hub_component=HUB_COMPONENT,
            offsets={
                split: capped_offsets[split]["hub_star"] for split in PREDICTION_SPLITS
            },
        ),
        "analytic_control_star_c48": _method(
            "measured_singletons_plus_control_incident_pair_interactions",
            control_star,
            hub_component=CONTROL_COMPONENT,
            offsets={
                split: capped_offsets[split]["control_star"]
                for split in PREDICTION_SPLITS
            },
        ),
    }
    if tuple(methods) != METHOD_MANIFEST:
        raise RuntimeError("E29.13 method manifest changed")
    for method in methods.values():
        for split, expected_shape in HELD_MASK_SHAPES.items():
            prediction = method["predictions"][split]
            if tuple(prediction.shape) != (expected_shape[0], CONTEXT_COUNT):
                raise RuntimeError(f"{split} prediction shape changed")
            if not bool(prediction.isfinite().all()):
                raise RuntimeError(f"{split} prediction is non-finite")

    return {
        "schema_version": "e29_13_absorbing_hub_predictions_v1",
        "status": "all_methods_frozen_before_either_held_response_table",
        "component_count": COMPONENT_COUNT,
        "context_count": CONTEXT_COUNT,
        "ridge": RIDGE,
        "saturation_constants": dict(SATURATION_CONSTANTS),
        "reference_mask_seeds": dict(REFERENCE_MASK_SEEDS),
        "branch_shape_scales": BRANCH_SHAPE_SCALES,
        "method_manifest": METHOD_MANIFEST,
        "prediction_masks": {
            split: clean_held[split].clone() for split in PREDICTION_SPLITS
        },
        "clean_endpoints": clean_endpoints.clone(),
        "all_components_responses": full_responses.clone(),
        "singleton_hub_responses": hub_level.clone(),
        "methods": methods,
    }


__all__ = [
    "BRANCH_SHAPE_SCALES",
    "METHOD_MANIFEST",
    "PREDICTION_SPLITS",
    "RIDGE",
    "analytic_star_score",
    "fit_absorbing_hub_models",
    "gate_shape_prediction",
    "saturation_fraction",
    "solve_capped_offset",
]
