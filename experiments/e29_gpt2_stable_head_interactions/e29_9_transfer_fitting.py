"""Pure fitting and prediction utilities for prospective E29.9 transfer.

The API accepts calibration and validation responses, plus response-blind
prediction masks.  Final and stress labels are deliberately absent, so every
model state and prediction can be frozen before outcome acquisition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_8_scaleup_protocol import (
    RIDGE_VALUES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_9_seven_gate_protocol import (
    e29_9_gate_results,
)
from experiments.e29_gpt2_stable_head_interactions.source_model_fitting import (
    SourceFitRecipe,
)
from experiments.e29_gpt2_stable_head_interactions.tensor_train_response import (
    count_tensor_train_parameters,
)

PRIMARY_METHOD = "mlp7_frozen_primary"
PRIMARY_BUDGETS = (32, 64, 96)
SCRATCH_SEEDS = tuple(range(29_911, 29_916))
RANDOM_FEATURE_SEEDS = tuple(range(29_921, 29_926))
PERMUTATION_SEEDS = tuple(range(29_931, 29_941))
RANDOM_FEATURE_METHODS = tuple(
    f"mlp7_random_seed{seed}" for seed in RANDOM_FEATURE_SEEDS
)
LEARNED_PERMUTATION_METHODS = (
    *(f"mlp7_permuted_seed{seed}" for seed in PERMUTATION_SEEDS),
    "mlp7_permuted_reverse",
    "mlp7_permuted_attn_mlp_swap",
)
E29_9_METHODS = (
    PRIMARY_METHOD,
    "tt2_frozen",
    "cp4_frozen",
    "additive",
    "pairwise",
    "cardinality_only",
    "mlp7_scratch",
    *RANDOM_FEATURE_METHODS,
    *LEARNED_PERMUTATION_METHODS,
    "ensemble_80_mlp7_20_tt2",
)
PREDICTION_SPLITS = ("final_size16", "stress_size24")
DEFAULT_SCRATCH_RECIPE = SourceFitRecipe(training_seeds=SCRATCH_SEEDS)


def scratch_recipe_metadata(
    recipe: SourceFitRecipe = DEFAULT_SCRATCH_RECIPE,
) -> dict[str, Any]:
    """Return the complete preregistration record for scratch optimization."""
    return {
        **recipe.to_metadata(),
        "selection": "lowest_joint_standardized_validation_mse_then_seed",
        "checkpointing": "best_validation_checkpoint_within_each_seed",
        "refit_after_validation": False,
    }


DEFAULT_SCRATCH_RECIPE_METADATA = scratch_recipe_metadata()


class ScratchMLP7(torch.nn.Module):
    """The capacity-matched seven-unit MLP trained on fresh calibration only."""

    def __init__(self, *, component_count: int, output_count: int) -> None:
        super().__init__()
        if component_count < 1 or output_count < 1:
            raise ValueError("model dimensions must be positive")
        self.component_count = component_count
        self.output_count = output_count
        self.input = torch.nn.Linear(component_count, 7)
        self.activation = torch.nn.GELU()
        self.output = torch.nn.Linear(7, output_count)

    def latent_state(self, masks: torch.Tensor) -> torch.Tensor:
        return self.activation(self.input(masks.float()))

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        return self.output(self.latent_state(masks))


def component_permutations(*, component_count: int = 48) -> dict[str, torch.Tensor]:
    """Return the twelve fixed response-blind learned-feature permutations."""
    if component_count < 2 or component_count % 2:
        raise ValueError("component_count must be positive and even")
    permutations = {
        f"seed{seed}": torch.randperm(
            component_count,
            generator=torch.Generator().manual_seed(seed),
        )
        for seed in PERMUTATION_SEEDS
    }
    permutations["reverse"] = torch.arange(component_count - 1, -1, -1)
    permutations["attn_mlp_swap"] = torch.tensor(
        [
            component
            for layer in range(component_count // 2)
            for component in (2 * layer + 1, 2 * layer)
        ],
        dtype=torch.long,
    )
    return permutations


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _validate_inputs(
    source_artifact: Mapping[str, Any],
    calibration_masks: torch.Tensor,
    calibration_responses: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_responses: torch.Tensor,
    prediction_masks: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if source_artifact.get("format_version") != 1:
        raise ValueError("unsupported E29.8 source artifact")
    if source_artifact.get("component_count") != 48:
        raise ValueError("E29.9 requires the frozen 48-component source artifact")
    required_source_methods = {
        "mlp_hidden7",
        "tt_natural_rank2",
        "cp_rank4",
    }
    if not required_source_methods <= source_artifact.get("methods", {}).keys():
        raise ValueError("source artifact omits a required frozen feature model")
    if calibration_masks.shape != (96, 48):
        raise ValueError("calibration_masks must have shape (96, 48)")
    if validation_masks.shape != (32, 48):
        raise ValueError("validation_masks must have shape (32, 48)")
    if calibration_responses.shape != (96, 12):
        raise ValueError("calibration_responses must have shape (96, 12)")
    if validation_responses.shape != (32, 12):
        raise ValueError("validation_responses must have shape (32, 12)")
    if tuple(prediction_masks) != PREDICTION_SPLITS:
        raise ValueError(f"prediction mask keys must be {PREDICTION_SPLITS}")
    expected_prediction_shapes = {
        "final_size16": (64, 48),
        "stress_size24": (32, 48),
    }
    for name, expected_shape in expected_prediction_shapes.items():
        if prediction_masks[name].shape != expected_shape:
            raise ValueError(f"{name} masks must have shape {expected_shape}")
    tensors = (
        calibration_masks,
        calibration_responses,
        validation_masks,
        validation_responses,
        *(prediction_masks[name] for name in PREDICTION_SPLITS),
    )
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("transfer inputs must be finite")
    for masks in (
        calibration_masks,
        validation_masks,
        *(prediction_masks[name] for name in PREDICTION_SPLITS),
    ):
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise ValueError("prediction and fit masks must be binary")
    return tuple(tensor.detach().cpu().double() for tensor in tensors)


def _source_feature_specs(
    source_artifact: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    source_methods = source_artifact["methods"]
    mlp_state = source_methods["mlp_hidden7"]["state"]
    if mlp_state["input.weight"].shape != (7, 48):
        raise ValueError("frozen source MLP does not have seven 48-input gates")
    return {
        "mlp7": {
            "family": "mlp7_features",
            "input_weight": mlp_state["input.weight"].detach().cpu().clone(),
            "input_bias": mlp_state["input.bias"].detach().cpu().clone(),
            "permutation": torch.arange(48),
        },
        "tt2": {
            "family": "tt_features",
            "rank": 2,
            "component_order": torch.tensor(
                source_methods["tt_natural_rank2"]["component_order"],
                dtype=torch.long,
            ),
            "state": _clone_state(source_methods["tt_natural_rank2"]["state"]),
        },
        "cp4": {
            "family": "cp_features",
            "rank": 4,
            "state": _clone_state(source_methods["cp_rank4"]["state"]),
        },
    }


def _random_mlp_feature_spec(seed: int) -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = torch.nn.Linear(48, 7)
    return {
        "family": "mlp7_features",
        "input_weight": layer.weight.detach().cpu().clone(),
        "input_bias": layer.bias.detach().cpu().clone(),
        "permutation": torch.arange(48),
        "random_feature_seed": seed,
    }


def _permuted_feature_spec(
    learned_spec: Mapping[str, Any], permutation: torch.Tensor
) -> dict[str, Any]:
    return {
        "family": "mlp7_features",
        "input_weight": learned_spec["input_weight"].clone(),
        "input_bias": learned_spec["input_bias"].clone(),
        "permutation": permutation.detach().cpu().clone(),
        "source": "frozen_learned_mlp7_with_response_blind_input_permutation",
    }


def _polynomial_features(masks: torch.Tensor, degree: int) -> torch.Tensor:
    masks = masks.double()
    if degree == 1:
        return masks
    if degree != 2:
        raise ValueError("polynomial degree must be one or two")
    pair_indices = torch.triu_indices(48, 48, offset=1)
    return torch.cat(
        (masks, masks[:, pair_indices[0]] * masks[:, pair_indices[1]]),
        dim=1,
    )


def _feature_state(spec: Mapping[str, Any], masks: torch.Tensor) -> torch.Tensor:
    family = spec["family"]
    masks = masks.detach().cpu()
    if family == "mlp7_features":
        permutation = spec["permutation"].long()
        return torch.nn.functional.gelu(
            torch.nn.functional.linear(
                masks.float()[:, permutation],
                spec["input_weight"].float(),
                spec["input_bias"].float(),
            )
        ).double()
    if family == "tt_features":
        state_record = spec["state"]
        ordered = masks[:, spec["component_order"].long()].long()
        state = state_record["first_core"][ordered[:, 0]]
        for component_index in range(1, 48):
            core = state_record[f"middle_cores.{component_index - 1}"]
            state = torch.einsum("bi,bij->bj", state, core[ordered[:, component_index]])
        return state.double()
    if family == "cp_features":
        factors = spec["state"]["factors"]
        indices = torch.arange(48)[None]
        return factors[indices, masks.long()].prod(dim=1).double()
    if family == "additive_features":
        return _polynomial_features(masks, 1)
    if family == "pairwise_features":
        return _polynomial_features(masks, 2)
    if family == "cardinality_features":
        return masks.double().sum(dim=1, keepdim=True)
    raise ValueError(f"unknown feature family {family}")


def _ridge_coefficients(
    features: torch.Tensor, targets: torch.Tensor, ridge: float
) -> torch.Tensor:
    if features.shape[1] <= features.shape[0]:
        gram = features.mT @ features
        return torch.linalg.solve(
            gram + ridge * torch.eye(gram.shape[0], dtype=features.dtype),
            features.mT @ targets,
        )
    gram = features @ features.mT
    return features.mT @ torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=features.dtype), targets
    )


def _fit_affine_readout(
    calibration_features: torch.Tensor,
    calibration_responses: torch.Tensor,
    validation_features: torch.Tensor,
    validation_responses: torch.Tensor,
    *,
    ridge_values: Sequence[float],
) -> dict[str, Any]:
    if not ridge_values or any(value < 0.0 for value in ridge_values):
        raise ValueError("ridge_values must be nonempty and nonnegative")
    calibration_features = calibration_features.double()
    validation_features = validation_features.double()
    calibration_responses = calibration_responses.double()
    validation_responses = validation_responses.double()
    feature_mean = calibration_features.mean(dim=0)
    feature_scale = calibration_features.std(dim=0)
    active = feature_scale > 1e-12
    standardized_calibration = (
        calibration_features[:, active] - feature_mean[active]
    ) / feature_scale[active]
    standardized_validation = (
        validation_features[:, active] - feature_mean[active]
    ) / feature_scale[active]
    response_mean = calibration_responses.mean(dim=0)
    response_scale = calibration_responses.std(dim=0).clamp_min(1e-6)
    calibration_targets = (calibration_responses - response_mean) / response_scale
    validation_targets = (validation_responses - response_mean) / response_scale
    candidates = []
    for ridge in ridge_values:
        coefficients = _ridge_coefficients(
            standardized_calibration, calibration_targets, float(ridge)
        )
        validation_loss = (
            (standardized_validation @ coefficients - validation_targets)
            .square()
            .mean()
        )
        if not bool(torch.isfinite(coefficients).all()) or not bool(
            torch.isfinite(validation_loss)
        ):
            raise RuntimeError("affine readout produced a non-finite value")
        candidates.append((float(validation_loss), float(ridge), coefficients))
    loss, ridge, coefficients = min(
        candidates, key=lambda candidate: (candidate[0], candidate[1])
    )
    return {
        "selected_ridge": ridge,
        "validation_standardized_mse": loss,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "active_features": active,
        "response_mean": response_mean,
        "response_scale": response_scale,
        "coefficients": coefficients,
        "refit_on_validation": False,
    }


def _apply_readout(readout: Mapping[str, Any], features: torch.Tensor) -> torch.Tensor:
    active = readout["active_features"]
    standardized = (
        features.double()[:, active] - readout["feature_mean"][active]
    ) / readout["feature_scale"][active]
    return (standardized @ readout["coefficients"]) * readout[
        "response_scale"
    ] + readout["response_mean"]


def _fit_feature_method(
    feature_spec: dict[str, Any],
    calibration_masks: torch.Tensor,
    calibration_responses: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_responses: torch.Tensor,
    *,
    ridge_values: Sequence[float],
    frozen_feature_parameter_count: int,
) -> dict[str, Any]:
    calibration_features = _feature_state(feature_spec, calibration_masks)
    validation_features = _feature_state(feature_spec, validation_masks)
    readout = _fit_affine_readout(
        calibration_features,
        calibration_responses,
        validation_features,
        validation_responses,
        ridge_values=ridge_values,
    )
    feature_width = calibration_features.shape[1]
    output_count = calibration_responses.shape[1]
    affine_parameter_count = (feature_width + 1) * output_count
    return {
        "family": "frozen_features_with_affine_readout",
        "feature_spec": feature_spec,
        "readout": readout,
        "selected_ridge": readout["selected_ridge"],
        "validation_standardized_mse": readout["validation_standardized_mse"],
        "frozen_feature_parameter_count": frozen_feature_parameter_count,
        "affine_readout_parameter_count": affine_parameter_count,
        "parameter_count": frozen_feature_parameter_count + affine_parameter_count,
        "calibration_rows": len(calibration_masks),
        "validation_rows": len(validation_masks),
        "refit_on_validation": False,
    }


def _fit_scratch_mlp(
    calibration_masks: torch.Tensor,
    calibration_responses: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_responses: torch.Tensor,
    recipe: SourceFitRecipe,
) -> dict[str, Any]:
    response_mean = calibration_responses.double().mean(dim=0)
    response_scale = calibration_responses.double().std(dim=0).clamp_min(1e-6)
    calibration_targets = (
        (calibration_responses.double() - response_mean) / response_scale
    ).float()
    validation_targets = (
        (validation_responses.double() - response_mean) / response_scale
    ).float()
    candidates = []
    seed_records = []
    for seed in recipe.training_seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ScratchMLP7(component_count=48, output_count=12)
            parameters = list(model.parameters())
            optimizer = torch.optim.Adam(
                parameters,
                lr=recipe.learning_rate,
                weight_decay=recipe.weight_decay,
            )
            best_loss = math.inf
            best_epoch = 0
            best_state = None
            epochs_completed = 0
            failure_stage = None
            for epoch in range(1, recipe.maximum_epochs + 1):
                optimizer.zero_grad()
                training_loss = (
                    (model(calibration_masks.float()) - calibration_targets)
                    .square()
                    .mean()
                )
                if not bool(torch.isfinite(training_loss)):
                    failure_stage = "training_loss"
                    break
                training_loss.backward()
                if any(
                    parameter.grad is None
                    or not bool(torch.isfinite(parameter.grad).all())
                    for parameter in parameters
                ):
                    failure_stage = "gradients"
                    break
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, recipe.gradient_clip_norm
                )
                if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
                    failure_stage = "gradient_norm"
                    break
                optimizer.step()
                if any(
                    not bool(torch.isfinite(parameter).all())
                    for parameter in parameters
                ):
                    failure_stage = "parameter_state"
                    break
                epochs_completed = epoch
                if (
                    epoch % recipe.evaluation_interval != 0
                    and epoch != recipe.maximum_epochs
                ):
                    continue
                with torch.no_grad():
                    validation_loss = (
                        (model(validation_masks.float()) - validation_targets)
                        .square()
                        .mean()
                    )
                if not bool(torch.isfinite(validation_loss)):
                    failure_stage = "validation_loss"
                    break
                value = float(validation_loss)
                if value < best_loss - 1e-7:
                    best_loss = value
                    best_epoch = epoch
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                if epoch - best_epoch >= recipe.patience_epochs:
                    break
            if failure_stage is not None:
                seed_records.append(
                    {
                        "seed": seed,
                        "status": "failed_nonfinite",
                        "failure_stage": failure_stage,
                        "epochs_completed": epochs_completed,
                    }
                )
                continue
            if best_state is None:
                raise RuntimeError("scratch MLP did not produce a checkpoint")
            candidate = {
                "seed": seed,
                "status": "success",
                "best_epoch": best_epoch,
                "epochs_completed": epochs_completed,
                "standardized_validation_mse": best_loss,
                "state": best_state,
            }
            candidates.append(candidate)
            seed_records.append(
                {key: value for key, value in candidate.items() if key != "state"}
            )
    if not candidates:
        raise RuntimeError("all scratch MLP seeds failed numerical checks")
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["standardized_validation_mse"],
            candidate["seed"],
        ),
    )
    return {
        "family": "scratch_mlp7",
        "state": selected["state"],
        "response_mean": response_mean,
        "response_scale": response_scale,
        "selected_seed": selected["seed"],
        "selected_epoch": selected["best_epoch"],
        "validation_standardized_mse": selected["standardized_validation_mse"],
        "seed_records": seed_records,
        "recipe": scratch_recipe_metadata(recipe),
        "parameter_count": 439,
        "calibration_rows": len(calibration_masks),
        "validation_rows": len(validation_masks),
        "refit_on_validation": False,
    }


def _predict_record(record: Mapping[str, Any], masks: torch.Tensor) -> torch.Tensor:
    if record["family"] == "frozen_features_with_affine_readout":
        return _apply_readout(
            record["readout"], _feature_state(record["feature_spec"], masks)
        )
    if record["family"] == "scratch_mlp7":
        model = ScratchMLP7(component_count=48, output_count=12)
        model.load_state_dict(record["state"])
        model.eval()
        with torch.no_grad():
            standardized = model(masks.float()).double()
        return standardized * record["response_scale"] + record["response_mean"]
    raise ValueError(f"method family {record['family']} is not directly predictive")


def _feature_preflight(features: torch.Tensor) -> dict[str, Any]:
    features = features.double()
    scale = features.std(dim=0)
    active = scale > 1e-12
    standardized = (features[:, active] - features[:, active].mean(dim=0)) / scale[
        active
    ]
    singular_values = torch.linalg.svdvals(standardized)
    rank = int(torch.linalg.matrix_rank(standardized))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) and singular_values[-1] > 0
        else float("inf")
    )
    passed = int(active.sum()) == 7 and rank == 7 and condition <= 1_000.0
    return {
        "active_feature_count": int(active.sum()),
        "centered_standardized_rank": rank,
        "condition_number": condition,
        "singular_values": singular_values,
        "threshold_rank": 7,
        "maximum_condition_number": 1_000.0,
        "passed": passed,
    }


def primary_feature_preflight(
    source_artifact: Mapping[str, Any], calibration_masks: torch.Tensor
) -> dict[str, Any]:
    """Compute the frozen primary design check without accepting any responses."""
    if source_artifact.get("format_version") != 1:
        raise ValueError("unsupported E29.8 source artifact")
    if source_artifact.get("component_count") != 48:
        raise ValueError("E29.9 requires the frozen 48-component source artifact")
    if calibration_masks.shape != (96, 48):
        raise ValueError("calibration_masks must have shape (96, 48)")
    if not bool(torch.isfinite(calibration_masks).all()) or not bool(
        torch.logical_or(calibration_masks == 0, calibration_masks == 1).all()
    ):
        raise ValueError("calibration_masks must be finite and binary")
    source_specs = _source_feature_specs(source_artifact)
    return _feature_preflight(
        _feature_state(source_specs["mlp7"], calibration_masks.detach().cpu())
    )


def _primary_feature_anatomy(feature_spec: Mapping[str, Any]) -> dict[str, Any]:
    weight = feature_spec["input_weight"].detach().cpu().clone()
    bias = feature_spec["input_bias"].detach().cpu().clone()
    singular_values = torch.linalg.svdvals(weight.double())
    units = []
    for unit_index, unit_weights in enumerate(weight):
        ordered = sorted(
            range(48),
            key=lambda component: (-abs(float(unit_weights[component])), component),
        )
        ranked = [
            {
                "rank": rank + 1,
                "component_index": component,
                "layer_index": component // 2,
                "sublayer_type": "attention" if component % 2 == 0 else "mlp",
                "weight": float(unit_weights[component]),
                "absolute_weight": abs(float(unit_weights[component])),
            }
            for rank, component in enumerate(ordered)
        ]
        units.append(
            {
                "unit_index": unit_index,
                "bias": float(bias[unit_index]),
                "ranked_component_weights": ranked,
                "top_absolute_weights": ranked[:8],
            }
        )
    float_weight = weight.float()
    float_bias = bias.float()
    origin = torch.nn.functional.gelu(float_bias)
    singleton_values = torch.nn.functional.gelu(float_bias[:, None] + float_weight)
    singleton_differences = singleton_values - origin[:, None]
    pair_indices = torch.triu_indices(48, 48, offset=1).mT
    pair_values = torch.nn.functional.gelu(
        float_bias[:, None]
        + float_weight[:, pair_indices[:, 0]]
        + float_weight[:, pair_indices[:, 1]]
    )
    pair_differences = (
        pair_values
        - singleton_values[:, pair_indices[:, 0]]
        - singleton_values[:, pair_indices[:, 1]]
        + origin[:, None]
    )
    return {
        "status": "response_blind_feature_anatomy_frozen_before_final_labels",
        "input_weight": weight,
        "input_bias": bias,
        "singular_values": singular_values,
        "units": units,
        "empty_origin_gate_values": origin,
        "singleton_finite_differences": singleton_differences,
        "pair_indices": pair_indices,
        "pair_finite_differences": pair_differences,
        "uses_response_labels": False,
        "uses_readout": False,
    }


def fit_transfer_models(
    source_artifact: Mapping[str, Any],
    calibration_masks: torch.Tensor,
    calibration_responses: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_responses: torch.Tensor,
    prediction_masks: Mapping[str, torch.Tensor],
    *,
    ridge_values: Sequence[float] = RIDGE_VALUES,
    scratch_recipe: SourceFitRecipe = DEFAULT_SCRATCH_RECIPE,
) -> dict[str, Any]:
    """Fit all E29.9 methods and freeze response-blind final predictions."""
    converted = _validate_inputs(
        source_artifact,
        calibration_masks,
        calibration_responses,
        validation_masks,
        validation_responses,
        prediction_masks,
    )
    calibration_masks = converted[0]
    calibration_responses = converted[1]
    validation_masks = converted[2]
    validation_responses = converted[3]
    prediction_masks = {
        name: converted[4 + index] for index, name in enumerate(PREDICTION_SPLITS)
    }
    source_specs = _source_feature_specs(source_artifact)
    preflight = primary_feature_preflight(source_artifact, calibration_masks)
    methods: dict[str, dict[str, Any]] = {}

    primary_curves = {}
    for budget in PRIMARY_BUDGETS:
        record = _fit_feature_method(
            source_specs["mlp7"],
            calibration_masks[:budget],
            calibration_responses[:budget],
            validation_masks,
            validation_responses,
            ridge_values=ridge_values,
            frozen_feature_parameter_count=343,
        )
        primary_curves[budget] = {
            "calibration_rows": budget,
            "confirmatory": budget == 96,
            "readout": record["readout"],
            "predictions": {
                split: _predict_record(record, masks)
                for split, masks in prediction_masks.items()
            },
        }
        if budget == 96:
            methods[PRIMARY_METHOD] = record

    methods["tt2_frozen"] = _fit_feature_method(
        source_specs["tt2"],
        calibration_masks,
        calibration_responses,
        validation_masks,
        validation_responses,
        ridge_values=ridge_values,
        frozen_feature_parameter_count=(
            count_tensor_train_parameters(48, 8, 2) - 2 * 8
        ),
    )
    methods["cp4_frozen"] = _fit_feature_method(
        source_specs["cp4"],
        calibration_masks,
        calibration_responses,
        validation_masks,
        validation_responses,
        ridge_values=ridge_values,
        frozen_feature_parameter_count=48 * 2 * 4,
    )
    direct_specs = {
        "additive": ({"family": "additive_features"}, 0),
        "pairwise": ({"family": "pairwise_features"}, 0),
        "cardinality_only": ({"family": "cardinality_features"}, 0),
    }
    for name, (spec, feature_parameters) in direct_specs.items():
        methods[name] = _fit_feature_method(
            spec,
            calibration_masks,
            calibration_responses,
            validation_masks,
            validation_responses,
            ridge_values=ridge_values,
            frozen_feature_parameter_count=feature_parameters,
        )
    methods["mlp7_scratch"] = _fit_scratch_mlp(
        calibration_masks,
        calibration_responses,
        validation_masks,
        validation_responses,
        scratch_recipe,
    )
    for seed, method_name in zip(
        RANDOM_FEATURE_SEEDS, RANDOM_FEATURE_METHODS, strict=True
    ):
        methods[method_name] = _fit_feature_method(
            _random_mlp_feature_spec(seed),
            calibration_masks,
            calibration_responses,
            validation_masks,
            validation_responses,
            ridge_values=ridge_values,
            frozen_feature_parameter_count=343,
        )
    permutations = component_permutations(component_count=48)
    for seed in PERMUTATION_SEEDS:
        method_name = f"mlp7_permuted_seed{seed}"
        methods[method_name] = _fit_feature_method(
            _permuted_feature_spec(source_specs["mlp7"], permutations[f"seed{seed}"]),
            calibration_masks,
            calibration_responses,
            validation_masks,
            validation_responses,
            ridge_values=ridge_values,
            frozen_feature_parameter_count=343,
        )
    for permutation_name, method_name in (
        ("reverse", "mlp7_permuted_reverse"),
        ("attn_mlp_swap", "mlp7_permuted_attn_mlp_swap"),
    ):
        methods[method_name] = _fit_feature_method(
            _permuted_feature_spec(
                source_specs["mlp7"], permutations[permutation_name]
            ),
            calibration_masks,
            calibration_responses,
            validation_masks,
            validation_responses,
            ridge_values=ridge_values,
            frozen_feature_parameter_count=343,
        )
    methods["ensemble_80_mlp7_20_tt2"] = {
        "family": "fixed_prediction_ensemble",
        "members": {PRIMARY_METHOD: 0.8, "tt2_frozen": 0.2},
        "parameter_count": methods[PRIMARY_METHOD]["parameter_count"]
        + methods["tt2_frozen"]["parameter_count"],
        "additional_parameter_count": 0,
        "calibration_rows": 96,
        "validation_rows": 32,
        "refit_on_validation": False,
    }
    methods = {name: methods[name] for name in E29_9_METHODS}

    artifact = {
        "format_version": 1,
        "status": "predictions_frozen_before_final_or_stress_labels",
        "component_count": 48,
        "output_count": 12,
        "method_manifest": E29_9_METHODS,
        "primary_method": PRIMARY_METHOD,
        "confirmatory_primary_budget": 96,
        "primary_learning_curves": primary_curves,
        "primary_feature_preflight": preflight,
        "primary_feature_anatomy": _primary_feature_anatomy(source_specs["mlp7"]),
        "component_permutations": permutations,
        "methods": methods,
        "prediction_masks": {
            name: masks.clone() for name, masks in prediction_masks.items()
        },
    }
    artifact["predictions"] = {
        method_name: {
            split: predict_transfer_method(artifact, method_name, masks)
            for split, masks in prediction_masks.items()
        }
        for method_name in E29_9_METHODS
    }
    return artifact


def predict_transfer_method(
    artifact: Mapping[str, Any], method_name: str, masks: torch.Tensor
) -> torch.Tensor:
    """Reconstruct any frozen E29.9 method prediction from serialized state."""
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported E29.9 artifact")
    if method_name not in artifact["methods"]:
        raise KeyError(f"unknown E29.9 method {method_name}")
    if masks.ndim != 2 or masks.shape[1] != artifact["component_count"]:
        raise ValueError("prediction masks have the wrong shape")
    if not bool(torch.logical_or(masks == 0, masks == 1).all()):
        raise ValueError("prediction masks must be binary")
    record = artifact["methods"][method_name]
    if record["family"] == "fixed_prediction_ensemble":
        return sum(
            weight * predict_transfer_method(artifact, member, masks)
            for member, weight in record["members"].items()
        )
    prediction = _predict_record(record, masks.detach().cpu().double())
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("reconstructed prediction is non-finite")
    return prediction


def predict_primary_budget(
    artifact: Mapping[str, Any], budget: int, masks: torch.Tensor
) -> torch.Tensor:
    """Reconstruct a nested-budget primary prediction from its frozen readout."""
    if budget not in PRIMARY_BUDGETS:
        raise ValueError(f"budget must be one of {PRIMARY_BUDGETS}")
    primary = artifact["methods"][PRIMARY_METHOD]
    features = _feature_state(primary["feature_spec"], masks)
    return _apply_readout(
        artifact["primary_learning_curves"][budget]["readout"], features
    )


__all__ = [
    "DEFAULT_SCRATCH_RECIPE",
    "DEFAULT_SCRATCH_RECIPE_METADATA",
    "E29_9_METHODS",
    "LEARNED_PERMUTATION_METHODS",
    "PERMUTATION_SEEDS",
    "PREDICTION_SPLITS",
    "PRIMARY_BUDGETS",
    "PRIMARY_METHOD",
    "RANDOM_FEATURE_METHODS",
    "RANDOM_FEATURE_SEEDS",
    "SCRATCH_SEEDS",
    "ScratchMLP7",
    "component_permutations",
    "e29_9_gate_results",
    "fit_transfer_models",
    "predict_primary_budget",
    "predict_transfer_method",
    "primary_feature_preflight",
    "scratch_recipe_metadata",
]
