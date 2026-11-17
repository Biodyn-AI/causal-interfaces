"""Pure source-response fitting for the E29.8 tensor scale experiment.

This module consumes frozen masks and response matrices.  It does not import a
language model or acquire outcomes.  Validation responses select checkpoints,
seeds, and ridge values; they are never included in an optimization update or
in a post-selection refit.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions import e29_8_scaleup_protocol
from experiments.e29_gpt2_stable_head_interactions.scaled_tensor_response import (
    BinaryCPResponse,
    BinaryMLPResponse,
    BinaryTensorTrain,
)

SOURCE_METHODS = e29_8_scaleup_protocol.SOURCE_METHODS


@dataclass(frozen=True)
class SourceFitRecipe:
    """Registration-ready optimization choices for frozen source responses."""

    training_seeds: tuple[int, ...] = e29_8_scaleup_protocol.TRAINING_SEEDS
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    maximum_epochs: int = 1_600
    evaluation_interval: int = 20
    patience_epochs: int = 400
    gradient_clip_norm: float = 5.0
    ridge_values: tuple[float, ...] = e29_8_scaleup_protocol.RIDGE_VALUES
    random_order_seed: int = e29_8_scaleup_protocol.ORDER_SEED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "training_seeds", tuple(int(seed) for seed in self.training_seeds)
        )
        object.__setattr__(
            self, "ridge_values", tuple(float(value) for value in self.ridge_values)
        )
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("training_seeds must be nonempty and distinct")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.weight_decay)
            or not math.isfinite(self.gradient_clip_norm)
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("optimizer values are invalid")
        if (
            self.maximum_epochs < 1
            or self.evaluation_interval < 1
            or self.patience_epochs < 1
        ):
            raise ValueError("epoch settings must be positive")
        if not self.ridge_values or any(value <= 0.0 for value in self.ridge_values):
            raise ValueError("ridge_values must be nonempty and positive")

    def to_metadata(self) -> dict[str, Any]:
        """Return a torch- and JSON-friendly description of the recipe."""
        return {
            "training_seeds": self.training_seeds,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "maximum_epochs": self.maximum_epochs,
            "evaluation_interval": self.evaluation_interval,
            "patience_epochs": self.patience_epochs,
            "gradient_clip_norm": self.gradient_clip_norm,
            "ridge_values": self.ridge_values,
            "random_order_seed": self.random_order_seed,
        }


DEFAULT_SOURCE_FIT_RECIPE = SourceFitRecipe()


def _validated_cpu_data(
    train_masks: torch.Tensor,
    train_responses: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_responses: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    tensors = (
        train_masks,
        train_responses,
        validation_masks,
        validation_responses,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("masks and responses must be tensors")
    if train_masks.ndim != 2 or validation_masks.ndim != 2:
        raise ValueError("mask arrays must be matrices")
    if train_masks.shape[1] < 2 or validation_masks.shape[1] != train_masks.shape[1]:
        raise ValueError("train and validation masks need the same component width")
    if train_responses.ndim != 2 or validation_responses.ndim != 2:
        raise ValueError("response arrays must be matrices")
    if len(train_masks) != len(train_responses) or len(validation_masks) != len(
        validation_responses
    ):
        raise ValueError("mask and response rows do not align")
    if len(train_masks) < 2 or len(validation_masks) < 1:
        raise ValueError(
            "source fitting needs at least two train and one validation row"
        )
    if (
        train_responses.shape[1] < 1
        or validation_responses.shape[1] != train_responses.shape[1]
    ):
        raise ValueError("train and validation responses need the same output width")
    for masks in (train_masks, validation_masks):
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise ValueError("masks must be binary")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("masks and responses must be finite")
    return tuple(tensor.detach().cpu().double() for tensor in tensors)


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


class _NonFiniteSeedError(RuntimeError):
    """Internal marker for a numerically divergent optimization seed."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"non-finite {stage}")


def _require_finite_tensor(value: torch.Tensor, stage: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise _NonFiniteSeedError(stage)


def _require_finite_state(state: dict[str, torch.Tensor], stage: str) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise _NonFiniteSeedError(stage)


def _fit_neural_model(
    model_factory: Callable[[], torch.nn.Module],
    train_masks: torch.Tensor,
    train_targets: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_targets: torch.Tensor,
    recipe: SourceFitRecipe,
) -> dict[str, Any]:
    """Select one training-only checkpoint and seed on joint validation MSE."""
    candidates = []
    seed_records = []
    for seed in recipe.training_seeds:
        epochs_completed = 0
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = model_factory()
                parameters = list(model.parameters())
                optimizer = torch.optim.Adam(
                    parameters,
                    lr=recipe.learning_rate,
                    weight_decay=recipe.weight_decay,
                )
                best_loss = math.inf
                best_epoch = 0
                best_state: dict[str, torch.Tensor] | None = None
                for epoch in range(1, recipe.maximum_epochs + 1):
                    optimizer.zero_grad()
                    training_loss = (
                        (model(train_masks.float()) - train_targets.float())
                        .square()
                        .mean()
                    )
                    _require_finite_tensor(training_loss, "training_loss")
                    training_loss.backward()
                    gradients = [
                        parameter.grad
                        for parameter in parameters
                        if parameter.grad is not None
                    ]
                    if not gradients or any(
                        not bool(torch.isfinite(gradient).all())
                        for gradient in gradients
                    ):
                        raise _NonFiniteSeedError("gradients")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, recipe.gradient_clip_norm
                    )
                    _require_finite_tensor(
                        torch.as_tensor(gradient_norm), "gradient_norm"
                    )
                    optimizer.step()
                    if any(
                        not bool(torch.isfinite(parameter).all())
                        for parameter in parameters
                    ):
                        raise _NonFiniteSeedError("parameter_state")
                    epochs_completed = epoch
                    if (
                        epoch % recipe.evaluation_interval != 0
                        and epoch != recipe.maximum_epochs
                    ):
                        continue
                    with torch.no_grad():
                        validation_loss = (
                            (
                                model(validation_masks.float())
                                - validation_targets.float()
                            )
                            .square()
                            .mean()
                        )
                    _require_finite_tensor(validation_loss, "validation_loss")
                    loss_value = float(validation_loss)
                    if loss_value < best_loss - 1e-7:
                        best_loss = loss_value
                        best_epoch = epoch
                        candidate_state = _clone_state(model)
                        _require_finite_state(candidate_state, "selected_state")
                        best_state = candidate_state
                    if epoch - best_epoch >= recipe.patience_epochs:
                        break
                if best_state is None:
                    raise RuntimeError(
                        "training did not produce a validation checkpoint"
                    )
                candidate = {
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "epochs_completed": epochs_completed,
                    "standardized_validation_mse": best_loss,
                    "status": "success",
                    "state": best_state,
                }
                candidates.append(candidate)
                seed_records.append(
                    {key: value for key, value in candidate.items() if key != "state"}
                )
        except _NonFiniteSeedError as error:
            seed_records.append(
                {
                    "seed": seed,
                    "status": "failed_nonfinite",
                    "failure_stage": error.stage,
                    "epochs_completed": epochs_completed,
                }
            )
            continue
    if not candidates:
        failure_summary = ", ".join(
            f"seed {record['seed']}: {record.get('failure_stage', 'unknown')}"
            for record in seed_records
        )
        raise RuntimeError(
            f"all registered training seeds failed numerical checks: {failure_summary}"
        )
    selected = min(
        candidates,
        key=lambda record: (
            record["standardized_validation_mse"],
            record["seed"],
        ),
    )
    return {
        "selected_seed": selected["seed"],
        "selected_epoch": selected["best_epoch"],
        "validation_standardized_mse": selected["standardized_validation_mse"],
        "seed_records": seed_records,
        "state": selected["state"],
        "numerical_status": "finite",
    }


def _polynomial_features(masks: torch.Tensor, degree: int) -> torch.Tensor:
    if degree not in (1, 2):
        raise ValueError("only additive and complete-pairwise features are supported")
    masks = masks.double()
    if degree == 1:
        return masks
    pair_indices = torch.triu_indices(masks.shape[1], masks.shape[1], offset=1)
    pair_features = masks[:, pair_indices[0]] * masks[:, pair_indices[1]]
    return torch.cat((masks, pair_features), dim=1)


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


def _fit_ridge_model(
    degree: int,
    train_masks: torch.Tensor,
    train_targets: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_targets: torch.Tensor,
    ridge_values: tuple[float, ...],
) -> dict[str, Any]:
    train_features = _polynomial_features(train_masks, degree)
    validation_features = _polynomial_features(validation_masks, degree)
    feature_mean = train_features.mean(dim=0)
    centered_train = train_features - feature_mean
    candidates = []
    for ridge in ridge_values:
        coefficients = _ridge_coefficients(centered_train, train_targets, ridge)
        if not bool(torch.isfinite(coefficients).all()):
            raise RuntimeError(f"ridge {ridge} produced non-finite coefficients")
        validation_prediction = (validation_features - feature_mean) @ coefficients
        validation_mse = (validation_prediction - validation_targets).square().mean()
        if not bool(torch.isfinite(validation_mse)):
            raise RuntimeError(f"ridge {ridge} produced non-finite validation loss")
        candidates.append(
            {
                "ridge": ridge,
                "standardized_validation_mse": float(validation_mse),
                "coefficients": coefficients,
            }
        )
    selected = min(
        candidates,
        key=lambda record: (
            record["standardized_validation_mse"],
            record["ridge"],
        ),
    )
    return {
        "selected_ridge": selected["ridge"],
        "validation_standardized_mse": selected["standardized_validation_mse"],
        "ridge_records": [
            {
                "ridge": record["ridge"],
                "standardized_validation_mse": record["standardized_validation_mse"],
            }
            for record in candidates
        ],
        "state": {
            "feature_mean": feature_mean.clone(),
            "coefficients": selected["coefficients"].clone(),
        },
        "feature_count": train_features.shape[1],
        "numerical_status": "finite",
    }


def _neural_specs(
    component_count: int, output_count: int, orders: dict[str, torch.Tensor]
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "tt_natural_rank1",
            "family": "tensor_train",
            "rank": 1,
            "component_order": orders["natural"],
        },
        {
            "name": "tt_natural_rank2",
            "family": "tensor_train",
            "rank": 2,
            "component_order": orders["natural"],
        },
        {
            "name": "tt_natural_rank4",
            "family": "tensor_train",
            "rank": 4,
            "component_order": orders["natural"],
        },
        {"name": "cp_rank4", "family": "cp", "rank": 4},
        {"name": "mlp_hidden7", "family": "mlp", "hidden_width": 7},
        {
            "name": "tt_reverse_rank2",
            "family": "tensor_train",
            "rank": 2,
            "component_order": orders["reverse"],
        },
        {
            "name": "tt_random_rank2",
            "family": "tensor_train",
            "rank": 2,
            "component_order": orders["random"],
        },
    )


def _build_neural_model(
    spec: dict[str, Any], component_count: int, output_count: int
) -> torch.nn.Module:
    if spec["family"] == "tensor_train":
        return BinaryTensorTrain(
            component_count=component_count,
            output_count=output_count,
            rank=spec["rank"],
            component_order=spec["component_order"],
        )
    if spec["family"] == "cp":
        return BinaryCPResponse(
            component_count=component_count,
            output_count=output_count,
            rank=spec["rank"],
        )
    if spec["family"] == "mlp":
        return BinaryMLPResponse(
            component_count=component_count,
            output_count=output_count,
            hidden_width=spec["hidden_width"],
        )
    raise ValueError(f"unknown neural family {spec['family']}")


def fit_source_models(
    train_masks: torch.Tensor,
    train_responses: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_responses: torch.Tensor,
    *,
    recipe: SourceFitRecipe = DEFAULT_SOURCE_FIT_RECIPE,
) -> dict[str, Any]:
    """Fit and serialize the registration-ready E29.8 source-model manifest."""
    (
        train_masks,
        train_responses,
        validation_masks,
        validation_responses,
    ) = _validated_cpu_data(
        train_masks,
        train_responses,
        validation_masks,
        validation_responses,
    )
    component_count = train_masks.shape[1]
    output_count = train_responses.shape[1]
    response_mean = train_responses.mean(dim=0)
    response_scale = train_responses.std(dim=0).clamp_min(1e-6)
    train_targets = (train_responses - response_mean) / response_scale
    validation_targets = (validation_responses - response_mean) / response_scale
    random_generator = torch.Generator().manual_seed(recipe.random_order_seed)
    orders = {
        "natural": torch.arange(component_count),
        "reverse": torch.arange(component_count - 1, -1, -1),
        "random": torch.randperm(component_count, generator=random_generator),
    }

    methods: dict[str, dict[str, Any]] = {}
    for spec in _neural_specs(component_count, output_count, orders):
        fitted = _fit_neural_model(
            lambda spec=spec: _build_neural_model(spec, component_count, output_count),
            train_masks,
            train_targets,
            validation_masks,
            validation_targets,
            recipe,
        )
        representative = _build_neural_model(spec, component_count, output_count)
        method_record = {
            "family": spec["family"],
            "parameter_count": sum(
                parameter.numel() for parameter in representative.parameters()
            ),
            "training_row_count": len(train_masks),
            "validation_row_count": len(validation_masks),
            "refit_on_validation": False,
            "selection_objective": "joint_standardized_validation_mse",
            **fitted,
        }
        if "rank" in spec:
            method_record["rank"] = spec["rank"]
        if "hidden_width" in spec:
            method_record["hidden_width"] = spec["hidden_width"]
        if "component_order" in spec:
            method_record["component_order"] = spec["component_order"].tolist()
        methods[spec["name"]] = method_record

    for name, degree in (("additive", 1), ("pairwise", 2)):
        fitted = _fit_ridge_model(
            degree,
            train_masks,
            train_targets,
            validation_masks,
            validation_targets,
            recipe.ridge_values,
        )
        feature_count = fitted.pop("feature_count")
        methods[name] = {
            "family": "polynomial_ridge",
            "degree": degree,
            "parameter_count": (feature_count + 1) * output_count,
            "training_row_count": len(train_masks),
            "validation_row_count": len(validation_masks),
            "refit_on_validation": False,
            "selection_objective": "joint_standardized_validation_mse",
            **fitted,
        }

    return {
        "format_version": 1,
        "component_count": component_count,
        "output_count": output_count,
        "method_manifest": SOURCE_METHODS,
        "recipe": recipe.to_metadata(),
        "normalization": {
            "response_mean": response_mean.clone(),
            "response_scale": response_scale.clone(),
            "source": "training_responses_only",
            "fit_rows": len(train_masks),
        },
        "orders": {name: order.clone() for name, order in orders.items()},
        "methods": {name: methods[name] for name in SOURCE_METHODS},
    }


def _validate_artifact_masks(
    artifact: dict[str, Any], method_name: str, masks: torch.Tensor
) -> torch.Tensor:
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported source-model artifact format")
    if method_name not in artifact["methods"]:
        raise KeyError(f"unknown source method {method_name}")
    component_count = artifact["component_count"]
    if masks.ndim != 2 or masks.shape[1] != component_count:
        raise ValueError(f"masks must have shape (rows, {component_count})")
    if not bool(torch.logical_or(masks == 0, masks == 1).all()):
        raise ValueError("masks must be binary")
    return masks.detach().cpu().double()


def _restore_neural_model(
    artifact: dict[str, Any], method_name: str
) -> torch.nn.Module:
    method = artifact["methods"][method_name]
    spec: dict[str, Any] = {
        "family": method["family"],
    }
    if "rank" in method:
        spec["rank"] = method["rank"]
    if "hidden_width" in method:
        spec["hidden_width"] = method["hidden_width"]
    if "component_order" in method:
        spec["component_order"] = torch.tensor(method["component_order"])
    model = _build_neural_model(
        spec,
        artifact["component_count"],
        artifact["output_count"],
    )
    model.load_state_dict(method["state"])
    restored_state = model.state_dict()
    _require_finite_state(restored_state, "restored_state")
    if "component_order" in method and not torch.equal(
        restored_state["component_order"],
        torch.tensor(method["component_order"], dtype=torch.long),
    ):
        raise ValueError(
            "restored tensor-train component order does not match metadata"
        )
    model.eval()
    return model


def frozen_latent_state(
    artifact: dict[str, Any], method_name: str, masks: torch.Tensor
) -> torch.Tensor:
    """Evaluate a selected frozen feature map without its source response head."""
    masks = _validate_artifact_masks(artifact, method_name, masks)
    method = artifact["methods"][method_name]
    if method["family"] == "polynomial_ridge":
        return _polynomial_features(masks, method["degree"])
    model = _restore_neural_model(artifact, method_name)
    with torch.no_grad():
        return model.latent_state(masks.float()).double()


def predict_source_response(
    artifact: dict[str, Any], method_name: str, masks: torch.Tensor
) -> torch.Tensor:
    """Evaluate the selected source head in the original response units."""
    masks = _validate_artifact_masks(artifact, method_name, masks)
    method = artifact["methods"][method_name]
    if method["family"] == "polynomial_ridge":
        features = _polynomial_features(masks, method["degree"])
        standardized = (features - method["state"]["feature_mean"]) @ method["state"][
            "coefficients"
        ]
    else:
        model = _restore_neural_model(artifact, method_name)
        with torch.no_grad():
            standardized = model(masks.float()).double()
    normalization = artifact["normalization"]
    return (
        standardized * normalization["response_scale"] + normalization["response_mean"]
    )


__all__ = [
    "DEFAULT_SOURCE_FIT_RECIPE",
    "SOURCE_METHODS",
    "SourceFitRecipe",
    "fit_source_models",
    "frozen_latent_state",
    "predict_source_response",
]
