"""General response models and response-blind masks for the E29.8 scale test.

The model classes deliberately separate a learned Boolean feature map from its
output head.  That distinction lets the experiment report both source-model
parameter counts and the operational cost of fitting a new affine readout.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch

from experiments.e29_gpt2_stable_head_interactions.tensor_train_response import (
    BinaryTensorTrain,
    count_tensor_train_parameters,
    mobius_latent_state,
)

E29_8_COMPONENT_COUNT = 48


def count_affine_readout_parameters(feature_count: int, output_count: int) -> int:
    """Return weights plus bias for a feature-to-response affine map."""
    if feature_count < 1 or output_count < 1:
        raise ValueError("readout dimensions must be positive")
    return feature_count * output_count + output_count


def count_tensor_train_feature_parameters(component_count: int, rank: int) -> int:
    """Count TT cores without the source-context output head."""
    if component_count < 2 or rank < 1:
        raise ValueError("invalid tensor-train dimensions")
    return 2 * rank + (component_count - 1) * 2 * rank * rank


def count_cp_feature_parameters(component_count: int, rank: int) -> int:
    """Count the binary CP factors without their output head."""
    if component_count < 1 or rank < 1:
        raise ValueError("invalid CP dimensions")
    return component_count * 2 * rank


def count_cp_response_parameters(
    component_count: int, output_count: int, rank: int
) -> int:
    """Count binary CP factors and the bias-free source response head."""
    if output_count < 1:
        raise ValueError("output_count must be positive")
    return count_cp_feature_parameters(component_count, rank) + rank * output_count


def count_mlp_response_parameters(
    component_count: int, output_count: int, hidden_width: int
) -> int:
    """Count both affine layers of the one-hidden-layer MLP control."""
    if component_count < 1 or output_count < 1 or hidden_width < 1:
        raise ValueError("invalid MLP dimensions")
    return (
        component_count * hidden_width
        + hidden_width
        + hidden_width * output_count
        + output_count
    )


def _validate_binary_masks(masks: torch.Tensor, component_count: int) -> None:
    if masks.ndim != 2 or masks.shape[1] != component_count:
        raise ValueError(f"masks must have shape (rows, {component_count})")
    if not bool(torch.logical_or(masks == 0, masks == 1).all()):
        raise ValueError("masks must be binary")


class BinaryCPResponse(torch.nn.Module):
    """A dimension-general CP factorization of a Boolean response tensor."""

    def __init__(self, *, component_count: int, output_count: int, rank: int) -> None:
        super().__init__()
        if component_count < 1 or output_count < 1 or rank < 1:
            raise ValueError("invalid CP dimensions")
        self.component_count = component_count
        self.output_count = output_count
        self.rank = rank
        self.factors = torch.nn.Parameter(torch.empty(component_count, 2, rank))
        self.output = torch.nn.Parameter(torch.empty(rank, output_count))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start products near one while breaking factor symmetry."""
        torch.nn.init.normal_(self.factors, mean=1.0, std=0.05)
        torch.nn.init.normal_(self.output, mean=0.0, std=0.1)

    def latent_state(self, masks: torch.Tensor) -> torch.Tensor:
        """Select one factor per component and multiply componentwise."""
        _validate_binary_masks(masks, self.component_count)
        component_indices = torch.arange(self.component_count, device=masks.device)[
            None
        ]
        selected = self.factors[component_indices, masks.long()]
        return selected.prod(dim=1)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        return self.latent_state(masks) @ self.output


class BinaryMLPResponse(torch.nn.Module):
    """A dimension-general one-hidden-layer nonlinear response control."""

    def __init__(
        self, *, component_count: int, output_count: int, hidden_width: int
    ) -> None:
        super().__init__()
        if component_count < 1 or output_count < 1 or hidden_width < 1:
            raise ValueError("invalid MLP dimensions")
        self.component_count = component_count
        self.output_count = output_count
        self.hidden_width = hidden_width
        self.input = torch.nn.Linear(component_count, hidden_width)
        self.activation = torch.nn.GELU()
        self.output = torch.nn.Linear(hidden_width, output_count)

    def latent_state(self, masks: torch.Tensor) -> torch.Tensor:
        """Return hidden activations used as frozen transfer features."""
        _validate_binary_masks(masks, self.component_count)
        return self.activation(self.input(masks.to(dtype=self.input.weight.dtype)))

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        return self.output(self.latent_state(masks))


def _masks_from_subsets(
    subsets: Sequence[Sequence[int]], component_count: int
) -> torch.Tensor:
    masks = torch.zeros(len(subsets), component_count, dtype=torch.float64)
    for row_index, subset in enumerate(subsets):
        masks[row_index, list(subset)] = 1.0
    return masks


def _sample_unique_subsets(
    *,
    component_count: int,
    cardinality: int,
    count: int,
    generator: random.Random,
) -> list[tuple[int, ...]]:
    """Sample fixed-cardinality subsets without enumerating a huge pool."""
    selected: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(selected) < count:
        subset = tuple(sorted(generator.sample(range(component_count), cardinality)))
        if subset not in seen:
            seen.add(subset)
            selected.append(subset)
    return selected


def _sample_balanced_subsets(
    *,
    component_count: int,
    cardinality: int,
    count: int,
    coverage: list[int],
    generator: random.Random,
) -> list[tuple[int, ...]]:
    """Greedily balance component incidence, randomizing every tied choice."""
    selected: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(selected) < count:
        randomized_components = list(range(component_count))
        generator.shuffle(randomized_components)
        randomized_components.sort(key=coverage.__getitem__)
        subset = tuple(sorted(randomized_components[:cardinality]))
        if subset in seen:
            # A collision is rare. Random sampling retains uniqueness without
            # changing the coverage guarantee already established by balancing.
            subset = tuple(
                sorted(generator.sample(range(component_count), cardinality))
            )
            if subset in seen:
                continue
        seen.add(subset)
        selected.append(subset)
        for component_index in subset:
            coverage[component_index] += 1
    return selected


def _require_e29_8_width(component_count: int) -> None:
    if component_count != E29_8_COMPONENT_COUNT:
        raise ValueError("the E29.8 mask protocol is pinned to 48 components")


def build_source_mask_protocol(
    *, seed: int, component_count: int = E29_8_COMPONENT_COUNT
) -> dict[str, torch.Tensor]:
    """Build the response-blind source-fit split for the 48-component model."""
    _require_e29_8_width(component_count)
    generator = random.Random(seed)
    coverage = [1] * component_count
    low_subsets: list[tuple[int, ...]] = [
        (component_index,) for component_index in range(component_count)
    ]
    low_subsets.extend(
        _sample_balanced_subsets(
            component_count=component_count,
            cardinality=2,
            count=72,
            coverage=coverage,
            generator=generator,
        )
    )
    low_subsets.extend(
        _sample_balanced_subsets(
            component_count=component_count,
            cardinality=3,
            count=72,
            coverage=coverage,
            generator=generator,
        )
    )
    generator.shuffle(low_subsets)

    size8 = _sample_unique_subsets(
        component_count=component_count,
        cardinality=8,
        count=192,
        generator=generator,
    )
    size16 = _sample_unique_subsets(
        component_count=component_count,
        cardinality=16,
        count=160,
        generator=generator,
    )
    return {
        "train_low": _masks_from_subsets(low_subsets, component_count),
        "train_size8": _masks_from_subsets(size8[:128], component_count),
        "train_size16": _masks_from_subsets(size16[:64], component_count),
        "validation_size8": _masks_from_subsets(size8[128:160], component_count),
        "validation_size16": _masks_from_subsets(size16[64:96], component_count),
        "test_size8": _masks_from_subsets(size8[160:], component_count),
        "test_size16": _masks_from_subsets(size16[96:], component_count),
    }


def build_transfer_mask_protocol(
    *, seed: int, component_count: int = E29_8_COMPONENT_COUNT
) -> dict[str, torch.Tensor]:
    """Build calibration and held-out masks for fresh-context transfer."""
    _require_e29_8_width(component_count)
    generator = random.Random(seed)
    singleton_components = list(range(component_count))
    coverage = [1] * component_count
    calibration_subsets: list[tuple[int, ...]] = [
        (component_index,) for component_index in singleton_components
    ]
    calibration_subsets.extend(
        _sample_balanced_subsets(
            component_count=component_count,
            cardinality=2,
            count=24,
            coverage=coverage,
            generator=generator,
        )
    )
    calibration_subsets.extend(
        _sample_balanced_subsets(
            component_count=component_count,
            cardinality=3,
            count=24,
            coverage=coverage,
            generator=generator,
        )
    )
    if min(coverage) < 1:
        raise RuntimeError("balanced transfer calibration missed a component")
    generator.shuffle(calibration_subsets)
    validation = _sample_unique_subsets(
        component_count=component_count,
        cardinality=8,
        count=32,
        generator=generator,
    )
    test = _sample_unique_subsets(
        component_count=component_count,
        cardinality=16,
        count=64,
        generator=generator,
    )
    return {
        "calibration": _masks_from_subsets(calibration_subsets, component_count),
        "validation_size8": _masks_from_subsets(validation, component_count),
        "test_size16": _masks_from_subsets(test, component_count),
    }


__all__ = [
    "BinaryCPResponse",
    "BinaryMLPResponse",
    "BinaryTensorTrain",
    "build_source_mask_protocol",
    "build_transfer_mask_protocol",
    "count_affine_readout_parameters",
    "count_cp_feature_parameters",
    "count_cp_response_parameters",
    "count_mlp_response_parameters",
    "count_tensor_train_feature_parameters",
    "count_tensor_train_parameters",
    "mobius_latent_state",
]
