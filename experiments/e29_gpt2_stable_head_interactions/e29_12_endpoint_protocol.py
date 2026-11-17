"""Pure response-blind helpers for prospective E29.12 endpoint confirmation.

The eventual acquisition may fit new per-context readouts on the parent
low/c8/c16 masks, but this module deliberately knows nothing about prompts or
fit responses.  It only constructs fresh held masks, applies a clean-logit
endpoint proxy to already-frozen prediction tables, and scores held outcomes.
It performs no transformer call and reads no scientific artifact.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_8_scaleup_protocol import (
    response_metrics,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    TEMPLATE_CONTEXTS,
)
from experiments.e29_gpt2_stable_head_interactions.e29_12_fitting import (
    DEVELOPMENT_EXPONENT,
    METHOD_MANIFEST,
)

COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
HELD_MASK_COUNT = 64

ENDPOINT_MASK_SEEDS = {
    "confirmation_size24": 31_241,
    "confirmation_size32": 31_242,
}
ENDPOINT_MASK_SHAPES = {
    "confirmation_size24": (HELD_MASK_COUNT, COMPONENT_COUNT),
    "confirmation_size32": (HELD_MASK_COUNT, COMPONENT_COUNT),
}
ENDPOINT_CARDINALITIES = {
    "confirmation_size24": 24,
    "confirmation_size32": 32,
}

MLP7_UNBOUNDED_METHOD = METHOD_MANIFEST[0]
ADDITIVE_UNBOUNDED_METHOD = METHOD_MANIFEST[1]
MLP7_CAPPED_METHOD = METHOD_MANIFEST[2]
ADDITIVE_CAPPED_METHOD = METHOD_MANIFEST[3]
MLP7_REANCHORED_METHOD = METHOD_MANIFEST[4]
ADDITIVE_REANCHORED_METHOD = METHOD_MANIFEST[5]
ENDPOINT_METHODS = METHOD_MANIFEST

# c24 is a preservation check because the parent predictor is already strong
# there.  c32 is the decisive endpoint-repair test.
ENDPOINT_SCALE_GATES = {
    "confirmation_size24": {
        "pooled_r2_min": 0.70,
        "median_context_r2_min": 0.65,
        "positive_context_r2_min_count": 11,
        "minimum_context_r2_min": 0.0,
        "capped_over_unbounded_mse_max": 1.05,
        "paired_bootstrap_required": False,
    },
    "confirmation_size32": {
        "pooled_r2_min": 0.40,
        "median_context_r2_min": 0.35,
        "positive_context_r2_min_count": 10,
        "minimum_context_r2_min": -0.5,
        "capped_over_unbounded_mse_max": 0.50,
        "paired_bootstrap_required": True,
    },
}
BOOTSTRAP_SEED = 31_243
BOOTSTRAP_REPLICATES = 5_000


def _mask_rows(masks: torch.Tensor) -> list[tuple[int, ...]]:
    return [
        tuple(torch.nonzero(row, as_tuple=False).flatten().tolist()) for row in masks
    ]


def _prior_rows_by_cardinality(
    prior_masks: Sequence[torch.Tensor],
) -> dict[int, set[tuple[int, ...]]]:
    if not prior_masks:
        raise ValueError("prior mask closure must be nonempty")
    collected: dict[int, set[tuple[int, ...]]] = {}
    for masks in prior_masks:
        if not isinstance(masks, torch.Tensor):
            raise TypeError("prior mask closure contains a non-tensor")
        masks = masks.detach().cpu()
        if masks.ndim != 2 or masks.shape[1] != COMPONENT_COUNT or not len(masks):
            raise ValueError("prior masks must have shape (rows, 48)")
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise ValueError("prior masks must be binary")
        for row in _mask_rows(masks):
            if not row:
                raise ValueError("prior mask closure contains an empty mask")
            collected.setdefault(len(row), set()).add(row)
    return collected


def _balanced_rows(
    *,
    cardinality: int,
    count: int,
    excluded: set[tuple[int, ...]],
    seed: int,
) -> list[tuple[int, ...]]:
    """Generate unique degree-balanced rows, then randomize by valid switches."""
    generator = random.Random(seed)
    selected: list[tuple[int, ...]] = []
    seen = set(excluded)
    coverage = [0] * COMPONENT_COUNT
    while len(selected) < count:
        for _ in range(10_000):
            components = list(range(COMPONENT_COUNT))
            generator.shuffle(components)
            components.sort(key=coverage.__getitem__)
            row = tuple(sorted(components[:cardinality]))
            if row not in seen:
                break
        else:
            raise RuntimeError(
                f"could not construct fresh cardinality-{cardinality} masks"
            )
        seen.add(row)
        selected.append(row)
        for component in row:
            coverage[component] += 1
    if max(coverage) - min(coverage) > 1:
        raise RuntimeError("constructed masks do not have balanced incidence")

    # The greedy construction is degree-balanced but overly regular.  Switches
    # preserve every row sum and column incidence while randomizing the design.
    row_sets = [set(row) for row in selected]
    selected_set = set(selected)
    for _ in range(max(10_000, count * 200)):
        left_index, right_index = generator.sample(range(count), 2)
        left = row_sets[left_index]
        right = row_sets[right_index]
        left_only = tuple(left - right)
        right_only = tuple(right - left)
        if not left_only or not right_only:
            continue
        remove_left = generator.choice(left_only)
        remove_right = generator.choice(right_only)
        new_left = tuple(sorted(left - {remove_left} | {remove_right}))
        new_right = tuple(sorted(right - {remove_right} | {remove_left}))
        old_left = tuple(sorted(left))
        old_right = tuple(sorted(right))
        occupied = selected_set - {old_left, old_right}
        if (
            new_left in excluded
            or new_right in excluded
            or new_left == new_right
            or new_left in occupied
            or new_right in occupied
        ):
            continue
        selected_set = occupied | {new_left, new_right}
        row_sets[left_index] = set(new_left)
        row_sets[right_index] = set(new_right)

    rows = [tuple(sorted(row)) for row in row_sets]
    if cardinality * 2 == COMPONENT_COUNT:
        row_set = set(rows)
        for row in rows:
            complement = tuple(
                component
                for component in range(COMPONENT_COUNT)
                if component not in row
            )
            if complement in row_set:
                raise RuntimeError("balanced half-cardinality masks contain a pair")
    return rows


def _complement_rows(rows: Sequence[tuple[int, ...]]) -> set[tuple[int, ...]]:
    return {
        tuple(
            component
            for component in range(COMPONENT_COUNT)
            if component not in row
        )
        for row in rows
    }


def _rows_to_masks(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    masks = torch.zeros((len(rows), COMPONENT_COUNT), dtype=torch.float64)
    for index, row in enumerate(rows):
        masks[index, list(row)] = 1.0
    return masks


def build_endpoint_masks(
    prior_masks: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build the two deterministic panels fresh from a supplied prior closure.

    The registration layer must supply every earlier scientific mask tensor.
    Freshness is therefore explicit rather than inferred from the filesystem.
    """
    prior = _prior_rows_by_cardinality(prior_masks)
    masks = {}
    for split in ENDPOINT_MASK_SHAPES:
        cardinality = ENDPOINT_CARDINALITIES[split]
        excluded = set(prior.get(cardinality, set()))
        excluded.update(
            _complement_rows(tuple(prior.get(COMPONENT_COUNT - cardinality, set())))
        )
        masks[split] = _rows_to_masks(
            _balanced_rows(
                cardinality=cardinality,
                count=HELD_MASK_COUNT,
                excluded=excluded,
                seed=ENDPOINT_MASK_SEEDS[split],
            )
        )
    return validate_endpoint_masks(masks, prior_masks=prior_masks)


def validate_endpoint_masks(
    masks: Mapping[str, torch.Tensor],
    *,
    prior_masks: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validate exact shape, balance, uniqueness, and prior-panel freshness."""
    if tuple(masks) != tuple(ENDPOINT_MASK_SHAPES):
        raise ValueError("endpoint mask split order or keys changed")
    prior = _prior_rows_by_cardinality(prior_masks)
    validated: dict[str, torch.Tensor] = {}
    for split, expected_shape in ENDPOINT_MASK_SHAPES.items():
        value = masks[split].detach().cpu().double()
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{split} must have shape {expected_shape}")
        if not bool(torch.logical_or(value == 0, value == 1).all()):
            raise ValueError(f"{split} masks must be binary")
        cardinality = ENDPOINT_CARDINALITIES[split]
        if set(value.sum(dim=1).tolist()) != {float(cardinality)}:
            raise ValueError(f"{split} has the wrong row cardinality")
        rows = _mask_rows(value)
        row_set = set(rows)
        if len(row_set) != len(rows):
            raise ValueError(f"{split} contains a duplicate mask")
        if not row_set.isdisjoint(prior.get(cardinality, set())):
            raise ValueError(f"{split} overlaps a prior mask")
        prior_complements = _complement_rows(
            tuple(prior.get(COMPONENT_COUNT - cardinality, set()))
        )
        if not row_set.isdisjoint(prior_complements):
            raise ValueError(f"{split} is the complement of a prior mask")
        incidence = value.sum(dim=0)
        if float(incidence.max() - incidence.min()) > 1.0:
            raise ValueError(f"{split} component incidence is not balanced")
        if cardinality * 2 == COMPONENT_COUNT:
            for row in rows:
                complement = tuple(
                    component
                    for component in range(COMPONENT_COUNT)
                    if component not in row
                )
                if complement in row_set:
                    raise ValueError(f"{split} contains complementary masks")
        validated[split] = value
    return validated


def _context_vector(values: Sequence[float] | torch.Tensor, *, label: str) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float64).detach().cpu().clone()
    if tuple(vector.shape) != (CONTEXT_COUNT,):
        raise ValueError(f"{label} must have shape (12,)")
    if not bool(vector.isfinite().all()):
        raise ValueError(f"{label} must be finite")
    return vector


def _prediction_table(predictions: torch.Tensor, *, label: str) -> torch.Tensor:
    value = torch.as_tensor(predictions, dtype=torch.float64).detach().cpu().clone()
    if value.ndim != 2 or value.shape[1] != CONTEXT_COUNT or not len(value):
        raise ValueError(f"{label} must have shape (rows, 12)")
    if not bool(value.isfinite().all()):
        raise ValueError(f"{label} must be finite")
    return value


def natural_endpoint_caps(
    clean_base_correct_minus_donor_margins: Sequence[float] | torch.Tensor,
    clean_donor_correct_minus_base_margins: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Return the exact clean base-to-donor response distance per context."""
    base = _context_vector(
        clean_base_correct_minus_donor_margins,
        label="clean base correct-minus-donor margins",
    )
    donor = _context_vector(
        clean_donor_correct_minus_base_margins,
        label="clean donor correct-minus-base margins",
    )
    if not bool((base > 0).all()) or not bool((donor > 0).all()):
        raise ValueError("both clean correctness margins must be strictly positive")
    endpoints = base + donor
    if not bool(endpoints.isfinite().all()):
        raise ValueError("natural endpoint caps must be finite")
    return endpoints


def upper_cap_predictions(
    predictions: torch.Tensor,
    endpoint_caps: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Apply only the registered natural upper bound; do not floor responses."""
    table = _prediction_table(predictions, label="predictions")
    caps = _context_vector(endpoint_caps, label="endpoint caps")
    if not bool((caps > 0).all()):
        raise ValueError("endpoint caps must be strictly positive")
    return torch.minimum(table, caps[None, :])


def endpoint_prediction_family(
    mlp7_predictions: torch.Tensor,
    additive_predictions: torch.Tensor,
    density_conditioned_additive_predictions: torch.Tensor,
    development_scale_additive_predictions: torch.Tensor,
    endpoint_caps: Sequence[float] | torch.Tensor,
    *,
    split: str,
) -> dict[str, torch.Tensor]:
    """Freeze the exact eight-method family already defined by the pure fitter."""
    if split not in ENDPOINT_CARDINALITIES:
        raise ValueError(f"unknown endpoint split: {split}")
    mlp7 = _prediction_table(mlp7_predictions, label="MLP7 predictions")
    additive = _prediction_table(additive_predictions, label="additive predictions")
    density_additive = _prediction_table(
        density_conditioned_additive_predictions,
        label="density-conditioned additive predictions",
    )
    development_additive = _prediction_table(
        development_scale_additive_predictions,
        label="development-scale additive predictions",
    )
    if (
        mlp7.shape != additive.shape
        or mlp7.shape != density_additive.shape
        or mlp7.shape != development_additive.shape
    ):
        raise ValueError("all prediction tables must have equal shape")
    caps = _context_vector(endpoint_caps, label="endpoint caps")
    if not bool((caps > 0).all()):
        raise ValueError("endpoint caps must be strictly positive")
    cardinality = ENDPOINT_CARDINALITIES[split]
    target_mean = caps * (
        1.0
        - (1.0 - cardinality / COMPONENT_COUNT) ** DEVELOPMENT_EXPONENT
    )

    def reanchor(predictions: torch.Tensor) -> torch.Tensor:
        return (
            predictions
            - predictions.mean(dim=0, keepdim=True)
            + target_mean[None, :]
        )

    return {
        MLP7_UNBOUNDED_METHOD: mlp7,
        ADDITIVE_UNBOUNDED_METHOD: additive,
        MLP7_CAPPED_METHOD: upper_cap_predictions(mlp7, caps),
        ADDITIVE_CAPPED_METHOD: upper_cap_predictions(additive, caps),
        MLP7_REANCHORED_METHOD: reanchor(mlp7),
        ADDITIVE_REANCHORED_METHOD: reanchor(additive),
        METHOD_MANIFEST[6]: density_additive,
        METHOD_MANIFEST[7]: development_additive,
    }


def _held_tables(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = _prediction_table(labels, label="held labels")
    predictions = _prediction_table(predictions, label="held predictions")
    expected_shape = (HELD_MASK_COUNT, CONTEXT_COUNT)
    if tuple(labels.shape) != expected_shape or tuple(predictions.shape) != expected_shape:
        raise ValueError(f"held tables must have shape {expected_shape}")
    if labels.shape != predictions.shape:
        raise ValueError("held labels and predictions must have equal shape")
    centered = labels - labels.mean(dim=0, keepdim=True)
    if not bool((centered.square().sum(dim=0) > 0).all()):
        raise ValueError("every held context must have response variation")
    return labels, predictions


def endpoint_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, Any]:
    """Return pooled, per-context, and fixed-template held metrics."""
    labels, predictions = _held_tables(labels, predictions)
    return {
        "overall": response_metrics(labels, predictions),
        "by_template": {
            name: response_metrics(labels[:, indices], predictions[:, indices])
            for name, indices in TEMPLATE_CONTEXTS.items()
        },
    }


def paired_stratified_bootstrap_mse_ratio(
    labels: torch.Tensor,
    numerator_predictions: torch.Tensor,
    denominator_predictions: torch.Tensor,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int | str]:
    """Resample masks and contexts within the four fixed template strata."""
    labels, numerator = _held_tables(labels, numerator_predictions)
    denominator = _prediction_table(
        denominator_predictions,
        label="bootstrap denominator predictions",
    )
    if denominator.shape != labels.shape:
        raise ValueError("paired bootstrap tables must have identical shapes")
    if replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    numerator_error = (labels - numerator).square()
    denominator_error = (labels - denominator).square()
    denominator_mse = float(denominator_error.mean())
    if denominator_mse <= 0:
        raise ValueError("bootstrap denominator MSE must be positive")

    generator = torch.Generator().manual_seed(seed)
    ratios = torch.empty(replicates, dtype=torch.float64)
    strata = tuple(TEMPLATE_CONTEXTS.values())
    for replicate in range(replicates):
        row_indices = torch.randint(
            HELD_MASK_COUNT,
            (HELD_MASK_COUNT,),
            generator=generator,
        )
        context_indices = torch.cat(
            [
                torch.tensor(stratum, dtype=torch.long)[
                    torch.randint(len(stratum), (len(stratum),), generator=generator)
                ]
                for stratum in strata
            ]
        )
        numerator_mse = numerator_error[row_indices][:, context_indices].mean()
        denominator_sample_mse = denominator_error[row_indices][
            :, context_indices
        ].mean()
        ratios[replicate] = numerator_mse / denominator_sample_mse.clamp_min(1e-18)
    return {
        "point_estimate": float(numerator_error.mean()) / denominator_mse,
        "lower_95": float(torch.quantile(ratios, 0.025)),
        "upper_95": float(torch.quantile(ratios, 0.975)),
        "probability_ratio_below_one": float((ratios < 1).double().mean()),
        "seed": seed,
        "replicates": replicates,
        "resampling": "masks_and_contexts_within_four_template_strata",
    }


def _clean_margin_gate(
    values: Sequence[float] | torch.Tensor,
    *,
    label: str,
) -> bool:
    margins = _context_vector(values, label=label)
    return bool((margins > 0).all())


def endpoint_gate_results(
    *,
    split: str,
    capped_metrics: Mapping[str, Any],
    capped_over_unbounded_bootstrap: Mapping[str, Any],
    clean_base_correct_minus_donor_margins: Sequence[float] | torch.Tensor,
    clean_donor_correct_minus_base_margins: Sequence[float] | torch.Tensor,
    numerical_integrity_passed: bool,
) -> dict[str, Any]:
    """Apply the registered preservation gate at c24 or repair gate at c32."""
    if split not in ENDPOINT_SCALE_GATES:
        raise ValueError(f"unknown endpoint split: {split}")
    if set(capped_metrics) != {"overall", "by_template"}:
        raise ValueError("capped metrics have unexpected keys")
    overall = capped_metrics["overall"]
    by_template = capped_metrics["by_template"]
    if set(by_template) != set(TEMPLATE_CONTEXTS):
        raise ValueError("template metric strata changed")
    context_r2 = [float(value) for value in overall["per_context_r2"]]
    if len(context_r2) != CONTEXT_COUNT:
        raise ValueError("endpoint gates require twelve context R2 values")
    point_ratio = float(capped_over_unbounded_bootstrap["point_estimate"])
    bootstrap_upper = float(capped_over_unbounded_bootstrap["upper_95"])
    if not torch.isfinite(torch.tensor([point_ratio, bootstrap_upper])).all():
        raise ValueError("endpoint comparison ratios must be finite")

    thresholds = ENDPOINT_SCALE_GATES[split]
    required_results = {
        "pooled_r2_passed": float(overall["r2"])
        >= thresholds["pooled_r2_min"],
        "median_context_r2_passed": float(overall["median_context_r2"])
        >= thresholds["median_context_r2_min"],
        "positive_context_count_passed": sum(value > 0 for value in context_r2)
        >= thresholds["positive_context_r2_min_count"],
        "minimum_context_r2_passed": float(overall["minimum_context_r2"])
        >= thresholds["minimum_context_r2_min"],
        "all_template_strata_positive": all(
            float(by_template[name]["r2"]) > 0 for name in TEMPLATE_CONTEXTS
        ),
        "mse_ratio_to_unbounded_passed": point_ratio
        <= thresholds["capped_over_unbounded_mse_max"],
        "all_clean_base_margins_positive": _clean_margin_gate(
            clean_base_correct_minus_donor_margins,
            label="clean base correct-minus-donor margins",
        ),
        "all_clean_donor_margins_positive": _clean_margin_gate(
            clean_donor_correct_minus_base_margins,
            label="clean donor correct-minus-base margins",
        ),
        "numerical_integrity_passed": bool(numerical_integrity_passed),
    }
    bootstrap_passed = bootstrap_upper < 1.0
    results = {
        **required_results,
        "paired_bootstrap_upper_below_one": bootstrap_passed,
    }
    all_passed = all(required_results.values()) and (
        bootstrap_passed if thresholds["paired_bootstrap_required"] else True
    )
    return {
        "split": split,
        "thresholds": dict(thresholds),
        "results": results,
        "paired_bootstrap_required": thresholds["paired_bootstrap_required"],
        "all_passed": all_passed,
    }


def bounded_mlp_vs_bounded_additive_results(
    *,
    bounded_mlp_metrics: Mapping[str, Any],
    bounded_additive_metrics: Mapping[str, Any],
    paired_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Report learned-feature superiority separately from absolute success."""
    mlp_mse = float(bounded_mlp_metrics["overall"]["mse"])
    additive_mse = float(bounded_additive_metrics["overall"]["mse"])
    bootstrap_upper = float(paired_bootstrap["upper_95"])
    results = {
        "bounded_mlp_mse_strictly_below_bounded_additive": mlp_mse < additive_mse,
        "paired_bootstrap_upper_below_one": bootstrap_upper < 1.0,
    }
    return {
        "results": results,
        "all_passed": all(results.values()),
        "required_for_absolute_endpoint_success": False,
    }


__all__ = [
    "ADDITIVE_CAPPED_METHOD",
    "ADDITIVE_REANCHORED_METHOD",
    "ADDITIVE_UNBOUNDED_METHOD",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "COMPONENT_COUNT",
    "CONTEXT_COUNT",
    "DEVELOPMENT_EXPONENT",
    "ENDPOINT_CARDINALITIES",
    "ENDPOINT_MASK_SEEDS",
    "ENDPOINT_MASK_SHAPES",
    "ENDPOINT_METHODS",
    "ENDPOINT_SCALE_GATES",
    "HELD_MASK_COUNT",
    "METHOD_MANIFEST",
    "MLP7_CAPPED_METHOD",
    "MLP7_REANCHORED_METHOD",
    "MLP7_UNBOUNDED_METHOD",
    "bounded_mlp_vs_bounded_additive_results",
    "build_endpoint_masks",
    "endpoint_gate_results",
    "endpoint_metrics",
    "endpoint_prediction_family",
    "natural_endpoint_caps",
    "paired_stratified_bootstrap_mse_ratio",
    "upper_cap_predictions",
    "validate_endpoint_masks",
]
