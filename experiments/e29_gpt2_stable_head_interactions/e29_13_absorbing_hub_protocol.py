"""Response-blind protocol primitives for prospective E29.13.

The module freezes the component identities, source-mask closure, held masks,
diagnostics, saturation constants, and query arithmetic.  It performs no file
I/O, transformer call, registration, acquisition, fitting, or scoring.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch

COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
COMPONENT_MAP = tuple(
    (layer, kind) for layer in range(24) for kind in ("attention", "mlp")
)
HUB_COMPONENT = 1
HUB_COMPONENT_SPEC = (0, "mlp")
CONTROL_COMPONENT = 0
CONTROL_COMPONENT_SPEC = (0, "attention")

SOURCE_INHERITED_SHAPES = {
    "calibration_low": (96, COMPONENT_COUNT),
    "calibration_size8": (32, COMPONENT_COUNT),
    "bridge_size16": (64, COMPONENT_COUNT),
}
SOURCE_STRATA = (
    *SOURCE_INHERITED_SHAPES,
    "all_singletons",
    "hub_incident_pairs",
    "control_incident_pairs",
)

HELD_MASK_SEEDS = {
    "confirmation_size24": 31_341,
    "confirmation_size32": 31_342,
}
REFERENCE_MASK_SEEDS = {
    "reference_size24": 31_343,
    "reference_size32": 31_344,
}
HELD_MASK_SHAPES = {
    "confirmation_size24": (48, COMPONENT_COUNT),
    "confirmation_size32": (48, COMPONENT_COUNT),
}
HELD_CARDINALITIES = {
    "confirmation_size24": 24,
    "confirmation_size32": 32,
}
HELD_HUB_ON_COUNTS = {
    "confirmation_size24": 24,
    "confirmation_size32": 32,
}
DIAGNOSTIC_MASK_SHAPES = {
    "all_components": (1, COMPONENT_COUNT),
    "all_minus_hub": (1, COMPONENT_COUNT),
}

SATURATION_CONSTANTS = {
    "a": 1.1823081383,
    "b": 3.5185046296,
    "p": 2.7800291320,
    "D_over_A": 0.9942213374,
    "reference_rows": 8_192,
}
BOOTSTRAP = {"seed": 31_345, "replicates": 5_000}
SCORING_GATES = {
    "primary_gate_shape_c48": {
        "pooled_r2_min": 0.80,
        "median_context_r2_min": 0.75,
        "positive_context_r2_min_count": 12,
        "all_template_r2_positive": True,
        "baseline_mse_ratio_max_exclusive": 1.0,
        "bootstrap_upper_95_max_exclusive_by_split": {
            "confirmation_size24": None,
            "confirmation_size32": 1.0,
        },
        "joint_two_scale_bootstrap_upper_95_max_exclusive": 1.0,
        "baselines": (
            "additive_endpoint_cap_ridge10",
            "mlp7_saturation_reanchored_c48",
        ),
    },
    "matched_nongated_superiority": {
        "primary_method": "gate_shape_c48_calibrated",
        "baseline_method": "capped_mlp_additive_ensemble_c48",
        "mse_ratio_max_exclusive": 1.0,
        "bootstrap_upper_95_max_exclusive_by_split": {
            "confirmation_size24": None,
            "confirmation_size32": 1.0,
        },
        "joint_two_scale_bootstrap_upper_95_max_exclusive": 1.0,
    },
    "hub_off_shape": {
        "primary_method": "gate_shape_c48_calibrated",
        "matched_baseline_method": "gate_only_c48_calibrated",
        "per_scale_mse_ratio_max_exclusive": 0.80,
        "joint_context_centered_r2_min_exclusive": 0.0,
        "joint_bootstrap_upper_95_max_exclusive": 1.0,
        "equal_cell_weight_per_scale": True,
    },
    "mechanistic_absorbing_hub": {
        "component_1_mean_accuracy_rank_max": 1,
        "pooled_ceiling_accuracy_min": 0.98,
        "pooled_ceiling_f1_min": 0.97,
        "hub_on_mean_response_over_clean_endpoint_min": 0.985,
        "hub_on_mean_response_over_clean_endpoint_max": 1.01,
        "ceiling_threshold_fraction_of_clean_endpoint": 0.99,
    },
    "clean_proxy_secondary": {
        "pooled_r2_min": 0.75,
        "positive_context_r2_min_count": 12,
        "mse_over_primary_max": 1.15,
    },
    "gate_only_secondary": {
        "pooled_r2_min": 0.75,
        "positive_context_r2_min_count": 12,
    },
}

SCHEMA_VERSION = "e29_13_absorbing_hub_response_blind_v1"


def _mask_rows(masks: torch.Tensor) -> list[tuple[int, ...]]:
    return [
        tuple(torch.nonzero(row, as_tuple=False).flatten().tolist()) for row in masks
    ]


def _rows_to_masks(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    masks = torch.zeros((len(rows), COMPONENT_COUNT), dtype=torch.float64)
    for row_index, row in enumerate(rows):
        masks[row_index, list(row)] = 1.0
    return masks


def _binary_panel(
    value: Any,
    *,
    label: str,
    shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    masks = value.detach().cpu().double().clone()
    if masks.ndim != 2 or masks.shape[1] != COMPONENT_COUNT:
        raise ValueError(f"{label} must have 48 columns")
    if shape is not None and tuple(masks.shape) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not len(masks):
        raise ValueError(f"{label} must be nonempty")
    if not bool(torch.logical_or(masks == 0.0, masks == 1.0).all()):
        raise ValueError(f"{label} must be binary")
    return masks


def _inherited_fit_panels(
    inherited_fit_masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(inherited_fit_masks, Mapping):
        raise TypeError("inherited fit masks must be a mapping")
    missing = set(SOURCE_INHERITED_SHAPES) - set(inherited_fit_masks)
    if missing:
        raise ValueError(f"inherited fit masks are missing {sorted(missing)}")
    panels = {
        split: _binary_panel(
            inherited_fit_masks[split],
            label=f"inherited {split}",
            shape=shape,
        )
        for split, shape in SOURCE_INHERITED_SHAPES.items()
    }
    expected_cardinalities = {
        "calibration_low": {1.0, 2.0, 3.0},
        "calibration_size8": {8.0},
        "bridge_size16": {16.0},
    }
    for split, panel in panels.items():
        if set(panel.sum(dim=1).tolist()) != expected_cardinalities[split]:
            raise ValueError(f"inherited {split} cardinalities changed")
        rows = _mask_rows(panel)
        if len(rows) != len(set(rows)):
            raise ValueError(f"inherited {split} contains a duplicate mask")
    return panels


def _construct_source_fit_protocol(
    inherited_fit_masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    inherited = _inherited_fit_panels(inherited_fit_masks)
    rows: list[tuple[int, ...]] = []
    row_to_index: dict[tuple[int, ...], int] = {}

    def indices_for(requested: Sequence[tuple[int, ...]]) -> torch.Tensor:
        indices = []
        for row in requested:
            if row not in row_to_index:
                row_to_index[row] = len(rows)
                rows.append(row)
            indices.append(row_to_index[row])
        return torch.tensor(indices, dtype=torch.long)

    strata: dict[str, torch.Tensor] = {}
    for split in SOURCE_INHERITED_SHAPES:
        strata[split] = indices_for(_mask_rows(inherited[split]))
    strata["all_singletons"] = indices_for(
        [(component,) for component in range(COMPONENT_COUNT)]
    )
    strata["hub_incident_pairs"] = indices_for(
        [
            tuple(sorted((HUB_COMPONENT, component)))
            for component in range(COMPONENT_COUNT)
            if component != HUB_COMPONENT
        ]
    )
    strata["control_incident_pairs"] = indices_for(
        [
            tuple(sorted((CONTROL_COMPONENT, component)))
            for component in range(COMPONENT_COUNT)
            if component != CONTROL_COMPONENT
        ]
    )
    return {"masks": _rows_to_masks(rows), "strata": strata}


def build_source_fit_protocol(
    inherited_fit_masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Return one de-duplicated source table plus overlapping stratum indices."""
    source = _construct_source_fit_protocol(inherited_fit_masks)
    return validate_source_fit_protocol(source, inherited_fit_masks)


def validate_source_fit_protocol(
    source: Mapping[str, Any],
    inherited_fit_masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Validate physical uniqueness and exact membership of every source stratum."""
    if not isinstance(source, Mapping) or tuple(source) != ("masks", "strata"):
        raise ValueError("source protocol keys or order changed")
    masks = _binary_panel(source["masks"], label="source masks")
    rows = _mask_rows(masks)
    if len(rows) != len(set(rows)):
        raise ValueError("source masks contain a physical duplicate")
    strata = source["strata"]
    if not isinstance(strata, Mapping) or tuple(strata) != SOURCE_STRATA:
        raise ValueError("source stratum keys or order changed")

    expected = _construct_source_fit_protocol(inherited_fit_masks)
    if not torch.equal(masks, expected["masks"]):
        raise ValueError("source de-duplicated union changed")
    validated_strata = {}
    for name in SOURCE_STRATA:
        indices = torch.as_tensor(strata[name], dtype=torch.long).detach().cpu().clone()
        if indices.ndim != 1 or not torch.equal(indices, expected["strata"][name]):
            raise ValueError(f"source stratum {name} changed")
        validated_strata[name] = indices
    return {"masks": masks, "strata": validated_strata}


def _prior_rows(prior_masks: Sequence[torch.Tensor]) -> set[tuple[int, ...]]:
    if not prior_masks:
        raise ValueError("prior mask closure must be nonempty")
    rows: set[tuple[int, ...]] = set()
    for panel_index, panel in enumerate(prior_masks):
        masks = _binary_panel(panel, label=f"prior masks {panel_index}")
        for row in _mask_rows(masks):
            if not row:
                raise ValueError("prior mask closure contains an empty mask")
            rows.add(row)
    return rows


def _complement(row: Sequence[int]) -> tuple[int, ...]:
    selected = set(row)
    return tuple(
        component for component in range(COMPONENT_COUNT) if component not in selected
    )


def _balanced_hub_stratum(
    *,
    cardinality: int,
    count: int,
    hub_on: bool,
    forbidden: set[tuple[int, ...]],
    seed: int,
) -> list[tuple[int, ...]]:
    """Construct a fixed-hub panel balanced over the other 47 components."""
    generator = random.Random(seed)
    other_components = [
        component for component in range(COMPONENT_COUNT) if component != HUB_COMPONENT
    ]
    selected_count = cardinality - int(hub_on)
    coverage = {component: 0 for component in other_components}
    rows: list[tuple[int, ...]] = []
    occupied = set(forbidden)
    while len(rows) < count:
        for _ in range(20_000):
            candidates = list(other_components)
            generator.shuffle(candidates)
            candidates.sort(key=coverage.__getitem__)
            selected = candidates[:selected_count]
            row = tuple(sorted([*selected, *([HUB_COMPONENT] if hub_on else [])]))
            if row not in occupied:
                break
        else:
            raise RuntimeError("could not construct a fresh balanced hub stratum")
        rows.append(row)
        occupied.add(row)
        for component in selected:
            coverage[component] += 1

    incidence = list(coverage.values())
    if max(incidence) - min(incidence) > 1:
        raise RuntimeError("hub stratum component incidence is not balanced")

    # Degree-preserving switches remove the greedy construction's ordering
    # regularity without changing row cardinality, hub state, or incidence.
    row_sets = [set(row) for row in rows]
    row_set = set(rows)
    for _ in range(max(10_000, count * 250)):
        left_index, right_index = generator.sample(range(count), 2)
        left = row_sets[left_index]
        right = row_sets[right_index]
        left_only = tuple((left - right) - {HUB_COMPONENT})
        right_only = tuple((right - left) - {HUB_COMPONENT})
        if not left_only or not right_only:
            continue
        remove_left = generator.choice(left_only)
        remove_right = generator.choice(right_only)
        new_left = tuple(sorted(left - {remove_left} | {remove_right}))
        new_right = tuple(sorted(right - {remove_right} | {remove_left}))
        old_left = tuple(sorted(left))
        old_right = tuple(sorted(right))
        remaining = row_set - {old_left, old_right}
        if (
            new_left in forbidden
            or new_right in forbidden
            or new_left == new_right
            or new_left in remaining
            or new_right in remaining
        ):
            continue
        row_set = remaining | {new_left, new_right}
        row_sets[left_index] = set(new_left)
        row_sets[right_index] = set(new_right)
    return [tuple(sorted(row)) for row in row_sets]


def build_held_masks(
    prior_masks: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build deterministic c24/c32 panels with fixed natural hub prevalence."""
    prior = _prior_rows(prior_masks)
    forbidden = prior | {_complement(row) for row in prior}
    held = {}
    for split, shape in HELD_MASK_SHAPES.items():
        cardinality = HELD_CARDINALITIES[split]
        hub_on_count = HELD_HUB_ON_COUNTS[split]
        seed = HELD_MASK_SEEDS[split]
        on_rows = _balanced_hub_stratum(
            cardinality=cardinality,
            count=hub_on_count,
            hub_on=True,
            forbidden=forbidden,
            seed=seed,
        )
        off_forbidden = forbidden | set(on_rows)
        if cardinality * 2 == COMPONENT_COUNT:
            off_forbidden |= {_complement(row) for row in on_rows}
        off_rows = _balanced_hub_stratum(
            cardinality=cardinality,
            count=shape[0] - hub_on_count,
            hub_on=False,
            forbidden=off_forbidden,
            seed=seed + 10_000,
        )
        rows = [*on_rows, *off_rows]
        random.Random(seed + 20_000).shuffle(rows)
        held[split] = _rows_to_masks(rows)
        forbidden.update(rows)
        forbidden.update(_complement(row) for row in rows)
    return validate_held_masks(held, prior_masks=prior_masks)


def validate_held_masks(
    masks: Mapping[str, torch.Tensor],
    *,
    prior_masks: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validate held shape, hub prevalence, balance, and mask/complement freshness."""
    if not isinstance(masks, Mapping) or tuple(masks) != tuple(HELD_MASK_SHAPES):
        raise ValueError("held mask keys or order changed")
    prior = _prior_rows(prior_masks)
    forbidden = prior | {_complement(row) for row in prior}
    validated = {}
    for split, shape in HELD_MASK_SHAPES.items():
        panel = _binary_panel(masks[split], label=split, shape=shape)
        cardinality = HELD_CARDINALITIES[split]
        if set(panel.sum(dim=1).tolist()) != {float(cardinality)}:
            raise ValueError(f"{split} row cardinality changed")
        rows = _mask_rows(panel)
        row_set = set(rows)
        if len(rows) != len(row_set):
            raise ValueError(f"{split} contains a duplicate mask")
        if not row_set.isdisjoint(forbidden):
            raise ValueError(f"{split} overlaps a prior mask or prior complement")
        hub_on = int(panel[:, HUB_COMPONENT].sum())
        if hub_on != HELD_HUB_ON_COUNTS[split]:
            raise ValueError(f"{split} hub-on row count changed")
        nonhub = [
            component
            for component in range(COMPONENT_COUNT)
            if component != HUB_COMPONENT
        ]
        for hub_state in (0.0, 1.0):
            stratum = panel[panel[:, HUB_COMPONENT] == hub_state]
            incidence = stratum[:, nonhub].sum(dim=0)
            if float(incidence.max() - incidence.min()) > 1.0:
                raise ValueError(f"{split} is not balanced within a hub stratum")
        if cardinality * 2 == COMPONENT_COUNT and any(
            _complement(row) in row_set for row in rows
        ):
            raise ValueError(f"{split} contains a complement pair")
        validated[split] = panel
    return validated


def diagnostic_masks() -> dict[str, torch.Tensor]:
    """Return the full-mask endpoint and the all-minus-hub diagnostic separately."""
    all_components = torch.ones((1, COMPONENT_COUNT), dtype=torch.float64)
    all_minus_hub = all_components.clone()
    all_minus_hub[0, HUB_COMPONENT] = 0.0
    return {
        "all_components": all_components,
        "all_minus_hub": all_minus_hub,
    }


def validate_diagnostic_masks(
    masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reject any drift in the two separately accounted diagnostic rows."""
    expected = diagnostic_masks()
    if not isinstance(masks, Mapping) or tuple(masks) != tuple(expected):
        raise ValueError("diagnostic mask keys or order changed")
    validated = {}
    for name, shape in DIAGNOSTIC_MASK_SHAPES.items():
        panel = _binary_panel(masks[name], label=name, shape=shape)
        if not torch.equal(panel, expected[name]):
            raise ValueError(f"{name} diagnostic changed")
        validated[name] = panel
    return validated


def _row_count(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def protocol_query_accounting(
    *,
    source_fit_rows: int,
    held_rows: int,
    diagnostic_rows: int,
) -> dict[str, int]:
    """Count patched context-mask cells without assuming an acquisition schedule."""
    source_fit_rows = _row_count(source_fit_rows, label="source fit rows")
    held_rows = _row_count(held_rows, label="held rows")
    diagnostic_rows = _row_count(diagnostic_rows, label="diagnostic rows")
    total_rows = source_fit_rows + held_rows + diagnostic_rows
    return {
        "contexts": CONTEXT_COUNT,
        "source_fit_rows": source_fit_rows,
        "held_rows": held_rows,
        "diagnostic_rows": diagnostic_rows,
        "total_mask_rows": total_rows,
        "source_fit_patched_context_mask_cells": source_fit_rows * CONTEXT_COUNT,
        "held_patched_context_mask_cells": held_rows * CONTEXT_COUNT,
        "diagnostic_patched_context_mask_cells": diagnostic_rows * CONTEXT_COUNT,
        "total_patched_context_mask_cells": total_rows * CONTEXT_COUNT,
    }


def _assembled_protocol(
    *,
    source_fit: Mapping[str, Any],
    held_masks: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    accounting = protocol_query_accounting(
        source_fit_rows=len(source_fit["masks"]),
        held_rows=sum(len(panel) for panel in held_masks.values()),
        diagnostic_rows=sum(len(panel) for panel in diagnostics.values()),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "component_map": COMPONENT_MAP,
        "hub_component": {"index": HUB_COMPONENT, "spec": HUB_COMPONENT_SPEC},
        "control_component": {
            "index": CONTROL_COMPONENT,
            "spec": CONTROL_COMPONENT_SPEC,
        },
        "seeds": {**HELD_MASK_SEEDS, **REFERENCE_MASK_SEEDS},
        "saturation_constants": dict(SATURATION_CONSTANTS),
        "bootstrap": dict(BOOTSTRAP),
        "scoring_gates": SCORING_GATES,
        "source_fit": source_fit,
        "held_masks": held_masks,
        "diagnostic_masks": diagnostics,
        "query_accounting": accounting,
    }


def build_absorbing_hub_protocol(
    inherited_fit_masks: Mapping[str, torch.Tensor],
    *,
    prior_masks: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Assemble the complete deterministic E29.13 response-blind foundation."""
    source = build_source_fit_protocol(inherited_fit_masks)
    held = build_held_masks([*prior_masks, source["masks"]])
    protocol = _assembled_protocol(
        source_fit=source,
        held_masks=held,
        diagnostics=diagnostic_masks(),
    )
    return validate_absorbing_hub_protocol(
        protocol,
        inherited_fit_masks=inherited_fit_masks,
        prior_masks=prior_masks,
    )


def validate_absorbing_hub_protocol(
    protocol: Mapping[str, Any],
    *,
    inherited_fit_masks: Mapping[str, torch.Tensor],
    prior_masks: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Purely validate the closed protocol without loading responses or files."""
    expected_keys = (
        "schema_version",
        "component_map",
        "hub_component",
        "control_component",
        "seeds",
        "saturation_constants",
        "bootstrap",
        "scoring_gates",
        "source_fit",
        "held_masks",
        "diagnostic_masks",
        "query_accounting",
    )
    if not isinstance(protocol, Mapping) or tuple(protocol) != expected_keys:
        raise ValueError("absorbing-hub protocol keys or order changed")
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("absorbing-hub schema version changed")
    if tuple(protocol["component_map"]) != COMPONENT_MAP:
        raise ValueError("component map changed")
    if protocol["hub_component"] != {
        "index": HUB_COMPONENT,
        "spec": HUB_COMPONENT_SPEC,
    }:
        raise ValueError("hub component changed")
    if protocol["control_component"] != {
        "index": CONTROL_COMPONENT,
        "spec": CONTROL_COMPONENT_SPEC,
    }:
        raise ValueError("control component changed")
    if dict(protocol["seeds"]) != {**HELD_MASK_SEEDS, **REFERENCE_MASK_SEEDS}:
        raise ValueError("protocol seeds changed")
    if dict(protocol["saturation_constants"]) != SATURATION_CONSTANTS:
        raise ValueError("saturation constants changed")
    if dict(protocol["bootstrap"]) != BOOTSTRAP:
        raise ValueError("bootstrap changed")
    if protocol["scoring_gates"] != SCORING_GATES:
        raise ValueError("scoring gates changed")

    source = validate_source_fit_protocol(protocol["source_fit"], inherited_fit_masks)
    held = validate_held_masks(
        protocol["held_masks"],
        prior_masks=[*prior_masks, source["masks"]],
    )
    diagnostics = validate_diagnostic_masks(protocol["diagnostic_masks"])
    validated = _assembled_protocol(
        source_fit=source,
        held_masks=held,
        diagnostics=diagnostics,
    )
    if dict(protocol["query_accounting"]) != validated["query_accounting"]:
        raise ValueError("query accounting changed")
    return validated


__all__ = [
    "BOOTSTRAP",
    "COMPONENT_COUNT",
    "COMPONENT_MAP",
    "CONTEXT_COUNT",
    "CONTROL_COMPONENT",
    "CONTROL_COMPONENT_SPEC",
    "DIAGNOSTIC_MASK_SHAPES",
    "HELD_CARDINALITIES",
    "HELD_HUB_ON_COUNTS",
    "HELD_MASK_SEEDS",
    "HELD_MASK_SHAPES",
    "HUB_COMPONENT",
    "HUB_COMPONENT_SPEC",
    "REFERENCE_MASK_SEEDS",
    "SATURATION_CONSTANTS",
    "SCHEMA_VERSION",
    "SCORING_GATES",
    "SOURCE_INHERITED_SHAPES",
    "SOURCE_STRATA",
    "build_absorbing_hub_protocol",
    "build_held_masks",
    "build_source_fit_protocol",
    "diagnostic_masks",
    "protocol_query_accounting",
    "validate_absorbing_hub_protocol",
    "validate_diagnostic_masks",
    "validate_held_masks",
    "validate_source_fit_protocol",
]
