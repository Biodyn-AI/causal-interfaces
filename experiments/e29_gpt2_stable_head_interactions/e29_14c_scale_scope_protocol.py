"""Pure response-blind protocol and scoring for the E29.14c scale ladder."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch

COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
HUB_COMPONENT = 1
MASK_SPLITS = ("scale36", "scale40", "scale44", "scale47")
MULTIROW_MASK_SPLITS = MASK_SPLITS[:-1]
MASK_SHAPES = {
    "scale36": (47, COMPONENT_COUNT),
    "scale40": (47, COMPONENT_COUNT),
    "scale44": (47, COMPONENT_COUNT),
    "scale47": (1, COMPONENT_COUNT),
}
MASK_CARDINALITIES = {
    "scale36": 36,
    "scale40": 40,
    "scale44": 44,
    "scale47": 47,
}
POSITION_SCOPE_ORDER = ("all_positions", "differing_lexical_positions")
SINGLETON_CONDITIONS = (
    {
        "name": "singleton_hub_all_positions",
        "component": HUB_COMPONENT,
        "scope": "all_positions",
        "alpha": 1.0,
    },
    {
        "name": "singleton_hub_differing_lexical",
        "component": HUB_COMPONENT,
        "scope": "differing_lexical_positions",
        "alpha": 1.0,
    },
)
TEMPLATE_STRATA = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))

# These offsets were chosen without model responses. Across 47 cyclic shifts,
# they give exact component marginals, nested 4/4/3 additions, bounded pair
# imbalance, and full-rank transition incidence matrices with condition < 8.
COMPONENT_ORDER_DOMAIN = "e29_14c_non_hub_order_v1|29143"
TRANSITION_SHELL_OFFSETS = {
    "scale36_to_scale40": (10, 16, 29, 42),
    "scale40_to_scale44": (15, 18, 25, 45),
    "scale44_to_scale47": (0, 12, 26),
}
OMISSION_OFFSETS = {
    "scale36": (0, 12, 26, 15, 18, 25, 45, 10, 16, 29, 42),
    "scale40": (0, 12, 26, 15, 18, 25, 45),
    "scale44": (0, 12, 26),
    "scale47": (),
}
TRANSITION_DESIGN_CONDITION_NUMBERS = {
    "scale36_to_scale40": 7.7035161832255845,
    "scale40_to_scale44": 6.354765333633317,
    "scale44_to_scale47": 7.532788816021114,
}
EXPECTED_MASK_CANONICAL_SHA256 = {
    "scale36": "7a5ef49c088e180685fff5d015596693b55bbb0dce7b6bf9311ef5cc500a5871",
    "scale40": "0d69fa100a61ad870c5a1fe36a988e656269811017dcf22236630244cbc17cbb",
    "scale44": "9e1af22d50f9bc5c0f31fe19fdae43e09835a68cf00d4df91d0075ad6003b4c3",
    "scale47": "2a249b2d3dc95d2c2f07571dfcb6003f9995ef706d57560f746032fdb19fba0f",
}

BOOTSTRAP = {
    "seed": 29_143,
    "replicates": 5_000,
    "lower_quantile_one_sided": 0.05,
    "upper_quantile_one_sided": 0.95,
    "context_resampling": "contexts_within_four_fixed_template_strata",
    "mask_resampling": "common_nested_chain_rows_across_scales",
    "joint_scale_weighting": "equal_one_quarter_weight_per_scale",
    "interpretation": (
        "fixed_design_sensitivity_analysis_not_random_mask_population_inference"
    ),
}
SCORING_GATES = {
    "singleton_transfer": {
        "endpoint_mean_min": 0.985,
        "endpoint_mean_max": 1.01,
        "endpoint_absolute_gap_bootstrap_upper_max": 0.03,
        "paired_absolute_gap_bootstrap_upper_max": 0.03,
    },
    "scope_equivalence": {
        "paired_absolute_gap_bootstrap_upper_max": 0.03,
        "context_mae_max": 0.05,
        "minimum_passing_contexts": 11,
        "template_mae_max": 0.05,
        "minimum_passing_templates": 4,
        "mask_mae_max": 0.05,
        "minimum_passing_multirow_masks": 46,
        "cell_absolute_gap_max": 0.10,
        "minimum_passing_multirow_cells": 536,
        "minimum_passing_scale47_cells": 11,
    },
    "joint_scope_equivalence": {
        "paired_absolute_gap_bootstrap_upper_max": 0.03,
        "equal_scale_weighting": True,
    },
    "c36_nonsaturation": {
        "all_position_endpoint_absolute_gap_bootstrap_lower_min": 0.03,
    },
    "c47_backup_endpoint": {
        "endpoint_mean_min": 0.95,
        "endpoint_mean_max": 1.01,
        "endpoint_absolute_gap_bootstrap_upper_max": 0.05,
    },
}
QUERY_ACCOUNTING = {
    "contexts": CONTEXT_COUNT,
    "mask_rows_by_split": {
        split: MASK_SHAPES[split][0] for split in MASK_SPLITS
    },
    "multi_component_masks": sum(MASK_SHAPES[split][0] for split in MASK_SPLITS),
    "position_scopes": len(POSITION_SCOPE_ORDER),
    "singleton_condition_rows_per_context": len(SINGLETON_CONDITIONS),
    "multi_component_condition_rows_per_context": sum(
        MASK_SHAPES[split][0] for split in MASK_SPLITS
    )
    * len(POSITION_SCOPE_ORDER),
    "patched_condition_rows_per_context": len(SINGLETON_CONDITIONS)
    + sum(MASK_SHAPES[split][0] for split in MASK_SPLITS)
    * len(POSITION_SCOPE_ORDER),
    "patched_context_condition_cells": CONTEXT_COUNT
    * (
        len(SINGLETON_CONDITIONS)
        + sum(MASK_SHAPES[split][0] for split in MASK_SPLITS)
        * len(POSITION_SCOPE_ORDER)
    ),
    "clean_forward_examples": 2 * CONTEXT_COUNT,
    "maximum_logical_queries": CONTEXT_COUNT
    * (
        len(SINGLETON_CONDITIONS)
        + sum(MASK_SHAPES[split][0] for split in MASK_SPLITS)
        * len(POSITION_SCOPE_ORDER)
        + 2
    ),
}
PHYSICAL_QUERY_ACCOUNTING = {
    **QUERY_ACCOUNTING,
    "batch_smoke_unique_condition_rows": QUERY_ACCOUNTING[
        "patched_condition_rows_per_context"
    ],
    "batch_smoke_physical_condition_cells": 2
    * QUERY_ACCOUNTING["patched_condition_rows_per_context"],
    "maximum_physical_model_examples": QUERY_ACCOUNTING["maximum_logical_queries"]
    + 2 * QUERY_ACCOUNTING["patched_condition_rows_per_context"],
}
SCHEMA_VERSION = "e29_14c_hub_off_scale_scope_ladder_response_blind_v1"


def _non_hub_component_order() -> tuple[int, ...]:
    components = (index for index in range(COMPONENT_COUNT) if index != HUB_COMPONENT)
    return tuple(
        sorted(
            components,
            key=lambda index: hashlib.sha256(
                f"{COMPONENT_ORDER_DOMAIN}|{index}".encode("ascii")
            ).digest(),
        )
    )


NON_HUB_COMPONENT_ORDER = _non_hub_component_order()


def _mask_rows(panel: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in torch.where(row == 1.0)[0])
        for row in panel
    )


def _canonical_mask_sha256(panel: torch.Tensor) -> str:
    payload = "\n".join(
        ",".join(str(index) for index in row) for row in _mask_rows(panel)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def build_scale_masks() -> dict[str, torch.Tensor]:
    """Construct the exact nested response-blind hub-off scale panels."""
    masks = {}
    for split in MULTIROW_MASK_SPLITS:
        panel = torch.zeros(MASK_SHAPES[split], dtype=torch.float64)
        for row_index in range(len(NON_HUB_COMPONENT_ORDER)):
            panel[row_index, list(NON_HUB_COMPONENT_ORDER)] = 1.0
            omitted = [
                NON_HUB_COMPONENT_ORDER[(row_index + offset) % 47]
                for offset in OMISSION_OFFSETS[split]
            ]
            panel[row_index, omitted] = 0.0
        masks[split] = panel
    scale47 = torch.zeros(MASK_SHAPES["scale47"], dtype=torch.float64)
    scale47[0, list(NON_HUB_COMPONENT_ORDER)] = 1.0
    masks["scale47"] = scale47
    return masks


def validate_scale_masks(
    masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Require exact rows, balance, nesting, and hashes for all four scales."""
    if not isinstance(masks, Mapping) or tuple(masks) != MASK_SPLITS:
        raise ValueError("scale-mask keys or order changed")
    validated = {}
    non_hub = torch.tensor(NON_HUB_COMPONENT_ORDER, dtype=torch.long)
    for split in MASK_SPLITS:
        value = masks[split]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{split} masks must be a tensor")
        panel = value.detach().cpu().double().clone()
        if tuple(panel.shape) != MASK_SHAPES[split]:
            raise ValueError(f"{split} mask shape changed")
        if not bool(torch.logical_or(panel == 0.0, panel == 1.0).all()):
            raise ValueError(f"{split} masks are not binary")
        if not bool((panel.sum(dim=1) == MASK_CARDINALITIES[split]).all()):
            raise ValueError(f"{split} mask cardinality changed")
        if not bool((panel[:, HUB_COMPONENT] == 0.0).all()):
            raise ValueError(f"{split} includes the excluded hub component")
        rows = _mask_rows(panel)
        if len(rows) != len(set(rows)):
            raise ValueError(f"{split} contains duplicate masks")
        if split in MULTIROW_MASK_SPLITS and not bool(
            (panel[:, non_hub].sum(dim=0) == MASK_CARDINALITIES[split]).all()
        ):
            raise ValueError(f"{split} component marginals are not exact")
        if _canonical_mask_sha256(panel) != EXPECTED_MASK_CANONICAL_SHA256[split]:
            raise ValueError(f"{split} canonical hash changed")
        validated[split] = panel

    if not bool((validated["scale36"] <= validated["scale40"]).all()):
        raise ValueError("scale36 masks are not nested in scale40")
    if not bool((validated["scale40"] <= validated["scale44"]).all()):
        raise ValueError("scale40 masks are not nested in scale44")
    if not bool((validated["scale44"] <= validated["scale47"]).all()):
        raise ValueError("scale44 masks are not nested in scale47")
    return validated


def _pair_count_histogram(offsets: Sequence[int]) -> dict[str, int]:
    counts = Counter(
        (first - second) % 47
        for first in offsets
        for second in offsets
        if first != second
    )
    histogram = Counter(counts[difference] for difference in range(1, 47))
    return {str(count): int(frequency) for count, frequency in sorted(histogram.items())}


def build_scale_scope_protocol() -> dict[str, Any]:
    """Build the complete file-free E29.14c scientific protocol."""
    masks = validate_scale_masks(build_scale_masks())
    return {
        "schema_version": SCHEMA_VERSION,
        "context_count": CONTEXT_COUNT,
        "component_count": COMPONENT_COUNT,
        "hub_component": HUB_COMPONENT,
        "component_order_domain": COMPONENT_ORDER_DOMAIN,
        "non_hub_component_order": list(NON_HUB_COMPONENT_ORDER),
        "mask_splits": list(MASK_SPLITS),
        "mask_shapes": _jsonable(MASK_SHAPES),
        "mask_cardinalities": dict(MASK_CARDINALITIES),
        "mask_canonical_sha256": dict(EXPECTED_MASK_CANONICAL_SHA256),
        "omission_offsets": _jsonable(OMISSION_OFFSETS),
        "omission_pair_count_histograms": {
            split: _pair_count_histogram(OMISSION_OFFSETS[split])
            for split in MULTIROW_MASK_SPLITS
        },
        "transition_shell_offsets": _jsonable(TRANSITION_SHELL_OFFSETS),
        "transition_design_condition_numbers": dict(
            TRANSITION_DESIGN_CONDITION_NUMBERS
        ),
        "mask_rows": {
            split: _jsonable(_mask_rows(masks[split])) for split in MASK_SPLITS
        },
        "position_scope_order": list(POSITION_SCOPE_ORDER),
        "singleton_conditions": _jsonable(SINGLETON_CONDITIONS),
        "template_strata": _jsonable(TEMPLATE_STRATA),
        "bootstrap": dict(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
    }


def _masks_from_rows(rows_by_split: Any) -> dict[str, torch.Tensor]:
    if not isinstance(rows_by_split, Mapping) or tuple(rows_by_split) != MASK_SPLITS:
        raise ValueError("protocol mask-row keys or order changed")
    masks = {}
    for split in MASK_SPLITS:
        rows = rows_by_split[split]
        if not isinstance(rows, Sequence) or len(rows) != MASK_SHAPES[split][0]:
            raise ValueError(f"{split} protocol mask rows changed")
        panel = torch.zeros(MASK_SHAPES[split], dtype=torch.float64)
        for row_index, selected in enumerate(rows):
            if not isinstance(selected, Sequence):
                raise TypeError("protocol mask row must be a sequence")
            indices = tuple(int(index) for index in selected)
            if len(indices) != len(set(indices)) or any(
                index < 0 or index >= COMPONENT_COUNT for index in indices
            ):
                raise ValueError("protocol mask row contains invalid indices")
            panel[row_index, list(indices)] = 1.0
        masks[split] = panel
    return masks


def validate_scale_scope_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any mutation of the fixed pure protocol or mask closure."""
    if not isinstance(protocol, Mapping):
        raise TypeError("scale-scope protocol must be a mapping")
    expected = build_scale_scope_protocol()
    if tuple(protocol) != tuple(expected):
        raise ValueError("scale-scope protocol keys or order changed")
    validate_scale_masks(_masks_from_rows(protocol["mask_rows"]))
    for key in expected:
        if protocol[key] != expected[key]:
            raise ValueError(f"scale-scope protocol {key.replace('_', ' ')} changed")
    return expected


def _finite_context_vector(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    vector = value.detach().cpu().double().clone()
    if tuple(vector.shape) != (CONTEXT_COUNT,) or not bool(vector.isfinite().all()):
        raise ValueError(f"{label} must be a finite 12-context vector")
    return vector


def _finite_response_tables(
    values: Mapping[str, torch.Tensor], *, label: str
) -> dict[str, torch.Tensor]:
    if not isinstance(values, Mapping) or tuple(values) != MASK_SPLITS:
        raise ValueError(f"{label} keys or order changed")
    tables = {}
    for split in MASK_SPLITS:
        value = values[split]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}/{split} must be a tensor")
        table = value.detach().cpu().double().clone()
        shape = (MASK_SHAPES[split][0], CONTEXT_COUNT)
        if tuple(table.shape) != shape or not bool(table.isfinite().all()):
            raise ValueError(f"{label}/{split} must be finite with shape {shape}")
        tables[split] = table
    return tables


def _context_samples(generator: torch.Generator, replicates: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.tensor(stratum, dtype=torch.long)[
                torch.randint(
                    len(stratum),
                    (replicates, len(stratum)),
                    generator=generator,
                )
            ]
            for stratum in TEMPLATE_STRATA
        ],
        dim=1,
    )


def _bootstrap_summary(estimates: torch.Tensor, point_estimate: float) -> dict[str, Any]:
    lower, upper = torch.quantile(
        estimates,
        torch.tensor(
            (
                float(BOOTSTRAP["lower_quantile_one_sided"]),
                float(BOOTSTRAP["upper_quantile_one_sided"]),
            ),
            dtype=torch.float64,
        ),
    )
    return {
        "point_estimate": point_estimate,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "lower_quantile_one_sided": float(BOOTSTRAP["lower_quantile_one_sided"]),
        "upper_quantile_one_sided": float(BOOTSTRAP["upper_quantile_one_sided"]),
        "seed": int(BOOTSTRAP["seed"]),
        "replicates": int(BOOTSTRAP["replicates"]),
    }


def _context_bootstrap_mean(values: torch.Tensor, *, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    contexts = _context_samples(generator, int(BOOTSTRAP["replicates"]))
    estimates = values[contexts].mean(dim=1)
    result = _bootstrap_summary(estimates, float(values.mean()))
    result["resampling"] = str(BOOTSTRAP["context_resampling"])
    return result


def _crossed_bootstrap_mean(values: torch.Tensor, *, seed: int) -> dict[str, Any]:
    replicates = int(BOOTSTRAP["replicates"])
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randint(len(values), (replicates, len(values)), generator=generator)
    contexts = _context_samples(generator, replicates)
    estimates = values[rows[:, :, None], contexts[:, None, :]].mean(dim=(1, 2))
    result = _bootstrap_summary(estimates, float(values.mean()))
    result["resampling"] = (
        f"{BOOTSTRAP['mask_resampling']}_crossed_with_"
        f"{BOOTSTRAP['context_resampling']}"
    )
    return result


def _joint_equal_scale_bootstrap(
    values: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    replicates = int(BOOTSTRAP["replicates"])
    generator = torch.Generator().manual_seed(int(BOOTSTRAP["seed"]))
    rows = torch.randint(47, (replicates, 47), generator=generator)
    contexts = _context_samples(generator, replicates)
    estimates = []
    points = []
    for split in MULTIROW_MASK_SPLITS:
        estimates.append(
            values[split][rows[:, :, None], contexts[:, None, :]].mean(dim=(1, 2))
        )
        points.append(values[split].mean())
    estimates.append(values["scale47"][0, contexts].mean(dim=1))
    points.append(values["scale47"].mean())
    joint_estimates = torch.stack(estimates).mean(dim=0)
    result = _bootstrap_summary(
        joint_estimates, float(torch.stack(points).mean())
    )
    result["resampling"] = (
        "common_nested_chain_rows_crossed_with_contexts_within_template_strata"
    )
    result["scale_weighting"] = str(BOOTSTRAP["joint_scale_weighting"])
    return result


def _scope_equivalence_claim(
    absolute_gap: torch.Tensor,
    *,
    split: str,
    integrity_passed: bool,
) -> dict[str, Any]:
    thresholds = SCORING_GATES["scope_equivalence"]
    if split == "scale47":
        bootstrap = _context_bootstrap_mean(
            absolute_gap[0], seed=int(BOOTSTRAP["seed"])
        )
    else:
        bootstrap = _crossed_bootstrap_mean(
            absolute_gap, seed=int(BOOTSTRAP["seed"])
        )
    context_mae = absolute_gap.mean(dim=0)
    mask_mae = absolute_gap.mean(dim=1)
    template_mae = [
        float(absolute_gap[:, torch.tensor(stratum, dtype=torch.long)].mean())
        for stratum in TEMPLATE_STRATA
    ]
    passing_contexts = int((context_mae <= thresholds["context_mae_max"]).sum())
    passing_templates = sum(
        value <= thresholds["template_mae_max"] for value in template_mae
    )
    passing_masks = int((mask_mae <= thresholds["mask_mae_max"]).sum())
    passing_cells = int(
        (absolute_gap <= thresholds["cell_absolute_gap_max"]).sum()
    )
    mask_gate = split == "scale47" or passing_masks >= thresholds[
        "minimum_passing_multirow_masks"
    ]
    minimum_cells = (
        thresholds["minimum_passing_scale47_cells"]
        if split == "scale47"
        else thresholds["minimum_passing_multirow_cells"]
    )
    passed = bool(
        integrity_passed
        and bootstrap["upper_95"]
        <= thresholds["paired_absolute_gap_bootstrap_upper_max"]
        and passing_contexts >= thresholds["minimum_passing_contexts"]
        and passing_templates >= thresholds["minimum_passing_templates"]
        and mask_gate
        and passing_cells >= minimum_cells
    )
    return {
        "paired_absolute_gap_bootstrap": bootstrap,
        "context_mae": [float(value) for value in context_mae],
        "passing_contexts": passing_contexts,
        "template_mae": template_mae,
        "passing_templates": passing_templates,
        "mask_mae": [float(value) for value in mask_mae],
        "passing_masks": passing_masks,
        "mask_count": len(mask_mae),
        "passing_cells": passing_cells,
        "cell_count": int(absolute_gap.numel()),
        "passed": passed,
    }


def _matrix_diagnostic_summary(values: torch.Tensor) -> dict[str, Any]:
    """Keep a compact summary and the raw normalized diagnostic matrix."""
    return {
        "normalized_mean": float(values.mean()),
        "normalized_values": _jsonable(values.tolist()),
        "row_means": [float(value) for value in values.mean(dim=1)],
        "context_means": [float(value) for value in values.mean(dim=0)],
        "template_means": [
            float(values[:, torch.tensor(stratum, dtype=torch.long)].mean())
            for stratum in TEMPLATE_STRATA
        ],
    }


def _scale_response_geometry(
    all_normalized: Mapping[str, torch.Tensor],
    lexical_normalized: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    geometry = {}
    for split in MASK_SPLITS:
        by_scope = {}
        for scope, values in (
            ("all_positions", all_normalized[split]),
            ("differing_lexical_positions", lexical_normalized[split]),
        ):
            summary = _matrix_diagnostic_summary(values)
            endpoint_distance = (values - 1.0).abs()
            summary.update(
                {
                    "endpoint_absolute_distance_mean": float(
                        endpoint_distance.mean()
                    ),
                    "endpoint_absolute_distance_row_means": [
                        float(value) for value in endpoint_distance.mean(dim=1)
                    ],
                    "endpoint_absolute_distance_context_means": [
                        float(value) for value in endpoint_distance.mean(dim=0)
                    ],
                    "endpoint_absolute_distance_template_means": [
                        float(
                            endpoint_distance[
                                :, torch.tensor(stratum, dtype=torch.long)
                            ].mean()
                        )
                        for stratum in TEMPLATE_STRATA
                    ],
                }
            )
            by_scope[scope] = summary
        geometry[split] = by_scope
    return geometry


def _expanded_following_table(
    values: Mapping[str, torch.Tensor], following: str
) -> torch.Tensor:
    table = values[following]
    return table.repeat(47, 1) if following == "scale47" else table


def _expanded_following_masks(
    masks: Mapping[str, torch.Tensor], following: str
) -> torch.Tensor:
    panel = masks[following]
    return panel.repeat(47, 1) if following == "scale47" else panel


def _transition_scope_gap_diagnostics(
    signed_gaps: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    transitions = (
        ("scale36", "scale40"),
        ("scale40", "scale44"),
        ("scale44", "scale47"),
    )
    diagnostics = {}
    for current, following in transitions:
        following_gap = _expanded_following_table(signed_gaps, following)
        following_masks = _expanded_following_masks(masks, following)
        signed_increment_gap = following_gap - signed_gaps[current]
        increment_gap = signed_increment_gap.abs()
        name = f"{current}_to_{following}"
        diagnostics[name] = {
            "diagnostic_only": True,
            "diagnostic_target": (
                "change_in_lexical_minus_all_scope_gap_not_endpoint_restoration"
            ),
            "absolute_scope_increment_gap_bootstrap": _crossed_bootstrap_mean(
                increment_gap, seed=int(BOOTSTRAP["seed"])
            ),
            "signed_scope_increment_gap": _jsonable(
                signed_increment_gap.tolist()
            ),
            "signed_scope_increment_gap_mean": float(
                signed_increment_gap.mean()
            ),
            "context_mae": [
                float(value) for value in increment_gap.mean(dim=0)
            ],
            "mask_mae": [float(value) for value in increment_gap.mean(dim=1)],
            "template_mae": [
                float(
                    increment_gap[
                        :, torch.tensor(stratum, dtype=torch.long)
                    ].mean()
                )
                for stratum in TEMPLATE_STRATA
            ],
            "added_component_rows": _jsonable(
                _mask_rows(following_masks - masks[current])
            ),
            "incidence_condition_number": TRANSITION_DESIGN_CONDITION_NUMBERS[
                name
            ],
        }
    return diagnostics


def _transition_response_increment_diagnostics(
    all_normalized: Mapping[str, torch.Tensor],
    lexical_normalized: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Preserve actual rung increments for response-blind failure localization."""
    transitions = (
        ("scale36", "scale40"),
        ("scale40", "scale44"),
        ("scale44", "scale47"),
    )
    diagnostics = {}
    for current, following in transitions:
        name = f"{current}_to_{following}"
        following_masks = _expanded_following_masks(masks, following)
        record = {
            "diagnostic_only": True,
            "diagnostic_target": "actual_normalized_response_restoration_by_shell",
            "added_component_rows": _jsonable(
                _mask_rows(following_masks - masks[current])
            ),
            "incidence_condition_number": TRANSITION_DESIGN_CONDITION_NUMBERS[
                name
            ],
        }
        for scope, values in (
            ("all_positions", all_normalized),
            ("differing_lexical_positions", lexical_normalized),
        ):
            increments = _expanded_following_table(values, following) - values[
                current
            ]
            record[scope] = _matrix_diagnostic_summary(increments)
        diagnostics[name] = record
    return diagnostics


def score_scale_scope_ladder(
    all_position_responses: Mapping[str, torch.Tensor],
    lexical_position_responses: Mapping[str, torch.Tensor],
    singleton_all_response: torch.Tensor,
    singleton_lexical_response: torch.Tensor,
    clean_endpoints: torch.Tensor,
    held_masks: Mapping[str, torch.Tensor],
    *,
    integrity_passed: bool,
) -> dict[str, Any]:
    """Score scope equivalence, endpoint state, and nested failure diagnostics."""
    if not isinstance(integrity_passed, bool):
        raise TypeError("integrity_passed must be a bool")
    masks = validate_scale_masks(held_masks)
    all_raw = _finite_response_tables(all_position_responses, label="all responses")
    lexical_raw = _finite_response_tables(
        lexical_position_responses, label="lexical responses"
    )
    singleton_all_raw = _finite_context_vector(
        singleton_all_response, label="singleton all response"
    )
    singleton_lexical_raw = _finite_context_vector(
        singleton_lexical_response, label="singleton lexical response"
    )
    endpoints = _finite_context_vector(clean_endpoints, label="clean endpoints")
    if not bool((endpoints > 0.0).all()):
        raise ValueError("clean endpoints must be positive")

    all_normalized = {split: all_raw[split] / endpoints[None] for split in MASK_SPLITS}
    lexical_normalized = {
        split: lexical_raw[split] / endpoints[None] for split in MASK_SPLITS
    }
    signed_gaps = {
        split: lexical_normalized[split] - all_normalized[split]
        for split in MASK_SPLITS
    }
    absolute_gaps = {split: signed_gaps[split].abs() for split in MASK_SPLITS}

    singleton_all = singleton_all_raw / endpoints
    singleton_lexical = singleton_lexical_raw / endpoints
    singleton_thresholds = SCORING_GATES["singleton_transfer"]
    singleton_all_endpoint = _context_bootstrap_mean(
        (singleton_all - 1.0).abs(), seed=int(BOOTSTRAP["seed"])
    )
    singleton_lexical_endpoint = _context_bootstrap_mean(
        (singleton_lexical - 1.0).abs(), seed=int(BOOTSTRAP["seed"])
    )
    singleton_paired = _context_bootstrap_mean(
        (singleton_lexical - singleton_all).abs(), seed=int(BOOTSTRAP["seed"])
    )
    singleton_claim = {
        "all_position_mean": float(singleton_all.mean()),
        "lexical_position_mean": float(singleton_lexical.mean()),
        "all_position_endpoint_absolute_gap_bootstrap": singleton_all_endpoint,
        "lexical_position_endpoint_absolute_gap_bootstrap": (
            singleton_lexical_endpoint
        ),
        "paired_absolute_gap_bootstrap": singleton_paired,
    }
    singleton_claim["passed"] = bool(
        integrity_passed
        and singleton_thresholds["endpoint_mean_min"]
        <= singleton_claim["all_position_mean"]
        <= singleton_thresholds["endpoint_mean_max"]
        and singleton_thresholds["endpoint_mean_min"]
        <= singleton_claim["lexical_position_mean"]
        <= singleton_thresholds["endpoint_mean_max"]
        and singleton_all_endpoint["upper_95"]
        <= singleton_thresholds["endpoint_absolute_gap_bootstrap_upper_max"]
        and singleton_lexical_endpoint["upper_95"]
        <= singleton_thresholds["endpoint_absolute_gap_bootstrap_upper_max"]
        and singleton_paired["upper_95"]
        <= singleton_thresholds["paired_absolute_gap_bootstrap_upper_max"]
    )

    scope_claims = {
        split: _scope_equivalence_claim(
            absolute_gaps[split], split=split, integrity_passed=integrity_passed
        )
        for split in MASK_SPLITS
    }
    joint_bootstrap = _joint_equal_scale_bootstrap(absolute_gaps)
    joint_claim = {
        "paired_absolute_gap_bootstrap": joint_bootstrap,
        "passed": bool(
            integrity_passed
            and joint_bootstrap["upper_95"]
            <= SCORING_GATES["joint_scope_equivalence"][
                "paired_absolute_gap_bootstrap_upper_max"
            ]
        ),
    }

    c36_endpoint = _crossed_bootstrap_mean(
        (all_normalized["scale36"] - 1.0).abs(), seed=int(BOOTSTRAP["seed"])
    )
    c36_nonsaturation = {
        "all_position_endpoint_absolute_gap_bootstrap": c36_endpoint,
        "passed": bool(
            integrity_passed
            and c36_endpoint["lower_95"]
            >= SCORING_GATES["c36_nonsaturation"][
                "all_position_endpoint_absolute_gap_bootstrap_lower_min"
            ]
        ),
    }

    c47_thresholds = SCORING_GATES["c47_backup_endpoint"]
    c47_endpoint_bootstraps = {
        "all_positions": _context_bootstrap_mean(
            (all_normalized["scale47"][0] - 1.0).abs(),
            seed=int(BOOTSTRAP["seed"]),
        ),
        "differing_lexical_positions": _context_bootstrap_mean(
            (lexical_normalized["scale47"][0] - 1.0).abs(),
            seed=int(BOOTSTRAP["seed"]),
        ),
    }
    c47_means = {
        "all_positions": float(all_normalized["scale47"].mean()),
        "differing_lexical_positions": float(
            lexical_normalized["scale47"].mean()
        ),
    }
    c47_backup = {
        "normalized_means": c47_means,
        "endpoint_absolute_gap_bootstrap": c47_endpoint_bootstraps,
    }
    c47_backup["passed"] = bool(
        integrity_passed
        and all(
            c47_thresholds["endpoint_mean_min"]
            <= mean
            <= c47_thresholds["endpoint_mean_max"]
            for mean in c47_means.values()
        )
        and all(
            bootstrap["upper_95"]
            <= c47_thresholds["endpoint_absolute_gap_bootstrap_upper_max"]
            for bootstrap in c47_endpoint_bootstraps.values()
        )
    )

    scope_authorized = bool(
        integrity_passed
        and singleton_claim["passed"]
        and all(claim["passed"] for claim in scope_claims.values())
        and joint_claim["passed"]
    )
    complete_ladder = bool(
        scope_authorized and c36_nonsaturation["passed"] and c47_backup["passed"]
    )
    claims = {
        "singleton_transfer": singleton_claim,
        "scope_equivalence_by_scale": scope_claims,
        "joint_equal_scale_scope_equivalence": joint_claim,
        "high_scale_lexical_scope_authorized": {"passed": scope_authorized},
        "informative_c36_nonsaturation": c36_nonsaturation,
        "c47_backup_endpoint": c47_backup,
        "complete_hub_off_scale_ladder": {"passed": complete_ladder},
    }
    return {
        "schema_version": "e29_14c_hub_off_scale_scope_ladder_score_v1",
        "integrity_passed": integrity_passed,
        "claims": claims,
        "scale_response_geometry": _scale_response_geometry(
            all_normalized, lexical_normalized
        ),
        "transition_response_increment_diagnostics": (
            _transition_response_increment_diagnostics(
                all_normalized, lexical_normalized, masks
            )
        ),
        "transition_scope_surrogate_failure_localization": (
            _transition_scope_gap_diagnostics(
                signed_gaps, masks
            )
        ),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "recommended_interpretation": (
            "hub_off_c36_c47_scope_equivalence_with_informative_transition_and_"
            "backup_endpoint_confirmed"
            if complete_ladder
            else (
                "hub_off_c36_c47_lexical_scope_surrogate_confirmed_with_"
                "separate_endpoint_claim_failure"
                if scope_authorized
                else "inspect_first_failing_scale_and_nested_transition_residuals"
            )
        ),
    }


__all__ = [
    "BOOTSTRAP",
    "COMPONENT_COUNT",
    "CONTEXT_COUNT",
    "EXPECTED_MASK_CANONICAL_SHA256",
    "HUB_COMPONENT",
    "MASK_CARDINALITIES",
    "MASK_SHAPES",
    "MASK_SPLITS",
    "NON_HUB_COMPONENT_ORDER",
    "PHYSICAL_QUERY_ACCOUNTING",
    "POSITION_SCOPE_ORDER",
    "QUERY_ACCOUNTING",
    "SCHEMA_VERSION",
    "SCORING_GATES",
    "SINGLETON_CONDITIONS",
    "TEMPLATE_STRATA",
    "TRANSITION_DESIGN_CONDITION_NUMBERS",
    "TRANSITION_SHELL_OFFSETS",
    "build_scale_masks",
    "build_scale_scope_protocol",
    "score_scale_scope_ladder",
    "validate_scale_masks",
    "validate_scale_scope_protocol",
]
