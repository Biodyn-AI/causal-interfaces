"""Pure protocol and scoring for the prospective E29.14b scope bridge."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch

COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
HUB_COMPONENT = 1
MASK_SPLITS = ("confirmation_size24", "confirmation_size32")
MASK_SHAPES = {
    "confirmation_size24": (48, COMPONENT_COUNT),
    "confirmation_size32": (48, COMPONENT_COUNT),
}
MASK_CARDINALITIES = {
    "confirmation_size24": 24,
    "confirmation_size32": 32,
}
MASK_HUB_ON_COUNTS = {
    "confirmation_size24": 24,
    "confirmation_size32": 32,
}
EXPECTED_MASK_CANONICAL_SHA256 = {
    "confirmation_size24": (
        "810cf7d705753b56cd7104144becd0793c8e3cfb38aaa1d3cbd6a7e3ecd19c9f"
    ),
    "confirmation_size32": (
        "ab8a05abb5c8620757ed5539468a1ccf4eadf6b1ea2143954dce01edff8fac36"
    ),
}
HUB_STATE_ROW_INDICES = {
    "confirmation_size24": {
        "hub_off": (3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 22, 25, 28, 30, 32, 35, 40, 41, 44, 45, 46, 47),
        "hub_on": (0, 1, 2, 4, 9, 16, 18, 19, 20, 21, 23, 24, 26, 27, 29, 31, 33, 34, 36, 37, 38, 39, 42, 43),
    },
    "confirmation_size32": {
        "hub_off": (1, 3, 6, 9, 10, 13, 16, 28, 36, 37, 38, 39, 41, 44, 46, 47),
        "hub_on": (0, 2, 4, 5, 7, 8, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 40, 42, 43, 45),
    },
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
BOOTSTRAP = {
    "seed": 29_142,
    "replicates": 5_000,
    "lower_quantile_one_sided": 0.05,
    "upper_quantile_one_sided": 0.95,
    "resampling": (
        "masks_within_scale_and_hub_state_crossed_with_contexts_within_"
        "four_fixed_template_strata"
    ),
    "joint_scale_weighting": "equal_one_half_weight_per_scale",
}
SCORING_GATES = {
    "singleton_transfer": {
        "endpoint_mean_min": 0.985,
        "endpoint_mean_max": 1.01,
        "endpoint_absolute_gap_bootstrap_upper_max": 0.03,
        "paired_absolute_gap_bootstrap_upper_max": 0.03,
    },
    "all_position_anchor": {
        "hub_on_endpoint_mean_min": 0.985,
        "hub_on_endpoint_mean_max": 1.01,
        "hub_on_endpoint_absolute_gap_bootstrap_upper_max": 0.03,
        "hub_off_endpoint_absolute_gap_bootstrap_lower_min": 0.05,
    },
    "scope_equivalence": {
        "paired_absolute_gap_bootstrap_upper_max": 0.03,
        "context_mae_max": 0.05,
        "minimum_passing_contexts": 11,
        "template_mae_max": 0.05,
        "minimum_passing_templates": 4,
        "mask_mae_max": 0.05,
        "maximum_failing_masks": 1,
    },
    "hub_on_lexical_endpoint": {
        "endpoint_mean_min": 0.985,
        "endpoint_mean_max": 1.01,
        "endpoint_absolute_gap_bootstrap_upper_max": 0.03,
    },
    "joint_scope_equivalence": {
        "paired_absolute_gap_bootstrap_upper_max": 0.03,
        "equal_scale_weighting": True,
    },
}
QUERY_ACCOUNTING = {
    "contexts": CONTEXT_COUNT,
    "mask_rows_by_split": {split: MASK_SHAPES[split][0] for split in MASK_SPLITS},
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
SCHEMA_VERSION = "e29_14b_component_position_bridge_response_blind_v1"


def _mask_rows(panel: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in torch.where(row == 1.0)[0]) for row in panel
    )


def _canonical_mask_sha256(panel: torch.Tensor) -> str:
    payload = "\n".join(
        ",".join(str(index) for index in row) for row in _mask_rows(panel)
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_bridge_masks(
    masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Require the exact sealed E29.13 c24 and c32 panels and row order."""
    if not isinstance(masks, Mapping) or tuple(masks) != MASK_SPLITS:
        raise ValueError("bridge mask keys or order changed")
    validated = {}
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
        rows = _mask_rows(panel)
        if len(rows) != len(set(rows)):
            raise ValueError(f"{split} contains duplicate masks")
        if int(panel[:, HUB_COMPONENT].sum()) != MASK_HUB_ON_COUNTS[split]:
            raise ValueError(f"{split} hub-on count changed")
        observed_indices = {
            "hub_off": tuple(
                int(index) for index in torch.where(panel[:, HUB_COMPONENT] == 0)[0]
            ),
            "hub_on": tuple(
                int(index) for index in torch.where(panel[:, HUB_COMPONENT] == 1)[0]
            ),
        }
        if observed_indices != HUB_STATE_ROW_INDICES[split]:
            raise ValueError(f"{split} hub-state row indices changed")
        if _canonical_mask_sha256(panel) != EXPECTED_MASK_CANONICAL_SHA256[split]:
            raise ValueError(f"{split} canonical hash changed")
        validated[split] = panel
    return validated


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def build_component_position_protocol(
    masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Build the complete file-free protocol from the exact sealed masks."""
    validated = validate_bridge_masks(masks)
    return {
        "schema_version": SCHEMA_VERSION,
        "context_count": CONTEXT_COUNT,
        "component_count": COMPONENT_COUNT,
        "hub_component": HUB_COMPONENT,
        "mask_splits": list(MASK_SPLITS),
        "mask_shapes": _jsonable(MASK_SHAPES),
        "mask_cardinalities": dict(MASK_CARDINALITIES),
        "mask_hub_on_counts": dict(MASK_HUB_ON_COUNTS),
        "mask_canonical_sha256": dict(EXPECTED_MASK_CANONICAL_SHA256),
        "hub_state_row_indices": _jsonable(HUB_STATE_ROW_INDICES),
        "position_scope_order": list(POSITION_SCOPE_ORDER),
        "singleton_conditions": _jsonable(SINGLETON_CONDITIONS),
        "mask_rows": {
            split: _jsonable(_mask_rows(validated[split])) for split in MASK_SPLITS
        },
        "template_strata": _jsonable(TEMPLATE_STRATA),
        "bootstrap": dict(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
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


def validate_component_position_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any mutation of the fixed pure protocol or exact mask closure."""
    if not isinstance(protocol, Mapping):
        raise TypeError("component-position protocol must be a mapping")
    expected_keys = (
        "schema_version",
        "context_count",
        "component_count",
        "hub_component",
        "mask_splits",
        "mask_shapes",
        "mask_cardinalities",
        "mask_hub_on_counts",
        "mask_canonical_sha256",
        "hub_state_row_indices",
        "position_scope_order",
        "singleton_conditions",
        "mask_rows",
        "template_strata",
        "bootstrap",
        "scoring_gates",
        "query_accounting",
    )
    if tuple(protocol) != expected_keys:
        raise ValueError("component-position protocol keys or order changed")
    masks = validate_bridge_masks(_masks_from_rows(protocol["mask_rows"]))
    expected = build_component_position_protocol(masks)
    if protocol["scoring_gates"] != expected["scoring_gates"]:
        raise ValueError("component-position scoring gates changed")
    for key in expected_keys:
        if key != "scoring_gates" and protocol[key] != expected[key]:
            raise ValueError(f"component-position {key.replace('_', ' ')} changed")
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
        expected = (MASK_SHAPES[split][0], CONTEXT_COUNT)
        if tuple(table.shape) != expected or not bool(table.isfinite().all()):
            raise ValueError(f"{label}/{split} must be finite with shape {expected}")
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
    result["resampling"] = "contexts_within_four_fixed_template_strata"
    return result


def _crossed_bootstrap_mean(
    values: torch.Tensor,
    row_indices: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    selected = values[torch.tensor(row_indices, dtype=torch.long)]
    replicates = int(BOOTSTRAP["replicates"])
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randint(len(selected), (replicates, len(selected)), generator=generator)
    contexts = _context_samples(generator, replicates)
    estimates = selected[rows[:, :, None], contexts[:, None, :]].mean(dim=(1, 2))
    result = _bootstrap_summary(estimates, float(selected.mean()))
    result["resampling"] = str(BOOTSTRAP["resampling"])
    return result


def _joint_crossed_bootstrap_mean(
    values: Mapping[str, torch.Tensor],
    *,
    hub_state: str,
) -> dict[str, Any]:
    replicates = int(BOOTSTRAP["replicates"])
    generator = torch.Generator().manual_seed(int(BOOTSTRAP["seed"]))
    contexts = _context_samples(generator, replicates)
    scale_estimates = []
    scale_points = []
    for split in MASK_SPLITS:
        selected = values[split][
            torch.tensor(HUB_STATE_ROW_INDICES[split][hub_state], dtype=torch.long)
        ]
        rows = torch.randint(
            len(selected), (replicates, len(selected)), generator=generator
        )
        scale_estimates.append(
            selected[rows[:, :, None], contexts[:, None, :]].mean(dim=(1, 2))
        )
        scale_points.append(selected.mean())
    estimates = torch.stack(scale_estimates).mean(dim=0)
    point = float(torch.stack(scale_points).mean())
    result = _bootstrap_summary(estimates, point)
    result["resampling"] = str(BOOTSTRAP["resampling"])
    result["scale_weighting"] = str(BOOTSTRAP["joint_scale_weighting"])
    return result


def _scope_equivalence_claim(
    absolute_gap: torch.Tensor,
    *,
    split: str,
    hub_state: str,
    integrity_passed: bool,
) -> dict[str, Any]:
    indices = HUB_STATE_ROW_INDICES[split][hub_state]
    selected = absolute_gap[torch.tensor(indices, dtype=torch.long)]
    bootstrap = _crossed_bootstrap_mean(
        absolute_gap,
        indices,
        seed=int(BOOTSTRAP["seed"]),
    )
    thresholds = SCORING_GATES["scope_equivalence"]
    context_mae = selected.mean(dim=0)
    mask_mae = selected.mean(dim=1)
    template_mae = [
        float(selected[:, torch.tensor(stratum, dtype=torch.long)].mean())
        for stratum in TEMPLATE_STRATA
    ]
    passing_contexts = int((context_mae <= thresholds["context_mae_max"]).sum())
    passing_masks = int((mask_mae <= thresholds["mask_mae_max"]).sum())
    passing_templates = sum(
        value <= thresholds["template_mae_max"] for value in template_mae
    )
    passed = bool(
        integrity_passed
        and bootstrap["upper_95"]
        <= thresholds["paired_absolute_gap_bootstrap_upper_max"]
        and passing_contexts >= thresholds["minimum_passing_contexts"]
        and passing_templates >= thresholds["minimum_passing_templates"]
        and passing_masks >= len(indices) - thresholds["maximum_failing_masks"]
    )
    return {
        "paired_absolute_gap_bootstrap": bootstrap,
        "context_mae": [float(value) for value in context_mae],
        "passing_contexts": passing_contexts,
        "template_mae": template_mae,
        "passing_templates": passing_templates,
        "mask_mae": [float(value) for value in mask_mae],
        "passing_masks": passing_masks,
        "mask_count": len(indices),
        "passed": passed,
    }


def score_component_position_bridge(
    all_position_responses: Mapping[str, torch.Tensor],
    lexical_position_responses: Mapping[str, torch.Tensor],
    singleton_all_response: torch.Tensor,
    singleton_lexical_response: torch.Tensor,
    clean_endpoints: torch.Tensor,
    held_masks: Mapping[str, torch.Tensor],
    *,
    integrity_passed: bool,
) -> dict[str, Any]:
    """Score the fixed fresh-context singleton, per-scale, and joint bridge."""
    if not isinstance(integrity_passed, bool):
        raise TypeError("integrity_passed must be a bool")
    validate_bridge_masks(held_masks)
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
    absolute_gaps = {
        split: (lexical_normalized[split] - all_normalized[split]).abs()
        for split in MASK_SPLITS
    }
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

    anchor_thresholds = SCORING_GATES["all_position_anchor"]
    anchors = {}
    hub_off_claims = {}
    hub_on_claims = {}
    for split in MASK_SPLITS:
        on_indices = HUB_STATE_ROW_INDICES[split]["hub_on"]
        off_indices = HUB_STATE_ROW_INDICES[split]["hub_off"]
        hub_on_all = all_normalized[split][torch.tensor(on_indices)]
        on_endpoint = _crossed_bootstrap_mean(
            (all_normalized[split] - 1.0).abs(),
            on_indices,
            seed=int(BOOTSTRAP["seed"]),
        )
        off_endpoint = _crossed_bootstrap_mean(
            (all_normalized[split] - 1.0).abs(),
            off_indices,
            seed=int(BOOTSTRAP["seed"]),
        )
        anchor = {
            "hub_on_all_position_mean": float(hub_on_all.mean()),
            "hub_on_endpoint_absolute_gap_bootstrap": on_endpoint,
            "hub_off_endpoint_absolute_gap_bootstrap": off_endpoint,
        }
        anchor["passed"] = bool(
            integrity_passed
            and anchor_thresholds["hub_on_endpoint_mean_min"]
            <= anchor["hub_on_all_position_mean"]
            <= anchor_thresholds["hub_on_endpoint_mean_max"]
            and on_endpoint["upper_95"]
            <= anchor_thresholds[
                "hub_on_endpoint_absolute_gap_bootstrap_upper_max"
            ]
            and off_endpoint["lower_95"]
            >= anchor_thresholds[
                "hub_off_endpoint_absolute_gap_bootstrap_lower_min"
            ]
        )
        anchors[split] = anchor

        hub_off_claims[split] = _scope_equivalence_claim(
            absolute_gaps[split],
            split=split,
            hub_state="hub_off",
            integrity_passed=integrity_passed,
        )
        hub_on_claim = _scope_equivalence_claim(
            absolute_gaps[split],
            split=split,
            hub_state="hub_on",
            integrity_passed=integrity_passed,
        )
        hub_on_lexical = lexical_normalized[split][torch.tensor(on_indices)]
        lexical_endpoint = _crossed_bootstrap_mean(
            (lexical_normalized[split] - 1.0).abs(),
            on_indices,
            seed=int(BOOTSTRAP["seed"]),
        )
        endpoint_thresholds = SCORING_GATES["hub_on_lexical_endpoint"]
        hub_on_claim["lexical_endpoint_mean"] = float(hub_on_lexical.mean())
        hub_on_claim["lexical_endpoint_absolute_gap_bootstrap"] = lexical_endpoint
        hub_on_claim["passed"] = bool(
            hub_on_claim["passed"]
            and endpoint_thresholds["endpoint_mean_min"]
            <= hub_on_claim["lexical_endpoint_mean"]
            <= endpoint_thresholds["endpoint_mean_max"]
            and lexical_endpoint["upper_95"]
            <= endpoint_thresholds[
                "endpoint_absolute_gap_bootstrap_upper_max"
            ]
        )
        hub_on_claims[split] = hub_on_claim

    joint_threshold = SCORING_GATES["joint_scope_equivalence"][
        "paired_absolute_gap_bootstrap_upper_max"
    ]
    joint_off_bootstrap = _joint_crossed_bootstrap_mean(
        absolute_gaps, hub_state="hub_off"
    )
    joint_on_bootstrap = _joint_crossed_bootstrap_mean(
        absolute_gaps, hub_state="hub_on"
    )
    joint_off = {
        "paired_absolute_gap_bootstrap": joint_off_bootstrap,
        "passed": bool(
            integrity_passed and joint_off_bootstrap["upper_95"] <= joint_threshold
        ),
    }
    joint_on = {
        "paired_absolute_gap_bootstrap": joint_on_bootstrap,
        "passed": bool(
            integrity_passed and joint_on_bootstrap["upper_95"] <= joint_threshold
        ),
    }
    anchor_passed = all(claim["passed"] for claim in anchors.values())
    hub_off_per_scale_passed = all(
        claim["passed"] for claim in hub_off_claims.values()
    )
    hub_on_per_scale_passed = all(claim["passed"] for claim in hub_on_claims.values())
    promotion_passed = bool(
        integrity_passed
        and singleton_claim["passed"]
        and anchor_passed
        and hub_off_per_scale_passed
        and joint_off["passed"]
    )
    full_passed = bool(
        promotion_passed and hub_on_per_scale_passed and joint_on["passed"]
    )
    claims = {
        "singleton_transfer": singleton_claim,
        "all_position_anchor_by_scale": anchors,
        "hub_off_scope_bridge_by_scale": hub_off_claims,
        "hub_on_scope_control_by_scale": hub_on_claims,
        "joint_hub_off_scope_bridge": joint_off,
        "joint_hub_on_scope_control": joint_on,
        "hub_off_scale_promotion_authorized": {"passed": promotion_passed},
        "full_component_position_bridge": {"passed": full_passed},
    }
    return {
        "schema_version": "e29_14b_component_position_bridge_score_v1",
        "integrity_passed": integrity_passed,
        "claims": claims,
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "recommended_interpretation": (
            "fresh_context_c24_c32_component_position_bridge_confirmed"
            if full_passed
            else (
                "fresh_context_hub_off_c24_c32_component_position_bridge_"
                "confirmed_without_general_all_mask_equivalence"
                if promotion_passed
                else "inspect_independent_claims_without_scope_promotion"
            )
        ),
    }


__all__ = [
    "BOOTSTRAP",
    "COMPONENT_COUNT",
    "CONTEXT_COUNT",
    "EXPECTED_MASK_CANONICAL_SHA256",
    "HUB_COMPONENT",
    "HUB_STATE_ROW_INDICES",
    "MASK_CARDINALITIES",
    "MASK_HUB_ON_COUNTS",
    "MASK_SHAPES",
    "MASK_SPLITS",
    "POSITION_SCOPE_ORDER",
    "QUERY_ACCOUNTING",
    "SCHEMA_VERSION",
    "SCORING_GATES",
    "SINGLETON_CONDITIONS",
    "TEMPLATE_STRATA",
    "build_component_position_protocol",
    "score_component_position_bridge",
    "validate_bridge_masks",
    "validate_component_position_protocol",
]
