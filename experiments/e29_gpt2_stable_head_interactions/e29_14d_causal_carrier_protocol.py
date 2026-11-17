"""Pure response-blind protocol for prospective E29.14d carrier tests.

The module constructs and validates intervention masks and their scientific
contrast mappings. It performs no file I/O, model loading, or model calls.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
HUB_COMPONENT = 1
NON_HUB_COMPONENTS = tuple(
    component for component in range(COMPONENT_COUNT) if component != HUB_COMPONENT
)
DESIGN_DOMAIN = "e29_14d_class_balanced_addone_v1|29144764"


def _response_blind_component_order(
    component_class: str, components: Sequence[int]
) -> tuple[int, ...]:
    """Order a fixed component class solely by the registered hash domain."""
    return tuple(
        sorted(
            components,
            key=lambda component: hashlib.sha256(
                f"{DESIGN_DOMAIN}|{component_class}|{component}".encode("ascii")
            ).digest(),
        )
    )


ATTENTION_COMPONENT_ORDER = _response_blind_component_order(
    "attention", range(0, COMPONENT_COUNT, 2)
)
MLP_COMPONENT_ORDER = _response_blind_component_order(
    "mlp", range(3, COMPONENT_COUNT, 2)
)
TARGET_COMPONENT_ORDER = (*ATTENTION_COMPONENT_ORDER, *MLP_COMPONENT_ORDER)
ADD_ONE_FAMILY_IDS = ("family_0", "family_1", "family_2")
ATTENTION_SAME_CLASS_STARTS = (0, 8, 16)
MLP_SAME_CLASS_STARTS = (0, 7, 14)
CROSS_CLASS_STARTS = (0, 8, 16)
CORE_ATTENTION_COMPONENTS = (36, 38, 40, 42)
MATCHED_MLP_COMPONENTS = (37, 39, 41, 43)
FLANKING_ATTENTION_COMPONENTS = (28, 30, 32, 34, 44, 46)
REGIONAL_FLANKING_ATTENTION_COMPONENTS = (32, 34, 44, 46)
CANDIDATE_GROUPS = {
    "late_attention_core": CORE_ATTENTION_COMPONENTS,
    "matched_same_layer_mlp": MATCHED_MLP_COMPONENTS,
    "flanking_attention_controls": FLANKING_ATTENTION_COMPONENTS,
}
POSITION_SCOPE_ORDER = ("all_positions", "differing_lexical_positions")
PRIMARY_POSITION_SCOPE = POSITION_SCOPE_ORDER[0]
SECONDARY_POSITION_SCOPE = POSITION_SCOPE_ORDER[1]
TEMPLATE_STRATA = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
CONDITION_CARDINALITY_COUNTS = {
    1: 1,
    32: 153,
    33: 101,
    34: 11,
    46: 47,
    47: 1,
}
CONDITION_FAMILY_ORDER = (
    "class_balanced_add_one_scan",
    "same_layer_top4_factorial",
    "full_network_single_dropout",
    "hub_singleton_control",
)
EXPECTED_ADD_ONE_RECORDS_SHA256 = (
    "0071e9f8dd2acde40d0c9b06f5709f2703d6ec58441f2021d911cbbcaadcd981"
)
EXPECTED_FACTORIAL_RECORDS_SHA256 = (
    "644c7b96a9a38d846ec6324f5199ab7fde7994ab449c84da7ba59c88d9065c87"
)
EXPECTED_CONDITION_MASKS_SHA256 = (
    "41411add7c84132dc93c767e2f6c0f0733af20b566a2eeee79b05ca5b6b72792"
)
EXPECTED_CONDITION_MANIFEST_SHA256 = (
    "562a1a53949e78488038e2bd6b5b14b5ca5720b7d3d31f850ae26a51a6bd7934"
)
EXPECTED_QUERY_MANIFEST_SHA256 = (
    "2f0c006e84c25f16e519a84baeff9f8388b45940b0784a7d8449da6a5ea01e33"
)

BOOTSTRAP = {
    "seed": 29_144,
    "replicates": 10_000,
    "lower_quantile_one_sided": 0.05,
    "upper_quantile_one_sided": 0.95,
    "template_strata": [list(stratum) for stratum in TEMPLATE_STRATA],
    "fixed_design_interpretation": (
        "sensitivity_intervals_for_exact_components_backgrounds_and_context_panel"
    ),
    "paired_draws": (
        "one common deterministic bootstrap stream shared across compared "
        "components cells and scopes within each estimator"
    ),
}
SCORING_GATES = {
    "integrity_anchors": {
        "hub_singleton_normalized_mean_min": 0.985,
        "hub_singleton_normalized_mean_max": 1.01,
        "hub_singleton_absolute_endpoint_gap_bootstrap_upper_max": 0.03,
        "c47_full_normalized_mean_min": 0.95,
        "c47_full_normalized_mean_max": 1.01,
        "c47_full_absolute_endpoint_gap_bootstrap_upper_max": 0.05,
    },
    "scope_secondary": {
        "condition_response_mean_absolute_gap_bootstrap_upper_max": 0.03,
        "add_one_effect_mean_absolute_gap_bootstrap_upper_max": 0.03,
        "factorial_contrast_mean_absolute_gap_bootstrap_upper_max": 0.03,
        "dropout_effect_mean_absolute_gap_bootstrap_upper_max": 0.03,
        "context_mean_absolute_gap_max": 0.05,
        "minimum_passing_contexts": 11,
        "conjunctive_with_primary_causal_class_claim": False,
    },
    "balanced_add_one": {
        "attention_class_mean_effect_bootstrap_lower_strict_min": 0.005,
        "matched_layer_attention_minus_mlp_bootstrap_lower_strict_min": 0.005,
        "attention_vs_mlp_component_auc_bootstrap_lower_strict_min": 0.75,
        "minimum_positive_matched_layer_contrasts": 18,
        "matched_layer_contrast_count": 23,
    },
    "top4_factorial": {
        "metrics": [
            "attention_effect_mlp_absent_A0",
            "attention_effect_mlp_present_A1",
            "average_attention_minus_mlp_marginal",
        ],
        "simultaneous_lower_band_strict_min": 0.005,
        "required_attention_components": list(CORE_ATTENTION_COMPONENTS),
        "simultaneous_cells": 12,
        "one_simultaneous_max_error_lower_band_required": True,
        "interaction_contrast_is_descriptive_not_gate": True,
    },
    "regional_flanking_secondary": {
        "late_attention_components": list(CORE_ATTENTION_COMPONENTS),
        "regional_flanking_attention_components": list(
            REGIONAL_FLANKING_ATTENTION_COMPONENTS
        ),
        "late_minus_regional_flanking_bootstrap_lower_strict_min": 0.005,
        "components_28_and_30_are_descriptive_only": True,
        "conjunctive_with_primary_causal_class_claim": False,
    },
    "single_dropout_secondary": {
        "attention_class_mean_loss_bootstrap_lower_strict_min": 0.002,
        "matched_layer_attention_minus_mlp_loss_lower_strict_min": 0.002,
        "top4_individual_loss_simultaneous_lower_strict_min": 0.002,
        "top4_attention_minus_matched_mlp_loss_simultaneous_lower_strict_min": (0.002),
        "top4_individual_claim_requires_both_simultaneous_bounds": True,
        "conjunctive_with_primary_carrier_claim": False,
    },
}
QUERY_ACCOUNTING = {
    "contexts": CONTEXT_COUNT,
    "unique_condition_masks": 314,
    "position_scopes": len(POSITION_SCOPE_ORDER),
    "patched_condition_rows_per_context": 628,
    "patched_context_condition_cells": 7_536,
    "clean_forward_examples": 24,
    "maximum_logical_queries": 7_560,
}
PHYSICAL_QUERY_ACCOUNTING = {
    **QUERY_ACCOUNTING,
    "batch_smoke_contexts": 1,
    "batch_smoke_batch_sizes": (1, 32),
    "batch_smoke_unique_query_rows": 628,
    "batch_smoke_physical_condition_cells": 1_256,
    "maximum_physical_model_examples": 8_816,
}
RESPONSE_SCHEMA = {
    "scope_order": POSITION_SCOPE_ORDER,
    "condition_context_shape_per_scope": (314, CONTEXT_COUNT),
    "flattened_query_context_shape": (628, CONTEXT_COUNT),
    "normalization": "raw_shift_divided_by_positive_symmetric_clean_endpoint",
    "primary_scope": PRIMARY_POSITION_SCOPE,
    "secondary_scope": SECONDARY_POSITION_SCOPE,
}
SCHEMA_VERSION = "e29_14d_class_balanced_causal_carrier_response_blind_v1"


def _mask(selected: Sequence[int]) -> torch.Tensor:
    row = torch.zeros(COMPONENT_COUNT, dtype=torch.float64)
    row[list(selected)] = 1.0
    return row


def _rows(panel: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in torch.where(row == 1.0)[0]) for row in panel
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_rows_sha256(rows: Sequence[Sequence[int]]) -> str:
    payload = "\n".join(",".join(str(value) for value in row) for row in rows)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _same_class_selection(
    order: tuple[int, ...], target: int, start: int, modulus: int
) -> tuple[int, ...]:
    target_index = order.index(target)
    return tuple(
        order[(target_index + 1 + ((start + offset) % modulus)) % len(order)]
        for offset in range(16)
    )


def _raw_add_one_records() -> tuple[dict[str, Any], ...]:
    records = []
    for target in TARGET_COMPONENT_ORDER:
        for family_index, family_id in enumerate(ADD_ONE_FAMILY_IDS):
            if target in ATTENTION_COMPONENT_ORDER:
                target_index = ATTENTION_COMPONENT_ORDER.index(target)
                same = _same_class_selection(
                    ATTENTION_COMPONENT_ORDER,
                    target,
                    ATTENTION_SAME_CLASS_STARTS[family_index],
                    23,
                )
                cross = tuple(
                    MLP_COMPONENT_ORDER[
                        (target_index + CROSS_CLASS_STARTS[family_index] + offset)
                        % len(MLP_COMPONENT_ORDER)
                    ]
                    for offset in range(16)
                )
                target_class = "attention"
            else:
                target_index = MLP_COMPONENT_ORDER.index(target)
                same = _same_class_selection(
                    MLP_COMPONENT_ORDER,
                    target,
                    MLP_SAME_CLASS_STARTS[family_index],
                    22,
                )
                cross = tuple(
                    ATTENTION_COMPONENT_ORDER[
                        (target_index + CROSS_CLASS_STARTS[family_index] + offset)
                        % len(ATTENTION_COMPONENT_ORDER)
                    ]
                    for offset in range(16)
                )
                target_class = "mlp"
            base = tuple(sorted((*same, *cross)))
            records.append(
                {
                    "pair_id": f"add_one_component_{target:02d}_{family_id}",
                    "target_component": target,
                    "target_class": target_class,
                    "family_id": family_id,
                    "family_index": family_index,
                    "base_components": base,
                    "plus_components": tuple(sorted((*base, target))),
                }
            )
    return tuple(records)


def _raw_factorial_records() -> tuple[dict[str, Any], ...]:
    records = []
    for attention, mlp in zip(
        CORE_ATTENTION_COMPONENTS, MATCHED_MLP_COMPONENTS, strict=True
    ):
        for background_index, family_id in enumerate(ADD_ONE_FAMILY_IDS):
            attention_side = _same_class_selection(
                ATTENTION_COMPONENT_ORDER,
                attention,
                ATTENTION_SAME_CLASS_STARTS[background_index],
                23,
            )
            mlp_side = _same_class_selection(
                MLP_COMPONENT_ORDER,
                mlp,
                MLP_SAME_CLASS_STARTS[background_index],
                22,
            )
            neither = tuple(sorted((*attention_side, *mlp_side)))
            records.append(
                {
                    "factorial_id": (
                        f"factorial_layer_{attention // 2:02d}_{family_id}"
                    ),
                    "attention_component": attention,
                    "mlp_component": mlp,
                    "layer": attention // 2,
                    "background_index": background_index,
                    "family_id": family_id,
                    "cells": {
                        "neither": neither,
                        "attention_only": tuple(sorted((*neither, attention))),
                        "mlp_only": tuple(sorted((*neither, mlp))),
                        "both": tuple(sorted((*neither, attention, mlp))),
                    },
                }
            )
    return tuple(records)


def _record_design_hashes() -> tuple[str, str]:
    add_lines = "\n".join(
        "|".join(
            (
                str(record["target_component"]),
                str(record["family_index"]),
                ",".join(str(value) for value in record["base_components"]),
                ",".join(str(value) for value in record["plus_components"]),
            )
        )
        for record in _raw_add_one_records()
    )
    factorial_lines = "\n".join(
        "|".join(
            (
                str(record["attention_component"]),
                str(record["mlp_component"]),
                str(record["background_index"]),
                *(
                    ",".join(str(value) for value in record["cells"][cell])
                    for cell in ("neither", "attention_only", "mlp_only", "both")
                ),
            )
        )
        for record in _raw_factorial_records()
    )
    return (
        hashlib.sha256(add_lines.encode("ascii")).hexdigest(),
        hashlib.sha256(factorial_lines.encode("ascii")).hexdigest(),
    )


def build_balanced_carrier_panels() -> dict[str, dict[str, torch.Tensor]]:
    """Return three exact class-balanced c32/base and c33/add-one panels."""
    records = _raw_add_one_records()
    return {
        family_id: {
            "c32_base": torch.stack(
                [
                    _mask(record["base_components"])
                    for record in records
                    if record["family_id"] == family_id
                ]
            ),
            "c33_add_one": torch.stack(
                [
                    _mask(record["plus_components"])
                    for record in records
                    if record["family_id"] == family_id
                ]
            ),
        }
        for family_id in ADD_ONE_FAMILY_IDS
    }


def build_factorial_panels() -> dict[str, torch.Tensor]:
    """Return twelve same-layer 2x2 factorials in fixed record order."""
    records = _raw_factorial_records()
    return {
        cell: torch.stack([_mask(record["cells"][cell]) for record in records])
        for cell in ("neither", "attention_only", "mlp_only", "both")
    }


def build_full_dropout_panels() -> dict[str, torch.Tensor]:
    """Return the c47 all-non-hub reference and 47 c46 dropouts."""
    full = tuple(sorted(NON_HUB_COMPONENTS))
    return {
        "c47_full": _mask(full)[None],
        "c46_leave_one_out": torch.stack(
            [
                _mask(tuple(value for value in full if value != target))
                for target in TARGET_COMPONENT_ORDER
            ]
        ),
    }


def _condition_blueprints() -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    indices: dict[tuple[int, ...], int] = {}

    def intern(selected: tuple[int, ...], condition_id: str, origin: str) -> int:
        if selected in indices:
            return indices[selected]
        index = len(records)
        indices[selected] = index
        records.append(
            {
                "condition_index": index,
                "condition_id": condition_id,
                "first_origin_family": origin,
                "cardinality": len(selected),
                "selected_components": selected,
            }
        )
        return index

    for record in _raw_add_one_records():
        prefix = (
            f"scan_component_{record['target_component']:02d}_{record['family_id']}"
        )
        intern(
            record["base_components"],
            f"{prefix}_c32_base",
            CONDITION_FAMILY_ORDER[0],
        )
        intern(
            record["plus_components"],
            f"{prefix}_c33_plus",
            CONDITION_FAMILY_ORDER[0],
        )
    for record in _raw_factorial_records():
        for cell in ("neither", "attention_only", "mlp_only", "both"):
            intern(
                record["cells"][cell],
                f"{record['factorial_id']}_{cell}",
                CONDITION_FAMILY_ORDER[1],
            )
    full = tuple(sorted(NON_HUB_COMPONENTS))
    intern(full, "full_non_hub_c47", CONDITION_FAMILY_ORDER[2])
    for target in TARGET_COMPONENT_ORDER:
        intern(
            tuple(value for value in full if value != target),
            f"full_drop_component_{target:02d}_c46",
            CONDITION_FAMILY_ORDER[2],
        )
    intern((HUB_COMPONENT,), "hub_component_01_singleton", CONDITION_FAMILY_ORDER[3])
    return tuple(records)


def build_condition_manifest() -> tuple[dict[str, Any], ...]:
    """Return the exact ordered manifest of 314 unique condition masks."""
    return deepcopy(_condition_blueprints())


def build_condition_masks() -> torch.Tensor:
    """Return the exact ordered 314-by-48 float64 condition-mask tensor."""
    return torch.stack(
        [_mask(record["selected_components"]) for record in _condition_blueprints()]
    )


def _condition_lookup() -> tuple[
    dict[tuple[int, ...], int], tuple[dict[str, Any], ...]
]:
    manifest = _condition_blueprints()
    return (
        {
            record["selected_components"]: record["condition_index"]
            for record in manifest
        },
        manifest,
    )


def build_add_one_records() -> tuple[dict[str, Any], ...]:
    """Map all 141 scientific add-one pairs onto deduplicated conditions."""
    lookup, manifest = _condition_lookup()
    output = []
    for record in _raw_add_one_records():
        base_index = lookup[record["base_components"]]
        plus_index = lookup[record["plus_components"]]
        output.append(
            {
                key: record[key]
                for key in (
                    "pair_id",
                    "target_component",
                    "target_class",
                    "family_id",
                    "family_index",
                )
            }
            | {
                "base_condition_index": base_index,
                "plus_condition_index": plus_index,
                "base_condition_id": manifest[base_index]["condition_id"],
                "plus_condition_id": manifest[plus_index]["condition_id"],
            }
        )
    return tuple(output)


def build_factorial_records() -> tuple[dict[str, Any], ...]:
    """Map twelve 2x2 factorials onto the deduplicated conditions."""
    lookup, manifest = _condition_lookup()
    output = []
    for record in _raw_factorial_records():
        cell_indices = {
            cell: lookup[selected] for cell, selected in record["cells"].items()
        }
        output.append(
            {
                key: record[key]
                for key in (
                    "factorial_id",
                    "attention_component",
                    "mlp_component",
                    "layer",
                    "background_index",
                    "family_id",
                )
            }
            | {
                "condition_indices": cell_indices,
                "condition_ids": {
                    cell: manifest[index]["condition_id"]
                    for cell, index in cell_indices.items()
                },
            }
        )
    return tuple(output)


def build_dropout_records() -> tuple[dict[str, Any], ...]:
    """Map all 47 full-versus-single-dropout contrasts to conditions."""
    lookup, manifest = _condition_lookup()
    full = tuple(sorted(NON_HUB_COMPONENTS))
    full_index = lookup[full]
    output = []
    for target in TARGET_COMPONENT_ORDER:
        dropout = tuple(value for value in full if value != target)
        dropout_index = lookup[dropout]
        output.append(
            {
                "dropout_id": f"dropout_component_{target:02d}",
                "dropped_component": target,
                "dropped_class": "attention" if target % 2 == 0 else "mlp",
                "full_condition_index": full_index,
                "dropout_condition_index": dropout_index,
                "full_condition_id": manifest[full_index]["condition_id"],
                "dropout_condition_id": manifest[dropout_index]["condition_id"],
            }
        )
    return tuple(output)


def query_scope_indices() -> dict[str, tuple[int, ...]]:
    """Return ordered *condition* indices evaluated under each scope."""
    indices = tuple(range(QUERY_ACCOUNTING["unique_condition_masks"]))
    return {scope: indices for scope in POSITION_SCOPE_ORDER}


def build_query_manifest() -> tuple[dict[str, Any], ...]:
    """Flatten scope-major queries: all-position first, then lexical."""
    conditions = build_condition_manifest()
    queries = []
    for scope in POSITION_SCOPE_ORDER:
        for condition_index in query_scope_indices()[scope]:
            condition = conditions[condition_index]
            queries.append(
                {
                    "query_index": len(queries),
                    "condition_index": condition_index,
                    "query_id": f"{scope}::{condition['condition_id']}",
                    "condition_id": condition["condition_id"],
                    "position_scope": scope,
                }
            )
    return tuple(queries)


def validate_condition_masks(masks: torch.Tensor) -> torch.Tensor:
    """Require exact CPU-float64 ordered closure of all 314 conditions."""
    if not isinstance(masks, torch.Tensor):
        raise TypeError("E29.14d condition masks must be a tensor")
    if masks.device.type != "cpu" or masks.dtype != torch.float64:
        raise ValueError("E29.14d condition masks must be CPU float64")
    panel = masks.detach().clone()
    if tuple(panel.shape) != (314, COMPONENT_COUNT):
        raise ValueError("E29.14d condition-mask shape changed")
    if not bool(torch.logical_or(panel == 0.0, panel == 1.0).all()):
        raise ValueError("E29.14d condition masks must be binary")
    expected = build_condition_masks()
    if not torch.equal(panel, expected):
        raise ValueError("E29.14d condition-mask rows or order changed")
    if len(set(_rows(panel))) != len(panel):
        raise ValueError("E29.14d condition masks are not deduplicated")
    if Counter(int(row.sum()) for row in panel) != Counter(
        CONDITION_CARDINALITY_COUNTS
    ):
        raise ValueError("E29.14d condition cardinalities changed")
    observed_hash = _ordered_rows_sha256(_rows(panel))
    if observed_hash != EXPECTED_CONDITION_MASKS_SHA256:
        raise ValueError("E29.14d condition-mask canonical hash changed")
    return panel


def validate_no_prior_c32_c33_overlap(
    masks: torch.Tensor, prior_panels: Sequence[torch.Tensor]
) -> dict[str, int]:
    """Reject overlap of current c32/c33 rows with supplied prior panels."""
    current = validate_condition_masks(masks)
    if not isinstance(prior_panels, Sequence):
        raise TypeError("prior mask panels must be a sequence")
    prior_rows: set[tuple[int, ...]] = set()
    for panel in prior_panels:
        if not isinstance(panel, torch.Tensor):
            raise TypeError("each prior mask panel must be a tensor")
        value = panel.detach().cpu()
        if value.ndim != 2 or value.shape[1] != COMPONENT_COUNT:
            raise ValueError("a prior mask panel has invalid shape")
        if not bool(torch.logical_or(value == 0, value == 1).all()):
            raise ValueError("a prior mask panel is not binary")
        prior_rows.update(row for row in _rows(value) if len(row) in (32, 33))
    current_rows = {row for row in _rows(current) if len(row) in (32, 33)}
    overlap = current_rows & prior_rows
    if overlap:
        raise ValueError("E29.14d masks overlap prior c32/c33 masks")
    return {
        "current_c32_c33_unique_rows": len(current_rows),
        "prior_c32_c33_unique_rows": len(prior_rows),
        "overlap_count": 0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def build_causal_carrier_protocol() -> dict[str, Any]:
    """Build the complete file-free serializable protocol."""
    add_hash, factorial_hash = _record_design_hashes()
    if add_hash != EXPECTED_ADD_ONE_RECORDS_SHA256:
        raise RuntimeError("E29.14d add-one construction hash changed")
    if factorial_hash != EXPECTED_FACTORIAL_RECORDS_SHA256:
        raise RuntimeError("E29.14d factorial construction hash changed")
    masks = build_condition_masks()
    manifest = build_condition_manifest()
    if _ordered_rows_sha256(_rows(masks)) != EXPECTED_CONDITION_MASKS_SHA256:
        raise RuntimeError("E29.14d condition-mask canonical hash changed")
    if _canonical_sha256(_jsonable(manifest)) != EXPECTED_CONDITION_MANIFEST_SHA256:
        raise RuntimeError("E29.14d condition-manifest canonical hash changed")
    queries = build_query_manifest()
    if _canonical_sha256(_jsonable(queries)) != EXPECTED_QUERY_MANIFEST_SHA256:
        raise RuntimeError("E29.14d query-manifest canonical hash changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "context_count": CONTEXT_COUNT,
        "component_count": COMPONENT_COUNT,
        "hub_component": HUB_COMPONENT,
        "non_hub_components": list(NON_HUB_COMPONENTS),
        "attention_component_order": list(ATTENTION_COMPONENT_ORDER),
        "mlp_component_order": list(MLP_COMPONENT_ORDER),
        "target_component_order": list(TARGET_COMPONENT_ORDER),
        "design_domain": DESIGN_DOMAIN,
        "add_one_family_ids": list(ADD_ONE_FAMILY_IDS),
        "attention_same_class_starts": list(ATTENTION_SAME_CLASS_STARTS),
        "mlp_same_class_starts": list(MLP_SAME_CLASS_STARTS),
        "cross_class_starts": list(CROSS_CLASS_STARTS),
        "candidate_groups": _jsonable(CANDIDATE_GROUPS),
        "position_scope_order": list(POSITION_SCOPE_ORDER),
        "primary_position_scope": PRIMARY_POSITION_SCOPE,
        "condition_family_order": list(CONDITION_FAMILY_ORDER),
        "condition_cardinality_counts": {
            str(key): value for key, value in CONDITION_CARDINALITY_COUNTS.items()
        },
        "condition_mask_rows": _jsonable(_rows(masks)),
        "condition_masks_sha256": EXPECTED_CONDITION_MASKS_SHA256,
        "condition_manifest": _jsonable(manifest),
        "condition_manifest_sha256": EXPECTED_CONDITION_MANIFEST_SHA256,
        "add_one_records": _jsonable(build_add_one_records()),
        "add_one_records_sha256": EXPECTED_ADD_ONE_RECORDS_SHA256,
        "factorial_records": _jsonable(build_factorial_records()),
        "factorial_records_sha256": EXPECTED_FACTORIAL_RECORDS_SHA256,
        "dropout_records": _jsonable(build_dropout_records()),
        "query_manifest": _jsonable(queries),
        "query_manifest_sha256": EXPECTED_QUERY_MANIFEST_SHA256,
        "template_strata": _jsonable(TEMPLATE_STRATA),
        "bootstrap": _jsonable(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "response_schema": _jsonable(RESPONSE_SCHEMA),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "claim_boundaries": {
            "e29_14d_responses_used_to_construct_masks_or_gates": False,
            "parent_posthoc_responses_motivated_attention_class_hypothesis": True,
            "parent_posthoc_responses_selected_top4_factorial_components": True,
            "parent_responses_used_for_hash_balanced_scan_background_layouts": False,
            "parent_responses_used_to_set_numeric_gates": False,
            "model_calls_in_protocol_module": False,
            "explicit_component_orders_are_authoritative": True,
            "design_domain_is_nonbehavioral_provenance_metadata": True,
            "fixed_component_and_context_panel_only": True,
            "perturbation_norm_role": (
                "mandatory_diagnostic_not_a_gate_or_matching_lane"
            ),
            "individual_add_one_effects_are_background_conditional": True,
            "dropout_claim_is_independent_secondary": True,
        },
    }


def validate_causal_carrier_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any mutation of the exact pure E29.14d protocol."""
    if not isinstance(protocol, Mapping):
        raise TypeError("E29.14d protocol must be a mapping")
    expected = build_causal_carrier_protocol()
    if tuple(protocol) != tuple(expected):
        raise ValueError("E29.14d protocol keys or order changed")
    if protocol.get("scoring_gates") != expected["scoring_gates"]:
        raise ValueError("E29.14d scoring gates changed")
    for key, value in expected.items():
        if key != "scoring_gates" and protocol.get(key) != value:
            raise ValueError(f"E29.14d protocol {key.replace('_', ' ')} changed")
    return expected


__all__ = [
    "ADD_ONE_FAMILY_IDS",
    "ATTENTION_COMPONENT_ORDER",
    "BOOTSTRAP",
    "CANDIDATE_GROUPS",
    "COMPONENT_COUNT",
    "CONDITION_CARDINALITY_COUNTS",
    "CONDITION_FAMILY_ORDER",
    "CONTEXT_COUNT",
    "DESIGN_DOMAIN",
    "EXPECTED_ADD_ONE_RECORDS_SHA256",
    "EXPECTED_CONDITION_MANIFEST_SHA256",
    "EXPECTED_CONDITION_MASKS_SHA256",
    "EXPECTED_FACTORIAL_RECORDS_SHA256",
    "EXPECTED_QUERY_MANIFEST_SHA256",
    "HUB_COMPONENT",
    "MLP_COMPONENT_ORDER",
    "NON_HUB_COMPONENTS",
    "PHYSICAL_QUERY_ACCOUNTING",
    "POSITION_SCOPE_ORDER",
    "PRIMARY_POSITION_SCOPE",
    "QUERY_ACCOUNTING",
    "REGIONAL_FLANKING_ATTENTION_COMPONENTS",
    "RESPONSE_SCHEMA",
    "SCHEMA_VERSION",
    "SCORING_GATES",
    "TARGET_COMPONENT_ORDER",
    "build_add_one_records",
    "build_balanced_carrier_panels",
    "build_causal_carrier_protocol",
    "build_condition_manifest",
    "build_condition_masks",
    "build_dropout_records",
    "build_factorial_panels",
    "build_factorial_records",
    "build_full_dropout_panels",
    "build_query_manifest",
    "query_scope_indices",
    "validate_causal_carrier_protocol",
    "validate_condition_masks",
    "validate_no_prior_c32_c33_overlap",
]
