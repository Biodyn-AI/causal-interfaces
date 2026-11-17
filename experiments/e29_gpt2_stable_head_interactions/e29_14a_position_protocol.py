"""Pure response-blind protocol and scoring primitives for E29.14a."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

CONTEXT_COUNT = 12
HUB_COMPONENT = 1
CONTROL_COMPONENT = 0
POSITION_SCOPE_NAMES = (
    "all_positions",
    "differing_lexical_positions",
    "shared_static_positions",
    "demonstration_target_position",
    "demonstration_prefix_positions",
    "repeated_query_positions",
    "differing_except_demonstration_target",
)


def _condition(
    name: str,
    *,
    component: int,
    scope: str,
    alpha: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "component": component,
        "scope": scope,
        "alpha": alpha,
    }


FORWARD_CONDITIONS = (
    _condition(
        "hub_all_positions_alpha1",
        component=HUB_COMPONENT,
        scope="all_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_differing_lexical_alpha1",
        component=HUB_COMPONENT,
        scope="differing_lexical_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_shared_static_alpha1",
        component=HUB_COMPONENT,
        scope="shared_static_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_demonstration_target_alpha1",
        component=HUB_COMPONENT,
        scope="demonstration_target_position",
        alpha=1.0,
    ),
    _condition(
        "hub_demonstration_prefix_alpha1",
        component=HUB_COMPONENT,
        scope="demonstration_prefix_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_repeated_query_alpha1",
        component=HUB_COMPONENT,
        scope="repeated_query_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_differing_except_target_alpha1",
        component=HUB_COMPONENT,
        scope="differing_except_demonstration_target",
        alpha=1.0,
    ),
    _condition(
        "hub_all_positions_alpha025",
        component=HUB_COMPONENT,
        scope="all_positions",
        alpha=0.25,
    ),
    _condition(
        "hub_all_positions_alpha050",
        component=HUB_COMPONENT,
        scope="all_positions",
        alpha=0.50,
    ),
    _condition(
        "hub_all_positions_alpha075",
        component=HUB_COMPONENT,
        scope="all_positions",
        alpha=0.75,
    ),
    _condition(
        "control_all_positions_alpha1",
        component=CONTROL_COMPONENT,
        scope="all_positions",
        alpha=1.0,
    ),
    _condition(
        "control_differing_lexical_alpha1",
        component=CONTROL_COMPONENT,
        scope="differing_lexical_positions",
        alpha=1.0,
    ),
)

REVERSE_CONDITIONS = (
    _condition(
        "hub_all_positions_alpha1",
        component=HUB_COMPONENT,
        scope="all_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_differing_lexical_alpha1",
        component=HUB_COMPONENT,
        scope="differing_lexical_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_shared_static_alpha1",
        component=HUB_COMPONENT,
        scope="shared_static_positions",
        alpha=1.0,
    ),
    _condition(
        "hub_demonstration_target_alpha1",
        component=HUB_COMPONENT,
        scope="demonstration_target_position",
        alpha=1.0,
    ),
)

TEMPLATE_STRATA = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
)
BOOTSTRAP = {
    "seed": 29_141,
    "replicates": 5_000,
    "resampling": "contexts_within_four_fixed_template_strata",
    "upper_quantile_one_sided": 0.95,
    "lower_quantile_descriptive": 0.025,
}
SCORING_GATES = {
    "replication": {
        "forward_all_mean_min": 0.985,
        "forward_endpoint_absolute_gap_bootstrap_upper_max": 0.03,
    },
    "forward_lexical_localization": {
        "paired_absolute_gap_bootstrap_upper_max": 0.03,
        "shared_static_mean_absolute_max": 0.02,
    },
    "component_specificity_control": {
        "maximum_control_mean_absolute_max": 0.05,
    },
    "bidirectional_switch": {
        "reverse_all_mean_min": 0.985,
        "reverse_endpoint_absolute_gap_bootstrap_upper_max": 0.03,
        "forward_reverse_absolute_gap_bootstrap_upper_max": 0.04,
    },
    "bidirectional_lexical_localization": {
        "reverse_lexical_absolute_gap_bootstrap_upper_max": 0.03,
        "reverse_shared_static_mean_absolute_max": 0.02,
    },
    "graded_dose_response": {
        "minimum_strictly_monotone_contexts": 10,
        "midpoint_mean_min_exclusive": 0.20,
        "midpoint_mean_max_exclusive": 0.80,
    },
}
QUERY_ACCOUNTING = {
    "contexts": CONTEXT_COUNT,
    "forward_condition_rows_per_context": len(FORWARD_CONDITIONS),
    "reverse_condition_rows_per_context": len(REVERSE_CONDITIONS),
    "patched_context_condition_cells": CONTEXT_COUNT
    * (len(FORWARD_CONDITIONS) + len(REVERSE_CONDITIONS)),
    "clean_forward_examples": 2 * CONTEXT_COUNT,
    "maximum_logical_queries": CONTEXT_COUNT
    * (len(FORWARD_CONDITIONS) + len(REVERSE_CONDITIONS) + 2),
}
SCHEMA_VERSION = "e29_14a_position_semantics_response_blind_v1"


def build_position_scopes(
    base_token_ids: torch.Tensor,
    donor_token_ids: torch.Tensor,
    *,
    base_target_token_id: int,
) -> dict[str, torch.Tensor]:
    """Return the exact aligned token-position partition used by E29.14a."""
    base = torch.as_tensor(base_token_ids, dtype=torch.long).detach().cpu()
    donor = torch.as_tensor(donor_token_ids, dtype=torch.long).detach().cpu()
    if base.ndim != 1 or donor.ndim != 1 or base.shape != donor.shape:
        raise ValueError(
            "base and donor token ids must be aligned one-dimensional rows"
        )
    if len(base) < 2:
        raise ValueError("prompt must contain at least two token positions")
    target = base == int(base_target_token_id)
    if int(target.sum()) != 1:
        raise ValueError("the base demonstration target must occur exactly once")
    target_position = int(torch.where(target)[0][0])
    differing = base != donor
    if not bool(differing[target_position]):
        raise ValueError("the donor token must differ at the target position")
    position = torch.arange(len(base))
    demonstration_prefix = differing & (position < target_position)
    repeated_query = differing & (position > target_position)
    if not bool(demonstration_prefix.any()) or not bool(repeated_query.any()):
        raise ValueError("differing positions must lie on both sides of the target")
    if not torch.equal(demonstration_prefix | target | repeated_query, differing):
        raise ValueError("differing positions do not form the required partition")
    return {
        "all_positions": torch.ones(len(base), dtype=torch.bool),
        "differing_lexical_positions": differing,
        "shared_static_positions": ~differing,
        "demonstration_target_position": target,
        "demonstration_prefix_positions": demonstration_prefix,
        "repeated_query_positions": repeated_query,
        "differing_except_demonstration_target": differing & ~target,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def build_position_protocol() -> dict[str, Any]:
    """Return the complete pure protocol payload for registration."""
    return {
        "schema_version": SCHEMA_VERSION,
        "context_count": CONTEXT_COUNT,
        "hub_component": HUB_COMPONENT,
        "control_component": CONTROL_COMPONENT,
        "position_scope_names": list(POSITION_SCOPE_NAMES),
        "forward_conditions": _jsonable(FORWARD_CONDITIONS),
        "reverse_conditions": _jsonable(REVERSE_CONDITIONS),
        "template_strata": _jsonable(TEMPLATE_STRATA),
        "bootstrap": dict(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": dict(QUERY_ACCOUNTING),
        "alpha_zero_role": "derived_clean_recipient_not_a_patched_query",
    }


def validate_position_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any mutation of the fixed E29.14a pure protocol."""
    if not isinstance(protocol, Mapping):
        raise TypeError("position protocol must be a mapping")
    expected = build_position_protocol()
    if set(protocol) != set(expected):
        raise ValueError("position protocol keys changed")
    for key in expected:
        if _jsonable(protocol[key]) != expected[key]:
            label = key.replace("_", " ")
            raise ValueError(f"{label} changed")
    return expected


def stratified_paired_bootstrap(
    paired_values: torch.Tensor,
    *,
    seed: int = int(BOOTSTRAP["seed"]),
    replicates: int = int(BOOTSTRAP["replicates"]),
) -> dict[str, Any]:
    """Bootstrap one paired context statistic within the four template strata."""
    values = torch.as_tensor(paired_values, dtype=torch.float64).detach().cpu()
    if values.shape != (CONTEXT_COUNT,) or not bool(values.isfinite().all()):
        raise ValueError("paired values must be a finite 12-context vector")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("bootstrap seed must be an integer")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("bootstrap replicates must be a positive integer")
    generator = torch.Generator().manual_seed(seed)
    estimates = torch.empty(replicates, dtype=torch.float64)
    strata = tuple(
        torch.tensor(indices, dtype=torch.long) for indices in TEMPLATE_STRATA
    )
    for replicate in range(replicates):
        selected = torch.cat(
            [
                stratum[
                    torch.randint(
                        len(stratum),
                        (len(stratum),),
                        generator=generator,
                    )
                ]
                for stratum in strata
            ]
        )
        estimates[replicate] = values[selected].mean()
    lower, upper = torch.quantile(
        estimates,
        torch.tensor(
            (
                float(BOOTSTRAP["lower_quantile_descriptive"]),
                float(BOOTSTRAP["upper_quantile_one_sided"]),
            ),
            dtype=torch.float64,
        ),
    )
    return {
        "point_estimate": float(values.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "lower_quantile_descriptive": float(BOOTSTRAP["lower_quantile_descriptive"]),
        "upper_quantile_one_sided": float(BOOTSTRAP["upper_quantile_one_sided"]),
        "seed": seed,
        "replicates": replicates,
        "resampling": str(BOOTSTRAP["resampling"]),
    }


def _finite_matrix(value: Any, *, shape: tuple[int, ...], label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if tensor.shape != shape or not bool(tensor.isfinite().all()):
        raise ValueError(f"{label} must be a finite tensor with shape {shape}")
    return tensor


def _indices(conditions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(condition["name"]): index for index, condition in enumerate(conditions)}


def _summary(values: torch.Tensor) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "mean_absolute": float(values.abs().mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "per_context": [float(value) for value in values],
    }


def score_position_semantics(
    forward_responses: torch.Tensor,
    reverse_responses: torch.Tensor,
    clean_endpoints: torch.Tensor,
    *,
    integrity_passed: bool,
) -> dict[str, Any]:
    """Score the six independent E29.14a claims without adaptive fitting."""
    forward = _finite_matrix(
        forward_responses,
        shape=(CONTEXT_COUNT, len(FORWARD_CONDITIONS)),
        label="forward responses",
    )
    reverse = _finite_matrix(
        reverse_responses,
        shape=(CONTEXT_COUNT, len(REVERSE_CONDITIONS)),
        label="reverse responses",
    )
    endpoints = _finite_matrix(
        clean_endpoints, shape=(CONTEXT_COUNT,), label="clean endpoints"
    )
    if bool((endpoints <= 0.0).any()):
        raise ValueError("clean endpoints must be positive")
    if not isinstance(integrity_passed, bool):
        raise TypeError("integrity_passed must be Boolean")
    forward = forward / endpoints[:, None]
    reverse = reverse / endpoints[:, None]
    fi = _indices(FORWARD_CONDITIONS)
    ri = _indices(REVERSE_CONDITIONS)

    f_all = forward[:, fi["hub_all_positions_alpha1"]]
    f_lexical = forward[:, fi["hub_differing_lexical_alpha1"]]
    f_static = forward[:, fi["hub_shared_static_alpha1"]]
    r_all = reverse[:, ri["hub_all_positions_alpha1"]]
    r_lexical = reverse[:, ri["hub_differing_lexical_alpha1"]]
    r_static = reverse[:, ri["hub_shared_static_alpha1"]]
    forward_endpoint_bootstrap = stratified_paired_bootstrap((f_all - 1.0).abs())
    forward_lexical_bootstrap = stratified_paired_bootstrap((f_lexical - f_all).abs())
    reverse_endpoint_bootstrap = stratified_paired_bootstrap((r_all - 1.0).abs())
    direction_bootstrap = stratified_paired_bootstrap((f_all - r_all).abs())
    reverse_lexical_bootstrap = stratified_paired_bootstrap((r_lexical - r_all).abs())

    replication_passed = (
        float(f_all.mean()) >= SCORING_GATES["replication"]["forward_all_mean_min"]
        and forward_endpoint_bootstrap["upper_95"]
        <= SCORING_GATES["replication"][
            "forward_endpoint_absolute_gap_bootstrap_upper_max"
        ]
        and integrity_passed
    )
    localization_passed = (
        forward_lexical_bootstrap["upper_95"]
        <= SCORING_GATES["forward_lexical_localization"][
            "paired_absolute_gap_bootstrap_upper_max"
        ]
        and float(f_static.abs().mean())
        <= SCORING_GATES["forward_lexical_localization"][
            "shared_static_mean_absolute_max"
        ]
        and integrity_passed
    )
    control_values = torch.stack(
        (
            forward[:, fi["control_all_positions_alpha1"]],
            forward[:, fi["control_differing_lexical_alpha1"]],
        )
    )
    control_maximum_mean_absolute = float(control_values.abs().mean(dim=1).max())
    control_passed = (
        control_maximum_mean_absolute
        <= SCORING_GATES["component_specificity_control"][
            "maximum_control_mean_absolute_max"
        ]
        and integrity_passed
    )
    bidirectional_passed = (
        float(r_all.mean())
        >= SCORING_GATES["bidirectional_switch"]["reverse_all_mean_min"]
        and reverse_endpoint_bootstrap["upper_95"]
        <= SCORING_GATES["bidirectional_switch"][
            "reverse_endpoint_absolute_gap_bootstrap_upper_max"
        ]
        and direction_bootstrap["upper_95"]
        <= SCORING_GATES["bidirectional_switch"][
            "forward_reverse_absolute_gap_bootstrap_upper_max"
        ]
        and integrity_passed
    )
    bidirectional_lexical_passed = (
        localization_passed
        and bidirectional_passed
        and reverse_lexical_bootstrap["upper_95"]
        <= SCORING_GATES["bidirectional_lexical_localization"][
            "reverse_lexical_absolute_gap_bootstrap_upper_max"
        ]
        and float(r_static.abs().mean())
        <= SCORING_GATES["bidirectional_lexical_localization"][
            "reverse_shared_static_mean_absolute_max"
        ]
    )

    positive_doses = torch.stack(
        (
            forward[:, fi["hub_all_positions_alpha025"]],
            forward[:, fi["hub_all_positions_alpha050"]],
            forward[:, fi["hub_all_positions_alpha075"]],
            f_all,
        ),
        dim=1,
    )
    doses = torch.cat((torch.zeros(CONTEXT_COUNT, 1), positive_doses), dim=1)
    dose_means = doses.mean(dim=0)
    aggregate_monotone = bool((dose_means[1:] > dose_means[:-1]).all())
    monotone_contexts = int((doses[:, 1:] > doses[:, :-1]).all(dim=1).sum())
    midpoint_mean = float(dose_means[2])
    dose_passed = (
        aggregate_monotone
        and monotone_contexts
        >= SCORING_GATES["graded_dose_response"]["minimum_strictly_monotone_contexts"]
        and SCORING_GATES["graded_dose_response"]["midpoint_mean_min_exclusive"]
        < midpoint_mean
        < SCORING_GATES["graded_dose_response"]["midpoint_mean_max_exclusive"]
        and integrity_passed
    )

    claims = {
        "replication": {
            "forward_all_mean": float(f_all.mean()),
            "forward_endpoint_absolute_gap_bootstrap": forward_endpoint_bootstrap,
            "passed": replication_passed,
        },
        "forward_lexical_localization": {
            "paired_absolute_gap_bootstrap": forward_lexical_bootstrap,
            "shared_static_mean_absolute": float(f_static.abs().mean()),
            "passed": localization_passed,
        },
        "component_specificity_control": {
            "maximum_control_mean_absolute": control_maximum_mean_absolute,
            "passed": control_passed,
        },
        "bidirectional_switch": {
            "reverse_all_mean": float(r_all.mean()),
            "reverse_endpoint_absolute_gap_bootstrap": reverse_endpoint_bootstrap,
            "forward_reverse_absolute_gap_bootstrap": direction_bootstrap,
            "passed": bidirectional_passed,
        },
        "bidirectional_lexical_localization": {
            "reverse_lexical_absolute_gap_bootstrap": reverse_lexical_bootstrap,
            "reverse_shared_static_mean_absolute": float(r_static.abs().mean()),
            "passed": bidirectional_lexical_passed,
        },
        "graded_dose_response": {
            "alpha": [0.0, 0.25, 0.50, 0.75, 1.0],
            "mean": [float(value) for value in dose_means],
            "aggregate_strictly_monotone": aggregate_monotone,
            "strictly_monotone_context_count": monotone_contexts,
            "midpoint_mean": midpoint_mean,
            "passed": dose_passed,
        },
    }
    full_interpretation = all(
        claims[name]["passed"]
        for name in (
            "replication",
            "forward_lexical_localization",
            "component_specificity_control",
            "bidirectional_switch",
            "bidirectional_lexical_localization",
            "graded_dose_response",
        )
    )
    return {
        "schema_version": "e29_14a_position_semantics_score_v1",
        "normalized_response_summaries": {
            "forward": {
                condition["name"]: _summary(forward[:, index])
                for index, condition in enumerate(FORWARD_CONDITIONS)
            },
            "reverse": {
                condition["name"]: _summary(reverse[:, index])
                for index, condition in enumerate(REVERSE_CONDITIONS)
            },
        },
        "claims": claims,
        "integrity_passed": integrity_passed,
        "recommended_interpretation": (
            "component_specific_bidirectional_lexical_state_switch_with_graded_dose_response"
            if full_interpretation
            else "report_claims_independently_and_follow_registered_decision_branches"
        ),
    }


__all__ = [
    "BOOTSTRAP",
    "CONTEXT_COUNT",
    "CONTROL_COMPONENT",
    "FORWARD_CONDITIONS",
    "HUB_COMPONENT",
    "POSITION_SCOPE_NAMES",
    "QUERY_ACCOUNTING",
    "REVERSE_CONDITIONS",
    "SCHEMA_VERSION",
    "SCORING_GATES",
    "TEMPLATE_STRATA",
    "build_position_protocol",
    "build_position_scopes",
    "score_position_semantics",
    "stratified_paired_bootstrap",
    "validate_position_protocol",
]
