from __future__ import annotations

import copy

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    BOOTSTRAP,
    CONTEXT_COUNT,
    FORWARD_CONDITIONS,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
    build_position_protocol,
    build_position_scopes,
    score_position_semantics,
    stratified_paired_bootstrap,
    validate_position_protocol,
)


def _condition_index(conditions: tuple[dict[str, object], ...], name: str) -> int:
    return next(index for index, row in enumerate(conditions) if row["name"] == name)


def _passing_responses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward = torch.zeros(CONTEXT_COUNT, len(FORWARD_CONDITIONS), dtype=torch.float64)
    reverse = torch.zeros(CONTEXT_COUNT, len(REVERSE_CONDITIONS), dtype=torch.float64)
    values = {
        "hub_all_positions_alpha1": 0.995,
        "hub_differing_lexical_alpha1": 0.996,
        "hub_shared_static_alpha1": 0.002,
        "hub_demonstration_target_alpha1": 0.82,
        "hub_demonstration_prefix_alpha1": 0.36,
        "hub_repeated_query_alpha1": 0.54,
        "hub_differing_except_target_alpha1": 0.14,
        "hub_all_positions_alpha025": 0.05,
        "hub_all_positions_alpha050": 0.45,
        "hub_all_positions_alpha075": 0.94,
        "control_all_positions_alpha1": 0.003,
        "control_differing_lexical_alpha1": 0.004,
    }
    for name, value in values.items():
        forward[:, _condition_index(FORWARD_CONDITIONS, name)] = value
    reverse_values = {
        "hub_all_positions_alpha1": 0.997,
        "hub_differing_lexical_alpha1": 0.998,
        "hub_shared_static_alpha1": 0.001,
        "hub_demonstration_target_alpha1": 0.84,
    }
    for name, value in reverse_values.items():
        reverse[:, _condition_index(REVERSE_CONDITIONS, name)] = value
    endpoints = torch.linspace(10.0, 21.0, CONTEXT_COUNT, dtype=torch.float64)
    return forward * endpoints[:, None], reverse * endpoints[:, None], endpoints


def test_fixed_manifest_and_query_arithmetic() -> None:
    assert len(FORWARD_CONDITIONS) == 12
    assert len(REVERSE_CONDITIONS) == 4
    assert len({row["name"] for row in FORWARD_CONDITIONS}) == 12
    assert len({row["name"] for row in REVERSE_CONDITIONS}) == 4
    assert QUERY_ACCOUNTING == {
        "contexts": 12,
        "forward_condition_rows_per_context": 12,
        "reverse_condition_rows_per_context": 4,
        "patched_context_condition_cells": 192,
        "clean_forward_examples": 24,
        "maximum_logical_queries": 216,
    }


def test_position_scope_partition_is_exact() -> None:
    base = torch.tensor([10, 11, 20, 21, 22, 23, 12, 20, 21, 22])
    donor = torch.tensor([10, 11, 30, 31, 32, 33, 12, 30, 31, 32])

    scopes = build_position_scopes(base, donor, base_target_token_id=23)

    assert torch.equal(
        scopes["demonstration_prefix_positions"]
        | scopes["demonstration_target_position"]
        | scopes["repeated_query_positions"],
        scopes["differing_lexical_positions"],
    )
    assert not bool(
        (
            scopes["differing_lexical_positions"] & scopes["shared_static_positions"]
        ).any()
    )
    assert torch.equal(
        scopes["differing_except_demonstration_target"]
        | scopes["demonstration_target_position"],
        scopes["differing_lexical_positions"],
    )


def test_protocol_round_trip_rejects_manifest_mutation() -> None:
    protocol = build_position_protocol()
    validated = validate_position_protocol(protocol)
    assert validated["query_accounting"] == QUERY_ACCOUNTING

    mutated = copy.deepcopy(protocol)
    mutated["forward_conditions"][0]["alpha"] = 0.9
    try:
        validate_position_protocol(mutated)
    except ValueError as error:
        assert "forward condition" in str(error)
    else:
        raise AssertionError("mutated condition was accepted")


def test_stratified_bootstrap_is_deterministic_and_template_balanced() -> None:
    values = torch.arange(CONTEXT_COUNT, dtype=torch.float64)

    first = stratified_paired_bootstrap(values)
    second = stratified_paired_bootstrap(values)

    assert first == second
    assert first["seed"] == BOOTSTRAP["seed"]
    assert first["replicates"] == BOOTSTRAP["replicates"]
    assert first["point_estimate"] == 5.5
    assert first["resampling"] == "contexts_within_four_fixed_template_strata"
    assert first["upper_quantile_one_sided"] == 0.95
    assert first["lower_quantile_descriptive"] == 0.025


def test_scoring_passes_claims_independently() -> None:
    forward, reverse, endpoints = _passing_responses()

    result = score_position_semantics(
        forward, reverse, endpoints, integrity_passed=True
    )

    assert result["claims"]["replication"]["passed"]
    assert result["claims"]["forward_lexical_localization"]["passed"]
    assert result["claims"]["component_specificity_control"]["passed"]
    assert result["claims"]["bidirectional_switch"]["passed"]
    assert result["claims"]["bidirectional_lexical_localization"]["passed"]
    assert result["claims"]["graded_dose_response"]["passed"]
    assert result["recommended_interpretation"] == (
        "component_specific_bidirectional_lexical_state_switch_with_graded_dose_response"
    )


def test_dose_failure_does_not_erase_localization() -> None:
    forward, reverse, endpoints = _passing_responses()
    midpoint = _condition_index(FORWARD_CONDITIONS, "hub_all_positions_alpha050")
    forward[:, midpoint] = 0.99 * endpoints

    result = score_position_semantics(
        forward, reverse, endpoints, integrity_passed=True
    )

    assert result["claims"]["forward_lexical_localization"]["passed"]
    assert not result["claims"]["graded_dose_response"]["passed"]


def test_negative_quarter_dose_is_not_graded_from_clean() -> None:
    forward, reverse, endpoints = _passing_responses()
    quarter = _condition_index(FORWARD_CONDITIONS, "hub_all_positions_alpha025")
    forward[:, quarter] = -0.01 * endpoints

    result = score_position_semantics(
        forward, reverse, endpoints, integrity_passed=True
    )

    assert not result["claims"]["graded_dose_response"]["passed"]
    assert result["claims"]["graded_dose_response"]["alpha"][0] == 0.0


def test_integrity_failure_forces_every_claim_to_fail() -> None:
    forward, reverse, endpoints = _passing_responses()

    result = score_position_semantics(
        forward, reverse, endpoints, integrity_passed=False
    )

    assert not result["integrity_passed"]
    assert not any(claim["passed"] for claim in result["claims"].values())


def test_endpoint_overshoot_cannot_pass_replication_or_bidirectionality() -> None:
    forward, reverse, endpoints = _passing_responses()
    forward_all = _condition_index(FORWARD_CONDITIONS, "hub_all_positions_alpha1")
    reverse_all = _condition_index(REVERSE_CONDITIONS, "hub_all_positions_alpha1")
    forward[:, forward_all] = 2.0 * endpoints
    reverse[:, reverse_all] = 2.0 * endpoints

    result = score_position_semantics(
        forward, reverse, endpoints, integrity_passed=True
    )

    assert not result["claims"]["replication"]["passed"]
    assert not result["claims"]["bidirectional_switch"]["passed"]
    assert (
        result["claims"]["replication"]["forward_endpoint_absolute_gap_bootstrap"][
            "point_estimate"
        ]
        == 1.0
    )
