"""Tests for the pure response-blind E29.14c scale-scope protocol."""

from __future__ import annotations

import copy

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    CONTEXT_COUNT,
    EXPECTED_MASK_CANONICAL_SHA256,
    HUB_COMPONENT,
    MASK_CARDINALITIES,
    MASK_SHAPES,
    MASK_SPLITS,
    PHYSICAL_QUERY_ACCOUNTING,
    QUERY_ACCOUNTING,
    TRANSITION_DESIGN_CONDITION_NUMBERS,
    build_scale_masks,
    build_scale_scope_protocol,
    score_scale_scope_ladder,
    validate_scale_masks,
    validate_scale_scope_protocol,
)


def _canonical_rows(panel: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in torch.where(row == 1.0)[0])
        for row in panel
    )


def _passing_responses() -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    endpoints = torch.linspace(18.0, 30.0, CONTEXT_COUNT, dtype=torch.float64)
    levels = {
        "scale36": 0.76,
        "scale40": 0.86,
        "scale44": 0.94,
        "scale47": 0.99,
    }
    all_position = {}
    lexical = {}
    context_offsets = torch.linspace(-0.004, 0.004, CONTEXT_COUNT)[None]
    for split in MASK_SPLITS:
        row_count = MASK_SHAPES[split][0]
        row_offsets = (
            torch.zeros(1, 1, dtype=torch.float64)
            if row_count == 1
            else torch.linspace(-0.006, 0.006, row_count)[:, None]
        )
        normalized = levels[split] + row_offsets + context_offsets
        all_position[split] = normalized * endpoints[None]
        lexical[split] = (normalized + 0.002) * endpoints[None]
    return (
        all_position,
        lexical,
        0.995 * endpoints,
        0.996 * endpoints,
        endpoints,
    )


def _score(
    all_position: dict[str, torch.Tensor],
    lexical: dict[str, torch.Tensor],
    singleton_all: torch.Tensor,
    singleton_lexical: torch.Tensor,
    endpoints: torch.Tensor,
    *,
    integrity_passed: bool = True,
) -> dict[str, object]:
    return score_scale_scope_ladder(
        all_position,
        lexical,
        singleton_all,
        singleton_lexical,
        endpoints,
        build_scale_masks(),
        integrity_passed=integrity_passed,
    )


def test_fixed_masks_are_exact_balanced_nested_hub_off_chains() -> None:
    masks = validate_scale_masks(build_scale_masks())

    assert tuple(masks) == MASK_SPLITS
    non_hub = [index for index in range(48) if index != HUB_COMPONENT]
    for split in MASK_SPLITS:
        panel = masks[split]
        assert tuple(panel.shape) == MASK_SHAPES[split]
        assert torch.all(panel[:, HUB_COMPONENT] == 0.0)
        assert torch.all(panel.sum(dim=1) == MASK_CARDINALITIES[split])
        assert len(set(_canonical_rows(panel))) == len(panel)
        if split != "scale47":
            assert torch.all(
                panel[:, non_hub].sum(dim=0) == MASK_CARDINALITIES[split]
            )

    assert torch.all(masks["scale36"] <= masks["scale40"])
    assert torch.all(masks["scale40"] <= masks["scale44"])
    assert torch.all(masks["scale44"] <= masks["scale47"])
    assert torch.all(
        (masks["scale40"] - masks["scale36"]).sum(dim=1) == 4
    )
    assert torch.all(
        (masks["scale44"] - masks["scale40"]).sum(dim=1) == 4
    )
    assert torch.all(
        (masks["scale47"] - masks["scale44"]).sum(dim=1) == 3
    )
    assert set(EXPECTED_MASK_CANONICAL_SHA256) == set(MASK_SPLITS)


def test_transition_incidence_is_full_rank_and_well_conditioned() -> None:
    masks = build_scale_masks()
    transitions = (
        ("scale36", "scale40"),
        ("scale40", "scale44"),
        ("scale44", "scale47"),
    )

    for current, following in transitions:
        following_panel = masks[following]
        if following == "scale47":
            following_panel = following_panel.repeat(47, 1)
        incidence = following_panel - masks[current]
        incidence = incidence[:, [index for index in range(48) if index != 1]]
        name = f"{current}_to_{following}"
        assert int(torch.linalg.matrix_rank(incidence)) == 47
        assert float(torch.linalg.cond(incidence)) == pytest.approx(
            TRANSITION_DESIGN_CONDITION_NUMBERS[name], rel=1e-12
        )


def test_protocol_round_trip_and_exact_query_accounting() -> None:
    protocol = build_scale_scope_protocol()

    assert validate_scale_scope_protocol(protocol) == protocol
    assert QUERY_ACCOUNTING == {
        "contexts": 12,
        "mask_rows_by_split": {
            "scale36": 47,
            "scale40": 47,
            "scale44": 47,
            "scale47": 1,
        },
        "multi_component_masks": 142,
        "position_scopes": 2,
        "singleton_condition_rows_per_context": 2,
        "multi_component_condition_rows_per_context": 284,
        "patched_condition_rows_per_context": 286,
        "patched_context_condition_cells": 3432,
        "clean_forward_examples": 24,
        "maximum_logical_queries": 3456,
    }
    assert PHYSICAL_QUERY_ACCOUNTING["maximum_physical_model_examples"] == 4028

    changed = copy.deepcopy(protocol)
    changed["scoring_gates"]["scope_equivalence"][
        "paired_absolute_gap_bootstrap_upper_max"
    ] = 0.031
    with pytest.raises(ValueError, match="scoring gates"):
        validate_scale_scope_protocol(changed)

    changed = copy.deepcopy(protocol)
    rows = changed["mask_rows"]["scale36"]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises((ValueError, TypeError), match="mask|protocol|hash"):
        validate_scale_scope_protocol(changed)


def test_passing_panel_authorizes_scope_scaling_and_separate_endpoint_claims() -> None:
    result = _score(*_passing_responses())
    claims = result["claims"]

    assert claims["singleton_transfer"]["passed"]
    assert all(
        claim["passed"]
        for claim in claims["scope_equivalence_by_scale"].values()
    )
    assert claims["joint_equal_scale_scope_equivalence"]["passed"]
    assert claims["high_scale_lexical_scope_authorized"]["passed"]
    assert claims["informative_c36_nonsaturation"]["passed"]
    assert claims["c47_backup_endpoint"]["passed"]
    assert claims["complete_hub_off_scale_ladder"]["passed"]
    assert set(result["transition_scope_surrogate_failure_localization"]) == {
        "scale36_to_scale40",
        "scale40_to_scale44",
        "scale44_to_scale47",
    }
    increments = result["transition_response_increment_diagnostics"]
    expected_increments = {
        "scale36_to_scale40": 0.10,
        "scale40_to_scale44": 0.08,
        "scale44_to_scale47": 0.05,
    }
    for name, expected in expected_increments.items():
        assert increments[name]["all_positions"]["normalized_mean"] == pytest.approx(
            expected
        )
        assert increments[name]["differing_lexical_positions"][
            "normalized_mean"
        ] == pytest.approx(expected)
    geometry = result["scale_response_geometry"]
    assert geometry["scale36"]["all_positions"][
        "normalized_mean"
    ] == pytest.approx(0.76)
    assert geometry["scale47"]["all_positions"][
        "endpoint_absolute_distance_mean"
    ] == pytest.approx(0.01)


def test_one_bad_rung_and_cell_coverage_fail_without_relabeling_other_claims() -> None:
    responses = _passing_responses()
    _all_position, lexical = responses[:2]
    lexical["scale40"] += 0.20 * responses[-1][None]
    result = _score(*responses)

    assert not result["claims"]["scope_equivalence_by_scale"]["scale40"][
        "passed"
    ]
    assert result["claims"]["scope_equivalence_by_scale"]["scale36"][
        "passed"
    ]
    assert not result["claims"]["high_scale_lexical_scope_authorized"][
        "passed"
    ]

    responses = _passing_responses()
    _all_position, lexical = responses[:2]
    flat = lexical["scale36"].flatten()
    flat[:29] += 0.11 * responses[-1].repeat(47)[:29]
    result = _score(*responses)
    scale36 = result["claims"]["scope_equivalence_by_scale"]["scale36"]
    assert scale36["passing_cells"] == 535
    assert not scale36["passed"]


def test_c47_uses_context_only_gate_and_endpoint_claims_are_independent() -> None:
    responses = _passing_responses()
    all_position, lexical = responses[:2]
    lexical["scale47"][0, :2] += 0.06 * responses[-1][:2]
    result = _score(*responses)
    c47 = result["claims"]["scope_equivalence_by_scale"]["scale47"]

    assert c47["mask_count"] == 1
    assert c47["passing_contexts"] == 10
    assert not c47["passed"]

    responses = _passing_responses()
    all_position, lexical = responses[:2]
    all_position["scale36"] = 0.99 * responses[-1][None].repeat(47, 1)
    lexical["scale36"] = all_position["scale36"] + 0.002 * responses[-1][None]
    result = _score(*responses)
    assert result["claims"]["scope_equivalence_by_scale"]["scale36"][
        "passed"
    ]
    assert not result["claims"]["informative_c36_nonsaturation"]["passed"]

    responses = _passing_responses()
    all_position, lexical = responses[:2]
    all_position["scale47"] = 0.90 * responses[-1][None]
    lexical["scale47"] = 0.902 * responses[-1][None]
    result = _score(*responses)
    assert result["claims"]["scope_equivalence_by_scale"]["scale47"][
        "passed"
    ]
    assert not result["claims"]["c47_backup_endpoint"]["passed"]


def test_integrity_is_a_hard_gate_and_bootstrap_is_deterministic() -> None:
    responses = _passing_responses()
    first = _score(*responses)
    second = _score(*responses)
    assert first == second

    failed = _score(*responses, integrity_passed=False)
    assert not failed["claims"]["singleton_transfer"]["passed"]
    assert not failed["claims"]["high_scale_lexical_scope_authorized"][
        "passed"
    ]
