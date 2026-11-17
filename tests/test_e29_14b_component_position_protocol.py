"""Focused tests for the pure E29.14b component-position bridge protocol."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    BOOTSTRAP,
    CONTEXT_COUNT,
    MASK_HUB_ON_COUNTS,
    MASK_SHAPES,
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    SINGLETON_CONDITIONS,
    build_component_position_protocol,
    score_component_position_bridge,
    validate_bridge_masks,
    validate_component_position_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEALED_MASKS = (
    REPOSITORY_ROOT
    / "runs/e29_13_absorbing_hub_confirmation_v1/protocol_masks.pt"
)


def _held_masks() -> dict[str, torch.Tensor]:
    artifact = torch.load(SEALED_MASKS, map_location="cpu", weights_only=True)
    return {
        split: artifact["protocol"]["held_masks"][split].clone()
        for split in MASK_SPLITS
    }


def _passing_responses() -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    masks = _held_masks()
    endpoints = torch.linspace(18.0, 30.0, CONTEXT_COUNT, dtype=torch.float64)
    all_position: dict[str, torch.Tensor] = {}
    lexical: dict[str, torch.Tensor] = {}
    for split in MASK_SPLITS:
        panel = masks[split]
        normalized = torch.empty(MASK_SHAPES[split][0], CONTEXT_COUNT)
        hub_on = panel[:, 1].bool()
        normalized[hub_on] = 0.995
        normalized[~hub_on] = 0.74 if split.endswith("24") else 0.86
        # Preserve nontrivial mask/context variation without approaching a gate.
        normalized += torch.linspace(-0.01, 0.01, len(panel))[:, None]
        normalized += torch.linspace(-0.005, 0.005, CONTEXT_COUNT)[None, :]
        all_position[split] = normalized.double() * endpoints[None, :]
        lexical[split] = (normalized.double() + 0.002) * endpoints[None, :]
    singleton_all = 0.995 * endpoints
    singleton_lexical = 0.996 * endpoints
    return all_position, lexical, singleton_all, singleton_lexical, endpoints


def _score(
    *,
    integrity_passed: bool = True,
) -> dict[str, object]:
    all_position, lexical, singleton_all, singleton_lexical, endpoints = (
        _passing_responses()
    )
    return score_component_position_bridge(
        all_position,
        lexical,
        singleton_all,
        singleton_lexical,
        endpoints,
        _held_masks(),
        integrity_passed=integrity_passed,
    )


def test_exact_masks_and_query_arithmetic() -> None:
    masks = validate_bridge_masks(_held_masks())

    assert tuple(masks) == MASK_SPLITS
    assert {split: tuple(panel.shape) for split, panel in masks.items()} == MASK_SHAPES
    assert {
        split: int(panel[:, 1].sum()) for split, panel in masks.items()
    } == MASK_HUB_ON_COUNTS
    assert POSITION_SCOPE_ORDER == (
        "all_positions",
        "differing_lexical_positions",
    )
    assert tuple(row["scope"] for row in SINGLETON_CONDITIONS) == (
        "all_positions",
        "differing_lexical_positions",
    )
    assert QUERY_ACCOUNTING == {
        "contexts": 12,
        "mask_rows_by_split": {
            "confirmation_size24": 48,
            "confirmation_size32": 48,
        },
        "multi_component_masks": 96,
        "position_scopes": 2,
        "singleton_condition_rows_per_context": 2,
        "multi_component_condition_rows_per_context": 192,
        "patched_condition_rows_per_context": 194,
        "patched_context_condition_cells": 2328,
        "clean_forward_examples": 24,
        "maximum_logical_queries": 2352,
    }


def test_exact_mask_hash_rejects_structurally_valid_row_reordering() -> None:
    masks = _held_masks()
    masks["confirmation_size24"][[0, 1]] = masks["confirmation_size24"][[1, 0]]

    with pytest.raises(ValueError, match="canonical hash"):
        validate_bridge_masks(masks)


def test_protocol_round_trip_rejects_gate_mutation() -> None:
    protocol = build_component_position_protocol(_held_masks())
    validated = validate_component_position_protocol(protocol)

    assert validated["query_accounting"] == QUERY_ACCOUNTING
    mutated = copy.deepcopy(protocol)
    mutated["scoring_gates"]["scope_equivalence"][
        "paired_absolute_gap_bootstrap_upper_max"
    ] = 0.04
    with pytest.raises(ValueError, match="scoring gates"):
        validate_component_position_protocol(mutated)


def test_all_bridge_claims_pass_on_a_clear_equivalence_fixture() -> None:
    result = _score()

    assert result["integrity_passed"]
    assert result["claims"]["singleton_transfer"]["passed"]
    assert all(
        claim["passed"]
        for claim in result["claims"]["all_position_anchor_by_scale"].values()
    )
    assert all(
        claim["passed"]
        for claim in result["claims"]["hub_off_scope_bridge_by_scale"].values()
    )
    assert all(
        claim["passed"]
        for claim in result["claims"]["hub_on_scope_control_by_scale"].values()
    )
    assert result["claims"]["joint_hub_off_scope_bridge"]["passed"]
    assert result["claims"]["joint_hub_on_scope_control"]["passed"]
    assert result["claims"]["hub_off_scale_promotion_authorized"]["passed"]
    assert result["claims"]["full_component_position_bridge"]["passed"]


def test_saturated_hub_off_panel_cannot_count_as_an_informative_bridge() -> None:
    all_position, lexical, singleton_all, singleton_lexical, endpoints = (
        _passing_responses()
    )
    masks = _held_masks()
    for split in MASK_SPLITS:
        hub_off = masks[split][:, 1] == 0
        all_position[split][hub_off] = endpoints
        lexical[split][hub_off] = endpoints

    result = score_component_position_bridge(
        all_position,
        lexical,
        singleton_all,
        singleton_lexical,
        endpoints,
        masks,
        integrity_passed=True,
    )

    assert not all(
        claim["passed"]
        for claim in result["claims"]["all_position_anchor_by_scale"].values()
    )
    assert not result["claims"]["hub_off_scale_promotion_authorized"]["passed"]
    assert not result["claims"]["full_component_position_bridge"]["passed"]


def test_composition_specific_scope_gap_does_not_erase_singleton_transfer() -> None:
    all_position, lexical, singleton_all, singleton_lexical, endpoints = (
        _passing_responses()
    )
    masks = _held_masks()
    for split in MASK_SPLITS:
        hub_off = masks[split][:, 1] == 0
        lexical[split][hub_off] += 0.08 * endpoints

    result = score_component_position_bridge(
        all_position,
        lexical,
        singleton_all,
        singleton_lexical,
        endpoints,
        masks,
        integrity_passed=True,
    )

    assert result["claims"]["singleton_transfer"]["passed"]
    assert not result["claims"]["joint_hub_off_scope_bridge"]["passed"]
    assert not result["claims"]["hub_off_scale_promotion_authorized"]["passed"]


def test_hub_off_promotion_is_reported_when_only_hub_on_control_fails() -> None:
    all_position, lexical, singleton_all, singleton_lexical, endpoints = (
        _passing_responses()
    )
    masks = _held_masks()
    for split in MASK_SPLITS:
        hub_on = masks[split][:, 1] == 1
        lexical[split][hub_on] += 0.08 * endpoints

    result = score_component_position_bridge(
        all_position,
        lexical,
        singleton_all,
        singleton_lexical,
        endpoints,
        masks,
        integrity_passed=True,
    )

    assert result["claims"]["hub_off_scale_promotion_authorized"]["passed"]
    assert not result["claims"]["full_component_position_bridge"]["passed"]
    assert result["recommended_interpretation"] == (
        "fresh_context_hub_off_c24_c32_component_position_bridge_confirmed_"
        "without_general_all_mask_equivalence"
    )


def test_endpoint_overshoot_and_integrity_failure_cannot_pass() -> None:
    all_position, lexical, singleton_all, singleton_lexical, endpoints = (
        _passing_responses()
    )
    singleton_all = 2.0 * endpoints
    singleton_lexical = 2.0 * endpoints
    overshoot = score_component_position_bridge(
        all_position,
        lexical,
        singleton_all,
        singleton_lexical,
        endpoints,
        _held_masks(),
        integrity_passed=True,
    )
    integrity = _score(integrity_passed=False)

    assert not overshoot["claims"]["singleton_transfer"]["passed"]
    assert not overshoot["claims"]["full_component_position_bridge"]["passed"]
    assert not integrity["integrity_passed"]
    assert not integrity["claims"]["singleton_transfer"]["passed"]
    assert not integrity["claims"]["full_component_position_bridge"]["passed"]
    assert BOOTSTRAP["seed"] == 29142
