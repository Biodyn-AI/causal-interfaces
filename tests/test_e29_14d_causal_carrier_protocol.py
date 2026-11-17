"""Focused tests for the pure E29.14d causal-carrier protocol."""

from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    ADD_ONE_FAMILY_IDS,
    ATTENTION_COMPONENT_ORDER,
    BOOTSTRAP,
    CANDIDATE_GROUPS,
    COMPONENT_COUNT,
    CONDITION_CARDINALITY_COUNTS,
    CONTEXT_COUNT,
    DESIGN_DOMAIN,
    EXPECTED_ADD_ONE_RECORDS_SHA256,
    EXPECTED_CONDITION_MANIFEST_SHA256,
    EXPECTED_CONDITION_MASKS_SHA256,
    EXPECTED_FACTORIAL_RECORDS_SHA256,
    HUB_COMPONENT,
    MLP_COMPONENT_ORDER,
    NON_HUB_COMPONENTS,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    PRIMARY_POSITION_SCOPE,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    TARGET_COMPONENT_ORDER,
    build_add_one_records,
    build_balanced_carrier_panels,
    build_causal_carrier_protocol,
    build_condition_manifest,
    build_condition_masks,
    build_dropout_records,
    build_factorial_panels,
    build_factorial_records,
    build_full_dropout_panels,
    build_query_manifest,
    query_scope_indices,
    validate_causal_carrier_protocol,
    validate_condition_masks,
    validate_no_prior_c32_c33_overlap,
)


def _rows(panel: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in torch.where(row == 1.0)[0]) for row in panel
    )


def test_component_classes_groups_and_scope_order_are_exact() -> None:
    expected_attention_order = tuple(
        sorted(
            range(0, 48, 2),
            key=lambda component: hashlib.sha256(
                f"{DESIGN_DOMAIN}|attention|{component}".encode("ascii")
            ).digest(),
        )
    )
    expected_mlp_order = tuple(
        sorted(
            range(3, 48, 2),
            key=lambda component: hashlib.sha256(
                f"{DESIGN_DOMAIN}|mlp|{component}".encode("ascii")
            ).digest(),
        )
    )

    assert len(ATTENTION_COMPONENT_ORDER) == 24
    assert len(MLP_COMPONENT_ORDER) == 23
    assert set(ATTENTION_COMPONENT_ORDER) == set(range(0, 48, 2))
    assert set(MLP_COMPONENT_ORDER) == set(range(3, 48, 2))
    assert ATTENTION_COMPONENT_ORDER == expected_attention_order
    assert MLP_COMPONENT_ORDER == expected_mlp_order
    assert TARGET_COMPONENT_ORDER == (
        *ATTENTION_COMPONENT_ORDER,
        *MLP_COMPONENT_ORDER,
    )
    assert NON_HUB_COMPONENTS == tuple(
        component for component in range(COMPONENT_COUNT) if component != HUB_COMPONENT
    )
    assert set(TARGET_COMPONENT_ORDER) == set(NON_HUB_COMPONENTS)
    assert CANDIDATE_GROUPS == {
        "late_attention_core": (36, 38, 40, 42),
        "matched_same_layer_mlp": (37, 39, 41, 43),
        "flanking_attention_controls": (28, 30, 32, 34, 44, 46),
    }
    assert POSITION_SCOPE_ORDER == (
        "all_positions",
        "differing_lexical_positions",
    )
    assert PRIMARY_POSITION_SCOPE == "all_positions"


def test_three_add_one_families_are_class_balanced_and_pair_balanced() -> None:
    panels = build_balanced_carrier_panels()
    records = build_add_one_records()

    assert tuple(panels) == ADD_ONE_FAMILY_IDS
    assert len(records) == 3 * 47 == 141
    directed_inclusion_counts: Counter[tuple[int, int]] = Counter()
    all_bases: list[tuple[int, ...]] = []
    all_adds: list[tuple[int, ...]] = []
    for family_index, family_id in enumerate(ADD_ONE_FAMILY_IDS):
        base = panels[family_id]["c32_base"]
        add = panels[family_id]["c33_add_one"]
        assert base.dtype == add.dtype == torch.float64
        assert tuple(base.shape) == tuple(add.shape) == (47, 48)
        assert bool((base.sum(dim=1) == 32).all())
        assert bool((add.sum(dim=1) == 33).all())
        assert bool((base[:, HUB_COMPONENT] == 0).all())
        assert bool((add[:, HUB_COMPONENT] == 0).all())
        assert bool((base[:, list(ATTENTION_COMPONENT_ORDER)].sum(dim=1) == 16).all())
        assert bool((base[:, list(MLP_COMPONENT_ORDER)].sum(dim=1) == 16).all())
        for target_index, target in enumerate(TARGET_COMPONENT_ORDER):
            assert base[target_index, target] == 0.0
            difference = add[target_index] - base[target_index]
            assert torch.equal(
                difference,
                torch.nn.functional.one_hot(
                    torch.tensor(target), num_classes=COMPONENT_COUNT
                ).to(torch.float64),
            )
            directed_inclusion_counts.update(
                (target, component)
                for component in torch.where(base[target_index] == 1.0)[0].tolist()
            )
        family_records = [
            record for record in records if record["family_index"] == family_index
        ]
        assert [record["target_component"] for record in family_records] == list(
            TARGET_COMPONENT_ORDER
        )
        all_bases.extend(_rows(base))
        all_adds.extend(_rows(add))

    assert len(set(all_bases)) == 141
    assert len(set(all_adds)) == 78
    assert len(set(all_bases) | set(all_adds)) == 219
    assert set(directed_inclusion_counts.values()) == {2, 3}
    assert Counter(directed_inclusion_counts.values()) == Counter({2: 1974, 3: 188})
    assert len(directed_inclusion_counts) == 47 * 46
    assert len(EXPECTED_ADD_ONE_RECORDS_SHA256) == 64


def test_factorials_are_exact_recompositions_and_deduplicate_as_declared() -> None:
    panels = build_factorial_panels()
    records = build_factorial_records()
    scan_rows = {
        row
        for family in build_balanced_carrier_panels().values()
        for kind in ("c32_base", "c33_add_one")
        for row in _rows(family[kind])
    }

    assert tuple(panels) == ("neither", "attention_only", "mlp_only", "both")
    assert all(tuple(panel.shape) == (12, 48) for panel in panels.values())
    assert len(records) == 12
    factorial_rows = []
    for row_index, record in enumerate(records):
        attention = record["attention_component"]
        mlp = record["mlp_component"]
        assert mlp == attention + 1
        neither = panels["neither"][row_index]
        attention_only = panels["attention_only"][row_index]
        mlp_only = panels["mlp_only"][row_index]
        both = panels["both"][row_index]
        assert int(neither.sum()) == 32
        assert int(attention_only.sum()) == int(mlp_only.sum()) == 33
        assert int(both.sum()) == 34
        assert neither[attention] == neither[mlp] == 0.0
        assert torch.equal(
            attention_only - neither,
            torch.nn.functional.one_hot(
                torch.tensor(attention), num_classes=COMPONENT_COUNT
            ).to(torch.float64),
        )
        assert torch.equal(
            mlp_only - neither,
            torch.nn.functional.one_hot(
                torch.tensor(mlp), num_classes=COMPONENT_COUNT
            ).to(torch.float64),
        )
        assert torch.equal(both, attention_only + mlp_only - neither)
        factorial_rows.extend(
            _rows(torch.stack((neither, attention_only, mlp_only, both)))
        )

    assert len(set(factorial_rows)) == 47
    assert len(set(factorial_rows) & scan_rows) == 1
    assert len(set(factorial_rows) - scan_rows) == 46
    assert len(EXPECTED_FACTORIAL_RECORDS_SHA256) == 64


def test_full_dropout_and_deduplicated_condition_manifests_are_exact() -> None:
    high = build_full_dropout_panels()
    masks = build_condition_masks()
    manifest = build_condition_manifest()
    dropout_records = build_dropout_records()

    assert tuple(high["c47_full"].shape) == (1, 48)
    assert tuple(high["c46_leave_one_out"].shape) == (47, 48)
    assert int(high["c47_full"].sum()) == 47
    assert bool((high["c46_leave_one_out"].sum(dim=1) == 46).all())
    assert bool((high["c47_full"][:, HUB_COMPONENT] == 0).all())
    assert len(dropout_records) == 47
    assert tuple(masks.shape) == (314, 48)
    assert masks.dtype == torch.float64 and masks.device.type == "cpu"
    assert len(manifest) == 314
    assert [record["condition_index"] for record in manifest] == list(range(314))
    assert len({record["condition_id"] for record in manifest}) == 314
    assert len(set(_rows(masks))) == 314
    assert Counter(int(row.sum()) for row in masks) == Counter(
        CONDITION_CARDINALITY_COUNTS
    )
    assert tuple(torch.where(masks[-1] == 1.0)[0].tolist()) == (HUB_COMPONENT,)
    assert manifest[-1]["condition_id"] == "hub_component_01_singleton"
    assert validate_condition_masks(masks).equal(masks)
    assert len(EXPECTED_CONDITION_MASKS_SHA256) == 64
    assert len(EXPECTED_CONDITION_MANIFEST_SHA256) == 64


def test_pair_and_factorial_records_reconstruct_exact_deduplicated_masks() -> None:
    masks = build_condition_masks()
    for record in build_add_one_records():
        base = masks[record["base_condition_index"]]
        plus = masks[record["plus_condition_index"]]
        target = record["target_component"]
        assert base[target] == 0.0 and plus[target] == 1.0
        assert torch.equal(
            plus - base,
            torch.nn.functional.one_hot(
                torch.tensor(target), num_classes=COMPONENT_COUNT
            ).to(torch.float64),
        )
    for record in build_factorial_records():
        cells = record["condition_indices"]
        neither = masks[cells["neither"]]
        attention_only = masks[cells["attention_only"]]
        mlp_only = masks[cells["mlp_only"]]
        both = masks[cells["both"]]
        assert torch.equal(both, attention_only + mlp_only - neither)
    for record in build_dropout_records():
        full = masks[record["full_condition_index"]]
        dropout = masks[record["dropout_condition_index"]]
        component = record["dropped_component"]
        assert full[component] == 1.0 and dropout[component] == 0.0
        assert torch.equal(
            full - dropout,
            torch.nn.functional.one_hot(
                torch.tensor(component), num_classes=COMPONENT_COUNT
            ).to(torch.float64),
        )


def test_query_order_and_accounting_cover_both_scopes_without_duplicate_masks() -> None:
    queries = build_query_manifest()
    by_scope = query_scope_indices()

    assert len(queries) == 628
    assert by_scope == {
        "all_positions": tuple(range(314)),
        "differing_lexical_positions": tuple(range(314)),
    }
    assert [query["query_index"] for query in queries] == list(range(628))
    assert queries[0]["position_scope"] == "all_positions"
    assert queries[0]["condition_index"] == 0
    assert queries[313]["condition_index"] == 313
    assert queries[314]["position_scope"] == "differing_lexical_positions"
    assert queries[314]["condition_index"] == 0
    assert queries[-1]["condition_index"] == 313
    assert len({query["query_id"] for query in queries}) == 628
    assert QUERY_ACCOUNTING == {
        "contexts": 12,
        "unique_condition_masks": 314,
        "position_scopes": 2,
        "patched_condition_rows_per_context": 628,
        "patched_context_condition_cells": 7_536,
        "clean_forward_examples": 24,
        "maximum_logical_queries": 7_560,
    }
    assert PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"] == 1_256
    assert PHYSICAL_QUERY_ACCOUNTING["maximum_physical_model_examples"] == 8_816


def test_protocol_round_trip_gates_bootstrap_and_prior_overlap_validation() -> None:
    protocol = build_causal_carrier_protocol()
    assert validate_causal_carrier_protocol(protocol) == protocol
    assert protocol["add_one_records"] == list(build_add_one_records())
    assert protocol["factorial_records"] == list(build_factorial_records())
    assert protocol["dropout_records"] == list(build_dropout_records())
    assert protocol["bootstrap"] == BOOTSTRAP
    assert protocol["scoring_gates"] == SCORING_GATES
    assert BOOTSTRAP["replicates"] == 10_000
    assert BOOTSTRAP["seed"] == 29_144

    changed = deepcopy(protocol)
    changed["scoring_gates"]["balanced_add_one"][
        "attention_class_mean_effect_bootstrap_lower_strict_min"
    ] += 0.001
    with pytest.raises(ValueError, match="scoring gates"):
        validate_causal_carrier_protocol(changed)

    masks = build_condition_masks()
    clean_prior = torch.zeros((2, COMPONENT_COUNT), dtype=torch.float64)
    clean_prior[0, :24] = 1.0
    clean_prior[1, :31] = 1.0
    overlap = validate_no_prior_c32_c33_overlap(masks, (clean_prior,))
    assert overlap["overlap_count"] == 0

    collision = masks[torch.where(masks.sum(dim=1) == 32)[0][0]].clone()[None]
    with pytest.raises(ValueError, match="overlap prior c32/c33 masks"):
        validate_no_prior_c32_c33_overlap(masks, (collision,))


def test_module_is_response_blind_and_has_no_model_runner_surface() -> None:
    assert CONTEXT_COUNT == 12
    protocol = build_causal_carrier_protocol()
    boundaries = protocol["claim_boundaries"]
    assert boundaries["e29_14d_responses_used_to_construct_masks_or_gates"] is False
    assert boundaries["parent_posthoc_responses_motivated_attention_class_hypothesis"]
    assert boundaries["parent_posthoc_responses_selected_top4_factorial_components"]
    assert (
        boundaries["parent_responses_used_for_hash_balanced_scan_background_layouts"]
        is False
    )
    assert boundaries["parent_responses_used_to_set_numeric_gates"] is False
    assert protocol["claim_boundaries"]["model_calls_in_protocol_module"] is False
    assert protocol["claim_boundaries"]["perturbation_norm_role"] == (
        "mandatory_diagnostic_not_a_gate_or_matching_lane"
    )
