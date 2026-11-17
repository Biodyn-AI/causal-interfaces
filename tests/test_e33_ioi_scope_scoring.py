from __future__ import annotations

from copy import deepcopy

from experiments.e33_ioi_scope_replication.protocol import (
    CONDITION_IDS,
    CONTEXT_COUNT,
    DIRECTIONS,
)
from experiments.e33_ioi_scope_replication.scoring import score_context_records


def _passing_contexts() -> list[dict]:
    transfer = {
        "mlp0_all": 0.80,
        "mlp0_lexical": 0.80,
        "mlp0_shared_static": 0.0,
        "mlp0_matched_static_a": 0.0,
        "mlp0_matched_static_b": 0.0,
        "attention0_lexical": 0.10,
        "mlp0_lexical_energy_scrambled": 0.0,
        "first12_all": 0.98,
        "first12_lexical": 0.98,
        "all24_all": 1.0,
        "all24_lexical": 0.99,
    }
    fidelity = {
        condition: (
            0.90
            if condition == "mlp0_lexical"
            else 0.97
            if condition == "first12_lexical"
            else 0.99
            if condition == "all24_lexical"
            else 0.0
        )
        for condition in CONDITION_IDS
    }
    contexts = []
    for context_index in range(CONTEXT_COUNT):
        clean = {
            direction: {
                "recipient_pairwise_correct": True,
                "source_pairwise_correct": True,
                "endpoint": 5.0,
                "recipient_target_top1": True,
                "source_target_top1": True,
            }
            for direction in DIRECTIONS
        }
        directions = {}
        for direction in DIRECTIONS:
            directions[direction] = {
                condition: {
                    "normalized_transfer": transfer[condition],
                    "js_fidelity": fidelity[condition],
                    "counterfactual_pairwise_win": condition
                    in {"mlp0_lexical", "first12_lexical", "all24_lexical"},
                    "counterfactual_target_top1": condition
                    in {"mlp0_lexical", "first12_lexical", "all24_lexical"},
                    "patched_counterfactual_margin": 1.0,
                }
                for condition in CONDITION_IDS
            }
        contexts.append(
            {
                "context_id": f"context_{context_index}",
                "clean": clean,
                "directions": directions,
            }
        )
    return contexts


def test_passing_fixture_passes_every_registered_gate() -> None:
    score = score_context_records(_passing_contexts())
    assert score["all_registered_gates_passed"] is True


def test_scope_failure_is_not_hidden_by_other_positive_results() -> None:
    contexts = deepcopy(_passing_contexts())
    for context in contexts:
        for direction in DIRECTIONS:
            context["directions"][direction]["mlp0_lexical"][
                "normalized_transfer"
            ] = 0.50
    score = score_context_records(contexts)
    assert score["all_registered_gates_passed"] is False
    assert not score["gates"]["mlp0_scope_equivalence"][
        "forward_lexical_mean_transfer"
    ]

