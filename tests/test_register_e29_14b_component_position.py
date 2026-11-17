"""Focused tests for response-blind E29.14b registration assembly."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    BATCH_INVARIANCE,
    PHYSICAL_QUERY_ACCOUNTING,
    SOURCE_FILES,
    validate_registration_payload,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    BOOTSTRAP,
    EXPECTED_MASK_CANONICAL_SHA256,
    MASK_SPLITS,
    QUERY_ACCOUNTING,
    SCORING_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_prompt_panel import (
    e29_14b_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14b_component_position import (
    PARENT_REGISTRATION,
    PRIOR_PROTOCOL_MASKS,
    build_registration_record,
    load_prior_bridge_masks,
)


def _checks() -> tuple[dict[str, object], ...]:
    checks = []
    for context_index, record in enumerate(e29_14b_prompt_panel()):
        checks.append(
            {
                "context_index": context_index,
                "token_length": 10,
                "target_positions_aligned": True,
                "sides": {
                    side: {
                        "target": record[f"{side}_target"],
                        "target_token_id": 100 + 2 * context_index + offset,
                        "demonstration_target_position": 4,
                        "prediction_position": 10,
                        "next_token_append_valid": True,
                        "word_token_lengths": {
                            word: 1 for word in record[f"{side}_words"]
                        },
                    }
                    for offset, side in enumerate(("base", "donor"))
                },
                **record,
            }
        )
    return tuple(checks)


def _record() -> dict[str, Any]:
    parent = json.loads(PARENT_REGISTRATION.read_text(encoding="utf-8"))
    masks = load_prior_bridge_masks()
    return build_registration_record(
        parent_registration_path=PARENT_REGISTRATION,
        parent=parent,
        prior_masks_path=PRIOR_PROTOCOL_MASKS,
        held_masks=masks,
        prompts=e29_14b_prompt_panel(),
        prompt_checks=_checks(),
        source_hashes={name: "1" * 64 for name in SOURCE_FILES},
        environment={"python": "test"},
    )


def test_registration_seals_exact_masks_gates_and_query_arithmetic() -> None:
    registration = _record()

    validate_registration_payload(registration, verify_external_files=False)

    assert registration["bootstrap"] == BOOTSTRAP
    assert registration["scoring_gates"] == SCORING_GATES
    assert registration["query_accounting"] == QUERY_ACCOUNTING
    assert registration["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert registration["batch_invariance"] == BATCH_INVARIANCE
    assert registration["prior_mask_canonical_sha256"] == (
        EXPECTED_MASK_CANONICAL_SHA256
    )
    protocol = registration["component_position_protocol"]
    assert protocol["query_accounting"] == QUERY_ACCOUNTING
    assert protocol["mask_splits"] == list(MASK_SPLITS)
    assert len(protocol["singleton_conditions"]) == 2
    assert sum(len(rows) for rows in protocol["mask_rows"].values()) == 96


def test_registration_rejects_protocol_query_and_checkpoint_mutations() -> None:
    registration = _record()

    changed_protocol = copy.deepcopy(registration)
    changed_protocol["component_position_protocol"]["mask_rows"][
        "confirmation_size24"
    ][0][0] = 47
    with pytest.raises((RuntimeError, ValueError), match="mask|protocol|hub"):
        validate_registration_payload(changed_protocol, verify_external_files=False)

    changed_query = copy.deepcopy(registration)
    changed_query["physical_query_accounting"][
        "maximum_physical_model_examples"
    ] += 1
    with pytest.raises(RuntimeError, match="physical query accounting"):
        validate_registration_payload(changed_query, verify_external_files=False)

    changed_checkpoint = copy.deepcopy(registration)
    changed_checkpoint["checkpoint_sha256"] = {}
    # The exact parent comparison occurs with external validation. The local
    # schema still rejects an empty checkpoint contract explicitly.
    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_registration_payload(changed_checkpoint, verify_external_files=False)


def test_mask_loader_reads_only_the_exact_response_blind_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = torch.load
    observed: list[tuple[Path, object, object]] = []

    def guarded_load(path: Path, *, map_location: object, weights_only: object) -> Any:
        observed.append((Path(path), map_location, weights_only))
        assert Path(path).resolve() == PRIOR_PROTOCOL_MASKS.resolve()
        assert "labels" not in Path(path).name
        return original_load(path, map_location=map_location, weights_only=weights_only)

    monkeypatch.setattr(torch, "load", guarded_load)

    masks = load_prior_bridge_masks()

    assert tuple(masks) == MASK_SPLITS
    assert observed == [(PRIOR_PROTOCOL_MASKS, "cpu", True)]


def test_registration_contains_no_prior_outcome_or_label_dependency() -> None:
    registration = _record()
    serialized = json.dumps(registration, sort_keys=True)

    assert "position_semantics_summary" not in serialized
    assert "position_labels" not in serialized
    assert "fit_labels" not in serialized
    assert "held_labels" not in serialized
    assert (
        registration["parent_e29_14a_registration_sha256"]
        == "f6ef1755bd6a4c3345a78fe180f00d7dd5e98086319348f34d9a1d9592ee23df"
    )
