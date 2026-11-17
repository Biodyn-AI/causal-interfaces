"""Focused tests for prospective E29.14a registration assembly."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_custody import (
    BATCH_INVARIANCE,
    PHYSICAL_QUERY_ACCOUNTING,
    SOURCE_FILES,
    canonical_sha256,
    validate_registration_payload,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    BOOTSTRAP,
    FORWARD_CONDITIONS,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
    SCORING_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_prompt_panel import (
    e29_14a_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14a_position import (
    build_registration_record,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "notes/E29.13-ABSORBING-HUB-CONFIRMATION-REGISTRATION.json"
DEVELOPMENT = (
    ROOT
    / "runs/e29_13_absorbing_hub_confirmation_v1/e29_13_position_semantics_posthoc.json"
)


def _checks() -> tuple[dict[str, object], ...]:
    checks = []
    for context_index, record in enumerate(e29_14a_prompt_panel()):
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


def _record() -> dict[str, object]:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    return build_registration_record(
        parent_registration_path=PARENT,
        parent=parent,
        development_path=DEVELOPMENT,
        development=development,
        prompts=e29_14a_prompt_panel(),
        prompt_checks=_checks(),
        source_hashes={name: "1" * 64 for name in SOURCE_FILES},
        environment={"python": "test"},
    )


def test_registration_seals_exact_manifest_gates_and_query_arithmetic() -> None:
    registration = _record()

    validate_registration_payload(registration, verify_external_files=False)
    assert registration["condition_manifest"] == {
        "forward": [dict(row) for row in FORWARD_CONDITIONS],
        "reverse": [dict(row) for row in REVERSE_CONDITIONS],
    }
    assert registration["bootstrap"] == BOOTSTRAP
    assert registration["scoring_gates"] == SCORING_GATES
    assert registration["query_accounting"] == QUERY_ACCOUNTING
    assert registration["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert registration["batch_invariance"] == BATCH_INVARIANCE
    assert registration["prompt_panel_sha256"] == canonical_sha256(
        registration["prompt_records"]
    )


def test_registration_rejects_condition_and_query_mutation() -> None:
    registration = _record()
    changed_condition = copy.deepcopy(registration)
    changed_condition["condition_manifest"]["forward"][0]["alpha"] = 0.9
    with pytest.raises(RuntimeError, match="condition manifest"):
        validate_registration_payload(
            changed_condition,
            verify_external_files=False,
        )

    changed_query = copy.deepcopy(registration)
    changed_query["physical_query_accounting"]["maximum_physical_model_examples"] = 249
    with pytest.raises(RuntimeError, match="physical query accounting"):
        validate_registration_payload(changed_query, verify_external_files=False)


def test_registration_rejects_tokenizer_and_checkpoint_mutation() -> None:
    registration = _record()
    changed_token = copy.deepcopy(registration)
    changed_token["prompt_checks"][0]["sides"]["donor"]["target_token_id"] = (
        changed_token["prompt_checks"][0]["sides"]["base"]["target_token_id"]
    )
    with pytest.raises(RuntimeError, match="target alignment"):
        validate_registration_payload(changed_token, verify_external_files=False)

    changed_checkpoint = copy.deepcopy(registration)
    changed_checkpoint["checkpoint_sha256"].pop("config.json")
    with pytest.raises(RuntimeError, match="checkpoint hashes keys changed"):
        validate_registration_payload(changed_checkpoint, verify_external_files=False)
