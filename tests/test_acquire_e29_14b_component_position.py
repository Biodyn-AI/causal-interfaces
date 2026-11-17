"""Focused tests for fail-closed E29.14b acquisition."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions import (
    acquire_e29_14b_component_position as acquisition,
)
from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14b_component_position import (
    LABEL_KEYS,
    acquire_component_position,
    acquire_component_position_panel_with_runner,
    build_condition_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    BATCH_INVARIANCE,
    PHYSICAL_QUERY_ACCOUNTING,
    sha256,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14b_component_position import (
    load_prior_bridge_masks,
)


def _checks() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "context_index": index,
            "template": f"template-{index // 3}",
            "base_words": [f"b{index}a", f"b{index}b", f"b{index}c", f"base-{index}"],
            "donor_words": [f"d{index}a", f"d{index}b", f"d{index}c", f"donor-{index}"],
            "base_prompt": f"base prompt {index}",
            "donor_prompt": f"donor prompt {index}",
            "base_target": f"base-{index}",
            "donor_target": f"donor-{index}",
            "token_length": 5,
            "target_positions_aligned": True,
            "sides": {
                "base": {
                    "target": f"base-{index}",
                    "target_token_id": 100 + 2 * index,
                    "demonstration_target_position": 2,
                    "prediction_position": 5,
                    "next_token_append_valid": True,
                    "word_token_lengths": {
                        f"b{index}a": 1,
                        f"b{index}b": 1,
                        f"b{index}c": 1,
                        f"base-{index}": 1,
                    },
                },
                "donor": {
                    "target": f"donor-{index}",
                    "target_token_id": 101 + 2 * index,
                    "demonstration_target_position": 2,
                    "prediction_position": 5,
                    "next_token_append_valid": True,
                    "word_token_lengths": {
                        f"d{index}a": 1,
                        f"d{index}b": 1,
                        f"d{index}c": 1,
                        f"donor-{index}": 1,
                    },
                },
            },
        }
        for index in range(12)
    )


class _FakeRunner:
    def __init__(self, *, bad_clean_index: int | None = None, smoke_pass: bool = True):
        self.bad_clean_index = bad_clean_index
        self.smoke_pass = smoke_pass
        self.prepare_count = 0
        self.paired_calls = 0

    def prepare_context(self, record: dict[str, str]) -> SimpleNamespace:
        index = int(record["base_target"].split("-")[-1])
        self.prepare_count += 1
        base_margin = 2.0 + index / 10
        if index == self.bad_clean_index:
            base_margin = -0.1
        donor_margin = 3.0 + index / 10
        base_ids = torch.tensor([[9, 10 + index, 100 + 2 * index, 11, 12]])
        donor_ids = torch.tensor([[9, 20 + index, 101 + 2 * index, 21, 22]])
        metadata = {
            **record,
            "base_target_token_id": 100 + 2 * index,
            "donor_target_token_id": 101 + 2 * index,
            "token_length": 5,
            "model_layer_count": 24,
            "component_count": 48,
            "component_semantics": "attention_and_mlp_residual_updates",
            "patch_direction": "donor_to_base_only",
            "position_scope_order": list(POSITION_SCOPE_ORDER),
            "position_scope_indices": {
                "all_positions": [0, 1, 2, 3, 4],
                "differing_lexical_positions": [1, 2, 3, 4],
            },
            "clean_base_correct_minus_donor_margin": base_margin,
            "clean_donor_correct_minus_base_margin": donor_margin,
            "clean_endpoint": base_margin + donor_margin,
            "clean_base_logit_absolute_max": 7.0,
            "clean_donor_logit_absolute_max": 8.0,
            "clean_logit_absolute_max": 8.0,
            "clean_logits_finite": True,
            "device": "cpu",
            "model_dtype": "float64",
        }
        return SimpleNamespace(
            base_input_ids=base_ids,
            donor_input_ids=donor_ids,
            clean_metadata=metadata,
        )

    def paired_responses_from_prepared(
        self, context: SimpleNamespace, masks: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        assert self.prepare_count == 12
        self.paired_calls += 1
        endpoint = float(context.clean_metadata["clean_endpoint"])
        all_values = endpoint * (
            0.70 + 0.001 * torch.arange(len(masks), dtype=torch.float64)
        )
        return {
            "all_positions": all_values,
            "differing_lexical_positions": all_values + 0.002 * endpoint,
        }

    def full_batch_invariance_smoke(
        self,
        _context: SimpleNamespace,
        masks: torch.Tensor,
        *,
        reference_batch_size: int,
        candidate_batch_size: int,
    ) -> dict[str, Any]:
        records = {
            scope: {
                "mask_rows": len(masks),
                "reference_batch_size": reference_batch_size,
                "candidate_batch_size": candidate_batch_size,
                "maximum_absolute_difference": 0.0,
                "maximum_relative_difference": 0.0,
                "atol": BATCH_INVARIANCE["absolute_tolerance"],
                "rtol": BATCH_INVARIANCE["relative_tolerance"],
                "passed": self.smoke_pass,
            }
            for scope in POSITION_SCOPE_ORDER
        }
        return {
            "records": records,
            "mask_rows": len(masks),
            "unique_mask_scope_cells": 2 * len(masks),
            "physical_mask_scope_cells": 4 * len(masks),
            "scientific_labels_produced": False,
            "passed": self.smoke_pass,
        }


def test_complete_panel_shapes_order_and_accounting() -> None:
    masks = load_prior_bridge_masks()
    runner = _FakeRunner()

    labels = acquire_component_position_panel_with_runner(
        runner, prompt_checks=_checks(), masks=masks
    )

    assert set(labels) == LABEL_KEYS
    assert runner.paired_calls == 12
    assert labels["singleton_all_response"].shape == (12,)
    assert labels["responses_in_registered_order"].shape == (194, 12)
    for split in MASK_SPLITS:
        assert labels["all_position_responses"][split].shape == (48, 12)
        assert labels["lexical_position_responses"][split].shape == (48, 12)
    manifest = labels["condition_manifest"]
    assert manifest == build_condition_manifest(masks)
    assert manifest[0]["scope"] == "all_positions"
    assert manifest[1]["scope"] == "differing_lexical_positions"
    torch.testing.assert_close(
        labels["responses_in_registered_order"][2],
        labels["all_position_responses"][MASK_SPLITS[0]][0],
    )
    torch.testing.assert_close(
        labels["responses_in_registered_order"][3],
        labels["lexical_position_responses"][MASK_SPLITS[0]][0],
    )
    assert labels["query_accounting"] == QUERY_ACCOUNTING
    assert labels["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING


def test_every_clean_context_is_checked_before_any_patch() -> None:
    runner = _FakeRunner(bad_clean_index=7)

    with pytest.raises(RuntimeError, match="clean correctness"):
        acquire_component_position_panel_with_runner(
            runner, prompt_checks=_checks(), masks=load_prior_bridge_masks()
        )

    assert runner.prepare_count == 12
    assert runner.paired_calls == 0


def test_failed_full_smoke_admits_no_label_artifact() -> None:
    runner = _FakeRunner(smoke_pass=False)

    with pytest.raises(RuntimeError, match="batch invariance"):
        acquire_component_position_panel_with_runner(
            runner, prompt_checks=_checks(), masks=load_prior_bridge_masks()
        )


def test_attempt_marker_precedes_load_and_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}\n", encoding="utf-8")
    expected_sha = sha256(registration_path)
    output = tmp_path / "output"
    registration = {
        "environment": {"python": "test"},
        "execution": {
            "batch_size": 32,
            "device": "cpu",
            "model_dtype": "float64",
            "attention_implementation": "eager",
        },
        "prior_protocol_masks": "masks.pt",
        "prior_protocol_masks_sha256": "1" * 64,
        "source_code_sha256": {"source.py": "2" * 64},
        "checkpoint_sha256": {"model.safetensors": "3" * 64},
        "prompt_panel_sha256": "4" * 64,
        "component_position_protocol_sha256": "5" * 64,
        "model_snapshot": "snapshot",
    }
    events: list[str] = []
    monkeypatch.setattr(acquisition, "load_verified_registration", lambda *a, **k: registration)
    monkeypatch.setattr(acquisition, "validate_registration_payload", lambda value, **k: value)
    monkeypatch.setattr(acquisition, "runtime_environment", lambda: registration["environment"])
    monkeypatch.setattr(acquisition, "load_prior_bridge_masks", lambda _path: load_prior_bridge_masks())

    def stop_at_tokenizer(_snapshot: Path) -> None:
        assert (output / "component_position_acquisition_attempt.json").is_file()
        events.append("tokenizer")
        raise RuntimeError("deliberate tokenizer stop")

    monkeypatch.setattr(acquisition, "_load_tokenizer", stop_at_tokenizer)
    monkeypatch.setattr(
        acquisition,
        "_load_model",
        lambda _snapshot: events.append("model"),
    )

    with pytest.raises(RuntimeError, match="deliberate tokenizer stop"):
        acquire_component_position(
            registration_path=registration_path,
            expected_registration_sha256=expected_sha,
            output_directory=output,
            batch_size=32,
        )
    assert events == ["tokenizer"]

    monkeypatch.setattr(
        acquisition,
        "load_verified_registration",
        lambda *a, **k: events.append("registration"),
    )
    with pytest.raises(FileExistsError, match="rerun"):
        acquire_component_position(
            registration_path=registration_path,
            expected_registration_sha256=expected_sha,
            output_directory=output,
            batch_size=32,
        )
    assert events == ["tokenizer"]
