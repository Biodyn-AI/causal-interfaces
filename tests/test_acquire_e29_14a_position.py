"""Pure-runner tests for sealed E29.14a positional acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions import (
    acquire_e29_14a_position as acquisition_module,
)
from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14a_position import (
    ATTEMPT_KEYS,
    LABEL_KEYS,
    acquire_position,
    acquire_position_panel_with_runner,
    build_acquisition_attempt_record,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_custody import (
    PHYSICAL_QUERY_ACCOUNTING,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    FORWARD_CONDITIONS,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
)


def _checks() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "context_index": index,
            "template": f"template-{index // 3}",
            "base_prompt": f"base prompt {index}",
            "donor_prompt": f"donor prompt {index}",
            "base_target": f"base-{index}",
            "donor_target": f"donor-{index}",
            "token_length": 6,
            "target_positions_aligned": True,
            "sides": {
                "base": {
                    "target": f"base-{index}",
                    "target_token_id": 100 + 2 * index,
                    "demonstration_target_position": 2,
                    "prediction_position": 6,
                    "next_token_append_valid": True,
                    "word_token_lengths": {f"base-word-{index}": 1},
                },
                "donor": {
                    "target": f"donor-{index}",
                    "target_token_id": 101 + 2 * index,
                    "demonstration_target_position": 2,
                    "prediction_position": 6,
                    "next_token_append_valid": True,
                    "word_token_lengths": {f"donor-word-{index}": 1},
                },
            },
        }
        for index in range(12)
    )


def _clean_metadata(record: dict[str, Any], index: int) -> dict[str, Any]:
    base_margin = 2.0 + index / 10
    donor_margin = 3.0 + index / 10
    return {
        "template": record["template"],
        "base_prompt": record["base_prompt"],
        "donor_prompt": record["donor_prompt"],
        "base_target": record["base_target"],
        "donor_target": record["donor_target"],
        "base_target_token_id": 100 + 2 * index,
        "donor_target_token_id": 101 + 2 * index,
        "token_length": 6,
        "position_scope_indices": {
            "all_positions": [0, 1, 2, 3, 4, 5],
            "differing_lexical_positions": [0, 1, 2, 4, 5],
            "shared_static_positions": [3],
            "demonstration_target_position": [2],
            "demonstration_prefix_positions": [0, 1],
            "repeated_query_positions": [4, 5],
            "differing_except_demonstration_target": [0, 1, 4, 5],
        },
        "clean_base_correct_minus_donor_margin": base_margin,
        "clean_donor_correct_minus_base_margin": donor_margin,
        "clean_endpoint": base_margin + donor_margin,
        "clean_logit_absolute_max": 7.0,
        "device": "cpu",
        "model_dtype": "float64",
    }


class _FakeRunner:
    def __init__(
        self,
        *,
        bad_clean_context: int | None = None,
        batch_passed: bool = True,
    ) -> None:
        self.bad_clean_context = bad_clean_context
        self.batch_passed = batch_passed
        self.registered_calls: list[int] = []
        self.smoke_calls: list[tuple[int, int, int]] = []

    def prepare_context(self, record: dict[str, Any]) -> SimpleNamespace:
        index = int(record["base_target"].split("-")[-1])
        metadata = _clean_metadata(record, index)
        if index == self.bad_clean_context:
            metadata["clean_base_correct_minus_donor_margin"] = -0.1
        return SimpleNamespace(index=index, clean_metadata=metadata)

    def registered_responses(
        self, prepared: SimpleNamespace
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.registered_calls.append(prepared.index)
        endpoint = float(prepared.clean_metadata["clean_endpoint"])
        forward = endpoint * (
            torch.arange(len(FORWARD_CONDITIONS), dtype=torch.float64) + 1.0
        ) / 20.0
        reverse = endpoint * (
            torch.arange(len(REVERSE_CONDITIONS), dtype=torch.float64) + 1.0
        ) / 20.0
        metadata = {
            **prepared.clean_metadata,
            "patched_logit_absolute_max": 8.0,
            "patched_response_absolute_max": float(
                torch.cat((forward, reverse)).abs().max()
            ),
            "all_logits_finite": True,
            "forward_condition_rows": len(FORWARD_CONDITIONS),
            "reverse_condition_rows": len(REVERSE_CONDITIONS),
            "patched_condition_cells": len(FORWARD_CONDITIONS)
            + len(REVERSE_CONDITIONS),
            "clean_forward_examples": 2,
        }
        return {"forward": forward, "reverse": reverse}, metadata

    def registered_batch_invariance_smoke(
        self,
        prepared: SimpleNamespace,
        *,
        reference_batch_size: int,
        candidate_batch_size: int,
    ) -> dict[str, Any]:
        self.smoke_calls.append(
            (prepared.index, reference_batch_size, candidate_batch_size)
        )
        records = {
            direction: {
                "direction": direction,
                "condition_rows": len(conditions),
                "reference_batch_size": reference_batch_size,
                "candidate_batch_size": candidate_batch_size,
                "maximum_absolute_difference": 0.0,
                "maximum_relative_difference": 0.0,
                "atol": 1e-9,
                "rtol": 1e-9,
                "passed": self.batch_passed,
            }
            for direction, conditions in (
                ("forward", FORWARD_CONDITIONS),
                ("reverse", REVERSE_CONDITIONS),
            )
        }
        return {
            "records": records,
            "unique_condition_rows": 16,
            "physical_condition_cells": 32,
            "scientific_labels_produced": False,
            "passed": self.batch_passed,
        }


def test_acquires_exact_nonadaptive_panel_and_one_full_manifest_smoke() -> None:
    runner = _FakeRunner()

    artifact = acquire_position_panel_with_runner(
        runner,
        prompt_checks=_checks(),
        registration_sha256="a" * 64,
        expected_registration_sha256_supplied="a" * 64,
    )

    assert set(artifact) == LABEL_KEYS
    assert artifact["forward_responses"].shape == (12, 12)
    assert artifact["reverse_responses"].shape == (12, 4)
    assert artifact["clean_endpoints"].shape == (12,)
    assert artifact["query_accounting"] == QUERY_ACCOUNTING
    assert artifact["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert runner.registered_calls == list(range(12))
    assert runner.smoke_calls == [(0, 1, 32)]
    assert artifact["batch_invariance_smoke"]["passed"] is True
    assert artifact["all_integrity_gates_passed"] is True


def test_clean_gate_failure_precedes_every_patched_condition() -> None:
    runner = _FakeRunner(bad_clean_context=4)

    with pytest.raises(RuntimeError, match="clean correctness"):
        acquire_position_panel_with_runner(runner, prompt_checks=_checks())

    assert runner.registered_calls == []
    assert runner.smoke_calls == []


def test_runtime_token_drift_precedes_every_patched_condition() -> None:
    runner = _FakeRunner()
    checks = list(_checks())
    checks[3] = {
        **checks[3],
        "sides": {
            **checks[3]["sides"],
            "base": {
                **checks[3]["sides"]["base"],
                "target_token_id": 999,
            },
        },
    }

    with pytest.raises(RuntimeError, match="target token"):
        acquire_position_panel_with_runner(runner, prompt_checks=checks)

    assert runner.registered_calls == []


def test_batch_invariance_failure_produces_no_label_record() -> None:
    runner = _FakeRunner(batch_passed=False)

    with pytest.raises(RuntimeError, match="batch invariance"):
        acquire_position_panel_with_runner(runner, prompt_checks=_checks())

    assert runner.registered_calls == list(range(12))
    assert runner.smoke_calls == [(0, 1, 32)]


class _BadShapeRunner(_FakeRunner):
    def registered_responses(
        self, prepared: SimpleNamespace
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        responses, metadata = super().registered_responses(prepared)
        responses["forward"] = responses["forward"][:-1]
        return responses, metadata


def test_response_shape_drift_fails_closed() -> None:
    runner = _BadShapeRunner()

    with pytest.raises(RuntimeError, match="forward response"):
        acquire_position_panel_with_runner(runner, prompt_checks=_checks())

    assert runner.smoke_calls == []


def test_attempt_marker_consumes_the_registered_query_budget() -> None:
    registration = {
        "source_code_sha256": {"source.py": "1" * 64},
        "checkpoint_sha256": {"model.safetensors": "2" * 64},
        "prompt_panel_sha256": "3" * 64,
        "position_protocol_sha256": "4" * 64,
        "environment": {"python": "3.12.9"},
        "execution": {"batch_size": 32},
    }
    marker = build_acquisition_attempt_record(
        registration=registration,
        registration_path=Path("registration.json"),
        observed_registration_sha256="5" * 64,
        expected_registration_sha256="5" * 64,
    )

    assert set(marker) == ATTEMPT_KEYS
    assert marker["status"] == "attempt_consumed_before_model_load"
    assert marker["query_accounting"] == QUERY_ACCOUNTING
    assert marker["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING


def test_entrypoint_writes_attempt_before_load_and_blocks_second_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}\n", encoding="utf-8")
    expected_sha = acquisition_module.sha256(registration_path)
    environment = {"python": "test"}
    registration = {
        "environment": environment,
        "execution": {
            "batch_size": 32,
            "device": "cpu",
            "model_dtype": "float64",
            "attention_implementation": "eager",
        },
        "source_code_sha256": {"source.py": "1" * 64},
        "checkpoint_sha256": {"model.safetensors": "2" * 64},
        "prompt_panel_sha256": "3" * 64,
        "position_protocol_sha256": "4" * 64,
        "model_snapshot": str(tmp_path / "snapshot"),
    }
    verification_calls = 0

    def load_registration(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal verification_calls
        verification_calls += 1
        return registration

    monkeypatch.setattr(
        acquisition_module, "load_verified_registration", load_registration
    )
    monkeypatch.setattr(
        acquisition_module,
        "validate_registration_payload",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        acquisition_module, "runtime_environment", lambda: environment
    )

    output_directory = tmp_path / "output"
    tokenizer_loads = 0
    model_loads = 0

    def stop_at_tokenizer_load(*args: Any, **kwargs: Any) -> None:
        nonlocal tokenizer_loads
        tokenizer_loads += 1
        attempt_path = output_directory / "position_acquisition_attempt.json"
        assert attempt_path.is_file()
        marker = json.loads(attempt_path.read_text(encoding="utf-8"))
        assert set(marker) == ATTEMPT_KEYS
        assert marker["status"] == "attempt_consumed_before_model_load"
        raise RuntimeError("test stop at tokenizer load")

    def reject_model_load(*args: Any, **kwargs: Any) -> None:
        nonlocal model_loads
        model_loads += 1
        raise AssertionError("model load preceded the attempt marker")

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", stop_at_tokenizer_load
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", reject_model_load
    )

    with pytest.raises(RuntimeError, match="test stop at tokenizer load"):
        acquire_position(
            registration_path=registration_path,
            expected_registration_sha256=expected_sha,
            output_directory=output_directory,
            batch_size=32,
        )
    assert verification_calls == 1
    assert tokenizer_loads == 1
    assert model_loads == 0

    with pytest.raises(FileExistsError, match="rerun"):
        acquire_position(
            registration_path=registration_path,
            expected_registration_sha256=expected_sha,
            output_directory=output_directory,
            batch_size=32,
        )
    assert verification_calls == 1
    assert tokenizer_loads == 1
    assert model_loads == 0
