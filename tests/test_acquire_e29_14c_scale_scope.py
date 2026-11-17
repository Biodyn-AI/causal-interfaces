"""Focused pure-runner and custody-order tests for E29.14c acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions import (
    acquire_e29_14c_scale_scope as acquisition_module,
)
from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14c_scale_scope import (
    CUSTODY_KEYS,
    LABEL_KEYS,
    acquire_component_position_panel_with_runner,
    acquire_scale_scope,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    BATCH_INVARIANCE,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    COMPONENT_COUNT,
    HUB_COMPONENT,
    MASK_SHAPES,
    MASK_SPLITS,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    build_scale_masks,
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
        "model_layer_count": 24,
        "component_count": 48,
        "component_semantics": "attention_and_mlp_residual_updates",
        "patch_direction": "donor_to_base_only",
        "position_scope_order": list(POSITION_SCOPE_ORDER),
        "position_scope_indices": {
            "all_positions": [0, 1, 2, 3, 4, 5],
            "differing_lexical_positions": [0, 1, 2, 4, 5],
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


def _clean_contract() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "context_index": index,
            "template": check["template"],
            "base_prompt": check["base_prompt"],
            "donor_prompt": check["donor_prompt"],
            "base_target": check["base_target"],
            "donor_target": check["donor_target"],
            "base_target_token_id": check["sides"]["base"]["target_token_id"],
            "donor_target_token_id": check["sides"]["donor"]["target_token_id"],
            "token_length": check["token_length"],
            "position_scope_indices": {
                "all_positions": [0, 1, 2, 3, 4, 5],
                "differing_lexical_positions": [0, 1, 2, 4, 5],
            },
        }
        for index, check in enumerate(_checks())
    )


class _FakeRunner:
    def __init__(
        self,
        *,
        bad_clean_context: int | None = None,
        batch_passed: bool = True,
    ) -> None:
        self.bad_clean_context = bad_clean_context
        self.batch_passed = batch_passed
        self.prepare_calls: list[int] = []
        self.paired_calls: list[tuple[int, torch.Tensor]] = []
        self.smoke_calls: list[tuple[int, int, int, int]] = []

    def prepare_context(self, record: dict[str, Any]) -> SimpleNamespace:
        index = int(record["base_target"].split("-")[-1])
        self.prepare_calls.append(index)
        metadata = _clean_metadata(record, index)
        if index == self.bad_clean_context:
            metadata["clean_base_correct_minus_donor_margin"] = -0.1
        base_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        donor_ids = torch.tensor([[20, 21, 22, 13, 24, 25]])
        return SimpleNamespace(
            index=index,
            base_input_ids=base_ids,
            donor_input_ids=donor_ids,
            clean_metadata=metadata,
        )

    def paired_responses_from_prepared(
        self, prepared: SimpleNamespace, masks: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.paired_calls.append((prepared.index, masks.clone()))
        base = torch.arange(len(masks), dtype=torch.float64) / 100.0
        base = base + prepared.index / 1_000.0
        return {
            "all_positions": base,
            "differing_lexical_positions": base + 0.25,
        }

    def full_batch_invariance_smoke(
        self,
        prepared: SimpleNamespace,
        masks: torch.Tensor,
        *,
        reference_batch_size: int,
        candidate_batch_size: int,
    ) -> dict[str, Any]:
        self.smoke_calls.append(
            (
                prepared.index,
                len(masks),
                reference_batch_size,
                candidate_batch_size,
            )
        )
        records = {
            scope: {
                "mask_rows": len(masks),
                "reference_batch_size": reference_batch_size,
                "candidate_batch_size": candidate_batch_size,
                "maximum_absolute_difference": 0.0,
                "maximum_relative_difference": 0.0,
                "atol": BATCH_INVARIANCE["absolute_tolerance"],
                "rtol": BATCH_INVARIANCE["relative_tolerance"],
                "passed": self.batch_passed,
            }
            for scope in POSITION_SCOPE_ORDER
        }
        return {
            "records": records,
            "mask_rows": len(masks),
            "unique_mask_scope_cells": 2 * len(masks),
            "physical_mask_scope_cells": 4 * len(masks),
            "scientific_labels_produced": False,
            "passed": self.batch_passed,
        }


def test_acquires_exact_fixed_panel_and_preserves_registered_table_order() -> None:
    runner = _FakeRunner()

    artifact = acquire_component_position_panel_with_runner(
        runner,
        prompt_checks=_checks(),
        masks=build_scale_masks(),
        clean_metadata_contract=_clean_contract(),
        registration_sha256="a" * 64,
        expected_registration_sha256_supplied="a" * 64,
    )

    assert set(artifact) == LABEL_KEYS
    assert artifact["singleton_all_response"].shape == (12,)
    assert artifact["singleton_lexical_response"].shape == (12,)
    assert tuple(artifact["all_position_responses"]) == MASK_SPLITS
    assert tuple(artifact["lexical_position_responses"]) == MASK_SPLITS
    assert {
        split: tuple(table.shape)
        for split, table in artifact["all_position_responses"].items()
    } == {split: (MASK_SHAPES[split][0], 12) for split in MASK_SPLITS}
    assert artifact["query_accounting"] == QUERY_ACCOUNTING
    assert artifact["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert runner.prepare_calls == list(range(12))
    assert [index for index, _masks in runner.paired_calls] == list(range(12))
    assert all(masks.shape == (143, 48) for _index, masks in runner.paired_calls)
    first_panel = runner.paired_calls[0][1]
    assert torch.equal(first_panel[0], torch.nn.functional.one_hot(
        torch.tensor(HUB_COMPONENT), COMPONENT_COUNT
    ).double())
    assert runner.smoke_calls == [(0, 143, 1, 32)]
    assert artifact["all_integrity_gates_passed"] is True

    manifest = artifact["condition_manifest"]
    assert len(manifest) == 286
    assert [row["condition_index"] for row in manifest] == list(range(286))
    assert [row["scope"] for row in manifest] == list(POSITION_SCOPE_ORDER) * 143
    assert manifest[:2] == [
        {
            "condition_index": 0,
            "mask_group": "singleton_hub",
            "mask_row": 0,
            "scope": "all_positions",
            "alpha": 1.0,
        },
        {
            "condition_index": 1,
            "mask_group": "singleton_hub",
            "mask_row": 0,
            "scope": "differing_lexical_positions",
            "alpha": 1.0,
        },
    ]


def test_all_clean_contexts_are_verified_before_any_patched_query() -> None:
    runner = _FakeRunner(bad_clean_context=4)

    with pytest.raises(RuntimeError, match="clean correctness"):
        acquire_component_position_panel_with_runner(
            runner,
            prompt_checks=_checks(),
            masks=build_scale_masks(),
            clean_metadata_contract=_clean_contract(),
        )

    assert runner.prepare_calls == list(range(12))
    assert runner.paired_calls == []
    assert runner.smoke_calls == []


def test_scope_drift_fails_before_any_patched_query() -> None:
    class _BadScopeRunner(_FakeRunner):
        def prepare_context(self, record: dict[str, Any]) -> SimpleNamespace:
            prepared = super().prepare_context(record)
            if prepared.index == 9:
                prepared.clean_metadata["position_scope_indices"][
                    "differing_lexical_positions"
                ] = [0, 1, 2]
            return prepared

    runner = _BadScopeRunner()
    with pytest.raises(RuntimeError, match="V2 seal"):
        acquire_component_position_panel_with_runner(
            runner,
            prompt_checks=_checks(),
            masks=build_scale_masks(),
            clean_metadata_contract=_clean_contract(),
        )

    assert runner.paired_calls == []
    assert runner.smoke_calls == []


def test_response_shape_and_batch_smoke_fail_closed() -> None:
    class _BadShapeRunner(_FakeRunner):
        def paired_responses_from_prepared(
            self, prepared: SimpleNamespace, masks: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            responses = super().paired_responses_from_prepared(prepared, masks)
            responses["all_positions"] = responses["all_positions"][:-1]
            return responses

    bad_shape = _BadShapeRunner()
    with pytest.raises(RuntimeError, match="response shape"):
        acquire_component_position_panel_with_runner(
            bad_shape,
            prompt_checks=_checks(),
            masks=build_scale_masks(),
            clean_metadata_contract=_clean_contract(),
        )
    assert bad_shape.smoke_calls == []

    bad_smoke = _FakeRunner(batch_passed=False)
    with pytest.raises(RuntimeError, match="batch invariance"):
        acquire_component_position_panel_with_runner(
            bad_smoke,
            prompt_checks=_checks(),
            masks=build_scale_masks(),
            clean_metadata_contract=_clean_contract(),
        )
    assert len(bad_smoke.paired_calls) == 12
    assert bad_smoke.smoke_calls == [(0, 143, 1, 32)]


def test_entrypoint_orders_mask_verification_marker_model_outcomes_and_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}\n", encoding="utf-8")
    expected_sha = acquisition_module.sha256(registration_path)
    environment = {"python": "test"}
    protocol_masks_path = tmp_path / "protocol_masks.pt"
    protocol_masks_path.write_bytes(b"sealed")
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
        "protocol_masks": str(protocol_masks_path),
        "protocol_masks_sha256": "3" * 64,
        "protocol_masks_schema": "test_masks_v1",
        "prompt_panel_sha256": "4" * 64,
        "clean_metadata_contract": list(_clean_contract()),
        "clean_metadata_contract_sha256": "6" * 64,
        "scale_scope_protocol_sha256": "5" * 64,
        "model_snapshot": str(tmp_path / "snapshot"),
        "component_map": [
            [layer, kind]
            for layer in range(24)
            for kind in ("attention", "mlp")
        ],
        "prompt_checks": list(_checks()),
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
    }
    events: list[str] = []

    def load_registration(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("load_registration")
        return registration

    def validate_registration(value: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        events.append("validate_registration")
        return value

    def load_masks(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        events.append("masks")
        return build_scale_masks()

    monkeypatch.setattr(
        acquisition_module, "load_verified_registration", load_registration
    )
    monkeypatch.setattr(
        acquisition_module, "validate_registration_payload", validate_registration
    )
    monkeypatch.setattr(
        acquisition_module, "load_verified_protocol_masks", load_masks
    )
    monkeypatch.setattr(
        acquisition_module, "runtime_environment", lambda: environment
    )
    monkeypatch.setattr(
        acquisition_module,
        "build_acquisition_attempt_record",
        lambda **kwargs: {"status": "attempt_consumed_before_model_load"},
    )

    output_directory = tmp_path / "output"

    def create_marker(record: dict[str, Any], path: Path) -> str:
        assert events == ["load_registration", "validate_registration", "masks"]
        events.append("marker")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return acquisition_module.sha256(path)

    monkeypatch.setattr(
        acquisition_module, "create_acquisition_attempt_marker", create_marker
    )

    class _TokenizerLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            assert (output_directory / "scale_scope_acquisition_attempt.json").is_file()
            events.append("tokenizer")
            return object()

    class _LoadedModel:
        def eval(self) -> _LoadedModel:
            return self

    class _ModelLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> _LoadedModel:
            assert events[-1] == "tokenizer"
            events.append("model")
            return _LoadedModel()

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers, "AutoTokenizer", _TokenizerLoader)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _ModelLoader)

    expected_components = tuple(
        (int(layer), str(kind)) for layer, kind in registration["component_map"]
    )

    def runner_factory(*args: Any, **kwargs: Any) -> SimpleNamespace:
        assert events[-1] == "model"
        events.append("runner")
        return SimpleNamespace(components=expected_components)

    monkeypatch.setattr(
        acquisition_module, "Float64ComponentPositionRunner", runner_factory
    )

    def acquire_panel(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert events[-1] == "runner"
        events.append("outcomes")
        return {
            "numerical_summary": {
                "combined_responses": {"absolute_max": 1.25}
            }
        }

    monkeypatch.setattr(
        acquisition_module,
        "acquire_component_position_panel_with_runner",
        acquire_panel,
    )

    def save_labels(record: Any, path: Path) -> None:
        assert events[-1] == "outcomes"
        events.append("labels")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, path)

    def save_custody(record: Any, path: Path) -> None:
        assert events[-1] == "labels"
        assert set(record) == CUSTODY_KEYS
        events.append("custody")
        path.write_text(json.dumps(record), encoding="utf-8")

    monkeypatch.setattr(acquisition_module, "atomic_torch_save", save_labels)
    monkeypatch.setattr(acquisition_module, "atomic_json_save", save_custody)

    result = acquire_scale_scope(
        registration_path=registration_path,
        expected_registration_sha256=expected_sha,
        output_directory=output_directory,
        batch_size=32,
    )

    assert events == [
        "load_registration",
        "validate_registration",
        "masks",
        "marker",
        "tokenizer",
        "model",
        "runner",
        "outcomes",
        "labels",
        "custody",
    ]
    assert set(result) == CUSTODY_KEYS | {"custody_sha256"}
    assert result["mask_rows"] == 143
    assert result["patched_condition_rows_per_context"] == 286
    assert result["batch_smoke_unique_condition_rows"] == 286
    assert result["batch_smoke_physical_condition_cells"] == 572
    assert result["maximum_logical_queries"] == 3_456
    assert result["maximum_physical_model_examples"] == 4_028

    with pytest.raises(FileExistsError, match="rerun"):
        acquire_scale_scope(
            registration_path=registration_path,
            expected_registration_sha256=expected_sha,
            output_directory=output_directory,
            batch_size=32,
        )
    assert events[-1] == "custody"
