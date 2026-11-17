"""Pure-runner tests for E29.14d acquisition and diagnostic custody."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions import (
    acquire_e29_14d_direct_backup as acquisition_module,
)
from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14d_direct_backup import (
    CUSTODY_KEYS,
    LABEL_KEYS,
    acquire_direct_backup,
    acquire_direct_backup_panel_with_runner,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    build_condition_manifest,
    build_condition_masks,
    build_query_manifest,
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
        for index in range(CONTEXT_COUNT)
    )


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
            "token_length": 6,
            "position_scope_indices": {
                "all_positions": [0, 1, 2, 3, 4, 5],
                "differing_lexical_positions": [0, 1, 2, 4, 5],
            },
        }
        for index, check in enumerate(_checks())
    )


def _metadata(record: dict[str, Any], index: int) -> dict[str, Any]:
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
        "component_count": COMPONENT_COUNT,
        "component_semantics": "attention_and_mlp_residual_updates",
        "patch_direction": "donor_to_base_only",
        "position_scope_order": ["all_positions", "differing_lexical_positions"],
        "position_scope_indices": {
            "all_positions": [0, 1, 2, 3, 4, 5],
            "differing_lexical_positions": [0, 1, 2, 4, 5],
        },
        "clean_base_correct_minus_donor_margin": 2.0 + index / 10,
        "clean_donor_correct_minus_base_margin": 3.0 + index / 10,
        "clean_endpoint": 5.0 + index / 5,
        "clean_base_logit_absolute_max": 7.0,
        "clean_donor_logit_absolute_max": 8.0,
        "clean_logit_absolute_max": 8.0,
        "clean_logits_finite": True,
        "device": "cpu",
        "model_dtype": "float64",
    }


class _FakeRunner:
    def __init__(self, *, bad_clean_context: int | None = None) -> None:
        self.bad_clean_context = bad_clean_context
        self.prepare_calls: list[int] = []
        self.response_calls: list[tuple[int, str, int, int | None]] = []

    def prepare_context(self, record: dict[str, Any]) -> SimpleNamespace:
        index = int(record["base_target"].split("-")[-1])
        self.prepare_calls.append(index)
        metadata = _metadata(record, index)
        if index == self.bad_clean_context:
            metadata["clean_base_correct_minus_donor_margin"] = -0.1
        base_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        donor_ids = torch.tensor([[20, 21, 22, 13, 24, 25]])
        base_updates = tuple(
            torch.zeros((6, 2), dtype=torch.float64) for _ in range(COMPONENT_COUNT)
        )
        donor_updates = tuple(
            torch.full((6, 2), float(component + 1), dtype=torch.float64)
            for component in range(COMPONENT_COUNT)
        )
        return SimpleNamespace(
            index=index,
            base_input_ids=base_ids,
            donor_input_ids=donor_ids,
            base_updates=base_updates,
            donor_updates=donor_updates,
            position_scopes={
                "all_positions": torch.ones(6, dtype=torch.bool),
                "differing_lexical_positions": torch.tensor(
                    [True, True, True, False, True, True]
                ),
            },
            clean_metadata=metadata,
        )

    def responses_from_prepared(
        self,
        prepared: SimpleNamespace,
        masks: torch.Tensor,
        *,
        position_scope: str,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        self.response_calls.append(
            (prepared.index, position_scope, len(masks), batch_size)
        )
        scope_offset = 0.25 if position_scope == "differing_lexical_positions" else 0
        return (
            torch.arange(len(masks), dtype=torch.float64) / 1_000
            + prepared.index / 100
            + scope_offset
        )


def test_acquires_both_scopes_in_exact_protocol_order_and_records_norms() -> None:
    runner = _FakeRunner()
    masks = build_condition_masks()

    artifact = acquire_direct_backup_panel_with_runner(
        runner,
        prompt_checks=_checks(),
        condition_masks=masks,
        clean_metadata_contract=_clean_contract(),
        registration_sha256="a" * 64,
        expected_registration_sha256_supplied="a" * 64,
    )

    assert set(artifact) == LABEL_KEYS
    assert artifact["condition_manifest"] == list(build_condition_manifest())
    assert artifact["query_manifest"] == list(build_query_manifest())
    for scope in POSITION_SCOPE_ORDER:
        assert artifact["responses_by_scope"][scope].shape == (
            len(masks),
            CONTEXT_COUNT,
        )
        assert artifact["clean_reference_update_frobenius_norms"][scope].shape == (
            COMPONENT_COUNT,
            CONTEXT_COUNT,
        )
        assert artifact["clean_reference_update_rms_norms"][scope].shape == (
            COMPONENT_COUNT,
            CONTEXT_COUNT,
        )
    assert artifact["query_accounting"] == QUERY_ACCOUNTING
    assert artifact["physical_query_accounting"] == PHYSICAL_QUERY_ACCOUNTING
    assert runner.prepare_calls == list(range(CONTEXT_COUNT))
    # Scientific acquisition is 12 contexts x 2 scopes. Smoke repeats context 0
    # at bs=1 and bs=32 for both scopes, without entering scientific labels.
    assert len(runner.response_calls) == CONTEXT_COUNT * 2 + 4
    assert all(call[2] == len(masks) for call in runner.response_calls)
    assert artifact["batch_invariance_smoke"]["passed"] is True
    assert artifact["batch_invariance_smoke"]["scientific_labels_produced"] is False
    assert artifact["all_integrity_gates_passed"] is True
    assert torch.allclose(
        artifact["clean_reference_update_rms_norms"]["all_positions"][:, 0],
        torch.arange(1, COMPONENT_COUNT + 1, dtype=torch.float64),
    )


def test_all_clean_contexts_are_checked_before_any_patched_response() -> None:
    runner = _FakeRunner(bad_clean_context=4)

    with pytest.raises(RuntimeError, match="clean base correctness"):
        acquire_direct_backup_panel_with_runner(
            runner,
            prompt_checks=_checks(),
            condition_masks=build_condition_masks(),
            clean_metadata_contract=_clean_contract(),
        )

    assert runner.prepare_calls == list(range(CONTEXT_COUNT))
    assert runner.response_calls == []


def test_query_budget_matches_two_scopes_and_full_batch_smoke() -> None:
    masks = build_condition_masks()
    assert len(masks) == 314
    assert len(build_query_manifest()) == 2 * len(masks) == 628
    assert QUERY_ACCOUNTING["patched_context_condition_cells"] == 7_536
    assert QUERY_ACCOUNTING["maximum_logical_queries"] == 7_560
    assert PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"] == 1_256
    assert PHYSICAL_QUERY_ACCOUNTING["maximum_physical_model_examples"] == 8_816


def test_entrypoint_writes_attempt_before_model_and_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}\n", encoding="utf-8")
    expected_sha = acquisition_module.sha256(registration_path)
    environment = {"python": "test"}
    protocol_path = tmp_path / "protocol_conditions.pt"
    protocol_path.write_bytes(b"sealed")
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
        "protocol_artifact": str(protocol_path),
        "protocol_artifact_sha256": "3" * 64,
        "prompt_panel_sha256": "4" * 64,
        "prompt_records_sha256": "9" * 64,
        "posthoc_candidate_provenance_sha256": "a" * 64,
        "prior_mask_audit_sha256": "b" * 64,
        "clean_metadata_contract_sha256": "5" * 64,
        "causal_carrier_protocol_sha256": "6" * 64,
        "condition_manifest_sha256": "7" * 64,
        "query_manifest_sha256": "8" * 64,
        "model_snapshot": str(tmp_path / "snapshot"),
        "component_map": [
            [layer, kind] for layer in range(24) for kind in ("attention", "mlp")
        ],
        "prompt_checks": list(_checks()),
        "clean_metadata_contract": list(_clean_contract()),
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": json.loads(json.dumps(PHYSICAL_QUERY_ACCOUNTING)),
    }
    events: list[str] = []
    monkeypatch.setattr(
        acquisition_module,
        "load_verified_registration",
        lambda *args, **kwargs: events.append("registration") or registration,
    )
    monkeypatch.setattr(
        acquisition_module,
        "validate_registration_payload",
        lambda value, **kwargs: events.append("validation") or value,
    )
    monkeypatch.setattr(
        acquisition_module,
        "load_verified_protocol_artifact",
        lambda *args, **kwargs: events.append("protocol") or build_condition_masks(),
    )
    monkeypatch.setattr(acquisition_module, "runtime_environment", lambda: environment)
    monkeypatch.setattr(
        acquisition_module,
        "build_acquisition_attempt_record",
        lambda **kwargs: {"status": "attempt_consumed_before_model_load"},
    )
    output = tmp_path / "output"

    def create_marker(record: dict[str, Any], path: Path) -> str:
        assert events == ["registration", "validation", "protocol"]
        events.append("attempt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return acquisition_module.sha256(path)

    monkeypatch.setattr(
        acquisition_module, "create_acquisition_attempt_marker", create_marker
    )

    class _TokenizerLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            assert (output / "direct_backup_acquisition_attempt.json").is_file()
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
    components = tuple(
        (int(layer), str(kind)) for layer, kind in registration["component_map"]
    )
    monkeypatch.setattr(
        acquisition_module,
        "Float64ComponentPositionRunner",
        lambda *args, **kwargs: (
            events.append("runner") or SimpleNamespace(components=components)
        ),
    )
    monkeypatch.setattr(
        acquisition_module,
        "acquire_direct_backup_panel_with_runner",
        lambda *args, **kwargs: (
            events.append("outcomes")
            or {"numerical_summary": {"combined_responses": {"absolute_max": 1.25}}}
        ),
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

    result = acquire_direct_backup(
        registration_path=registration_path,
        expected_registration_sha256=expected_sha,
        output_directory=output,
        batch_size=32,
    )

    assert events == [
        "registration",
        "validation",
        "protocol",
        "attempt",
        "tokenizer",
        "model",
        "runner",
        "outcomes",
        "labels",
        "custody",
    ]
    assert set(result) == CUSTODY_KEYS | {"custody_sha256"}
    assert result["condition_masks"] == 314
    assert result["patched_condition_rows_per_context"] == 628
    assert result["maximum_logical_queries"] == 7_560
    assert result["maximum_physical_model_examples"] == 8_816

    with pytest.raises(FileExistsError, match="rerun"):
        acquire_direct_backup(
            registration_path=registration_path,
            expected_registration_sha256=expected_sha,
            output_directory=output,
            batch_size=32,
        )
