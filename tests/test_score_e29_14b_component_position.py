"""Focused tests for fail-closed E29.14b custody-bound scoring."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14b_component_position import (
    CUSTODY_KEYS,
    LABEL_KEYS,
    build_condition_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    PHYSICAL_QUERY_ACCOUNTING,
    PRIOR_PROTOCOL_MASKS_SHA256,
    build_acquisition_attempt_record,
    sha256,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    CONTEXT_COUNT,
    MASK_SHAPES,
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    build_component_position_protocol,
)
from experiments.e29_gpt2_stable_head_interactions.score_e29_14b_component_position import (
    _validate_attempt_record,
    _validate_custody_summary,
    _validate_scoring_environment,
    score_from_custody,
    score_label_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEALED_MASKS = (
    REPOSITORY_ROOT
    / "runs/e29_13_absorbing_hub_confirmation_v1/protocol_masks.pt"
)


def _masks() -> dict[str, torch.Tensor]:
    artifact = torch.load(SEALED_MASKS, map_location="cpu", weights_only=True)
    return {
        split: artifact["protocol"]["held_masks"][split].clone()
        for split in MASK_SPLITS
    }


def _registered_order(
    singleton_all: torch.Tensor,
    singleton_lexical: torch.Tensor,
    all_position: dict[str, torch.Tensor],
    lexical: dict[str, torch.Tensor],
) -> torch.Tensor:
    rows = [singleton_all, singleton_lexical]
    for split in MASK_SPLITS:
        for row_index in range(MASK_SHAPES[split][0]):
            rows.extend(
                (all_position[split][row_index], lexical[split][row_index])
            )
    return torch.stack(rows)


def _smoke() -> dict[str, object]:
    records = {
        scope: {
            "mask_rows": 97,
            "reference_batch_size": 1,
            "candidate_batch_size": 32,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "atol": 1e-9,
            "rtol": 1e-9,
            "passed": True,
        }
        for scope in POSITION_SCOPE_ORDER
    }
    return {
        "records": records,
        "mask_rows": 97,
        "unique_mask_scope_cells": 194,
        "physical_mask_scope_cells": 388,
        "scientific_labels_produced": False,
        "passed": True,
    }


def _labels(registration_sha: str = "a" * 64) -> dict[str, object]:
    masks = _masks()
    base_margins = torch.linspace(
        9.0, 15.0, CONTEXT_COUNT, dtype=torch.float64
    )
    donor_margins = base_margins.clone()
    endpoints = base_margins + donor_margins
    all_position: dict[str, torch.Tensor] = {}
    lexical: dict[str, torch.Tensor] = {}
    for split in MASK_SPLITS:
        normalized = torch.empty(MASK_SHAPES[split][0], CONTEXT_COUNT)
        hub_on = masks[split][:, 1].bool()
        normalized[hub_on] = 0.995
        normalized[~hub_on] = 0.74 if split.endswith("24") else 0.86
        normalized += torch.linspace(-0.01, 0.01, len(normalized))[:, None]
        normalized += torch.linspace(-0.005, 0.005, CONTEXT_COUNT)[None]
        all_position[split] = normalized.double() * endpoints[None]
        lexical[split] = (normalized.double() + 0.002) * endpoints[None]
    singleton_all = 0.995 * endpoints
    singleton_lexical = 0.996 * endpoints
    labels: dict[str, object] = {
        "schema_version": "e29_14b_component_position_labels_v1",
        "registration_sha256": registration_sha,
        "expected_registration_sha256_supplied": registration_sha,
        "condition_manifest": build_condition_manifest(masks),
        "held_masks": masks,
        "singleton_all_response": singleton_all,
        "singleton_lexical_response": singleton_lexical,
        "all_position_responses": all_position,
        "lexical_position_responses": lexical,
        "responses_in_registered_order": _registered_order(
            singleton_all, singleton_lexical, all_position, lexical
        ),
        "clean_metadata": [{} for _ in range(CONTEXT_COUNT)],
        "clean_base_correct_minus_donor_margins": base_margins,
        "clean_donor_correct_minus_base_margins": donor_margins,
        "clean_endpoints": endpoints,
        "batch_invariance_smoke": _smoke(),
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": {
            name: {"passed": True}
            for name in (
                "singleton_all_response",
                "singleton_lexical_response",
                "responses_in_registered_order",
                "combined_responses",
                "clean_endpoints",
            )
        },
        "all_integrity_gates_passed": True,
    }
    assert set(labels) == LABEL_KEYS
    return labels


def test_valid_labels_call_only_registered_scoring_with_integrity_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _labels()
    protocol = build_component_position_protocol(_masks())
    observed: dict[str, Any] = {}

    def fake_score(*args: object, **kwargs: object) -> dict[str, object]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"schema_version": "pure-score", "claims": {}}

    monkeypatch.setattr(
        "experiments.e29_gpt2_stable_head_interactions.score_e29_14b_component_position.score_component_position_bridge",
        fake_score,
    )

    summary = score_label_artifact(
        labels,
        expected_registration_sha256="a" * 64,
        registered_protocol=protocol,
    )

    assert observed["kwargs"] == {"integrity_passed": True}
    assert len(observed["args"]) == 6
    assert summary["all_registered_integrity_gates_passed"] is True
    assert summary["query_accounting"] == QUERY_ACCOUNTING


def test_labels_fail_closed_on_manifest_mask_order_and_endpoint_drift() -> None:
    protocol = build_component_position_protocol(_masks())

    manifest = copy.deepcopy(_labels())
    manifest["condition_manifest"][2]["scope"] = "differing_lexical_positions"
    with pytest.raises(RuntimeError, match="condition manifest"):
        score_label_artifact(
            manifest,
            expected_registration_sha256="a" * 64,
            registered_protocol=protocol,
        )

    masks = copy.deepcopy(_labels())
    masks["held_masks"]["confirmation_size24"][[0, 1]] = masks["held_masks"][
        "confirmation_size24"
    ][[1, 0]]
    with pytest.raises((RuntimeError, ValueError), match="mask|hash"):
        score_label_artifact(
            masks,
            expected_registration_sha256="a" * 64,
            registered_protocol=protocol,
        )

    endpoints = copy.deepcopy(_labels())
    endpoints["clean_endpoints"][0] += 1.0
    with pytest.raises(RuntimeError, match="endpoint arithmetic"):
        score_label_artifact(
            endpoints,
            expected_registration_sha256="a" * 64,
            registered_protocol=protocol,
        )


def test_labels_fail_closed_on_order_accounting_and_integrity_drift() -> None:
    protocol = build_component_position_protocol(_masks())
    ordered = copy.deepcopy(_labels())
    ordered["responses_in_registered_order"][[0, 1]] = ordered[
        "responses_in_registered_order"
    ][[1, 0]]
    with pytest.raises(RuntimeError, match="registered response order"):
        score_label_artifact(
            ordered,
            expected_registration_sha256="a" * 64,
            registered_protocol=protocol,
        )

    accounting = copy.deepcopy(_labels())
    accounting["query_accounting"]["maximum_logical_queries"] += 1
    with pytest.raises(RuntimeError, match="query accounting"):
        score_label_artifact(
            accounting,
            expected_registration_sha256="a" * 64,
            registered_protocol=protocol,
        )

    integrity = _labels()
    integrity["all_integrity_gates_passed"] = False
    with pytest.raises(RuntimeError, match="integrity"):
        score_label_artifact(
            integrity,
            expected_registration_sha256="a" * 64,
            registered_protocol=protocol,
        )

    smoke = copy.deepcopy(_labels())
    smoke["batch_invariance_smoke"]["records"][POSITION_SCOPE_ORDER[0]][
        "candidate_batch_size"
    ] = 16
    with pytest.raises(RuntimeError, match="batch-invariance"):
        score_label_artifact(
            smoke,
            expected_registration_sha256="a" * 64,
            registered_protocol=protocol,
        )


def _registration(path: Path, registration_sha: str) -> dict[str, object]:
    return {
        "environment": {"python": "test"},
        "execution": {"batch_size": 32},
        "source_code_sha256": {"source.py": "1" * 64},
        "checkpoint_sha256": {"model.safetensors": "2" * 64},
        "prior_protocol_masks_sha256": PRIOR_PROTOCOL_MASKS_SHA256,
        "prompt_panel_sha256": "3" * 64,
        "component_position_protocol_sha256": "4" * 64,
        "component_position_protocol": build_component_position_protocol(_masks()),
        "_path": str(path),
        "_sha": registration_sha,
    }


def _custody(
    *,
    output_directory: Path,
    registration_path: Path,
    registration_sha: str,
    labels_sha: str,
    attempt_sha: str,
    response_absolute_max: float = 30.0,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "e29_14b_component_position_acquisition_custody_v1",
        "status": "all_registered_conditions_acquired_after_integrity_gates",
        "registration": str(registration_path.resolve()),
        "registration_sha256": registration_sha,
        "expected_registration_sha256_supplied": registration_sha,
        "source_code_sha256_verified_before_acquisition": {
            "source.py": "1" * 64
        },
        "checkpoint_sha256_verified_before_acquisition": {
            "model.safetensors": "2" * 64
        },
        "prior_protocol_masks_sha256_verified_before_acquisition": (
            PRIOR_PROTOCOL_MASKS_SHA256
        ),
        "acquisition_attempt": str(
            (output_directory / "component_position_acquisition_attempt.json").resolve()
        ),
        "acquisition_attempt_sha256": attempt_sha,
        "component_position_labels": str(
            (output_directory / "component_position_labels.pt").resolve()
        ),
        "component_position_labels_sha256": labels_sha,
        "prompt_panel_sha256": "3" * 64,
        "component_position_protocol_sha256": "4" * 64,
        "contexts": 12,
        "patched_condition_rows_per_context": 194,
        "patched_context_condition_cells": 2328,
        "clean_forward_examples": 24,
        "batch_smoke_unique_condition_rows": 194,
        "batch_smoke_physical_condition_cells": 388,
        "maximum_logical_queries": 2352,
        "maximum_physical_model_examples": 2740,
        "batch_size": 32,
        "batch_invariance_passed": True,
        "all_clean_and_numerical_gates_passed": True,
        "device": "cpu",
        "dtype": "float64",
        "response_absolute_max": response_absolute_max,
        "runtime_seconds": 1.0,
    }
    assert set(value) == CUSTODY_KEYS
    return value


def test_custody_attempt_environment_and_exact_counts_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_path = tmp_path / "registration.json"
    registration_sha = "a" * 64
    registration = _registration(registration_path, registration_sha)
    attempt = build_acquisition_attempt_record(
        registration=registration,
        registration_path=registration_path,
        observed_registration_sha256=registration_sha,
        expected_registration_sha256=registration_sha,
    )
    custody = _custody(
        output_directory=tmp_path,
        registration_path=registration_path,
        registration_sha=registration_sha,
        labels_sha="b" * 64,
        attempt_sha="c" * 64,
    )

    _validate_custody_summary(
        custody, expected_registration_sha256=registration_sha
    )
    _validate_attempt_record(
        attempt,
        registration=registration,
        registration_path=registration_path,
        expected_registration_sha256=registration_sha,
    )
    changed = dict(custody)
    changed["patched_condition_rows_per_context"] = 193
    with pytest.raises(RuntimeError, match="custody contract"):
        _validate_custody_summary(
            changed, expected_registration_sha256=registration_sha
        )

    monkeypatch.setattr(
        "experiments.e29_gpt2_stable_head_interactions.score_e29_14b_component_position.runtime_environment",
        lambda: {"python": "drift"},
    )
    with pytest.raises(RuntimeError, match="scoring environment"):
        _validate_scoring_environment(registration)


def test_full_custody_chain_writes_once_and_requests_external_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    registration_path = tmp_path / "registration.json"
    registration_path.write_text('{"sealed":true}\n', encoding="utf-8")
    registration_sha = sha256(registration_path)
    registration = _registration(registration_path, registration_sha)
    labels_path = output / "component_position_labels.pt"
    labels = _labels(registration_sha)
    torch.save(labels, labels_path)
    attempt = build_acquisition_attempt_record(
        registration=registration,
        registration_path=registration_path,
        observed_registration_sha256=registration_sha,
        expected_registration_sha256=registration_sha,
    )
    attempt_path = output / "component_position_acquisition_attempt.json"
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
    custody = _custody(
        output_directory=output,
        registration_path=registration_path,
        registration_sha=registration_sha,
        labels_sha=sha256(labels_path),
        attempt_sha=sha256(attempt_path),
        response_absolute_max=float(
            labels["responses_in_registered_order"].abs().max()
        ),
    )
    custody_path = output / "component_position_acquisition_custody.json"
    custody_path.write_text(json.dumps(custody), encoding="utf-8")
    observed_validation: dict[str, object] = {}

    def fake_load_registration(
        path: Path, *, expected_sha256: str, required_keys: object
    ) -> dict[str, object]:
        assert path.resolve() == registration_path.resolve()
        assert expected_sha256 == registration_sha
        assert required_keys is not None
        return registration

    def fake_validate_registration(
        value: dict[str, object],
        *,
        source_directory: Path,
        verify_external_files: bool,
    ) -> dict[str, object]:
        observed_validation["source_directory"] = source_directory
        observed_validation["verify_external_files"] = verify_external_files
        return value

    monkeypatch.setattr(
        "experiments.e29_gpt2_stable_head_interactions.score_e29_14b_component_position.load_verified_registration",
        fake_load_registration,
    )
    monkeypatch.setattr(
        "experiments.e29_gpt2_stable_head_interactions.score_e29_14b_component_position.validate_registration_payload",
        fake_validate_registration,
    )
    monkeypatch.setattr(
        "experiments.e29_gpt2_stable_head_interactions.score_e29_14b_component_position.runtime_environment",
        lambda: {"python": "test"},
    )

    result = score_from_custody(
        output_directory=output,
        expected_registration_sha256=registration_sha,
        expected_acquisition_custody_sha256=sha256(custody_path),
    )

    summary_path = output / "component_position_bridge_summary.json"
    assert summary_path.is_file()
    assert result["summary_sha256"] == sha256(summary_path)
    assert observed_validation["verify_external_files"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        score_from_custody(
            output_directory=output,
            expected_registration_sha256=registration_sha,
            expected_acquisition_custody_sha256=sha256(custody_path),
        )
