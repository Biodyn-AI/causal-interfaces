"""Focused tests for fail-closed E29.14a positional scoring."""

from __future__ import annotations

import copy

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14a_position import (
    LABEL_KEYS,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_custody import (
    PHYSICAL_QUERY_ACCOUNTING,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    CONTEXT_COUNT,
    FORWARD_CONDITIONS,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
)
from experiments.e29_gpt2_stable_head_interactions.score_e29_14a_position import (
    _validate_scoring_environment,
    score_label_artifact,
)


def _condition_index(conditions: tuple[dict[str, object], ...], name: str) -> int:
    return next(index for index, row in enumerate(conditions) if row["name"] == name)


def _labels() -> dict[str, object]:
    endpoints = torch.linspace(10.0, 21.0, CONTEXT_COUNT, dtype=torch.float64)
    forward = torch.zeros(CONTEXT_COUNT, len(FORWARD_CONDITIONS), dtype=torch.float64)
    reverse = torch.zeros(CONTEXT_COUNT, len(REVERSE_CONDITIONS), dtype=torch.float64)
    forward_values = {
        "hub_all_positions_alpha1": 0.995,
        "hub_differing_lexical_alpha1": 0.996,
        "hub_shared_static_alpha1": 0.002,
        "hub_demonstration_target_alpha1": 0.82,
        "hub_demonstration_prefix_alpha1": 0.36,
        "hub_repeated_query_alpha1": 0.54,
        "hub_differing_except_target_alpha1": 0.14,
        "hub_all_positions_alpha025": 0.05,
        "hub_all_positions_alpha050": 0.45,
        "hub_all_positions_alpha075": 0.94,
        "control_all_positions_alpha1": 0.003,
        "control_differing_lexical_alpha1": 0.004,
    }
    reverse_values = {
        "hub_all_positions_alpha1": 0.997,
        "hub_differing_lexical_alpha1": 0.998,
        "hub_shared_static_alpha1": 0.001,
        "hub_demonstration_target_alpha1": 0.84,
    }
    for name, value in forward_values.items():
        forward[:, _condition_index(FORWARD_CONDITIONS, name)] = value
    for name, value in reverse_values.items():
        reverse[:, _condition_index(REVERSE_CONDITIONS, name)] = value
    base_margins = endpoints * 0.4
    donor_margins = endpoints - base_margins
    registration_sha = "a" * 64
    labels: dict[str, object] = {
        "schema_version": "e29_14a_position_labels_v1",
        "registration_sha256": registration_sha,
        "expected_registration_sha256_supplied": registration_sha,
        "condition_manifest": {
            "forward": [dict(row) for row in FORWARD_CONDITIONS],
            "reverse": [dict(row) for row in REVERSE_CONDITIONS],
        },
        "forward_responses": forward * endpoints[:, None],
        "reverse_responses": reverse * endpoints[:, None],
        "clean_metadata": [{} for _ in range(CONTEXT_COUNT)],
        "clean_base_correct_minus_donor_margins": base_margins,
        "clean_donor_correct_minus_base_margins": donor_margins,
        "clean_endpoints": endpoints,
        "batch_invariance_smoke": {"passed": True},
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": {
            name: {"passed": True}
            for name in (
                "forward_responses",
                "reverse_responses",
                "combined_responses",
                "clean_endpoints",
            )
        },
        "all_integrity_gates_passed": True,
    }
    assert set(labels) == LABEL_KEYS
    return labels


def test_scoring_accepts_only_integrity_bound_labels() -> None:
    summary = score_label_artifact(
        _labels(), expected_registration_sha256="a" * 64
    )

    assert summary["all_registered_integrity_gates_passed"] is True
    assert all(claim["passed"] for claim in summary["claims"].values())
    assert summary["recommended_interpretation"].endswith(
        "graded_dose_response"
    )


def test_scoring_rejects_false_integrity_instead_of_scoring_optimistically() -> None:
    labels = _labels()
    labels["all_integrity_gates_passed"] = False

    with pytest.raises(RuntimeError, match="integrity"):
        score_label_artifact(labels, expected_registration_sha256="a" * 64)


def test_scoring_rejects_manifest_or_registration_drift() -> None:
    labels = _labels()
    changed_manifest = copy.deepcopy(labels)
    changed_manifest["condition_manifest"]["forward"][0]["alpha"] = 0.9
    with pytest.raises(RuntimeError, match="condition manifest"):
        score_label_artifact(
            changed_manifest, expected_registration_sha256="a" * 64
        )

    with pytest.raises(RuntimeError, match="registration"):
        score_label_artifact(labels, expected_registration_sha256="b" * 64)


def test_scoring_rejects_runtime_environment_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    sealed_environment = {
        "python": "3.12.9",
        "torch": "2.13.0",
        "transformers": "5.1.0",
        "platform": "sealed-platform",
    }
    monkeypatch.setattr(
        "experiments.e29_gpt2_stable_head_interactions.score_e29_14a_position.runtime_environment",
        lambda: {**sealed_environment, "torch": "2.11.0"},
    )

    with pytest.raises(RuntimeError, match="scoring environment"):
        _validate_scoring_environment({"environment": sealed_environment})
