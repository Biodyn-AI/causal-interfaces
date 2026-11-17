"""Focused tests for fail-closed E29.14c scale/scope scoring."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14c_scale_scope import (
    CLEAN_METADATA_KEYS,
    LABEL_KEYS,
    build_condition_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    tensor_numerical_summary,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    BATCH_INVARIANCE,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    CONTEXT_COUNT,
    MASK_SHAPES,
    MASK_SPLITS,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    build_scale_masks,
    build_scale_scope_protocol,
)
from experiments.e29_gpt2_stable_head_interactions.score_e29_14c_scale_scope import (
    score_label_artifact,
)


def _smoke() -> dict[str, object]:
    mask_rows = 143
    records = {
        scope: {
            "mask_rows": mask_rows,
            "reference_batch_size": BATCH_INVARIANCE["reference_batch_size"],
            "candidate_batch_size": BATCH_INVARIANCE["candidate_batch_size"],
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "atol": BATCH_INVARIANCE["absolute_tolerance"],
            "rtol": BATCH_INVARIANCE["relative_tolerance"],
            "passed": True,
        }
        for scope in POSITION_SCOPE_ORDER
    }
    return {
        "records": records,
        "mask_rows": mask_rows,
        "unique_mask_scope_cells": 286,
        "physical_mask_scope_cells": 572,
        "scientific_labels_produced": False,
        "passed": True,
    }


def _clean_metadata(index: int, endpoint: float) -> dict[str, object]:
    base_margin = endpoint / 2.0
    donor_margin = endpoint - base_margin
    metadata: dict[str, object] = {
        "template": f"template-{index // 3}",
        "base_prompt": f"base prompt {index}",
        "donor_prompt": f"donor prompt {index}",
        "base_target": f"base-{index}",
        "donor_target": f"donor-{index}",
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
        "clean_endpoint": endpoint,
        "clean_base_logit_absolute_max": 7.0,
        "clean_donor_logit_absolute_max": 8.0,
        "clean_logit_absolute_max": 8.0,
        "clean_logits_finite": True,
        "device": "cpu",
        "model_dtype": "float64",
    }
    assert set(metadata) == CLEAN_METADATA_KEYS
    return metadata


def _clean_metadata_contract() -> list[dict[str, object]]:
    records = []
    for index in range(CONTEXT_COUNT):
        metadata = _clean_metadata(index, 1.0)
        records.append(
            {
                "context_index": index,
                "template": metadata["template"],
                "base_prompt": metadata["base_prompt"],
                "donor_prompt": metadata["donor_prompt"],
                "base_target": metadata["base_target"],
                "donor_target": metadata["donor_target"],
                "base_target_token_id": metadata["base_target_token_id"],
                "donor_target_token_id": metadata["donor_target_token_id"],
                "token_length": metadata["token_length"],
                "position_scope_indices": copy.deepcopy(
                    metadata["position_scope_indices"]
                ),
            }
        )
    return records


def _scoring_kwargs() -> dict[str, object]:
    return {
        "expected_registration_sha256": "a" * 64,
        "registered_protocol": build_scale_scope_protocol(),
        "held_masks": build_scale_masks(),
        "clean_metadata_contract": _clean_metadata_contract(),
    }


def _labels(registration_sha: str = "a" * 64) -> dict[str, object]:
    endpoints = torch.linspace(18.0, 30.0, CONTEXT_COUNT, dtype=torch.float64)
    levels = {
        "scale36": 0.76,
        "scale40": 0.86,
        "scale44": 0.94,
        "scale47": 0.99,
    }
    all_position = {}
    lexical = {}
    context_offsets = torch.linspace(-0.004, 0.004, CONTEXT_COUNT)[None]
    for split in MASK_SPLITS:
        row_count = MASK_SHAPES[split][0]
        row_offsets = torch.linspace(-0.006, 0.006, row_count)[:, None]
        normalized = levels[split] + row_offsets + context_offsets
        all_position[split] = normalized.double() * endpoints[None]
        lexical[split] = (normalized.double() + 0.002) * endpoints[None]
    singleton_all = 0.995 * endpoints
    singleton_lexical = 0.996 * endpoints
    combined = torch.cat(
        (
            singleton_all,
            singleton_lexical,
            *(all_position[split].flatten() for split in MASK_SPLITS),
            *(lexical[split].flatten() for split in MASK_SPLITS),
        )
    )
    numerical_summary = {
        "singleton_all_response": tensor_numerical_summary(singleton_all),
        "singleton_lexical_response": tensor_numerical_summary(singleton_lexical),
        "all_position_responses": {
            split: tensor_numerical_summary(all_position[split])
            for split in MASK_SPLITS
        },
        "lexical_position_responses": {
            split: tensor_numerical_summary(lexical[split]) for split in MASK_SPLITS
        },
        "combined_responses": tensor_numerical_summary(combined),
        "clean_endpoints": tensor_numerical_summary(endpoints),
    }
    labels: dict[str, object] = {
        "schema_version": "e29_14c_scale_scope_labels_v2",
        "registration_sha256": registration_sha,
        "expected_registration_sha256_supplied": registration_sha,
        "condition_manifest": build_condition_manifest(),
        "singleton_all_response": singleton_all,
        "singleton_lexical_response": singleton_lexical,
        "all_position_responses": all_position,
        "lexical_position_responses": lexical,
        "clean_metadata": [
            _clean_metadata(index, float(endpoints[index]))
            for index in range(CONTEXT_COUNT)
        ],
        "clean_base_correct_minus_donor_margins": endpoints / 2.0,
        "clean_donor_correct_minus_base_margins": endpoints / 2.0,
        "clean_endpoints": endpoints,
        "batch_invariance_smoke": _smoke(),
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": numerical_summary,
        "all_integrity_gates_passed": True,
    }
    assert set(labels) == LABEL_KEYS
    return labels


def test_valid_labels_call_only_registered_scoring_with_integrity_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_score(*args: object, **kwargs: object) -> dict[str, object]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "schema_version": "e29_14c_hub_off_scale_scope_ladder_score_v1",
            "claims": {},
        }

    monkeypatch.setattr(
        "experiments.e29_gpt2_stable_head_interactions.score_e29_14c_scale_scope.score_scale_scope_ladder",
        fake_score,
    )
    summary = score_label_artifact(
        _labels(),
        **_scoring_kwargs(),
    )

    assert observed["kwargs"] == {"integrity_passed": True}
    assert len(observed["args"]) == 6
    assert summary["schema_version"] == ("e29_14c_hub_off_scale_scope_ladder_score_v2")
    assert summary["scientific_score_schema_version"] == (
        "e29_14c_hub_off_scale_scope_ladder_score_v1"
    )
    assert summary["all_registered_integrity_gates_passed"] is True
    assert summary["query_accounting"] == QUERY_ACCOUNTING


def test_labels_fail_closed_on_manifest_order_shape_and_endpoint_drift() -> None:
    kwargs = _scoring_kwargs()
    manifest = copy.deepcopy(_labels())
    manifest["condition_manifest"][2]["scope"] = "differing_lexical_positions"
    with pytest.raises(RuntimeError, match="condition manifest"):
        score_label_artifact(manifest, **kwargs)

    order = copy.deepcopy(_labels())
    order["all_position_responses"] = {
        split: order["all_position_responses"][split] for split in reversed(MASK_SPLITS)
    }
    with pytest.raises((RuntimeError, ValueError), match="response.*order|keys"):
        score_label_artifact(order, **kwargs)

    shape = copy.deepcopy(_labels())
    shape["lexical_position_responses"]["scale44"] = shape[
        "lexical_position_responses"
    ]["scale44"][:-1]
    with pytest.raises((RuntimeError, ValueError), match="shape"):
        score_label_artifact(shape, **kwargs)

    endpoint = copy.deepcopy(_labels())
    endpoint["clean_endpoints"][0] += 1.0
    with pytest.raises(RuntimeError, match="endpoint arithmetic"):
        score_label_artifact(endpoint, **kwargs)


def test_labels_fail_closed_on_smoke_accounting_numerics_and_metadata() -> None:
    kwargs = _scoring_kwargs()
    smoke = copy.deepcopy(_labels())
    smoke["batch_invariance_smoke"]["records"][POSITION_SCOPE_ORDER[0]][
        "candidate_batch_size"
    ] = 16
    with pytest.raises(RuntimeError, match="batch-invariance"):
        score_label_artifact(smoke, **kwargs)

    accounting = copy.deepcopy(_labels())
    accounting["query_accounting"]["maximum_logical_queries"] += 1
    with pytest.raises(RuntimeError, match="query accounting"):
        score_label_artifact(accounting, **kwargs)

    numerics = copy.deepcopy(_labels())
    numerics["numerical_summary"]["combined_responses"]["passed"] = False
    with pytest.raises(RuntimeError, match="numerical"):
        score_label_artifact(numerics, **kwargs)

    metadata = copy.deepcopy(_labels())
    del metadata["clean_metadata"][0]["patch_direction"]
    with pytest.raises(RuntimeError, match="metadata"):
        score_label_artifact(metadata, **kwargs)

    dtype = copy.deepcopy(_labels())
    dtype["singleton_all_response"] = dtype["singleton_all_response"].float()
    with pytest.raises(RuntimeError, match="device or dtype"):
        score_label_artifact(dtype, **kwargs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("base_prompt", "an unregistered base prompt"),
        ("base_target_token_id", 999_999),
    ),
)
def test_labels_reject_registered_prompt_or_target_identity_mutation(
    field: str, replacement: object
) -> None:
    labels = copy.deepcopy(_labels())
    labels["clean_metadata"][0][field] = replacement

    with pytest.raises(RuntimeError, match="registered clean metadata contract"):
        score_label_artifact(labels, **_scoring_kwargs())


def test_labels_reject_structurally_valid_wrong_lexical_scope() -> None:
    labels = copy.deepcopy(_labels())
    labels["clean_metadata"][0]["position_scope_indices"][
        "differing_lexical_positions"
    ] = [0]

    with pytest.raises(RuntimeError, match="registered clean metadata contract"):
        score_label_artifact(labels, **_scoring_kwargs())


def test_labels_reject_consistently_permuted_context_order() -> None:
    labels = copy.deepcopy(_labels())
    permutation = torch.tensor((1, 0, *range(2, CONTEXT_COUNT)), dtype=torch.int64)
    labels["clean_metadata"] = [
        labels["clean_metadata"][index] for index in permutation.tolist()
    ]
    for key in (
        "clean_base_correct_minus_donor_margins",
        "clean_donor_correct_minus_base_margins",
        "clean_endpoints",
        "singleton_all_response",
        "singleton_lexical_response",
    ):
        labels[key] = labels[key].index_select(0, permutation)
    for key in ("all_position_responses", "lexical_position_responses"):
        labels[key] = {
            split: labels[key][split].index_select(1, permutation)
            for split in MASK_SPLITS
        }

    with pytest.raises(RuntimeError, match="registered clean metadata contract"):
        score_label_artifact(labels, **_scoring_kwargs())
