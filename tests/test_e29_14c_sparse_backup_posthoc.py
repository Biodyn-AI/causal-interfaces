"""Focused tests for the E29.14c exploratory sparse-backup analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.analyze_e29_14c_sparse_backup_posthoc import (
    DEFAULT_OUTPUT,
    DEFAULT_SUMMARY,
    NON_HUB_COMPONENTS,
    TRANSITIONS,
    analyze_score_summary,
    analyze_summary_file,
    centered_pseudoinverse_coefficients,
    transition_incidence,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    CONTEXT_COUNT,
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    build_scale_masks,
)


def _synthetic_summary(component_weights: torch.Tensor) -> dict[str, object]:
    if tuple(component_weights.shape) != (48, CONTEXT_COUNT):
        raise ValueError("synthetic weights must have component/context shape")
    masks = build_scale_masks()
    context_offsets = torch.linspace(-0.02, 0.02, CONTEXT_COUNT)[None]
    geometry = {}
    for split in MASK_SPLITS:
        geometry[split] = {}
        for scope, multiplier in zip(POSITION_SCOPE_ORDER, (1.0, 0.97), strict=True):
            values = (
                0.72 + context_offsets + multiplier * masks[split] @ component_weights
            )
            geometry[split][scope] = {"normalized_values": values.tolist()}
    return {
        "schema_version": "e29_14c_hub_off_scale_scope_ladder_score_v2",
        "scientific_score_schema_version": (
            "e29_14c_hub_off_scale_scope_ladder_score_v1"
        ),
        "integrity_passed": True,
        "scale_response_geometry": geometry,
    }


def _stable_sparse_weights() -> torch.Tensor:
    scale = torch.linspace(0.8, 1.2, CONTEXT_COUNT)
    weights = torch.zeros(48, CONTEXT_COUNT, dtype=torch.float64)
    weights[7] = 0.06 * scale
    weights[33] = 0.035 * scale
    weights[12] = -0.025 * scale
    return weights


def test_centered_pseudoinverse_recovers_relative_additive_weights() -> None:
    incidence = transition_incidence("scale36", "scale40")
    non_hub_weights = _stable_sparse_weights()[list(NON_HUB_COMPONENTS)]
    increments = incidence @ non_hub_weights

    result = centered_pseudoinverse_coefficients(incidence, increments)
    expected = non_hub_weights - non_hub_weights.mean(dim=0, keepdim=True)

    assert result["effective_rank"] == 46
    assert result["maximum_interpolation_error"] < 1e-12
    assert torch.allclose(result["coefficients"], expected, atol=1e-12, rtol=1e-12)


def test_stable_sparse_signal_is_ranked_consistently_across_all_panels() -> None:
    result = analyze_score_summary(
        _synthetic_summary(_stable_sparse_weights()), top_k=5
    )

    assert (
        result["status"] == "exploratory_posthoc_candidate_generation_not_confirmatory"
    )
    assert result["source_score_schema_version"] == (
        "e29_14c_hub_off_scale_scope_ladder_score_v2"
    )
    assert result["source_scientific_score_schema_version"] == (
        "e29_14c_hub_off_scale_scope_ladder_score_v1"
    )
    assert set(result["transition_analyses"]) == {
        f"{current}_to_{following}" for current, following in TRANSITIONS
    }
    assert result["overall_consensus"]["top_candidates"][0]["component"] == 7
    assert result["overall_consensus"]["top_candidates"][1]["component"] == 33
    assert all(
        transition["joint_top_candidates"][0]["component"] == 7
        for transition in result["transition_analyses"].values()
    )
    for transition in result["transition_analyses"].values():
        assert transition["cross_scope_mean_coefficient_correlation"] == pytest.approx(
            1.0
        )
        for scope in POSITION_SCOPE_ORDER:
            stability = transition["scopes"][scope]["leave_one_context_out_row_pattern"]
            assert stability["mean_response_pattern_correlation"] == pytest.approx(1.0)
            assert transition["scopes"][scope][
                "in_sample_interpolation_expected_by_design"
            ]
    assert all(
        record["component"] != 1
        for record in result["overall_consensus"]["component_ranking"]
    )
    assert "not independent replication" in " ".join(result["limitations"])


def test_leave_one_context_out_exposes_a_context_reversing_nomination() -> None:
    weights = torch.zeros(48, CONTEXT_COUNT, dtype=torch.float64)
    weights[7, :6] = 0.06
    weights[7, 6:] = -0.06
    weights[33, :6] = -0.03
    weights[33, 6:] = 0.03

    result = analyze_score_summary(_synthetic_summary(weights), top_k=5)

    for transition in result["transition_analyses"].values():
        for scope in POSITION_SCOPE_ORDER:
            stability = transition["scopes"][scope]["leave_one_context_out_row_pattern"]
            assert stability["mean_response_pattern_correlation"] == pytest.approx(-1.0)
            assert stability["mean_coefficient_pattern_correlation"] == pytest.approx(
                -1.0
            )


def test_summary_validation_and_file_provenance_are_strict(tmp_path: Path) -> None:
    summary = _synthetic_summary(_stable_sparse_weights())
    assert DEFAULT_SUMMARY.parent.name == "e29_14c_hub_off_scale_ladder_v2"
    assert DEFAULT_OUTPUT.parent == DEFAULT_SUMMARY.parent

    bad = dict(summary)
    bad["integrity_passed"] = False
    with pytest.raises(RuntimeError, match="integrity-passing"):
        analyze_score_summary(bad)
    with pytest.raises(ValueError, match="top_k"):
        analyze_score_summary(summary, top_k=47)

    old_wrapper = dict(summary)
    old_wrapper["schema_version"] = "e29_14c_hub_off_scale_scope_ladder_score_v1"
    with pytest.raises(ValueError, match="V2 wrapper"):
        analyze_score_summary(old_wrapper)

    wrong_scientific_schema = dict(summary)
    wrong_scientific_schema["scientific_score_schema_version"] = "changed"
    with pytest.raises(ValueError, match="underlying scientific"):
        analyze_score_summary(wrong_scientific_schema)

    no_scientific_field = dict(summary)
    del no_scientific_field["scientific_score_schema_version"]
    without_optional_field = analyze_score_summary(no_scientific_field)
    assert "source_scientific_score_schema_version" not in without_optional_field

    summary_path = tmp_path / "score.json"
    output_path = tmp_path / "posthoc.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    written = analyze_summary_file(summary_path, output_path, top_k=4)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert written["output_path"] == str(output_path.resolve())
    assert artifact["source_score_summary"] == str(summary_path.resolve())
    assert len(artifact["source_score_summary_sha256"]) == 64
    with pytest.raises(FileExistsError, match="overwrite"):
        analyze_summary_file(summary_path, output_path, top_k=4)
