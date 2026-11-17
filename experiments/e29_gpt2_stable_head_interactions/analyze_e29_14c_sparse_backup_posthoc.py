"""Exploratory sparse-backup nominations from the scored E29.14c ladder.

This is a response-only, post-hoc analysis.  It makes no model calls and does
not alter the sealed E29.14c protocol or its verdict.  Each nested transition
adds a balanced multi-component shell.  We center that shell-incidence matrix
and use its Moore--Penrose pseudoinverse to express mask-row response variation
as *relative component associations*.  Leave-one-context-out prediction tests
whether those associations repeat across contexts.

The resulting component rankings are candidate generators, not identified
individual causal effects.  Multi-component interactions and the changing
background mask are aliased into the coefficients, and the centered design
cannot recover a common effect shared by every component.  Fresh targeted
interventions are required to confirm any nomination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    HUB_COMPONENT,
    MASK_SHAPES,
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    build_scale_masks,
    validate_scale_masks,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_DIRECTORY = REPOSITORY_ROOT / "runs/e29_14c_hub_off_scale_ladder_v2"
DEFAULT_SUMMARY = RUN_DIRECTORY / "scale_scope_summary.json"
DEFAULT_OUTPUT = RUN_DIRECTORY / "sparse_backup_nominations_posthoc.json"
DEFAULT_TOP_K = 8
PINV_RTOL = 1e-12
TRANSITIONS = (
    ("scale36", "scale40"),
    ("scale40", "scale44"),
    ("scale44", "scale47"),
)
NON_HUB_COMPONENTS = tuple(
    index for index in range(COMPONENT_COUNT) if index != HUB_COMPONENT
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_matrix(
    value: Any,
    *,
    shape: tuple[int, int],
    label: str,
) -> torch.Tensor:
    try:
        matrix = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if tuple(matrix.shape) != shape or not bool(matrix.isfinite().all()):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return matrix.detach().cpu().clone()


def normalized_response_tables(
    score_summary: Mapping[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Validate and extract the normalized response matrices from a score."""
    if not isinstance(score_summary, Mapping):
        raise TypeError("E29.14c score summary must be a mapping")
    if (
        score_summary.get("schema_version")
        != "e29_14c_hub_off_scale_scope_ladder_score_v2"
    ):
        raise ValueError("E29.14c V2 wrapper score schema changed")
    scientific_schema = score_summary.get("scientific_score_schema_version")
    if (
        scientific_schema is not None
        and scientific_schema != "e29_14c_hub_off_scale_scope_ladder_score_v1"
    ):
        raise ValueError("E29.14c underlying scientific score schema changed")
    if score_summary.get("integrity_passed") is not True:
        raise RuntimeError("post-hoc analysis requires an integrity-passing score")
    geometry = score_summary.get("scale_response_geometry")
    if not isinstance(geometry, Mapping) or tuple(geometry) != MASK_SPLITS:
        raise ValueError("E29.14c scale response geometry changed")

    tables: dict[str, dict[str, torch.Tensor]] = {
        scope: {} for scope in POSITION_SCOPE_ORDER
    }
    for split in MASK_SPLITS:
        split_geometry = geometry[split]
        if (
            not isinstance(split_geometry, Mapping)
            or tuple(split_geometry) != POSITION_SCOPE_ORDER
        ):
            raise ValueError(f"E29.14c {split} scope geometry changed")
        for scope in POSITION_SCOPE_ORDER:
            scope_geometry = split_geometry[scope]
            if not isinstance(scope_geometry, Mapping):
                raise TypeError(f"E29.14c {split}/{scope} geometry must be a mapping")
            tables[scope][split] = _finite_matrix(
                scope_geometry.get("normalized_values"),
                shape=(MASK_SHAPES[split][0], CONTEXT_COUNT),
                label=f"E29.14c {split}/{scope} normalized values",
            )
    return tables


def transition_incidence(
    current: str,
    following: str,
    *,
    masks: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return one 47-by-47 non-hub added-component incidence matrix."""
    if (current, following) not in TRANSITIONS:
        raise ValueError("unknown E29.14c ladder transition")
    held_masks = validate_scale_masks(build_scale_masks() if masks is None else masks)
    following_panel = held_masks[following]
    if following == "scale47":
        following_panel = following_panel.repeat(47, 1)
    incidence = following_panel - held_masks[current]
    incidence = incidence[:, list(NON_HUB_COMPONENTS)]
    if (
        tuple(incidence.shape) != (47, 47)
        or int(torch.linalg.matrix_rank(incidence)) != 47
    ):
        raise RuntimeError("E29.14c transition incidence lost full rank")
    return incidence


def centered_pseudoinverse_coefficients(
    incidence: torch.Tensor,
    increments: torch.Tensor,
) -> dict[str, torch.Tensor | float | int]:
    """Map row variation to zero-mean relative component associations.

    The centered incidence has rank 46 and spans every zero-mean 47-vector.
    Consequently its in-sample interpolation is guaranteed by design and is
    not evidence for an additive causal model.  Cross-context stability is the
    informative diagnostic returned by the higher-level analysis.
    """
    design = _finite_matrix(
        incidence,
        shape=(47, 47),
        label="transition incidence",
    )
    responses = _finite_matrix(
        increments,
        shape=(47, CONTEXT_COUNT),
        label="transition increments",
    )
    centered_design = design - design.mean(dim=0, keepdim=True)
    centered_responses = responses - responses.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered_design)
    positive = singular_values[singular_values > PINV_RTOL * singular_values.max()]
    if len(positive) != 46:
        raise RuntimeError("centered E29.14c incidence must have rank 46")
    coefficients = (
        torch.linalg.pinv(centered_design, rtol=PINV_RTOL) @ centered_responses
    )
    coefficients -= coefficients.mean(dim=0, keepdim=True)
    fitted = centered_design @ coefficients
    return {
        "coefficients": coefficients,
        "centered_responses": centered_responses,
        "fitted": fitted,
        "effective_rank": len(positive),
        "effective_condition_number": float(positive.max() / positive.min()),
        "maximum_interpolation_error": float((fitted - centered_responses).abs().max()),
    }


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float | None:
    first = first.flatten() - first.mean()
    second = second.flatten() - second.mean()
    denominator = first.norm() * second.norm()
    if denominator <= 1e-15:
        return None
    return float(first @ second / denominator)


def _explained_energy(exact: torch.Tensor, predicted: torch.Tensor) -> float | None:
    energy = exact.square().sum()
    if energy <= 1e-30:
        return None
    return float(1.0 - (exact - predicted).square().sum() / energy)


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return None if not defined else float(sum(defined) / len(defined))


def _percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return ascending ranks in [0, 1], with high coefficients ranked high."""
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(len(values), dtype=torch.float64) / (len(values) - 1)
    return ranks


def _standardized(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean()
    rms = centered.square().mean().sqrt()
    return torch.zeros_like(centered) if rms <= 1e-30 else centered / rms


def _scope_analysis(
    incidence: torch.Tensor,
    increments: torch.Tensor,
    *,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    decomposition = centered_pseudoinverse_coefficients(incidence, increments)
    coefficients = decomposition["coefficients"]
    centered_responses = decomposition["centered_responses"]
    if not isinstance(coefficients, torch.Tensor) or not isinstance(
        centered_responses, torch.Tensor
    ):
        raise TypeError("internal coefficient typing failed")

    context_percentiles = torch.stack(
        [_percentile_ranks(coefficients[:, index]) for index in range(CONTEXT_COUNT)],
        dim=1,
    )
    loco_coefficients = []
    loco_correlations: list[float | None] = []
    loco_explained: list[float | None] = []
    loco_coefficient_correlations: list[float | None] = []
    centered_design = incidence - incidence.mean(dim=0, keepdim=True)
    for held_index in range(CONTEXT_COUNT):
        training_indices = [
            index for index in range(CONTEXT_COUNT) if index != held_index
        ]
        training_coefficient = coefficients[:, training_indices].mean(dim=1)
        held_pattern = centered_responses[:, held_index]
        predicted_pattern = centered_design @ training_coefficient
        loco_coefficients.append(training_coefficient)
        loco_correlations.append(_pearson(held_pattern, predicted_pattern))
        loco_explained.append(_explained_energy(held_pattern, predicted_pattern))
        loco_coefficient_correlations.append(
            _pearson(coefficients[:, held_index], training_coefficient)
        )
    loco = torch.stack(loco_coefficients, dim=1)
    loco_top = torch.zeros_like(loco, dtype=torch.bool)
    for held_index in range(CONTEXT_COUNT):
        selected = torch.topk(loco[:, held_index], top_k).indices
        loco_top[selected, held_index] = True

    mean_coefficients = coefficients.mean(dim=1)
    component_records = []
    for local_index, component in enumerate(NON_HUB_COMPONENTS):
        component_records.append(
            {
                "component": component,
                "mean_relative_coefficient": float(mean_coefficients[local_index]),
                "context_standard_deviation": float(
                    coefficients[local_index].std(unbiased=False)
                ),
                "positive_context_fraction": float(
                    (coefficients[local_index] > 0.0).double().mean()
                ),
                "mean_context_percentile": float(
                    context_percentiles[local_index].mean()
                ),
                "loco_top_k_fraction": float(loco_top[local_index].double().mean()),
                "loco_positive_training_fraction": float(
                    (loco[local_index] > 0.0).double().mean()
                ),
            }
        )
    component_records.sort(
        key=lambda record: (
            record["mean_context_percentile"],
            record["loco_top_k_fraction"],
            record["mean_relative_coefficient"],
        ),
        reverse=True,
    )
    for rank, record in enumerate(component_records, start=1):
        record["rank"] = rank

    summary = {
        "increment_mean": float(increments.mean()),
        "increment_standard_deviation": float(increments.std(unbiased=False)),
        "row_pattern_rms": float(centered_responses.square().mean().sqrt()),
        "centered_incidence_effective_rank": int(decomposition["effective_rank"]),
        "centered_incidence_effective_condition_number": float(
            decomposition["effective_condition_number"]
        ),
        "maximum_in_sample_interpolation_error": float(
            decomposition["maximum_interpolation_error"]
        ),
        "in_sample_interpolation_expected_by_design": True,
        "leave_one_context_out_row_pattern": {
            "informative_contexts": sum(
                value is not None for value in loco_correlations
            ),
            "mean_response_pattern_correlation": _mean_defined(loco_correlations),
            "minimum_response_pattern_correlation": (
                None
                if not any(value is not None for value in loco_correlations)
                else min(value for value in loco_correlations if value is not None)
            ),
            "mean_centered_explained_energy": _mean_defined(loco_explained),
            "mean_coefficient_pattern_correlation": _mean_defined(
                loco_coefficient_correlations
            ),
            "per_context_response_pattern_correlation": loco_correlations,
            "per_context_centered_explained_energy": loco_explained,
            "evaluation_target": (
                "mask_row_pattern_after_removing_each_held_contexts_mean_increment"
            ),
        },
        "component_ranking": component_records,
        "top_candidates": component_records[:top_k],
    }
    internals = {
        "mean_coefficients": mean_coefficients,
        "standardized_mean_coefficients": _standardized(mean_coefficients),
        "context_percentiles": context_percentiles,
        "loco_top": loco_top,
        "positive_context": coefficients > 0.0,
    }
    return summary, internals


def _rank_component_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records.sort(
        key=lambda record: (
            record["mean_context_percentile"],
            record["loco_top_k_fraction"],
            record["mean_standardized_relative_coefficient"],
        ),
        reverse=True,
    )
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def analyze_score_summary(
    score_summary: Mapping[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Generate explicitly exploratory component nominations from E29.14c."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k < 47:
        raise ValueError("top_k must be an integer from 1 through 46")
    tables = normalized_response_tables(score_summary)
    masks = validate_scale_masks(build_scale_masks())
    transition_results: dict[str, Any] = {}
    panel_internals: list[dict[str, torch.Tensor]] = []

    for current, following in TRANSITIONS:
        name = f"{current}_to_{following}"
        incidence = transition_incidence(current, following, masks=masks)
        scopes = {}
        scope_internals = []
        for scope in POSITION_SCOPE_ORDER:
            following_values = tables[scope][following]
            if following == "scale47":
                following_values = following_values.repeat(47, 1)
            increments = following_values - tables[scope][current]
            scope_summary, internals = _scope_analysis(
                incidence,
                increments,
                top_k=top_k,
            )
            scopes[scope] = scope_summary
            scope_internals.append(internals)
            panel_internals.append(internals)

        joint_records = []
        for local_index, component in enumerate(NON_HUB_COMPONENTS):
            joint_records.append(
                {
                    "component": component,
                    "mean_standardized_relative_coefficient": float(
                        torch.stack(
                            [
                                internals["standardized_mean_coefficients"][local_index]
                                for internals in scope_internals
                            ]
                        ).mean()
                    ),
                    "mean_context_percentile": float(
                        torch.cat(
                            [
                                internals["context_percentiles"][local_index]
                                for internals in scope_internals
                            ]
                        ).mean()
                    ),
                    "positive_context_scope_fraction": float(
                        torch.cat(
                            [
                                internals["positive_context"][local_index]
                                for internals in scope_internals
                            ]
                        )
                        .double()
                        .mean()
                    ),
                    "loco_top_k_fraction": float(
                        torch.cat(
                            [
                                internals["loco_top"][local_index]
                                for internals in scope_internals
                            ]
                        )
                        .double()
                        .mean()
                    ),
                }
            )
        joint_records = _rank_component_records(joint_records)
        transition_results[name] = {
            "from_scale": current,
            "to_scale": following,
            "shell_size": int(incidence[0].sum()),
            "scopes": scopes,
            "cross_scope_mean_coefficient_correlation": _pearson(
                scope_internals[0]["mean_coefficients"],
                scope_internals[1]["mean_coefficients"],
            ),
            "joint_component_ranking": joint_records,
            "joint_top_candidates": joint_records[:top_k],
        }

    overall_records = []
    for local_index, component in enumerate(NON_HUB_COMPONENTS):
        panel_percentiles = torch.cat(
            [panel["context_percentiles"][local_index] for panel in panel_internals]
        )
        overall_records.append(
            {
                "component": component,
                "mean_standardized_relative_coefficient": float(
                    torch.stack(
                        [
                            panel["standardized_mean_coefficients"][local_index]
                            for panel in panel_internals
                        ]
                    ).mean()
                ),
                "mean_context_percentile": float(panel_percentiles.mean()),
                "minimum_transition_scope_mean_percentile": float(
                    min(
                        panel["context_percentiles"][local_index].mean()
                        for panel in panel_internals
                    )
                ),
                "positive_context_panel_fraction": float(
                    torch.cat(
                        [
                            panel["positive_context"][local_index]
                            for panel in panel_internals
                        ]
                    )
                    .double()
                    .mean()
                ),
                "positive_panel_mean_fraction": float(
                    torch.stack(
                        [
                            panel["mean_coefficients"][local_index]
                            for panel in panel_internals
                        ]
                    )
                    .gt(0.0)
                    .double()
                    .mean()
                ),
                "loco_top_k_fraction": float(
                    torch.cat(
                        [panel["loco_top"][local_index] for panel in panel_internals]
                    )
                    .double()
                    .mean()
                ),
            }
        )
    overall_records = _rank_component_records(overall_records)

    result = {
        "schema_version": "e29_14c_sparse_backup_nominations_posthoc_v1",
        "status": "exploratory_posthoc_candidate_generation_not_confirmatory",
        "source_score_schema_version": score_summary["schema_version"],
        "source_integrity_passed": True,
        "method": {
            "estimator": "centered_moore_penrose_pseudoinverse",
            "pseudoinverse_relative_tolerance": PINV_RTOL,
            "component_parameterization": (
                "zero_mean_relative_association_among_47_non_hub_components"
            ),
            "context_stability": (
                "leave_one_context_out_prediction_of_centered_mask_row_pattern"
            ),
            "ranking": (
                "descending_mean_within_context_percentile_across_relevant_panels"
            ),
            "top_k": top_k,
            "position_scopes_analyzed_separately": list(POSITION_SCOPE_ORDER),
        },
        "transition_analyses": transition_results,
        "overall_consensus": {
            "panels": len(panel_internals),
            "component_ranking": overall_records,
            "top_candidates": overall_records[:top_k],
        },
        "limitations": [
            (
                "Each row adds three or four components together; nonlinear interactions "
                "and the row-specific background mask are aliased into the coefficients."
            ),
            (
                "Centering removes the transition mean, so the analysis estimates only "
                "relative component associations and cannot allocate a common shell effect."
            ),
            (
                "The centered 47-row design spans every zero-mean row pattern; exact "
                "in-sample reconstruction is automatic and is not model evidence."
            ),
            (
                "Leave-one-context-out stability reuses the same fixed masks and 12-context "
                "cohort; it is not independent replication."
            ),
            (
                "The ranking and top-k shortlist are post-hoc. Candidates require fresh "
                "single-component, dropout, or targeted recomposition interventions."
            ),
        ],
        "recommended_next_test": (
            "freeze a small shortlist before new queries, then compare each candidate's "
            "add-one and leave-one-out effect on fresh contexts at the scale where it ranked"
        ),
    }
    if "scientific_score_schema_version" in score_summary:
        result["source_scientific_score_schema_version"] = score_summary[
            "scientific_score_schema_version"
        ]
    return result


def analyze_summary_file(
    summary_path: Path,
    output_path: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Read one scored summary and atomically write one post-hoc artifact."""
    if output_path.exists():
        raise FileExistsError("refusing to overwrite E29.14c post-hoc nominations")
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("E29.14c score summary file must contain a JSON object")
    result = analyze_score_summary(loaded, top_k=top_k)
    result["source_score_summary"] = str(summary_path.resolve())
    result["source_score_summary_sha256"] = sha256(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return {
        **result,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    arguments = parser.parse_args()
    print(
        json.dumps(
            analyze_summary_file(
                arguments.summary,
                arguments.output,
                top_k=arguments.top_k,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SUMMARY",
    "DEFAULT_TOP_K",
    "NON_HUB_COMPONENTS",
    "TRANSITIONS",
    "analyze_score_summary",
    "analyze_summary_file",
    "centered_pseudoinverse_coefficients",
    "normalized_response_tables",
    "transition_incidence",
]
