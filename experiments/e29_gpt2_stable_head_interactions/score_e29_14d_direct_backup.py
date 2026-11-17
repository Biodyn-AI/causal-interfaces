"""Custody-bound pure scoring for the prospective E29.14d carrier panel.

The scientific functions in this module operate only on immutable response
tensors and the response-blind protocol.  They never load or call the model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14d_direct_backup import (
    CLEAN_METADATA_KEYS,
    CUSTODY_KEYS,
    DEFAULT_OUTPUT,
    LABEL_KEYS,
)
from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
    tensor_numerical_summary,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    ADD_ONE_FAMILY_IDS,
    ATTENTION_COMPONENT_ORDER,
    BOOTSTRAP,
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    CORE_ATTENTION_COMPONENTS,
    MLP_COMPONENT_ORDER,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    PRIMARY_POSITION_SCOPE,
    QUERY_ACCOUNTING,
    REGIONAL_FLANKING_ATTENTION_COMPONENTS,
    SCORING_GATES,
    TARGET_COMPONENT_ORDER,
    build_add_one_records,
    build_condition_manifest,
    build_dropout_records,
    build_factorial_records,
    build_query_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_direct_backup_custody import (
    ATTEMPT_REQUIRED_KEYS,
    BATCH_INVARIANCE,
    PROTOCOL_ARTIFACT_SCHEMA,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    build_acquisition_attempt_record,
    load_verified_protocol_artifact,
    load_verified_registration,
    require_exact_keys,
    runtime_environment,
    sha256,
    validate_registration_payload,
    verify_file_hash,
)

SOURCE_DIRECTORY = Path(__file__).resolve().parent


def _finite_float64_tensor(
    value: Any, *, shape: tuple[int, ...], label: str
) -> torch.Tensor:
    """Return an isolated tensor while rejecting silent dtype/device coercion."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise RuntimeError(f"{label} must be CPU float64")
    tensor = value.detach().clone()
    if tuple(tensor.shape) != shape or not bool(tensor.isfinite().all()):
        raise RuntimeError(f"{label} must be finite with shape {shape}")
    return tensor


def _stratified_context_draws(
    *,
    replicates: int,
    strata: Sequence[Sequence[int]],
    generator: torch.Generator,
) -> torch.Tensor:
    """Resample three contexts within each fixed prompt-template stratum."""
    draws = []
    for stratum in strata:
        indices = torch.tensor(tuple(stratum), dtype=torch.long)
        draws.append(
            indices[
                torch.randint(
                    len(indices),
                    (replicates, len(indices)),
                    generator=generator,
                )
            ]
        )
    return torch.cat(draws, dim=1)


def _quantile_summary(estimates: torch.Tensor, point: float) -> dict[str, float]:
    lower, upper = torch.quantile(
        estimates,
        torch.tensor((0.05, 0.95), dtype=torch.float64),
    )
    return {
        "point_estimate": float(point),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _auc(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Mann-Whitney AUC on the final two axes, with half credit for ties."""
    pairwise = positive.unsqueeze(-1) - negative.unsqueeze(-2)
    return (pairwise.gt(0).double() + 0.5 * pairwise.eq(0).double()).mean(dim=(-2, -1))


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first_centered = first.flatten() - first.double().mean()
    second_centered = second.flatten() - second.double().mean()
    denominator = torch.linalg.vector_norm(first_centered) * torch.linalg.vector_norm(
        second_centered
    )
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(first_centered, second_centered) else 0.0
    return float((first_centered * second_centered).sum() / denominator)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for a one-dimensional tensor, including exact ties."""
    if values.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(len(values), dtype=torch.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and bool(sorted_values[stop] == sorted_values[start]):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    return _pearson(_average_ranks(first.flatten()), _average_ranks(second.flatten()))


def _component_bootstrap_estimates(
    effects: torch.Tensor,
    *,
    replicates: int,
    seed: int,
    strata: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Cross common family draws with common stratified context draws.

    ``effects`` has shape component x family x context.  The exact same family
    and context draws are used for every component, preserving paired class and
    ranking comparisons.
    """
    if effects.ndim != 3:
        raise ValueError("add-one effects must be component x family x context")
    generator = torch.Generator().manual_seed(seed)
    family_count = effects.shape[1]
    family_draws = torch.randint(
        family_count,
        (replicates, family_count),
        generator=generator,
    )
    context_draws = _stratified_context_draws(
        replicates=replicates,
        strata=strata,
        generator=generator,
    )
    # Chunking caps transient memory while leaving the registered draws exact.
    estimates = torch.empty((replicates, effects.shape[0]), dtype=torch.float64)
    for start in range(0, replicates, 500):
        stop = min(start + 500, replicates)
        selected = effects[
            :,
            family_draws[start:stop, :, None],
            context_draws[start:stop, None, :],
        ]
        estimates[start:stop] = selected.mean(dim=(-2, -1)).transpose(0, 1)
    return estimates


def _score_add_one_class(
    effects: torch.Tensor,
    *,
    components: Sequence[int],
    attention_components: Sequence[int],
    mlp_components: Sequence[int],
    matched_pairs: Sequence[tuple[int, int]],
    top_components: Sequence[int],
    flanking_components: Sequence[int],
    bootstrap: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    component_order = tuple(int(component) for component in components)
    row = {component: index for index, component in enumerate(component_order)}
    attention_rows = torch.tensor(
        [row[int(component)] for component in attention_components], dtype=torch.long
    )
    mlp_rows = torch.tensor(
        [row[int(component)] for component in mlp_components], dtype=torch.long
    )
    matched_attention_rows = torch.tensor(
        [row[int(attention)] for attention, _ in matched_pairs], dtype=torch.long
    )
    matched_mlp_rows = torch.tensor(
        [row[int(mlp)] for _, mlp in matched_pairs], dtype=torch.long
    )
    top_rows = torch.tensor(
        [row[int(component)] for component in top_components], dtype=torch.long
    )
    flank_rows = torch.tensor(
        [row[int(component)] for component in flanking_components], dtype=torch.long
    )

    point_components = effects.mean(dim=(1, 2))
    estimates = _component_bootstrap_estimates(
        effects,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        strata=bootstrap["template_strata"],
    )
    attention_estimates = estimates[:, attention_rows].mean(dim=1)
    matched_estimates = (
        estimates[:, matched_attention_rows] - estimates[:, matched_mlp_rows]
    ).mean(dim=1)
    auc_estimates = _auc(estimates[:, attention_rows], estimates[:, mlp_rows])
    top_minus_flank_estimates = estimates[:, top_rows].mean(dim=1) - estimates[
        :, flank_rows
    ].mean(dim=1)

    attention_point = float(point_components[attention_rows].mean())
    matched_contrasts = (
        point_components[matched_attention_rows] - point_components[matched_mlp_rows]
    )
    matched_point = float(matched_contrasts.mean())
    auc_point = float(
        _auc(point_components[attention_rows], point_components[mlp_rows])
    )
    top_minus_flank_point = float(
        point_components[top_rows].mean() - point_components[flank_rows].mean()
    )
    attention_summary = _quantile_summary(attention_estimates, attention_point)
    matched_summary = _quantile_summary(matched_estimates, matched_point)
    auc_summary = _quantile_summary(auc_estimates, auc_point)
    top_minus_flank_summary = _quantile_summary(
        top_minus_flank_estimates, top_minus_flank_point
    )
    positive_layers = int((matched_contrasts > 0.0).sum())
    class_passed = bool(
        attention_summary["lower_95"] > float(gates["attention_mean_lower_95_min"])
        and matched_summary["lower_95"]
        > float(gates["matched_attention_minus_mlp_lower_95_min"])
        and auc_summary["lower_95"] > float(gates["auc_lower_95_min"])
        and positive_layers >= int(gates["minimum_positive_layer_contrasts"])
    )
    flank_passed = bool(
        top_minus_flank_summary["lower_95"]
        > float(gates["top_minus_flanking_lower_95_min"])
    )
    return {
        "component_effect_means": {
            str(component): float(point_components[index])
            for index, component in enumerate(component_order)
        },
        "attention_mean": attention_summary,
        "matched_L1_L23_attention_minus_mlp": matched_summary,
        "attention_vs_mlp_auc": auc_summary,
        "positive_matched_layer_contrasts": positive_layers,
        "required_positive_matched_layer_contrasts": int(
            gates["minimum_positive_layer_contrasts"]
        ),
        "class_confirmation_passed": class_passed,
        "top4_minus_flanking_attention": top_minus_flank_summary,
        "top4_vs_flanking_secondary_passed": flank_passed,
    }


def _simultaneous_lower_bounds(
    estimates: torch.Tensor, point: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """One-sided 95% max-error bounds over every cell of ``point``."""
    flattened_point = point.flatten()
    flattened_estimates = estimates.reshape(estimates.shape[0], -1)
    maximum_errors = (flattened_point[None] - flattened_estimates).max(dim=1).values
    critical_value = float(torch.quantile(maximum_errors, 0.95))
    return point - critical_value, critical_value


def _factorial_bootstrap_estimates(
    metrics: torch.Tensor,
    *,
    replicates: int,
    seed: int,
    strata: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Bootstrap q x background x context x metric factorial effects."""
    if metrics.ndim != 4:
        raise ValueError("factorial metrics must be q x background x context x metric")
    generator = torch.Generator().manual_seed(seed)
    background_count = metrics.shape[1]
    background_draws = torch.randint(
        background_count,
        (replicates, background_count),
        generator=generator,
    )
    context_draws = _stratified_context_draws(
        replicates=replicates,
        strata=strata,
        generator=generator,
    )
    estimates = torch.empty(
        (replicates, metrics.shape[0], metrics.shape[-1]), dtype=torch.float64
    )
    for start in range(0, replicates, 500):
        stop = min(start + 500, replicates)
        selected = metrics[
            :,
            background_draws[start:stop, :, None],
            context_draws[start:stop, None, :],
            :,
        ]
        estimates[start:stop] = selected.mean(dim=(-3, -2)).permute(1, 0, 2)
    return estimates


def _score_factorial(
    cells: torch.Tensor,
    *,
    top_components: Sequence[int],
    bootstrap: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Score B, B+A, B+M, B+A+M cells for each q and background."""
    if cells.ndim != 4 or cells.shape[-1] != 4:
        raise ValueError(
            "factorial cells must be q x background x context x four cells"
        )
    metrics = _factorial_metrics(cells)
    point = metrics.mean(dim=(1, 2))
    estimates = _factorial_bootstrap_estimates(
        metrics,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        strata=bootstrap["template_strata"],
    )
    lower, critical = _simultaneous_lower_bounds(estimates, point)
    metric_names = ("A0", "A1", "average_A_minus_M_marginal")
    by_component = {}
    for q_index, component in enumerate(top_components):
        by_component[str(component)] = {
            metric: {
                "point_estimate": float(point[q_index, metric_index]),
                "simultaneous_lower_95": float(lower[q_index, metric_index]),
            }
            for metric_index, metric in enumerate(metric_names)
        }
    return {
        "by_attention_component": by_component,
        "simultaneous_max_error_critical_value": critical,
        "familywise_claim_count": int(point.numel()),
        "threshold": float(threshold),
        "all_simultaneous_lower_bounds_passed": bool((lower > threshold).all()),
    }


def _factorial_metrics(cells: torch.Tensor) -> torch.Tensor:
    """Return A0, A1, and the average A-minus-M marginal."""
    base, add_attention, add_mlp, add_both = cells.unbind(dim=-1)
    a0 = add_attention - base
    a1 = add_both - add_mlp
    m0 = add_mlp - base
    m1 = add_both - add_attention
    average_a_minus_m = 0.5 * (a0 + a1 - m0 - m1)
    return torch.stack((a0, a1, average_a_minus_m), dim=-1)


def _context_bootstrap_component_estimates(
    effects: torch.Tensor,
    *,
    replicates: int,
    seed: int,
    strata: Sequence[Sequence[int]],
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    contexts = _stratified_context_draws(
        replicates=replicates,
        strata=strata,
        generator=generator,
    )
    return effects[:, contexts].mean(dim=-1).transpose(0, 1)


def _score_dropout(
    effects: torch.Tensor,
    *,
    components: Sequence[int],
    attention_components: Sequence[int],
    mlp_components: Sequence[int],
    matched_pairs: Sequence[tuple[int, int]],
    top_components: Sequence[int],
    bootstrap: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    component_order = tuple(int(component) for component in components)
    row = {component: index for index, component in enumerate(component_order)}
    attention_rows = torch.tensor(
        [row[int(component)] for component in attention_components], dtype=torch.long
    )
    matched_attention_rows = torch.tensor(
        [row[int(attention)] for attention, _ in matched_pairs], dtype=torch.long
    )
    matched_mlp_rows = torch.tensor(
        [row[int(mlp)] for _, mlp in matched_pairs], dtype=torch.long
    )
    estimates = _context_bootstrap_component_estimates(
        effects,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        strata=bootstrap["template_strata"],
    )
    point_components = effects.mean(dim=1)
    attention_estimates = estimates[:, attention_rows].mean(dim=1)
    matched_estimates = (
        estimates[:, matched_attention_rows] - estimates[:, matched_mlp_rows]
    ).mean(dim=1)
    attention_summary = _quantile_summary(
        attention_estimates, float(point_components[attention_rows].mean())
    )
    matched_summary = _quantile_summary(
        matched_estimates,
        float(
            (
                point_components[matched_attention_rows]
                - point_components[matched_mlp_rows]
            ).mean()
        ),
    )
    class_passed = bool(
        attention_summary["lower_95"] > float(gates["attention_mean_lower_95_min"])
        and matched_summary["lower_95"]
        > float(gates["matched_attention_minus_mlp_lower_95_min"])
    )

    individual_points = []
    individual_estimates = []
    individual_names = []
    for component in top_components:
        attention_row = row[int(component)]
        mlp_row = row[int(component) + 1]
        individual_names.extend(
            (f"d_{component}", f"d_{component}_minus_d_{component + 1}")
        )
        individual_points.extend(
            (
                point_components[attention_row],
                point_components[attention_row] - point_components[mlp_row],
            )
        )
        individual_estimates.extend(
            (
                estimates[:, attention_row],
                estimates[:, attention_row] - estimates[:, mlp_row],
            )
        )
    individual_point = torch.stack(individual_points)
    individual_bootstrap = torch.stack(individual_estimates, dim=1)
    individual_lower, critical = _simultaneous_lower_bounds(
        individual_bootstrap, individual_point
    )
    individual = {
        name: {
            "point_estimate": float(individual_point[index]),
            "simultaneous_lower_95": float(individual_lower[index]),
        }
        for index, name in enumerate(individual_names)
    }
    individual_threshold = float(gates["individual_top4_simultaneous_lower_95_min"])
    matched_threshold = float(gates["matched_top4_simultaneous_lower_95_min"])
    thresholds = torch.tensor(
        [
            individual_threshold if index % 2 == 0 else matched_threshold
            for index in range(len(individual_names))
        ],
        dtype=torch.float64,
    )
    return {
        "component_effect_means": {
            str(component): float(point_components[index])
            for index, component in enumerate(component_order)
        },
        "attention_mean": attention_summary,
        "matched_L1_L23_attention_minus_mlp": matched_summary,
        "class_necessity_passed": class_passed,
        "top4_individual_and_matched_contrasts": individual,
        "top4_simultaneous_max_error_critical_value": critical,
        "top4_individual_loss_threshold": individual_threshold,
        "top4_attention_minus_matched_mlp_threshold": matched_threshold,
        "top4_individual_necessity_passed": bool((individual_lower > thresholds).all()),
    }


def _extract_add_one_effects(
    normalized_responses: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    components: Sequence[int],
    family_count: int,
) -> torch.Tensor:
    """Reconstruct the 47 x 3 x context paired add-one estimands."""
    row = {int(component): index for index, component in enumerate(components)}
    effects = torch.full(
        (len(components), family_count, normalized_responses.shape[1]),
        torch.nan,
        dtype=torch.float64,
    )
    observed: set[tuple[int, int]] = set()
    required_keys = {
        "pair_id",
        "target_component",
        "target_class",
        "family_id",
        "family_index",
        "base_condition_index",
        "plus_condition_index",
        "base_condition_id",
        "plus_condition_id",
    }
    for record in records:
        if set(record) != required_keys:
            raise RuntimeError("E29.14d add-one record keys changed")
        component = int(record["target_component"])
        family = int(record["family_index"])
        key = (component, family)
        if (
            component not in row
            or family < 0
            or family >= family_count
            or key in observed
        ):
            raise RuntimeError("E29.14d add-one record identity changed")
        base = int(record["base_condition_index"])
        plus = int(record["plus_condition_index"])
        if min(base, plus) < 0 or max(base, plus) >= normalized_responses.shape[0]:
            raise RuntimeError("E29.14d add-one condition index changed")
        effects[row[component], family] = (
            normalized_responses[plus] - normalized_responses[base]
        )
        observed.add(key)
    expected = {
        (int(component), family)
        for component in components
        for family in range(family_count)
    }
    if observed != expected or not bool(effects.isfinite().all()):
        raise RuntimeError("E29.14d add-one record coverage changed")
    return effects


def _extract_factorial_cells(
    normalized_responses: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    top_components: Sequence[int],
    background_count: int,
) -> torch.Tensor:
    """Reconstruct q x background x context x (B, BA, BM, BAM)."""
    row = {int(component): index for index, component in enumerate(top_components)}
    cells = torch.full(
        (
            len(top_components),
            background_count,
            normalized_responses.shape[1],
            4,
        ),
        torch.nan,
        dtype=torch.float64,
    )
    required_keys = {
        "factorial_id",
        "attention_component",
        "mlp_component",
        "layer",
        "background_index",
        "family_id",
        "condition_indices",
        "condition_ids",
    }
    observed: set[tuple[int, int]] = set()
    for record in records:
        if set(record) != required_keys:
            raise RuntimeError("E29.14d factorial record keys changed")
        attention = int(record["attention_component"])
        mlp = int(record["mlp_component"])
        background = int(record["background_index"])
        key = (attention, background)
        if (
            attention not in row
            or mlp != attention + 1
            or background < 0
            or background >= background_count
            or key in observed
        ):
            raise RuntimeError("E29.14d factorial record identity changed")
        condition_indices = record["condition_indices"]
        if not isinstance(condition_indices, Mapping) or tuple(condition_indices) != (
            "neither",
            "attention_only",
            "mlp_only",
            "both",
        ):
            raise RuntimeError("E29.14d factorial cell mapping changed")
        indices = tuple(int(condition_indices[name]) for name in condition_indices)
        if min(indices) < 0 or max(indices) >= normalized_responses.shape[0]:
            raise RuntimeError("E29.14d factorial condition index changed")
        cells[row[attention], background] = normalized_responses[
            torch.tensor(indices, dtype=torch.long)
        ].transpose(0, 1)
        observed.add(key)
    expected = {
        (int(component), background)
        for component in top_components
        for background in range(background_count)
    }
    if observed != expected or not bool(cells.isfinite().all()):
        raise RuntimeError("E29.14d factorial record coverage changed")
    return cells


def _extract_dropout_effects(
    normalized_responses: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    components: Sequence[int],
) -> torch.Tensor:
    """Reconstruct full-c47 minus leave-one-component-out effects."""
    row = {int(component): index for index, component in enumerate(components)}
    effects = torch.full(
        (len(components), normalized_responses.shape[1]),
        torch.nan,
        dtype=torch.float64,
    )
    required_keys = {
        "dropout_id",
        "dropped_component",
        "dropped_class",
        "full_condition_index",
        "dropout_condition_index",
        "full_condition_id",
        "dropout_condition_id",
    }
    observed: set[int] = set()
    full_indices: set[int] = set()
    for record in records:
        if set(record) != required_keys:
            raise RuntimeError("E29.14d dropout record keys changed")
        component = int(record["dropped_component"])
        full = int(record["full_condition_index"])
        dropout = int(record["dropout_condition_index"])
        if (
            component not in row
            or component in observed
            or min(full, dropout) < 0
            or max(full, dropout) >= normalized_responses.shape[0]
        ):
            raise RuntimeError("E29.14d dropout record identity changed")
        effects[row[component]] = (
            normalized_responses[full] - normalized_responses[dropout]
        )
        observed.add(component)
        full_indices.add(full)
    if observed != set(components) or len(full_indices) != 1:
        raise RuntimeError("E29.14d dropout record coverage changed")
    return effects


def _effect_concordance(
    all_position: torch.Tensor, lexical: torch.Tensor
) -> dict[str, Any]:
    gap = lexical - all_position
    return {
        "pearson": _pearson(all_position, lexical),
        "spearman": _spearman(all_position, lexical),
        "mean_absolute_gap": float(gap.abs().mean()),
        "maximum_absolute_gap": float(gap.abs().max()),
        "same_sign_fraction": float(
            ((all_position > 0.0) == (lexical > 0.0)).double().mean()
        ),
    }


def _norm_summary(
    norms: torch.Tensor,
    *,
    component_effects: Mapping[str, float],
    components: Sequence[int],
    attention_components: Sequence[int],
    mlp_components: Sequence[int],
    matched_pairs: Sequence[tuple[int, int]],
    top_components: Sequence[int],
    flanking_components: Sequence[int],
) -> dict[str, Any]:
    component_means = norms.mean(dim=1)
    effects = torch.tensor(
        [float(component_effects[str(component)]) for component in components],
        dtype=torch.float64,
    )
    nonhub_norms = component_means[torch.tensor(tuple(components), dtype=torch.long)]
    attention_rows = torch.tensor(tuple(attention_components), dtype=torch.long)
    mlp_rows = torch.tensor(tuple(mlp_components), dtype=torch.long)
    matched_attention = torch.tensor(
        [attention for attention, _ in matched_pairs], dtype=torch.long
    )
    matched_mlp = torch.tensor([mlp for _, mlp in matched_pairs], dtype=torch.long)
    top_rows = torch.tensor(tuple(top_components), dtype=torch.long)
    flank_rows = torch.tensor(tuple(flanking_components), dtype=torch.long)
    return {
        "minimum": float(norms.min()),
        "maximum": float(norms.max()),
        "mean": float(norms.mean()),
        "component_means": {
            str(component): float(component_means[component])
            for component in range(len(component_means))
        },
        "attention_mean": float(component_means[attention_rows].mean()),
        "nonhub_mlp_mean": float(component_means[mlp_rows].mean()),
        "matched_L1_L23_attention_minus_mlp": float(
            (component_means[matched_attention] - component_means[matched_mlp]).mean()
        ),
        "top4_minus_flanking_attention": float(
            component_means[top_rows].mean() - component_means[flank_rows].mean()
        ),
        "add_one_effect_vs_norm_pearson": _pearson(effects, nonhub_norms),
        "add_one_effect_vs_norm_spearman": _spearman(effects, nonhub_norms),
        "warning": (
            "clean reference donor-minus-base update norms are descriptive; "
            "they are not actual multi-component injected norms and do not "
            "enter any gate"
        ),
    }


def _interpretation_branch(
    *,
    class_passed: bool,
    factorial_passed: bool,
    dropout_class_passed: bool,
    dropout_top4_passed: bool,
) -> str:
    """Name add-one evidence first, then append independent dropout evidence."""
    if class_passed and factorial_passed:
        add_one = "broad_attention_and_late_top4_causal_add_one_confirmed"
    elif class_passed:
        add_one = "broad_attention_backup_without_specific_late_top4_confirmation"
    elif factorial_passed:
        add_one = "localized_late_top4_effect_without_broad_attention_class_support"
    else:
        add_one = "no_prospective_add_one_attention_carrier_confirmation"

    if dropout_class_passed and dropout_top4_passed:
        dropout = "c47_class_and_top4_necessity_supported"
    elif dropout_class_passed:
        dropout = "c47_class_necessity_supported_but_top4_necessity_not_supported"
    elif dropout_top4_passed:
        dropout = "c47_top4_necessity_supported_but_class_necessity_not_supported"
    else:
        dropout = "c47_dropout_necessity_not_supported"
    return f"{add_one}__{dropout}"


def _context_bootstrap_fixed_panel_mean(
    values: torch.Tensor,
    *,
    replicates: int,
    seed: int,
    strata: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Resample contexts while keeping every prespecified panel row fixed."""
    generator = torch.Generator().manual_seed(seed)
    contexts = _stratified_context_draws(
        replicates=replicates, strata=strata, generator=generator
    )
    flattened = values.reshape(-1, values.shape[-1])
    return flattened[:, contexts].mean(dim=(0, 2))


def _score_endpoint_anchor(
    normalized: torch.Tensor,
    *,
    mean_min: float,
    mean_max: float,
    gap_upper_max: float,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    strata = bootstrap["template_strata"]
    mean = float(normalized.mean())
    gap = (normalized - 1.0).abs()
    gap_estimates = _context_bootstrap_fixed_panel_mean(
        gap, replicates=replicates, seed=seed, strata=strata
    )
    gap_summary = _quantile_summary(gap_estimates, float(gap.mean()))
    return {
        "normalized_mean": mean,
        "normalized_mean_min": float(mean_min),
        "normalized_mean_max": float(mean_max),
        "absolute_endpoint_gap": gap_summary,
        "absolute_endpoint_gap_upper_95_max": float(gap_upper_max),
        "passed": bool(
            mean_min <= mean <= mean_max and gap_summary["upper_95"] <= gap_upper_max
        ),
    }


def _score_scope_secondary(
    normalized_by_scope: Mapping[str, torch.Tensor],
    add_effects: Mapping[str, torch.Tensor],
    factorial_cells: Mapping[str, torch.Tensor],
    dropout_effects: Mapping[str, torch.Tensor],
    *,
    scopes: Sequence[str],
    bootstrap: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    first, second = scopes
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    strata = bootstrap["template_strata"]

    response_gap = (normalized_by_scope[first] - normalized_by_scope[second]).abs()
    response_estimates = _context_bootstrap_fixed_panel_mean(
        response_gap, replicates=replicates, seed=seed, strata=strata
    )
    response_summary = _quantile_summary(response_estimates, float(response_gap.mean()))

    add_gap = (add_effects[first] - add_effects[second]).abs()
    add_estimates = _component_bootstrap_estimates(
        add_gap,
        replicates=replicates,
        seed=seed,
        strata=strata,
    ).mean(dim=1)
    add_summary = _quantile_summary(add_estimates, float(add_gap.mean()))

    factorial_gap = (
        _factorial_metrics(factorial_cells[first])
        - _factorial_metrics(factorial_cells[second])
    ).abs()
    factorial_estimates = _factorial_bootstrap_estimates(
        factorial_gap,
        replicates=replicates,
        seed=seed,
        strata=strata,
    ).mean(dim=(1, 2))
    factorial_summary = _quantile_summary(
        factorial_estimates, float(factorial_gap.mean())
    )

    dropout_gap = (dropout_effects[first] - dropout_effects[second]).abs()
    dropout_estimates = _context_bootstrap_component_estimates(
        dropout_gap,
        replicates=replicates,
        seed=seed,
        strata=strata,
    ).mean(dim=1)
    dropout_summary = _quantile_summary(dropout_estimates, float(dropout_gap.mean()))
    context_mae = response_gap.mean(dim=0)
    context_threshold = float(gates["context_mean_absolute_gap_max"])
    passing_contexts = int((context_mae <= context_threshold).sum())
    passed = bool(
        response_summary["upper_95"]
        <= float(gates["condition_response_mean_absolute_gap_bootstrap_upper_max"])
        and add_summary["upper_95"]
        <= float(gates["add_one_effect_mean_absolute_gap_bootstrap_upper_max"])
        and factorial_summary["upper_95"]
        <= float(gates["factorial_contrast_mean_absolute_gap_bootstrap_upper_max"])
        and dropout_summary["upper_95"]
        <= float(gates["dropout_effect_mean_absolute_gap_bootstrap_upper_max"])
        and passing_contexts >= int(gates["minimum_passing_contexts"])
    )
    return {
        "condition_response_mean_absolute_gap": response_summary,
        "add_one_effect_mean_absolute_gap": add_summary,
        "factorial_contrast_mean_absolute_gap": factorial_summary,
        "dropout_effect_mean_absolute_gap": dropout_summary,
        "context_mean_absolute_gaps": [float(value) for value in context_mae],
        "passing_contexts": passing_contexts,
        "minimum_passing_contexts": int(gates["minimum_passing_contexts"]),
        "passed": passed,
        "conjunctive_with_primary_causal_class_claim": False,
    }


def score_direct_backup_response_tensors(
    responses_by_scope: Mapping[str, torch.Tensor],
    clean_endpoints: torch.Tensor,
    clean_reference_update_frobenius_norms: Mapping[str, torch.Tensor],
    clean_reference_update_rms_norms: Mapping[str, torch.Tensor],
    *,
    condition_count: int,
    context_count: int,
    position_scopes: Sequence[str],
    primary_scope: str,
    components: Sequence[int],
    attention_components: Sequence[int],
    mlp_components: Sequence[int],
    matched_pairs: Sequence[tuple[int, int]],
    top_components: Sequence[int],
    flanking_components: Sequence[int],
    add_one_records: Sequence[Mapping[str, Any]],
    factorial_records: Sequence[Mapping[str, Any]],
    dropout_records: Sequence[Mapping[str, Any]],
    family_count: int,
    factorial_background_count: int,
    hub_singleton_condition_index: int,
    full_condition_index: int,
    bootstrap: Mapping[str, Any],
    scoring_gates: Mapping[str, Any],
    integrity_passed: bool,
) -> dict[str, Any]:
    """Pure registered estimator and gate implementation, with no file I/O."""
    if integrity_passed is not True:
        raise RuntimeError("E29.14d scientific scoring requires passed integrity")
    scopes = tuple(str(scope) for scope in position_scopes)
    if (
        not isinstance(responses_by_scope, Mapping)
        or tuple(responses_by_scope) != scopes
    ):
        raise RuntimeError("E29.14d response scope keys or order changed")
    if primary_scope != scopes[0]:
        raise RuntimeError("E29.14d primary scope must remain first")
    endpoints = _finite_float64_tensor(
        clean_endpoints, shape=(context_count,), label="clean endpoints"
    )
    if bool((endpoints <= 0.0).any()):
        raise RuntimeError("E29.14d clean endpoints must be positive")
    if not isinstance(bootstrap, Mapping) or (
        int(bootstrap["seed"]) != 29_144 or int(bootstrap["replicates"]) != 10_000
    ):
        raise RuntimeError("E29.14d bootstrap seed or replicate count changed")

    normalized_by_scope = {}
    add_effects = {}
    factorial_cells = {}
    dropout_effects = {}
    analyses = {}
    for scope in scopes:
        responses = _finite_float64_tensor(
            responses_by_scope[scope],
            shape=(condition_count, context_count),
            label=f"{scope} responses",
        )
        normalized = responses / endpoints[None]
        normalized_by_scope[scope] = normalized
        add_effects[scope] = _extract_add_one_effects(
            normalized,
            add_one_records,
            components=components,
            family_count=family_count,
        )
        factorial_cells[scope] = _extract_factorial_cells(
            normalized,
            factorial_records,
            top_components=top_components,
            background_count=factorial_background_count,
        )
        dropout_effects[scope] = _extract_dropout_effects(
            normalized,
            dropout_records,
            components=components,
        )
        analyses[scope] = {
            "add_one": _score_add_one_class(
                add_effects[scope],
                components=components,
                attention_components=attention_components,
                mlp_components=mlp_components,
                matched_pairs=matched_pairs,
                top_components=top_components,
                flanking_components=flanking_components,
                bootstrap=bootstrap,
                gates={
                    "attention_mean_lower_95_min": scoring_gates["balanced_add_one"][
                        "attention_class_mean_effect_bootstrap_lower_strict_min"
                    ],
                    "matched_attention_minus_mlp_lower_95_min": scoring_gates[
                        "balanced_add_one"
                    ]["matched_layer_attention_minus_mlp_bootstrap_lower_strict_min"],
                    "auc_lower_95_min": scoring_gates["balanced_add_one"][
                        "attention_vs_mlp_component_auc_bootstrap_lower_strict_min"
                    ],
                    "minimum_positive_layer_contrasts": scoring_gates[
                        "balanced_add_one"
                    ]["minimum_positive_matched_layer_contrasts"],
                    "top_minus_flanking_lower_95_min": scoring_gates[
                        "regional_flanking_secondary"
                    ]["late_minus_regional_flanking_bootstrap_lower_strict_min"],
                },
            ),
            "factorial": _score_factorial(
                factorial_cells[scope],
                top_components=top_components,
                bootstrap=bootstrap,
                threshold=float(
                    scoring_gates["top4_factorial"][
                        "simultaneous_lower_band_strict_min"
                    ]
                ),
            ),
            "dropout": _score_dropout(
                dropout_effects[scope],
                components=components,
                attention_components=attention_components,
                mlp_components=mlp_components,
                matched_pairs=matched_pairs,
                top_components=top_components,
                bootstrap=bootstrap,
                gates={
                    "attention_mean_lower_95_min": scoring_gates[
                        "single_dropout_secondary"
                    ]["attention_class_mean_loss_bootstrap_lower_strict_min"],
                    "matched_attention_minus_mlp_lower_95_min": scoring_gates[
                        "single_dropout_secondary"
                    ]["matched_layer_attention_minus_mlp_loss_lower_strict_min"],
                    "individual_top4_simultaneous_lower_95_min": scoring_gates[
                        "single_dropout_secondary"
                    ]["top4_individual_loss_simultaneous_lower_strict_min"],
                    "matched_top4_simultaneous_lower_95_min": scoring_gates[
                        "single_dropout_secondary"
                    ][
                        "top4_attention_minus_matched_mlp_loss_simultaneous_lower_strict_min"
                    ],
                },
            ),
        }

    if (
        min(hub_singleton_condition_index, full_condition_index) < 0
        or max(hub_singleton_condition_index, full_condition_index) >= condition_count
    ):
        raise RuntimeError("E29.14d anchor condition index changed")
    anchor_gates = scoring_gates["integrity_anchors"]
    anchors = {
        "hub_singleton": _score_endpoint_anchor(
            normalized_by_scope[primary_scope][hub_singleton_condition_index],
            mean_min=float(anchor_gates["hub_singleton_normalized_mean_min"]),
            mean_max=float(anchor_gates["hub_singleton_normalized_mean_max"]),
            gap_upper_max=float(
                anchor_gates["hub_singleton_absolute_endpoint_gap_bootstrap_upper_max"]
            ),
            bootstrap=bootstrap,
        ),
        "full_nonhub_c47": _score_endpoint_anchor(
            normalized_by_scope[primary_scope][full_condition_index],
            mean_min=float(anchor_gates["c47_full_normalized_mean_min"]),
            mean_max=float(anchor_gates["c47_full_normalized_mean_max"]),
            gap_upper_max=float(
                anchor_gates["c47_full_absolute_endpoint_gap_bootstrap_upper_max"]
            ),
            bootstrap=bootstrap,
        ),
    }
    anchors["passed"] = bool(
        anchors["hub_singleton"]["passed"] and anchors["full_nonhub_c47"]["passed"]
    )

    primary = analyses[primary_scope]
    class_passed = bool(primary["add_one"]["class_confirmation_passed"])
    factorial_passed = bool(
        primary["factorial"]["all_simultaneous_lower_bounds_passed"]
    )
    dropout_class_passed = bool(primary["dropout"]["class_necessity_passed"])
    dropout_top4_passed = bool(primary["dropout"]["top4_individual_necessity_passed"])
    causal_add_one_program_passed = bool(
        anchors["passed"] and class_passed and factorial_passed
    )
    interpretation = _interpretation_branch(
        class_passed=class_passed,
        factorial_passed=factorial_passed,
        dropout_class_passed=dropout_class_passed,
        dropout_top4_passed=dropout_top4_passed,
    )
    if not anchors["passed"]:
        interpretation = (
            "integrity_anchors_failed_scientific_claims_not_authorized__"
            f"{interpretation}"
        )

    concordance = {
        "normalized_responses": _effect_concordance(
            normalized_by_scope[scopes[0]], normalized_by_scope[scopes[1]]
        ),
        "add_one_effect_cells": _effect_concordance(
            add_effects[scopes[0]], add_effects[scopes[1]]
        ),
        "add_one_component_means": _effect_concordance(
            add_effects[scopes[0]].mean(dim=(1, 2)),
            add_effects[scopes[1]].mean(dim=(1, 2)),
        ),
        "factorial_cells": _effect_concordance(
            factorial_cells[scopes[0]], factorial_cells[scopes[1]]
        ),
        "dropout_effect_cells": _effect_concordance(
            dropout_effects[scopes[0]], dropout_effects[scopes[1]]
        ),
    }
    scope_secondary = _score_scope_secondary(
        normalized_by_scope,
        add_effects,
        factorial_cells,
        dropout_effects,
        scopes=scopes,
        bootstrap=bootstrap,
        gates=scoring_gates["scope_secondary"],
    )

    norm_diagnostics = {}
    for norm_name, values_by_scope in (
        ("frobenius", clean_reference_update_frobenius_norms),
        ("rms", clean_reference_update_rms_norms),
    ):
        if not isinstance(values_by_scope, Mapping) or tuple(values_by_scope) != scopes:
            raise RuntimeError(f"E29.14d {norm_name} norm scope order changed")
        norm_diagnostics[norm_name] = {}
        for scope in scopes:
            norms = _finite_float64_tensor(
                values_by_scope[scope],
                shape=(48, context_count),
                label=f"{scope} {norm_name} clean reference norms",
            )
            if bool((norms < 0.0).any()):
                raise RuntimeError("E29.14d clean reference norms cannot be negative")
            norm_diagnostics[norm_name][scope] = _norm_summary(
                norms,
                component_effects=analyses[scope]["add_one"]["component_effect_means"],
                components=components,
                attention_components=attention_components,
                mlp_components=mlp_components,
                matched_pairs=matched_pairs,
                top_components=top_components,
                flanking_components=flanking_components,
            )

    return {
        "schema_version": "e29_14d_direct_causal_carrier_score_v1",
        "integrity_passed": True,
        "response_normalization": "raw_response_divided_by_positive_clean_endpoint",
        "primary_position_scope": primary_scope,
        "secondary_position_scope": scopes[1],
        "analyses_by_scope": analyses,
        "integrity_anchor_claims": anchors,
        "primary_claims": {
            "broad_attention_add_one_confirmation": bool(
                anchors["passed"] and class_passed
            ),
            "late_top4_factorial_confirmation": bool(
                anchors["passed"] and factorial_passed
            ),
            "causal_add_one_program_passed": causal_add_one_program_passed,
            "top4_vs_flanking_attention_secondary": bool(
                anchors["passed"]
                and primary["add_one"]["top4_vs_flanking_secondary_passed"]
            ),
        },
        "independent_dropout_claims": {
            "broad_attention_c47_dropout_necessity": bool(
                anchors["passed"] and dropout_class_passed
            ),
            "late_top4_individual_c47_dropout_necessity": bool(
                anchors["passed"] and dropout_top4_passed
            ),
            "all_registered_dropout_claims_passed": bool(
                anchors["passed"] and dropout_class_passed and dropout_top4_passed
            ),
        },
        "all_claims_passed_descriptive_conjunction": bool(
            anchors["passed"]
            and class_passed
            and factorial_passed
            and dropout_class_passed
            and dropout_top4_passed
        ),
        "all_claims_descriptive_conjunction_is_a_promotion_gate": False,
        "scope_concordance_descriptive": concordance,
        "scope_secondary_registered_claim": scope_secondary,
        "clean_reference_update_norm_diagnostics": norm_diagnostics,
        "claim_boundaries": {
            "causal_intervention": "full_donor_update_replacement_at_registered_positions",
            "clean_reference_norms_are_not_actual_injected_multi_mask_norms": True,
            "norm_matched_causal_lane_was_run": False,
            "carrier_effects_may_reflect": [
                "component_identity",
                "donor_minus_base_update_magnitude",
                "both_identity_and_magnitude",
            ],
            "identity_only_carrier_claim_authorized": False,
        },
        "recommended_interpretation": interpretation,
        "bootstrap": dict(bootstrap),
        "scoring_gates": dict(scoring_gates),
    }


def _validate_clean_label_metadata(
    metadata: Any,
    *,
    clean_contract: Sequence[Mapping[str, Any]],
    metadata_keys: Sequence[str],
    base_margins: torch.Tensor,
    donor_margins: torch.Tensor,
    endpoints: torch.Tensor,
) -> None:
    if not isinstance(metadata, list) or len(metadata) != len(clean_contract):
        raise RuntimeError("E29.14d clean metadata count changed")
    contract_fields = {
        "template",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
        "base_target_token_id",
        "donor_target_token_id",
        "token_length",
        "position_scope_indices",
    }
    for context_index, (record, sealed) in enumerate(
        zip(metadata, clean_contract, strict=True)
    ):
        if not isinstance(record, Mapping) or set(record) != set(metadata_keys):
            raise RuntimeError("E29.14d clean metadata keys changed")
        if (
            not isinstance(sealed, Mapping)
            or int(sealed["context_index"]) != context_index
        ):
            raise RuntimeError("E29.14d clean metadata contract order changed")
        if any(record[key] != sealed[key] for key in contract_fields):
            raise RuntimeError("E29.14d clean metadata differs from registration")
        fixed = {
            "model_layer_count": 24,
            "component_count": 48,
            "component_semantics": "attention_and_mlp_residual_updates",
            "patch_direction": "donor_to_base_only",
            "position_scope_order": [
                "all_positions",
                "differing_lexical_positions",
            ],
            "clean_logits_finite": True,
            "device": "cpu",
            "model_dtype": "float64",
        }
        if any(record[key] != value for key, value in fixed.items()):
            raise RuntimeError("E29.14d clean metadata runtime contract changed")
        base = float(record["clean_base_correct_minus_donor_margin"])
        donor = float(record["clean_donor_correct_minus_base_margin"])
        endpoint = float(record["clean_endpoint"])
        if (
            base != float(base_margins[context_index])
            or donor != float(donor_margins[context_index])
            or endpoint != float(endpoints[context_index])
            or not math.isclose(endpoint, base + donor, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise RuntimeError("E29.14d clean endpoint arithmetic changed")
        clean_max = float(record["clean_logit_absolute_max"])
        if (
            not math.isfinite(clean_max)
            or clean_max < 0.0
            or clean_max > float(NUMERICAL_GATES["clean_logit_absolute_max"])
        ):
            raise RuntimeError("E29.14d clean logit numerical gate changed")


def _expected_label_numerical_summary(
    responses: Mapping[str, torch.Tensor],
    endpoints: torch.Tensor,
    frobenius: Mapping[str, torch.Tensor],
    rms: Mapping[str, torch.Tensor],
    scopes: Sequence[str],
) -> dict[str, Any]:
    combined = torch.cat([responses[scope].flatten() for scope in scopes])
    return {
        "responses_by_scope": {
            scope: tensor_numerical_summary(responses[scope]) for scope in scopes
        },
        "combined_responses": tensor_numerical_summary(combined),
        "clean_endpoints": tensor_numerical_summary(endpoints),
        "clean_reference_update_frobenius_norms": {
            scope: {
                "shape": list(frobenius[scope].shape),
                "dtype": "float64",
                "finite": bool(frobenius[scope].isfinite().all()),
                "absolute_max": float(frobenius[scope].max()),
            }
            for scope in scopes
        },
        "clean_reference_update_rms_norms": {
            scope: {
                "shape": list(rms[scope].shape),
                "dtype": "float64",
                "finite": bool(rms[scope].isfinite().all()),
                "absolute_max": float(rms[scope].max()),
            }
            for scope in scopes
        },
    }


def _validate_batch_smoke(smoke: Any) -> None:
    if not isinstance(smoke, Mapping):
        raise TypeError("E29.14d batch-invariance smoke must be a mapping")
    require_exact_keys(
        smoke,
        {
            "records",
            "mask_rows",
            "unique_condition_rows",
            "physical_condition_cells",
            "scientific_labels_produced",
            "passed",
        },
        label="E29.14d batch-invariance smoke",
    )
    if (
        smoke["mask_rows"] != QUERY_ACCOUNTING["unique_condition_masks"]
        or smoke["unique_condition_rows"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_unique_query_rows"]
        or smoke["physical_condition_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"]
        or smoke["scientific_labels_produced"] is not False
        or smoke["passed"] is not True
    ):
        raise RuntimeError("E29.14d batch-invariance smoke contract changed")
    records = smoke["records"]
    if not isinstance(records, Mapping) or tuple(records) != POSITION_SCOPE_ORDER:
        raise RuntimeError("E29.14d batch-invariance scope order changed")
    for record in records.values():
        if not isinstance(record, Mapping):
            raise TypeError("E29.14d batch-invariance record must be a mapping")
        require_exact_keys(
            record,
            {
                "mask_rows",
                "reference_batch_size",
                "candidate_batch_size",
                "maximum_absolute_difference",
                "maximum_relative_difference",
                "atol",
                "rtol",
                "passed",
            },
            label="E29.14d batch-invariance record",
        )
        if (
            record["mask_rows"] != QUERY_ACCOUNTING["unique_condition_masks"]
            or record["reference_batch_size"]
            != BATCH_INVARIANCE["reference_batch_size"]
            or record["candidate_batch_size"]
            != BATCH_INVARIANCE["candidate_batch_size"]
            or record["atol"] != BATCH_INVARIANCE["absolute_tolerance"]
            or record["rtol"] != BATCH_INVARIANCE["relative_tolerance"]
            or record["passed"] is not True
        ):
            raise RuntimeError("E29.14d batch-invariance record changed")
        for key in ("maximum_absolute_difference", "maximum_relative_difference"):
            value = float(record[key])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError("E29.14d batch-invariance difference is invalid")


def _registered_scientific_arguments() -> dict[str, Any]:
    conditions = build_condition_manifest()
    hub_indices = [
        int(record["condition_index"])
        for record in conditions
        if tuple(record["selected_components"]) == (1,)
    ]
    dropouts = build_dropout_records()
    full_indices = {int(record["full_condition_index"]) for record in dropouts}
    if len(hub_indices) != 1 or len(full_indices) != 1:
        raise RuntimeError("E29.14d registered anchor mapping changed")
    return {
        "condition_count": QUERY_ACCOUNTING["unique_condition_masks"],
        "context_count": CONTEXT_COUNT,
        "position_scopes": POSITION_SCOPE_ORDER,
        "primary_scope": PRIMARY_POSITION_SCOPE,
        "components": TARGET_COMPONENT_ORDER,
        "attention_components": ATTENTION_COMPONENT_ORDER,
        "mlp_components": MLP_COMPONENT_ORDER,
        "matched_pairs": tuple((2 * layer, 2 * layer + 1) for layer in range(1, 24)),
        "top_components": CORE_ATTENTION_COMPONENTS,
        "flanking_components": REGIONAL_FLANKING_ATTENTION_COMPONENTS,
        "add_one_records": build_add_one_records(),
        "factorial_records": build_factorial_records(),
        "dropout_records": dropouts,
        "family_count": len(ADD_ONE_FAMILY_IDS),
        "factorial_background_count": len(ADD_ONE_FAMILY_IDS),
        "hub_singleton_condition_index": hub_indices[0],
        "full_condition_index": full_indices.pop(),
        "bootstrap": BOOTSTRAP,
        "scoring_gates": SCORING_GATES,
    }


def score_label_artifact(
    labels: Mapping[str, Any],
    *,
    expected_registration_sha256: str,
    expected_label_keys: Sequence[str],
    expected_condition_manifest: Sequence[Mapping[str, Any]],
    expected_query_manifest: Sequence[Mapping[str, Any]],
    expected_query_accounting: Mapping[str, Any],
    expected_physical_query_accounting: Mapping[str, Any],
    clean_metadata_contract: Sequence[Mapping[str, Any]],
    clean_metadata_keys: Sequence[str],
    scientific_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate immutable labels and call only the pure scientific scorer."""
    if not isinstance(labels, Mapping) or set(labels) != set(expected_label_keys):
        raise RuntimeError("E29.14d label keys changed")
    if (
        labels["schema_version"] != "e29_14d_direct_backup_labels_v1"
        or labels["registration_sha256"] != expected_registration_sha256
        or labels["expected_registration_sha256_supplied"]
        != expected_registration_sha256
    ):
        raise RuntimeError("E29.14d labels differ from the expected registration")
    if labels["condition_manifest"] != list(expected_condition_manifest):
        raise RuntimeError("E29.14d label condition manifest changed")
    if labels["query_manifest"] != list(expected_query_manifest):
        raise RuntimeError("E29.14d label query manifest or order changed")
    if labels["query_accounting"] != expected_query_accounting:
        raise RuntimeError("E29.14d label query accounting changed")
    if labels["physical_query_accounting"] != expected_physical_query_accounting:
        raise RuntimeError("E29.14d label physical query accounting changed")
    if labels["all_integrity_gates_passed"] is not True:
        raise RuntimeError("E29.14d label integrity did not pass")

    context_count = int(scientific_arguments["context_count"])
    condition_count = int(scientific_arguments["condition_count"])
    scopes = tuple(scientific_arguments["position_scopes"])
    endpoints = _finite_float64_tensor(
        labels["clean_endpoints"], shape=(context_count,), label="clean endpoints"
    )
    base_margins = _finite_float64_tensor(
        labels["clean_base_correct_minus_donor_margins"],
        shape=(context_count,),
        label="clean base correctness margins",
    )
    donor_margins = _finite_float64_tensor(
        labels["clean_donor_correct_minus_base_margins"],
        shape=(context_count,),
        label="clean donor correctness margins",
    )
    if (
        bool((base_margins <= 0.0).any())
        or bool((donor_margins <= 0.0).any())
        or not torch.equal(endpoints, base_margins + donor_margins)
    ):
        raise RuntimeError("E29.14d clean correctness or endpoint arithmetic changed")
    _validate_clean_label_metadata(
        labels["clean_metadata"],
        clean_contract=clean_metadata_contract,
        metadata_keys=clean_metadata_keys,
        base_margins=base_margins,
        donor_margins=donor_margins,
        endpoints=endpoints,
    )

    response_mapping = labels["responses_by_scope"]
    frobenius_mapping = labels["clean_reference_update_frobenius_norms"]
    rms_mapping = labels["clean_reference_update_rms_norms"]
    for mapping, label in (
        (response_mapping, "responses"),
        (frobenius_mapping, "Frobenius norms"),
        (rms_mapping, "RMS norms"),
    ):
        if not isinstance(mapping, Mapping) or tuple(mapping) != scopes:
            raise RuntimeError(f"E29.14d {label} scope keys or order changed")
    responses = {}
    frobenius = {}
    rms = {}
    for scope in scopes:
        responses[scope] = _finite_float64_tensor(
            response_mapping[scope],
            shape=(condition_count, context_count),
            label=f"{scope} responses",
        )
        if float(responses[scope].abs().max()) > float(
            NUMERICAL_GATES["response_absolute_max"]
        ):
            raise RuntimeError("E29.14d response magnitude gate changed")
        frobenius[scope] = _finite_float64_tensor(
            frobenius_mapping[scope],
            shape=(48, context_count),
            label=f"{scope} clean Frobenius norms",
        )
        rms[scope] = _finite_float64_tensor(
            rms_mapping[scope],
            shape=(48, context_count),
            label=f"{scope} clean RMS norms",
        )
    semantics = {
        "quantity": "clean_donor_update_minus_clean_base_update",
        "scopes": list(scopes),
        "frobenius": "l2_norm_over_selected_positions_and_hidden_coordinates",
        "rms": "root_mean_square_over_selected_positions_and_hidden_coordinates",
        "not_actual_multi_mask_injected_norms": True,
        "not_used_for_condition_selection_or_dosing": True,
        "descriptive_only": True,
    }
    if labels["clean_reference_norm_semantics"] != semantics:
        raise RuntimeError("E29.14d clean reference norm semantics changed")
    expected_numerical = _expected_label_numerical_summary(
        responses, endpoints, frobenius, rms, scopes
    )
    if labels["numerical_summary"] != expected_numerical:
        raise RuntimeError("E29.14d label numerical summary differs from tensors")
    _validate_batch_smoke(labels["batch_invariance_smoke"])

    score = score_direct_backup_response_tensors(
        responses,
        endpoints,
        frobenius,
        rms,
        integrity_passed=True,
        **scientific_arguments,
    )
    return {
        **score,
        "all_registered_integrity_gates_passed": True,
        "clean_summary": {
            "contexts": context_count,
            "positive_base_correctness_contexts": int((base_margins > 0.0).sum()),
            "positive_donor_correctness_contexts": int((donor_margins > 0.0).sum()),
            "minimum_clean_endpoint": float(endpoints.min()),
            "maximum_clean_endpoint": float(endpoints.max()),
        },
        "query_accounting": dict(expected_query_accounting),
        "physical_query_accounting": dict(expected_physical_query_accounting),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _validate_custody_summary(
    custody: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    expected_registration_sha256: str,
) -> None:
    require_exact_keys(custody, CUSTODY_KEYS, label="E29.14d acquisition custody")
    fixed = {
        "schema_version": "e29_14d_direct_backup_acquisition_custody_v1",
        "status": (
            "all_registered_direct_backup_conditions_acquired_after_integrity_gates"
        ),
        "registration_sha256": expected_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "source_code_sha256_verified_before_acquisition": registration[
            "source_code_sha256"
        ],
        "checkpoint_sha256_verified_before_acquisition": registration[
            "checkpoint_sha256"
        ],
        "protocol_artifact_sha256_verified_before_acquisition": registration[
            "protocol_artifact_sha256"
        ],
        "protocol_artifact_schema": PROTOCOL_ARTIFACT_SCHEMA,
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "prompt_records_sha256": registration["prompt_records_sha256"],
        "posthoc_candidate_provenance_sha256": registration[
            "posthoc_candidate_provenance_sha256"
        ],
        "prior_mask_audit_sha256": registration["prior_mask_audit_sha256"],
        "clean_metadata_contract_sha256": registration[
            "clean_metadata_contract_sha256"
        ],
        "causal_carrier_protocol_sha256": registration[
            "causal_carrier_protocol_sha256"
        ],
        "condition_manifest_sha256": registration["condition_manifest_sha256"],
        "query_manifest_sha256": registration["query_manifest_sha256"],
        "contexts": QUERY_ACCOUNTING["contexts"],
        "condition_masks": QUERY_ACCOUNTING["unique_condition_masks"],
        "position_scopes": QUERY_ACCOUNTING["position_scopes"],
        "patched_condition_rows_per_context": QUERY_ACCOUNTING[
            "patched_condition_rows_per_context"
        ],
        "patched_context_condition_cells": QUERY_ACCOUNTING[
            "patched_context_condition_cells"
        ],
        "clean_forward_examples": QUERY_ACCOUNTING["clean_forward_examples"],
        "batch_smoke_unique_query_rows": PHYSICAL_QUERY_ACCOUNTING[
            "batch_smoke_unique_query_rows"
        ],
        "batch_smoke_physical_condition_cells": PHYSICAL_QUERY_ACCOUNTING[
            "batch_smoke_physical_condition_cells"
        ],
        "maximum_logical_queries": QUERY_ACCOUNTING["maximum_logical_queries"],
        "maximum_physical_model_examples": PHYSICAL_QUERY_ACCOUNTING[
            "maximum_physical_model_examples"
        ],
        "batch_size": registration["execution"]["batch_size"],
        "batch_invariance_passed": True,
        "all_clean_and_numerical_gates_passed": True,
        "clean_reference_norms_are_diagnostic_only": True,
        "device": "cpu",
        "dtype": "float64",
    }
    if any(custody[key] != expected for key, expected in fixed.items()):
        raise RuntimeError("E29.14d acquisition custody contract changed")
    for key in ("response_absolute_max", "runtime_seconds"):
        value = float(custody[key])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"E29.14d custody {key} is invalid")


def score_from_custody(
    *,
    output_directory: Path,
    expected_registration_sha256: str,
    expected_acquisition_custody_sha256: str,
) -> dict[str, Any]:
    """Verify the complete immutable chain and write one scored summary."""
    custody_path = output_directory / "direct_backup_acquisition_custody.json"
    labels_path = output_directory / "direct_backup_labels.pt"
    attempt_path = output_directory / "direct_backup_acquisition_attempt.json"
    summary_path = output_directory / "direct_backup_summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to overwrite E29.14d scored summary")

    verify_file_hash(custody_path, expected_acquisition_custody_sha256)
    custody = _load_json(custody_path)
    registration_path = Path(custody["registration"])
    registration = load_verified_registration(
        registration_path,
        expected_sha256=expected_registration_sha256,
        required_keys=REGISTRATION_REQUIRED_KEYS,
    )
    registration = validate_registration_payload(
        registration,
        source_directory=SOURCE_DIRECTORY,
        verify_external_files=True,
    )
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14d scoring environment differs from registration")
    _validate_custody_summary(
        custody,
        registration=registration,
        expected_registration_sha256=expected_registration_sha256,
    )
    if Path(custody["registration"]).resolve() != registration_path.resolve():
        raise RuntimeError("E29.14d custody registration path changed")
    if Path(custody["direct_backup_labels"]).resolve() != labels_path.resolve():
        raise RuntimeError("E29.14d custody points to another label artifact")
    if Path(custody["acquisition_attempt"]).resolve() != attempt_path.resolve():
        raise RuntimeError("E29.14d custody points to another acquisition attempt")
    protocol_path = Path(registration["protocol_artifact"])
    if Path(custody["protocol_artifact"]).resolve() != protocol_path.resolve():
        raise RuntimeError("E29.14d custody points to another protocol artifact")

    verify_file_hash(labels_path, str(custody["direct_backup_labels_sha256"]))
    verify_file_hash(attempt_path, str(custody["acquisition_attempt_sha256"]))
    masks = load_verified_protocol_artifact(
        protocol_path, expected_sha256=registration["protocol_artifact_sha256"]
    )
    if tuple(masks.shape) != (
        QUERY_ACCOUNTING["unique_condition_masks"],
        COMPONENT_COUNT,
    ):
        raise RuntimeError("E29.14d verified protocol mask shape changed")
    attempt = _load_json(attempt_path)
    require_exact_keys(attempt, ATTEMPT_REQUIRED_KEYS, label="E29.14d attempt")
    expected_attempt = build_acquisition_attempt_record(
        registration=registration,
        registration_path=registration_path,
        observed_registration_sha256=expected_registration_sha256,
        expected_registration_sha256=expected_registration_sha256,
    )
    if attempt != expected_attempt:
        raise RuntimeError("E29.14d acquisition attempt differs from registration")

    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(labels, Mapping):
        raise TypeError("E29.14d label artifact must contain a mapping")
    scientific_arguments = _registered_scientific_arguments()
    summary = score_label_artifact(
        labels,
        expected_registration_sha256=expected_registration_sha256,
        expected_label_keys=LABEL_KEYS,
        expected_condition_manifest=build_condition_manifest(),
        expected_query_manifest=build_query_manifest(),
        expected_query_accounting=QUERY_ACCOUNTING,
        expected_physical_query_accounting=PHYSICAL_QUERY_ACCOUNTING,
        clean_metadata_contract=registration["clean_metadata_contract"],
        clean_metadata_keys=CLEAN_METADATA_KEYS,
        scientific_arguments=scientific_arguments,
    )
    observed_response_max = max(
        float(labels["responses_by_scope"][scope].abs().max())
        for scope in POSITION_SCOPE_ORDER
    )
    if float(custody["response_absolute_max"]) != observed_response_max:
        raise RuntimeError("E29.14d custody response maximum differs from labels")
    summary["custody"] = {
        "registration": str(registration_path.resolve()),
        "registration_sha256": expected_registration_sha256,
        "acquisition_attempt": str(attempt_path.resolve()),
        "acquisition_attempt_sha256": str(custody["acquisition_attempt_sha256"]),
        "acquisition_custody": str(custody_path.resolve()),
        "acquisition_custody_sha256": expected_acquisition_custody_sha256,
        "direct_backup_labels": str(labels_path.resolve()),
        "direct_backup_labels_sha256": str(custody["direct_backup_labels_sha256"]),
        "protocol_artifact": str(protocol_path.resolve()),
        "protocol_artifact_sha256": str(registration["protocol_artifact_sha256"]),
        "clean_metadata_contract_sha256": str(
            registration["clean_metadata_contract_sha256"]
        ),
    }
    atomic_json_save(summary, summary_path)
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--expected-acquisition-custody-sha256", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            score_from_custody(
                output_directory=arguments.output_directory,
                expected_registration_sha256=arguments.expected_registration_sha256,
                expected_acquisition_custody_sha256=(
                    arguments.expected_acquisition_custody_sha256
                ),
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "score_direct_backup_response_tensors",
    "score_from_custody",
    "score_label_artifact",
]
