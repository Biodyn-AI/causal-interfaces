"""Frozen constants and scoring rules for the staged E29.8 scale-up."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

MODEL_SNAPSHOT = Path(
    "/Users/ihorkendiukhov/.cache/huggingface/hub/"
    "models--openai-community--gpt2-medium/snapshots/"
    "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
)
MODEL_REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
SOURCE_MASK_SEED = 29_801
TRANSFER_MASK_SEED = 29_808
ORDER_SEED = 29_809
TRAINING_SEEDS = (29_811, 29_812, 29_813, 29_814, 29_815)
BOOTSTRAP_SEED = 29_816
RIDGE_VALUES = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0)
SOURCE_METHODS = (
    "tt_natural_rank1",
    "tt_natural_rank2",
    "tt_natural_rank4",
    "cp_rank4",
    "mlp_hidden7",
    "tt_reverse_rank2",
    "tt_random_rank2",
    "additive",
    "pairwise",
)
PRIMARY_METHOD = "tt_natural_rank2"


def response_metrics(
    labels: torch.Tensor, predictions: torch.Tensor
) -> dict[str, Any]:
    """Score predictions after centering each prompt context independently."""
    labels = labels.double()
    predictions = predictions.double()
    if labels.ndim != 2 or labels.shape != predictions.shape:
        raise ValueError("labels and predictions must be same-shaped matrices")
    residual = labels - predictions
    centered = labels - labels.mean(dim=0, keepdim=True)
    denominator = centered.square().sum()
    if denominator <= 0:
        raise ValueError("response labels have no context-centered variation")
    per_context_r2 = []
    for context_index in range(labels.shape[1]):
        context_denominator = centered[:, context_index].square().sum()
        if context_denominator <= 0:
            per_context_r2.append(float("-inf"))
        else:
            per_context_r2.append(
                float(
                    1.0
                    - residual[:, context_index].square().sum()
                    / context_denominator
                )
            )
    return {
        "r2": float(1.0 - residual.square().sum() / denominator),
        "mse": float(residual.square().mean()),
        "mae": float(residual.abs().mean()),
        "per_context_r2": per_context_r2,
        # torch.median intentionally uses the conservative lower middle value.
        "median_context_r2": float(torch.tensor(per_context_r2).median()),
        "minimum_context_r2": min(per_context_r2),
    }


def source_gate_results(
    method_metrics: Mapping[str, Mapping[str, Any]],
    *,
    clean_margins: Sequence[float],
    transfer_latent_rank: int,
    transfer_latent_condition: float,
) -> dict[str, Any]:
    """Apply every preregistered source-scale promotion condition."""
    required = {PRIMARY_METHOD, "additive", "mlp_hidden7"}
    if not required <= method_metrics.keys():
        raise ValueError("method metrics omit a required source-gate control")
    primary = method_metrics[PRIMARY_METHOD]
    contexts = [float(value) for value in primary["per_context_r2"]]
    if len(contexts) != 8:
        raise ValueError("the source gate requires exactly eight contexts")
    results = {
        "test_r2_at_least_0_70": float(primary["r2"]) >= 0.70,
        "median_context_r2_at_least_0_60": (
            float(primary["median_context_r2"]) >= 0.60
        ),
        "at_least_7_of_8_contexts_r2_at_least_0_30": (
            sum(value >= 0.30 for value in contexts) >= 7
        ),
        "no_context_r2_below_zero": min(contexts) >= 0.0,
        "mse_at_most_0_70_additive": (
            float(primary["mse"]) <= 0.70 * float(method_metrics["additive"]["mse"])
        ),
        "mse_at_most_0_90_mlp7": (
            float(primary["mse"])
            <= 0.90 * float(method_metrics["mlp_hidden7"]["mse"])
        ),
        "all_clean_induction_margins_positive": (
            len(clean_margins) == 8
            and all(float(margin) > 0.0 for margin in clean_margins)
        ),
        "transfer_calibration_latent_rank_is_two": transfer_latent_rank == 2,
        "transfer_calibration_latent_condition_at_most_1000": (
            float(transfer_latent_condition) <= 1_000.0
        ),
    }
    return {"results": results, "all_passed": all(results.values())}


__all__ = [
    "BOOTSTRAP_SEED",
    "MODEL_REVISION",
    "MODEL_SNAPSHOT",
    "ORDER_SEED",
    "PRIMARY_METHOD",
    "RIDGE_VALUES",
    "SOURCE_MASK_SEED",
    "SOURCE_METHODS",
    "TRAINING_SEEDS",
    "TRANSFER_MASK_SEED",
    "response_metrics",
    "source_gate_results",
]
