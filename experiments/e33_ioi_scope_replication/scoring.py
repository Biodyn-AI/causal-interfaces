"""Pure scoring for the preregistered E33.1 IOI replication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e33_ioi_scope_replication.protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONDITION_IDS,
    CONTEXT_COUNT,
    DIRECTIONS,
    SCORING_GATES,
)


def _stratified_upper(values: torch.Tensor, *, absolute: bool = False) -> float:
    vector = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    if vector.shape != (CONTEXT_COUNT,) or not bool(vector.isfinite().all()):
        raise ValueError("bootstrap values must be one finite value per context")
    if absolute:
        vector = vector.abs()
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    estimates = torch.empty(BOOTSTRAP_REPLICATES, dtype=torch.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = torch.cat(
            [
                start
                + torch.randint(4, (4,), generator=generator, dtype=torch.long)
                for start in (0, 4, 8, 12)
            ]
        )
        estimates[replicate] = vector[selected].mean()
    return float(torch.quantile(estimates, 0.95))


def _validated_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    copied = tuple(dict(record) for record in records)
    if len(copied) != CONTEXT_COUNT:
        raise ValueError("E33.1 scoring requires exactly sixteen context records")
    context_ids = tuple(record.get("context_id") for record in copied)
    if len(set(context_ids)) != CONTEXT_COUNT:
        raise ValueError("E33.1 context IDs must be unique")
    for record in copied:
        if set(record.get("directions", {})) != set(DIRECTIONS):
            raise ValueError("a context is missing a direction")
        for direction in DIRECTIONS:
            if tuple(record["directions"][direction]) != CONDITION_IDS:
                raise ValueError("condition identities or order changed")
    return copied


def score_context_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score every frozen gate without model access or adaptive choices."""
    contexts = _validated_records(records)
    condition_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    vectors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    clean_summary: dict[str, Any] = {}
    for direction in DIRECTIONS:
        clean_rows = [context["clean"][direction] for context in contexts]
        clean_viable = torch.tensor(
            [
                bool(row["recipient_pairwise_correct"])
                and bool(row["source_pairwise_correct"])
                for row in clean_rows
            ],
            dtype=torch.bool,
        )
        endpoints = torch.tensor(
            [float(row["endpoint"]) for row in clean_rows], dtype=torch.float64
        )
        clean_summary[direction] = {
            "pairwise_correct_contexts": int(clean_viable.sum()),
            "minimum_endpoint": float(endpoints.min()),
            "mean_endpoint": float(endpoints.mean()),
            "recipient_target_top1_contexts": sum(
                bool(row["recipient_target_top1"]) for row in clean_rows
            ),
            "source_target_top1_contexts": sum(
                bool(row["source_target_top1"]) for row in clean_rows
            ),
        }
        vectors[direction] = {}
        condition_metrics[direction] = {}
        for condition_id in CONDITION_IDS:
            rows = [
                context["directions"][direction][condition_id]
                for context in contexts
            ]
            transfer = torch.tensor(
                [float(row["normalized_transfer"]) for row in rows],
                dtype=torch.float64,
            )
            fidelity = torch.tensor(
                [float(row["js_fidelity"]) for row in rows],
                dtype=torch.float64,
            )
            if not bool(transfer.isfinite().all() and fidelity.isfinite().all()):
                raise ValueError("a scientific metric is non-finite")
            pairwise = torch.tensor(
                [bool(row["counterfactual_pairwise_win"]) for row in rows],
                dtype=torch.bool,
            )
            top1 = torch.tensor(
                [bool(row["counterfactual_target_top1"]) for row in rows],
                dtype=torch.bool,
            )
            vectors[direction][condition_id] = {
                "transfer": transfer,
                "fidelity": fidelity,
                "pairwise": pairwise,
                "top1": top1,
            }
            condition_metrics[direction][condition_id] = {
                "mean_normalized_transfer": float(transfer.mean()),
                "mean_absolute_normalized_transfer": float(transfer.abs().mean()),
                "minimum_normalized_transfer": float(transfer.min()),
                "maximum_normalized_transfer": float(transfer.max()),
                "mean_js_fidelity": float(fidelity.mean()),
                "minimum_js_fidelity": float(fidelity.min()),
                "counterfactual_pairwise_wins": int(pairwise.sum()),
                "counterfactual_target_top1": int(top1.sum()),
            }

    gap_metrics: dict[str, dict[str, float]] = {}
    gates: dict[str, dict[str, bool]] = {}
    clean_gate = SCORING_GATES["clean_viability"]
    gates["clean_viability"] = {
        f"{direction}_all_contexts_pairwise_correct": (
            clean_summary[direction]["pairwise_correct_contexts"]
            >= int(clean_gate["minimum_pairwise_correct_contexts_each_direction"])
        )
        for direction in DIRECTIONS
    }
    gates["clean_viability"].update(
        {
            f"{direction}_minimum_endpoint": (
                clean_summary[direction]["minimum_endpoint"]
                >= float(clean_gate["minimum_clean_endpoint"])
            )
            for direction in DIRECTIONS
        }
    )

    scope_gate = SCORING_GATES["mlp0_scope_equivalence"]
    gates["mlp0_scope_equivalence"] = {}
    gates["matched_static_controls"] = {}
    gates["neighbor_component_control"] = {}
    gates["energy_matched_control"] = {}
    gates["mesoscale_first12"] = {}
    gates["mesoscale_all24"] = {}
    gates["full_distribution"] = {}
    gates["counterfactual_answer"] = {}
    for direction in DIRECTIONS:
        direction_vectors = vectors[direction]
        mlp_all = direction_vectors["mlp0_all"]["transfer"]
        mlp_lexical = direction_vectors["mlp0_lexical"]["transfer"]
        mlp_shared = direction_vectors["mlp0_shared_static"]["transfer"]
        attention = direction_vectors["attention0_lexical"]["transfer"]
        scrambled = direction_vectors["mlp0_lexical_energy_scrambled"]["transfer"]
        first12_all = direction_vectors["first12_all"]["transfer"]
        first12_lexical = direction_vectors["first12_lexical"]["transfer"]
        all24_lexical = direction_vectors["all24_lexical"]["transfer"]
        gap_metrics[direction] = {
            "mlp0_lexical_vs_all_mean_absolute_gap": float(
                (mlp_lexical - mlp_all).abs().mean()
            ),
            "mlp0_lexical_vs_all_bootstrap_upper": _stratified_upper(
                mlp_lexical - mlp_all, absolute=True
            ),
            "first12_lexical_vs_all_mean_absolute_gap": float(
                (first12_lexical - first12_all).abs().mean()
            ),
            "first12_lexical_vs_all_bootstrap_upper": _stratified_upper(
                first12_lexical - first12_all, absolute=True
            ),
            "all24_lexical_endpoint_mean_absolute_gap": float(
                (all24_lexical - 1.0).abs().mean()
            ),
            "all24_lexical_endpoint_bootstrap_upper": _stratified_upper(
                all24_lexical - 1.0, absolute=True
            ),
            "mlp0_minus_attention0_mean_transfer": float(
                mlp_lexical.mean() - attention.mean()
            ),
            "mlp0_minus_scrambled_mean_transfer": float(
                mlp_lexical.mean() - scrambled.mean()
            ),
        }
        gates["mlp0_scope_equivalence"].update(
            {
                f"{direction}_all_mean_transfer": float(mlp_all.mean())
                >= float(scope_gate["minimum_mean_transfer_each_direction"]),
                f"{direction}_lexical_mean_transfer": float(mlp_lexical.mean())
                >= float(scope_gate["minimum_mean_transfer_each_direction"]),
                f"{direction}_lexical_vs_all_gap": gap_metrics[direction][
                    "mlp0_lexical_vs_all_bootstrap_upper"
                ]
                <= float(
                    scope_gate[
                        "maximum_one_sided_95_bootstrap_mean_absolute_gap"
                    ]
                ),
                f"{direction}_shared_static": float(mlp_shared.abs().mean())
                <= float(scope_gate["maximum_mean_absolute_shared_static"]),
            }
        )
        static_limit = float(
            SCORING_GATES["matched_static_controls"][
                "maximum_mean_absolute_each_control"
            ]
        )
        for suffix in ("a", "b"):
            static = direction_vectors[f"mlp0_matched_static_{suffix}"]["transfer"]
            gates["matched_static_controls"][f"{direction}_{suffix}"] = float(
                static.abs().mean()
            ) <= static_limit
        gates["neighbor_component_control"][direction] = gap_metrics[direction][
            "mlp0_minus_attention0_mean_transfer"
        ] >= float(
            SCORING_GATES["neighbor_component_control"][
                "minimum_mlp0_minus_attention0_mean_transfer"
            ]
        )
        energy_gate = SCORING_GATES["energy_matched_control"]
        gates["energy_matched_control"].update(
            {
                f"{direction}_contrast": gap_metrics[direction][
                    "mlp0_minus_scrambled_mean_transfer"
                ]
                >= float(energy_gate["minimum_mlp0_minus_scrambled_mean_transfer"]),
                f"{direction}_scrambled_absolute": float(scrambled.abs().mean())
                <= float(energy_gate["maximum_scrambled_mean_absolute_transfer"]),
            }
        )
        first_gate = SCORING_GATES["mesoscale_first12"]
        gates["mesoscale_first12"].update(
            {
                f"{direction}_mean_transfer": float(first12_lexical.mean())
                >= float(first_gate["minimum_mean_lexical_transfer_each_direction"]),
                f"{direction}_scope_gap": gap_metrics[direction][
                    "first12_lexical_vs_all_bootstrap_upper"
                ]
                <= float(
                    first_gate[
                        "maximum_one_sided_95_bootstrap_mean_absolute_gap"
                    ]
                ),
            }
        )
        all_gate = SCORING_GATES["mesoscale_all24"]
        gates["mesoscale_all24"].update(
            {
                f"{direction}_mean_transfer": float(all24_lexical.mean())
                >= float(all_gate["minimum_mean_lexical_transfer_each_direction"]),
                f"{direction}_endpoint_gap": gap_metrics[direction][
                    "all24_lexical_endpoint_bootstrap_upper"
                ]
                <= float(
                    all_gate["maximum_one_sided_95_bootstrap_endpoint_gap"]
                ),
            }
        )
        distribution_gate = SCORING_GATES["full_distribution"]
        for condition_id, threshold_key in (
            ("mlp0_lexical", "minimum_mlp0_lexical_js_fidelity_each_direction"),
            (
                "first12_lexical",
                "minimum_first12_lexical_js_fidelity_each_direction",
            ),
            ("all24_lexical", "minimum_all24_lexical_js_fidelity_each_direction"),
        ):
            gates["full_distribution"][f"{direction}_{condition_id}"] = float(
                direction_vectors[condition_id]["fidelity"].mean()
            ) >= float(distribution_gate[threshold_key])
        answer_gate = SCORING_GATES["counterfactual_answer"]
        gates["counterfactual_answer"].update(
            {
                f"{direction}_mlp0": int(
                    direction_vectors["mlp0_lexical"]["pairwise"].sum()
                )
                >= int(answer_gate["minimum_mlp0_pairwise_wins_each_direction"]),
                f"{direction}_all24": int(
                    direction_vectors["all24_lexical"]["pairwise"].sum()
                )
                >= int(answer_gate["minimum_all24_pairwise_wins_each_direction"]),
            }
        )

    all_gates_passed = all(
        passed for family in gates.values() for passed in family.values()
    )
    return {
        "schema_version": "e33_1_ioi_scope_replication_score_v1",
        "context_count": CONTEXT_COUNT,
        "clean_summary": clean_summary,
        "condition_metrics": condition_metrics,
        "gap_metrics": gap_metrics,
        "gates": gates,
        "all_registered_gates_passed": all_gates_passed,
        "recommended_interpretation": (
            "Independent model-and-behavior replication of token-position scope "
            "equivalence across a focal early MLP and 12/24-sublayer compositions."
            if all_gates_passed
            else "The registered independent replication is mixed or negative."
        ),
        "claim_boundary": (
            "GPT-2-small IOI panel only; no universality, minimality, or abstract-"
            "variable identification claim."
        ),
    }


__all__ = ["score_context_records"]
