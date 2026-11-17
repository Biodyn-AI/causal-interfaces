"""Response-blind primitives for prospective E29.11 multiscale transfer.

The module defines prompts, deterministic masks, methods, query accounting,
and gates. It performs no transformer call, accepts no response labels, and
does not register or mutate any prior artifact.
"""

from __future__ import annotations

import itertools
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
MASK_SEEDS = {
    "calibration_low": 30_101,
    "calibration_size8": 30_102,
    "bridge_size16": 30_103,
    "confirmation_size24": 30_104,
    "conditional_size32": 30_105,
    "count_control_size8": 30_106,
}
FIT_SPLITS = ("calibration_low", "calibration_size8", "bridge_size16")
FIT_ACQUISITION_SPLITS = (*FIT_SPLITS, "count_control_size8")
PREDICTION_SPLITS = ("confirmation_size24", "conditional_size32")
MASK_SHAPES = {
    "calibration_low": (96, COMPONENT_COUNT),
    "calibration_size8": (32, COMPONENT_COUNT),
    "bridge_size16": (64, COMPONENT_COUNT),
    "count_control_size8": (64, COMPONENT_COUNT),
    "confirmation_size24": (64, COMPONENT_COUNT),
    "conditional_size32": (32, COMPONENT_COUNT),
}
TEMPLATES = (
    "Replay the sequence: {a} {b} {c} {d} | {a} {b} {c}",
    "Loop example: {a} {b} {c} {d}; continue {a} {b} {c}",
    "After {a} {b} {c} {d}, repeat {a} {b} {c}",
    "Pattern: {a} {b} {c} {d}. Continue: {a} {b} {c}",
)
TEMPLATE_CONTEXTS = {
    "replay": (0, 1, 2),
    "loop": (3, 4, 5),
    "adversarial_after_repeat": (6, 7, 8),
    "new_pattern_paraphrase": (9, 10, 11),
}
ADVERSARIAL_TEMPLATE_CONTEXTS = TEMPLATE_CONTEXTS["adversarial_after_repeat"]

PRIMARY_METHOD = "mlp7_multiscale_ridge10"
SECONDARY_METHOD = "tt2_multiscale_ridge1e-4"
COUNT_MATCHED_CONTROL_METHOD = "mlp7_low_size8_count_matched_ridge10"
PRIMARY_RIDGE = 10.0
SECONDARY_RIDGE = 1e-4
RANDOM_FEATURE_SEEDS = tuple(range(29_921, 29_926))
COMPONENT_PERMUTATION_SEEDS = tuple(range(29_931, 29_941))
METHOD_MANIFEST = (
    PRIMARY_METHOD,
    SECONDARY_METHOD,
    "mlp7_low_only_ridge10",
    "mlp7_low_size8_ridge10",
    COUNT_MATCHED_CONTROL_METHOD,
    "tt2_low_only_ridge1e-4",
    "tt2_low_size8_ridge1e-4",
    "additive_multiscale_ridge10",
    *(f"mlp7_random_seed{seed}_multiscale_ridge10" for seed in RANDOM_FEATURE_SEEDS),
    *(
        f"mlp7_permuted_seed{seed}_multiscale_ridge10"
        for seed in COMPONENT_PERMUTATION_SEEDS
    ),
    "mlp7_permuted_reverse_multiscale_ridge10",
    "mlp7_permuted_attn_mlp_swap_multiscale_ridge10",
    "mlp7_low_slope_multiscale_intercept",
    "tt2_low_slope_multiscale_intercept",
)
RANDOM_FEATURE_METHODS = METHOD_MANIFEST[8:13]
RANDOM_CONTROL_METHODS = RANDOM_FEATURE_METHODS
PERMUTATION_CONTROL_METHODS = METHOD_MANIFEST[13:25]
INTERCEPT_SECONDARY_METHODS = METHOD_MANIFEST[25:]
MULTISCALE_FIT_PROTOCOL = {
    "primary_fit_splits": FIT_SPLITS,
    "low_only_ablation_fit_splits": ("calibration_low",),
    "low_size8_ablation_fit_splits": (
        "calibration_low",
        "calibration_size8",
    ),
    "size8_count_matched_control_fit_splits": (
        "calibration_low",
        "calibration_size8",
        "count_control_size8",
    ),
    "nested_report_only_budgets": {
        "calibration_low": (64, 96),
        "calibration_size8": (16, 32),
        "bridge_size16": (16, 32, 64),
    },
    "equal_n_exchange_contrast": {
        "without_size16": (
            ("calibration_low", 96),
            ("calibration_size8", 32),
        ),
        "with_size16": (
            ("calibration_low", 64),
            ("calibration_size8", 32),
            ("bridge_size16", 32),
        ),
        "fit_rows_each": 128,
    },
    "fit_weighting": "equal_weight_per_mask_row",
    "feature_and_response_standardization": "fit_rows_only",
    "ridge_selection": "fixed_from_e29_10_cached_diagnosis_no_selection",
    "refit_after_confirmation": False,
    "confirmation_split": "confirmation_size24",
    "conditional_promotion_split": "conditional_size32",
    "conditional_size32_predictions_frozen_before_size24_labels": True,
    "conditional_size32_outcomes_acquired_only_if_primary_size24_passes": True,
}
SIZE24_CONFIRMATION_GATES = {
    "pooled_context_centered_r2_min": 0.55,
    "median_context_r2_min": 0.50,
    "positive_context_r2_min_count": 10,
    "minimum_context_r2_min": -2.0,
    "adversarial_template_pooled_r2_strict_min": 0.0,
    "new_pattern_paraphrase_pooled_r2_strict_min": 0.0,
    "mse_over_low_size8_mlp_max": 0.75,
    "bridge_benefit_bootstrap_ratio_upper_strict_max": 1.0,
    "normalized_bridge_benefit_bootstrap_ratio_upper_strict_max": 1.0,
    "mse_over_size8_count_matched_control_strict_max": 1.0,
    "size8_count_matched_bootstrap_ratio_upper_strict_max": 1.0,
    "mse_over_additive_max": 0.80,
    "mse_over_median_random_control_max": 0.80,
    "mse_over_median_permutation_control_max": 0.80,
    "positive_clean_margins_required": 12,
    "positive_donor_clean_margins_required": 12,
    "numerical_integrity_required": True,
}
TT2_SECONDARY_GATES = {
    "pooled_context_centered_r2_min": 0.30,
    "median_context_r2_min": 0.35,
    "positive_context_r2_min_count": 9,
    "minimum_context_r2_min": -3.0,
    "mse_over_low_size8_tt2_max": 1.05,
}
MLP7_OVER_TT2_SUPERIORITY_GATES = {
    "primary_mse_over_tt2_strict_max": 1.0,
    "paired_bootstrap_ratio_upper_strict_max": 1.0,
}
C32_PROMOTION_GATES = {
    "pooled_context_centered_r2_min": 0.40,
    "median_context_r2_min": 0.35,
    "positive_context_r2_min_count": 9,
    "positive_clean_margins_required": 12,
    "positive_donor_clean_margins_required": 12,
    "numerical_integrity_required": True,
}
QUERY_BUDGET = {
    "contexts": 12,
    "fit_masks": 256,
    "primary_fit_masks": 192,
    "size8_count_control_masks": 64,
    "confirmation_masks": 64,
    "conditional_promotion_masks": 32,
    "patched_queries_before_conditional_promotion": 3_840,
    "clean_queries_before_conditional_promotion": 72,
    "logical_queries_before_conditional_promotion": 3_912,
    "conditional_patched_queries": 384,
    "conditional_clean_queries": 36,
    "conditional_logical_queries": 420,
    "maximum_patched_queries": 4_224,
    "maximum_clean_queries": 108,
    "maximum_logical_queries": 4_332,
}


def _induction_record(
    template: str,
    base_words: tuple[str, str, str, str],
    donor_words: tuple[str, str, str, str],
) -> dict[str, Any]:
    def render(words: tuple[str, str, str, str]) -> str:
        return template.format(a=words[0], b=words[1], c=words[2], d=words[3])

    return {
        "template": template,
        "base_words": base_words,
        "donor_words": donor_words,
        "base_prompt": render(base_words),
        "donor_prompt": render(donor_words),
        "base_target": base_words[3],
        "donor_target": donor_words[3],
    }


def e29_11_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Return twelve fresh pairs over three E29.9 templates plus one new one."""
    pairs = (
        (("navy", "cyan", "cream", "golden"), ("eagle", "shark", "turtle", "camel")),
        (
            ("banana", "lemon", "cherry", "mango"),
            ("cabinet", "couch", "bench", "shelf"),
        ),
        (
            ("Monday", "Tuesday", "Wednesday", "Thursday"),
            ("diamond", "pearl", "ruby", "crystal"),
        ),
        (("storm", "thunder", "lightning", "fog"), ("leg", "knee", "neck", "face")),
        (
            ("farmer", "baker", "driver", "artist"),
            ("butter", "honey", "pasta", "pizza"),
        ),
        (("house", "hotel", "church", "tower"), ("jacket", "skirt", "glove", "scarf")),
        (("taxi", "subway", "rocket", "wagon"), ("drill", "saw", "chain", "wire")),
        (("soccer", "tennis", "golf", "chess"), ("minute", "noon", "midnight", "dawn")),
        (("turn", "throw", "catch", "climb"), ("joy", "grief", "shame", "envy")),
        (("leather", "wool", "silk", "brick"), ("canyon", "cave", "pond", "shore")),
        (("organ", "horn", "bass", "trumpet"), ("mother", "father", "son", "daughter")),
        (("ten", "twenty", "thirty", "forty"), ("sweet", "sour", "bitter", "spicy")),
    )
    return tuple(
        _induction_record(TEMPLATES[index // 3], *pair)
        for index, pair in enumerate(pairs)
    )


def validate_e29_11_prompt_panel(
    tokenizer: Any,
    records: tuple[dict[str, Any], ...],
    *,
    prior_group_words: Collection[str],
) -> tuple[dict[str, Any], ...]:
    """Validate lexical novelty and exact pinned-tokenizer alignment."""
    if len(records) != CONTEXT_COUNT:
        raise ValueError("E29.11 requires exactly twelve contexts")
    template_counts = Counter(str(record["template"]) for record in records)
    if len(template_counts) != 4 or set(template_counts.values()) != {3}:
        raise ValueError("E29.11 requires four templates used three times each")
    groups = [
        tuple(str(word) for word in record[key])
        for record in records
        for key in ("base_words", "donor_words")
    ]
    words = {word for group in groups for word in group}
    if any(len(group) != 4 for group in groups):
        raise ValueError("each E29.11 group must have four words")
    if len(groups) != 24 or len(set(groups)) != 24 or len(words) != 96:
        raise ValueError("E29.11 requires 24 groups and 96 unique words")
    overlap = sorted(words & {str(word) for word in prior_group_words})
    if overlap:
        raise ValueError(f"E29.11 words overlap prior E29 panels: {overlap}")

    checks = []
    for context_index, record in enumerate(records):
        prompt_ids = {
            side: tokenizer.encode(record[f"{side}_prompt"], add_special_tokens=False)
            for side in ("base", "donor")
        }
        if len(prompt_ids["base"]) != len(prompt_ids["donor"]):
            raise ValueError(f"context {context_index} prompt lengths differ")
        side_checks = {}
        for side in ("base", "donor"):
            side_words = tuple(str(word) for word in record[f"{side}_words"])
            word_token_lengths = {
                word: len(tokenizer.encode(" " + word, add_special_tokens=False))
                for word in side_words
            }
            if set(word_token_lengths.values()) != {1}:
                raise ValueError(f"context {context_index} has a split group word")
            target = str(record[f"{side}_target"])
            if target != side_words[3]:
                raise ValueError(f"context {context_index} target is not word d")
            target_id = tokenizer.encode(" " + target, add_special_tokens=False)[0]
            target_positions = [
                index
                for index, token_id in enumerate(prompt_ids[side])
                if token_id == target_id
            ]
            if len(target_positions) != 1:
                raise ValueError(f"context {context_index} target is ambiguous")
            extended = tokenizer.encode(
                record[f"{side}_prompt"] + " " + target,
                add_special_tokens=False,
            )
            if extended != [*prompt_ids[side], target_id]:
                raise ValueError(f"context {context_index} target is not exact append")
            side_checks[side] = {
                "target": target,
                "target_token_id": int(target_id),
                "demonstration_target_position": target_positions[0],
                "prediction_position": len(prompt_ids[side]),
                "next_token_append_valid": True,
                "word_token_lengths": word_token_lengths,
            }
        if (
            side_checks["base"]["demonstration_target_position"]
            != side_checks["donor"]["demonstration_target_position"]
        ):
            raise ValueError(f"context {context_index} target positions differ")
        if (
            side_checks["base"]["target_token_id"]
            == side_checks["donor"]["target_token_id"]
        ):
            raise ValueError(f"context {context_index} target IDs coincide")
        checks.append(
            {
                "context_index": context_index,
                "token_length": len(prompt_ids["base"]),
                "target_positions_aligned": True,
                "sides": side_checks,
                **deepcopy(record),
            }
        )
    return tuple(checks)


def _rows(masks: torch.Tensor) -> list[tuple[int, ...]]:
    return [
        tuple(torch.nonzero(row, as_tuple=False).flatten().tolist()) for row in masks
    ]


def _collect_prior_rows(
    prior: Mapping[str, Any], label: str
) -> dict[int, set[tuple[int, ...]]]:
    tensors: list[torch.Tensor] = []
    if {"source", "transfer"} <= set(prior):
        for family in ("source", "transfer"):
            if not isinstance(prior[family], Mapping):
                raise TypeError(f"{label}/{family} is not a mapping")
            tensors.extend(prior[family].values())
    elif isinstance(prior.get("masks"), Mapping):
        tensors.extend(prior["masks"].values())
    else:
        tensors.extend(
            value for value in prior.values() if isinstance(value, torch.Tensor)
        )
    if not tensors:
        raise ValueError(f"{label} has no mask tensors")
    collected: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} contains a non-tensor split")
        if tensor.ndim != 2 or tensor.shape[1] != COMPONENT_COUNT or not len(tensor):
            raise ValueError(f"{label} contains an invalid mask shape")
        if not bool(torch.logical_or(tensor == 0, tensor == 1).all()):
            raise ValueError(f"{label} contains non-binary masks")
        for row in _rows(tensor):
            if not row:
                raise ValueError(f"{label} contains an empty mask")
            collected[len(row)].add(row)
    return dict(collected)


def _masks(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    masks = torch.zeros((len(rows), COMPONENT_COUNT), dtype=torch.float64)
    for index, row in enumerate(rows):
        masks[index, list(row)] = 1.0
    return masks


def _balanced_rows(
    cardinality: int,
    count: int,
    coverage: list[int],
    excluded: set[tuple[int, ...]],
    generator: random.Random,
) -> list[tuple[int, ...]]:
    candidates = [
        row
        for row in itertools.combinations(range(COMPONENT_COUNT), cardinality)
        if row not in excluded
    ]
    generator.shuffle(candidates)
    selected = []
    for _ in range(count):
        if not candidates:
            raise ValueError(f"too few fresh cardinality-{cardinality} masks")
        best = min(
            range(len(candidates)),
            key=lambda index: (
                max(coverage[item] for item in candidates[index]),
                sum(coverage[item] for item in candidates[index]),
            ),
        )
        row = candidates.pop(best)
        selected.append(row)
        for item in row:
            coverage[item] += 1
    return selected


def _sample_rows(
    cardinality: int,
    count: int,
    excluded: set[tuple[int, ...]],
    generator: random.Random,
) -> list[tuple[int, ...]]:
    """Sample unique masks while keeping component incidence within one."""
    selected: list[tuple[int, ...]] = []
    seen = set(excluded)
    coverage = [0] * COMPONENT_COUNT
    while len(selected) < count:
        for _ in range(10_000):
            components = list(range(COMPONENT_COUNT))
            generator.shuffle(components)
            components.sort(key=coverage.__getitem__)
            row = tuple(sorted(components[:cardinality]))
            if row not in seen:
                break
        else:
            raise RuntimeError(
                f"could not find a fresh balanced cardinality-{cardinality} mask"
            )
        seen.add(row)
        selected.append(row)
        for component in row:
            coverage[component] += 1
    if max(coverage) - min(coverage) > 1:
        raise RuntimeError(
            f"cardinality-{cardinality} component incidence is not balanced"
        )

    # Greedy load balancing can create artificial antithetic pairs (especially
    # at cardinality 24). Degree-preserving switches randomize the design while
    # retaining exact row sums and column incidence.
    row_sets = [set(row) for row in selected]
    selected_set = set(selected)
    for _ in range(max(10_000, count * 200)):
        left_index, right_index = generator.sample(range(count), 2)
        left = row_sets[left_index]
        right = row_sets[right_index]
        left_only = tuple(left - right)
        right_only = tuple(right - left)
        if not left_only or not right_only:
            continue
        remove_left = generator.choice(left_only)
        remove_right = generator.choice(right_only)
        new_left = tuple(sorted(left - {remove_left} | {remove_right}))
        new_right = tuple(sorted(right - {remove_right} | {remove_left}))
        old_left = tuple(sorted(left))
        old_right = tuple(sorted(right))
        occupied = selected_set - {old_left, old_right}
        if (
            new_left in excluded
            or new_right in excluded
            or new_left == new_right
            or new_left in occupied
            or new_right in occupied
        ):
            continue
        selected_set = occupied | {new_left, new_right}
        row_sets[left_index] = set(new_left)
        row_sets[right_index] = set(new_right)
    selected = [tuple(sorted(row)) for row in row_sets]
    if cardinality * 2 == COMPONENT_COUNT:
        row_set = set(selected)
        complements = {
            tuple(component for component in range(COMPONENT_COUNT) if component not in row)
            for row in selected
        }
        if row_set & complements:
            raise RuntimeError("balanced half-cardinality masks retain complement pairs")
    return selected


def _balanced_mask_order(
    rows: Sequence[tuple[int, ...]],
    cardinality_schedule: Sequence[int],
    generator: random.Random,
) -> list[tuple[int, ...]]:
    """Order a fixed row set so every registered prefix spreads incidence."""
    if len(rows) != len(cardinality_schedule):
        raise ValueError("mask rows and cardinality schedule differ in length")
    remaining: defaultdict[int, list[tuple[int, ...]]] = defaultdict(list)
    for row in rows:
        remaining[len(row)].append(row)
    coverage = [0] * COMPONENT_COUNT
    ordered: list[tuple[int, ...]] = []
    for cardinality in cardinality_schedule:
        candidates = remaining[cardinality]
        if not candidates:
            raise ValueError("cardinality schedule requests an unavailable mask")
        generator.shuffle(candidates)

        def imbalance(row: tuple[int, ...]) -> tuple[int, int, int]:
            updated = [
                count + int(component in row)
                for component, count in enumerate(coverage)
            ]
            return (
                max(updated) - min(updated),
                sum(count * count for count in updated),
                max(updated),
            )

        selected_index = min(range(len(candidates)), key=lambda i: imbalance(candidates[i]))
        selected = candidates.pop(selected_index)
        ordered.append(selected)
        for component in selected:
            coverage[component] += 1
    if any(remaining.values()):
        raise RuntimeError("balanced mask ordering left unused rows")
    return ordered


def build_e29_11_mask_protocol(
    sealed_e29_8_masks: Mapping[str, Any],
    sealed_e29_9_masks: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build fresh E29.11 rows, intentionally reusing all singletons only."""
    old8 = _collect_prior_rows(sealed_e29_8_masks, "E29.8")
    old9 = _collect_prior_rows(sealed_e29_9_masks, "E29.9")
    old = {
        cardinality: old8.get(cardinality, set()) | old9.get(cardinality, set())
        for cardinality in set(old8) | set(old9)
    }
    low_generator = random.Random(MASK_SEEDS["calibration_low"])
    coverage = [1] * COMPONENT_COUNT
    low_rows: list[tuple[int, ...]] = [(index,) for index in range(COMPONENT_COUNT)]
    low_rows.extend(_balanced_rows(2, 24, coverage, old.get(2, set()), low_generator))
    low_rows.extend(_balanced_rows(3, 24, coverage, old.get(3, set()), low_generator))
    if min(coverage) < 3:
        raise RuntimeError("E29.11 low-calibration incidence is below three")
    low_rows = _balanced_mask_order(
        low_rows,
        (1, 1, 2, 3) * 24,
        random.Random(MASK_SEEDS["calibration_low"] + 1_000),
    )
    calibration_size8_rows = _sample_rows(
            8,
            32,
            old.get(8, set()),
            random.Random(MASK_SEEDS["calibration_size8"]),
        )
    calibration_size8 = _masks(
        _balanced_mask_order(
            calibration_size8_rows,
            (8,) * 32,
            random.Random(MASK_SEEDS["calibration_size8"] + 1_000),
        )
    )
    count_control_excluded = old.get(8, set()) | set(_rows(calibration_size8))
    protocol = {
        "calibration_low": _masks(low_rows),
        "calibration_size8": calibration_size8,
        "bridge_size16": _masks(
            _balanced_mask_order(
                _sample_rows(
                    16,
                    64,
                    old.get(16, set()),
                    random.Random(MASK_SEEDS["bridge_size16"]),
                ),
                (16,) * 64,
                random.Random(MASK_SEEDS["bridge_size16"] + 1_000),
            )
        ),
        "count_control_size8": _masks(
            _sample_rows(
                8,
                64,
                count_control_excluded,
                random.Random(MASK_SEEDS["count_control_size8"]),
            )
        ),
        "confirmation_size24": _masks(
            _sample_rows(
                24,
                64,
                old.get(24, set()),
                random.Random(MASK_SEEDS["confirmation_size24"]),
            )
        ),
        "conditional_size32": _masks(
            _sample_rows(
                32,
                32,
                old.get(32, set()),
                random.Random(MASK_SEEDS["conditional_size32"]),
            )
        ),
    }
    split_rows: defaultdict[int, list[set[tuple[int, ...]]]] = defaultdict(list)
    for split, masks in protocol.items():
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise RuntimeError(f"generated {split} masks are non-binary")
        for cardinality in masks.sum(1).to(torch.int64).unique().tolist():
            rows = {row for row in _rows(masks) if len(row) == cardinality}
            expected = int((masks.sum(1) == cardinality).sum())
            if len(rows) != expected:
                raise RuntimeError(f"generated {split} contains duplicate masks")
            if cardinality >= 2 and not rows.isdisjoint(old.get(cardinality, set())):
                raise RuntimeError(f"generated {split} overlaps a prior mask")
            split_rows[cardinality].append(rows)
    for cardinality, row_sets in split_rows.items():
        if cardinality < 2:
            continue
        for left, right in itertools.combinations(row_sets, 2):
            if not left.isdisjoint(right):
                raise RuntimeError(
                    f"generated cardinality-{cardinality} splits overlap"
                )
    return protocol


def validate_method_manifest(methods: Sequence[str]) -> tuple[str, ...]:
    manifest = tuple(methods)
    if manifest != METHOD_MANIFEST:
        raise ValueError(
            f"method manifest mismatch: expected {METHOD_MANIFEST}, got {manifest}"
        )
    return manifest


def size24_confirmation_gate_results(
    *,
    primary_metrics: Mapping[str, Any],
    low_size8_mlp_mse: float,
    count_matched_size8_mlp_mse: float,
    additive_mse: float,
    random_control_mses: Sequence[float],
    permutation_control_mses: Sequence[float],
    adversarial_template_pooled_r2: float,
    new_pattern_paraphrase_pooled_r2: float,
    bridge_benefit_bootstrap_ratio_upper: float,
    normalized_bridge_benefit_bootstrap_ratio_upper: float,
    size8_count_matched_bootstrap_ratio_upper: float,
    clean_margins: Sequence[float],
    donor_clean_margins: Sequence[float],
    numerical_integrity_passed: bool,
) -> dict[str, Any]:
    context_r2 = [float(value) for value in primary_metrics["per_context_r2"]]
    if len(context_r2) != CONTEXT_COUNT:
        raise ValueError("E29.11 size-24 gate requires twelve contexts")
    gates = SIZE24_CONFIRMATION_GATES
    if len(random_control_mses) != len(RANDOM_CONTROL_METHODS):
        raise ValueError("E29.11 requires five random-feature controls")
    if len(permutation_control_mses) != len(PERMUTATION_CONTROL_METHODS):
        raise ValueError("E29.11 requires twelve component-permutation controls")
    random_median = statistics.median(float(value) for value in random_control_mses)
    permutation_median = statistics.median(
        float(value) for value in permutation_control_mses
    )
    primary_mse = float(primary_metrics["mse"])
    results = {
        "pooled_r2_at_least_0_55": float(primary_metrics["r2"])
        >= gates["pooled_context_centered_r2_min"],
        "median_context_r2_at_least_0_50": float(primary_metrics["median_context_r2"])
        >= gates["median_context_r2_min"],
        "at_least_ten_contexts_r2_positive": sum(value > 0 for value in context_r2)
        >= gates["positive_context_r2_min_count"],
        "minimum_context_r2_at_least_minus_2": min(context_r2)
        >= gates["minimum_context_r2_min"],
        "adversarial_after_template_pooled_r2_positive": float(
            adversarial_template_pooled_r2
        )
        > gates["adversarial_template_pooled_r2_strict_min"],
        "new_pattern_paraphrase_pooled_r2_positive": float(
            new_pattern_paraphrase_pooled_r2
        )
        > gates["new_pattern_paraphrase_pooled_r2_strict_min"],
        "mse_at_most_0_75_low_size8_mlp": primary_mse
        <= gates["mse_over_low_size8_mlp_max"] * float(low_size8_mlp_mse),
        "paired_bridge_benefit_bootstrap_upper_below_1": float(
            bridge_benefit_bootstrap_ratio_upper
        )
        < gates["bridge_benefit_bootstrap_ratio_upper_strict_max"],
        "normalized_bridge_bootstrap_upper_below_1": float(
            normalized_bridge_benefit_bootstrap_ratio_upper
        )
        < gates["normalized_bridge_benefit_bootstrap_ratio_upper_strict_max"],
        "mse_below_size8_count_matched_control": primary_mse
        < gates["mse_over_size8_count_matched_control_strict_max"]
        * float(count_matched_size8_mlp_mse),
        "size8_count_matched_bootstrap_upper_below_1": float(
            size8_count_matched_bootstrap_ratio_upper
        )
        < gates["size8_count_matched_bootstrap_ratio_upper_strict_max"],
        "mse_at_most_0_80_additive": primary_mse
        <= gates["mse_over_additive_max"] * float(additive_mse),
        "mse_at_most_0_80_median_random_control": primary_mse
        <= gates["mse_over_median_random_control_max"] * random_median,
        "mse_at_most_0_80_median_permutation_control": primary_mse
        <= gates["mse_over_median_permutation_control_max"] * permutation_median,
        "all_twelve_clean_margins_positive": len(clean_margins)
        == gates["positive_clean_margins_required"]
        and all(float(value) > 0 for value in clean_margins),
        "all_twelve_donor_clean_margins_positive": len(donor_clean_margins)
        == gates["positive_donor_clean_margins_required"]
        and all(float(value) > 0 for value in donor_clean_margins),
        "numerical_integrity_passed": bool(numerical_integrity_passed),
    }
    return {
        "results": results,
        "all_passed": all(results.values()),
        "median_random_control_mse": random_median,
        "median_permutation_control_mse": permutation_median,
        "tt2_superiority_is_separate": True,
    }


def tt2_secondary_gate_results(
    *, secondary_metrics: Mapping[str, Any], low_size8_tt2_mse: float
) -> dict[str, Any]:
    """Apply TT2 robustness gates separately from the primary claim."""
    context_r2 = [float(value) for value in secondary_metrics["per_context_r2"]]
    if len(context_r2) != CONTEXT_COUNT:
        raise ValueError("E29.11 TT2 gate requires twelve contexts")
    gates = TT2_SECONDARY_GATES
    results = {
        "pooled_r2_at_least_0_30": float(secondary_metrics["r2"])
        >= gates["pooled_context_centered_r2_min"],
        "median_context_r2_at_least_0_35": float(secondary_metrics["median_context_r2"])
        >= gates["median_context_r2_min"],
        "at_least_nine_contexts_r2_positive": sum(value > 0 for value in context_r2)
        >= gates["positive_context_r2_min_count"],
        "minimum_context_r2_at_least_minus_3": min(context_r2)
        >= gates["minimum_context_r2_min"],
        "mse_at_most_1_05_low_size8_tt2": float(secondary_metrics["mse"])
        <= gates["mse_over_low_size8_tt2_max"] * float(low_size8_tt2_mse),
    }
    return {"results": results, "all_passed": all(results.values())}


def mlp7_over_tt2_superiority_gate_results(
    *, primary_mse: float, tt2_mse: float, paired_bootstrap_ratio_upper: float
) -> dict[str, Any]:
    """Report MLP7-over-TT2 superiority without gating basic confirmation."""
    gates = MLP7_OVER_TT2_SUPERIORITY_GATES
    results = {
        "primary_mse_strictly_below_tt2": float(primary_mse)
        < gates["primary_mse_over_tt2_strict_max"] * float(tt2_mse),
        "paired_bootstrap_ratio_upper_below_1": float(paired_bootstrap_ratio_upper)
        < gates["paired_bootstrap_ratio_upper_strict_max"],
    }
    return {
        "results": results,
        "all_passed": all(results.values()),
        "required_for_scale_ladder_confirmation": False,
    }


def conditional_size32_gate_results(
    metrics: Mapping[str, Any],
    *,
    size24_confirmation_passed: bool,
    clean_margins: Sequence[float],
    donor_clean_margins: Sequence[float],
    numerical_integrity_passed: bool,
) -> dict[str, Any]:
    if not size24_confirmation_passed:
        return {
            "queried": False,
            "all_passed": None,
            "reason": "size24_confirmation_did_not_pass",
        }
    context_r2 = [float(value) for value in metrics["per_context_r2"]]
    if len(context_r2) != CONTEXT_COUNT:
        raise ValueError("E29.11 size-32 gate requires twelve contexts")
    gates = C32_PROMOTION_GATES
    results = {
        "pooled_r2_at_least_0_40": float(metrics["r2"])
        >= gates["pooled_context_centered_r2_min"],
        "median_context_r2_at_least_0_35": float(metrics["median_context_r2"])
        >= gates["median_context_r2_min"],
        "at_least_nine_contexts_r2_positive": sum(value > 0 for value in context_r2)
        >= gates["positive_context_r2_min_count"],
        "all_twelve_clean_margins_positive": len(clean_margins)
        == gates["positive_clean_margins_required"]
        and all(float(value) > 0 for value in clean_margins),
        "all_twelve_donor_clean_margins_positive": len(donor_clean_margins)
        == gates["positive_donor_clean_margins_required"]
        and all(float(value) > 0 for value in donor_clean_margins),
        "numerical_integrity_passed": bool(numerical_integrity_passed),
    }
    return {"queried": True, "results": results, "all_passed": all(results.values())}


__all__ = [
    "ADVERSARIAL_TEMPLATE_CONTEXTS",
    "C32_PROMOTION_GATES",
    "COMPONENT_COUNT",
    "COMPONENT_PERMUTATION_SEEDS",
    "CONTEXT_COUNT",
    "COUNT_MATCHED_CONTROL_METHOD",
    "FIT_ACQUISITION_SPLITS",
    "FIT_SPLITS",
    "INTERCEPT_SECONDARY_METHODS",
    "MASK_SEEDS",
    "MASK_SHAPES",
    "METHOD_MANIFEST",
    "MLP7_OVER_TT2_SUPERIORITY_GATES",
    "MULTISCALE_FIT_PROTOCOL",
    "PERMUTATION_CONTROL_METHODS",
    "PREDICTION_SPLITS",
    "PRIMARY_METHOD",
    "PRIMARY_RIDGE",
    "QUERY_BUDGET",
    "RANDOM_CONTROL_METHODS",
    "RANDOM_FEATURE_METHODS",
    "RANDOM_FEATURE_SEEDS",
    "SECONDARY_METHOD",
    "SECONDARY_RIDGE",
    "SIZE24_CONFIRMATION_GATES",
    "TEMPLATES",
    "TEMPLATE_CONTEXTS",
    "TT2_SECONDARY_GATES",
    "build_e29_11_mask_protocol",
    "conditional_size32_gate_results",
    "e29_11_prompt_panel",
    "mlp7_over_tt2_superiority_gate_results",
    "size24_confirmation_gate_results",
    "tt2_secondary_gate_results",
    "validate_e29_11_prompt_panel",
    "validate_method_manifest",
]
