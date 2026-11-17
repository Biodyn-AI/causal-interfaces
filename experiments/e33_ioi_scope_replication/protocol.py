"""Pure, response-blind protocol for the E33.1 IOI scope replication.

The panel, interventions, controls, and gates in this module are fixed without
loading a language model.  The sixteen confirmation prompts use names absent
from the earlier E29.15 IOI panels and from the four-context development pilot.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

SCHEMA_VERSION = "e33_1_ioi_scope_replication_protocol_v1"
MODEL_ID = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
LAYER_COUNT = 12
COMPONENT_COUNT = 24
CONTEXT_COUNT = 16
BOOTSTRAP_SEED = 33_101
BOOTSTRAP_REPLICATES = 5_000

TEMPLATES = (
    (
        "when_market",
        "When {a} and {b} went to the market, {b} gave a book to",
    ),
    (
        "after_station",
        "After {a} and {b} arrived at the station, {b} passed a drink to",
    ),
    (
        "while_office",
        "While {a} and {b} worked in the office, {b} sent a letter to",
    ),
    (
        "because_school",
        "Because {a} and {b} met at the school, {b} brought a gift to",
    ),
)

# All names are single leading-space GPT-2 tokens and are absent from the
# frozen E29.15 name pool.  Four unique names define each context, so no name
# is reused anywhere in the confirmation panel.
CONFIRMATION_NAMES = tuple(
    "Alex Amber Angela Ann Anna Anthony Ashley Austin Barry Ben Beverly Billy "
    "Bobby Brad Brandon Bryan Caleb Catherine Cheryl Chloe Christian Christina "
    "Christopher Cindy Connor Craig Crystal Dale Dana Dean Dennis Douglas Dustin "
    "Earl Eleanor Elijah Elizabeth Erin Gabriel James Jamie Jean Jeffrey Jeremy "
    "Jordan Jose Joseph Joshua Julie Justin Katherine Kathryn Katie Kimberly Larry "
    "Lauren Lawrence Liam Madison Maria Marie Marilyn Mason Michelle".split()
)

POSITION_SCOPES = (
    "all_positions",
    "differing_lexical_positions",
    "shared_static_positions",
    "matched_static_a",
    "matched_static_b",
)


def _condition(
    condition_id: str,
    components: Sequence[int],
    scope: str,
    *,
    patch_mode: str = "source_replace",
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "components": list(components),
        "scope": scope,
        "patch_mode": patch_mode,
    }


CONDITIONS = (
    _condition("mlp0_all", (1,), "all_positions"),
    _condition("mlp0_lexical", (1,), "differing_lexical_positions"),
    _condition("mlp0_shared_static", (1,), "shared_static_positions"),
    _condition("mlp0_matched_static_a", (1,), "matched_static_a"),
    _condition("mlp0_matched_static_b", (1,), "matched_static_b"),
    _condition("attention0_lexical", (0,), "differing_lexical_positions"),
    _condition(
        "mlp0_lexical_energy_scrambled",
        (1,),
        "differing_lexical_positions",
        patch_mode="half_roll_delta",
    ),
    _condition("first12_all", tuple(range(12)), "all_positions"),
    _condition(
        "first12_lexical", tuple(range(12)), "differing_lexical_positions"
    ),
    _condition("all24_all", tuple(range(24)), "all_positions"),
    _condition(
        "all24_lexical", tuple(range(24)), "differing_lexical_positions"
    ),
)

CONDITION_IDS = tuple(condition["condition_id"] for condition in CONDITIONS)
DIRECTIONS = ("forward", "reverse")

# Thresholds were fixed from a separate four-context development panel using
# disjoint names.  The primary claim is scope equivalence, not exact endpoint
# transfer by the layer-0 MLP alone.
SCORING_GATES = {
    "clean_viability": {
        "minimum_pairwise_correct_contexts_each_direction": 16,
        "minimum_clean_endpoint": 0.50,
    },
    "mlp0_scope_equivalence": {
        "minimum_mean_transfer_each_direction": 0.60,
        "maximum_one_sided_95_bootstrap_mean_absolute_gap": 0.04,
        "maximum_mean_absolute_shared_static": 0.03,
    },
    "matched_static_controls": {
        "maximum_mean_absolute_each_control": 0.05,
    },
    "neighbor_component_control": {
        "minimum_mlp0_minus_attention0_mean_transfer": 0.45,
    },
    "energy_matched_control": {
        "minimum_mlp0_minus_scrambled_mean_transfer": 0.50,
        "maximum_scrambled_mean_absolute_transfer": 0.15,
    },
    "mesoscale_first12": {
        "minimum_mean_lexical_transfer_each_direction": 0.90,
        "maximum_one_sided_95_bootstrap_mean_absolute_gap": 0.05,
    },
    "mesoscale_all24": {
        "minimum_mean_lexical_transfer_each_direction": 0.92,
        "maximum_one_sided_95_bootstrap_endpoint_gap": 0.08,
    },
    "full_distribution": {
        "minimum_mlp0_lexical_js_fidelity_each_direction": 0.80,
        "minimum_first12_lexical_js_fidelity_each_direction": 0.94,
        "minimum_all24_lexical_js_fidelity_each_direction": 0.95,
    },
    "counterfactual_answer": {
        "minimum_mlp0_pairwise_wins_each_direction": 14,
        "minimum_all24_pairwise_wins_each_direction": 15,
    },
}


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with one canonical encoding."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_prompt_manifest() -> tuple[dict[str, str], ...]:
    """Return the immutable sixteen-context, four-template IOI panel."""
    if len(CONFIRMATION_NAMES) != 4 * CONTEXT_COUNT:
        raise RuntimeError("the confirmation name panel must contain 64 names")
    records = []
    for context_index in range(CONTEXT_COUNT):
        template_index = context_index // 4
        template_id, template = TEMPLATES[template_index]
        a, b, c, d = CONFIRMATION_NAMES[4 * context_index : 4 * context_index + 4]
        records.append(
            {
                "context_id": f"ioi_scope_{context_index:02d}",
                "stratum_id": template_id,
                "template_id": template_id,
                "template": template,
                "base_name_a": a,
                "base_name_b": b,
                "donor_name_a": c,
                "donor_name_b": d,
                "base_prompt": template.format(a=a, b=b),
                "donor_prompt": template.format(a=c, b=d),
                "base_target": a,
                "donor_target": c,
            }
        )
    return tuple(records)


def validate_prompt_manifest(
    tokenizer: Any, records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Validate exact prompt identity and tokenizer alignment without responses."""
    copied = tuple(dict(record) for record in records)
    if copied != build_prompt_manifest():
        raise ValueError("the E33.1 prompt manifest changed")
    checks = []
    observed_names: set[str] = set()
    for record in copied:
        names = (
            record["base_name_a"],
            record["base_name_b"],
            record["donor_name_a"],
            record["donor_name_b"],
        )
        if observed_names.intersection(names):
            raise ValueError("confirmation names must be globally unique")
        observed_names.update(names)
        name_token_ids = {}
        for name in names:
            token_ids = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"name is not one leading-space token: {name}")
            name_token_ids[name] = int(token_ids[0])
        base_ids = tokenizer.encode(record["base_prompt"], add_special_tokens=False)
        donor_ids = tokenizer.encode(record["donor_prompt"], add_special_tokens=False)
        if len(base_ids) != len(donor_ids):
            raise ValueError(f"unaligned prompts: {record['context_id']}")
        differing = tuple(
            index
            for index, (base_id, donor_id) in enumerate(
                zip(base_ids, donor_ids, strict=True)
            )
            if base_id != donor_id
        )
        if len(differing) != 3:
            raise ValueError(
                f"{record['context_id']} must differ at exactly three name positions"
            )
        checks.append(
            {
                **deepcopy(record),
                "token_count": len(base_ids),
                "differing_positions": list(differing),
                "name_token_ids": name_token_ids,
            }
        )
    return tuple(checks)


def matched_static_scope(
    context_id: str,
    shared_positions: torch.Tensor,
    *,
    replicate: str,
    count: int,
) -> torch.Tensor:
    """Select a response-blind, equal-cardinality subset of shared positions."""
    shared = torch.as_tensor(shared_positions, dtype=torch.bool).detach().cpu()
    if shared.ndim != 1 or count < 1 or int(shared.sum()) < count:
        raise ValueError("shared position mask cannot supply the requested control")
    candidates = torch.where(shared)[0].tolist()
    ranked = sorted(
        candidates,
        key=lambda index: hashlib.sha256(
            f"e33.1|{context_id}|{replicate}|{index}".encode("utf-8")
        ).digest(),
    )
    scope = torch.zeros_like(shared)
    scope[ranked[:count]] = True
    return scope


def build_protocol() -> dict[str, Any]:
    """Return the complete JSON-compatible protocol payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_count": LAYER_COUNT,
        "component_count": COMPONENT_COUNT,
        "context_count": CONTEXT_COUNT,
        "directions": list(DIRECTIONS),
        "position_scopes": list(POSITION_SCOPES),
        "conditions": list(deepcopy(CONDITIONS)),
        "prompt_manifest": list(build_prompt_manifest()),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling": "four_contexts_with_replacement_within_each_template",
            "one_sided_upper_quantile": 0.95,
        },
        "scoring_gates": deepcopy(SCORING_GATES),
        "response_coordinate": (
            "(patched_source_minus_recipient_logit_margin-"
            "clean_recipient_margin)/(clean_source_margin-clean_recipient_margin)"
        ),
        "distribution_coordinate": (
            "1-JS(patched,clean_source)/JS(clean_recipient,clean_source)"
        ),
        "query_accounting": {
            "clean_forward_examples": 2 * CONTEXT_COUNT,
            "patched_examples": (
                len(DIRECTIONS) * len(CONDITIONS) * CONTEXT_COUNT
            ),
            "total_logical_examples": (
                (2 + len(DIRECTIONS) * len(CONDITIONS)) * CONTEXT_COUNT
            ),
        },
        "claim_boundary": (
            "Independent GPT-2-small IOI replication of token-position scope "
            "equivalence; not model universality, a minimal circuit, or an "
            "identified abstract variable."
        ),
    }


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_protocol()
    if dict(protocol) != expected:
        raise ValueError("the E33.1 response-blind protocol changed")
    return expected


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "COMPONENT_COUNT",
    "CONDITIONS",
    "CONDITION_IDS",
    "CONTEXT_COUNT",
    "DIRECTIONS",
    "MODEL_ID",
    "MODEL_REVISION",
    "SCORING_GATES",
    "build_prompt_manifest",
    "build_protocol",
    "canonical_sha256",
    "matched_static_scope",
    "validate_prompt_manifest",
    "validate_protocol",
]
