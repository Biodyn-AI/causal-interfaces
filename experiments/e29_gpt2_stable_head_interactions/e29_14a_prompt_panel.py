"""Response-blind lexical prompt panel for prospective E29.14a work.

This module defines fresh prompt strings and tokenizer-level checks only.  It
does not load a transformer model, query responses, or define a protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    TEMPLATES,
    validate_e29_11_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.e29_13_prompt_panel import (
    e29_13_prompt_panel,
    prior_e29_8_to_e29_12_group_words,
)


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


def _group_words(records: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(word)
        for record in records
        for side in ("base_words", "donor_words")
        for word in record[side]
    }


def prior_e29_8_to_e29_13_group_words() -> frozenset[str]:
    """Return the complete lexical exclusion set through E29.13."""
    return frozenset(
        set(prior_e29_8_to_e29_12_group_words()) | _group_words(e29_13_prompt_panel())
    )


def e29_14a_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Return twelve fresh context pairs over 96 distinct group words."""
    groups = (
        ("crimson", "coral", "olive", "bright"),
        ("elephant", "donkey", "dolphin", "spider"),
        ("onion", "garlic", "carrot", "potato"),
        ("carpet", "curtain", "blanket", "basket"),
        ("January", "March", "June", "August"),
        ("ceramic", "granite", "ivory", "carbon"),
        ("heat", "rainbow", "smoke", "dust"),
        ("finger", "thumb", "chest", "waist"),
        ("captain", "officer", "manager", "worker"),
        ("bacon", "cookie", "beef", "pork"),
        ("library", "museum", "airport", "station"),
        ("jeans", "cap", "tie", "robe"),
        ("sedan", "motor", "rail", "jet"),
        ("sword", "blade", "gun", "pipe"),
        ("football", "basketball", "volleyball", "skiing"),
        ("today", "tomorrow", "daily", "annual"),
        ("speak", "listen", "touch", "hold"),
        ("honor", "trust", "faith", "mercy"),
        ("ink", "oil", "wax", "foam"),
        ("coast", "harbor", "creek", "jungle"),
        ("melody", "rhythm", "choir", "tune"),
        ("family", "friend", "neighbor", "guest"),
        ("zero", "nine", "fifty", "dozen"),
        ("dry", "wet", "rough", "smooth"),
    )
    return tuple(
        _induction_record(
            TEMPLATES[context_index // 3],
            groups[2 * context_index],
            groups[2 * context_index + 1],
        )
        for context_index in range(12)
    )


def validate_e29_14a_prompt_panel(
    tokenizer: Any,
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Validate novelty through E29.13 and pinned-tokenizer alignment."""
    return validate_e29_11_prompt_panel(
        tokenizer,
        records,
        prior_group_words=prior_e29_8_to_e29_13_group_words(),
    )


__all__ = [
    "e29_14a_prompt_panel",
    "prior_e29_8_to_e29_13_group_words",
    "validate_e29_14a_prompt_panel",
]
