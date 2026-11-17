"""Pure, response-blind lexical prompt panel for prospective E29.13 work.

The panel extends the lexical holdout through E29.12.  This module defines
prompt strings and tokenizer-level validation only; it never loads a model or
scientific response artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    TEMPLATES,
    validate_e29_11_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.e29_12_prompt_panel import (
    e29_12_prompt_panel,
    prior_e29_8_to_e29_11_group_words,
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


def prior_e29_8_to_e29_12_group_words() -> frozenset[str]:
    """Return the complete lexical exclusion set through E29.12."""
    return frozenset(
        set(prior_e29_8_to_e29_11_group_words()) | _group_words(e29_12_prompt_panel())
    )


def e29_13_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Return twelve new context pairs using 96 distinct group words."""
    groups = (
        ("red", "blue", "green", "yellow"),
        ("wolf", "bear", "lion", "tiger"),
        ("apple", "pear", "peach", "grape"),
        ("table", "chair", "desk", "lamp"),
        ("spring", "summer", "autumn", "winter"),
        ("gold", "lead", "tin", "nickel"),
        ("rain", "snow", "cloud", "wind"),
        ("hand", "foot", "arm", "head"),
        ("doctor", "nurse", "editor", "dancer"),
        ("bread", "rice", "salad", "cheese"),
        ("hospital", "prison", "garage", "factory"),
        ("shirt", "coat", "sock", "shoe"),
        ("train", "bus", "car", "plane"),
        ("pen", "key", "ruler", "brush"),
        ("cricket", "poker", "racing", "hunting"),
        ("morning", "evening", "night", "day"),
        ("walk", "run", "jump", "dance"),
        ("anger", "fear", "hope", "love"),
        ("wood", "glass", "stone", "paper"),
        ("river", "lake", "ocean", "beach"),
        ("opera", "jazz", "rock", "blues"),
        ("king", "queen", "prince", "princess"),
        ("one", "two", "three", "four"),
        ("hot", "cold", "warm", "fresh"),
    )
    return tuple(
        _induction_record(
            TEMPLATES[context_index // 3],
            groups[2 * context_index],
            groups[2 * context_index + 1],
        )
        for context_index in range(12)
    )


def validate_e29_13_prompt_panel(
    tokenizer: Any,
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Validate novelty through E29.12 and pinned-tokenizer alignment."""
    return validate_e29_11_prompt_panel(
        tokenizer,
        records,
        prior_group_words=prior_e29_8_to_e29_12_group_words(),
    )


__all__ = [
    "e29_13_prompt_panel",
    "prior_e29_8_to_e29_12_group_words",
    "validate_e29_13_prompt_panel",
]
