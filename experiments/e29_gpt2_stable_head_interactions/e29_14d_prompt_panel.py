"""Response-blind fresh lexical prompt panel for prospective E29.14d.

The words were selected using lexical exclusions and tokenizer structure only.
This module never loads a transformer model or reads scientific responses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    TEMPLATES,
    validate_e29_11_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_prompt_panel import (
    e29_14b_prompt_panel,
    prior_e29_8_to_e29_14a_group_words,
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


def prior_e29_8_to_e29_14c_group_words() -> frozenset[str]:
    """Return the full lexical exclusion set through E29.14c.

    E29.14c reused E29.14b's exact prompt panel to isolate component scale, so
    adding E29.14b once gives the exact union through E29.14c.
    """
    return frozenset(
        set(prior_e29_8_to_e29_14a_group_words()) | _group_words(e29_14b_prompt_panel())
    )


def e29_14d_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Return twelve fresh context pairs over 96 distinct group words."""
    groups = (
        ("ant", "bee", "fly", "worm"),
        ("pumpkin", "spinach", "cabbage", "lettuce"),
        ("balcony", "porch", "attic", "basement"),
        ("February", "September", "November", "December"),
        ("aluminum", "titanium", "brass", "chrome"),
        ("tornado", "hurricane", "volcano", "earthquake"),
        ("kidney", "liver", "lung", "stomach"),
        ("architect", "dentist", "surgeon", "accountant"),
        ("shrimp", "salmon", "tuna", "lobster"),
        ("theater", "stadium", "restaurant", "university"),
        ("gown", "sleeve", "collar", "pocket"),
        ("aircraft", "motorcycle", "helicopter", "ambulance"),
        ("scissors", "ladder", "torch", "razor"),
        ("surfing", "fencing", "sailing", "wrestling"),
        ("yesterday", "tonight", "weekly", "monthly"),
        ("smile", "laugh", "shout", "whisper"),
        ("curious", "gentle", "clever", "honest"),
        ("soap", "glue", "paint", "gravel"),
        ("plateau", "marsh", "glacier", "waterfall"),
        ("comedy", "drama", "ballet", "poetry"),
        ("bride", "groom", "widow", "orphan"),
        ("fifteen", "sixteen", "seventeen", "eighteen"),
        ("tall", "short", "heavy", "thin"),
        ("laptop", "tablet", "camera", "radio"),
    )
    return tuple(
        _induction_record(
            TEMPLATES[context_index // 3],
            groups[2 * context_index],
            groups[2 * context_index + 1],
        )
        for context_index in range(12)
    )


def validate_e29_14d_prompt_panel(
    tokenizer: Any,
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Validate novelty through E29.14c and pinned-tokenizer alignment."""
    return validate_e29_11_prompt_panel(
        tokenizer,
        records,
        prior_group_words=prior_e29_8_to_e29_14c_group_words(),
    )


__all__ = [
    "e29_14d_prompt_panel",
    "prior_e29_8_to_e29_14c_group_words",
    "validate_e29_14d_prompt_panel",
]
