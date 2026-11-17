"""Pure, response-blind lexical prompt panel for prospective E29.12 work.

This module defines prompts and validates tokenizer-level preconditions only.
It never loads or queries a transformer model and does not read outcome data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_8_prompt_panel import (
    e29_4_source_prompt_panel,
    e29_8_transfer_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.e29_9_seven_gate_protocol import (
    e29_9_transfer_prompt_panel,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    TEMPLATES,
    e29_11_prompt_panel,
    validate_e29_11_prompt_panel,
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


def prior_e29_8_to_e29_11_group_words() -> frozenset[str]:
    """Return every group word used by the E29.8, E29.9, and E29.11 panels.

    E29.8 contains both its copied source panel and its fresh transfer panel,
    so both are included. E29.10 reused existing contexts and adds no panel.
    """
    panels = (
        e29_4_source_prompt_panel(),
        e29_8_transfer_prompt_panel(),
        e29_9_transfer_prompt_panel(),
        e29_11_prompt_panel(),
    )
    return frozenset(word for panel in panels for word in _group_words(panel))


def e29_12_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Return twelve wholly new lexical induction context pairs.

    The 24 groups contain 96 distinct words. Every group word was selected to
    be one GPT-2 token when preceded by a space; the validator checks that
    property against the actual pinned tokenizer before any outcomes are run.
    """
    groups = (
        ("amber", "gray", "tan", "violet"),
        ("owl", "hawk", "crane", "crow"),
        ("plum", "lime", "coconut", "tomato"),
        ("stool", "drawer", "closet", "pillow"),
        ("Friday", "Saturday", "Sunday", "weekend"),
        ("quartz", "jewel", "gem", "coal"),
        ("breeze", "frost", "hail", "mist"),
        ("elbow", "wrist", "ankle", "shoulder"),
        ("clerk", "judge", "mayor", "priest"),
        ("flour", "jam", "cereal", "beans"),
        ("castle", "temple", "palace", "cabin"),
        ("sweater", "boots", "helmet", "mask"),
        ("canoe", "ferry", "van", "yacht"),
        ("shovel", "axe", "hook", "needle"),
        ("hockey", "baseball", "boxing", "rugby"),
        ("century", "decade", "season", "era"),
        ("bend", "kick", "shake", "roll"),
        ("pride", "guilt", "courage", "patience"),
        ("linen", "velvet", "cement", "marble"),
        ("cliff", "bay", "swamp", "reef"),
        ("keyboard", "bell", "whistle", "speaker"),
        ("cousin", "nephew", "niece", "wife"),
        ("hundred", "thousand", "million", "billion"),
        ("salty", "creamy", "crispy", "juicy"),
    )
    return tuple(
        _induction_record(
            TEMPLATES[context_index // 3],
            groups[2 * context_index],
            groups[2 * context_index + 1],
        )
        for context_index in range(12)
    )


def validate_e29_12_prompt_panel(
    tokenizer: Any,
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Validate novelty, token alignment, targets, and exact next-token appends."""
    return validate_e29_11_prompt_panel(
        tokenizer,
        records,
        prior_group_words=prior_e29_8_to_e29_11_group_words(),
    )


__all__ = [
    "e29_12_prompt_panel",
    "prior_e29_8_to_e29_11_group_words",
    "validate_e29_12_prompt_panel",
]
