"""Response-blind prompt panels for the E29.8 GPT-2 scale transfer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

GPT2_MEDIUM_SNAPSHOT = Path(
    "/Users/ihorkendiukhov/.cache/huggingface/hub/"
    "models--openai-community--gpt2-medium/snapshots/"
    "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
)

E29_4_GROUPS = (
    ("red", "blue", "green", "yellow"),
    ("cat", "dog", "bird", "fish"),
    ("one", "two", "three", "four"),
    ("sun", "moon", "star", "sky"),
    ("apple", "pear", "peach", "grape"),
    ("chair", "table", "lamp", "clock"),
    ("river", "ocean", "lake", "beach"),
    ("king", "queen", "prince", "knight"),
    ("white", "orange", "purple", "brown"),
    ("horse", "sheep", "duck", "mouse"),
    ("five", "six", "seven", "eight"),
    ("north", "south", "east", "west"),
)
E29_5_GROUP_WORDS = frozenset(
    (
        "black", "silver", "gold", "pink", "lion", "tiger", "bear", "wolf",
        "spring", "summer", "autumn", "winter", "bread", "cheese", "milk",
        "rice", "stone", "wood", "glass", "steel", "happy", "sad", "angry",
        "calm", "walk", "run", "jump", "swim", "small", "large", "short",
        "long", "coffee", "tea", "juice", "water", "car", "bus", "train",
        "plane", "shirt", "shoe", "coat", "hat", "book", "paper", "pen",
        "desk", "morning", "evening", "night", "day", "hot", "cold", "warm",
        "cool", "open", "close", "start", "stop", "left", "right", "up", "down",
    )
)
E29_6_GROUP_WORDS = frozenset(
    (
        "rock", "sand", "clay", "mud", "rain", "snow", "wind", "cloud", "hand",
        "foot", "head", "arm", "door", "wall", "floor", "roof", "love", "hate",
        "fear", "hope", "fast", "slow", "high", "low", "young", "old", "new",
        "ancient", "city", "town", "village", "farm", "music", "song", "dance",
        "art", "phone", "radio", "screen", "light", "tree", "grass", "leaf",
        "flower", "doctor", "nurse", "teacher", "student", "fire", "earth", "air",
        "space", "box", "bag", "cup", "bowl", "smile", "laugh", "cry", "sleep",
        "road", "street", "bridge", "park",
    )
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


def e29_8_transfer_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Return eight fresh prompt pairs for prospective GPT-2-medium transfer."""
    groups = (
        ("copper", "bronze", "iron", "zinc"),
        ("fox", "deer", "goat", "frog"),
        ("soup", "meat", "cake", "fruit"),
        ("sit", "stand", "read", "write"),
        ("knife", "spoon", "plate", "fork"),
        ("kitchen", "bedroom", "garden", "office"),
        ("lawyer", "pilot", "chef", "guard"),
        ("eye", "nose", "mouth", "ear"),
        ("proud", "shy", "brave", "kind"),
        ("hill", "field", "forest", "sea"),
        ("bike", "ship", "truck", "boat"),
        ("dress", "pants", "socks", "belt"),
        ("week", "month", "year", "hour"),
        ("math", "science", "history", "language"),
        ("movie", "game", "story", "news"),
        ("bed", "sofa", "mirror", "window"),
    )
    templates = (
        "Sequence replay: {a} {b} {c} {d} => {a} {b} {c}",
        "Continue this loop: {a} {b} {c} {d}; again {a} {b} {c}",
    )
    return tuple(
        _induction_record(
            templates[context_index // 4],
            groups[2 * context_index],
            groups[2 * context_index + 1],
        )
        for context_index in range(8)
    )


def e29_4_source_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Expose defensive copies of the exact eight E29.4 source contexts."""
    test_groups = E29_4_GROUPS[8:]
    templates = (
        "Pattern: {a} {b} {c} {d} | {a} {b} {c}",
        "Repeat {a} {b} {c} {d}; then {a} {b} {c}",
    )
    records = tuple(
        _induction_record(
            template,
            group,
            test_groups[(index + 1) % len(test_groups)],
        )
        for template in templates
        for index, group in enumerate(test_groups)
    )
    return deepcopy(records)


def validate_fresh_prompt_novelty(
    records: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    """Fail unless all 16 groups and 64 words are new relative to E29.4--6."""
    groups = [
        tuple(str(word) for word in record[key])
        for record in records
        for key in ("base_words", "donor_words")
    ]
    fresh_words = {word for group in groups for word in group}
    old_e29_4_words = {word for group in E29_4_GROUPS for word in group}
    overlap = sorted(
        fresh_words & (old_e29_4_words | E29_5_GROUP_WORDS | E29_6_GROUP_WORDS)
    )
    if len(groups) != 16 or len(set(groups)) != 16 or len(fresh_words) != 64:
        raise ValueError("fresh panel must contain 16 distinct four-word groups")
    if overlap:
        raise ValueError(f"fresh prompt words overlap E29.4--E29.6: {overlap}")
    return {
        "fresh_group_count": len(groups),
        "fresh_unique_word_count": len(fresh_words),
        "overlap_with_e29_4_to_e29_6": overlap,
    }


def validate_prompt_panel(
    tokenizer: Any, records: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    """Validate all token-level preconditions without querying model outcomes."""
    checks = []
    for context_index, record in enumerate(records):
        prompt_ids = {
            side: tokenizer.encode(
                record[f"{side}_prompt"], add_special_tokens=False
            )
            for side in ("base", "donor")
        }
        if len(prompt_ids["base"]) != len(prompt_ids["donor"]):
            raise ValueError(
                f"context {context_index} base and donor token lengths differ"
            )

        side_checks = {}
        for side in ("base", "donor"):
            target = record[f"{side}_target"]
            leading_target_ids = tokenizer.encode(
                " " + target, add_special_tokens=False
            )
            if len(leading_target_ids) != 1:
                raise ValueError(
                    f"context {context_index} {side} target is not one "
                    "leading-space token"
                )
            word_token_lengths = {
                word: len(
                    tokenizer.encode(" " + word, add_special_tokens=False)
                )
                for word in record[f"{side}_words"]
            }
            if set(word_token_lengths.values()) != {1}:
                raise ValueError(
                    f"context {context_index} {side} contains a split group word"
                )

            target_token_id = leading_target_ids[0]
            target_positions = [
                position
                for position, token_id in enumerate(prompt_ids[side])
                if token_id == target_token_id
            ]
            if len(target_positions) != 1:
                raise ValueError(
                    f"context {context_index} {side} target does not have one "
                    "unambiguous demonstration position"
                )
            extended_ids = tokenizer.encode(
                record[f"{side}_prompt"] + " " + target,
                add_special_tokens=False,
            )
            next_token_append_valid = extended_ids == [
                *prompt_ids[side],
                target_token_id,
            ]
            if not next_token_append_valid:
                raise ValueError(
                    f"context {context_index} {side} target does not append at "
                    "the next-token prediction position"
                )
            side_checks[side] = {
                "target": target,
                "target_token_id": int(target_token_id),
                "leading_space_token_length": len(leading_target_ids),
                "demonstration_target_position": target_positions[0],
                "prediction_position": len(prompt_ids[side]),
                "next_token_append_valid": next_token_append_valid,
                "word_token_lengths": word_token_lengths,
            }

        base_position = side_checks["base"]["demonstration_target_position"]
        donor_position = side_checks["donor"]["demonstration_target_position"]
        if base_position != donor_position:
            raise ValueError(
                f"context {context_index} demonstration target positions differ"
            )
        if side_checks["base"]["target_token_id"] == side_checks["donor"][
            "target_token_id"
        ]:
            raise ValueError(f"context {context_index} base and donor targets coincide")
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
