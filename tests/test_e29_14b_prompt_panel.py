"""Focused tests for the response-blind E29.14b lexical prompt panel."""

from __future__ import annotations

from collections import Counter

import pytest

from experiments.e29_gpt2_stable_head_interactions.e29_8_prompt_panel import (
    GPT2_MEDIUM_SNAPSHOT,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    TEMPLATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_prompt_panel import (
    e29_14a_prompt_panel,
    prior_e29_8_to_e29_13_group_words,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_prompt_panel import (
    e29_14b_prompt_panel,
    prior_e29_8_to_e29_14a_group_words,
    validate_e29_14b_prompt_panel,
)


def _words(records: tuple[dict[str, object], ...]) -> set[str]:
    return {
        str(word)
        for record in records
        for side in ("base_words", "donor_words")
        for word in record[side]
    }


def test_panel_has_12_balanced_contexts_and_exact_new_groups() -> None:
    panel = e29_14b_prompt_panel()
    groups = tuple(
        tuple(record[side])
        for record in panel
        for side in ("base_words", "donor_words")
    )
    expected_groups = (
        ("pink", "silver", "black", "grey"),
        ("cat", "dog", "cow", "pig"),
        ("egg", "milk", "tea", "coffee"),
        ("door", "wall", "floor", "roof"),
        ("April", "May", "July", "October"),
        ("steel", "clay", "sand", "cloth"),
        ("sun", "ice", "fire", "water"),
        ("back", "skin", "hair", "lip"),
        ("teacher", "student", "soldier", "police"),
        ("fish", "chicken", "turkey", "ham"),
        ("park", "street", "bridge", "road"),
        ("hat", "vest", "suit", "uniform"),
        ("cart", "cab", "tractor", "bicycle"),
        ("wrench", "bolt", "clamp", "rod"),
        ("swimming", "running", "cycling", "skating"),
        ("early", "late", "soon", "never"),
        ("open", "close", "start", "stop"),
        ("happy", "sad", "calm", "wise"),
        ("asphalt", "mud", "chalk", "plaster"),
        ("city", "town", "village", "country"),
        ("song", "voice", "drums", "music"),
        ("husband", "grandma", "childhood", "relative"),
        ("eleven", "twelve", "thirteen", "fourteen"),
        ("big", "small", "fast", "slow"),
    )

    assert len(panel) == 12
    assert groups == expected_groups
    assert Counter(record["template"] for record in panel) == Counter(
        {template: 3 for template in TEMPLATES}
    )
    assert len(_words(panel)) == 96
    assert _words(panel).isdisjoint(prior_e29_8_to_e29_14a_group_words())
    assert all(record["base_target"] == record["base_words"][3] for record in panel)
    assert all(
        record["donor_target"] == record["donor_words"][3] for record in panel
    )


def test_prior_exclusion_set_is_exact_union_through_e29_14a() -> None:
    expected = set(prior_e29_8_to_e29_13_group_words()) | _words(
        e29_14a_prompt_panel()
    )

    assert prior_e29_8_to_e29_14a_group_words() == frozenset(expected)


def test_validator_rejects_prior_overlap_before_tokenization() -> None:
    panel = list(e29_14b_prompt_panel())
    prior_word = min(prior_e29_8_to_e29_14a_group_words())
    panel[0] = {
        **panel[0],
        "base_words": (*panel[0]["base_words"][:3], prior_word),
        "base_target": prior_word,
    }

    with pytest.raises(ValueError, match="overlap prior E29 panels"):
        validate_e29_14b_prompt_panel(object(), tuple(panel))


@pytest.mark.skipif(
    not (GPT2_MEDIUM_SNAPSHOT / "tokenizer.json").is_file(),
    reason="pinned GPT-2-medium tokenizer is unavailable",
)
def test_pinned_tokenizer_validates_every_alignment_condition() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        GPT2_MEDIUM_SNAPSHOT,
        local_files_only=True,
    )

    checks = validate_e29_14b_prompt_panel(tokenizer, e29_14b_prompt_panel())

    assert [record["token_length"] for record in checks] == (
        [13] * 3 + [12] * 3 + [10] * 3 + [12] * 3
    )
    assert all(record["target_positions_aligned"] for record in checks)
    for record in checks:
        base = record["sides"]["base"]
        donor = record["sides"]["donor"]
        assert base["demonstration_target_position"] == donor[
            "demonstration_target_position"
        ]
        assert base["prediction_position"] == donor["prediction_position"]
        assert base["target_token_id"] != donor["target_token_id"]
        for side in (base, donor):
            assert side["next_token_append_valid"] is True
            assert set(side["word_token_lengths"].values()) == {1}
