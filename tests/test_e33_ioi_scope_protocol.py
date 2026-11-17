from __future__ import annotations

import re
import hashlib
import json

import torch

from experiments.e33_ioi_scope_replication.protocol import (
    CONDITION_IDS,
    CONTEXT_COUNT,
    build_prompt_manifest,
    build_protocol,
    matched_static_scope,
    validate_prompt_manifest,
    validate_protocol,
)


class WordTokenizer:
    """Minimal aligned tokenizer sufficient for response-blind protocol tests."""

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        words = re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text)
        return [
            int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
            for word in words
        ]


def test_prompt_manifest_is_unique_aligned_and_frozen() -> None:
    manifest = build_prompt_manifest()
    checks = validate_prompt_manifest(WordTokenizer(), manifest)
    assert len(manifest) == CONTEXT_COUNT == 16
    assert len({record["context_id"] for record in manifest}) == 16
    assert len(
        {
            name
            for record in manifest
            for name in (
                record["base_name_a"],
                record["base_name_b"],
                record["donor_name_a"],
                record["donor_name_b"],
            )
        }
    ) == 64
    assert all(len(check["differing_positions"]) == 3 for check in checks)


def test_matched_static_scopes_are_deterministic_and_equal_cardinality() -> None:
    shared = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=torch.bool)
    first = matched_static_scope("context", shared, replicate="a", count=3)
    replay = matched_static_scope("context", shared, replicate="a", count=3)
    second = matched_static_scope("context", shared, replicate="b", count=3)
    assert torch.equal(first, replay)
    assert int(first.sum()) == int(second.sum()) == 3
    assert not bool((first & ~shared).any())
    assert not bool((second & ~shared).any())


def test_protocol_roundtrip_fixes_every_condition() -> None:
    protocol = build_protocol()
    assert validate_protocol(protocol) == protocol
    assert tuple(row["condition_id"] for row in protocol["conditions"]) == CONDITION_IDS
    assert json.loads(json.dumps(protocol)) == protocol

