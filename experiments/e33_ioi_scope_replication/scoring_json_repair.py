"""Outcome-blind JSON-order adapter for the sealed E33.1 scorer.

The write-once label serializer used ``sort_keys=True``, so JSON dictionaries
reload in alphabetical rather than protocol order.  The original scorer
correctly rejects changed condition order.  This adapter reconstructs the
registered order from ``CONDITION_IDS`` and delegates every calculation and
gate to the unchanged sealed scorer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from experiments.e33_ioi_scope_replication.protocol import CONDITION_IDS, DIRECTIONS
from experiments.e33_ioi_scope_replication.scoring import score_context_records


def restore_registered_condition_order(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    restored = []
    for source_record in records:
        record = deepcopy(dict(source_record))
        directions = record.get("directions")
        if not isinstance(directions, Mapping) or set(directions) != set(DIRECTIONS):
            raise ValueError("a label record is missing a registered direction")
        ordered_directions = {}
        for direction in DIRECTIONS:
            conditions = directions[direction]
            if not isinstance(conditions, Mapping) or set(conditions) != set(
                CONDITION_IDS
            ):
                raise ValueError("a label record changed registered conditions")
            ordered_directions[direction] = {
                condition_id: deepcopy(conditions[condition_id])
                for condition_id in CONDITION_IDS
            }
        record["directions"] = ordered_directions
        restored.append(record)
    return tuple(restored)


def score_json_context_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return score_context_records(restore_registered_condition_order(records))


__all__ = ["restore_registered_condition_order", "score_json_context_records"]

