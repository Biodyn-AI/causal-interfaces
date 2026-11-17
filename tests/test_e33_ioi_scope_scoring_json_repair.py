from __future__ import annotations

import json

from experiments.e33_ioi_scope_replication.scoring_json_repair import (
    score_json_context_records,
)
from test_e33_ioi_scope_scoring import _passing_contexts


def test_sorted_json_roundtrip_preserves_registered_scientific_score() -> None:
    records = json.loads(json.dumps(_passing_contexts(), sort_keys=True))
    score = score_json_context_records(records)
    assert score["all_registered_gates_passed"] is True

