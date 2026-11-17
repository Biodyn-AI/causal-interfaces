from __future__ import annotations

import torch

from experiments.e33_ioi_scope_replication.runner import _jensen_shannon


def test_full_distribution_divergence_is_symmetric_and_zero_on_identity() -> None:
    first = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    second = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    assert _jensen_shannon(first, first) == 0.0
    assert abs(_jensen_shannon(first, second) - _jensen_shannon(second, first)) < 1e-15
    assert _jensen_shannon(first, second) > 0.0

