from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    FORWARD_CONDITIONS,
    REVERSE_CONDITIONS,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_runner import (
    Float64PositionDoseRunner,
)


class _Projection(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.scale * hidden


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.c_proj = _Projection(0.1)
        self.mlp = torch.nn.Module()
        self.mlp.c_proj = _Projection(0.9)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_layer=1)
        self.embedding = torch.nn.Embedding(16, 1, dtype=torch.float64)
        weights = torch.zeros(16, 1, dtype=torch.float64)
        weights[[1, 4, 5, 6]] = 1.0
        weights[[2, 7, 8, 9]] = -1.0
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([_Block()])

    def forward(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        attention_update = self.transformer.h[0].attn.c_proj(hidden)
        residual = hidden + attention_update
        mlp_update = self.transformer.h[0].mlp.c_proj(residual)
        final = residual + mlp_update
        score = final.sum(dim=(1, 2))
        logits = torch.zeros(
            len(input_ids), input_ids.shape[1], 16, dtype=torch.float64
        )
        logits[:, :, 1] = score[:, None]
        logits[:, :, 2] = -score[:, None]
        return SimpleNamespace(logits=logits)


class _AsymmetricFakeModel(_FakeModel):
    """Nonlinear readout makes the two interchange directions distinguishable."""

    def forward(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        attention_update = self.transformer.h[0].attn.c_proj(hidden)
        residual = hidden + attention_update
        mlp_update = self.transformer.h[0].mlp.c_proj(residual)
        score = (residual + mlp_update).sum(dim=(1, 2))
        logits = torch.zeros(
            len(input_ids), input_ids.shape[1], 16, dtype=torch.float64
        )
        logits[:, :, 1] = score[:, None].pow(3)
        logits[:, :, 2] = -score[:, None].pow(3)
        return SimpleNamespace(logits=logits)


class _FakeTokenizer:
    prompt_ids: ClassVar[dict[str, list[int]]] = {
        "base prompt": [3, 4, 1, 5, 6],
        "donor prompt": [3, 7, 2, 8, 9],
    }
    targets: ClassVar[dict[str, int]] = {"base": 1, "donor": 2}

    def __call__(self, prompt: str, *, return_tensors: str) -> SimpleNamespace:
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([self.prompt_ids[prompt]]))

    def encode(self, target: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.targets[target.strip()]]


def _record() -> dict[str, str]:
    return {
        "template": "test",
        "base_prompt": "base prompt",
        "donor_prompt": "donor prompt",
        "base_target": "base",
        "donor_target": "donor",
    }


def _condition_index(conditions: tuple[dict[str, object], ...], name: str) -> int:
    return next(index for index, row in enumerate(conditions) if row["name"] == name)


def test_registered_forward_reverse_and_alpha_interpolation() -> None:
    model = _FakeModel().eval()
    runner = Float64PositionDoseRunner(model, _FakeTokenizer(), batch_size=32)
    prepared = runner.prepare_context(_record())

    responses, metadata = runner.registered_responses(prepared)

    assert tuple(responses) == ("forward", "reverse")
    assert responses["forward"].shape == (len(FORWARD_CONDITIONS),)
    assert responses["reverse"].shape == (len(REVERSE_CONDITIONS),)
    assert bool(responses["forward"].isfinite().all())
    assert prepared.clean_endpoint > 0.0
    assert metadata["position_scope_indices"]["shared_static_positions"] == [0]

    all_one = responses["forward"][
        _condition_index(FORWARD_CONDITIONS, "hub_all_positions_alpha1")
    ]
    reverse_all = responses["reverse"][
        _condition_index(REVERSE_CONDITIONS, "hub_all_positions_alpha1")
    ]
    assert torch.allclose(all_one, reverse_all, atol=1e-12, rtol=0.0)
    for alpha, suffix in ((0.25, "025"), (0.5, "050"), (0.75, "075")):
        value = responses["forward"][
            _condition_index(FORWARD_CONDITIONS, f"hub_all_positions_alpha{suffix}")
        ]
        assert torch.allclose(value, alpha * all_one, atol=1e-12, rtol=0.0)

    alpha_zero = runner.responses_from_prepared(
        prepared,
        direction="forward",
        conditions=(
            {
                "name": "alpha_zero_test",
                "component": 1,
                "scope": "all_positions",
                "alpha": 0.0,
            },
        ),
    )
    assert torch.equal(alpha_zero, torch.zeros_like(alpha_zero))


def test_mixed_component_rows_are_isolated() -> None:
    runner = Float64PositionDoseRunner(
        _FakeModel().eval(), _FakeTokenizer(), batch_size=32
    )
    prepared = runner.prepare_context(_record())
    hub = next(
        row for row in FORWARD_CONDITIONS if row["name"] == "hub_all_positions_alpha1"
    )
    control = next(
        row
        for row in FORWARD_CONDITIONS
        if row["name"] == "control_all_positions_alpha1"
    )

    joint = runner.responses_from_prepared(
        prepared, direction="forward", conditions=(hub, control)
    )
    hub_only = runner.responses_from_prepared(
        prepared, direction="forward", conditions=(hub,)
    )
    control_only = runner.responses_from_prepared(
        prepared, direction="forward", conditions=(control,)
    )

    assert torch.equal(joint, torch.cat((hub_only, control_only)))


def test_forward_and_reverse_have_exact_asymmetric_sign_semantics() -> None:
    model = _AsymmetricFakeModel().eval()
    with torch.no_grad():
        model.embedding.weight[[2, 7, 8, 9]] = -2.0
    runner = Float64PositionDoseRunner(model, _FakeTokenizer(), batch_size=32)
    prepared = runner.prepare_context(_record())

    responses, _ = runner.registered_responses(prepared)
    forward_target = responses["forward"][
        _condition_index(
            FORWARD_CONDITIONS, "hub_demonstration_target_alpha1"
        )
    ]
    reverse_target = responses["reverse"][
        _condition_index(
            REVERSE_CONDITIONS, "hub_demonstration_target_alpha1"
        )
    ]

    assert prepared.clean_forward_margin == pytest.approx(-1168.554112)
    assert prepared.clean_reverse_margin == pytest.approx(-9348.432896)
    assert forward_target.item() == pytest.approx(855.372474)
    assert reverse_target.item() == pytest.approx(4149.214146)
    assert forward_target > 0.0
    assert reverse_target > forward_target


def test_batch_invariance_utility() -> None:
    runner = Float64PositionDoseRunner(
        _FakeModel().eval(), _FakeTokenizer(), batch_size=2
    )
    prepared = runner.prepare_context(_record())

    result = runner.batch_invariance_smoke(
        prepared,
        direction="forward",
        conditions=FORWARD_CONDITIONS,
        reference_batch_size=1,
        candidate_batch_size=7,
    )

    assert result["passed"]
    assert result["maximum_absolute_difference"] == 0.0

    registered = runner.registered_batch_invariance_smoke(
        prepared, reference_batch_size=1, candidate_batch_size=7
    )
    assert registered["passed"]
    assert registered["unique_condition_rows"] == 16
    assert registered["physical_condition_cells"] == 32
    assert not registered["scientific_labels_produced"]


def test_runner_rejects_non_float64_model() -> None:
    try:
        Float64PositionDoseRunner(
            _FakeModel().float().eval(), _FakeTokenizer(), batch_size=2
        )
    except ValueError as error:
        assert "float64" in str(error)
    else:
        raise AssertionError("float32 model was accepted")
