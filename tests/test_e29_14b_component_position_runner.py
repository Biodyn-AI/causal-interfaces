from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_runner import (
    Float64GPT2SublayerPatchRunner,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_runner import (
    Float64ComponentPositionRunner,
)


class _Projection(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.scale * hidden


class _Block(torch.nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.c_proj = _Projection(0.0010 * (layer + 1))
        self.mlp = torch.nn.Module()
        self.mlp.c_proj = _Projection(0.0007 * (layer + 1))


class _NonlinearToyModel(torch.nn.Module):
    """A 24-layer causal-patching toy with cross-position dependence."""

    def __init__(self, *, layers: int = 24) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_layer=layers)
        self.embedding = torch.nn.Embedding(32, 1, dtype=torch.float64)
        weights = torch.zeros(32, 1, dtype=torch.float64)
        weights[3] = 0.2
        weights[[1, 4, 5, 6]] = 0.6
        weights[[2, 7, 8, 9]] = -1.1
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList(
            [_Block(layer) for layer in range(layers)]
        )

    def forward(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        for block in self.transformer.h:
            # The shared token's update depends on lexical tokens. Consequently,
            # all-position and differing-position patches are genuinely distinct.
            attention_input = hidden + 0.05 * hidden.mean(dim=1, keepdim=True)
            hidden = hidden + block.attn.c_proj(attention_input)
            mlp_input = hidden + 0.03 * hidden.mean(dim=1, keepdim=True)
            hidden = hidden + block.mlp.c_proj(mlp_input)
        score = hidden.sum(dim=(1, 2))
        logits = torch.zeros(
            len(input_ids), input_ids.shape[1], 32, dtype=torch.float64
        )
        logits[:, :, 1] = score[:, None].pow(3)
        logits[:, :, 2] = -score[:, None].pow(3)
        return SimpleNamespace(logits=logits)


class _Tokenizer:
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


def _record(*, reverse: bool = False) -> dict[str, str]:
    if reverse:
        return {
            "template": "toy",
            "base_prompt": "donor prompt",
            "donor_prompt": "base prompt",
            "base_target": "donor",
            "donor_target": "base",
        }
    return {
        "template": "toy",
        "base_prompt": "base prompt",
        "donor_prompt": "donor prompt",
        "base_target": "base",
        "donor_target": "donor",
    }


def _masks() -> torch.Tensor:
    masks = torch.zeros(7, 48, dtype=torch.float64)
    masks[1, 0] = 1
    masks[2, 1] = 1
    masks[3, 17] = 1
    masks[4, 47] = 1
    masks[5, [1, 5, 20, 47]] = 1
    masks[6] = 1
    return masks


def test_prepare_context_captures_all_components_logits_endpoint_and_scopes() -> None:
    runner = Float64ComponentPositionRunner(
        _NonlinearToyModel().eval(), _Tokenizer(), batch_size=4
    )

    prepared = runner.prepare_context(_record())

    assert runner.component_count == 48
    assert len(prepared.base_updates) == len(prepared.donor_updates) == 48
    assert all(update.shape == (5, 1) for update in prepared.base_updates)
    assert all(update.shape == (5, 1) for update in prepared.donor_updates)
    assert prepared.base_logits.shape == prepared.donor_logits.shape == (32,)
    assert prepared.clean_forward_margin < 0.0
    assert prepared.clean_reverse_margin < 0.0
    assert prepared.clean_endpoint > 0.0
    assert tuple(prepared.position_scopes) == (
        "all_positions",
        "differing_lexical_positions",
    )
    assert prepared.position_scopes["all_positions"].tolist() == [True] * 5
    assert prepared.position_scopes["differing_lexical_positions"].tolist() == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert prepared.clean_metadata["component_count"] == 48
    assert prepared.clean_metadata["patch_direction"] == "donor_to_base_only"


def test_alpha_no_op_and_mask_validation() -> None:
    runner = Float64ComponentPositionRunner(
        _NonlinearToyModel().eval(), _Tokenizer(), batch_size=3
    )
    prepared = runner.prepare_context(_record())
    masks = _masks()

    zero_mask = runner.responses_from_prepared(
        prepared, masks[:1], position_scope="all_positions"
    )
    alpha_zero = runner.responses_from_prepared(
        prepared, masks[-1:], position_scope="all_positions", alpha=0.0
    )
    alpha_half = runner.responses_from_prepared(
        prepared, masks[-1:], position_scope="all_positions", alpha=0.5
    )

    assert torch.equal(zero_mask, torch.zeros_like(zero_mask))
    assert torch.equal(alpha_zero, torch.zeros_like(alpha_zero))
    assert alpha_half.shape == (1,)
    assert bool(alpha_half.isfinite().all())

    with pytest.raises(ValueError, match="shape"):
        runner.responses_from_prepared(
            prepared, torch.zeros(1, 47), position_scope="all_positions"
        )
    with pytest.raises(ValueError, match="at least one row"):
        runner.responses_from_prepared(
            prepared, torch.zeros(0, 48), position_scope="all_positions"
        )
    invalid = torch.zeros(1, 48)
    invalid[0, 3] = 0.5
    with pytest.raises(ValueError, match="binary"):
        runner.responses_from_prepared(
            prepared, invalid, position_scope="all_positions"
        )
    with pytest.raises(ValueError, match="alpha"):
        runner.responses_from_prepared(
            prepared, masks, position_scope="all_positions", alpha=float("nan")
        )
    with pytest.raises(ValueError, match="alpha"):
        runner.responses_from_prepared(
            prepared, masks, position_scope="all_positions", alpha=1.01
        )
    with pytest.raises(ValueError, match="position_scope"):
        runner.responses_from_prepared(prepared, masks, position_scope="static")


def test_all_position_alpha_one_matches_legacy_runner() -> None:
    model = _NonlinearToyModel().eval()
    tokenizer = _Tokenizer()
    new_runner = Float64ComponentPositionRunner(model, tokenizer, batch_size=3)
    legacy_runner = Float64GPT2SublayerPatchRunner(model, tokenizer, batch_size=3)

    new_prepared = new_runner.prepare_context(_record())
    legacy_prepared = legacy_runner.prepare_context(_record())
    actual = new_runner.responses_from_prepared(
        new_prepared, _masks(), position_scope="all_positions"
    )
    expected, _ = legacy_runner.responses_from_prepared(legacy_prepared, _masks())

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_paired_order_and_forward_only_nonlinear_asymmetry() -> None:
    runner = Float64ComponentPositionRunner(
        _NonlinearToyModel().eval(), _Tokenizer(), batch_size=3
    )
    prepared = runner.prepare_context(_record())
    masks = _masks()[2:]

    paired = runner.paired_responses_from_prepared(prepared, masks)

    assert tuple(paired) == ("all_positions", "differing_lexical_positions")
    assert (
        paired["all_positions"].shape
        == paired["differing_lexical_positions"].shape
        == (len(masks),)
    )
    assert not torch.equal(
        paired["all_positions"], paired["differing_lexical_positions"]
    )

    reverse_prepared = runner.prepare_context(_record(reverse=True))
    forward = runner.responses_from_prepared(
        prepared, masks[:1], position_scope="all_positions"
    )
    reverse = runner.responses_from_prepared(
        reverse_prepared, masks[:1], position_scope="all_positions"
    )
    assert abs(prepared.clean_forward_margin) != pytest.approx(
        abs(prepared.clean_reverse_margin)
    )
    assert forward.item() != pytest.approx(reverse.item())


def test_full_two_scope_batch_invariance_smoke() -> None:
    runner = Float64ComponentPositionRunner(
        _NonlinearToyModel().eval(), _Tokenizer(), batch_size=2
    )
    prepared = runner.prepare_context(_record())
    masks = _masks()

    smoke = runner.full_batch_invariance_smoke(
        prepared,
        masks,
        reference_batch_size=1,
        candidate_batch_size=5,
    )

    assert smoke["passed"]
    assert tuple(smoke["records"]) == (
        "all_positions",
        "differing_lexical_positions",
    )
    assert smoke["unique_mask_scope_cells"] == 2 * len(masks)
    assert smoke["physical_mask_scope_cells"] == 4 * len(masks)
    assert all(
        record["maximum_absolute_difference"] == 0.0
        for record in smoke["records"].values()
    )


def test_runner_requires_exact_gpt2_medium_component_count_and_float64() -> None:
    with pytest.raises(ValueError, match="48 components"):
        Float64ComponentPositionRunner(
            _NonlinearToyModel(layers=1).eval(), _Tokenizer(), batch_size=2
        )
    with pytest.raises(ValueError, match="float64"):
        Float64ComponentPositionRunner(
            _NonlinearToyModel().float().eval(), _Tokenizer(), batch_size=2
        )
