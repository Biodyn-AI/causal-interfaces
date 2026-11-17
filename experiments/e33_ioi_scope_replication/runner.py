"""CPU-float64 GPT-2-small runner for the sealed E33.1 panel."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from experiments.e33_ioi_scope_replication.protocol import (
    COMPONENT_COUNT,
    CONDITIONS,
    CONDITION_IDS,
    LAYER_COUNT,
    matched_static_scope,
)


@dataclass(frozen=True, eq=False)
class PreparedContext:
    """Clean endpoints and all donor/recipient sublayer updates for one pair."""

    record: Mapping[str, str]
    base_input_ids: torch.Tensor = field(repr=False)
    donor_input_ids: torch.Tensor = field(repr=False)
    base_updates: tuple[torch.Tensor, ...] = field(repr=False)
    donor_updates: tuple[torch.Tensor, ...] = field(repr=False)
    base_logits: torch.Tensor = field(repr=False)
    donor_logits: torch.Tensor = field(repr=False)
    base_target_token_id: int
    donor_target_token_id: int
    position_scopes: Mapping[str, torch.Tensor] = field(repr=False)
    _runner: Any = field(repr=False)


def _target_token_id(tokenizer: Any, target: str) -> int:
    token_ids = tokenizer.encode(" " + target, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"target is not one leading-space token: {target!r}")
    return int(token_ids[0])


def _jensen_shannon(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return finite Jensen-Shannon divergence for two full-vocabulary logits."""
    first_log = torch.log_softmax(first.double(), dim=-1)
    second_log = torch.log_softmax(second.double(), dim=-1)
    first_probability = first_log.exp()
    second_probability = second_log.exp()
    mixture = 0.5 * (first_probability + second_probability)
    mixture_log = mixture.log()
    divergence = 0.5 * (
        (first_probability * (first_log - mixture_log)).sum()
        + (second_probability * (second_log - mixture_log)).sum()
    )
    value = float(divergence)
    if not math.isfinite(value) or value < -1e-12:
        raise RuntimeError("invalid Jensen-Shannon divergence")
    return max(value, 0.0)


class IOIPositionScopeRunner:
    """Patch any fixed GPT-2-small sublayer set at fixed token scopes."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        if model.training:
            raise ValueError("model must be in evaluation mode")
        state = (*tuple(model.parameters()), *tuple(model.buffers()))
        if not state or any(tensor.device.type != "cpu" for tensor in state):
            raise ValueError("E33.1 requires a CPU model")
        if any(
            tensor.dtype != torch.float64
            for tensor in state
            if tensor.is_floating_point()
        ):
            raise ValueError("E33.1 requires float64 model state")
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            raise TypeError("model must expose Hugging Face GPT-2 blocks")
        if len(model.transformer.h) != LAYER_COUNT:
            raise ValueError("E33.1 requires GPT-2 small with twelve layers")
        self.model = model
        self.tokenizer = tokenizer
        self.modules = tuple(
            module
            for block in model.transformer.h
            for module in (block.attn.c_proj, block.mlp.c_proj)
        )
        if len(self.modules) != COMPONENT_COUNT:
            raise RuntimeError("unexpected GPT-2 component count")

    def _input_ids(self, prompt: str) -> torch.Tensor:
        encoded: SimpleNamespace = self.tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids.detach().cpu()

    def _capture(
        self, input_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for component, module in enumerate(self.modules):

            def capture(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                index: int = component,
            ) -> None:
                captured[index] = output[0].detach().clone()

            handles.append(module.register_forward_hook(capture))
        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0, -1].detach().clone()
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(range(COMPONENT_COUNT)):
            raise RuntimeError("failed to capture every sublayer update")
        if not bool(logits.isfinite().all()):
            raise RuntimeError("clean logits are non-finite")
        return tuple(captured[index] for index in range(COMPONENT_COUNT)), logits

    def prepare_context(self, record: Mapping[str, Any]) -> PreparedContext:
        """Acquire both clean endpoints before any patched evaluation."""
        required = {
            "context_id",
            "template",
            "base_prompt",
            "donor_prompt",
            "base_target",
            "donor_target",
        }
        if not isinstance(record, Mapping) or not required <= set(record):
            raise ValueError("prompt record omits required fields")
        frozen_record = {key: str(value) for key, value in record.items()}
        base_ids = self._input_ids(frozen_record["base_prompt"])
        donor_ids = self._input_ids(frozen_record["donor_prompt"])
        if base_ids.ndim != 2 or base_ids.shape[0] != 1 or base_ids.shape != donor_ids.shape:
            raise ValueError("base and donor prompts must be aligned one-row tensors")
        differing = base_ids[0] != donor_ids[0]
        if int(differing.sum()) != 3:
            raise ValueError("each E33.1 pair must differ at exactly three positions")
        shared = ~differing
        scopes = {
            "all_positions": torch.ones_like(differing),
            "differing_lexical_positions": differing,
            "shared_static_positions": shared,
            "matched_static_a": matched_static_scope(
                frozen_record["context_id"], shared, replicate="a", count=3
            ),
            "matched_static_b": matched_static_scope(
                frozen_record["context_id"], shared, replicate="b", count=3
            ),
        }
        base_target = _target_token_id(self.tokenizer, frozen_record["base_target"])
        donor_target = _target_token_id(self.tokenizer, frozen_record["donor_target"])
        if base_target == donor_target:
            raise ValueError("counterfactual targets must differ")
        base_updates, base_logits = self._capture(base_ids)
        donor_updates, donor_logits = self._capture(donor_ids)
        return PreparedContext(
            record=MappingProxyType(frozen_record),
            base_input_ids=base_ids.clone(),
            donor_input_ids=donor_ids.clone(),
            base_updates=base_updates,
            donor_updates=donor_updates,
            base_logits=base_logits,
            donor_logits=donor_logits,
            base_target_token_id=base_target,
            donor_target_token_id=donor_target,
            position_scopes=MappingProxyType(
                {name: scope.clone() for name, scope in scopes.items()}
            ),
            _runner=self,
        )

    def _patched_logits(
        self,
        prepared: PreparedContext,
        *,
        direction: str,
        condition: Mapping[str, Any],
    ) -> torch.Tensor:
        if prepared._runner is not self:
            raise ValueError("prepared context belongs to another runner")
        if direction == "forward":
            input_ids = prepared.base_input_ids
            source_updates = prepared.donor_updates
        elif direction == "reverse":
            input_ids = prepared.donor_input_ids
            source_updates = prepared.base_updates
        else:
            raise ValueError("direction must be forward or reverse")
        components = tuple(int(index) for index in condition["components"])
        scope = prepared.position_scopes[str(condition["scope"])][None, :, None]
        patch_mode = str(condition["patch_mode"])
        handles = []
        for component in components:
            module = self.modules[component]
            source = source_updates[component][None]

            def patch(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                source_output: torch.Tensor = source,
                position_scope: torch.Tensor = scope,
                mode: str = patch_mode,
            ) -> torch.Tensor:
                delta = source_output.to(output) - output
                if mode == "half_roll_delta":
                    delta = torch.roll(
                        delta, shifts=delta.shape[-1] // 2, dims=-1
                    )
                elif mode != "source_replace":
                    raise ValueError(f"unknown patch mode: {mode}")
                return output + position_scope.to(output) * delta

            handles.append(module.register_forward_hook(patch))
        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0, -1].detach().clone()
        finally:
            for handle in handles:
                handle.remove()
        if not bool(logits.isfinite().all()):
            raise RuntimeError("patched logits are non-finite")
        return logits

    @staticmethod
    def _direction_endpoints(
        prepared: PreparedContext, direction: str
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, float]:
        if direction == "forward":
            recipient_logits = prepared.base_logits
            source_logits = prepared.donor_logits
            recipient_target = prepared.base_target_token_id
            source_target = prepared.donor_target_token_id
        elif direction == "reverse":
            recipient_logits = prepared.donor_logits
            source_logits = prepared.base_logits
            recipient_target = prepared.donor_target_token_id
            source_target = prepared.base_target_token_id
        else:
            raise ValueError("direction must be forward or reverse")
        recipient_margin = float(
            recipient_logits[source_target] - recipient_logits[recipient_target]
        )
        source_margin = float(
            source_logits[source_target] - source_logits[recipient_target]
        )
        endpoint = source_margin - recipient_margin
        if not math.isfinite(endpoint) or endpoint <= 0.0:
            raise RuntimeError("clean counterfactual endpoint must be positive")
        return (
            recipient_logits,
            source_logits,
            recipient_target,
            source_target,
            endpoint,
        )

    def clean_metrics(self, prepared: PreparedContext) -> dict[str, Any]:
        """Return clean viability and full-distribution denominators."""
        records = {}
        for direction in ("forward", "reverse"):
            recipient, source, recipient_target, source_target, endpoint = (
                self._direction_endpoints(prepared, direction)
            )
            recipient_margin = float(
                recipient[source_target] - recipient[recipient_target]
            )
            source_margin = float(source[source_target] - source[recipient_target])
            baseline_js = _jensen_shannon(recipient, source)
            if baseline_js <= 1e-12:
                raise RuntimeError("clean endpoint distributions are indistinguishable")
            records[direction] = {
                "recipient_counterfactual_margin": recipient_margin,
                "source_counterfactual_margin": source_margin,
                "endpoint": endpoint,
                "recipient_pairwise_correct": recipient_margin < 0.0,
                "source_pairwise_correct": source_margin > 0.0,
                "recipient_target_top1": int(recipient.argmax()) == recipient_target,
                "source_target_top1": int(source.argmax()) == source_target,
                "clean_endpoint_js": baseline_js,
            }
        return records

    def evaluate_context(self, prepared: PreparedContext) -> dict[str, Any]:
        """Evaluate all fixed conditions in both directions."""
        clean = self.clean_metrics(prepared)
        directional = {}
        for direction in ("forward", "reverse"):
            recipient, source, recipient_target, source_target, endpoint = (
                self._direction_endpoints(prepared, direction)
            )
            clean_margin = float(
                recipient[source_target] - recipient[recipient_target]
            )
            clean_js = float(clean[direction]["clean_endpoint_js"])
            rows = {}
            for condition in CONDITIONS:
                logits = self._patched_logits(
                    prepared, direction=direction, condition=condition
                )
                patched_margin = float(logits[source_target] - logits[recipient_target])
                rows[str(condition["condition_id"])] = {
                    "normalized_transfer": (patched_margin - clean_margin) / endpoint,
                    "js_fidelity": 1.0 - _jensen_shannon(logits, source) / clean_js,
                    "counterfactual_pairwise_win": patched_margin > 0.0,
                    "counterfactual_target_top1": int(logits.argmax()) == source_target,
                    "patched_counterfactual_margin": patched_margin,
                }
            if tuple(rows) != CONDITION_IDS:
                raise RuntimeError("condition order changed during evaluation")
            directional[direction] = rows
        return {
            "context_id": prepared.record["context_id"],
            "record": dict(prepared.record),
            "token_count": int(prepared.base_input_ids.shape[1]),
            "position_scopes": {
                name: torch.where(scope)[0].tolist()
                for name, scope in prepared.position_scopes.items()
            },
            "clean": clean,
            "directions": directional,
        }


__all__ = ["IOIPositionScopeRunner", "PreparedContext", "_jensen_shannon"]
