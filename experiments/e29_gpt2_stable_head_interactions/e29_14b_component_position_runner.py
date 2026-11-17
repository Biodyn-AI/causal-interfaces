"""Forward-only float64 component-by-position patches for E29.14b."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    build_position_scopes,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    COMPONENT_COUNT,
    POSITION_SCOPE_ORDER,
)
from experiments.e29_gpt2_stable_head_interactions.generalized_sublayer_patch import (
    SublayerComponent,
    gpt2_sublayer_components,
)


@dataclass(frozen=True, eq=False)
class PreparedComponentPositionContext:
    """Two clean prompts and every source/recipient sublayer update."""

    base_input_ids: torch.Tensor = field(repr=False)
    donor_input_ids: torch.Tensor = field(repr=False)
    base_updates: tuple[torch.Tensor, ...] = field(repr=False)
    donor_updates: tuple[torch.Tensor, ...] = field(repr=False)
    base_logits: torch.Tensor = field(repr=False)
    donor_logits: torch.Tensor = field(repr=False)
    position_scopes: Mapping[str, torch.Tensor] = field(repr=False)
    base_target_token_id: int
    donor_target_token_id: int
    clean_forward_margin: float
    clean_reverse_margin: float
    clean_endpoint: float
    clean_logit_absolute_max: float
    clean_metadata: Mapping[str, Any]
    _runner: Any = field(repr=False)


def _target_token_id(tokenizer: Any, target: str) -> int:
    token_ids = tokenizer.encode(" " + target, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"target is not one token: {target!r} -> {token_ids}")
    return int(token_ids[0])


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool):
        raise TypeError("alpha must be a finite number in [0, 1]")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be a finite number in [0, 1]") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")
    return value


class Float64ComponentPositionRunner:
    """Patch any of GPT-2-medium's 48 sublayers at either fixed scope.

    Component ``2 * layer`` is the attention residual update and component
    ``2 * layer + 1`` is the MLP residual update. Patches always run from the
    clean donor into the base recipient; reverse-direction data require a
    separately prepared, explicitly swapped prompt record.
    """

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int) -> None:
        self.batch_size = _positive_integer(batch_size, name="batch_size")
        if model.training:
            raise ValueError("dropout must be disabled with model.eval()")
        parameters = tuple(model.parameters())
        if not parameters:
            raise ValueError("model has no parameters")
        state = (*parameters, *tuple(model.buffers()))
        if any(tensor.device.type != "cpu" for tensor in state):
            raise ValueError("E29.14b requires a CPU model")
        if any(
            tensor.dtype != torch.float64
            for tensor in state
            if tensor.is_floating_point()
        ):
            raise ValueError("E29.14b requires a float64 model")
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            raise TypeError("model must expose the GPT-2 transformer block layout")
        components = gpt2_sublayer_components(model.config)
        if len(components) != COMPONENT_COUNT:
            raise ValueError("E29.14b requires exactly 48 components (24 layers)")
        if len(model.transformer.h) != int(model.config.n_layer):
            raise ValueError("config.n_layer does not match transformer blocks")
        for layer, _kind in components:
            block = model.transformer.h[layer]
            if not hasattr(block, "attn") or not hasattr(block.attn, "c_proj"):
                raise TypeError(f"layer {layer} attention projection is unavailable")
            if not hasattr(block, "mlp") or not hasattr(block.mlp, "c_proj"):
                raise TypeError(f"layer {layer} MLP projection is unavailable")
        self.model = model
        self.tokenizer = tokenizer
        self.components = components
        self.device = parameters[0].device

    @property
    def component_count(self) -> int:
        return len(self.components)

    def _module(self, layer: int, kind: str) -> Any:
        block = self.model.transformer.h[layer]
        return block.attn.c_proj if kind == "attention" else block.mlp.c_proj

    def _input_ids(self, prompt: str) -> torch.Tensor:
        encoded: SimpleNamespace = self.tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids.detach().to(self.device)

    @staticmethod
    def _check_logits(logits: torch.Tensor) -> None:
        if not bool(logits.isfinite().all()):
            raise RuntimeError("a clean or patched logit is non-finite")
        if float(logits.abs().max()) > float(
            NUMERICAL_GATES["clean_logit_absolute_max"]
        ):
            raise RuntimeError("a clean or patched logit exceeds the magnitude gate")

    def _capture_updates_and_logits(
        self, input_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for component_index, (layer, kind) in enumerate(self.components):
            module = self._module(layer, kind)

            def capture(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                index: int = component_index,
            ) -> None:
                captured[index] = output[0].detach().clone()

            handles.append(module.register_forward_hook(capture))
        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0, -1].detach().clone()
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(range(self.component_count)):
            raise RuntimeError("failed to capture all 48 component updates")
        self._check_logits(logits)
        return (
            tuple(captured[index] for index in range(self.component_count)),
            logits,
        )

    def _validate_masks(self, masks: torch.Tensor) -> None:
        if not isinstance(masks, torch.Tensor):
            raise TypeError("masks must be a tensor with shape (rows, 48)")
        if masks.ndim != 2 or masks.shape[1] != COMPONENT_COUNT:
            raise ValueError("masks must have shape (rows, 48)")
        if len(masks) == 0:
            raise ValueError("masks must have shape (rows, 48) with at least one row")
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise ValueError("masks must be binary")

    def prepare_context(
        self, record: Mapping[str, Any]
    ) -> PreparedComponentPositionContext:
        """Cache both clean endpoints and all 48 base/donor updates."""
        required = {
            "template",
            "base_prompt",
            "donor_prompt",
            "base_target",
            "donor_target",
        }
        if not isinstance(record, Mapping) or not required <= set(record):
            raise ValueError("prompt record omits a required field")
        base_prompt = str(record["base_prompt"])
        donor_prompt = str(record["donor_prompt"])
        base_target = str(record["base_target"])
        donor_target = str(record["donor_target"])
        base_token = _target_token_id(self.tokenizer, base_target)
        donor_token = _target_token_id(self.tokenizer, donor_target)
        base_ids = self._input_ids(base_prompt)
        donor_ids = self._input_ids(donor_prompt)
        if (
            base_ids.shape != donor_ids.shape
            or base_ids.ndim != 2
            or len(base_ids) != 1
        ):
            raise ValueError("base and donor prompts must be one aligned token row")

        all_scopes = build_position_scopes(
            base_ids[0], donor_ids[0], base_target_token_id=base_token
        )
        scopes = {name: all_scopes[name] for name in POSITION_SCOPE_ORDER}
        base_updates, base_logits = self._capture_updates_and_logits(base_ids)
        donor_updates, donor_logits = self._capture_updates_and_logits(donor_ids)
        forward_margin = float(base_logits[donor_token] - base_logits[base_token])
        reverse_margin = float(donor_logits[base_token] - donor_logits[donor_token])
        if forward_margin >= 0.0 or reverse_margin >= 0.0:
            raise RuntimeError("a clean base or donor correctness gate failed")
        endpoint = -forward_margin - reverse_margin
        if endpoint <= 0.0:
            raise RuntimeError("clean endpoint must be positive")
        clean_logit_max = max(
            float(base_logits.abs().max()), float(donor_logits.abs().max())
        )
        metadata = {
            "template": str(record["template"]),
            "base_prompt": base_prompt,
            "donor_prompt": donor_prompt,
            "base_target": base_target,
            "donor_target": donor_target,
            "base_target_token_id": base_token,
            "donor_target_token_id": donor_token,
            "token_length": int(base_ids.shape[1]),
            "model_layer_count": int(self.model.config.n_layer),
            "component_count": self.component_count,
            "component_semantics": "attention_and_mlp_residual_updates",
            "patch_direction": "donor_to_base_only",
            "position_scope_order": list(POSITION_SCOPE_ORDER),
            "position_scope_indices": {
                name: [int(index) for index in torch.where(scope)[0]]
                for name, scope in scopes.items()
            },
            "clean_base_correct_minus_donor_margin": -forward_margin,
            "clean_donor_correct_minus_base_margin": -reverse_margin,
            "clean_endpoint": endpoint,
            "clean_base_logit_absolute_max": float(base_logits.abs().max()),
            "clean_donor_logit_absolute_max": float(donor_logits.abs().max()),
            "clean_logit_absolute_max": clean_logit_max,
            "clean_logits_finite": True,
            "device": "cpu",
            "model_dtype": "float64",
        }
        return PreparedComponentPositionContext(
            base_input_ids=base_ids.clone(),
            donor_input_ids=donor_ids.clone(),
            base_updates=tuple(update.clone() for update in base_updates),
            donor_updates=tuple(update.clone() for update in donor_updates),
            base_logits=base_logits.clone(),
            donor_logits=donor_logits.clone(),
            position_scopes=MappingProxyType(
                {name: scope.clone() for name, scope in scopes.items()}
            ),
            base_target_token_id=base_token,
            donor_target_token_id=donor_token,
            clean_forward_margin=forward_margin,
            clean_reverse_margin=reverse_margin,
            clean_endpoint=endpoint,
            clean_logit_absolute_max=clean_logit_max,
            clean_metadata=MappingProxyType(metadata),
            _runner=self,
        )

    def responses_from_prepared(
        self,
        prepared: PreparedComponentPositionContext,
        masks: torch.Tensor,
        *,
        position_scope: str,
        alpha: float = 1.0,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Evaluate one fixed scope for every binary component mask."""
        self._validate_masks(masks)
        if prepared._runner is not self:
            raise ValueError("prepared context belongs to a different runner")
        if position_scope not in POSITION_SCOPE_ORDER:
            raise ValueError(f"position_scope must be one of {POSITION_SCOPE_ORDER!r}")
        dose = _validated_alpha(alpha)
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        effective_batch_size = _positive_integer(
            effective_batch_size, name="batch_size"
        )
        position = prepared.position_scopes[position_scope].double()

        response_batches = []
        with torch.no_grad():
            for mask_batch in masks.split(effective_batch_size):
                mask_batch = mask_batch.to(device=self.device, dtype=torch.float64)
                repeated_ids = prepared.base_input_ids.expand(len(mask_batch), -1)
                handles = []
                for component_index, (layer, kind) in enumerate(self.components):
                    module = self._module(layer, kind)
                    selected = mask_batch[:, component_index] * dose
                    weights = selected[:, None, None] * position[None, :, None]
                    source = prepared.donor_updates[component_index][None]

                    def patch_component(
                        _module: Any,
                        _inputs: tuple[torch.Tensor, ...],
                        output: torch.Tensor,
                        *,
                        position_weights: torch.Tensor = weights,
                        source_output: torch.Tensor = source,
                    ) -> torch.Tensor:
                        recipient = output
                        return recipient + position_weights.to(recipient) * (
                            source_output.to(recipient) - recipient
                        )

                    handles.append(module.register_forward_hook(patch_component))
                try:
                    logits = self.model(input_ids=repeated_ids).logits[:, -1]
                finally:
                    for handle in handles:
                        handle.remove()
                self._check_logits(logits)
                margins = (
                    logits[:, prepared.donor_target_token_id]
                    - logits[:, prepared.base_target_token_id]
                )
                response_batches.append(
                    margins.detach().cpu().double() - prepared.clean_forward_margin
                )
        responses = torch.cat(response_batches)
        if not bool(responses.isfinite().all()):
            raise RuntimeError("a component-position response is non-finite")
        if float(responses.abs().max()) > float(
            NUMERICAL_GATES["response_absolute_max"]
        ):
            raise RuntimeError(
                "a component-position response exceeds the magnitude gate"
            )
        return responses

    def paired_responses_from_prepared(
        self,
        prepared: PreparedComponentPositionContext,
        masks: torch.Tensor,
        *,
        alpha: float = 1.0,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return all-position responses first and lexical responses second."""
        self._validate_masks(masks)
        return {
            scope: self.responses_from_prepared(
                prepared,
                masks,
                position_scope=scope,
                alpha=alpha,
                batch_size=batch_size,
            )
            for scope in POSITION_SCOPE_ORDER
        }

    def full_batch_invariance_smoke(
        self,
        prepared: PreparedComponentPositionContext,
        masks: torch.Tensor,
        *,
        alpha: float = 1.0,
        reference_batch_size: int = 1,
        candidate_batch_size: int = 32,
    ) -> dict[str, Any]:
        """Compare every mask at both scopes under two physical batch sizes."""
        self._validate_masks(masks)
        reference_batch_size = _positive_integer(
            reference_batch_size, name="reference_batch_size"
        )
        candidate_batch_size = _positive_integer(
            candidate_batch_size, name="candidate_batch_size"
        )
        dose = _validated_alpha(alpha)
        atol = float(NUMERICAL_GATES["batch_size_smoke_atol"])
        rtol = float(NUMERICAL_GATES["batch_size_smoke_rtol"])
        records = {}
        for scope in POSITION_SCOPE_ORDER:
            reference = self.responses_from_prepared(
                prepared,
                masks,
                position_scope=scope,
                alpha=dose,
                batch_size=reference_batch_size,
            )
            candidate = self.responses_from_prepared(
                prepared,
                masks,
                position_scope=scope,
                alpha=dose,
                batch_size=candidate_batch_size,
            )
            differences = (candidate - reference).abs()
            records[scope] = {
                "mask_rows": len(masks),
                "reference_batch_size": reference_batch_size,
                "candidate_batch_size": candidate_batch_size,
                "maximum_absolute_difference": float(differences.max()),
                "maximum_relative_difference": float(
                    (differences / reference.abs().clamp_min(atol)).max()
                ),
                "atol": atol,
                "rtol": rtol,
                "passed": bool(
                    torch.allclose(candidate, reference, atol=atol, rtol=rtol)
                ),
            }
        return {
            "records": records,
            "mask_rows": len(masks),
            "unique_mask_scope_cells": len(POSITION_SCOPE_ORDER) * len(masks),
            "physical_mask_scope_cells": 2 * len(POSITION_SCOPE_ORDER) * len(masks),
            "scientific_labels_produced": False,
            "passed": all(record["passed"] for record in records.values()),
        }


__all__ = [
    "COMPONENT_COUNT",
    "POSITION_SCOPE_ORDER",
    "Float64ComponentPositionRunner",
    "PreparedComponentPositionContext",
    "SublayerComponent",
]
