"""CPU-float64 position and dose patch runner for prospective E29.14a."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    CONTROL_COMPONENT,
    FORWARD_CONDITIONS,
    HUB_COMPONENT,
    POSITION_SCOPE_NAMES,
    REVERSE_CONDITIONS,
    build_position_scopes,
)


@dataclass(frozen=True, eq=False)
class PreparedPositionDoseContext:
    """Two clean directions and their layer-0 updates cached before patching."""

    base_input_ids: torch.Tensor = field(repr=False)
    donor_input_ids: torch.Tensor = field(repr=False)
    base_updates: Mapping[int, torch.Tensor] = field(repr=False)
    donor_updates: Mapping[int, torch.Tensor] = field(repr=False)
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


def _clean_conditions(
    conditions: Sequence[Mapping[str, Any]], *, direction: str
) -> tuple[dict[str, Any], ...]:
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'")
    if not isinstance(conditions, Sequence) or not len(conditions):
        raise ValueError("conditions must be a nonempty sequence")
    cleaned = []
    names = set()
    for condition in conditions:
        if not isinstance(condition, Mapping) or set(condition) != {
            "name",
            "component",
            "scope",
            "alpha",
        }:
            raise ValueError("each condition must have the four fixed fields")
        name = str(condition["name"])
        component = int(condition["component"])
        scope = str(condition["scope"])
        alpha = float(condition["alpha"])
        if not name or name in names:
            raise ValueError("condition names must be nonempty and unique")
        if component not in {CONTROL_COMPONENT, HUB_COMPONENT}:
            raise ValueError("condition component must be the hub or matched control")
        if direction == "reverse" and component != HUB_COMPONENT:
            raise ValueError("reverse conditions are restricted to the hub component")
        if scope not in POSITION_SCOPE_NAMES:
            raise ValueError("condition uses an unknown position scope")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("condition alpha must lie in [0, 1]")
        names.add(name)
        cleaned.append(
            {"name": name, "component": component, "scope": scope, "alpha": alpha}
        )
    return tuple(cleaned)


class Float64PositionDoseRunner:
    """Patch layer-0 attention/MLP updates by position, dose, and direction."""

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if model.training:
            raise ValueError("dropout must be disabled with model.eval()")
        parameters = tuple(model.parameters())
        if not parameters:
            raise ValueError("model has no parameters")
        state = (*parameters, *tuple(model.buffers()))
        if any(tensor.device.type != "cpu" for tensor in state):
            raise ValueError("E29.14a requires a CPU model")
        if any(
            tensor.dtype != torch.float64
            for tensor in state
            if tensor.is_floating_point()
        ):
            raise ValueError("E29.14a requires a float64 model")
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            raise TypeError("model must expose the GPT-2 transformer block layout")
        if len(model.transformer.h) < 1:
            raise ValueError("model must contain a layer-0 block")
        block = model.transformer.h[0]
        if not hasattr(block, "attn") or not hasattr(block.attn, "c_proj"):
            raise TypeError("layer-0 attention projection is unavailable")
        if not hasattr(block, "mlp") or not hasattr(block.mlp, "c_proj"):
            raise TypeError("layer-0 MLP projection is unavailable")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.modules = {
            CONTROL_COMPONENT: block.attn.c_proj,
            HUB_COMPONENT: block.mlp.c_proj,
        }

    def _input_ids(self, prompt: str) -> torch.Tensor:
        encoded: SimpleNamespace = self.tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids.detach().cpu()

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
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for component, module in self.modules.items():

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
        if set(captured) != set(self.modules):
            raise RuntimeError("failed to capture both registered component updates")
        self._check_logits(logits)
        return captured, logits

    def prepare_context(self, record: Mapping[str, Any]) -> PreparedPositionDoseContext:
        """Run the two clean directions and freeze their layer-0 source updates."""
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
        base_ids = self._input_ids(base_prompt)
        donor_ids = self._input_ids(donor_prompt)
        if (
            base_ids.shape != donor_ids.shape
            or base_ids.ndim != 2
            or len(base_ids) != 1
        ):
            raise ValueError("base and donor prompts must be one aligned token row")
        base_token = _target_token_id(self.tokenizer, base_target)
        donor_token = _target_token_id(self.tokenizer, donor_target)
        scopes = build_position_scopes(
            base_ids[0], donor_ids[0], base_target_token_id=base_token
        )
        base_updates, base_logits = self._capture_updates_and_logits(base_ids)
        donor_updates, donor_logits = self._capture_updates_and_logits(donor_ids)
        forward_margin = float(base_logits[donor_token] - base_logits[base_token])
        reverse_margin = float(donor_logits[base_token] - donor_logits[donor_token])
        if forward_margin >= 0.0 or reverse_margin >= 0.0:
            raise RuntimeError("a clean base or donor correctness gate failed")
        endpoint = -forward_margin - reverse_margin
        if endpoint <= 0.0:
            raise RuntimeError("clean endpoint must be positive")
        scope_indices = {
            name: [int(index) for index in torch.where(scope)[0]]
            for name, scope in scopes.items()
        }
        metadata = {
            "template": str(record["template"]),
            "base_prompt": base_prompt,
            "donor_prompt": donor_prompt,
            "base_target": base_target,
            "donor_target": donor_target,
            "base_target_token_id": base_token,
            "donor_target_token_id": donor_token,
            "token_length": int(base_ids.shape[1]),
            "position_scope_indices": scope_indices,
            "clean_base_correct_minus_donor_margin": -forward_margin,
            "clean_donor_correct_minus_base_margin": -reverse_margin,
            "clean_endpoint": endpoint,
            "clean_logit_absolute_max": max(
                float(base_logits.abs().max()), float(donor_logits.abs().max())
            ),
            "device": "cpu",
            "model_dtype": "float64",
        }
        return PreparedPositionDoseContext(
            base_input_ids=base_ids.clone(),
            donor_input_ids=donor_ids.clone(),
            base_updates=MappingProxyType(
                {
                    component: update.clone()
                    for component, update in base_updates.items()
                }
            ),
            donor_updates=MappingProxyType(
                {
                    component: update.clone()
                    for component, update in donor_updates.items()
                }
            ),
            position_scopes=MappingProxyType(
                {name: scope.clone() for name, scope in scopes.items()}
            ),
            base_target_token_id=base_token,
            donor_target_token_id=donor_token,
            clean_forward_margin=forward_margin,
            clean_reverse_margin=reverse_margin,
            clean_endpoint=endpoint,
            clean_logit_absolute_max=metadata["clean_logit_absolute_max"],
            clean_metadata=MappingProxyType(metadata),
            _runner=self,
        )

    def _evaluate(
        self,
        prepared: PreparedPositionDoseContext,
        *,
        direction: str,
        conditions: Sequence[Mapping[str, Any]],
        batch_size: int,
    ) -> tuple[torch.Tensor, float]:
        clean_conditions = _clean_conditions(conditions, direction=direction)
        if prepared._runner is not self:
            raise ValueError("prepared context belongs to a different runner")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if direction == "forward":
            recipient_ids = prepared.base_input_ids
            source_updates = prepared.donor_updates
            recipient_target = prepared.base_target_token_id
            source_target = prepared.donor_target_token_id
            clean_margin = prepared.clean_forward_margin
        else:
            recipient_ids = prepared.donor_input_ids
            source_updates = prepared.base_updates
            recipient_target = prepared.donor_target_token_id
            source_target = prepared.base_target_token_id
            clean_margin = prepared.clean_reverse_margin

        response_batches = []
        maximum_logit_absolute = 0.0
        with torch.no_grad():
            for start in range(0, len(clean_conditions), batch_size):
                condition_batch = clean_conditions[start : start + batch_size]
                repeated_ids = recipient_ids.expand(len(condition_batch), -1)
                handles = []
                for component, module in self.modules.items():
                    position_weights = torch.stack(
                        [
                            prepared.position_scopes[condition["scope"]].double()
                            * (
                                condition["alpha"]
                                if condition["component"] == component
                                else 0.0
                            )
                            for condition in condition_batch
                        ]
                    )[:, :, None]
                    source = source_updates[component][None]

                    def patch(
                        _module: Any,
                        _inputs: tuple[torch.Tensor, ...],
                        output: torch.Tensor,
                        *,
                        weights: torch.Tensor = position_weights,
                        source_output: torch.Tensor = source,
                    ) -> torch.Tensor:
                        return output + weights.to(output) * (
                            source_output.to(output) - output
                        )

                    handles.append(module.register_forward_hook(patch))
                try:
                    logits = self.model(input_ids=repeated_ids).logits[:, -1]
                finally:
                    for handle in handles:
                        handle.remove()
                self._check_logits(logits)
                maximum_logit_absolute = max(
                    maximum_logit_absolute, float(logits.abs().max())
                )
                margin = logits[:, source_target] - logits[:, recipient_target]
                response_batches.append(margin.detach().cpu().double() - clean_margin)
        responses = torch.cat(response_batches)
        if not bool(responses.isfinite().all()):
            raise RuntimeError("a position-dose response is non-finite")
        if float(responses.abs().max()) > float(
            NUMERICAL_GATES["response_absolute_max"]
        ):
            raise RuntimeError("a position-dose response exceeds the magnitude gate")
        return responses, maximum_logit_absolute

    def responses_from_prepared(
        self,
        prepared: PreparedPositionDoseContext,
        *,
        direction: str,
        conditions: Sequence[Mapping[str, Any]],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Evaluate an explicit condition sequence without another clean forward."""
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        responses, _ = self._evaluate(
            prepared,
            direction=direction,
            conditions=conditions,
            batch_size=effective_batch_size,
        )
        return responses

    def registered_responses(
        self, prepared: PreparedPositionDoseContext
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Evaluate the exact 12 forward and four reverse registered conditions."""
        forward, forward_logit_max = self._evaluate(
            prepared,
            direction="forward",
            conditions=FORWARD_CONDITIONS,
            batch_size=self.batch_size,
        )
        reverse, reverse_logit_max = self._evaluate(
            prepared,
            direction="reverse",
            conditions=REVERSE_CONDITIONS,
            batch_size=self.batch_size,
        )
        responses = {"forward": forward, "reverse": reverse}
        metadata = {
            **dict(prepared.clean_metadata),
            "patched_logit_absolute_max": max(forward_logit_max, reverse_logit_max),
            "patched_response_absolute_max": float(
                torch.cat((forward, reverse)).abs().max()
            ),
            "all_logits_finite": True,
            "forward_condition_rows": len(FORWARD_CONDITIONS),
            "reverse_condition_rows": len(REVERSE_CONDITIONS),
            "patched_condition_cells": len(FORWARD_CONDITIONS)
            + len(REVERSE_CONDITIONS),
            "clean_forward_examples": 2,
        }
        return responses, metadata

    def batch_invariance_smoke(
        self,
        prepared: PreparedPositionDoseContext,
        *,
        direction: str,
        conditions: Sequence[Mapping[str, Any]],
        reference_batch_size: int = 1,
        candidate_batch_size: int = 32,
    ) -> dict[str, Any]:
        """Compare identical conditions at two physical batch sizes."""
        reference = self.responses_from_prepared(
            prepared,
            direction=direction,
            conditions=conditions,
            batch_size=reference_batch_size,
        )
        candidate = self.responses_from_prepared(
            prepared,
            direction=direction,
            conditions=conditions,
            batch_size=candidate_batch_size,
        )
        differences = (candidate - reference).abs()
        atol = float(NUMERICAL_GATES["batch_size_smoke_atol"])
        rtol = float(NUMERICAL_GATES["batch_size_smoke_rtol"])
        return {
            "direction": direction,
            "condition_rows": len(conditions),
            "reference_batch_size": reference_batch_size,
            "candidate_batch_size": candidate_batch_size,
            "maximum_absolute_difference": float(differences.max()),
            "maximum_relative_difference": float(
                (differences / reference.abs().clamp_min(atol)).max()
            ),
            "atol": atol,
            "rtol": rtol,
            "passed": bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol)),
        }

    def registered_batch_invariance_smoke(
        self,
        prepared: PreparedPositionDoseContext,
        *,
        reference_batch_size: int = 1,
        candidate_batch_size: int = 32,
    ) -> dict[str, Any]:
        """Run the fixed full-manifest numerical smoke in both directions."""
        records = {
            "forward": self.batch_invariance_smoke(
                prepared,
                direction="forward",
                conditions=FORWARD_CONDITIONS,
                reference_batch_size=reference_batch_size,
                candidate_batch_size=candidate_batch_size,
            ),
            "reverse": self.batch_invariance_smoke(
                prepared,
                direction="reverse",
                conditions=REVERSE_CONDITIONS,
                reference_batch_size=reference_batch_size,
                candidate_batch_size=candidate_batch_size,
            ),
        }
        return {
            "records": records,
            "unique_condition_rows": len(FORWARD_CONDITIONS) + len(REVERSE_CONDITIONS),
            "physical_condition_cells": 2
            * (len(FORWARD_CONDITIONS) + len(REVERSE_CONDITIONS)),
            "scientific_labels_produced": False,
            "passed": all(record["passed"] for record in records.values()),
        }


__all__ = [
    "Float64PositionDoseRunner",
    "PreparedPositionDoseContext",
]
