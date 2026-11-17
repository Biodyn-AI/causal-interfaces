"""Acquire the sealed E29.14a position and dose panel exactly once.

This stage verifies the full prospective source/checkpoint closure before it
loads the model.  It does not import or invoke the E29.14a scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    BATCH_SIZE,
    NUMERICAL_GATES,
    tensor_numerical_summary,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_custody import (
    BATCH_INVARIANCE,
    PHYSICAL_QUERY_ACCOUNTING,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    atomic_torch_save,
    load_verified_registration,
    require_exact_keys,
    runtime_environment,
    sha256,
    validate_registration_payload,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    CONTEXT_COUNT,
    FORWARD_CONDITIONS,
    POSITION_SCOPE_NAMES,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_runner import (
    Float64PositionDoseRunner,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REGISTRATION = (
    REPOSITORY_ROOT
    / "notes/E29.14a-POSITIONAL-LEXICAL-INTERCHANGE-REGISTRATION.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14a_position_lexical_interchange_v1"

PROMPT_KEYS = (
    "template",
    "base_prompt",
    "donor_prompt",
    "base_target",
    "donor_target",
)
CLEAN_METADATA_KEYS = frozenset(
    {
        *PROMPT_KEYS,
        "base_target_token_id",
        "donor_target_token_id",
        "token_length",
        "position_scope_indices",
        "clean_base_correct_minus_donor_margin",
        "clean_donor_correct_minus_base_margin",
        "clean_endpoint",
        "clean_logit_absolute_max",
        "device",
        "model_dtype",
    }
)
RUNTIME_METADATA_KEYS = frozenset(
    {
        *CLEAN_METADATA_KEYS,
        "patched_logit_absolute_max",
        "patched_response_absolute_max",
        "all_logits_finite",
        "forward_condition_rows",
        "reverse_condition_rows",
        "patched_condition_cells",
        "clean_forward_examples",
    }
)
LABEL_KEYS = frozenset(
    {
        "schema_version",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "condition_manifest",
        "forward_responses",
        "reverse_responses",
        "clean_metadata",
        "clean_base_correct_minus_donor_margins",
        "clean_donor_correct_minus_base_margins",
        "clean_endpoints",
        "batch_invariance_smoke",
        "query_accounting",
        "physical_query_accounting",
        "numerical_summary",
        "all_integrity_gates_passed",
    }
)
ATTEMPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "registration",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "source_code_sha256_verified_before_attempt",
        "checkpoint_sha256_verified_before_attempt",
        "prompt_panel_sha256",
        "position_protocol_sha256",
        "query_accounting",
        "physical_query_accounting",
        "environment",
        "execution",
    }
)
CUSTODY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "registration",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "source_code_sha256_verified_before_acquisition",
        "checkpoint_sha256_verified_before_acquisition",
        "acquisition_attempt",
        "acquisition_attempt_sha256",
        "position_labels",
        "position_labels_sha256",
        "prompt_panel_sha256",
        "position_protocol_sha256",
        "contexts",
        "forward_condition_rows_per_context",
        "reverse_condition_rows_per_context",
        "patched_context_condition_cells",
        "clean_forward_examples",
        "batch_smoke_unique_condition_rows",
        "batch_smoke_physical_condition_cells",
        "maximum_logical_queries",
        "maximum_physical_model_examples",
        "batch_size",
        "batch_invariance_passed",
        "all_clean_and_numerical_gates_passed",
        "device",
        "dtype",
        "response_absolute_max",
        "runtime_seconds",
    }
)
_SMOKE_KEYS = frozenset(
    {
        "records",
        "unique_condition_rows",
        "physical_condition_cells",
        "scientific_labels_produced",
        "passed",
    }
)
_SMOKE_RECORD_KEYS = frozenset(
    {
        "direction",
        "condition_rows",
        "reference_batch_size",
        "candidate_batch_size",
        "maximum_absolute_difference",
        "maximum_relative_difference",
        "atol",
        "rtol",
        "passed",
    }
)


def prompt_records(
    prompt_checks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Extract only the five strings consumed by the registered runner."""
    if not isinstance(prompt_checks, Sequence):
        raise TypeError("E29.14a prompt checks must be a sequence")
    records = tuple(
        {key: str(check[key]) for key in PROMPT_KEYS} for check in prompt_checks
    )
    if len(records) != CONTEXT_COUNT:
        raise RuntimeError("E29.14a requires exactly twelve registered prompts")
    return records


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be a finite number")
    return number


def _validate_scope_indices(
    value: Any,
    *,
    token_length: int,
    target_position: int,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("runtime position scopes must be a mapping")
    require_exact_keys(value, POSITION_SCOPE_NAMES, label="runtime position scopes")
    scopes: dict[str, set[int]] = {}
    for name in POSITION_SCOPE_NAMES:
        indices = value[name]
        if not isinstance(indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise RuntimeError("runtime position scopes must contain integer lists")
        if indices != sorted(set(indices)) or any(
            index < 0 or index >= token_length for index in indices
        ):
            raise RuntimeError("runtime position scope indices are invalid")
        scopes[name] = set(indices)

    all_positions = set(range(token_length))
    differing = scopes["differing_lexical_positions"]
    static = scopes["shared_static_positions"]
    target = scopes["demonstration_target_position"]
    prefix = scopes["demonstration_prefix_positions"]
    repeated = scopes["repeated_query_positions"]
    except_target = scopes["differing_except_demonstration_target"]
    if scopes["all_positions"] != all_positions:
        raise RuntimeError("runtime all-position scope changed")
    if target != {target_position} or target_position not in differing:
        raise RuntimeError("runtime demonstration-target position changed")
    if differing & static or differing | static != all_positions:
        raise RuntimeError("runtime lexical/static position partition changed")
    if (
        not prefix
        or not repeated
        or any(index >= target_position for index in prefix)
        or any(index <= target_position for index in repeated)
        or differing != prefix | target | repeated
        or except_target != prefix | repeated
    ):
        raise RuntimeError("runtime differing-position partition changed")


def _verify_clean_metadata(
    prompt_checks: Sequence[Mapping[str, Any]],
    prepared: Sequence[Any],
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(prepared) != CONTEXT_COUNT:
        raise RuntimeError("E29.14a prepared-context count changed")
    metadata = []
    base_margins = []
    donor_margins = []
    endpoints = []
    logit_limit = float(NUMERICAL_GATES["clean_logit_absolute_max"])
    for registered, context in zip(prompt_checks, prepared, strict=True):
        observed = dict(context.clean_metadata)
        require_exact_keys(
            observed,
            CLEAN_METADATA_KEYS,
            label="E29.14a clean runtime metadata",
        )
        for key in PROMPT_KEYS:
            if str(observed[key]) != str(registered[key]):
                raise RuntimeError("runtime prompt metadata differs from registration")
        token_length = int(registered["token_length"])
        if int(observed["token_length"]) != token_length:
            raise RuntimeError("runtime prompt token length differs from registration")
        for side in ("base", "donor"):
            expected_token = int(registered["sides"][side]["target_token_id"])
            if int(observed[f"{side}_target_token_id"]) != expected_token:
                raise RuntimeError("runtime target token differs from registration")
        target_position = int(
            registered["sides"]["base"]["demonstration_target_position"]
        )
        _validate_scope_indices(
            observed["position_scope_indices"],
            token_length=token_length,
            target_position=target_position,
        )
        base_margin = _finite_number(
            observed["clean_base_correct_minus_donor_margin"],
            label="clean base correctness margin",
        )
        donor_margin = _finite_number(
            observed["clean_donor_correct_minus_base_margin"],
            label="clean donor correctness margin",
        )
        endpoint = _finite_number(
            observed["clean_endpoint"], label="clean endpoint"
        )
        logit_maximum = _finite_number(
            observed["clean_logit_absolute_max"], label="clean logit maximum"
        )
        if base_margin <= 0.0 or donor_margin <= 0.0:
            raise RuntimeError("E29.14a clean correctness gate failed")
        if endpoint <= 0.0 or endpoint != base_margin + donor_margin:
            raise RuntimeError("E29.14a clean endpoint arithmetic changed")
        if logit_maximum < 0.0 or logit_maximum > logit_limit:
            raise RuntimeError("E29.14a clean numerical magnitude gate failed")
        if observed["device"] != "cpu" or observed["model_dtype"] != "float64":
            raise RuntimeError("E29.14a runtime device or dtype changed")
        metadata.append(observed)
        base_margins.append(base_margin)
        donor_margins.append(donor_margin)
        endpoints.append(endpoint)
    return (
        metadata,
        torch.tensor(base_margins, dtype=torch.float64),
        torch.tensor(donor_margins, dtype=torch.float64),
        torch.tensor(endpoints, dtype=torch.float64),
    )


def _response_vector(value: Any, *, rows: int, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} response must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise RuntimeError(f"{label} response device or dtype changed")
    if tuple(value.shape) != (rows,):
        raise RuntimeError(f"{label} response shape changed")
    if not bool(value.isfinite().all()):
        raise RuntimeError(f"{label} response is non-finite")
    if float(value.abs().max()) > float(NUMERICAL_GATES["response_absolute_max"]):
        raise RuntimeError(f"{label} response exceeds the magnitude gate")
    return value.detach().clone()


def _verify_patched_metadata(
    observed_value: Any,
    *,
    clean: Mapping[str, Any],
    responses: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    if not isinstance(observed_value, Mapping):
        raise TypeError("E29.14a patched runtime metadata must be a mapping")
    observed = dict(observed_value)
    require_exact_keys(
        observed,
        RUNTIME_METADATA_KEYS,
        label="E29.14a patched runtime metadata",
    )
    if any(observed[key] != clean[key] for key in CLEAN_METADATA_KEYS):
        raise RuntimeError("patched metadata differs from verified clean metadata")
    expected_counts = {
        "forward_condition_rows": len(FORWARD_CONDITIONS),
        "reverse_condition_rows": len(REVERSE_CONDITIONS),
        "patched_condition_cells": len(FORWARD_CONDITIONS)
        + len(REVERSE_CONDITIONS),
        "clean_forward_examples": 2,
    }
    if any(observed[key] != expected for key, expected in expected_counts.items()):
        raise RuntimeError("E29.14a per-context query accounting changed")
    if observed["all_logits_finite"] is not True:
        raise RuntimeError("E29.14a patched logits are non-finite")
    logit_maximum = _finite_number(
        observed["patched_logit_absolute_max"], label="patched logit maximum"
    )
    response_maximum = _finite_number(
        observed["patched_response_absolute_max"],
        label="patched response maximum",
    )
    actual_maximum = float(torch.cat(responses).abs().max())
    if (
        logit_maximum < 0.0
        or logit_maximum > float(NUMERICAL_GATES["clean_logit_absolute_max"])
        or response_maximum < 0.0
        or response_maximum > float(NUMERICAL_GATES["response_absolute_max"])
        or response_maximum != actual_maximum
    ):
        raise RuntimeError("E29.14a patched numerical metadata gate failed")
    return observed


def _verify_batch_smoke(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("E29.14a batch smoke must be a mapping")
    smoke = dict(value)
    require_exact_keys(smoke, _SMOKE_KEYS, label="E29.14a batch smoke")
    records = smoke["records"]
    if not isinstance(records, Mapping) or tuple(records) != ("forward", "reverse"):
        raise RuntimeError("E29.14a batch-smoke directions changed")
    for direction, conditions in (
        ("forward", FORWARD_CONDITIONS),
        ("reverse", REVERSE_CONDITIONS),
    ):
        record = records[direction]
        if not isinstance(record, Mapping):
            raise TypeError("E29.14a batch-smoke record must be a mapping")
        require_exact_keys(
            record,
            _SMOKE_RECORD_KEYS,
            label=f"E29.14a {direction} batch-smoke record",
        )
        maximum_absolute = _finite_number(
            record["maximum_absolute_difference"],
            label=f"{direction} batch-smoke absolute difference",
        )
        maximum_relative = _finite_number(
            record["maximum_relative_difference"],
            label=f"{direction} batch-smoke relative difference",
        )
        expected = {
            "direction": direction,
            "condition_rows": len(conditions),
            "reference_batch_size": BATCH_INVARIANCE["reference_batch_size"],
            "candidate_batch_size": BATCH_INVARIANCE["candidate_batch_size"],
            "atol": BATCH_INVARIANCE["absolute_tolerance"],
            "rtol": BATCH_INVARIANCE["relative_tolerance"],
            "passed": True,
        }
        if any(record[key] != expected_value for key, expected_value in expected.items()):
            raise RuntimeError("E29.14a batch invariance gate failed")
        if maximum_absolute < 0.0 or maximum_relative < 0.0:
            raise RuntimeError("E29.14a batch-invariance differences are invalid")
    if (
        smoke["unique_condition_rows"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_unique_condition_rows"]
        or smoke["physical_condition_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"]
        or smoke["scientific_labels_produced"] is not False
        or smoke["passed"] is not True
    ):
        raise RuntimeError("E29.14a batch invariance gate failed")
    return smoke


def acquire_position_panel_with_runner(
    runner: Any,
    *,
    prompt_checks: Sequence[Mapping[str, Any]],
    registration_sha256: str = "pure_runner",
    expected_registration_sha256_supplied: str = "pure_runner",
) -> dict[str, Any]:
    """Acquire all fixed conditions after clean gates, then run one smoke."""
    records = prompt_records(prompt_checks)
    prepared = tuple(runner.prepare_context(record) for record in records)
    clean_metadata, base_margins, donor_margins, endpoints = (
        _verify_clean_metadata(prompt_checks, prepared)
    )

    forward_rows = []
    reverse_rows = []
    patched_metadata = []
    for context, clean in zip(prepared, clean_metadata, strict=True):
        responses, observed_metadata = runner.registered_responses(context)
        if not isinstance(responses, Mapping) or tuple(responses) != (
            "forward",
            "reverse",
        ):
            raise RuntimeError("E29.14a registered response directions changed")
        forward = _response_vector(
            responses["forward"], rows=len(FORWARD_CONDITIONS), label="forward"
        )
        reverse = _response_vector(
            responses["reverse"], rows=len(REVERSE_CONDITIONS), label="reverse"
        )
        patched_metadata.append(
            _verify_patched_metadata(
                observed_metadata,
                clean=clean,
                responses=(forward, reverse),
            )
        )
        forward_rows.append(forward)
        reverse_rows.append(reverse)

    forward_responses = torch.stack(forward_rows)
    reverse_responses = torch.stack(reverse_rows)
    smoke = runner.registered_batch_invariance_smoke(
        prepared[int(BATCH_INVARIANCE["context_index"])],
        reference_batch_size=int(BATCH_INVARIANCE["reference_batch_size"]),
        candidate_batch_size=int(BATCH_INVARIANCE["candidate_batch_size"]),
    )
    smoke = _verify_batch_smoke(smoke)
    combined = torch.cat((forward_responses.flatten(), reverse_responses.flatten()))
    numerical_summary = {
        "forward_responses": tensor_numerical_summary(forward_responses),
        "reverse_responses": tensor_numerical_summary(reverse_responses),
        "combined_responses": tensor_numerical_summary(combined),
        "clean_endpoints": tensor_numerical_summary(endpoints),
    }
    if not all(summary["passed"] for summary in numerical_summary.values()):
        raise RuntimeError("E29.14a final numerical summary gate failed")

    artifact = {
        "schema_version": "e29_14a_position_labels_v1",
        "registration_sha256": registration_sha256,
        "expected_registration_sha256_supplied": (
            expected_registration_sha256_supplied
        ),
        "condition_manifest": {
            "forward": [dict(condition) for condition in FORWARD_CONDITIONS],
            "reverse": [dict(condition) for condition in REVERSE_CONDITIONS],
        },
        "forward_responses": forward_responses,
        "reverse_responses": reverse_responses,
        "clean_metadata": patched_metadata,
        "clean_base_correct_minus_donor_margins": base_margins,
        "clean_donor_correct_minus_base_margins": donor_margins,
        "clean_endpoints": endpoints,
        "batch_invariance_smoke": smoke,
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": numerical_summary,
        "all_integrity_gates_passed": True,
    }
    require_exact_keys(artifact, LABEL_KEYS, label="E29.14a position labels")
    return artifact


def build_acquisition_attempt_record(
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    observed_registration_sha256: str,
    expected_registration_sha256: str,
) -> dict[str, Any]:
    """Build the immutable marker that consumes this registered attempt."""
    marker = {
        "schema_version": "e29_14a_position_acquisition_attempt_v1",
        "status": "attempt_consumed_before_model_load",
        "registration": str(registration_path.resolve()),
        "registration_sha256": observed_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "source_code_sha256_verified_before_attempt": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_attempt": dict(
            registration["checkpoint_sha256"]
        ),
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "position_protocol_sha256": registration["position_protocol_sha256"],
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "environment": dict(registration["environment"]),
        "execution": dict(registration["execution"]),
    }
    require_exact_keys(marker, ATTEMPT_KEYS, label="E29.14a acquisition attempt")
    return marker


def acquire_position(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    output_directory: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Verify the seal and checkpoint, acquire once, and write custody once."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    attempt_path = output_directory / "position_acquisition_attempt.json"
    labels_path = output_directory / "position_labels.pt"
    custody_path = output_directory / "position_acquisition_custody.json"
    if attempt_path.exists() or labels_path.exists() or custody_path.exists():
        raise FileExistsError("refusing to rerun or overwrite E29.14a position custody")
    registration = load_verified_registration(
        registration_path,
        expected_sha256=expected_registration_sha256,
        required_keys=REGISTRATION_REQUIRED_KEYS,
    )
    registration = validate_registration_payload(
        registration,
        source_directory=SOURCE_DIRECTORY,
        verify_external_files=True,
    )
    observed_registration_sha = sha256(registration_path)
    if observed_registration_sha != expected_registration_sha256:
        raise RuntimeError("E29.14a registration changed during verification")
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14a runtime environment differs from registration")
    execution = registration["execution"]
    if (
        batch_size != BATCH_SIZE
        or batch_size != execution["batch_size"]
        or execution["device"] != "cpu"
        or execution["model_dtype"] != "float64"
        or execution["attention_implementation"] != "eager"
    ):
        raise RuntimeError("E29.14a runtime execution differs from registration")

    attempt = build_acquisition_attempt_record(
        registration=registration,
        registration_path=registration_path,
        observed_registration_sha256=observed_registration_sha,
        expected_registration_sha256=expected_registration_sha256,
    )
    atomic_json_save(attempt, attempt_path)

    snapshot = Path(registration["model_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float64,
    ).eval()
    runner = Float64PositionDoseRunner(model, tokenizer, batch_size=batch_size)
    if tuple(runner.modules) != (0, 1):
        raise RuntimeError("E29.14a runtime component map changed")
    artifact = acquire_position_panel_with_runner(
        runner,
        prompt_checks=registration["prompt_checks"],
        registration_sha256=observed_registration_sha,
        expected_registration_sha256_supplied=expected_registration_sha256,
    )

    response_maximum = max(
        float(artifact["forward_responses"].abs().max()),
        float(artifact["reverse_responses"].abs().max()),
    )
    atomic_torch_save(artifact, labels_path)
    custody = {
        "schema_version": "e29_14a_position_acquisition_custody_v1",
        "status": "all_registered_conditions_acquired_after_integrity_gates",
        "registration": str(registration_path.resolve()),
        "registration_sha256": observed_registration_sha,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "source_code_sha256_verified_before_acquisition": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_acquisition": dict(
            registration["checkpoint_sha256"]
        ),
        "acquisition_attempt": str(attempt_path.resolve()),
        "acquisition_attempt_sha256": sha256(attempt_path),
        "position_labels": str(labels_path.resolve()),
        "position_labels_sha256": sha256(labels_path),
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "position_protocol_sha256": registration["position_protocol_sha256"],
        "contexts": QUERY_ACCOUNTING["contexts"],
        "forward_condition_rows_per_context": QUERY_ACCOUNTING[
            "forward_condition_rows_per_context"
        ],
        "reverse_condition_rows_per_context": QUERY_ACCOUNTING[
            "reverse_condition_rows_per_context"
        ],
        "patched_context_condition_cells": QUERY_ACCOUNTING[
            "patched_context_condition_cells"
        ],
        "clean_forward_examples": QUERY_ACCOUNTING["clean_forward_examples"],
        "batch_smoke_unique_condition_rows": PHYSICAL_QUERY_ACCOUNTING[
            "batch_smoke_unique_condition_rows"
        ],
        "batch_smoke_physical_condition_cells": PHYSICAL_QUERY_ACCOUNTING[
            "batch_smoke_physical_condition_cells"
        ],
        "maximum_logical_queries": QUERY_ACCOUNTING["maximum_logical_queries"],
        "maximum_physical_model_examples": PHYSICAL_QUERY_ACCOUNTING[
            "maximum_physical_model_examples"
        ],
        "batch_size": batch_size,
        "batch_invariance_passed": True,
        "all_clean_and_numerical_gates_passed": True,
        "device": "cpu",
        "dtype": "float64",
        "response_absolute_max": response_maximum,
        "runtime_seconds": time.perf_counter() - started,
    }
    require_exact_keys(custody, CUSTODY_KEYS, label="E29.14a acquisition custody")
    atomic_json_save(custody, custody_path)
    return {**custody, "custody_sha256": sha256(custody_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()
    print(
        json.dumps(
            acquire_position(
                registration_path=arguments.registration,
                expected_registration_sha256=arguments.expected_registration_sha256,
                output_directory=arguments.output_directory,
                batch_size=arguments.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ATTEMPT_KEYS",
    "CLEAN_METADATA_KEYS",
    "CUSTODY_KEYS",
    "LABEL_KEYS",
    "RUNTIME_METADATA_KEYS",
    "acquire_position",
    "acquire_position_panel_with_runner",
    "build_acquisition_attempt_record",
    "prompt_records",
]
