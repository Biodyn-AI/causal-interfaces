"""Acquire the sealed E29.14c high-cardinality scale/scope panel once.

The pure acquisition function prepares and validates every clean context before
making any patched query.  The entrypoint verifies all sealed inputs, loads the
response-blind masks, and consumes an immutable attempt before loading either
the tokenizer or model.  This module never imports the E29.14c scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    BATCH_SIZE,
    NUMERICAL_GATES,
    tensor_numerical_summary,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_runner import (
    Float64ComponentPositionRunner,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    BATCH_INVARIANCE,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    atomic_torch_save,
    build_acquisition_attempt_record,
    create_acquisition_attempt_marker,
    load_verified_protocol_masks,
    load_verified_registration,
    runtime_environment,
    sha256,
    validate_clean_metadata_contract,
    validate_registration_payload,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    HUB_COMPONENT,
    MASK_SHAPES,
    MASK_SPLITS,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    validate_scale_masks,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.14c-HUB-OFF-SCALE-LADDER-REGISTRATION-V2.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14c_hub_off_scale_ladder_v2"

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
        "model_layer_count",
        "component_count",
        "component_semantics",
        "patch_direction",
        "position_scope_order",
        "position_scope_indices",
        "clean_base_correct_minus_donor_margin",
        "clean_donor_correct_minus_base_margin",
        "clean_endpoint",
        "clean_base_logit_absolute_max",
        "clean_donor_logit_absolute_max",
        "clean_logit_absolute_max",
        "clean_logits_finite",
        "device",
        "model_dtype",
    }
)
LABEL_KEYS = frozenset(
    {
        "schema_version",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "condition_manifest",
        "singleton_all_response",
        "singleton_lexical_response",
        "all_position_responses",
        "lexical_position_responses",
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
CUSTODY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "registration",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "source_code_sha256_verified_before_acquisition",
        "checkpoint_sha256_verified_before_acquisition",
        "protocol_masks",
        "protocol_masks_sha256_verified_before_acquisition",
        "protocol_masks_schema",
        "acquisition_attempt",
        "acquisition_attempt_sha256",
        "scale_scope_labels",
        "scale_scope_labels_sha256",
        "prompt_panel_sha256",
        "clean_metadata_contract_sha256",
        "scale_scope_protocol_sha256",
        "contexts",
        "mask_rows",
        "position_scopes",
        "patched_condition_rows_per_context",
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
        "mask_rows",
        "unique_mask_scope_cells",
        "physical_mask_scope_cells",
        "scientific_labels_produced",
        "passed",
    }
)
_SMOKE_RECORD_KEYS = frozenset(
    {
        "mask_rows",
        "reference_batch_size",
        "candidate_batch_size",
        "maximum_absolute_difference",
        "maximum_relative_difference",
        "atol",
        "rtol",
        "passed",
    }
)


def _require_exact_keys(
    record: Mapping[str, Any], expected: Collection[str], *, label: str
) -> None:
    observed = set(record)
    required = set(expected)
    if observed != required:
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(required - observed)}, "
            f"extra={sorted(observed - required)}"
        )


def prompt_records(
    prompt_checks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Extract only the five registered strings consumed by the runner."""
    if not isinstance(prompt_checks, Sequence):
        raise TypeError("E29.14c prompt checks must be a sequence")
    records = tuple(
        {key: str(check[key]) for key in PROMPT_KEYS} for check in prompt_checks
    )
    if len(records) != CONTEXT_COUNT:
        raise RuntimeError("E29.14c requires exactly twelve registered prompts")
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


def _verify_position_scopes(context: Any, observed: Mapping[str, Any]) -> None:
    scopes = observed["position_scope_indices"]
    if not isinstance(scopes, Mapping) or tuple(scopes) != POSITION_SCOPE_ORDER:
        raise RuntimeError("E29.14c runtime position-scope order changed")
    token_length = int(observed["token_length"])
    all_indices = scopes["all_positions"]
    lexical_indices = scopes["differing_lexical_positions"]
    for name, indices in scopes.items():
        if not isinstance(indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise RuntimeError(f"E29.14c {name} indices must be an integer list")
        if indices != sorted(set(indices)) or any(
            index < 0 or index >= token_length for index in indices
        ):
            raise RuntimeError(f"E29.14c {name} indices are invalid")
    if all_indices != list(range(token_length)):
        raise RuntimeError("E29.14c all-position scope changed")
    base_ids = context.base_input_ids.detach().cpu()
    donor_ids = context.donor_input_ids.detach().cpu()
    if tuple(base_ids.shape) != (1, token_length) or base_ids.shape != donor_ids.shape:
        raise RuntimeError("E29.14c prepared token rows changed")
    expected_lexical = [
        int(index) for index in torch.where(base_ids[0] != donor_ids[0])[0]
    ]
    if not expected_lexical or lexical_indices != expected_lexical:
        raise RuntimeError("E29.14c differing-lexical scope changed")


def _verify_clean_metadata(
    prompt_checks: Sequence[Mapping[str, Any]],
    prepared: Sequence[Any],
    clean_metadata_contract: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify every clean gate before any patched outcome is requested."""
    if len(prepared) != CONTEXT_COUNT:
        raise RuntimeError("E29.14c prepared-context count changed")
    metadata = []
    base_margins = []
    donor_margins = []
    endpoints = []
    logit_limit = float(NUMERICAL_GATES["clean_logit_absolute_max"])
    for registered, sealed, context in zip(
        prompt_checks, clean_metadata_contract, prepared, strict=True
    ):
        observed = dict(context.clean_metadata)
        _require_exact_keys(
            observed, CLEAN_METADATA_KEYS, label="E29.14c clean runtime metadata"
        )
        if any(str(observed[key]) != str(registered[key]) for key in PROMPT_KEYS):
            raise RuntimeError("runtime prompt metadata differs from registration")
        sealed_observed = {
            "context_index": int(sealed["context_index"]),
            **{key: str(observed[key]) for key in PROMPT_KEYS},
            "base_target_token_id": int(observed["base_target_token_id"]),
            "donor_target_token_id": int(observed["donor_target_token_id"]),
            "token_length": int(observed["token_length"]),
            "position_scope_indices": observed["position_scope_indices"],
        }
        if sealed_observed != sealed:
            raise RuntimeError("runtime clean metadata differs from V2 seal")
        if int(observed["token_length"]) != int(registered["token_length"]):
            raise RuntimeError("runtime prompt token length differs from registration")
        for side in ("base", "donor"):
            expected_token = int(registered["sides"][side]["target_token_id"])
            if int(observed[f"{side}_target_token_id"]) != expected_token:
                raise RuntimeError("runtime target token differs from registration")
        expected_fixed = {
            "model_layer_count": 24,
            "component_count": COMPONENT_COUNT,
            "component_semantics": "attention_and_mlp_residual_updates",
            "patch_direction": "donor_to_base_only",
            "position_scope_order": list(POSITION_SCOPE_ORDER),
            "clean_logits_finite": True,
            "device": "cpu",
            "model_dtype": "float64",
        }
        if any(observed[key] != value for key, value in expected_fixed.items()):
            raise RuntimeError("E29.14c clean runtime contract changed")
        _verify_position_scopes(context, observed)

        base_margin = _finite_number(
            observed["clean_base_correct_minus_donor_margin"],
            label="clean base correctness margin",
        )
        donor_margin = _finite_number(
            observed["clean_donor_correct_minus_base_margin"],
            label="clean donor correctness margin",
        )
        endpoint = _finite_number(observed["clean_endpoint"], label="clean endpoint")
        base_logit_max = _finite_number(
            observed["clean_base_logit_absolute_max"],
            label="clean base logit maximum",
        )
        donor_logit_max = _finite_number(
            observed["clean_donor_logit_absolute_max"],
            label="clean donor logit maximum",
        )
        logit_max = _finite_number(
            observed["clean_logit_absolute_max"], label="clean logit maximum"
        )
        if base_margin <= 0.0 or donor_margin <= 0.0:
            raise RuntimeError("E29.14c clean correctness gate failed")
        if endpoint <= 0.0 or endpoint != base_margin + donor_margin:
            raise RuntimeError("E29.14c clean endpoint arithmetic changed")
        if (
            min(base_logit_max, donor_logit_max, logit_max) < 0.0
            or logit_max != max(base_logit_max, donor_logit_max)
            or logit_max > logit_limit
        ):
            raise RuntimeError("E29.14c clean numerical magnitude gate failed")
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


def _full_mask_panel(
    masks: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    validated = validate_scale_masks(masks)
    singleton = torch.zeros((1, COMPONENT_COUNT), dtype=torch.float64)
    singleton[0, HUB_COMPONENT] = 1.0
    full = torch.cat((singleton, *(validated[split] for split in MASK_SPLITS)))
    expected_rows = 1 + sum(MASK_SHAPES[split][0] for split in MASK_SPLITS)
    if tuple(full.shape) != (expected_rows, COMPONENT_COUNT):
        raise RuntimeError("E29.14c full mask-panel shape changed")
    return validated, full


def build_condition_manifest() -> list[dict[str, Any]]:
    """Return the exact flat logical label order sealed for E29.14c."""
    manifest = []
    condition_index = 0
    groups = (("singleton_hub", 1),) + tuple(
        (split, MASK_SHAPES[split][0]) for split in MASK_SPLITS
    )
    for group, rows in groups:
        for row in range(rows):
            for scope in POSITION_SCOPE_ORDER:
                manifest.append(
                    {
                        "condition_index": condition_index,
                        "mask_group": group,
                        "mask_row": row,
                        "scope": scope,
                        "alpha": 1.0,
                    }
                )
                condition_index += 1
    if len(manifest) != QUERY_ACCOUNTING["patched_condition_rows_per_context"]:
        raise RuntimeError("E29.14c condition-manifest length changed")
    return manifest


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


def _verify_batch_smoke(value: Any, *, mask_rows: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("E29.14c batch smoke must be a mapping")
    smoke = dict(value)
    _require_exact_keys(smoke, _SMOKE_KEYS, label="E29.14c batch smoke")
    records = smoke["records"]
    if not isinstance(records, Mapping) or tuple(records) != POSITION_SCOPE_ORDER:
        raise RuntimeError("E29.14c batch-smoke scope order changed")
    for scope in POSITION_SCOPE_ORDER:
        record = records[scope]
        if not isinstance(record, Mapping):
            raise TypeError("E29.14c batch-smoke record must be a mapping")
        _require_exact_keys(
            record, _SMOKE_RECORD_KEYS, label=f"E29.14c {scope} smoke record"
        )
        expected = {
            "mask_rows": mask_rows,
            "reference_batch_size": BATCH_INVARIANCE["reference_batch_size"],
            "candidate_batch_size": BATCH_INVARIANCE["candidate_batch_size"],
            "atol": BATCH_INVARIANCE["absolute_tolerance"],
            "rtol": BATCH_INVARIANCE["relative_tolerance"],
            "passed": True,
        }
        if any(record[key] != expected_value for key, expected_value in expected.items()):
            raise RuntimeError("E29.14c batch invariance gate failed")
        absolute = _finite_number(
            record["maximum_absolute_difference"],
            label=f"{scope} smoke absolute difference",
        )
        relative = _finite_number(
            record["maximum_relative_difference"],
            label=f"{scope} smoke relative difference",
        )
        if absolute < 0.0 or relative < 0.0:
            raise RuntimeError("E29.14c batch-invariance differences are invalid")
    if (
        smoke["mask_rows"] != mask_rows
        or smoke["unique_mask_scope_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_unique_condition_rows"]
        or smoke["physical_mask_scope_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"]
        or smoke["scientific_labels_produced"] is not False
        or smoke["passed"] is not True
    ):
        raise RuntimeError("E29.14c batch invariance gate failed")
    return smoke


def acquire_component_position_panel_with_runner(
    runner: Any,
    *,
    prompt_checks: Sequence[Mapping[str, Any]],
    masks: Mapping[str, torch.Tensor],
    clean_metadata_contract: Sequence[Mapping[str, Any]],
    registration_sha256: str = "pure_runner",
    expected_registration_sha256_supplied: str = "pure_runner",
) -> dict[str, Any]:
    """Acquire all 286 conditions per context after every clean gate passes."""
    records = prompt_records(prompt_checks)
    sealed_clean_metadata = validate_clean_metadata_contract(
        clean_metadata_contract, prompt_checks
    )
    validated_masks, full_masks = _full_mask_panel(masks)
    prepared = tuple(runner.prepare_context(record) for record in records)
    clean_metadata, base_margins, donor_margins, endpoints = (
        _verify_clean_metadata(prompt_checks, prepared, sealed_clean_metadata)
    )

    all_rows = []
    lexical_rows = []
    for context in prepared:
        paired = runner.paired_responses_from_prepared(context, full_masks)
        if not isinstance(paired, Mapping) or tuple(paired) != POSITION_SCOPE_ORDER:
            raise RuntimeError("E29.14c paired response scope order changed")
        all_rows.append(
            _response_vector(
                paired["all_positions"],
                rows=len(full_masks),
                label="all-position",
            )
        )
        lexical_rows.append(
            _response_vector(
                paired["differing_lexical_positions"],
                rows=len(full_masks),
                label="differing-lexical-position",
            )
        )

    all_matrix = torch.stack(all_rows, dim=1)
    lexical_matrix = torch.stack(lexical_rows, dim=1)
    smoke = runner.full_batch_invariance_smoke(
        prepared[int(BATCH_INVARIANCE["context_index"])],
        full_masks,
        reference_batch_size=int(BATCH_INVARIANCE["reference_batch_size"]),
        candidate_batch_size=int(BATCH_INVARIANCE["candidate_batch_size"]),
    )
    smoke = _verify_batch_smoke(smoke, mask_rows=len(full_masks))

    offset = 1
    all_responses = {}
    lexical_responses = {}
    for split in MASK_SPLITS:
        next_offset = offset + MASK_SHAPES[split][0]
        all_responses[split] = all_matrix[offset:next_offset].clone()
        lexical_responses[split] = lexical_matrix[offset:next_offset].clone()
        offset = next_offset
    if offset != len(full_masks):
        raise RuntimeError("E29.14c response-table slicing changed")

    singleton_all = all_matrix[0].clone()
    singleton_lexical = lexical_matrix[0].clone()
    combined = torch.cat((all_matrix.flatten(), lexical_matrix.flatten()))
    numerical_summary = {
        "singleton_all_response": tensor_numerical_summary(singleton_all),
        "singleton_lexical_response": tensor_numerical_summary(singleton_lexical),
        "all_position_responses": {
            split: tensor_numerical_summary(all_responses[split])
            for split in MASK_SPLITS
        },
        "lexical_position_responses": {
            split: tensor_numerical_summary(lexical_responses[split])
            for split in MASK_SPLITS
        },
        "combined_responses": tensor_numerical_summary(combined),
        "clean_endpoints": tensor_numerical_summary(endpoints),
    }
    summaries = (
        numerical_summary["singleton_all_response"],
        numerical_summary["singleton_lexical_response"],
        *numerical_summary["all_position_responses"].values(),
        *numerical_summary["lexical_position_responses"].values(),
        numerical_summary["combined_responses"],
        numerical_summary["clean_endpoints"],
    )
    if not all(summary["passed"] for summary in summaries):
        raise RuntimeError("E29.14c final numerical summary gate failed")

    artifact = {
        "schema_version": "e29_14c_scale_scope_labels_v2",
        "registration_sha256": registration_sha256,
        "expected_registration_sha256_supplied": (
            expected_registration_sha256_supplied
        ),
        "condition_manifest": build_condition_manifest(),
        "singleton_all_response": singleton_all,
        "singleton_lexical_response": singleton_lexical,
        "all_position_responses": all_responses,
        "lexical_position_responses": lexical_responses,
        "clean_metadata": clean_metadata,
        "clean_base_correct_minus_donor_margins": base_margins,
        "clean_donor_correct_minus_base_margins": donor_margins,
        "clean_endpoints": endpoints,
        "batch_invariance_smoke": smoke,
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": numerical_summary,
        "all_integrity_gates_passed": True,
    }
    _require_exact_keys(artifact, LABEL_KEYS, label="E29.14c scale/scope labels")
    # Retain this assertion beside construction so no validated split is silently
    # omitted merely because both mappings happen to have matching shapes.
    if tuple(all_responses) != tuple(validated_masks):
        raise RuntimeError("E29.14c response split order changed")
    return artifact


def acquire_scale_scope(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    output_directory: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Verify the seal, consume one attempt, acquire, and write custody once."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    attempt_path = output_directory / "scale_scope_acquisition_attempt.json"
    labels_path = output_directory / "scale_scope_labels.pt"
    custody_path = output_directory / "scale_scope_acquisition_custody.json"
    if any(path.exists() for path in (attempt_path, labels_path, custody_path)):
        raise FileExistsError("refusing to rerun or overwrite E29.14c custody")

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
        raise RuntimeError("E29.14c registration changed during verification")
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14c runtime environment differs from registration")
    execution = registration["execution"]
    if (
        batch_size != BATCH_SIZE
        or batch_size != execution["batch_size"]
        or execution["device"] != "cpu"
        or execution["model_dtype"] != "float64"
        or execution["attention_implementation"] != "eager"
    ):
        raise RuntimeError("E29.14c runtime execution differs from registration")
    if (
        registration["query_accounting"] != QUERY_ACCOUNTING
        or registration["physical_query_accounting"] != PHYSICAL_QUERY_ACCOUNTING
    ):
        raise RuntimeError("E29.14c registered query accounting changed")

    masks = load_verified_protocol_masks(
        Path(registration["protocol_masks"]),
        expected_sha256=str(registration["protocol_masks_sha256"]),
    )
    masks = validate_scale_masks(masks)
    attempt = build_acquisition_attempt_record(
        registration=registration,
        registration_path=registration_path,
        observed_registration_sha256=observed_registration_sha,
        expected_registration_sha256=expected_registration_sha256,
    )
    attempt_sha = create_acquisition_attempt_marker(attempt, attempt_path)

    snapshot = Path(registration["model_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float64,
    ).eval()
    runner = Float64ComponentPositionRunner(model, tokenizer, batch_size=batch_size)
    expected_components = tuple(
        (int(layer), str(kind)) for layer, kind in registration["component_map"]
    )
    if runner.components != expected_components:
        raise RuntimeError("E29.14c runtime component map changed")
    artifact = acquire_component_position_panel_with_runner(
        runner,
        prompt_checks=registration["prompt_checks"],
        masks=masks,
        clean_metadata_contract=registration["clean_metadata_contract"],
        registration_sha256=observed_registration_sha,
        expected_registration_sha256_supplied=expected_registration_sha256,
    )

    response_maximum = float(
        artifact["numerical_summary"]["combined_responses"]["absolute_max"]
    )
    atomic_torch_save(artifact, labels_path)
    custody = {
        "schema_version": "e29_14c_scale_scope_acquisition_custody_v2",
        "status": "all_registered_scale_scope_conditions_acquired_after_integrity_gates",
        "registration": str(registration_path.resolve()),
        "registration_sha256": observed_registration_sha,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "source_code_sha256_verified_before_acquisition": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_acquisition": dict(
            registration["checkpoint_sha256"]
        ),
        "protocol_masks": str(Path(registration["protocol_masks"]).resolve()),
        "protocol_masks_sha256_verified_before_acquisition": registration[
            "protocol_masks_sha256"
        ],
        "protocol_masks_schema": registration["protocol_masks_schema"],
        "acquisition_attempt": str(attempt_path.resolve()),
        "acquisition_attempt_sha256": attempt_sha,
        "scale_scope_labels": str(labels_path.resolve()),
        "scale_scope_labels_sha256": sha256(labels_path),
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "clean_metadata_contract_sha256": registration[
            "clean_metadata_contract_sha256"
        ],
        "scale_scope_protocol_sha256": registration["scale_scope_protocol_sha256"],
        "contexts": QUERY_ACCOUNTING["contexts"],
        # The runner panel contains the registered singleton followed by the
        # 142 multi-component rows.  The singleton remains separate in the
        # logical accounting because both of its scope responses are controls.
        "mask_rows": 1 + QUERY_ACCOUNTING["multi_component_masks"],
        "position_scopes": QUERY_ACCOUNTING["position_scopes"],
        "patched_condition_rows_per_context": QUERY_ACCOUNTING[
            "patched_condition_rows_per_context"
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
    _require_exact_keys(custody, CUSTODY_KEYS, label="E29.14c acquisition custody")
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
            acquire_scale_scope(
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
    "CLEAN_METADATA_KEYS",
    "CUSTODY_KEYS",
    "LABEL_KEYS",
    "acquire_component_position_panel_with_runner",
    "acquire_scale_scope",
    "build_condition_manifest",
    "prompt_records",
]
