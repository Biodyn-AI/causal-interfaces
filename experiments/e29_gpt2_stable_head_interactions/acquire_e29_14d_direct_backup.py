"""Acquire the sealed E29.14d direct causal-carrier panel once.

All clean contexts and response-blind update-norm diagnostics are validated
before any patched response.  The immutable attempt marker is written before
the tokenizer or model is loaded.  Batch-smoke outputs never enter labels.
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
from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    build_condition_manifest,
    build_query_manifest,
    validate_condition_masks,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_direct_backup_custody import (
    BATCH_INVARIANCE,
    PROTOCOL_ARTIFACT_SCHEMA,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    atomic_torch_save,
    build_acquisition_attempt_record,
    create_acquisition_attempt_marker,
    load_verified_protocol_artifact,
    load_verified_registration,
    runtime_environment,
    sha256,
    validate_clean_metadata_contract,
    validate_registration_payload,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.14d-DIRECT-CAUSAL-CARRIER-REGISTRATION.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14d_direct_backup_v1"

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
        "query_manifest",
        "responses_by_scope",
        "clean_metadata",
        "clean_base_correct_minus_donor_margins",
        "clean_donor_correct_minus_base_margins",
        "clean_endpoints",
        "clean_reference_update_frobenius_norms",
        "clean_reference_update_rms_norms",
        "clean_reference_norm_semantics",
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
        "protocol_artifact",
        "protocol_artifact_sha256_verified_before_acquisition",
        "protocol_artifact_schema",
        "acquisition_attempt",
        "acquisition_attempt_sha256",
        "direct_backup_labels",
        "direct_backup_labels_sha256",
        "prompt_panel_sha256",
        "prompt_records_sha256",
        "posthoc_candidate_provenance_sha256",
        "prior_mask_audit_sha256",
        "clean_metadata_contract_sha256",
        "causal_carrier_protocol_sha256",
        "condition_manifest_sha256",
        "query_manifest_sha256",
        "contexts",
        "condition_masks",
        "position_scopes",
        "patched_condition_rows_per_context",
        "patched_context_condition_cells",
        "clean_forward_examples",
        "batch_smoke_unique_query_rows",
        "batch_smoke_physical_condition_cells",
        "maximum_logical_queries",
        "maximum_physical_model_examples",
        "batch_size",
        "batch_invariance_passed",
        "all_clean_and_numerical_gates_passed",
        "clean_reference_norms_are_diagnostic_only",
        "device",
        "dtype",
        "response_absolute_max",
        "runtime_seconds",
    }
)


def _jsonable(value: Any) -> Any:
    """Match strict-JSON registration encoding for tuples and mappings."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


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
    if not isinstance(prompt_checks, Sequence):
        raise TypeError("E29.14d prompt checks must be a sequence")
    records = tuple(
        {key: str(check[key]) for key in PROMPT_KEYS} for check in prompt_checks
    )
    if len(records) != CONTEXT_COUNT:
        raise RuntimeError("E29.14d requires exactly twelve registered prompts")
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
    expected_order = ("all_positions", "differing_lexical_positions")
    if not isinstance(scopes, Mapping) or tuple(scopes) != expected_order:
        raise RuntimeError("E29.14d runtime position-scope order changed")
    token_length = int(observed["token_length"])
    for name, indices in scopes.items():
        if not isinstance(indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise RuntimeError(f"E29.14d {name} indices must be an integer list")
        if indices != sorted(set(indices)) or any(
            index < 0 or index >= token_length for index in indices
        ):
            raise RuntimeError(f"E29.14d {name} indices are invalid")
    if scopes["all_positions"] != list(range(token_length)):
        raise RuntimeError("E29.14d all-position scope changed")
    base_ids = context.base_input_ids.detach().cpu()
    donor_ids = context.donor_input_ids.detach().cpu()
    if tuple(base_ids.shape) != (1, token_length) or base_ids.shape != donor_ids.shape:
        raise RuntimeError("E29.14d prepared token rows changed")
    expected_lexical = [
        int(index) for index in torch.where(base_ids[0] != donor_ids[0])[0]
    ]
    if (
        not expected_lexical
        or scopes["differing_lexical_positions"] != expected_lexical
    ):
        raise RuntimeError("E29.14d differing-lexical scope changed")


def _verify_clean_metadata(
    prompt_checks: Sequence[Mapping[str, Any]],
    prepared: Sequence[Any],
    clean_metadata_contract: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate all clean contexts before the first patched response."""
    if len(prepared) != CONTEXT_COUNT:
        raise RuntimeError("E29.14d prepared-context count changed")
    metadata: list[dict[str, Any]] = []
    base_margins: list[float] = []
    donor_margins: list[float] = []
    endpoints: list[float] = []
    logit_limit = float(NUMERICAL_GATES["clean_logit_absolute_max"])
    for registered, sealed, context in zip(
        prompt_checks, clean_metadata_contract, prepared, strict=True
    ):
        observed = dict(context.clean_metadata)
        _require_exact_keys(
            observed, CLEAN_METADATA_KEYS, label="E29.14d clean runtime metadata"
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
            raise RuntimeError("runtime clean metadata differs from its seal")
        for side in ("base", "donor"):
            if int(observed[f"{side}_target_token_id"]) != int(
                registered["sides"][side]["target_token_id"]
            ):
                raise RuntimeError("runtime target token differs from registration")
        fixed = {
            "model_layer_count": 24,
            "component_count": COMPONENT_COUNT,
            "component_semantics": "attention_and_mlp_residual_updates",
            "patch_direction": "donor_to_base_only",
            "position_scope_order": [
                "all_positions",
                "differing_lexical_positions",
            ],
            "clean_logits_finite": True,
            "device": "cpu",
            "model_dtype": "float64",
        }
        if any(observed[key] != value for key, value in fixed.items()):
            raise RuntimeError("E29.14d clean runtime contract changed")
        _verify_position_scopes(context, observed)
        base = _finite_number(
            observed["clean_base_correct_minus_donor_margin"],
            label="clean base correctness margin",
        )
        donor = _finite_number(
            observed["clean_donor_correct_minus_base_margin"],
            label="clean donor correctness margin",
        )
        endpoint = _finite_number(observed["clean_endpoint"], label="clean endpoint")
        if base <= 0.0:
            raise RuntimeError("clean base correctness gate failed")
        if donor <= 0.0:
            raise RuntimeError("clean donor correctness gate failed")
        if endpoint <= 0.0 or not math.isclose(
            endpoint, base + donor, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError("clean endpoint gate failed")
        clean_max = _finite_number(
            observed["clean_logit_absolute_max"], label="clean logit maximum"
        )
        if clean_max < 0.0 or clean_max > logit_limit:
            raise RuntimeError("clean logit magnitude gate failed")
        metadata.append(observed)
        base_margins.append(base)
        donor_margins.append(donor)
        endpoints.append(endpoint)
    return (
        metadata,
        torch.tensor(base_margins, dtype=torch.float64),
        torch.tensor(donor_margins, dtype=torch.float64),
        torch.tensor(endpoints, dtype=torch.float64),
    )


def _clean_reference_update_norms(
    prepared: Sequence[Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Measure clean donor-minus-base deltas; these are not injected norms."""
    frobenius: dict[str, list[torch.Tensor]] = {
        scope: [] for scope in POSITION_SCOPE_ORDER
    }
    rms: dict[str, list[torch.Tensor]] = {scope: [] for scope in POSITION_SCOPE_ORDER}
    for context in prepared:
        if (
            len(context.base_updates) != COMPONENT_COUNT
            or len(context.donor_updates) != COMPONENT_COUNT
        ):
            raise RuntimeError("E29.14d clean component-update count changed")
        for scope in POSITION_SCOPE_ORDER:
            position = context.position_scopes[scope].detach().cpu().bool()
            component_frobenius = []
            component_rms = []
            for base, donor in zip(
                context.base_updates, context.donor_updates, strict=True
            ):
                delta = donor.detach().cpu().double() - base.detach().cpu().double()
                if delta.ndim != 2 or len(delta) != len(position):
                    raise RuntimeError("E29.14d clean update shape changed")
                selected = delta[position]
                if selected.numel() == 0 or not bool(selected.isfinite().all()):
                    raise RuntimeError("E29.14d clean reference delta is invalid")
                component_frobenius.append(torch.linalg.vector_norm(selected))
                component_rms.append(torch.sqrt(torch.mean(selected.square())))
            frobenius[scope].append(torch.stack(component_frobenius))
            rms[scope].append(torch.stack(component_rms))
    return (
        {scope: torch.stack(rows, dim=1) for scope, rows in frobenius.items()},
        {scope: torch.stack(rows, dim=1) for scope, rows in rms.items()},
    )


def _response_vector(value: Any, *, rows: int, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} response must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise RuntimeError(f"{label} response device or dtype changed")
    if tuple(value.shape) != (rows,) or not bool(value.isfinite().all()):
        raise RuntimeError(f"{label} response shape or finiteness changed")
    if float(value.abs().max()) > float(NUMERICAL_GATES["response_absolute_max"]):
        raise RuntimeError(f"{label} response exceeds the magnitude gate")
    return value.detach().clone()


def _batch_invariance_smoke(
    runner: Any, context: Any, masks: torch.Tensor
) -> dict[str, Any]:
    records = {}
    for scope in POSITION_SCOPE_ORDER:
        reference = _response_vector(
            runner.responses_from_prepared(
                context,
                masks,
                position_scope=scope,
                batch_size=int(BATCH_INVARIANCE["reference_batch_size"]),
            ),
            rows=len(masks),
            label=f"{scope} reference batch smoke",
        )
        candidate = _response_vector(
            runner.responses_from_prepared(
                context,
                masks,
                position_scope=scope,
                batch_size=int(BATCH_INVARIANCE["candidate_batch_size"]),
            ),
            rows=len(masks),
            label=f"{scope} candidate batch smoke",
        )
        differences = (candidate - reference).abs()
        atol = float(BATCH_INVARIANCE["absolute_tolerance"])
        rtol = float(BATCH_INVARIANCE["relative_tolerance"])
        records[scope] = {
            "mask_rows": len(masks),
            "reference_batch_size": BATCH_INVARIANCE["reference_batch_size"],
            "candidate_batch_size": BATCH_INVARIANCE["candidate_batch_size"],
            "maximum_absolute_difference": float(differences.max()),
            "maximum_relative_difference": float(
                (differences / reference.abs().clamp_min(atol)).max()
            ),
            "atol": atol,
            "rtol": rtol,
            "passed": bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol)),
        }
    smoke = {
        "records": records,
        "mask_rows": len(masks),
        "unique_condition_rows": len(POSITION_SCOPE_ORDER) * len(masks),
        "physical_condition_cells": 2 * len(POSITION_SCOPE_ORDER) * len(masks),
        "scientific_labels_produced": False,
        "passed": all(record["passed"] for record in records.values()),
    }
    if (
        smoke["unique_condition_rows"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_unique_query_rows"]
        or smoke["physical_condition_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"]
        or smoke["passed"] is not True
    ):
        raise RuntimeError("E29.14d batch invariance gate failed")
    return smoke


def acquire_direct_backup_panel_with_runner(
    runner: Any,
    *,
    prompt_checks: Sequence[Mapping[str, Any]],
    condition_masks: torch.Tensor,
    clean_metadata_contract: Sequence[Mapping[str, Any]],
    registration_sha256: str = "pure_runner",
    expected_registration_sha256_supplied: str = "pure_runner",
) -> dict[str, Any]:
    """Acquire every fixed condition at both scopes after all clean gates."""
    records = prompt_records(prompt_checks)
    sealed_clean = validate_clean_metadata_contract(
        clean_metadata_contract, prompt_checks
    )
    validated = validate_condition_masks(condition_masks)
    masks = condition_masks if validated is None else validated
    masks = masks.detach().cpu().to(torch.float64)
    prepared = tuple(runner.prepare_context(record) for record in records)
    clean_metadata, base_margins, donor_margins, endpoints = _verify_clean_metadata(
        prompt_checks, prepared, sealed_clean
    )
    frobenius, rms = _clean_reference_update_norms(prepared)

    responses_by_scope = {}
    for scope in POSITION_SCOPE_ORDER:
        rows = [
            _response_vector(
                runner.responses_from_prepared(context, masks, position_scope=scope),
                rows=len(masks),
                label=scope,
            )
            for context in prepared
        ]
        responses_by_scope[scope] = torch.stack(rows, dim=1)
    smoke = _batch_invariance_smoke(
        runner, prepared[int(BATCH_INVARIANCE["context_index"])], masks
    )

    combined = torch.cat(
        [responses_by_scope[scope].flatten() for scope in POSITION_SCOPE_ORDER]
    )
    numerical_summary = {
        "responses_by_scope": {
            scope: tensor_numerical_summary(responses_by_scope[scope])
            for scope in POSITION_SCOPE_ORDER
        },
        "combined_responses": tensor_numerical_summary(combined),
        "clean_endpoints": tensor_numerical_summary(endpoints),
        "clean_reference_update_frobenius_norms": {
            scope: {
                "shape": list(frobenius[scope].shape),
                "dtype": "float64",
                "finite": bool(frobenius[scope].isfinite().all()),
                "absolute_max": float(frobenius[scope].max()),
            }
            for scope in POSITION_SCOPE_ORDER
        },
        "clean_reference_update_rms_norms": {
            scope: {
                "shape": list(rms[scope].shape),
                "dtype": "float64",
                "finite": bool(rms[scope].isfinite().all()),
                "absolute_max": float(rms[scope].max()),
            }
            for scope in POSITION_SCOPE_ORDER
        },
    }
    if not all(
        summary["passed"]
        for summary in (
            *numerical_summary["responses_by_scope"].values(),
            numerical_summary["combined_responses"],
            numerical_summary["clean_endpoints"],
        )
    ) or not all(
        summary["finite"]
        for group in (
            numerical_summary["clean_reference_update_frobenius_norms"],
            numerical_summary["clean_reference_update_rms_norms"],
        )
        for summary in group.values()
    ):
        raise RuntimeError("E29.14d final numerical summary gate failed")

    artifact = {
        "schema_version": "e29_14d_direct_backup_labels_v1",
        "registration_sha256": registration_sha256,
        "expected_registration_sha256_supplied": (
            expected_registration_sha256_supplied
        ),
        "condition_manifest": list(build_condition_manifest()),
        "query_manifest": list(build_query_manifest()),
        "responses_by_scope": responses_by_scope,
        "clean_metadata": clean_metadata,
        "clean_base_correct_minus_donor_margins": base_margins,
        "clean_donor_correct_minus_base_margins": donor_margins,
        "clean_endpoints": endpoints,
        "clean_reference_update_frobenius_norms": frobenius,
        "clean_reference_update_rms_norms": rms,
        "clean_reference_norm_semantics": {
            "quantity": "clean_donor_update_minus_clean_base_update",
            "scopes": list(POSITION_SCOPE_ORDER),
            "frobenius": "l2_norm_over_selected_positions_and_hidden_coordinates",
            "rms": "root_mean_square_over_selected_positions_and_hidden_coordinates",
            "not_actual_multi_mask_injected_norms": True,
            "not_used_for_condition_selection_or_dosing": True,
            "descriptive_only": True,
        },
        "batch_invariance_smoke": smoke,
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": numerical_summary,
        "all_integrity_gates_passed": True,
    }
    _require_exact_keys(artifact, LABEL_KEYS, label="E29.14d direct-backup labels")
    return artifact


def acquire_direct_backup(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    output_directory: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Verify the seal, consume one attempt, acquire, and write custody once."""
    started = time.perf_counter()
    attempt_path = output_directory / "direct_backup_acquisition_attempt.json"
    labels_path = output_directory / "direct_backup_labels.pt"
    custody_path = output_directory / "direct_backup_acquisition_custody.json"
    if any(path.exists() for path in (attempt_path, labels_path, custody_path)):
        raise FileExistsError("refusing to rerun or overwrite E29.14d custody")

    registration = load_verified_registration(
        registration_path,
        expected_sha256=expected_registration_sha256,
        required_keys=REGISTRATION_REQUIRED_KEYS,
    )
    registration = validate_registration_payload(
        registration, source_directory=SOURCE_DIRECTORY, verify_external_files=True
    )
    observed_registration_sha = sha256(registration_path)
    if observed_registration_sha != expected_registration_sha256:
        raise RuntimeError("E29.14d registration changed during verification")
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14d runtime environment differs from registration")
    execution = registration["execution"]
    if (
        batch_size != BATCH_SIZE
        or batch_size != execution["batch_size"]
        or execution["device"] != "cpu"
        or execution["model_dtype"] != "float64"
        or execution["attention_implementation"] != "eager"
    ):
        raise RuntimeError("E29.14d runtime execution differs from registration")
    if registration["query_accounting"] != _jsonable(QUERY_ACCOUNTING) or registration[
        "physical_query_accounting"
    ] != _jsonable(PHYSICAL_QUERY_ACCOUNTING):
        raise RuntimeError("E29.14d registered query accounting changed")
    masks = load_verified_protocol_artifact(
        Path(registration["protocol_artifact"]),
        expected_sha256=registration["protocol_artifact_sha256"],
    )
    attempt = build_acquisition_attempt_record(
        registration=registration,
        registration_path=registration_path,
        observed_registration_sha256=observed_registration_sha,
        expected_registration_sha256=expected_registration_sha256,
    )
    attempt_sha = create_acquisition_attempt_marker(attempt, attempt_path)

    # This import and both loads deliberately occur only after the attempt is
    # durable. Any crash from here onward consumes the registered attempt.
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        raise RuntimeError("E29.14d runtime component map changed")
    labels = acquire_direct_backup_panel_with_runner(
        runner,
        prompt_checks=registration["prompt_checks"],
        condition_masks=masks,
        clean_metadata_contract=registration["clean_metadata_contract"],
        registration_sha256=observed_registration_sha,
        expected_registration_sha256_supplied=expected_registration_sha256,
    )
    response_maximum = float(
        labels["numerical_summary"]["combined_responses"]["absolute_max"]
    )
    atomic_torch_save(labels, labels_path)
    custody = {
        "schema_version": "e29_14d_direct_backup_acquisition_custody_v1",
        "status": "all_registered_direct_backup_conditions_acquired_after_integrity_gates",
        "registration": str(registration_path.resolve()),
        "registration_sha256": observed_registration_sha,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "source_code_sha256_verified_before_acquisition": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_acquisition": dict(
            registration["checkpoint_sha256"]
        ),
        "protocol_artifact": str(Path(registration["protocol_artifact"]).resolve()),
        "protocol_artifact_sha256_verified_before_acquisition": registration[
            "protocol_artifact_sha256"
        ],
        "protocol_artifact_schema": PROTOCOL_ARTIFACT_SCHEMA,
        "acquisition_attempt": str(attempt_path.resolve()),
        "acquisition_attempt_sha256": attempt_sha,
        "direct_backup_labels": str(labels_path.resolve()),
        "direct_backup_labels_sha256": sha256(labels_path),
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "prompt_records_sha256": registration["prompt_records_sha256"],
        "posthoc_candidate_provenance_sha256": registration[
            "posthoc_candidate_provenance_sha256"
        ],
        "prior_mask_audit_sha256": registration["prior_mask_audit_sha256"],
        "clean_metadata_contract_sha256": registration[
            "clean_metadata_contract_sha256"
        ],
        "causal_carrier_protocol_sha256": registration[
            "causal_carrier_protocol_sha256"
        ],
        "condition_manifest_sha256": registration["condition_manifest_sha256"],
        "query_manifest_sha256": registration["query_manifest_sha256"],
        "contexts": QUERY_ACCOUNTING["contexts"],
        "condition_masks": QUERY_ACCOUNTING["unique_condition_masks"],
        "position_scopes": QUERY_ACCOUNTING["position_scopes"],
        "patched_condition_rows_per_context": QUERY_ACCOUNTING[
            "patched_condition_rows_per_context"
        ],
        "patched_context_condition_cells": QUERY_ACCOUNTING[
            "patched_context_condition_cells"
        ],
        "clean_forward_examples": QUERY_ACCOUNTING["clean_forward_examples"],
        "batch_smoke_unique_query_rows": PHYSICAL_QUERY_ACCOUNTING[
            "batch_smoke_unique_query_rows"
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
        "clean_reference_norms_are_diagnostic_only": True,
        "device": "cpu",
        "dtype": "float64",
        "response_absolute_max": response_maximum,
        "runtime_seconds": time.perf_counter() - started,
    }
    _require_exact_keys(custody, CUSTODY_KEYS, label="E29.14d acquisition custody")
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
            acquire_direct_backup(
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
    "acquire_direct_backup",
    "acquire_direct_backup_panel_with_runner",
    "prompt_records",
]
