"""Verify custody and score the sealed E29.14c scale/scope ladder once."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14c_scale_scope import (
    CLEAN_METADATA_KEYS,
    CUSTODY_KEYS,
    DEFAULT_OUTPUT,
    LABEL_KEYS,
    build_condition_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
    tensor_numerical_summary,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    ATTEMPT_REQUIRED_KEYS,
    BATCH_INVARIANCE,
    CLEAN_METADATA_CONTRACT_KEYS,
    MASK_ARTIFACT_SCHEMA,
    PARENT_E29_14B_REGISTRATION_SHA256,
    PARENT_E29_14B_SCORE_SUMMARY_SHA256,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    load_verified_protocol_masks,
    load_verified_registration,
    require_exact_keys,
    runtime_environment,
    sha256,
    validate_registration_payload,
    verify_file_hash,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    MASK_SHAPES,
    MASK_SPLITS,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    score_scale_scope_ladder,
    validate_scale_masks,
    validate_scale_scope_protocol,
)

SOURCE_DIRECTORY = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be finite")
    return number


def _finite_vector(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise RuntimeError(f"{label} device or dtype changed")
    vector = value.detach().clone()
    if tuple(vector.shape) != (CONTEXT_COUNT,) or not bool(vector.isfinite().all()):
        raise RuntimeError(f"{label} shape or finiteness changed")
    return vector


def _finite_response_tables(value: Any, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or tuple(value) != MASK_SPLITS:
        raise RuntimeError(f"{label} response keys or order changed")
    tables = {}
    for split in MASK_SPLITS:
        table = value[split]
        if not isinstance(table, torch.Tensor):
            raise TypeError(f"{label}/{split} must be a tensor")
        if table.device.type != "cpu" or table.dtype != torch.float64:
            raise RuntimeError(f"{label}/{split} response device or dtype changed")
        table = table.detach().clone()
        expected_shape = (MASK_SHAPES[split][0], CONTEXT_COUNT)
        if tuple(table.shape) != expected_shape:
            raise RuntimeError(f"{label}/{split} response shape changed")
        if not bool(table.isfinite().all()):
            raise RuntimeError(f"{label}/{split} response is non-finite")
        if float(table.abs().max()) > float(NUMERICAL_GATES["response_absolute_max"]):
            raise RuntimeError(f"{label}/{split} response magnitude changed")
        tables[split] = table
    return tables


def _validate_smoke(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("E29.14c batch-invariance record must be a mapping")
    top_keys = {
        "records",
        "mask_rows",
        "unique_mask_scope_cells",
        "physical_mask_scope_cells",
        "scientific_labels_produced",
        "passed",
    }
    require_exact_keys(value, top_keys, label="E29.14c batch smoke")
    expected_mask_rows = 1 + QUERY_ACCOUNTING["multi_component_masks"]
    if (
        value["mask_rows"] != expected_mask_rows
        or value["unique_mask_scope_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_unique_condition_rows"]
        or value["physical_mask_scope_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"]
        or value["scientific_labels_produced"] is not False
        or value["passed"] is not True
    ):
        raise RuntimeError("E29.14c batch-invariance integrity changed")
    records = value["records"]
    if not isinstance(records, Mapping) or tuple(records) != POSITION_SCOPE_ORDER:
        raise RuntimeError("E29.14c batch-invariance scope order changed")
    record_keys = {
        "mask_rows",
        "reference_batch_size",
        "candidate_batch_size",
        "maximum_absolute_difference",
        "maximum_relative_difference",
        "atol",
        "rtol",
        "passed",
    }
    for record in records.values():
        if not isinstance(record, Mapping):
            raise TypeError("E29.14c smoke record must be a mapping")
        require_exact_keys(record, record_keys, label="E29.14c smoke record")
        if (
            record["mask_rows"] != expected_mask_rows
            or record["reference_batch_size"]
            != BATCH_INVARIANCE["reference_batch_size"]
            or record["candidate_batch_size"]
            != BATCH_INVARIANCE["candidate_batch_size"]
            or record["atol"] != BATCH_INVARIANCE["absolute_tolerance"]
            or record["rtol"] != BATCH_INVARIANCE["relative_tolerance"]
            or record["passed"] is not True
        ):
            raise RuntimeError("E29.14c batch-invariance scope failed")
        for key in ("maximum_absolute_difference", "maximum_relative_difference"):
            if _finite_number(record[key], label=key) < 0.0:
                raise RuntimeError("E29.14c batch-invariance difference is negative")


def _validate_clean_metadata(
    value: Any,
    *,
    base_margins: torch.Tensor,
    donor_margins: torch.Tensor,
    endpoints: torch.Tensor,
    clean_metadata_contract: Any,
) -> None:
    if not isinstance(value, list) or len(value) != CONTEXT_COUNT:
        raise RuntimeError("E29.14c clean metadata count changed")
    if (
        not isinstance(clean_metadata_contract, (tuple, list))
        or len(clean_metadata_contract) != CONTEXT_COUNT
    ):
        raise RuntimeError("E29.14c registered clean metadata contract count changed")
    for index, (record, registered_record) in enumerate(
        zip(value, clean_metadata_contract, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise TypeError("E29.14c clean metadata must be mappings")
        if not isinstance(registered_record, Mapping):
            raise TypeError(
                "E29.14c registered clean metadata contract must contain mappings"
            )
        require_exact_keys(record, CLEAN_METADATA_KEYS, label="E29.14c clean metadata")
        require_exact_keys(
            registered_record,
            CLEAN_METADATA_CONTRACT_KEYS,
            label="E29.14c registered clean metadata contract",
        )
        if registered_record["context_index"] != index:
            raise RuntimeError(
                "E29.14c registered clean metadata contract order changed"
            )
        registered_scopes = registered_record["position_scope_indices"]
        registered_token_length = registered_record["token_length"]
        if (
            isinstance(registered_token_length, bool)
            or not isinstance(registered_token_length, int)
            or registered_token_length < 1
            or not isinstance(registered_scopes, Mapping)
            or tuple(registered_scopes) != POSITION_SCOPE_ORDER
            or registered_scopes["all_positions"]
            != list(range(registered_token_length))
        ):
            raise RuntimeError(
                "E29.14c registered clean metadata contract position scopes changed"
            )
        registered_lexical = registered_scopes["differing_lexical_positions"]
        if (
            not isinstance(registered_lexical, list)
            or not registered_lexical
            or registered_lexical != sorted(set(registered_lexical))
            or any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                or position >= registered_token_length
                for position in registered_lexical
            )
        ):
            raise RuntimeError(
                "E29.14c registered clean metadata contract lexical positions changed"
            )
        registered_fields = CLEAN_METADATA_CONTRACT_KEYS - {"context_index"}
        if any(record[key] != registered_record[key] for key in registered_fields):
            raise RuntimeError(
                "E29.14c clean metadata differs from registered clean metadata contract"
            )
        fixed = {
            "model_layer_count": 24,
            "component_count": COMPONENT_COUNT,
            "component_semantics": "attention_and_mlp_residual_updates",
            "patch_direction": "donor_to_base_only",
            "position_scope_order": list(POSITION_SCOPE_ORDER),
            "clean_logits_finite": True,
            "device": "cpu",
            "model_dtype": "float64",
        }
        if any(record[key] != expected for key, expected in fixed.items()):
            raise RuntimeError("E29.14c clean metadata runtime contract changed")
        if (
            float(record["clean_base_correct_minus_donor_margin"])
            != float(base_margins[index])
            or float(record["clean_donor_correct_minus_base_margin"])
            != float(donor_margins[index])
            or float(record["clean_endpoint"]) != float(endpoints[index])
        ):
            raise RuntimeError("E29.14c clean metadata endpoint arithmetic changed")
        token_length = record["token_length"]
        scopes = record["position_scope_indices"]
        if (
            isinstance(token_length, bool)
            or not isinstance(token_length, int)
            or token_length < 1
            or not isinstance(scopes, Mapping)
            or tuple(scopes) != POSITION_SCOPE_ORDER
            or scopes["all_positions"] != list(range(token_length))
        ):
            raise RuntimeError("E29.14c clean metadata position scopes changed")
        lexical = scopes["differing_lexical_positions"]
        if (
            not isinstance(lexical, list)
            or not lexical
            or lexical != sorted(set(lexical))
            or any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                or position >= token_length
                for position in lexical
            )
        ):
            raise RuntimeError("E29.14c clean metadata lexical positions changed")
        base_max = _finite_number(
            record["clean_base_logit_absolute_max"], label="clean base logit maximum"
        )
        donor_max = _finite_number(
            record["clean_donor_logit_absolute_max"],
            label="clean donor logit maximum",
        )
        total_max = _finite_number(
            record["clean_logit_absolute_max"], label="clean logit maximum"
        )
        if (
            min(base_max, donor_max, total_max) < 0.0
            or total_max != max(base_max, donor_max)
            or total_max > float(NUMERICAL_GATES["clean_logit_absolute_max"])
        ):
            raise RuntimeError("E29.14c clean metadata numerical gate changed")


def _validate_numerical_summary(
    value: Any,
    *,
    singleton_all: torch.Tensor,
    singleton_lexical: torch.Tensor,
    all_position: Mapping[str, torch.Tensor],
    lexical: Mapping[str, torch.Tensor],
    endpoints: torch.Tensor,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("E29.14c numerical summary must be a mapping")
    required = {
        "singleton_all_response",
        "singleton_lexical_response",
        "all_position_responses",
        "lexical_position_responses",
        "combined_responses",
        "clean_endpoints",
    }
    require_exact_keys(value, required, label="E29.14c numerical summary")
    expected_all = {
        split: tensor_numerical_summary(all_position[split]) for split in MASK_SPLITS
    }
    expected_lexical = {
        split: tensor_numerical_summary(lexical[split]) for split in MASK_SPLITS
    }
    combined = torch.cat(
        (
            singleton_all,
            *(all_position[split].flatten() for split in MASK_SPLITS),
            singleton_lexical,
            *(lexical[split].flatten() for split in MASK_SPLITS),
        )
    )
    expected = {
        "singleton_all_response": tensor_numerical_summary(singleton_all),
        "singleton_lexical_response": tensor_numerical_summary(singleton_lexical),
        "all_position_responses": expected_all,
        "lexical_position_responses": expected_lexical,
        "combined_responses": tensor_numerical_summary(combined),
        "clean_endpoints": tensor_numerical_summary(endpoints),
    }
    if value != expected:
        raise RuntimeError("E29.14c numerical summary differs from labels")
    records = (
        expected["singleton_all_response"],
        expected["singleton_lexical_response"],
        *expected_all.values(),
        *expected_lexical.values(),
        expected["combined_responses"],
        expected["clean_endpoints"],
    )
    if not all(record["passed"] for record in records):
        raise RuntimeError("E29.14c numerical integrity failed")


def _validate_label_integrity(
    labels: Mapping[str, Any],
    *,
    expected_registration_sha256: str,
    registered_protocol: Mapping[str, Any],
    held_masks: Mapping[str, torch.Tensor],
    clean_metadata_contract: Any,
) -> dict[str, Any]:
    require_exact_keys(labels, LABEL_KEYS, label="E29.14c labels")
    if labels["schema_version"] != "e29_14c_scale_scope_labels_v2":
        raise RuntimeError("E29.14c label schema changed")
    if (
        labels["registration_sha256"] != expected_registration_sha256
        or labels["expected_registration_sha256_supplied"]
        != expected_registration_sha256
    ):
        raise RuntimeError("E29.14c labels differ from the expected registration")
    validate_scale_scope_protocol(registered_protocol)
    masks = validate_scale_masks(held_masks)
    if labels["condition_manifest"] != build_condition_manifest():
        raise RuntimeError("E29.14c condition manifest changed")
    if labels["query_accounting"] != QUERY_ACCOUNTING:
        raise RuntimeError("E29.14c query accounting changed")
    if labels["physical_query_accounting"] != PHYSICAL_QUERY_ACCOUNTING:
        raise RuntimeError("E29.14c physical query accounting changed")
    if labels["all_integrity_gates_passed"] is not True:
        raise RuntimeError("E29.14c label integrity did not pass")
    _validate_smoke(labels["batch_invariance_smoke"])

    base_margins = _finite_vector(
        labels["clean_base_correct_minus_donor_margins"],
        label="clean base correctness margins",
    )
    donor_margins = _finite_vector(
        labels["clean_donor_correct_minus_base_margins"],
        label="clean donor correctness margins",
    )
    endpoints = _finite_vector(labels["clean_endpoints"], label="clean endpoints")
    if bool((base_margins <= 0.0).any() or (donor_margins <= 0.0).any()):
        raise RuntimeError("E29.14c clean correctness integrity changed")
    if not torch.equal(endpoints, base_margins + donor_margins):
        raise RuntimeError("E29.14c clean endpoint arithmetic changed")
    _validate_clean_metadata(
        labels["clean_metadata"],
        base_margins=base_margins,
        donor_margins=donor_margins,
        endpoints=endpoints,
        clean_metadata_contract=clean_metadata_contract,
    )
    singleton_all = _finite_vector(
        labels["singleton_all_response"], label="singleton all response"
    )
    singleton_lexical = _finite_vector(
        labels["singleton_lexical_response"], label="singleton lexical response"
    )
    all_position = _finite_response_tables(
        labels["all_position_responses"], label="all-position"
    )
    lexical = _finite_response_tables(
        labels["lexical_position_responses"], label="lexical-position"
    )
    _validate_numerical_summary(
        labels["numerical_summary"],
        singleton_all=singleton_all,
        singleton_lexical=singleton_lexical,
        all_position=all_position,
        lexical=lexical,
        endpoints=endpoints,
    )
    return {
        "masks": masks,
        "base_margins": base_margins,
        "donor_margins": donor_margins,
        "endpoints": endpoints,
        "singleton_all": singleton_all,
        "singleton_lexical": singleton_lexical,
        "all_position": all_position,
        "lexical": lexical,
    }


def score_label_artifact(
    labels: Mapping[str, Any],
    *,
    expected_registration_sha256: str,
    registered_protocol: Mapping[str, Any],
    held_masks: Mapping[str, torch.Tensor],
    clean_metadata_contract: Any,
) -> dict[str, Any]:
    """Validate immutable labels, then call only the registered pure scorer."""
    if not isinstance(labels, Mapping):
        raise TypeError("E29.14c labels must be a mapping")
    validated = _validate_label_integrity(
        labels,
        expected_registration_sha256=expected_registration_sha256,
        registered_protocol=registered_protocol,
        held_masks=held_masks,
        clean_metadata_contract=clean_metadata_contract,
    )
    scored = score_scale_scope_ladder(
        validated["all_position"],
        validated["lexical"],
        validated["singleton_all"],
        validated["singleton_lexical"],
        validated["endpoints"],
        validated["masks"],
        integrity_passed=True,
    )
    scientific_score_schema_version = scored.get("schema_version")
    if scientific_score_schema_version != "e29_14c_hub_off_scale_scope_ladder_score_v1":
        raise RuntimeError("E29.14c scientific score schema changed")
    return {
        **scored,
        "schema_version": "e29_14c_hub_off_scale_scope_ladder_score_v2",
        "scientific_score_schema_version": scientific_score_schema_version,
        "all_registered_integrity_gates_passed": True,
        "clean_summary": {
            "positive_base_correctness_contexts": int(
                (validated["base_margins"] > 0.0).sum()
            ),
            "positive_donor_correctness_contexts": int(
                (validated["donor_margins"] > 0.0).sum()
            ),
            "minimum_clean_endpoint": float(validated["endpoints"].min()),
            "maximum_clean_endpoint": float(validated["endpoints"].max()),
        },
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
    }


def _validate_custody_summary(
    custody: Mapping[str, Any], *, expected_registration_sha256: str
) -> None:
    require_exact_keys(custody, CUSTODY_KEYS, label="E29.14c acquisition custody")
    fixed = {
        "schema_version": "e29_14c_scale_scope_acquisition_custody_v2",
        "status": "all_registered_scale_scope_conditions_acquired_after_integrity_gates",
        "registration_sha256": expected_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "protocol_masks_schema": MASK_ARTIFACT_SCHEMA,
        "contexts": QUERY_ACCOUNTING["contexts"],
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
        "batch_invariance_passed": True,
        "all_clean_and_numerical_gates_passed": True,
        "device": "cpu",
        "dtype": "float64",
    }
    if any(custody[key] != expected for key, expected in fixed.items()):
        raise RuntimeError("E29.14c acquisition custody contract changed")
    for key in ("response_absolute_max", "runtime_seconds"):
        if _finite_number(custody[key], label=f"custody {key}") < 0.0:
            raise RuntimeError(f"E29.14c custody {key} is negative")


def _validate_attempt_record(
    attempt: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    expected_registration_sha256: str,
) -> None:
    require_exact_keys(attempt, ATTEMPT_REQUIRED_KEYS, label="E29.14c attempt")
    fixed = {
        "schema_version": "e29_14c_scale_scope_acquisition_attempt_v2",
        "status": "attempt_consumed_before_model_load",
        "registration": str(registration_path.resolve()),
        "registration_sha256": expected_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "parent_e29_14b_registration_sha256_verified_before_attempt": (
            PARENT_E29_14B_REGISTRATION_SHA256
        ),
        "parent_e29_14b_score_summary_sha256_verified_before_attempt": (
            PARENT_E29_14B_SCORE_SUMMARY_SHA256
        ),
        "superseded_e29_14c_v1_registration_sha256_verified_before_attempt": (
            registration["supersedes_e29_14c_v1_registration_sha256"]
        ),
        "superseded_e29_14c_v1_protocol_masks_sha256_verified_before_attempt": (
            registration["supersedes_e29_14c_v1_protocol_masks_sha256"]
        ),
        "no_v1_attempt_or_labels_or_custody_or_summary_before_v2_verified_before_attempt": (
            registration["supersession"][
                "no_v1_attempt_or_labels_or_custody_or_summary_before_v2"
            ]
        ),
        "source_code_sha256_verified_before_attempt": registration[
            "source_code_sha256"
        ],
        "checkpoint_sha256_verified_before_attempt": registration["checkpoint_sha256"],
        "protocol_masks_sha256_verified_before_attempt": registration[
            "protocol_masks_sha256"
        ],
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "clean_metadata_contract_sha256": registration[
            "clean_metadata_contract_sha256"
        ],
        "scale_scope_protocol_sha256": registration["scale_scope_protocol_sha256"],
        "query_accounting": QUERY_ACCOUNTING,
        "physical_query_accounting": PHYSICAL_QUERY_ACCOUNTING,
        "environment": registration["environment"],
        "execution": registration["execution"],
    }
    if any(attempt[key] != expected for key, expected in fixed.items()):
        raise RuntimeError("E29.14c acquisition attempt differs from registration")


def _response_absolute_max(labels: Mapping[str, Any]) -> float:
    tensors = [labels["singleton_all_response"], labels["singleton_lexical_response"]]
    tensors.extend(labels["all_position_responses"][split] for split in MASK_SPLITS)
    tensors.extend(labels["lexical_position_responses"][split] for split in MASK_SPLITS)
    return max(float(tensor.abs().max()) for tensor in tensors)


def score_from_custody(
    *,
    output_directory: Path,
    expected_registration_sha256: str,
    expected_acquisition_custody_sha256: str,
) -> dict[str, Any]:
    """Verify the complete chain, score once, and write one immutable summary."""
    custody_path = output_directory / "scale_scope_acquisition_custody.json"
    labels_path = output_directory / "scale_scope_labels.pt"
    attempt_path = output_directory / "scale_scope_acquisition_attempt.json"
    summary_path = output_directory / "scale_scope_summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to overwrite E29.14c scored summary")
    verify_file_hash(custody_path, expected_acquisition_custody_sha256)
    custody = _load_json(custody_path)
    _validate_custody_summary(
        custody, expected_registration_sha256=expected_registration_sha256
    )
    if Path(custody["scale_scope_labels"]).resolve() != labels_path.resolve():
        raise RuntimeError("E29.14c custody points to another label artifact")
    verify_file_hash(labels_path, str(custody["scale_scope_labels_sha256"]))

    registration_path = Path(custody["registration"])
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
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14c scoring environment differs from registration")
    if custody["batch_size"] != registration["execution"]["batch_size"]:
        raise RuntimeError("E29.14c acquisition batch size differs from registration")
    if Path(custody["acquisition_attempt"]).resolve() != attempt_path.resolve():
        raise RuntimeError("E29.14c custody points to another acquisition attempt")
    verify_file_hash(attempt_path, str(custody["acquisition_attempt_sha256"]))
    _validate_attempt_record(
        _load_json(attempt_path),
        registration=registration,
        registration_path=registration_path,
        expected_registration_sha256=expected_registration_sha256,
    )
    protocol_masks_path = Path(registration["protocol_masks"])
    if Path(custody["protocol_masks"]).resolve() != protocol_masks_path.resolve():
        raise RuntimeError("E29.14c custody points to another protocol-mask artifact")
    masks = load_verified_protocol_masks(
        protocol_masks_path,
        expected_sha256=str(registration["protocol_masks_sha256"]),
    )
    if (
        custody["source_code_sha256_verified_before_acquisition"]
        != registration["source_code_sha256"]
        or custody["checkpoint_sha256_verified_before_acquisition"]
        != registration["checkpoint_sha256"]
        or custody["protocol_masks_sha256_verified_before_acquisition"]
        != registration["protocol_masks_sha256"]
        or custody["prompt_panel_sha256"] != registration["prompt_panel_sha256"]
        or custody["clean_metadata_contract_sha256"]
        != registration["clean_metadata_contract_sha256"]
        or custody["scale_scope_protocol_sha256"]
        != registration["scale_scope_protocol_sha256"]
    ):
        raise RuntimeError("E29.14c custody differs from the sealed registration")

    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(labels, Mapping):
        raise TypeError("E29.14c label artifact must contain a mapping")
    summary = score_label_artifact(
        labels,
        expected_registration_sha256=expected_registration_sha256,
        registered_protocol=registration["scale_scope_protocol"],
        held_masks=masks,
        clean_metadata_contract=registration["clean_metadata_contract"],
    )
    if float(custody["response_absolute_max"]) != _response_absolute_max(labels):
        raise RuntimeError("E29.14c custody response maximum differs from labels")
    summary["custody"] = {
        "registration": str(registration_path.resolve()),
        "registration_sha256": expected_registration_sha256,
        "acquisition_attempt": str(attempt_path.resolve()),
        "acquisition_attempt_sha256": str(custody["acquisition_attempt_sha256"]),
        "acquisition_custody": str(custody_path.resolve()),
        "acquisition_custody_sha256": expected_acquisition_custody_sha256,
        "scale_scope_labels": str(labels_path.resolve()),
        "scale_scope_labels_sha256": str(custody["scale_scope_labels_sha256"]),
        "protocol_masks": str(protocol_masks_path.resolve()),
        "protocol_masks_sha256": str(registration["protocol_masks_sha256"]),
        "clean_metadata_contract_sha256": str(
            registration["clean_metadata_contract_sha256"]
        ),
    }
    atomic_json_save(summary, summary_path)
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--expected-acquisition-custody-sha256", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            score_from_custody(
                output_directory=arguments.output_directory,
                expected_registration_sha256=arguments.expected_registration_sha256,
                expected_acquisition_custody_sha256=(
                    arguments.expected_acquisition_custody_sha256
                ),
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["score_from_custody", "score_label_artifact"]
