"""Verify custody and score the sealed E29.14b bridge exactly once."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14b_component_position import (
    CUSTODY_KEYS,
    LABEL_KEYS,
    build_condition_manifest,
)
from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    ATTEMPT_REQUIRED_KEYS,
    BATCH_INVARIANCE,
    PHYSICAL_QUERY_ACCOUNTING,
    PRIOR_PROTOCOL_MASKS_SHA256,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    load_verified_registration,
    require_exact_keys,
    runtime_environment,
    sha256,
    validate_registration_payload,
    verify_file_hash,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    CONTEXT_COUNT,
    MASK_SHAPES,
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    score_component_position_bridge,
    validate_bridge_masks,
    validate_component_position_protocol,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14b_component_position import (
    DEFAULT_OUTPUT,
)

SOURCE_DIRECTORY = Path(__file__).resolve().parent


def _validate_scoring_environment(registration: Mapping[str, Any]) -> None:
    """Reject scoring-library drift from the response-blind sealed runtime."""
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14b scoring environment differs from registration")


def _validate_custody_summary(
    custody: Mapping[str, Any], *, expected_registration_sha256: str
) -> None:
    """Validate every acquisition status, count, and scalar integrity field."""
    require_exact_keys(custody, CUSTODY_KEYS, label="E29.14b acquisition custody")
    fixed = {
        "schema_version": "e29_14b_component_position_acquisition_custody_v1",
        "status": "all_registered_conditions_acquired_after_integrity_gates",
        "registration_sha256": expected_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "contexts": QUERY_ACCOUNTING["contexts"],
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
        raise RuntimeError("E29.14b acquisition custody contract changed")
    for key in ("response_absolute_max", "runtime_seconds"):
        value = custody[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"E29.14b custody {key} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise RuntimeError(f"E29.14b custody {key} is invalid")


def _validate_attempt_record(
    attempt: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    expected_registration_sha256: str,
) -> None:
    """Bind the response-preceding attempt marker to every sealed dependency."""
    require_exact_keys(attempt, ATTEMPT_REQUIRED_KEYS, label="E29.14b attempt")
    fixed = {
        "schema_version": "e29_14b_component_position_acquisition_attempt_v1",
        "status": "attempt_consumed_before_model_load",
        "registration": str(registration_path.resolve()),
        "registration_sha256": expected_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "source_code_sha256_verified_before_attempt": registration[
            "source_code_sha256"
        ],
        "checkpoint_sha256_verified_before_attempt": registration[
            "checkpoint_sha256"
        ],
        "prior_protocol_masks_sha256_verified_before_attempt": registration[
            "prior_protocol_masks_sha256"
        ],
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "component_position_protocol_sha256": registration[
            "component_position_protocol_sha256"
        ],
        "query_accounting": QUERY_ACCOUNTING,
        "physical_query_accounting": PHYSICAL_QUERY_ACCOUNTING,
        "environment": registration["environment"],
        "execution": registration["execution"],
    }
    if any(attempt[key] != expected for key, expected in fixed.items()):
        raise RuntimeError("E29.14b acquisition attempt differs from registration")


def _finite_vector(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise RuntimeError(f"{label} device or dtype changed")
    vector = value.detach().clone()
    if tuple(vector.shape) != (CONTEXT_COUNT,) or not bool(vector.isfinite().all()):
        raise RuntimeError(f"{label} must be one finite 12-context vector")
    return vector


def _finite_response_tables(
    values: Any, *, label: str
) -> dict[str, torch.Tensor]:
    if not isinstance(values, Mapping) or tuple(values) != MASK_SPLITS:
        raise RuntimeError(f"{label} split keys or order changed")
    tables = {}
    for split in MASK_SPLITS:
        value = values[split]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}/{split} must be a tensor")
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise RuntimeError(f"{label}/{split} device or dtype changed")
        table = value.detach().clone()
        shape = (MASK_SHAPES[split][0], CONTEXT_COUNT)
        if tuple(table.shape) != shape or not bool(table.isfinite().all()):
            raise RuntimeError(f"{label}/{split} must be finite with shape {shape}")
        tables[split] = table
    return tables


def _registered_masks(protocol: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    rows_by_split = protocol["mask_rows"]
    if not isinstance(rows_by_split, Mapping) or tuple(rows_by_split) != MASK_SPLITS:
        raise RuntimeError("E29.14b registered mask-row order changed")
    masks = {}
    for split in MASK_SPLITS:
        rows = rows_by_split[split]
        if not isinstance(rows, list) or len(rows) != MASK_SHAPES[split][0]:
            raise RuntimeError("E29.14b registered mask-row count changed")
        panel = torch.zeros(MASK_SHAPES[split], dtype=torch.float64)
        for row_index, selected in enumerate(rows):
            if not isinstance(selected, list):
                raise TypeError("E29.14b registered mask row must be a list")
            indices = [int(index) for index in selected]
            if len(indices) != len(set(indices)):
                raise RuntimeError("E29.14b registered mask row contains duplicates")
            panel[row_index, indices] = 1.0
        masks[split] = panel
    return validate_bridge_masks(masks)


def _validate_smoke(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("E29.14b batch-invariance record must be a mapping")
    required = {
        "records",
        "mask_rows",
        "unique_mask_scope_cells",
        "physical_mask_scope_cells",
        "scientific_labels_produced",
        "passed",
    }
    require_exact_keys(value, required, label="E29.14b batch smoke")
    if (
        value["mask_rows"] != 97
        or value["unique_mask_scope_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_unique_condition_rows"]
        or value["physical_mask_scope_cells"]
        != PHYSICAL_QUERY_ACCOUNTING["batch_smoke_physical_condition_cells"]
        or value["scientific_labels_produced"] is not False
        or value["passed"] is not True
    ):
        raise RuntimeError("E29.14b batch-invariance integrity changed")
    records = value["records"]
    if not isinstance(records, Mapping) or tuple(records) != POSITION_SCOPE_ORDER:
        raise RuntimeError("E29.14b batch-smoke scope order changed")
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
            raise TypeError("E29.14b batch-invariance record must be a mapping")
        require_exact_keys(record, record_keys, label="E29.14b smoke record")
        if (
            record["mask_rows"] != 97
            or record["reference_batch_size"]
            != BATCH_INVARIANCE["reference_batch_size"]
            or record["candidate_batch_size"]
            != BATCH_INVARIANCE["candidate_batch_size"]
            or record["atol"] != BATCH_INVARIANCE["absolute_tolerance"]
            or record["rtol"] != BATCH_INVARIANCE["relative_tolerance"]
            or record["passed"] is not True
        ):
            raise RuntimeError("E29.14b batch-invariance scope failed")
        for key in ("maximum_absolute_difference", "maximum_relative_difference"):
            difference = record[key]
            if (
                isinstance(difference, bool)
                or not isinstance(difference, (int, float))
                or not math.isfinite(float(difference))
                or float(difference) < 0.0
            ):
                raise RuntimeError("E29.14b batch-invariance difference is invalid")


def _validate_label_integrity(
    labels: Mapping[str, Any],
    *,
    expected_registration_sha256: str,
    registered_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_keys(labels, LABEL_KEYS, label="E29.14b labels")
    if labels["schema_version"] != "e29_14b_component_position_labels_v1":
        raise RuntimeError("E29.14b label schema changed")
    if (
        labels["registration_sha256"] != expected_registration_sha256
        or labels["expected_registration_sha256_supplied"]
        != expected_registration_sha256
    ):
        raise RuntimeError("E29.14b labels differ from the expected registration")
    protocol = validate_component_position_protocol(registered_protocol)
    registered_masks = _registered_masks(protocol)
    label_masks = validate_bridge_masks(labels["held_masks"])
    if any(
        not torch.equal(label_masks[split], registered_masks[split])
        for split in MASK_SPLITS
    ):
        raise RuntimeError("E29.14b label masks differ from registration")
    if labels["condition_manifest"] != build_condition_manifest(registered_masks):
        raise RuntimeError("E29.14b label condition manifest changed")
    if labels["query_accounting"] != QUERY_ACCOUNTING:
        raise RuntimeError("E29.14b logical query accounting changed")
    if labels["physical_query_accounting"] != PHYSICAL_QUERY_ACCOUNTING:
        raise RuntimeError("E29.14b physical query accounting changed")
    if labels["all_integrity_gates_passed"] is not True:
        raise RuntimeError("E29.14b label integrity did not pass")
    _validate_smoke(labels["batch_invariance_smoke"])
    numerical = labels["numerical_summary"]
    expected_numerical = {
        "singleton_all_response",
        "singleton_lexical_response",
        "responses_in_registered_order",
        "combined_responses",
        "clean_endpoints",
    }
    if not isinstance(numerical, Mapping) or set(numerical) != expected_numerical:
        raise RuntimeError("E29.14b numerical summary changed")
    if any(
        not isinstance(record, Mapping) or record.get("passed") is not True
        for record in numerical.values()
    ):
        raise RuntimeError("E29.14b numerical integrity did not pass")
    metadata = labels["clean_metadata"]
    if (
        not isinstance(metadata, list)
        or len(metadata) != CONTEXT_COUNT
        or any(not isinstance(record, Mapping) for record in metadata)
    ):
        raise RuntimeError("E29.14b clean metadata count changed")

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
        raise RuntimeError("E29.14b clean correctness integrity changed")
    if not torch.equal(endpoints, base_margins + donor_margins):
        raise RuntimeError("E29.14b clean endpoint arithmetic changed")
    singleton_all = _finite_vector(
        labels["singleton_all_response"], label="singleton all response"
    )
    singleton_lexical = _finite_vector(
        labels["singleton_lexical_response"], label="singleton lexical response"
    )
    all_position = _finite_response_tables(
        labels["all_position_responses"], label="all-position responses"
    )
    lexical = _finite_response_tables(
        labels["lexical_position_responses"], label="lexical-position responses"
    )
    ordered = labels["responses_in_registered_order"]
    if not isinstance(ordered, torch.Tensor):
        raise TypeError("registered-order responses must be a tensor")
    if ordered.device.type != "cpu" or ordered.dtype != torch.float64:
        raise RuntimeError("registered-order response device or dtype changed")
    ordered = ordered.detach().clone()
    expected_shape = (
        QUERY_ACCOUNTING["patched_condition_rows_per_context"],
        CONTEXT_COUNT,
    )
    if tuple(ordered.shape) != expected_shape or not bool(ordered.isfinite().all()):
        raise RuntimeError("registered-order response shape or finiteness changed")
    expected_rows = [singleton_all, singleton_lexical]
    for split in MASK_SPLITS:
        for row_index in range(MASK_SHAPES[split][0]):
            expected_rows.extend(
                (all_position[split][row_index], lexical[split][row_index])
            )
    if not torch.equal(ordered, torch.stack(expected_rows)):
        raise RuntimeError("E29.14b registered response order changed")
    if float(ordered.abs().max()) > float(NUMERICAL_GATES["response_absolute_max"]):
        raise RuntimeError("E29.14b response magnitude gate changed")
    return {
        "masks": label_masks,
        "base_margins": base_margins,
        "donor_margins": donor_margins,
        "endpoints": endpoints,
        "singleton_all": singleton_all,
        "singleton_lexical": singleton_lexical,
        "all_position": all_position,
        "lexical": lexical,
        "ordered": ordered,
    }


def score_label_artifact(
    labels: Mapping[str, Any],
    *,
    expected_registration_sha256: str,
    registered_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable label artifact and call only registered scoring."""
    if not isinstance(labels, Mapping):
        raise TypeError("E29.14b labels must be a mapping")
    validated = _validate_label_integrity(
        labels,
        expected_registration_sha256=expected_registration_sha256,
        registered_protocol=registered_protocol,
    )
    scored = score_component_position_bridge(
        validated["all_position"],
        validated["lexical"],
        validated["singleton_all"],
        validated["singleton_lexical"],
        validated["endpoints"],
        validated["masks"],
        integrity_passed=True,
    )
    return {
        **scored,
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def score_from_custody(
    *,
    output_directory: Path,
    expected_registration_sha256: str,
    expected_acquisition_custody_sha256: str,
) -> dict[str, Any]:
    """Verify the complete chain, score once, and write one immutable summary."""
    custody_path = output_directory / "component_position_acquisition_custody.json"
    labels_path = output_directory / "component_position_labels.pt"
    attempt_path = output_directory / "component_position_acquisition_attempt.json"
    summary_path = output_directory / "component_position_bridge_summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to overwrite E29.14b scored summary")
    verify_file_hash(custody_path, expected_acquisition_custody_sha256)
    custody = _load_json(custody_path)
    _validate_custody_summary(
        custody, expected_registration_sha256=expected_registration_sha256
    )
    if Path(custody["component_position_labels"]).resolve() != labels_path.resolve():
        raise RuntimeError("E29.14b custody points to another label artifact")
    verify_file_hash(labels_path, str(custody["component_position_labels_sha256"]))

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
    _validate_scoring_environment(registration)
    if custody["batch_size"] != registration["execution"]["batch_size"]:
        raise RuntimeError("E29.14b acquisition batch size differs from registration")
    if Path(custody["acquisition_attempt"]).resolve() != attempt_path.resolve():
        raise RuntimeError("E29.14b custody points to another acquisition attempt")
    verify_file_hash(attempt_path, str(custody["acquisition_attempt_sha256"]))
    _validate_attempt_record(
        _load_json(attempt_path),
        registration=registration,
        registration_path=registration_path,
        expected_registration_sha256=expected_registration_sha256,
    )
    if (
        custody["source_code_sha256_verified_before_acquisition"]
        != registration["source_code_sha256"]
        or custody["checkpoint_sha256_verified_before_acquisition"]
        != registration["checkpoint_sha256"]
        or custody["prior_protocol_masks_sha256_verified_before_acquisition"]
        != PRIOR_PROTOCOL_MASKS_SHA256
        or custody["prior_protocol_masks_sha256_verified_before_acquisition"]
        != registration["prior_protocol_masks_sha256"]
        or custody["prompt_panel_sha256"] != registration["prompt_panel_sha256"]
        or custody["component_position_protocol_sha256"]
        != registration["component_position_protocol_sha256"]
    ):
        raise RuntimeError("E29.14b custody differs from the sealed registration")

    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(labels, Mapping):
        raise TypeError("E29.14b label artifact must contain a mapping")
    summary = score_label_artifact(
        labels,
        expected_registration_sha256=expected_registration_sha256,
        registered_protocol=registration["component_position_protocol"],
    )
    response_maximum = float(labels["responses_in_registered_order"].abs().max())
    if float(custody["response_absolute_max"]) != response_maximum:
        raise RuntimeError("E29.14b custody response maximum differs from labels")
    summary["custody"] = {
        "registration": str(registration_path.resolve()),
        "registration_sha256": expected_registration_sha256,
        "acquisition_attempt": str(attempt_path.resolve()),
        "acquisition_attempt_sha256": str(custody["acquisition_attempt_sha256"]),
        "acquisition_custody": str(custody_path.resolve()),
        "acquisition_custody_sha256": expected_acquisition_custody_sha256,
        "component_position_labels": str(labels_path.resolve()),
        "component_position_labels_sha256": str(
            custody["component_position_labels_sha256"]
        ),
        "prior_protocol_masks_sha256": PRIOR_PROTOCOL_MASKS_SHA256,
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
