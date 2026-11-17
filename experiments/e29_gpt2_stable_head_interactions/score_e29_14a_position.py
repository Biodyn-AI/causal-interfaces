"""Verify custody and score the sealed E29.14a position panel exactly once."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14a_position import (
    ATTEMPT_KEYS,
    CUSTODY_KEYS,
    LABEL_KEYS,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_custody import (
    PHYSICAL_QUERY_ACCOUNTING,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    load_verified_registration,
    require_exact_keys,
    runtime_environment,
    sha256,
    validate_registration_payload,
    verify_file_hash,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    CONTEXT_COUNT,
    FORWARD_CONDITIONS,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
    score_position_semantics,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14a_position_lexical_interchange_v1"


def _validate_scoring_environment(registration: Mapping[str, Any]) -> None:
    """Reject scoring-library drift from the response-blind sealed runtime."""
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14a scoring environment differs from registration")


def _validate_custody_summary(
    custody: Mapping[str, Any], *, expected_registration_sha256: str
) -> None:
    """Validate acquisition status, accounting, and scalar integrity fields."""
    require_exact_keys(custody, CUSTODY_KEYS, label="E29.14a acquisition custody")
    fixed = {
        "schema_version": "e29_14a_position_acquisition_custody_v1",
        "status": "all_registered_conditions_acquired_after_integrity_gates",
        "registration_sha256": expected_registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
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
        "batch_invariance_passed": True,
        "all_clean_and_numerical_gates_passed": True,
        "device": "cpu",
        "dtype": "float64",
    }
    if any(custody[key] != expected for key, expected in fixed.items()):
        raise RuntimeError("E29.14a acquisition custody contract changed")
    for key in ("response_absolute_max", "runtime_seconds"):
        value = custody[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"E29.14a custody {key} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise RuntimeError(f"E29.14a custody {key} is invalid")


def _validate_attempt_record(
    attempt: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    expected_registration_sha256: str,
) -> None:
    """Require the response-preceding attempt marker to match the seal."""
    require_exact_keys(attempt, ATTEMPT_KEYS, label="E29.14a acquisition attempt")
    fixed = {
        "schema_version": "e29_14a_position_acquisition_attempt_v1",
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
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "position_protocol_sha256": registration["position_protocol_sha256"],
        "query_accounting": QUERY_ACCOUNTING,
        "physical_query_accounting": PHYSICAL_QUERY_ACCOUNTING,
        "environment": registration["environment"],
        "execution": registration["execution"],
    }
    if any(attempt[key] != expected for key, expected in fixed.items()):
        raise RuntimeError("E29.14a acquisition attempt differs from registration")


def _finite_vector(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    vector = value.detach().cpu().double()
    if tuple(vector.shape) != (CONTEXT_COUNT,) or not bool(vector.isfinite().all()):
        raise RuntimeError(f"{label} must be one finite 12-context vector")
    return vector


def _validate_label_integrity(
    labels: Mapping[str, Any], *, expected_registration_sha256: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_exact_keys(labels, LABEL_KEYS, label="E29.14a position labels")
    if labels["schema_version"] != "e29_14a_position_labels_v1":
        raise RuntimeError("E29.14a label schema changed")
    if (
        labels["registration_sha256"] != expected_registration_sha256
        or labels["expected_registration_sha256_supplied"]
        != expected_registration_sha256
    ):
        raise RuntimeError("E29.14a labels differ from the expected registration")
    expected_manifest = {
        "forward": [dict(row) for row in FORWARD_CONDITIONS],
        "reverse": [dict(row) for row in REVERSE_CONDITIONS],
    }
    if labels["condition_manifest"] != expected_manifest:
        raise RuntimeError("E29.14a label condition manifest changed")
    if labels["query_accounting"] != QUERY_ACCOUNTING:
        raise RuntimeError("E29.14a logical query accounting changed")
    if labels["physical_query_accounting"] != PHYSICAL_QUERY_ACCOUNTING:
        raise RuntimeError("E29.14a physical query accounting changed")
    if labels["all_integrity_gates_passed"] is not True:
        raise RuntimeError("E29.14a label integrity did not pass")
    smoke = labels["batch_invariance_smoke"]
    if not isinstance(smoke, Mapping) or smoke.get("passed") is not True:
        raise RuntimeError("E29.14a batch-invariance integrity did not pass")
    numerical = labels["numerical_summary"]
    expected_numerical = {
        "forward_responses",
        "reverse_responses",
        "combined_responses",
        "clean_endpoints",
    }
    if not isinstance(numerical, Mapping) or set(numerical) != expected_numerical:
        raise RuntimeError("E29.14a numerical summary changed")
    if any(
        not isinstance(record, Mapping) or record.get("passed") is not True
        for record in numerical.values()
    ):
        raise RuntimeError("E29.14a numerical integrity did not pass")
    metadata = labels["clean_metadata"]
    if not isinstance(metadata, list) or len(metadata) != CONTEXT_COUNT:
        raise RuntimeError("E29.14a clean metadata count changed")

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
        raise RuntimeError("E29.14a clean correctness integrity changed")
    if not torch.equal(endpoints, base_margins + donor_margins):
        raise RuntimeError("E29.14a clean endpoint arithmetic changed")
    return base_margins, donor_margins, endpoints


def score_label_artifact(
    labels: Mapping[str, Any], *, expected_registration_sha256: str
) -> dict[str, Any]:
    """Validate an acquired label artifact and apply only registered scoring."""
    if not isinstance(labels, Mapping):
        raise TypeError("E29.14a labels must be a mapping")
    base_margins, donor_margins, endpoints = _validate_label_integrity(
        labels, expected_registration_sha256=expected_registration_sha256
    )
    scored = score_position_semantics(
        labels["forward_responses"],
        labels["reverse_responses"],
        endpoints,
        integrity_passed=True,
    )
    return {
        **scored,
        "all_registered_integrity_gates_passed": True,
        "clean_summary": {
            "positive_base_correctness_contexts": int((base_margins > 0.0).sum()),
            "positive_donor_correctness_contexts": int((donor_margins > 0.0).sum()),
            "minimum_clean_endpoint": float(endpoints.min()),
            "maximum_clean_endpoint": float(endpoints.max()),
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
    """Verify the complete registration/label chain and write one summary."""
    custody_path = output_directory / "position_acquisition_custody.json"
    labels_path = output_directory / "position_labels.pt"
    summary_path = output_directory / "position_semantics_summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to overwrite E29.14a scored summary")
    verify_file_hash(custody_path, expected_acquisition_custody_sha256)
    custody = _load_json(custody_path)
    _validate_custody_summary(
        custody, expected_registration_sha256=expected_registration_sha256
    )
    if Path(custody["position_labels"]).resolve() != labels_path.resolve():
        raise RuntimeError("E29.14a custody points to another label artifact")
    verify_file_hash(labels_path, str(custody["position_labels_sha256"]))

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
        raise RuntimeError("E29.14a acquisition batch size differs from registration")
    attempt_path = output_directory / "position_acquisition_attempt.json"
    if Path(custody["acquisition_attempt"]).resolve() != attempt_path.resolve():
        raise RuntimeError("E29.14a custody points to another acquisition attempt")
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
        or custody["prompt_panel_sha256"] != registration["prompt_panel_sha256"]
        or custody["position_protocol_sha256"]
        != registration["position_protocol_sha256"]
    ):
        raise RuntimeError("E29.14a custody differs from the sealed registration")

    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(labels, Mapping):
        raise TypeError("E29.14a label artifact must contain a mapping")
    summary = score_label_artifact(
        labels, expected_registration_sha256=expected_registration_sha256
    )
    summary["custody"] = {
        "registration": str(registration_path.resolve()),
        "registration_sha256": expected_registration_sha256,
        "acquisition_custody": str(custody_path.resolve()),
        "acquisition_custody_sha256": expected_acquisition_custody_sha256,
        "position_labels": str(labels_path.resolve()),
        "position_labels_sha256": str(custody["position_labels_sha256"]),
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
