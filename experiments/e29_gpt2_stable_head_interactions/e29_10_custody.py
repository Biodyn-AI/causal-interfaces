"""Write-once artifacts and strict hash closure for E29.10."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    ATTENTION_IMPLEMENTATION,
    BATCH_SIZE,
    CHECKPOINT_FILES,
    DEVICE,
    E29_9_METHODS,
    FORENSIC_REFERENCE_ACCESS_POLICY,
    FORENSIC_REFERENCE_MANIFEST_SHA256,
    MODEL_DTYPE,
    NUMERICAL_GATES,
    QUERY_BUDGET,
    SOURCE_FILES,
)

REGISTRATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "forensic_scope",
        "parent_e29_9_registration",
        "parent_e29_9_registration_sha256",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "component_map",
        "patch_scope",
        "component_semantics",
        "protocol_masks",
        "protocol_masks_sha256",
        "explicit_masks",
        "mask_shapes",
        "prompt_checks",
        "prompt_panel_sha256",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
        "expected_primary_feature_preflight",
        "method_manifest",
        "primary_method",
        "primary_budgets",
        "ridge_values",
        "scratch_mlp_recipe",
        "primary_size16_gates",
        "stress_size24_gates",
        "bootstrap",
        "query_budget",
        "numerical_gates",
        "batch_invariance_smoke",
        "forensic_reference",
        "source_code_sha256",
        "outcome_access_policy",
        "protocol_note",
        "protocol_note_sha256",
        "protocol_deviations",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"hash mismatch for {path}")
    return observed


def require_exact_keys(
    record: Mapping[str, Any], expected: Collection[str], *, label: str
) -> None:
    observed = set(record)
    required = set(expected)
    if observed != required:
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(required - observed)}, "
            f"extra={sorted(observed - required)}"
        )


def atomic_torch_save(record: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix="tmp-", suffix=".pt"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(record, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json_save(record: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix="tmp-", suffix=".json"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    verify_file_hash(path, expected_sha256)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _runtime_environment() -> dict[str, str]:
    import safetensors
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }


def load_verified_registration(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Verify the E29.10 seal without opening any old float32 outcome file."""
    registration = load_verified_json(path, expected_sha256)
    require_exact_keys(registration, REGISTRATION_KEYS, label="E29.10 registration")
    if registration["schema_version"] != "e29_10_float64_forensic_replication_v1":
        raise RuntimeError("unexpected E29.10 registration schema")
    if registration["status"] != "sealed_before_any_e29_10_scientific_outcome":
        raise RuntimeError("E29.10 is not in its sealed state")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("registration contains a protocol deviation")
    if registration["environment"] != _runtime_environment():
        raise RuntimeError("runtime environment differs from E29.10 seal")
    expected_execution = {
        "device": DEVICE,
        "model_dtype": MODEL_DTYPE,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "batch_size": BATCH_SIZE,
        "torch_num_threads_during_fitting": 1,
    }
    if registration["execution"] != expected_execution:
        raise RuntimeError("E29.10 execution recipe changed")
    if registration["numerical_gates"] != NUMERICAL_GATES:
        raise RuntimeError("E29.10 numerical gates changed")
    if registration["query_budget"] != QUERY_BUDGET:
        raise RuntimeError("E29.10 query budget differs from E29.9")
    if registration["bootstrap"] != {"seed": 29_905, "replicates": 2_000}:
        raise RuntimeError("E29.10 bootstrap differs from E29.9")
    if tuple(registration["method_manifest"]) != E29_9_METHODS:
        raise RuntimeError("E29.10 method manifest differs from E29.9")
    forensic_contract = registration["forensic_reference"]
    require_exact_keys(
        forensic_contract,
        {
            "artifact_keys",
            "expected_sha256_keys",
            "reference_manifest_sha256",
            "access_policy",
        },
        label="deferred forensic-reference contract",
    )
    expected_forensic_contract = {
        "artifact_keys": [
            "registration",
            "calibration_validation_labels",
            "final_stress_labels",
            "predictions",
            "summary",
        ],
        "expected_sha256_keys": [
            "registration_sha256",
            "calibration_validation_labels_sha256",
            "final_stress_labels_sha256",
            "predictions_sha256",
            "summary_sha256",
        ],
        "reference_manifest_sha256": FORENSIC_REFERENCE_MANIFEST_SHA256,
        "access_policy": FORENSIC_REFERENCE_ACCESS_POLICY,
    }
    if forensic_contract != expected_forensic_contract:
        raise RuntimeError("deferred E29.9 forensic reference contract changed")
    preflight = registration["expected_primary_feature_preflight"]
    if (
        preflight.get("uses_response_labels") is not False
        or preflight.get("passed") is not True
        or int(preflight.get("active_feature_count", -1)) != 7
        or int(preflight.get("centered_standardized_rank", -1)) != 7
        or float(preflight.get("condition_number", float("inf"))) > 1_000.0
    ):
        raise RuntimeError("registered source-feature preflight is invalid")
    if registration["outcome_access_policy"] != {
        "pre_score_old_float32_labels_forbidden": True,
        "old_float32_labels_first_opened_by": "score_e29_10_float64.py",
        "scoring_requires_frozen_float64_predictions_and_all_float64_labels": True,
    }:
        raise RuntimeError("old-outcome access policy changed")

    # The exact hashed parent is response-blind with respect to E29.10 and is
    # the authoritative semantic contract for every reused scientific field.
    from experiments.e29_gpt2_stable_head_interactions.e29_9_custody import (
        load_verified_registration as load_parent_registration,
    )

    parent = load_parent_registration(
        Path(registration["parent_e29_9_registration"]),
        expected_sha256=registration["parent_e29_9_registration_sha256"],
    )
    reused_fields = (
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "component_map",
        "patch_scope",
        "component_semantics",
        "protocol_masks",
        "protocol_masks_sha256",
        "explicit_masks",
        "mask_shapes",
        "prompt_checks",
        "prompt_panel_sha256",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
        "method_manifest",
        "primary_method",
        "primary_budgets",
        "ridge_values",
        "scratch_mlp_recipe",
        "primary_size16_gates",
        "stress_size24_gates",
        "bootstrap",
        "query_budget",
    )
    for field in reused_fields:
        if registration[field] != parent[field]:
            raise RuntimeError(f"E29.10 changed reused E29.9 field {field}")
    if (
        registration["expected_primary_feature_preflight"]
        != parent["observed_primary_feature_preflight"]
    ):
        raise RuntimeError("E29.10 changed parent primary-feature preflight")
    smoke = registration["batch_invariance_smoke"]
    require_exact_keys(
        smoke,
        {
            "context_index",
            "mask_rows",
            "mask_sha256",
            "reference_batch_size",
            "candidate_batch_size",
            "maximum_absolute_difference",
            "maximum_relative_difference",
            "atol",
            "rtol",
            "passed",
            "uses_scientific_masks",
            "used_for_model_selection",
            "outside_scientific_query_accounting",
        },
        label="batch invariance smoke",
    )
    if smoke["passed"] is not True or smoke["uses_scientific_masks"] is not False:
        raise RuntimeError("float64 batch-invariance smoke did not pass cleanly")
    if smoke["used_for_model_selection"] is not False:
        raise RuntimeError("numerical smoke cannot select a scientific model")
    if smoke["outside_scientific_query_accounting"] is not True:
        raise RuntimeError("numerical smoke accounting declaration changed")

    # These are response-blind dependencies and are safe to verify at every stage.
    for key in (
        "parent_e29_9_registration",
        "protocol_masks",
        "e29_8_feature_artifact",
        "protocol_note",
    ):
        verify_file_hash(Path(registration[key]), registration[f"{key}_sha256"])
    require_exact_keys(
        registration["checkpoint_sha256"],
        CHECKPOINT_FILES,
        label="checkpoint hash manifest",
    )
    for filename, expected in registration["checkpoint_sha256"].items():
        verify_file_hash(Path(registration["model_snapshot"]) / filename, expected)
    source_directory = (
        path.parent.parent / "experiments/e29_gpt2_stable_head_interactions"
    )
    require_exact_keys(
        registration["source_code_sha256"],
        SOURCE_FILES,
        label="source-code hash manifest",
    )
    for filename, expected in registration["source_code_sha256"].items():
        verify_file_hash(source_directory / filename, expected)
    return registration


__all__ = [
    "REGISTRATION_KEYS",
    "atomic_json_save",
    "atomic_torch_save",
    "load_verified_json",
    "load_verified_registration",
    "require_exact_keys",
    "sha256",
    "verify_file_hash",
]
