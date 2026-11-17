"""Write-once artifacts and strict hash custody for prospective E29.11."""

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
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_multiscale_protocol import (
    C32_PROMOTION_GATES,
    FIT_ACQUISITION_SPLITS,
    MASK_SHAPES,
    METHOD_MANIFEST,
    MLP7_OVER_TT2_SUPERIORITY_GATES,
    MULTISCALE_FIT_PROTOCOL,
    PREDICTION_SPLITS,
    PRIMARY_METHOD,
    PRIMARY_RIDGE,
    QUERY_BUDGET,
    SECONDARY_METHOD,
    SECONDARY_RIDGE,
    SIZE24_CONFIRMATION_GATES,
    TT2_SECONDARY_GATES,
)

SOURCE_FILES = (
    "e29_11_multiscale_protocol.py",
    "e29_11_fitting.py",
    "e29_11_custody.py",
    "register_e29_11_scale_ladder.py",
    "acquire_e29_11_fit.py",
    "fit_e29_11_multiscale.py",
    "acquire_e29_11_held.py",
    "score_e29_11_multiscale.py",
    "acquire_e29_11_conditional.py",
    "e29_10_custody.py",
    "e29_10_float64_runner.py",
    "e29_10_float64_protocol.py",
    "e29_9_custody.py",
    "e29_9_transfer_fitting.py",
    "e29_9_seven_gate_protocol.py",
    "e29_8_prompt_panel.py",
    "e29_8_scaleup_protocol.py",
    "source_model_fitting.py",
    "scaled_tensor_response.py",
    "tensor_train_response.py",
    "generalized_sublayer_patch.py",
)

REGISTRATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "prospective_scope",
        "parent_e29_10_registration",
        "parent_e29_10_registration_sha256",
        "prior_prompt_registrations",
        "prior_mask_artifacts",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "component_map",
        "patch_scope",
        "component_semantics",
        "protocol_note",
        "protocol_note_sha256",
        "protocol_masks",
        "protocol_masks_sha256",
        "mask_split_order",
        "explicit_masks",
        "mask_shapes",
        "prompt_checks",
        "prompt_panel_sha256",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
        "source_feature_preflight",
        "method_manifest",
        "primary_method",
        "secondary_method",
        "primary_ridge",
        "secondary_ridge",
        "multiscale_fit_protocol",
        "size24_confirmation_gates",
        "tt2_secondary_gates",
        "mlp7_over_tt2_superiority_gates",
        "conditional_size32_gates",
        "bootstrap",
        "query_budget",
        "numerical_gates",
        "batch_invariance_smoke",
        "outcome_custody",
        "source_code_sha256",
        "protocol_deviations",
    }
)


def sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected: str) -> str:
    """Fail closed when a required artifact is absent or has changed."""
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
    """Atomically create a Torch artifact, refusing every overwrite."""
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
    """Atomically create strict JSON, refusing NaNs and every overwrite."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix="tmp-", suffix=".json"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    verify_file_hash(path, expected_sha256)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def runtime_environment() -> dict[str, str]:
    import safetensors
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }


def _jsonable(value: Any) -> Any:
    """Mirror the tuple-to-list conversion performed by strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _expected_outcome_custody() -> dict[str, Any]:
    return {
        "registration_sha_required_for_fit_acquisition": True,
        "fit_acquisition_queries_only_registered_fit_and_matched_count_control": True,
        "fit_custody_sha_required_for_fitting": True,
        "every_acquisition_requires_base_and_donor_clean_correctness": True,
        "all_size24_and_conditional_size32_predictions_hashed_before_size24_labels": True,
        "preparation_sha_required_for_size24_acquisition": True,
        "size24_acquisition_imports_no_fitting_or_scoring_module": True,
        "conditional_size32_requires_passing_size24_summary_sha": True,
    }


def load_verified_registration(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Verify the E29.11 seal and all response-blind dependencies."""
    registration = load_verified_json(path, expected_sha256)
    require_exact_keys(registration, REGISTRATION_KEYS, label="E29.11 registration")
    if registration["schema_version"] != "e29_11_scale_ladder_registration_v1":
        raise RuntimeError("unexpected E29.11 registration schema")
    if registration["status"] != "sealed_before_any_e29_11_scientific_outcome":
        raise RuntimeError("E29.11 is not in its sealed state")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("registration contains a protocol deviation")
    if registration["environment"] != runtime_environment():
        raise RuntimeError("runtime environment differs from E29.11 seal")
    expected_execution = {
        "device": "cpu",
        "model_dtype": "float64",
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "batch_size": BATCH_SIZE,
        "torch_num_threads_during_fitting": 1,
    }
    if registration["execution"] != expected_execution:
        raise RuntimeError("E29.11 execution recipe changed")
    expected_splits = (*FIT_ACQUISITION_SPLITS, *PREDICTION_SPLITS)
    if tuple(registration["mask_split_order"]) != expected_splits:
        raise RuntimeError("E29.11 split order changed")
    if registration["mask_shapes"] != {
        name: list(shape) for name, shape in MASK_SHAPES.items()
    }:
        raise RuntimeError("E29.11 mask shapes changed")
    if tuple(registration["method_manifest"]) != METHOD_MANIFEST:
        raise RuntimeError("E29.11 method manifest changed")
    expected_science = {
        "primary_method": PRIMARY_METHOD,
        "secondary_method": SECONDARY_METHOD,
        "primary_ridge": PRIMARY_RIDGE,
        "secondary_ridge": SECONDARY_RIDGE,
        "multiscale_fit_protocol": _jsonable(MULTISCALE_FIT_PROTOCOL),
        "size24_confirmation_gates": SIZE24_CONFIRMATION_GATES,
        "tt2_secondary_gates": TT2_SECONDARY_GATES,
        "mlp7_over_tt2_superiority_gates": MLP7_OVER_TT2_SUPERIORITY_GATES,
        "conditional_size32_gates": C32_PROMOTION_GATES,
        "query_budget": QUERY_BUDGET,
        "numerical_gates": NUMERICAL_GATES,
    }
    for key, expected in expected_science.items():
        if registration[key] != expected:
            raise RuntimeError(f"E29.11 registered field changed: {key}")
    if registration["bootstrap"] != {"seed": 29_111, "replicates": 5_000}:
        raise RuntimeError("E29.11 bootstrap contract changed")
    if registration["outcome_custody"] != _expected_outcome_custody():
        raise RuntimeError("E29.11 outcome-custody contract changed")

    smoke = registration["batch_invariance_smoke"]
    require_exact_keys(
        smoke,
        {
            "context_index",
            "prompt_source",
            "prompt_source_registration_sha256",
            "uses_e29_11_prompt_context",
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
        label="E29.11 batch smoke",
    )
    if (
        not smoke["passed"]
        or smoke["uses_scientific_masks"]
        or smoke["uses_e29_11_prompt_context"]
        or smoke["prompt_source"] != "sealed_e29_10_context_0"
        or smoke["prompt_source_registration_sha256"]
        != registration["parent_e29_10_registration_sha256"]
    ):
        raise RuntimeError("E29.11 response-blind batch smoke failed")
    if smoke["used_for_model_selection"] or not smoke[
        "outside_scientific_query_accounting"
    ]:
        raise RuntimeError("E29.11 smoke role changed")

    feature_preflight = registration["source_feature_preflight"]
    if feature_preflight != {
        "uses_e29_11_response_labels": False,
        "component_count": 48,
        "mlp7_hidden_width": 7,
        "mlp7_selected_seed": 29_814,
        "tt2_rank": 2,
        "tt2_natural_order": True,
        "all_state_tensors_finite": True,
        "passed": True,
    }:
        raise RuntimeError("E29.11 source-feature preflight changed")

    for key in (
        "parent_e29_10_registration",
        "protocol_note",
        "protocol_masks",
        "e29_8_feature_artifact",
    ):
        verify_file_hash(Path(registration[key]), registration[f"{key}_sha256"])
    for family in ("prior_prompt_registrations", "prior_mask_artifacts"):
        for record in registration[family].values():
            require_exact_keys(record, {"path", "sha256"}, label=family)
            verify_file_hash(Path(record["path"]), record["sha256"])

    require_exact_keys(
        registration["checkpoint_sha256"],
        CHECKPOINT_FILES,
        label="E29.11 checkpoint hash manifest",
    )
    snapshot = Path(registration["model_snapshot"])
    for filename, expected in registration["checkpoint_sha256"].items():
        verify_file_hash(snapshot / filename, expected)
    require_exact_keys(
        registration["source_code_sha256"],
        SOURCE_FILES,
        label="E29.11 source hash manifest",
    )
    source_directory = Path(__file__).resolve().parent
    for filename, expected in registration["source_code_sha256"].items():
        verify_file_hash(source_directory / filename, expected)

    # This parent loader verifies the exact checkpoint, current float64 runtime,
    # patch semantics, and frozen E29.8 feature artifact transitively.
    from experiments.e29_gpt2_stable_head_interactions.e29_10_custody import (
        load_verified_registration as load_parent_registration,
    )

    parent = load_parent_registration(
        Path(registration["parent_e29_10_registration"]),
        expected_sha256=registration["parent_e29_10_registration_sha256"],
    )
    for key in (
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "component_map",
        "patch_scope",
        "component_semantics",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
    ):
        if registration[key] != parent[key]:
            raise RuntimeError(f"E29.11 changed parent field {key}")
    return registration


__all__ = [
    "REGISTRATION_KEYS",
    "SOURCE_FILES",
    "atomic_json_save",
    "atomic_torch_save",
    "load_verified_json",
    "load_verified_registration",
    "require_exact_keys",
    "runtime_environment",
    "sha256",
    "verify_file_hash",
]
