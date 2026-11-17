"""Write-once storage and strict hash custody for prospective E29.9 stages."""

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

REGISTRATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "adaptive_origin",
        "e29_8_feature_artifact",
        "e29_8_feature_artifact_sha256",
        "e29_8_mlp7_origin_verification",
        "e29_8_source_gate_result",
        "e29_8_source_gate_result_sha256",
        "e29_8_source_preparation",
        "e29_8_source_preparation_sha256",
        "e29_8_registration",
        "e29_8_registration_sha256",
        "e29_8_protocol_masks",
        "e29_8_protocol_masks_sha256",
        "model_snapshot",
        "model_revision",
        "environment",
        "checkpoint_sha256",
        "component_map",
        "patch_scope",
        "component_semantics",
        "protocol_note",
        "protocol_note_sha256",
        "protocol_masks",
        "protocol_masks_sha256",
        "prompt_checks",
        "prompt_panel_sha256",
        "explicit_masks",
        "mask_shapes",
        "method_manifest",
        "primary_method",
        "primary_budgets",
        "ridge_values",
        "observed_primary_feature_preflight",
        "scratch_mlp_recipe",
        "control_seeds",
        "canonical_permutation_gate_methods",
        "primary_size16_gates",
        "stress_size24_gates",
        "bootstrap",
        "query_budget",
        "execution",
        "outcome_custody",
        "source_code_sha256",
        "protocol_deviations",
    }
)
CHECKPOINT_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    }
)


def sha256(path: Path) -> str:
    """Return a file SHA-256 using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected: str) -> str:
    """Fail closed if ``path`` is absent or differs from its expected digest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"hash mismatch for {path}")
    return observed


def require_exact_keys(
    record: Mapping[str, Any], expected: Collection[str], *, label: str
) -> None:
    """Reject missing and extra keys in a custody-critical mapping."""
    observed = set(record)
    required = set(expected)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise RuntimeError(f"{label} keys changed; missing={missing}, extra={extra}")


def atomic_torch_save(record: Any, path: Path) -> None:
    """Atomically create a torch artifact and never overwrite an existing path."""
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
    """Atomically create formatted JSON and never overwrite an existing path."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix="tmp-", suffix=".json"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify an external handoff digest before parsing its JSON artifact."""
    verify_file_hash(path, expected_sha256)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def load_verified_registration(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Verify the external E29.9 seal and every file hash embedded in it."""
    registration = load_verified_json(path, expected_sha256)
    require_exact_keys(registration, REGISTRATION_KEYS, label="E29.9 registration")
    if registration["schema_version"] != (
        "e29_9_gpt2_medium_adaptive_transfer_registration_v1"
    ):
        raise RuntimeError("unexpected E29.9 registration schema")
    if registration["status"] != "sealed_before_any_e29_9_model_outcome":
        raise RuntimeError("E29.9 registration is not sealed")
    adaptive = registration["adaptive_origin"]
    if adaptive != {
        "status": "separate_post_e29_8_adaptive_experiment",
        "not_e29_8_conditional_stage_b": True,
        "e29_8_source_gate_passed": False,
        "sole_failed_condition": "mse_at_most_0_90_mlp7",
        "e29_8_fresh_outcomes_remain_forbidden": True,
    }:
        raise RuntimeError("E29.9 adaptive-origin declaration changed")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("registration contains an unreviewed protocol deviation")
    if registration["e29_8_mlp7_origin_verification"] != {
        "training_seeds": list(range(29_811, 29_816)),
        "successful_seed_records": 5,
        "selected_seed": 29_814,
        "selection": "lowest_joint_standardized_validation_mse_then_seed",
        "replayed_during_e29_9_registration": False,
        "verified_from_exact_hashed_e29_8_feature_artifact": True,
    }:
        raise RuntimeError("E29.8 MLP7 origin verification changed")
    import safetensors
    import transformers

    expected_environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }
    if registration["environment"] != expected_environment:
        raise RuntimeError("runtime environment differs from E29.9 registration")
    preflight = registration["observed_primary_feature_preflight"]
    require_exact_keys(
        preflight,
        {
            "active_feature_count",
            "centered_standardized_rank",
            "condition_number",
            "singular_values",
            "threshold_rank",
            "maximum_condition_number",
            "passed",
            "uses_response_labels",
        },
        label="observed primary feature preflight",
    )
    if preflight["uses_response_labels"] is not False:
        raise RuntimeError("primary preflight must be response-blind")
    if preflight["passed"] is not True:
        raise RuntimeError("registered primary feature preflight did not pass")
    if preflight["active_feature_count"] != 7 or preflight[
        "centered_standardized_rank"
    ] != 7:
        raise RuntimeError("registered primary feature design is not rank seven")
    condition = float(preflight["condition_number"])
    if not (condition <= 1_000.0):
        raise RuntimeError("registered primary feature design is ill-conditioned")
    expected_seed_families = {
        "scratch_mlp_training": list(range(29_911, 29_916)),
        "untrained_random_features": list(range(29_921, 29_926)),
        "learned_component_permutations": list(range(29_931, 29_941)),
    }
    if registration["control_seeds"] != expected_seed_families:
        raise RuntimeError("E29.9 control seed families changed")
    expected_permutation_gate_methods = [
        *(f"mlp7_permuted_seed{seed}" for seed in range(29_931, 29_941)),
        "mlp7_permuted_reverse",
        "mlp7_permuted_attn_mlp_swap",
    ]
    if registration["canonical_permutation_gate_methods"] != (
        expected_permutation_gate_methods
    ):
        raise RuntimeError("canonical permutation gate does not use all 12 controls")
    scratch_recipe = registration["scratch_mlp_recipe"]
    expected_scratch_recipe = {
        "training_seeds": list(range(29_911, 29_916)),
        "learning_rate": 0.01,
        "weight_decay": 1e-4,
        "maximum_epochs": 1_600,
        "evaluation_interval": 20,
        "patience_epochs": 400,
        "gradient_clip_norm": 5.0,
        "ridge_values": [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0],
        "random_order_seed": 29_809,
        "selection": "lowest_joint_standardized_validation_mse_then_seed",
        "checkpointing": "best_validation_checkpoint_within_each_seed",
        "refit_after_validation": False,
    }
    if scratch_recipe != expected_scratch_recipe:
        raise RuntimeError("E29.9 scratch MLP recipe changed")

    for key in (
        "e29_8_feature_artifact",
        "e29_8_source_gate_result",
        "e29_8_source_preparation",
        "e29_8_registration",
        "e29_8_protocol_masks",
        "protocol_note",
        "protocol_masks",
    ):
        verify_file_hash(Path(registration[key]), registration[f"{key}_sha256"])
    snapshot = Path(registration["model_snapshot"])
    if snapshot.name != registration["model_revision"]:
        raise RuntimeError("model snapshot and registered revision differ")
    require_exact_keys(
        registration["checkpoint_sha256"],
        CHECKPOINT_FILES,
        label="checkpoint hash manifest",
    )
    for name, expected in registration["checkpoint_sha256"].items():
        verify_file_hash(snapshot / name, expected)
    source_directory = Path(__file__).resolve().parent
    for name, expected in registration["source_code_sha256"].items():
        verify_file_hash(source_directory / name, expected)
    return registration


__all__ = [
    "atomic_json_save",
    "atomic_torch_save",
    "load_verified_json",
    "load_verified_registration",
    "require_exact_keys",
    "sha256",
    "verify_file_hash",
]
