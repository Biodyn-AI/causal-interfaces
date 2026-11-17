"""Strict, response-blind custody for prospective E29.14b artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    BOOTSTRAP,
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    EXPECTED_MASK_CANONICAL_SHA256,
    HUB_COMPONENT,
    MASK_CARDINALITIES,
    MASK_HUB_ON_COUNTS,
    MASK_SHAPES,
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    SINGLETON_CONDITIONS,
    validate_component_position_protocol,
)

PARENT_E29_14A_REGISTRATION_SHA256 = (
    "f6ef1755bd6a4c3345a78fe180f00d7dd5e98086319348f34d9a1d9592ee23df"
)
PRIOR_PROTOCOL_MASKS_SHA256 = (
    "014b87e0ddf2cf913ff745e41e12bb6444420716753d0eb712bd465eaa0c83fd"
)

# Acquisition and scoring are deliberately named before they are implemented.
# The registration cannot seal until the complete executable source closure exists.
SOURCE_FILES = (
    "acquire_e29_14b_component_position.py",
    "e29_14b_component_position_custody.py",
    "e29_14b_component_position_protocol.py",
    "e29_14b_component_position_runner.py",
    "e29_14b_prompt_panel.py",
    "register_e29_14b_component_position.py",
    "score_e29_14b_component_position.py",
    # Transitive local dependencies of the prompt panel and runner.
    "e29_8_prompt_panel.py",
    "e29_9_seven_gate_protocol.py",
    "e29_10_float64_protocol.py",
    "e29_11_multiscale_protocol.py",
    "e29_12_prompt_panel.py",
    "e29_13_prompt_panel.py",
    "e29_14a_prompt_panel.py",
    "e29_14a_position_protocol.py",
    "generalized_sublayer_patch.py",
)

BATCH_INVARIANCE = {
    "context_index": 0,
    "condition_rows": QUERY_ACCOUNTING["patched_condition_rows_per_context"],
    "reference_batch_size": 1,
    "candidate_batch_size": 32,
    "absolute_tolerance": 1e-9,
    "relative_tolerance": 1e-9,
    "physical_condition_cells": (
        2 * QUERY_ACCOUNTING["patched_condition_rows_per_context"]
    ),
    "smoke_outputs_enter_scientific_labels": False,
}
PHYSICAL_QUERY_ACCOUNTING = {
    **QUERY_ACCOUNTING,
    "batch_smoke_unique_condition_rows": QUERY_ACCOUNTING[
        "patched_condition_rows_per_context"
    ],
    "batch_smoke_physical_condition_cells": (
        2 * QUERY_ACCOUNTING["patched_condition_rows_per_context"]
    ),
    "maximum_physical_model_examples": (
        QUERY_ACCOUNTING["maximum_logical_queries"]
        + 2 * QUERY_ACCOUNTING["patched_condition_rows_per_context"]
    ),
}
OUTCOME_CUSTODY = {
    "registration_sha_required_before_acquisition": True,
    "source_checkpoint_mask_and_environment_verified_before_attempt": True,
    "write_once_attempt_marker_created_before_model_load": True,
    "any_attempt_marker_blocks_silent_rerun": True,
    "all_194_conditions_acquired_nonadaptively_per_context": True,
    "batch_smoke_runs_before_scientific_label_write": True,
    "batch_smoke_outputs_excluded_from_scientific_labels": True,
    "clean_and_numerical_gates_required_before_label_write": True,
    "attempt_labels_and_custody_are_write_once": True,
}

REGISTRATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "prospective_scope",
        "parent_e29_14a_registration",
        "parent_e29_14a_registration_sha256",
        "protocol_note",
        "protocol_note_sha256",
        "prior_protocol_masks",
        "prior_protocol_masks_sha256",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "component_map",
        "hub_component",
        "patch_semantics",
        "prompt_records",
        "prompt_checks",
        "prompt_panel_sha256",
        "component_position_protocol",
        "component_position_protocol_sha256",
        "prior_mask_canonical_sha256",
        "bootstrap",
        "scoring_gates",
        "query_accounting",
        "physical_query_accounting",
        "batch_invariance",
        "numerical_gates",
        "outcome_custody",
        "source_code_sha256",
        "protocol_deviations",
    }
)

ATTEMPT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "registration",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "source_code_sha256_verified_before_attempt",
        "checkpoint_sha256_verified_before_attempt",
        "prior_protocol_masks_sha256_verified_before_attempt",
        "prompt_panel_sha256",
        "component_position_protocol_sha256",
        "query_accounting",
        "physical_query_accounting",
        "environment",
        "execution",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using a stable encoding."""
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected: str) -> str:
    """Fail closed if a required file is absent or its digest differs."""
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


def atomic_json_save(record: Any, path: Path) -> None:
    """Atomically create strict JSON and refuse every overwrite."""
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


def atomic_torch_save(record: Any, path: Path) -> None:
    """Atomically create a Torch artifact and refuse every overwrite."""
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


def load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify and load one JSON object."""
    verify_file_hash(path, expected_sha256)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def load_verified_registration(
    path: Path,
    *,
    expected_sha256: str,
    required_keys: Collection[str] | None = None,
) -> dict[str, Any]:
    """Load one exact registration and optionally enforce its complete schema."""
    registration = load_verified_json(path, expected_sha256)
    if required_keys is not None:
        require_exact_keys(registration, required_keys, label="E29.14b registration")
    return registration


def runtime_environment() -> dict[str, str]:
    """Return the exact dependency/runtime identity sealed for acquisition."""
    import safetensors
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }


def verify_source_closure(
    source_directory: Path, filenames: Sequence[str]
) -> dict[str, str]:
    """Hash an exact ordered list of plain local source filenames."""
    names = tuple(str(name) for name in filenames)
    if not names or len(names) != len(set(names)):
        raise ValueError("source filenames must be nonempty and unique")
    if any(Path(name).name != name for name in names):
        raise ValueError("source closure entries must be plain filenames")
    return {name: sha256(source_directory / name) for name in names}


def verify_checkpoint_closure(
    snapshot: Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Verify exactly the checkpoint filenames inherited from E29.14a."""
    if not expected_hashes:
        raise RuntimeError("E29.14b checkpoint closure is empty")
    verified = {}
    for name, digest in expected_hashes.items():
        if Path(str(name)).name != str(name):
            raise RuntimeError("checkpoint closure contains a non-plain filename")
        verified[str(name)] = verify_file_hash(snapshot / str(name), str(digest))
    return verified


def build_acquisition_attempt_record(
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    observed_registration_sha256: str,
    expected_registration_sha256: str,
) -> dict[str, Any]:
    """Build the marker that irrevocably consumes one registered attempt."""
    marker = {
        "schema_version": "e29_14b_component_position_acquisition_attempt_v1",
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
        "prior_protocol_masks_sha256_verified_before_attempt": registration[
            "prior_protocol_masks_sha256"
        ],
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "component_position_protocol_sha256": registration[
            "component_position_protocol_sha256"
        ],
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "environment": dict(registration["environment"]),
        "execution": dict(registration["execution"]),
    }
    require_exact_keys(marker, ATTEMPT_REQUIRED_KEYS, label="E29.14b attempt")
    return marker


def create_acquisition_attempt_marker(record: Mapping[str, Any], path: Path) -> str:
    """Write the attempt marker once; acquisition must call this before model load."""
    require_exact_keys(record, ATTEMPT_REQUIRED_KEYS, label="E29.14b attempt")
    if record["status"] != "attempt_consumed_before_model_load":
        raise RuntimeError("E29.14b attempt status changed")
    atomic_json_save(dict(record), path)
    return sha256(path)


def _validate_prompt_contract(registration: Mapping[str, Any]) -> None:
    records = registration["prompt_records"]
    checks = registration["prompt_checks"]
    if not isinstance(records, list) or not isinstance(checks, list):
        raise TypeError("E29.14b prompts and checks must be lists")
    if len(records) != CONTEXT_COUNT or len(checks) != CONTEXT_COUNT:
        raise RuntimeError("E29.14b requires twelve prompt contexts")
    record_keys = {
        "template",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
    }
    check_keys = {
        "context_index",
        "token_length",
        "target_positions_aligned",
        "sides",
        "template",
        "base_words",
        "donor_words",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
    }
    side_keys = {
        "target",
        "target_token_id",
        "demonstration_target_position",
        "prediction_position",
        "next_token_append_valid",
        "word_token_lengths",
    }
    for context_index, (record, check) in enumerate(
        zip(records, checks, strict=True)
    ):
        require_exact_keys(record, record_keys, label="E29.14b prompt record")
        require_exact_keys(check, check_keys, label="E29.14b prompt check")
        if check["context_index"] != context_index:
            raise RuntimeError("E29.14b prompt-check order changed")
        if any(str(record[key]) != str(check[key]) for key in record_keys):
            raise RuntimeError("E29.14b prompt check differs from prompt record")
        if check["target_positions_aligned"] is not True:
            raise RuntimeError("E29.14b target positions are not aligned")
        token_length = check["token_length"]
        if isinstance(token_length, bool) or not isinstance(token_length, int):
            raise TypeError("E29.14b token length is invalid")
        sides = check["sides"]
        if not isinstance(sides, Mapping) or set(sides) != {"base", "donor"}:
            raise RuntimeError("E29.14b tokenizer side checks changed")
        for side_name, side in sides.items():
            require_exact_keys(
                side, side_keys, label=f"E29.14b {side_name} tokenizer check"
            )
            words = check[f"{side_name}_words"]
            lengths = side["word_token_lengths"]
            if (
                not isinstance(words, list)
                or len(words) != 4
                or not isinstance(lengths, Mapping)
                or set(lengths) != set(words)
                or set(lengths.values()) != {1}
                or side["target"] != record[f"{side_name}_target"]
                or side["target"] != words[3]
                or side["next_token_append_valid"] is not True
            ):
                raise RuntimeError("E29.14b tokenizer word contract failed")
            target_position = side["demonstration_target_position"]
            if (
                isinstance(target_position, bool)
                or not isinstance(target_position, int)
                or not 0 <= target_position < token_length
                or side["prediction_position"] != token_length
            ):
                raise RuntimeError("E29.14b tokenizer position metadata is invalid")
        if (
            sides["base"]["target_token_id"] == sides["donor"]["target_token_id"]
            or sides["base"]["demonstration_target_position"]
            != sides["donor"]["demonstration_target_position"]
        ):
            raise RuntimeError("E29.14b tokenizer target alignment changed")
    if registration["prompt_panel_sha256"] != canonical_sha256(records):
        raise RuntimeError("E29.14b prompt-panel hash changed")


def validate_registration_payload(
    registration: Mapping[str, Any],
    *,
    source_directory: Path | None = None,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Validate the exact E29.14b scientific and custody contract."""
    if not isinstance(registration, Mapping):
        raise TypeError("E29.14b registration must be a mapping")
    require_exact_keys(
        registration, REGISTRATION_REQUIRED_KEYS, label="E29.14b registration"
    )
    if (
        registration["schema_version"]
        != "e29_14b_component_position_registration_v1"
        or registration["status"]
        != "prospective_sealed_before_any_e29_14b_model_response"
    ):
        raise RuntimeError("E29.14b registration status changed")
    if (
        registration["parent_e29_14a_registration_sha256"]
        != PARENT_E29_14A_REGISTRATION_SHA256
        or registration["prior_protocol_masks_sha256"]
        != PRIOR_PROTOCOL_MASKS_SHA256
    ):
        raise RuntimeError("E29.14b immutable parent hash changed")

    protocol = validate_component_position_protocol(
        registration["component_position_protocol"]
    )
    if registration["component_position_protocol_sha256"] != canonical_sha256(
        protocol
    ):
        raise RuntimeError("E29.14b protocol canonical hash changed")
    if registration["prior_mask_canonical_sha256"] != _jsonable(
        EXPECTED_MASK_CANONICAL_SHA256
    ):
        raise RuntimeError("E29.14b held-mask canonical hash changed")
    if registration["bootstrap"] != _jsonable(BOOTSTRAP):
        raise RuntimeError("E29.14b bootstrap changed")
    if registration["scoring_gates"] != _jsonable(SCORING_GATES):
        raise RuntimeError("E29.14b scoring gates changed")
    if registration["query_accounting"] != _jsonable(QUERY_ACCOUNTING):
        raise RuntimeError("E29.14b logical query accounting changed")
    if registration["physical_query_accounting"] != PHYSICAL_QUERY_ACCOUNTING:
        raise RuntimeError("E29.14b physical query accounting changed")
    if registration["batch_invariance"] != BATCH_INVARIANCE:
        raise RuntimeError("E29.14b batch-invariance contract changed")
    if registration["outcome_custody"] != OUTCOME_CUSTODY:
        raise RuntimeError("E29.14b outcome custody changed")
    if registration["hub_component"] != HUB_COMPONENT:
        raise RuntimeError("E29.14b hub component changed")
    checkpoint_hashes = registration["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_hashes, Mapping)
        or not checkpoint_hashes
        or any(Path(str(name)).name != str(name) for name in checkpoint_hashes)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in checkpoint_hashes.values()
        )
    ):
        raise RuntimeError("E29.14b checkpoint hashes are invalid")
    if registration["protocol_deviations"] != []:
        raise RuntimeError("E29.14b deviations must be empty at seal")
    if not isinstance(registration["source_code_sha256"], Mapping):
        raise TypeError("E29.14b source-code hashes must be a mapping")
    require_exact_keys(
        registration["source_code_sha256"], SOURCE_FILES, label="E29.14b sources"
    )
    _validate_prompt_contract(registration)

    expected_protocol_facts = {
        "component_count": COMPONENT_COUNT,
        "context_count": CONTEXT_COUNT,
        "mask_splits": list(MASK_SPLITS),
        "mask_shapes": _jsonable(MASK_SHAPES),
        "mask_cardinalities": _jsonable(MASK_CARDINALITIES),
        "mask_hub_on_counts": _jsonable(MASK_HUB_ON_COUNTS),
        "position_scope_order": list(POSITION_SCOPE_ORDER),
        "singleton_conditions": _jsonable(SINGLETON_CONDITIONS),
    }
    for key, expected in expected_protocol_facts.items():
        if protocol[key] != expected:
            raise RuntimeError(f"E29.14b protocol {key} changed")

    if verify_external_files:
        parent_path = Path(registration["parent_e29_14a_registration"])
        parent = load_verified_json(parent_path, PARENT_E29_14A_REGISTRATION_SHA256)
        inherited = (
            "model_snapshot",
            "model_revision",
            "checkpoint_sha256",
            "execution",
            "component_map",
            "numerical_gates",
        )
        if any(
            _jsonable(registration[key]) != _jsonable(parent[key])
            for key in inherited
        ):
            raise RuntimeError("E29.14b inherited E29.14a execution changed")
        verify_file_hash(
            Path(registration["prior_protocol_masks"]), PRIOR_PROTOCOL_MASKS_SHA256
        )
        verify_file_hash(
            Path(registration["protocol_note"]),
            str(registration["protocol_note_sha256"]),
        )
        verify_checkpoint_closure(
            Path(registration["model_snapshot"]), registration["checkpoint_sha256"]
        )
        directory = source_directory or Path(__file__).resolve().parent
        observed_sources = verify_source_closure(directory, SOURCE_FILES)
        if registration["source_code_sha256"] != observed_sources:
            raise RuntimeError("E29.14b source closure changed after registration")
    return dict(registration)


__all__ = [
    "ATTEMPT_REQUIRED_KEYS",
    "BATCH_INVARIANCE",
    "OUTCOME_CUSTODY",
    "PARENT_E29_14A_REGISTRATION_SHA256",
    "PHYSICAL_QUERY_ACCOUNTING",
    "PRIOR_PROTOCOL_MASKS_SHA256",
    "REGISTRATION_REQUIRED_KEYS",
    "SOURCE_FILES",
    "atomic_json_save",
    "atomic_torch_save",
    "build_acquisition_attempt_record",
    "canonical_sha256",
    "create_acquisition_attempt_marker",
    "load_verified_json",
    "load_verified_registration",
    "require_exact_keys",
    "runtime_environment",
    "sha256",
    "validate_registration_payload",
    "verify_checkpoint_closure",
    "verify_file_hash",
    "verify_source_closure",
]
