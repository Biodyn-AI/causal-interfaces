"""Strict hash and write-once custody for prospective E29.14a artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    CHECKPOINT_FILES,
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_11_custody import (
    atomic_json_save,
    atomic_torch_save,
    load_verified_json,
    require_exact_keys,
    runtime_environment,
    sha256,
    verify_file_hash,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    BOOTSTRAP,
    FORWARD_CONDITIONS,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
    SCORING_GATES,
    validate_position_protocol,
)

PARENT_E29_13_REGISTRATION_SHA256 = (
    "5ac67250844bfa84797b7fbb1263a290f371a96b47446415523a6e3a444b5b4d"
)
DEVELOPMENT_POSITION_POSTHOC_SHA256 = (
    "9874f22dc28fc749ff47584695f5ec97997144dd71a9efd96210c9787e9232e3"
)
SOURCE_FILES = (
    "acquire_e29_14a_position.py",
    "e29_10_custody.py",
    "e29_10_float64_protocol.py",
    "e29_11_custody.py",
    "e29_11_fitting.py",
    "e29_11_multiscale_protocol.py",
    "e29_12_custody.py",
    "e29_12_endpoint_protocol.py",
    "e29_12_fitting.py",
    "e29_12_prompt_panel.py",
    "e29_13_absorbing_hub_fitting.py",
    "e29_13_absorbing_hub_protocol.py",
    "e29_13_custody.py",
    "e29_13_prompt_panel.py",
    "e29_14a_prompt_panel.py",
    "e29_14a_position_protocol.py",
    "e29_14a_position_runner.py",
    "e29_14a_position_custody.py",
    "e29_8_prompt_panel.py",
    "e29_8_scaleup_protocol.py",
    "e29_9_custody.py",
    "e29_9_seven_gate_protocol.py",
    "e29_9_transfer_fitting.py",
    "register_e29_14a_position.py",
    "scaled_tensor_response.py",
    "score_e29_14a_position.py",
    "source_model_fitting.py",
    "tensor_train_response.py",
)
BATCH_INVARIANCE = {
    "context_index": 0,
    "directions": ["forward", "reverse"],
    "forward_conditions": [dict(row) for row in FORWARD_CONDITIONS],
    "reverse_conditions": [dict(row) for row in REVERSE_CONDITIONS],
    "reference_batch_size": 1,
    "candidate_batch_size": 32,
    "absolute_tolerance": 1e-9,
    "relative_tolerance": 1e-9,
    "smoke_outputs_enter_scientific_labels": False,
}
PHYSICAL_QUERY_ACCOUNTING = {
    **QUERY_ACCOUNTING,
    "batch_smoke_unique_condition_rows": 16,
    "batch_smoke_physical_condition_cells": 32,
    "maximum_physical_model_examples": 248,
}
OUTCOME_CUSTODY = {
    "registration_sha_required_before_acquisition": True,
    "write_once_attempt_marker_created_before_model_load": True,
    "any_attempt_marker_blocks_silent_rerun": True,
    "all_16_conditions_acquired_nonadaptively_per_context": True,
    "forward_and_reverse_acquired_in_one_uninterrupted_stage": True,
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
        "parent_e29_13_registration",
        "parent_e29_13_registration_sha256",
        "development_evidence",
        "protocol_note",
        "protocol_note_sha256",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "component_map",
        "hub_component",
        "control_component",
        "patch_semantics",
        "prompt_records",
        "prompt_checks",
        "prompt_panel_sha256",
        "position_protocol",
        "position_protocol_sha256",
        "condition_manifest",
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using a stable canonical encoding."""
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_source_closure(
    source_directory: Path,
    filenames: Sequence[str],
) -> dict[str, str]:
    """Hash an exact ordered list of plain source filenames."""
    names = tuple(str(name) for name in filenames)
    if not names or len(names) != len(set(names)):
        raise ValueError("source filenames must be nonempty and unique")
    if any(Path(name).name != name for name in names):
        raise ValueError("source closure entries must be plain filenames")
    hashes = {}
    for name in names:
        path = source_directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
    return hashes


def verify_checkpoint_closure(
    snapshot: Path,
    expected_hashes: Mapping[str, str],
    required_files: Sequence[str],
) -> dict[str, str]:
    """Require exact checkpoint filenames and verify every digest."""
    names = tuple(str(name) for name in required_files)
    if not names or len(names) != len(set(names)):
        raise ValueError("checkpoint filenames must be nonempty and unique")
    require_exact_keys(expected_hashes, names, label="E29.14a checkpoint hashes")
    verified = {}
    for name in names:
        verified[name] = verify_file_hash(snapshot / name, str(expected_hashes[name]))
    return verified


def load_verified_registration(
    path: Path,
    *,
    expected_sha256: str,
    required_keys: Collection[str] | None = None,
) -> dict[str, Any]:
    """Load one exact registration hash and optionally enforce its key set."""
    registration = load_verified_json(path, expected_sha256)
    if required_keys is not None:
        require_exact_keys(
            registration,
            required_keys,
            label="E29.14a registration",
        )
    return registration


def validate_registration_payload(
    registration: Mapping[str, Any],
    *,
    source_directory: Path | None = None,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Validate the exact sealed scientific and custody contract."""
    if not isinstance(registration, Mapping):
        raise TypeError("E29.14a registration must be a mapping")
    require_exact_keys(
        registration,
        REGISTRATION_REQUIRED_KEYS,
        label="E29.14a registration",
    )
    if (
        registration["schema_version"] != "e29_14a_position_registration_v1"
        or registration["status"]
        != "prospective_sealed_before_any_e29_14a_model_response"
    ):
        raise RuntimeError("E29.14a registration status changed")
    if (
        registration["parent_e29_13_registration_sha256"]
        != PARENT_E29_13_REGISTRATION_SHA256
    ):
        raise RuntimeError("E29.14a parent registration changed")

    protocol = validate_position_protocol(registration["position_protocol"])
    if registration["position_protocol_sha256"] != canonical_sha256(protocol):
        raise RuntimeError("E29.14a protocol canonical hash changed")
    expected_manifest = {
        "forward": _jsonable(FORWARD_CONDITIONS),
        "reverse": _jsonable(REVERSE_CONDITIONS),
    }
    if registration["condition_manifest"] != expected_manifest:
        raise RuntimeError("E29.14a condition manifest changed")
    if registration["bootstrap"] != BOOTSTRAP:
        raise RuntimeError("E29.14a bootstrap changed")
    if registration["scoring_gates"] != _jsonable(SCORING_GATES):
        raise RuntimeError("E29.14a scoring gates changed")
    if registration["query_accounting"] != QUERY_ACCOUNTING:
        raise RuntimeError("E29.14a logical query accounting changed")
    if registration["physical_query_accounting"] != PHYSICAL_QUERY_ACCOUNTING:
        raise RuntimeError("E29.14a physical query accounting changed")
    if registration["batch_invariance"] != BATCH_INVARIANCE:
        raise RuntimeError("E29.14a batch-invariance contract changed")
    if registration["numerical_gates"] != NUMERICAL_GATES:
        raise RuntimeError("E29.14a numerical gates changed")
    if registration["outcome_custody"] != OUTCOME_CUSTODY:
        raise RuntimeError("E29.14a outcome custody changed")
    if registration["hub_component"] != 1 or registration["control_component"] != 0:
        raise RuntimeError("E29.14a component indices changed")
    require_exact_keys(
        registration["checkpoint_sha256"],
        CHECKPOINT_FILES,
        label="E29.14a checkpoint hashes",
    )

    component_map = registration["component_map"]
    if (
        not isinstance(component_map, list)
        or len(component_map) <= 1
        or component_map[0] != [0, "attention"]
        or component_map[1] != [0, "mlp"]
    ):
        raise RuntimeError("E29.14a component map changed")
    records = registration["prompt_records"]
    checks = registration["prompt_checks"]
    if not isinstance(records, list) or not isinstance(checks, list):
        raise TypeError("E29.14a prompts and checks must be lists")
    if len(records) != 12 or len(checks) != 12:
        raise RuntimeError("E29.14a requires twelve registered prompt contexts")
    prompt_keys = {
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
        require_exact_keys(record, prompt_keys, label="E29.14a prompt record")
        require_exact_keys(check, check_keys, label="E29.14a tokenizer check")
        if check["context_index"] != context_index:
            raise RuntimeError("E29.14a tokenizer-check order changed")
        for key in prompt_keys:
            if str(record[key]) != str(check[key]):
                raise RuntimeError("E29.14a prompt check differs from prompt record")
        if check.get("target_positions_aligned") is not True:
            raise RuntimeError("E29.14a prompt target positions are not aligned")
        token_length = check["token_length"]
        if (
            isinstance(token_length, bool)
            or not isinstance(token_length, int)
            or token_length < 2
        ):
            raise RuntimeError("E29.14a prompt token length is invalid")
        sides = check.get("sides")
        if not isinstance(sides, Mapping) or set(sides) != {"base", "donor"}:
            raise RuntimeError("E29.14a tokenizer side checks changed")
        for name, side in sides.items():
            require_exact_keys(
                side,
                side_keys,
                label=f"E29.14a {name} tokenizer side check",
            )
            words = check[f"{name}_words"]
            word_lengths = side["word_token_lengths"]
            if (
                not isinstance(words, list)
                or len(words) != 4
                or len(set(words)) != 4
                or not isinstance(word_lengths, Mapping)
                or set(word_lengths) != set(words)
                or set(word_lengths.values()) != {1}
                or side["target"] != record[f"{name}_target"]
                or side["target"] != words[3]
                or side["next_token_append_valid"] is not True
            ):
                raise RuntimeError("E29.14a tokenizer word/append check failed")
            token_id = side["target_token_id"]
            target_position = side["demonstration_target_position"]
            prediction_position = side["prediction_position"]
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or isinstance(target_position, bool)
                or not isinstance(target_position, int)
                or not 0 <= target_position < token_length
                or prediction_position != token_length
            ):
                raise RuntimeError("E29.14a tokenizer target metadata is invalid")
        if (
            sides["base"]["target_token_id"] == sides["donor"]["target_token_id"]
            or sides["base"]["demonstration_target_position"]
            != sides["donor"]["demonstration_target_position"]
        ):
            raise RuntimeError("E29.14a tokenizer target alignment changed")
    if registration["prompt_panel_sha256"] != canonical_sha256(records):
        raise RuntimeError("E29.14a prompt-panel canonical hash changed")
    require_exact_keys(
        registration["source_code_sha256"],
        SOURCE_FILES,
        label="E29.14a source-code hashes",
    )
    if registration["protocol_deviations"] != []:
        raise RuntimeError("E29.14a protocol deviations must be empty at seal")

    development = registration["development_evidence"]
    require_exact_keys(
        development,
        {
            "position_semantics_posthoc",
            "position_semantics_posthoc_sha256",
            "schema_version",
            "status",
            "confirmatory_status",
        },
        label="E29.14a development evidence",
    )
    if (
        development["position_semantics_posthoc_sha256"]
        != DEVELOPMENT_POSITION_POSTHOC_SHA256
    ):
        raise RuntimeError("E29.14a development evidence hash changed")

    if verify_external_files:
        parent_path = Path(registration["parent_e29_13_registration"])
        verify_file_hash(
            parent_path,
            PARENT_E29_13_REGISTRATION_SHA256,
        )
        parent = load_verified_json(parent_path, PARENT_E29_13_REGISTRATION_SHA256)
        inherited_keys = (
            "model_snapshot",
            "model_revision",
            "checkpoint_sha256",
            "execution",
            "component_map",
        )
        if any(
            _jsonable(registration[key]) != _jsonable(parent[key])
            for key in inherited_keys
        ):
            raise RuntimeError("E29.14a inherited parent execution changed")
        verify_file_hash(
            Path(development["position_semantics_posthoc"]),
            DEVELOPMENT_POSITION_POSTHOC_SHA256,
        )
        verify_file_hash(
            Path(registration["protocol_note"]),
            str(registration["protocol_note_sha256"]),
        )
        verify_checkpoint_closure(
            Path(registration["model_snapshot"]),
            registration["checkpoint_sha256"],
            CHECKPOINT_FILES,
        )
        if source_directory is None:
            source_directory = Path(__file__).resolve().parent
        observed_sources = verify_source_closure(source_directory, SOURCE_FILES)
        if registration["source_code_sha256"] != observed_sources:
            raise RuntimeError("E29.14a source closure changed after registration")
    return dict(registration)


__all__ = [
    "BATCH_INVARIANCE",
    "DEVELOPMENT_POSITION_POSTHOC_SHA256",
    "OUTCOME_CUSTODY",
    "PARENT_E29_13_REGISTRATION_SHA256",
    "PHYSICAL_QUERY_ACCOUNTING",
    "REGISTRATION_REQUIRED_KEYS",
    "SOURCE_FILES",
    "atomic_json_save",
    "atomic_torch_save",
    "canonical_sha256",
    "load_verified_registration",
    "require_exact_keys",
    "runtime_environment",
    "sha256",
    "validate_registration_payload",
    "verify_checkpoint_closure",
    "verify_file_hash",
    "verify_source_closure",
]
