"""Register, acquire, and score the bounded E33.1 confirmation exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from experiments.e33_ioi_scope_replication.protocol import (
    MODEL_ID,
    MODEL_REVISION,
    build_prompt_manifest,
    build_protocol,
    canonical_sha256,
    validate_prompt_manifest,
    validate_protocol,
)
from experiments.e33_ioi_scope_replication.scoring import score_context_records

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REGISTRATION = REPOSITORY_ROOT / "notes/E33.1-IOI-SCOPE-REGISTRATION.json"
DEFAULT_PROTOCOL_NOTE = REPOSITORY_ROOT / "notes/E33.1-IOI-SCOPE-PROTOCOL.md"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e33_1_ioi_scope_replication_v1"
SOURCE_FILES = ("protocol.py", "runner.py", "scoring.py", "workflow.py")
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_once_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _model_hashes(snapshot: Path) -> dict[str, str]:
    missing = [name for name in MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model snapshot is incomplete: {missing}")
    return {name: sha256(snapshot / name) for name in MODEL_FILES}


def _source_hashes() -> dict[str, str]:
    return {name: sha256(SOURCE_DIRECTORY / name) for name in SOURCE_FILES}


def register(
    *,
    registration_path: Path,
    protocol_note_path: Path,
    model_snapshot: Path,
) -> dict[str, Any]:
    """Seal code, prompts, gates, tokenizer checks, and checkpoint bytes."""
    if registration_path.exists():
        raise FileExistsError(f"refusing to overwrite {registration_path}")
    if not protocol_note_path.is_file():
        raise FileNotFoundError("the human-readable protocol note must exist first")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    prompt_checks = validate_prompt_manifest(tokenizer, build_prompt_manifest())
    protocol = build_protocol()
    validate_protocol(protocol)
    registration = {
        "schema_version": "e33_1_ioi_scope_registration_v1",
        "status": "registered_before_confirmation_model_load",
        "registered_unix_time": time.time(),
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "required_files_sha256": _model_hashes(model_snapshot),
            "snapshot_path_is_not_part_of_scientific_identity": True,
        },
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "prompt_tokenization_checks": list(prompt_checks),
        "prompt_tokenization_checks_sha256": canonical_sha256(prompt_checks),
        "protocol_note_sha256": sha256(protocol_note_path),
        "source_files_sha256": _source_hashes(),
        "environment_at_registration": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
        "development_boundary": {
            "context_count": 4,
            "names_disjoint_from_confirmation": True,
            "used_only_to_fix_conditions_and_thresholds": True,
            "confirmation_outcomes_unseen": True,
        },
    }
    _write_once_json(registration, registration_path)
    return {
        "registration": str(registration_path),
        "registration_sha256": sha256(registration_path),
    }


def _verified_registration(
    registration_path: Path,
    expected_sha256: str,
    model_snapshot: Path,
    protocol_note_path: Path,
) -> dict[str, Any]:
    if sha256(registration_path) != expected_sha256:
        raise RuntimeError("registration SHA-256 mismatch")
    registration = _load_json(registration_path)
    if registration.get("status") != "registered_before_confirmation_model_load":
        raise ValueError("registration status is not executable")
    validate_protocol(registration["protocol"])
    if canonical_sha256(registration["protocol"]) != registration["protocol_sha256"]:
        raise RuntimeError("protocol digest mismatch")
    if _source_hashes() != registration["source_files_sha256"]:
        raise RuntimeError("a registered acquisition or scoring source changed")
    if sha256(protocol_note_path) != registration["protocol_note_sha256"]:
        raise RuntimeError("human-readable protocol note changed")
    if _model_hashes(model_snapshot) != registration["model"][
        "required_files_sha256"
    ]:
        raise RuntimeError("model snapshot bytes changed")
    return registration


def acquire(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    protocol_note_path: Path,
    model_snapshot: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Acquire the fixed panel after consuming an immutable attempt marker."""
    registration = _verified_registration(
        registration_path,
        expected_registration_sha256,
        model_snapshot,
        protocol_note_path,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    attempt_path = output_directory / "attempt.json"
    labels_path = output_directory / "labels.json"
    custody_path = output_directory / "custody.json"
    clean_failure_path = output_directory / "clean_infeasible.json"
    if any(path.exists() for path in (attempt_path, labels_path, custody_path, clean_failure_path)):
        raise FileExistsError("E33.1 attempt or output already exists")
    attempt = {
        "schema_version": "e33_1_ioi_scope_attempt_v1",
        "status": "attempt_consumed_before_model_load",
        "registration_sha256": expected_registration_sha256,
        "started_unix_time": time.time(),
    }
    _write_once_json(attempt, attempt_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.e33_ioi_scope_replication.runner import IOIPositionScopeRunner

    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    prompt_checks = validate_prompt_manifest(tokenizer, build_prompt_manifest())
    if canonical_sha256(prompt_checks) != registration[
        "prompt_tokenization_checks_sha256"
    ]:
        raise RuntimeError("tokenization checks changed after registration")
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float64,
    ).to(device="cpu", dtype=torch.float64)
    model.eval()
    runner = IOIPositionScopeRunner(model, tokenizer)

    # Every clean endpoint is acquired and checked before the first patched row.
    prepared = tuple(runner.prepare_context(record) for record in build_prompt_manifest())
    clean = tuple(runner.clean_metrics(context) for context in prepared)
    minimum_endpoint = min(
        float(metrics[direction]["endpoint"])
        for metrics in clean
        for direction in ("forward", "reverse")
    )
    clean_correct = {
        direction: sum(
            bool(metrics[direction]["recipient_pairwise_correct"])
            and bool(metrics[direction]["source_pairwise_correct"])
            for metrics in clean
        )
        for direction in ("forward", "reverse")
    }
    clean_gate = registration["protocol"]["scoring_gates"]["clean_viability"]
    clean_passed = (
        minimum_endpoint >= float(clean_gate["minimum_clean_endpoint"])
        and all(
            count
            >= int(clean_gate["minimum_pairwise_correct_contexts_each_direction"])
            for count in clean_correct.values()
        )
    )
    if not clean_passed:
        failure = {
            "schema_version": "e33_1_ioi_scope_clean_infeasible_v1",
            "status": "confirmation_clean_panel_infeasible_no_patched_calls",
            "minimum_endpoint": minimum_endpoint,
            "pairwise_correct_contexts": clean_correct,
            "clean_records": clean,
        }
        _write_once_json(failure, clean_failure_path)
        return failure

    context_records = [runner.evaluate_context(context) for context in prepared]
    labels = {
        "schema_version": "e33_1_ioi_scope_labels_v1",
        "registration_sha256": expected_registration_sha256,
        "protocol_sha256": registration["protocol_sha256"],
        "context_records": context_records,
        "query_accounting": registration["protocol"]["query_accounting"],
    }
    _write_once_json(labels, labels_path)
    custody = {
        "schema_version": "e33_1_ioi_scope_custody_v1",
        "status": "confirmation_labels_acquired_write_once",
        "registration_sha256": expected_registration_sha256,
        "attempt_sha256": sha256(attempt_path),
        "labels_sha256": sha256(labels_path),
        "model_files_sha256": registration["model"]["required_files_sha256"],
        "source_files_sha256": registration["source_files_sha256"],
        "completed_unix_time": time.time(),
    }
    _write_once_json(custody, custody_path)
    return custody


def score(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    protocol_note_path: Path,
    model_snapshot: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Score sealed labels without model calls or adaptive choices."""
    registration = _verified_registration(
        registration_path,
        expected_registration_sha256,
        model_snapshot,
        protocol_note_path,
    )
    labels_path = output_directory / "labels.json"
    custody_path = output_directory / "custody.json"
    score_path = output_directory / "score.json"
    if score_path.exists():
        raise FileExistsError(f"refusing to overwrite {score_path}")
    labels = _load_json(labels_path)
    custody = _load_json(custody_path)
    if sha256(labels_path) != custody["labels_sha256"]:
        raise RuntimeError("label custody mismatch")
    if labels["registration_sha256"] != expected_registration_sha256:
        raise RuntimeError("labels belong to a different registration")
    scored = score_context_records(labels["context_records"])
    scored.update(
        {
            "registration_sha256": expected_registration_sha256,
            "labels_sha256": custody["labels_sha256"],
            "protocol_sha256": registration["protocol_sha256"],
        }
    )
    _write_once_json(scored, score_path)
    return {"score": str(score_path), "score_sha256": sha256(score_path), **scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("register", "acquire", "score"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
        subparser.add_argument("--protocol-note", type=Path, default=DEFAULT_PROTOCOL_NOTE)
        subparser.add_argument("--model-snapshot", type=Path, required=True)
        if command != "register":
            subparser.add_argument("--registration-sha256", required=True)
            subparser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
            subparser.add_argument("--confirm-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "register":
        result = register(
            registration_path=arguments.registration,
            protocol_note_path=arguments.protocol_note,
            model_snapshot=arguments.model_snapshot,
        )
    else:
        if not arguments.confirm_run:
            raise SystemExit("refusing confirmation work without --confirm-run")
        function = acquire if arguments.command == "acquire" else score
        result = function(
            registration_path=arguments.registration,
            expected_registration_sha256=arguments.registration_sha256,
            protocol_note_path=arguments.protocol_note,
            model_snapshot=arguments.model_snapshot,
            output_directory=arguments.output,
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()


__all__ = ["acquire", "register", "score", "sha256"]
