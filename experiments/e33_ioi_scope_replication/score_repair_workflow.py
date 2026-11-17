"""Register and execute the minimal E33.1 JSON-order scoring repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from experiments.e33_ioi_scope_replication.scoring_json_repair import (
    score_json_context_records,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PARENT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E33.1-IOI-SCOPE-REGISTRATION-V2.json"
)
DEFAULT_ADDENDUM = REPOSITORY_ROOT / "notes/E33.1-SCORING-REPAIR-REGISTRATION.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e33_1_ioi_scope_replication_v1"
REPAIR_SOURCES = ("protocol.py", "scoring.py", "scoring_json_repair.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _write_once_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _source_hashes() -> dict[str, str]:
    return {name: sha256(SOURCE_DIRECTORY / name) for name in REPAIR_SOURCES}


def register_repair(
    *,
    parent_registration: Path,
    expected_parent_sha256: str,
    output_directory: Path,
    addendum_path: Path,
) -> dict[str, Any]:
    if addendum_path.exists():
        raise FileExistsError(f"refusing to overwrite {addendum_path}")
    if sha256(parent_registration) != expected_parent_sha256:
        raise RuntimeError("parent registration SHA-256 mismatch")
    labels_path = output_directory / "labels.json"
    custody_path = output_directory / "custody.json"
    custody = _load_json(custody_path)
    labels_sha = sha256(labels_path)
    if labels_sha != custody["labels_sha256"]:
        raise RuntimeError("sealed label custody mismatch")
    addendum = {
        "schema_version": "e33_1_scoring_repair_registration_v1",
        "status": "registered_before_successful_scientific_scoring",
        "registered_unix_time": time.time(),
        "parent_registration_sha256": expected_parent_sha256,
        "sealed_labels_sha256": labels_sha,
        "repair_sources_sha256": _source_hashes(),
        "repair_scope": (
            "Restore CONDITION_IDS dictionary order after sort_keys JSON "
            "serialization, then delegate all calculations and gates to the "
            "unchanged parent-registered scorer."
        ),
        "scientific_conditions_changed": False,
        "scientific_thresholds_changed": False,
        "scientific_calculations_changed": False,
        "prior_scoring_attempt_reached_scientific_values": False,
    }
    _write_once_json(addendum, addendum_path)
    return {"addendum": str(addendum_path), "addendum_sha256": sha256(addendum_path)}


def execute_repair(
    *,
    addendum_path: Path,
    expected_addendum_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    if sha256(addendum_path) != expected_addendum_sha256:
        raise RuntimeError("scoring-repair registration SHA-256 mismatch")
    addendum = _load_json(addendum_path)
    if _source_hashes() != addendum["repair_sources_sha256"]:
        raise RuntimeError("a scoring-repair source changed")
    labels_path = output_directory / "labels.json"
    score_path = output_directory / "score.json"
    if score_path.exists():
        raise FileExistsError(f"refusing to overwrite {score_path}")
    if sha256(labels_path) != addendum["sealed_labels_sha256"]:
        raise RuntimeError("sealed labels changed after repair registration")
    labels = _load_json(labels_path)
    score = score_json_context_records(labels["context_records"])
    score.update(
        {
            "parent_registration_sha256": addendum[
                "parent_registration_sha256"
            ],
            "scoring_repair_registration_sha256": expected_addendum_sha256,
            "labels_sha256": addendum["sealed_labels_sha256"],
        }
    )
    _write_once_json(score, score_path)
    return {"score": str(score_path), "score_sha256": sha256(score_path), **score}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument(
        "--parent-registration", type=Path, default=DEFAULT_PARENT_REGISTRATION
    )
    register_parser.add_argument("--parent-registration-sha256", required=True)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    register_parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    score_parser.add_argument("--addendum-sha256", required=True)
    score_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    score_parser.add_argument("--confirm-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "register":
        result = register_repair(
            parent_registration=arguments.parent_registration,
            expected_parent_sha256=arguments.parent_registration_sha256,
            output_directory=arguments.output,
            addendum_path=arguments.addendum,
        )
    else:
        if not arguments.confirm_run:
            raise SystemExit("refusing scoring without --confirm-run")
        result = execute_repair(
            addendum_path=arguments.addendum,
            expected_addendum_sha256=arguments.addendum_sha256,
            output_directory=arguments.output,
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

