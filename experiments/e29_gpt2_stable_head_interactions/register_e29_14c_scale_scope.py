"""Dry-run or seal the promoted E29.14c hub-off scale-scope ladder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    validate_registration_payload as validate_e29_14b_registration,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    BATCH_INVARIANCE,
    MASK_ARTIFACT_SCHEMA,
    OUTCOME_CUSTODY,
    PARENT_E29_14B_REGISTRATION_SHA256,
    PARENT_E29_14B_SCORE_SUMMARY_SHA256,
    PHYSICAL_QUERY_ACCOUNTING,
    REGISTRATION_SCHEMA,
    REGISTRATION_STATUS,
    SOURCE_FILES,
    SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256,
    SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256,
    SUPERSEDED_PROTOCOL_MASKS,
    SUPERSEDED_REGISTRATION,
    atomic_json_save,
    atomic_torch_save,
    build_clean_metadata_contract,
    build_protocol_mask_artifact,
    canonical_sha256,
    load_verified_json,
    runtime_environment,
    sha256,
    validate_clean_metadata_contract,
    validate_parent_full_pass,
    validate_registration_payload,
    verify_checkpoint_closure,
    verify_source_closure,
    verify_v1_preacquisition_state,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_protocol import (
    BOOTSTRAP,
    EXPECTED_MASK_CANONICAL_SHA256,
    HUB_COMPONENT,
    MASK_SPLITS,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    build_scale_masks,
    build_scale_scope_protocol,
    validate_scale_masks,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PARENT_REGISTRATION = (
    REPOSITORY_ROOT
    / "notes/E29.14b-COMPONENT-POSITION-COMMUTATION-REGISTRATION.json"
)
PARENT_SCORE_SUMMARY = (
    REPOSITORY_ROOT
    / "runs/e29_14b_component_position_commutation_v1/"
    "component_position_bridge_summary.json"
)
PROTOCOL_NOTE = (
    REPOSITORY_ROOT / "notes/E29.14c-HUB-OFF-SCALE-LADDER-PROTOCOL-DRAFT.md"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14c_hub_off_scale_ladder_v2"
DEFAULT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.14c-HUB-OFF-SCALE-LADDER-REGISTRATION-V2.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    return value


def build_registration_record(
    *,
    parent_registration_path: Path,
    parent: Mapping[str, Any],
    parent_score_summary_path: Path,
    parent_score_summary: Mapping[str, Any],
    protocol_note_path: Path,
    protocol_masks_path: Path,
    held_masks: Mapping[str, Any],
    clean_metadata_contract: Any,
    supersession: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the complete E29.14c seal without retaining parent outcomes."""
    masks = validate_scale_masks(held_masks)
    protocol = build_scale_scope_protocol()
    pass_contract = validate_parent_full_pass(parent_score_summary)
    prompt_records = _jsonable(parent["prompt_records"])
    prompt_checks = _jsonable(parent["prompt_checks"])
    clean_contract = validate_clean_metadata_contract(
        clean_metadata_contract, prompt_checks
    )
    if canonical_sha256(prompt_records) != parent["prompt_panel_sha256"]:
        raise RuntimeError("E29.14c inherited E29.14b prompt records changed")
    if not protocol_masks_path.is_file():
        raise FileNotFoundError(protocol_masks_path)
    if not protocol_note_path.is_file():
        raise FileNotFoundError(protocol_note_path)
    record = {
        "schema_version": REGISTRATION_SCHEMA,
        "status": REGISTRATION_STATUS,
        "question": (
            "Does the E29.14b lexical-position scope surrogate remain accurate "
            "for balanced hub-off masks from c36 through the c47 backup regime?"
        ),
        "prospective_scope": {
            "prompt_contexts": 12,
            "prompt_panel": "exact_E29.14b_panel_reused_to_isolate_scale",
            "hub_state": "component_1_absent_in_every_scale_mask",
            "scales": [36, 40, 44, 47],
            "paired_scopes": [
                "all_positions",
                "differing_lexical_positions",
            ],
            "nested_response_blind_mask_chains": True,
            "parent_numeric_responses_used_to_construct_masks_or_gates": False,
            "parent_summary_used_only_for_exact_full_pass_contract": True,
        },
        "parent_e29_14b_registration": str(parent_registration_path.resolve()),
        "parent_e29_14b_registration_sha256": (
            PARENT_E29_14B_REGISTRATION_SHA256
        ),
        "parent_e29_14b_score_summary": str(
            parent_score_summary_path.resolve()
        ),
        "parent_e29_14b_score_summary_sha256": (
            PARENT_E29_14B_SCORE_SUMMARY_SHA256
        ),
        "parent_e29_14b_pass_contract": pass_contract,
        "supersedes_e29_14c_v1_registration": str(
            SUPERSEDED_REGISTRATION.resolve()
        ),
        "supersedes_e29_14c_v1_registration_sha256": (
            SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
        ),
        "supersedes_e29_14c_v1_protocol_masks": str(
            SUPERSEDED_PROTOCOL_MASKS.resolve()
        ),
        "supersedes_e29_14c_v1_protocol_masks_sha256": (
            SUPERSEDED_E29_14C_V1_PROTOCOL_MASKS_SHA256
        ),
        "supersession": _jsonable(supersession),
        "protocol_note": str(protocol_note_path.resolve()),
        "protocol_note_sha256": sha256(protocol_note_path),
        "protocol_masks": str(protocol_masks_path.resolve()),
        "protocol_masks_sha256": sha256(protocol_masks_path),
        "protocol_masks_schema": MASK_ARTIFACT_SCHEMA,
        "model_snapshot": str(parent["model_snapshot"]),
        "model_revision": str(parent["model_revision"]),
        "checkpoint_sha256": dict(parent["checkpoint_sha256"]),
        "environment": dict(environment),
        "execution": dict(parent["execution"]),
        "component_map": _jsonable(parent["component_map"]),
        "hub_component": HUB_COMPONENT,
        "patch_semantics": {
            "direction": "forward_donor_to_base_only",
            "selected_component": (
                "recipient_update_plus_position_mask_times_"
                "donor_minus_recipient_update"
            ),
            "unselected_component": "recipient_update_unchanged",
            "position_scopes": [
                "all_positions",
                "differing_lexical_positions",
            ],
            "condition_order": (
                "singleton_all_then_lexical_then_each_scale36_mask_paired_then_"
                "each_scale40_mask_paired_then_each_scale44_mask_paired_then_"
                "the_unique_scale47_mask_paired"
            ),
            "response": "clean-endpoint-normalized_donor-minus-base_margin_change",
        },
        "prompt_records": prompt_records,
        "prompt_checks": prompt_checks,
        "prompt_panel_sha256": str(parent["prompt_panel_sha256"]),
        "clean_metadata_contract": clean_contract,
        "clean_metadata_contract_sha256": canonical_sha256(clean_contract),
        "scale_scope_protocol": protocol,
        "scale_scope_protocol_sha256": canonical_sha256(protocol),
        "mask_canonical_sha256": _jsonable(EXPECTED_MASK_CANONICAL_SHA256),
        "bootstrap": _jsonable(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        "batch_invariance": dict(BATCH_INVARIANCE),
        "numerical_gates": _jsonable(parent["numerical_gates"]),
        "outcome_custody": dict(OUTCOME_CUSTODY),
        "source_code_sha256": dict(source_hashes),
        "protocol_deviations": [],
    }
    # Make accidental non-JSON values or NaNs fail before any seal write.
    json.dumps(record, sort_keys=True, allow_nan=False)
    if tuple(masks) != MASK_SPLITS:
        raise RuntimeError("E29.14c scale-mask order changed")
    return record


def register_e29_14c(
    *, output_directory: Path, registration_path: Path, dry_run: bool
) -> dict[str, Any]:
    """Validate every dependency and optionally create two write-once seals."""
    masks_path = output_directory / "protocol_masks.pt"
    for path in (registration_path, masks_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if not PROTOCOL_NOTE.is_file():
        raise FileNotFoundError(PROTOCOL_NOTE)

    # Finish every validation that does not require serialized mask bytes before
    # creating either seal artifact. A dry run therefore leaves no files behind.
    supersession = verify_v1_preacquisition_state()
    source_hashes = verify_source_closure(SOURCE_DIRECTORY, SOURCE_FILES)
    parent = load_verified_json(
        PARENT_REGISTRATION, PARENT_E29_14B_REGISTRATION_SHA256
    )
    parent = validate_e29_14b_registration(
        parent,
        source_directory=SOURCE_DIRECTORY,
        verify_external_files=True,
    )
    parent_summary = load_verified_json(
        PARENT_SCORE_SUMMARY, PARENT_E29_14B_SCORE_SUMMARY_SHA256
    )
    validate_parent_full_pass(parent_summary)
    verify_checkpoint_closure(
        Path(parent["model_snapshot"]), parent["checkpoint_sha256"]
    )
    environment = runtime_environment()
    if environment != parent["environment"]:
        raise RuntimeError("E29.14c runtime differs from the E29.14b parent")
    masks = validate_scale_masks(build_scale_masks())
    mask_artifact = build_protocol_mask_artifact(masks)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path(parent["model_snapshot"]), local_files_only=True
    )
    clean_metadata_contract = build_clean_metadata_contract(
        parent["prompt_checks"], tokenizer
    )

    if dry_run:
        return {
            "status": "dry_run_validated_no_artifacts_written",
            "would_write_registration": str(registration_path.resolve()),
            "would_write_protocol_masks": str(masks_path.resolve()),
            "prompt_contexts": len(parent["prompt_checks"]),
            "mask_rows_by_split": dict(QUERY_ACCOUNTING["mask_rows_by_split"]),
            "condition_rows_per_context": QUERY_ACCOUNTING[
                "patched_condition_rows_per_context"
            ],
            "superseded_e29_14c_v1_registration_sha256": (
                SUPERSEDED_E29_14C_V1_REGISTRATION_SHA256
            ),
            "no_v1_attempt_or_labels_or_custody_or_summary_before_v2": True,
            "clean_metadata_contract_sha256": canonical_sha256(
                clean_metadata_contract
            ),
            "query_accounting": dict(QUERY_ACCOUNTING),
            "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        }

    # The mask artifact is written first because the JSON seal binds its actual
    # bytes. Any later failure leaves an obvious consumed partial seal and never
    # permits silent overwrite or reuse.
    atomic_torch_save(mask_artifact, masks_path)
    registration = build_registration_record(
        parent_registration_path=PARENT_REGISTRATION,
        parent=parent,
        parent_score_summary_path=PARENT_SCORE_SUMMARY,
        parent_score_summary=parent_summary,
        protocol_note_path=PROTOCOL_NOTE,
        protocol_masks_path=masks_path,
        held_masks=masks,
        clean_metadata_contract=clean_metadata_contract,
        supersession=supersession,
        source_hashes=source_hashes,
        environment=environment,
    )
    validate_registration_payload(
        registration,
        source_directory=SOURCE_DIRECTORY,
        verify_external_files=True,
    )
    atomic_json_save(registration, registration_path)
    return registration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    arguments = parser.parse_args()
    if not arguments.dry_run and not arguments.confirm_reviewed:
        raise SystemExit("refusing to seal without --confirm-reviewed")
    result = register_e29_14c(
        output_directory=arguments.output,
        registration_path=arguments.registration,
        dry_run=arguments.dry_run,
    )
    if not arguments.dry_run:
        result = {
            "status": result["status"],
            "registration": str(arguments.registration.resolve()),
            "registration_sha256": sha256(arguments.registration),
            "protocol_masks": str((arguments.output / "protocol_masks.pt").resolve()),
            "protocol_masks_sha256": sha256(arguments.output / "protocol_masks.pt"),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_REGISTRATION",
    "PARENT_REGISTRATION",
    "PARENT_SCORE_SUMMARY",
    "PROTOCOL_NOTE",
    "SUPERSEDED_PROTOCOL_MASKS",
    "SUPERSEDED_REGISTRATION",
    "build_registration_record",
    "register_e29_14c",
]
