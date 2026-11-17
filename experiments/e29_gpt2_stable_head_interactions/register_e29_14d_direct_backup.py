"""Dry-run or seal the prospective E29.14d direct causal-carrier panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
    validate_registration_payload as validate_e29_14c_registration,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_causal_carrier_protocol import (
    BOOTSTRAP,
    CANDIDATE_GROUPS,
    HUB_COMPONENT,
    PHYSICAL_QUERY_ACCOUNTING,
    POSITION_SCOPE_ORDER,
    PRIMARY_POSITION_SCOPE,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    build_causal_carrier_protocol,
    build_condition_manifest,
    build_condition_masks,
    build_query_manifest,
    validate_causal_carrier_protocol,
    validate_condition_masks,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_direct_backup_custody import (
    BATCH_INVARIANCE,
    EXPECTED_PROMPT_PANEL_SHA256,
    EXPECTED_PROMPT_RECORDS_SHA256,
    OUTCOME_CUSTODY,
    PARENT_E29_14C_REGISTRATION_SHA256,
    PARENT_E29_14C_SCORE_SUMMARY_SHA256,
    PARENT_REGISTRATION,
    PARENT_SCORE_SUMMARY,
    PROTOCOL_ARTIFACT_SCHEMA,
    REGISTRATION_SCHEMA,
    REGISTRATION_STATUS,
    SOURCE_FILES,
    atomic_json_save,
    atomic_torch_save,
    audit_prior_mask_nonoverlap,
    build_clean_metadata_contract,
    build_protocol_artifact,
    canonical_sha256,
    load_verified_json,
    runtime_environment,
    sha256,
    validate_clean_metadata_contract,
    validate_parent_full_pass,
    validate_posthoc_candidate_provenance,
    validate_registration_payload,
    verify_checkpoint_closure,
    verify_source_closure,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14d_prompt_panel import (
    e29_14d_prompt_panel,
    validate_e29_14d_prompt_panel,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14d_direct_backup_v1"
DEFAULT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.14d-DIRECT-CAUSAL-CARRIER-REGISTRATION.json"
)

_PROMPT_RECORD_KEYS = (
    "template",
    "base_prompt",
    "donor_prompt",
    "base_target",
    "donor_target",
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
    protocol_artifact_path: Path,
    prompt_records: Any,
    prompt_checks: Any,
    clean_metadata_contract: Any,
    source_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the full prospective seal without copying parent outcomes."""
    if not protocol_artifact_path.is_file():
        raise FileNotFoundError(protocol_artifact_path)
    protocol = validate_causal_carrier_protocol(build_causal_carrier_protocol())
    masks = validate_condition_masks(build_condition_masks())
    if masks is None:
        masks = build_condition_masks()
    conditions = _jsonable(build_condition_manifest())
    queries = _jsonable(build_query_manifest())
    records = _jsonable(prompt_records)
    checks = _jsonable(prompt_checks)
    clean_contract = validate_clean_metadata_contract(clean_metadata_contract, checks)
    if canonical_sha256(records) != EXPECTED_PROMPT_RECORDS_SHA256:
        raise RuntimeError("E29.14d prompt records differ from their reviewed hash")
    pass_contract = validate_parent_full_pass(parent_score_summary)
    posthoc = validate_posthoc_candidate_provenance()
    prior_mask_audit = audit_prior_mask_nonoverlap(masks)
    record = {
        "schema_version": REGISTRATION_SCHEMA,
        "status": REGISTRATION_STATUS,
        "question": (
            "Which non-hub sublayers causally carry the high-cardinality donor "
            "transfer, and is the late-attention concentration robust across "
            "balanced backgrounds, direct dropouts, and same-layer MLP controls?"
        ),
        "prospective_scope": {
            "prompt_contexts": len(records),
            "prompt_panel": "fresh_E29.14d_response_blind_lexical_panel",
            "candidate_groups": _jsonable(CANDIDATE_GROUPS),
            "hub_component": HUB_COMPONENT,
            "all_conditions_use_both_position_scopes": True,
            "primary_position_scope": PRIMARY_POSITION_SCOPE,
            "position_scope_order": list(POSITION_SCOPE_ORDER),
            "class_balanced_backgrounds": True,
            "parent_posthoc_responses_selected_top4_factorial_components": True,
            "parent_posthoc_responses_motivated_attention_vs_mlp_primary_hypothesis": (
                True
            ),
            "parent_posthoc_selection_artifact_bound_exactly": True,
            "parent_responses_used_to_construct_class_balanced_scan_backgrounds": False,
            "parent_responses_used_to_set_numeric_gates": False,
            "e29_14d_responses_used_to_construct_any_condition_or_gate": False,
            "parent_main_summary_used_only_for_exact_complete_ladder_pass": True,
            "clean_update_norms_are_response_blind_diagnostics_only": True,
            "no_norm_adaptive_conditions_or_dosing": True,
        },
        "parent_e29_14c_registration": str(parent_registration_path.resolve()),
        "parent_e29_14c_registration_sha256": (PARENT_E29_14C_REGISTRATION_SHA256),
        "parent_e29_14c_score_summary": str(parent_score_summary_path.resolve()),
        "parent_e29_14c_score_summary_sha256": (PARENT_E29_14C_SCORE_SUMMARY_SHA256),
        "parent_e29_14c_pass_contract": pass_contract,
        "posthoc_candidate_provenance": posthoc,
        "posthoc_candidate_provenance_sha256": canonical_sha256(posthoc),
        "prior_mask_audit": prior_mask_audit,
        "prior_mask_audit_sha256": canonical_sha256(prior_mask_audit),
        "protocol_artifact": str(protocol_artifact_path.resolve()),
        "protocol_artifact_sha256": sha256(protocol_artifact_path),
        "protocol_artifact_schema": PROTOCOL_ARTIFACT_SCHEMA,
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
                "recipient_update_plus_position_mask_times_donor_minus_recipient_update"
            ),
            "unselected_component": "recipient_update_unchanged",
            "position_scopes": list(POSITION_SCOPE_ORDER),
            "condition_order": "exact_condition_manifest_then_both_scopes",
            "query_order": "exact_query_manifest",
            "response": "donor-minus-base_logit_margin_change",
            "normalization_deferred_to_scorer": "divide_by_positive_clean_endpoint",
        },
        "prompt_records": records,
        "prompt_records_sha256": EXPECTED_PROMPT_RECORDS_SHA256,
        "prompt_checks": checks,
        "prompt_panel_sha256": EXPECTED_PROMPT_PANEL_SHA256,
        "clean_metadata_contract": clean_contract,
        "clean_metadata_contract_sha256": canonical_sha256(clean_contract),
        "causal_carrier_protocol": _jsonable(protocol),
        "causal_carrier_protocol_sha256": canonical_sha256(protocol),
        "condition_masks_canonical_sha256": canonical_sha256(masks),
        "condition_manifest": conditions,
        "condition_manifest_sha256": canonical_sha256(conditions),
        "query_manifest": queries,
        "query_manifest_sha256": canonical_sha256(queries),
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
    json.dumps(record, sort_keys=True, allow_nan=False)
    return record


def register_e29_14d(
    *, output_directory: Path, registration_path: Path, dry_run: bool
) -> dict[str, Any]:
    """Validate every dependency and optionally create two write-once seals."""
    artifact_path = output_directory / "protocol_conditions.pt"
    for path in (registration_path, artifact_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    source_hashes = verify_source_closure(SOURCE_DIRECTORY, SOURCE_FILES)
    parent = load_verified_json(PARENT_REGISTRATION, PARENT_E29_14C_REGISTRATION_SHA256)
    parent = validate_e29_14c_registration(
        parent, source_directory=SOURCE_DIRECTORY, verify_external_files=True
    )
    parent_summary = load_verified_json(
        PARENT_SCORE_SUMMARY, PARENT_E29_14C_SCORE_SUMMARY_SHA256
    )
    validate_parent_full_pass(parent_summary)
    verify_checkpoint_closure(
        Path(parent["model_snapshot"]), parent["checkpoint_sha256"]
    )
    environment = runtime_environment()
    if environment != parent["environment"]:
        raise RuntimeError("E29.14d runtime differs from the E29.14c parent")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path(parent["model_snapshot"]), local_files_only=True
    )
    panel = e29_14d_prompt_panel()
    if canonical_sha256(panel) != EXPECTED_PROMPT_PANEL_SHA256:
        raise RuntimeError("E29.14d fresh prompt-panel source hash changed")
    prompt_checks = validate_e29_14d_prompt_panel(tokenizer, panel)
    prompt_records = [
        {key: str(check[key]) for key in _PROMPT_RECORD_KEYS} for check in prompt_checks
    ]
    if canonical_sha256(prompt_records) != EXPECTED_PROMPT_RECORDS_SHA256:
        raise RuntimeError("E29.14d prompt-record hash changed")
    clean_metadata_contract = build_clean_metadata_contract(prompt_checks, tokenizer)
    masks = build_condition_masks()
    protocol_artifact = build_protocol_artifact(masks)

    if dry_run:
        return {
            "status": "dry_run_validated_no_artifacts_written",
            "would_write_registration": str(registration_path.resolve()),
            "would_write_protocol_artifact": str(artifact_path.resolve()),
            "prompt_contexts": len(prompt_records),
            "condition_masks": len(masks),
            "patched_condition_rows_per_context": QUERY_ACCOUNTING[
                "patched_condition_rows_per_context"
            ],
            "prompt_panel_sha256": EXPECTED_PROMPT_PANEL_SHA256,
            "prompt_records_sha256": EXPECTED_PROMPT_RECORDS_SHA256,
            "posthoc_candidate_provenance_sha256": canonical_sha256(
                validate_posthoc_candidate_provenance()
            ),
            "prior_mask_audit_sha256": canonical_sha256(
                audit_prior_mask_nonoverlap(masks)
            ),
            "clean_metadata_contract_sha256": canonical_sha256(clean_metadata_contract),
            "query_accounting": _jsonable(QUERY_ACCOUNTING),
            "physical_query_accounting": _jsonable(PHYSICAL_QUERY_ACCOUNTING),
        }

    atomic_torch_save(protocol_artifact, artifact_path)
    registration = build_registration_record(
        parent_registration_path=PARENT_REGISTRATION,
        parent=parent,
        parent_score_summary_path=PARENT_SCORE_SUMMARY,
        parent_score_summary=parent_summary,
        protocol_artifact_path=artifact_path,
        prompt_records=prompt_records,
        prompt_checks=prompt_checks,
        clean_metadata_contract=clean_metadata_contract,
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
    result = register_e29_14d(
        output_directory=arguments.output,
        registration_path=arguments.registration,
        dry_run=arguments.dry_run,
    )
    if not arguments.dry_run:
        result = {
            "status": result["status"],
            "registration": str(arguments.registration.resolve()),
            "registration_sha256": sha256(arguments.registration),
            "protocol_artifact": str(
                (arguments.output / "protocol_conditions.pt").resolve()
            ),
            "protocol_artifact_sha256": sha256(
                arguments.output / "protocol_conditions.pt"
            ),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_REGISTRATION",
    "PARENT_REGISTRATION",
    "PARENT_SCORE_SUMMARY",
    "build_registration_record",
    "register_e29_14d",
]
