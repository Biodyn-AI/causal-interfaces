"""Dry-run or seal the prospective E29.14a positional confirmation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    CHECKPOINT_FILES,
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_13_custody import (
    load_verified_registration as load_parent_registration,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_custody import (
    BATCH_INVARIANCE,
    DEVELOPMENT_POSITION_POSTHOC_SHA256,
    OUTCOME_CUSTODY,
    PARENT_E29_13_REGISTRATION_SHA256,
    PHYSICAL_QUERY_ACCOUNTING,
    SOURCE_FILES,
    atomic_json_save,
    canonical_sha256,
    runtime_environment,
    sha256,
    validate_registration_payload,
    verify_checkpoint_closure,
    verify_file_hash,
    verify_source_closure,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_protocol import (
    BOOTSTRAP,
    CONTROL_COMPONENT,
    FORWARD_CONDITIONS,
    HUB_COMPONENT,
    QUERY_ACCOUNTING,
    REVERSE_CONDITIONS,
    SCORING_GATES,
    build_position_protocol,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14a_prompt_panel import (
    e29_14a_prompt_panel,
    validate_e29_14a_prompt_panel,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PARENT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.13-ABSORBING-HUB-CONFIRMATION-REGISTRATION.json"
)
DEVELOPMENT_POSTHOC = (
    REPOSITORY_ROOT
    / "runs/e29_13_absorbing_hub_confirmation_v1/e29_13_position_semantics_posthoc.json"
)
PROTOCOL_NOTE = (
    REPOSITORY_ROOT
    / "notes/E29.14a-POSITIONAL-LEXICAL-INTERCHANGE-PROTOCOL-DRAFT.md"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14a_position_lexical_interchange_v1"
DEFAULT_REGISTRATION = (
    REPOSITORY_ROOT
    / "notes/E29.14a-POSITIONAL-LEXICAL-INTERCHANGE-REGISTRATION.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _prompt_records(
    prompts: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    keys = ("template", "base_prompt", "donor_prompt", "base_target", "donor_target")
    records = [{key: str(prompt[key]) for key in keys} for prompt in prompts]
    if len(records) != 12:
        raise RuntimeError("E29.14a requires twelve prompt contexts")
    return records


def build_registration_record(
    *,
    parent_registration_path: Path,
    parent: Mapping[str, Any],
    development_path: Path,
    development: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    prompt_checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the complete response-blind registration payload."""
    records = _prompt_records(prompts)
    protocol = build_position_protocol()
    return {
        "schema_version": "e29_14a_position_registration_v1",
        "status": "prospective_sealed_before_any_e29_14a_model_response",
        "question": (
            "Does GPT-2-medium's layer-0 MLP act as a component-specific, "
            "bidirectional lexical-state switch localized to differing token positions?"
        ),
        "prospective_scope": (
            "fresh lexical contexts; fixed forward-12 and reverse-4 condition "
            "manifests; no adaptive conditions or response-dependent fitting"
        ),
        "parent_e29_13_registration": str(parent_registration_path.resolve()),
        "parent_e29_13_registration_sha256": PARENT_E29_13_REGISTRATION_SHA256,
        "development_evidence": {
            "position_semantics_posthoc": str(development_path.resolve()),
            "position_semantics_posthoc_sha256": (
                DEVELOPMENT_POSITION_POSTHOC_SHA256
            ),
            "schema_version": str(development["schema_version"]),
            "status": str(development["status"]),
            "confirmatory_status": str(development["confirmatory_status"]),
        },
        "protocol_note": str(PROTOCOL_NOTE.resolve()),
        "protocol_note_sha256": sha256(PROTOCOL_NOTE),
        "model_snapshot": str(parent["model_snapshot"]),
        "model_revision": str(parent["model_revision"]),
        "checkpoint_sha256": dict(parent["checkpoint_sha256"]),
        "environment": dict(environment),
        "execution": dict(parent["execution"]),
        "component_map": _jsonable(parent["component_map"]),
        "hub_component": HUB_COMPONENT,
        "control_component": CONTROL_COMPONENT,
        "patch_semantics": {
            "forward": "base_recipient_plus_donor_minus_base_component_update",
            "reverse": "donor_recipient_plus_base_minus_donor_component_update",
            "position_scope": "registered_aligned_token_position_boolean_mask",
            "dose": "recipient_update_plus_alpha_times_source_minus_recipient_update",
            "response": "recipient_wrong_minus_correct_logit_improvement",
        },
        "prompt_records": records,
        "prompt_checks": _jsonable(prompt_checks),
        "prompt_panel_sha256": canonical_sha256(records),
        "position_protocol": protocol,
        "position_protocol_sha256": canonical_sha256(protocol),
        "condition_manifest": {
            "forward": _jsonable(FORWARD_CONDITIONS),
            "reverse": _jsonable(REVERSE_CONDITIONS),
        },
        "bootstrap": dict(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "batch_invariance": dict(BATCH_INVARIANCE),
        "numerical_gates": dict(NUMERICAL_GATES),
        "outcome_custody": dict(OUTCOME_CUSTODY),
        "source_code_sha256": dict(source_hashes),
        "protocol_deviations": [],
    }


def _load_development_evidence() -> dict[str, Any]:
    verify_file_hash(DEVELOPMENT_POSTHOC, DEVELOPMENT_POSITION_POSTHOC_SHA256)
    value = json.loads(DEVELOPMENT_POSTHOC.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("E29.14a development evidence must be a JSON object")
    expected = {
        "schema_version": "e29_13_position_semantics_posthoc_v1",
        "status": "posthoc_development_only_new_queries_on_opened_e29_13_prompts",
        "confirmatory_status": (
            "not_confirmatory_and_cannot_change_any_sealed_e29_13_verdict"
        ),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"E29.14a development evidence {key} changed")
    return value


def register_e29_14a(
    *,
    registration_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Validate all dependencies and optionally create the write-once seal."""
    from transformers import AutoTokenizer

    if registration_path.exists():
        raise FileExistsError(f"refusing to overwrite {registration_path}")
    if not PROTOCOL_NOTE.is_file():
        raise FileNotFoundError(PROTOCOL_NOTE)
    parent = load_parent_registration(
        PARENT_REGISTRATION,
        expected_sha256=PARENT_E29_13_REGISTRATION_SHA256,
    )
    verify_checkpoint_closure(
        Path(parent["model_snapshot"]),
        parent["checkpoint_sha256"],
        CHECKPOINT_FILES,
    )
    development = _load_development_evidence()
    source_hashes = verify_source_closure(SOURCE_DIRECTORY, SOURCE_FILES)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(parent["model_snapshot"]), local_files_only=True
    )
    prompts = e29_14a_prompt_panel()
    checks = validate_e29_14a_prompt_panel(tokenizer, prompts)
    registration = build_registration_record(
        parent_registration_path=PARENT_REGISTRATION,
        parent=parent,
        development_path=DEVELOPMENT_POSTHOC,
        development=development,
        prompts=prompts,
        prompt_checks=checks,
        source_hashes=source_hashes,
        environment=runtime_environment(),
    )
    validate_registration_payload(
        registration,
        source_directory=SOURCE_DIRECTORY,
        verify_external_files=True,
    )
    if dry_run:
        return {
            "status": "dry_run_validated_no_artifact_written",
            "would_write_registration": str(registration_path.resolve()),
            "prompt_contexts": len(checks),
            "forward_conditions": len(FORWARD_CONDITIONS),
            "reverse_conditions": len(REVERSE_CONDITIONS),
            "query_accounting": dict(QUERY_ACCOUNTING),
            "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        }
    atomic_json_save(registration, registration_path)
    return registration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    arguments = parser.parse_args()
    if not arguments.dry_run and not arguments.confirm_reviewed:
        raise SystemExit("refusing to seal without --confirm-reviewed")
    result = register_e29_14a(
        registration_path=arguments.registration,
        dry_run=arguments.dry_run,
    )
    if not arguments.dry_run:
        result = {
            "status": result["status"],
            "registration": str(arguments.registration.resolve()),
            "registration_sha256": sha256(arguments.registration),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_REGISTRATION",
    "build_registration_record",
    "register_e29_14a",
]
