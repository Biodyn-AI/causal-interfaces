"""Dry-run or seal the response-blind E29.14b commutation bridge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    BATCH_INVARIANCE,
    OUTCOME_CUSTODY,
    PARENT_E29_14A_REGISTRATION_SHA256,
    PHYSICAL_QUERY_ACCOUNTING,
    PRIOR_PROTOCOL_MASKS_SHA256,
    SOURCE_FILES,
    atomic_json_save,
    canonical_sha256,
    load_verified_json,
    runtime_environment,
    sha256,
    validate_registration_payload,
    verify_checkpoint_closure,
    verify_file_hash,
    verify_source_closure,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    BOOTSTRAP,
    EXPECTED_MASK_CANONICAL_SHA256,
    HUB_COMPONENT,
    MASK_SPLITS,
    QUERY_ACCOUNTING,
    SCORING_GATES,
    build_component_position_protocol,
    validate_bridge_masks,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_prompt_panel import (
    e29_14b_prompt_panel,
    validate_e29_14b_prompt_panel,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PARENT_REGISTRATION = (
    REPOSITORY_ROOT
    / "notes/E29.14a-POSITIONAL-LEXICAL-INTERCHANGE-REGISTRATION.json"
)
PRIOR_PROTOCOL_MASKS = (
    REPOSITORY_ROOT
    / "runs/e29_13_absorbing_hub_confirmation_v1/protocol_masks.pt"
)
PROTOCOL_NOTE = (
    REPOSITORY_ROOT
    / "notes/E29.14b-COMPONENT-POSITION-COMMUTATION-PROTOCOL-DRAFT.md"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e29_14b_component_position_commutation_v1"
DEFAULT_REGISTRATION = (
    REPOSITORY_ROOT
    / "notes/E29.14b-COMPONENT-POSITION-COMMUTATION-REGISTRATION.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _prompt_records(
    prompts: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    keys = ("template", "base_prompt", "donor_prompt", "base_target", "donor_target")
    records = [{key: str(prompt[key]) for key in keys} for prompt in prompts]
    if len(records) != 12:
        raise RuntimeError("E29.14b requires twelve prompt contexts")
    return records


def load_prior_bridge_masks(path: Path = PRIOR_PROTOCOL_MASKS) -> dict[str, Any]:
    """Load only the exact response-blind E29.13 mask artifact."""
    verify_file_hash(path, PRIOR_PROTOCOL_MASKS_SHA256)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping):
        raise TypeError("E29.13 protocol-mask artifact must be a mapping")
    if artifact.get("schema_version") != "e29_13_absorbing_hub_protocol_artifact_v1":
        raise RuntimeError("E29.13 protocol-mask artifact schema changed")
    protocol = artifact.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("E29.13 response-blind protocol payload is absent")
    held = protocol.get("held_masks")
    if not isinstance(held, Mapping) or set(held) != set(MASK_SPLITS):
        raise RuntimeError("E29.13 held-mask split contract changed")
    masks = {split_name: held[split_name].detach().cpu() for split_name in MASK_SPLITS}
    validated = validate_bridge_masks(masks)
    if validated is not None:
        masks = dict(validated)
    return masks


def build_registration_record(
    *,
    parent_registration_path: Path,
    parent: Mapping[str, Any],
    prior_masks_path: Path,
    held_masks: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    prompt_checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the complete response-blind E29.14b registration."""
    records = _prompt_records(prompts)
    protocol = build_component_position_protocol(held_masks)
    return {
        "schema_version": "e29_14b_component_position_registration_v1",
        "status": "prospective_sealed_before_any_e29_14b_model_response",
        "question": (
            "Does differing-lexical-position patching preserve the paired causal "
            "response of the same c24/c32 component masks?"
        ),
        "prospective_scope": (
            "twelve fresh lexical contexts; exact sealed E29.13 c24/c32 held "
            "masks; paired all-position and lexical-position acquisition only"
        ),
        "parent_e29_14a_registration": str(parent_registration_path.resolve()),
        "parent_e29_14a_registration_sha256": (
            PARENT_E29_14A_REGISTRATION_SHA256
        ),
        "protocol_note": str(PROTOCOL_NOTE.resolve()),
        "protocol_note_sha256": sha256(PROTOCOL_NOTE),
        "prior_protocol_masks": str(prior_masks_path.resolve()),
        "prior_protocol_masks_sha256": PRIOR_PROTOCOL_MASKS_SHA256,
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
            "position_scopes": ["all_positions", "differing_lexical_positions"],
            "condition_order": (
                "singleton_all_then_lexical_then_each_c24_mask_all_then_lexical_"
                "then_each_c32_mask_all_then_lexical"
            ),
            "response": "clean-endpoint-normalized_donor-minus-base_margin_change",
        },
        "prompt_records": records,
        "prompt_checks": _jsonable(prompt_checks),
        "prompt_panel_sha256": canonical_sha256(records),
        "component_position_protocol": protocol,
        "component_position_protocol_sha256": canonical_sha256(protocol),
        "prior_mask_canonical_sha256": _jsonable(EXPECTED_MASK_CANONICAL_SHA256),
        "bootstrap": _jsonable(BOOTSTRAP),
        "scoring_gates": _jsonable(SCORING_GATES),
        "query_accounting": _jsonable(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "batch_invariance": dict(BATCH_INVARIANCE),
        "numerical_gates": _jsonable(parent["numerical_gates"]),
        "outcome_custody": dict(OUTCOME_CUSTODY),
        "source_code_sha256": dict(source_hashes),
        "protocol_deviations": [],
    }


def register_e29_14b(
    *, registration_path: Path, dry_run: bool
) -> dict[str, Any]:
    """Validate all response-blind dependencies and optionally seal once."""
    from transformers import AutoTokenizer

    if registration_path.exists():
        raise FileExistsError(f"refusing to overwrite {registration_path}")
    if not PROTOCOL_NOTE.is_file():
        raise FileNotFoundError(PROTOCOL_NOTE)
    parent = load_verified_json(
        PARENT_REGISTRATION, PARENT_E29_14A_REGISTRATION_SHA256
    )
    if (
        parent.get("schema_version") != "e29_14a_position_registration_v1"
        or parent.get("status")
        != "prospective_sealed_before_any_e29_14a_model_response"
    ):
        raise RuntimeError("sealed E29.14a parent registration changed")
    verify_checkpoint_closure(
        Path(parent["model_snapshot"]), parent["checkpoint_sha256"]
    )
    held_masks = load_prior_bridge_masks(PRIOR_PROTOCOL_MASKS)
    source_hashes = verify_source_closure(SOURCE_DIRECTORY, SOURCE_FILES)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(parent["model_snapshot"]), local_files_only=True
    )
    prompts = e29_14b_prompt_panel()
    checks = validate_e29_14b_prompt_panel(tokenizer, prompts)
    registration = build_registration_record(
        parent_registration_path=PARENT_REGISTRATION,
        parent=parent,
        prior_masks_path=PRIOR_PROTOCOL_MASKS,
        held_masks=held_masks,
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
            "condition_rows_per_context": QUERY_ACCOUNTING[
                "patched_condition_rows_per_context"
            ],
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
    result = register_e29_14b(
        registration_path=arguments.registration, dry_run=arguments.dry_run
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
    "load_prior_bridge_masks",
    "register_e29_14b",
]
