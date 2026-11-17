"""Acquire the sealed E29.14b component-by-position panel exactly once."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    BATCH_SIZE,
    NUMERICAL_GATES,
    tensor_numerical_summary,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_custody import (
    BATCH_INVARIANCE,
    PHYSICAL_QUERY_ACCOUNTING,
    REGISTRATION_REQUIRED_KEYS,
    atomic_json_save,
    atomic_torch_save,
    build_acquisition_attempt_record,
    create_acquisition_attempt_marker,
    load_verified_registration,
    require_exact_keys,
    runtime_environment,
    sha256,
    validate_registration_payload,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_protocol import (
    COMPONENT_COUNT,
    CONTEXT_COUNT,
    HUB_COMPONENT,
    MASK_SHAPES,
    MASK_SPLITS,
    POSITION_SCOPE_ORDER,
    QUERY_ACCOUNTING,
    validate_bridge_masks,
)
from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_runner import (
    Float64ComponentPositionRunner,
)
from experiments.e29_gpt2_stable_head_interactions.register_e29_14b_component_position import (
    DEFAULT_OUTPUT,
    DEFAULT_REGISTRATION,
    load_prior_bridge_masks,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = Path(__file__).resolve().parent

PROMPT_KEYS = (
    "template",
    "base_prompt",
    "donor_prompt",
    "base_target",
    "donor_target",
)
CLEAN_METADATA_KEYS = frozenset(
    {
        *PROMPT_KEYS,
        "base_target_token_id",
        "donor_target_token_id",
        "token_length",
        "model_layer_count",
        "component_count",
        "component_semantics",
        "patch_direction",
        "position_scope_order",
        "position_scope_indices",
        "clean_base_correct_minus_donor_margin",
        "clean_donor_correct_minus_base_margin",
        "clean_endpoint",
        "clean_base_logit_absolute_max",
        "clean_donor_logit_absolute_max",
        "clean_logit_absolute_max",
        "clean_logits_finite",
        "device",
        "model_dtype",
    }
)
LABEL_KEYS = frozenset(
    {
        "schema_version",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "condition_manifest",
        "held_masks",
        "singleton_all_response",
        "singleton_lexical_response",
        "all_position_responses",
        "lexical_position_responses",
        "responses_in_registered_order",
        "clean_metadata",
        "clean_base_correct_minus_donor_margins",
        "clean_donor_correct_minus_base_margins",
        "clean_endpoints",
        "batch_invariance_smoke",
        "query_accounting",
        "physical_query_accounting",
        "numerical_summary",
        "all_integrity_gates_passed",
    }
)
CUSTODY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "registration",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "source_code_sha256_verified_before_acquisition",
        "checkpoint_sha256_verified_before_acquisition",
        "prior_protocol_masks_sha256_verified_before_acquisition",
        "acquisition_attempt",
        "acquisition_attempt_sha256",
        "component_position_labels",
        "component_position_labels_sha256",
        "prompt_panel_sha256",
        "component_position_protocol_sha256",
        "contexts",
        "patched_condition_rows_per_context",
        "patched_context_condition_cells",
        "clean_forward_examples",
        "batch_smoke_unique_condition_rows",
        "batch_smoke_physical_condition_cells",
        "maximum_logical_queries",
        "maximum_physical_model_examples",
        "batch_size",
        "batch_invariance_passed",
        "all_clean_and_numerical_gates_passed",
        "device",
        "dtype",
        "response_absolute_max",
        "runtime_seconds",
    }
)


def prompt_records(
    prompt_checks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Extract the exact five-field runtime prompt records."""
    records = tuple(
        {key: str(check[key]) for key in PROMPT_KEYS} for check in prompt_checks
    )
    if len(records) != CONTEXT_COUNT:
        raise RuntimeError("E29.14b requires twelve prompt contexts")
    return records


def build_condition_manifest(
    masks: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Return the exact singleton, then c24/c32 paired condition order."""
    panels = validate_bridge_masks(masks)
    singleton = torch.zeros(COMPONENT_COUNT, dtype=torch.float64)
    singleton[HUB_COMPONENT] = 1.0
    manifest = [
        {
            "kind": "singleton",
            "split": "singleton_hub",
            "mask_row": 0,
            "scope": scope,
            "selected_components": [HUB_COMPONENT],
        }
        for scope in POSITION_SCOPE_ORDER
    ]
    for split in MASK_SPLITS:
        for row_index, row in enumerate(panels[split]):
            selected = [int(index) for index in torch.where(row == 1.0)[0]]
            for scope in POSITION_SCOPE_ORDER:
                manifest.append(
                    {
                        "kind": "multi_component",
                        "split": split,
                        "mask_row": row_index,
                        "scope": scope,
                        "selected_components": selected,
                    }
                )
    if len(manifest) != QUERY_ACCOUNTING["patched_condition_rows_per_context"]:
        raise RuntimeError("E29.14b condition-manifest count changed")
    return manifest


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be finite")
    return number


def _verify_clean_contexts(
    prompt_checks: Sequence[Mapping[str, Any]], prepared: Sequence[Any]
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(prepared) != CONTEXT_COUNT:
        raise RuntimeError("E29.14b prepared-context count changed")
    metadata = []
    base_margins = []
    donor_margins = []
    endpoints = []
    logit_limit = float(NUMERICAL_GATES["clean_logit_absolute_max"])
    for registered, context in zip(prompt_checks, prepared, strict=True):
        observed = dict(context.clean_metadata)
        require_exact_keys(
            observed, CLEAN_METADATA_KEYS, label="E29.14b clean runtime metadata"
        )
        if any(str(observed[key]) != str(registered[key]) for key in PROMPT_KEYS):
            raise RuntimeError("runtime prompt metadata differs from registration")
        token_length = int(registered["token_length"])
        if observed["token_length"] != token_length:
            raise RuntimeError("runtime prompt token length differs from registration")
        for side in ("base", "donor"):
            expected_token = int(registered["sides"][side]["target_token_id"])
            if observed[f"{side}_target_token_id"] != expected_token:
                raise RuntimeError("runtime target token differs from registration")
        expected_scopes = {
            "all_positions": list(range(token_length)),
            "differing_lexical_positions": [
                int(index)
                for index in torch.where(
                    context.base_input_ids[0] != context.donor_input_ids[0]
                )[0]
            ],
        }
        if (
            observed["position_scope_order"] != list(POSITION_SCOPE_ORDER)
            or observed["position_scope_indices"] != expected_scopes
        ):
            raise RuntimeError("runtime position scopes differ from registration")
        base_margin = _finite_number(
            observed["clean_base_correct_minus_donor_margin"],
            label="clean base correctness margin",
        )
        donor_margin = _finite_number(
            observed["clean_donor_correct_minus_base_margin"],
            label="clean donor correctness margin",
        )
        endpoint = _finite_number(observed["clean_endpoint"], label="clean endpoint")
        if base_margin <= 0.0 or donor_margin <= 0.0:
            raise RuntimeError("E29.14b clean correctness gate failed")
        if endpoint != base_margin + donor_margin or endpoint <= 0.0:
            raise RuntimeError("E29.14b clean endpoint arithmetic changed")
        if (
            observed["clean_logits_finite"] is not True
            or observed["device"] != "cpu"
            or observed["model_dtype"] != "float64"
            or observed["model_layer_count"] != 24
            or observed["component_count"] != COMPONENT_COUNT
            or observed["patch_direction"] != "donor_to_base_only"
        ):
            raise RuntimeError("E29.14b clean runtime contract changed")
        for key in (
            "clean_base_logit_absolute_max",
            "clean_donor_logit_absolute_max",
            "clean_logit_absolute_max",
        ):
            magnitude = _finite_number(observed[key], label=key)
            if magnitude < 0.0 or magnitude > logit_limit:
                raise RuntimeError("E29.14b clean logit magnitude gate failed")
        metadata.append(observed)
        base_margins.append(base_margin)
        donor_margins.append(donor_margin)
        endpoints.append(endpoint)
    return (
        metadata,
        torch.tensor(base_margins, dtype=torch.float64),
        torch.tensor(donor_margins, dtype=torch.float64),
        torch.tensor(endpoints, dtype=torch.float64),
    )


def _verify_smoke(smoke: Any, *, mask_rows: int) -> dict[str, Any]:
    if not isinstance(smoke, Mapping):
        raise TypeError("E29.14b batch smoke must be a mapping")
    value = dict(smoke)
    expected_top = {
        "records",
        "mask_rows",
        "unique_mask_scope_cells",
        "physical_mask_scope_cells",
        "scientific_labels_produced",
        "passed",
    }
    require_exact_keys(value, expected_top, label="E29.14b batch smoke")
    if (
        value["mask_rows"] != mask_rows
        or value["unique_mask_scope_cells"]
        != BATCH_INVARIANCE["condition_rows"]
        or value["physical_mask_scope_cells"]
        != BATCH_INVARIANCE["physical_condition_cells"]
        or value["scientific_labels_produced"] is not False
        or value["passed"] is not True
    ):
        raise RuntimeError("E29.14b batch invariance gate failed")
    records = value["records"]
    if not isinstance(records, Mapping) or tuple(records) != POSITION_SCOPE_ORDER:
        raise RuntimeError("E29.14b batch-smoke scope order changed")
    record_keys = {
        "mask_rows",
        "reference_batch_size",
        "candidate_batch_size",
        "maximum_absolute_difference",
        "maximum_relative_difference",
        "atol",
        "rtol",
        "passed",
    }
    for record in records.values():
        require_exact_keys(record, record_keys, label="E29.14b smoke record")
        if (
            record["mask_rows"] != mask_rows
            or record["reference_batch_size"]
            != BATCH_INVARIANCE["reference_batch_size"]
            or record["candidate_batch_size"]
            != BATCH_INVARIANCE["candidate_batch_size"]
            or record["atol"] != BATCH_INVARIANCE["absolute_tolerance"]
            or record["rtol"] != BATCH_INVARIANCE["relative_tolerance"]
            or record["passed"] is not True
        ):
            raise RuntimeError("E29.14b batch invariance gate failed")
        for key in ("maximum_absolute_difference", "maximum_relative_difference"):
            if _finite_number(record[key], label=key) < 0.0:
                raise RuntimeError("E29.14b batch difference is negative")
    return value


def acquire_component_position_panel_with_runner(
    runner: Any,
    *,
    prompt_checks: Sequence[Mapping[str, Any]],
    masks: Mapping[str, torch.Tensor],
    registration_sha256: str = "pure_runner",
    expected_registration_sha256_supplied: str = "pure_runner",
) -> dict[str, Any]:
    """Acquire the complete fixed panel after validating every clean context."""
    panels = validate_bridge_masks(masks)
    records = prompt_records(prompt_checks)
    prepared = tuple(runner.prepare_context(record) for record in records)
    clean_metadata, base_margins, donor_margins, endpoints = _verify_clean_contexts(
        prompt_checks, prepared
    )
    singleton_mask = torch.zeros(1, COMPONENT_COUNT, dtype=torch.float64)
    singleton_mask[0, HUB_COMPONENT] = 1.0
    combined_masks = torch.cat(
        (singleton_mask, *(panels[split] for split in MASK_SPLITS)), dim=0
    )
    if len(combined_masks) != 97:
        raise RuntimeError("E29.14b combined mask count changed")

    singleton_all = []
    singleton_lexical = []
    all_by_split: dict[str, list[torch.Tensor]] = {split: [] for split in MASK_SPLITS}
    lexical_by_split: dict[str, list[torch.Tensor]] = {
        split: [] for split in MASK_SPLITS
    }
    ordered_by_context = []
    for context in prepared:
        paired = runner.paired_responses_from_prepared(context, combined_masks)
        if not isinstance(paired, Mapping) or tuple(paired) != POSITION_SCOPE_ORDER:
            raise RuntimeError("E29.14b paired response scope order changed")
        all_values = paired["all_positions"].detach().cpu().double()
        lexical_values = paired["differing_lexical_positions"].detach().cpu().double()
        if (
            tuple(all_values.shape) != (97,)
            or tuple(lexical_values.shape) != (97,)
            or not bool(all_values.isfinite().all() and lexical_values.isfinite().all())
        ):
            raise RuntimeError("E29.14b paired response shape or finiteness changed")
        singleton_all.append(all_values[0])
        singleton_lexical.append(lexical_values[0])
        offset = 1
        ordered = [all_values[0], lexical_values[0]]
        for split in MASK_SPLITS:
            rows = MASK_SHAPES[split][0]
            all_split = all_values[offset : offset + rows]
            lexical_split = lexical_values[offset : offset + rows]
            all_by_split[split].append(all_split)
            lexical_by_split[split].append(lexical_split)
            for row_index in range(rows):
                ordered.extend((all_split[row_index], lexical_split[row_index]))
            offset += rows
        ordered_by_context.append(torch.stack(ordered))

    singleton_all_tensor = torch.stack(singleton_all)
    singleton_lexical_tensor = torch.stack(singleton_lexical)
    all_tables = {split: torch.stack(rows, dim=1) for split, rows in all_by_split.items()}
    lexical_tables = {
        split: torch.stack(rows, dim=1) for split, rows in lexical_by_split.items()
    }
    registered_order = torch.stack(ordered_by_context, dim=1)
    smoke = _verify_smoke(
        runner.full_batch_invariance_smoke(
            prepared[int(BATCH_INVARIANCE["context_index"])],
            combined_masks,
            reference_batch_size=int(BATCH_INVARIANCE["reference_batch_size"]),
            candidate_batch_size=int(BATCH_INVARIANCE["candidate_batch_size"]),
        ),
        mask_rows=len(combined_masks),
    )
    combined_responses = torch.cat(
        (
            singleton_all_tensor,
            singleton_lexical_tensor,
            *(table.flatten() for table in all_tables.values()),
            *(table.flatten() for table in lexical_tables.values()),
        )
    )
    numerical_summary = {
        "singleton_all_response": tensor_numerical_summary(singleton_all_tensor),
        "singleton_lexical_response": tensor_numerical_summary(
            singleton_lexical_tensor
        ),
        "responses_in_registered_order": tensor_numerical_summary(registered_order),
        "combined_responses": tensor_numerical_summary(combined_responses),
        "clean_endpoints": tensor_numerical_summary(endpoints),
    }
    if not all(record["passed"] for record in numerical_summary.values()):
        raise RuntimeError("E29.14b numerical summary gate failed")
    artifact = {
        "schema_version": "e29_14b_component_position_labels_v1",
        "registration_sha256": registration_sha256,
        "expected_registration_sha256_supplied": (
            expected_registration_sha256_supplied
        ),
        "condition_manifest": build_condition_manifest(panels),
        "held_masks": {split: panels[split].clone() for split in MASK_SPLITS},
        "singleton_all_response": singleton_all_tensor,
        "singleton_lexical_response": singleton_lexical_tensor,
        "all_position_responses": all_tables,
        "lexical_position_responses": lexical_tables,
        "responses_in_registered_order": registered_order,
        "clean_metadata": clean_metadata,
        "clean_base_correct_minus_donor_margins": base_margins,
        "clean_donor_correct_minus_base_margins": donor_margins,
        "clean_endpoints": endpoints,
        "batch_invariance_smoke": smoke,
        "query_accounting": dict(QUERY_ACCOUNTING),
        "physical_query_accounting": dict(PHYSICAL_QUERY_ACCOUNTING),
        "numerical_summary": numerical_summary,
        "all_integrity_gates_passed": True,
    }
    require_exact_keys(artifact, LABEL_KEYS, label="E29.14b labels")
    return artifact


def _load_tokenizer(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def _load_model(snapshot: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float64,
    ).eval()


def acquire_component_position(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    output_directory: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Verify the seal, consume the attempt, acquire, and write custody once."""
    started = time.perf_counter()
    attempt_path = output_directory / "component_position_acquisition_attempt.json"
    labels_path = output_directory / "component_position_labels.pt"
    custody_path = output_directory / "component_position_acquisition_custody.json"
    if attempt_path.exists() or labels_path.exists() or custody_path.exists():
        raise FileExistsError("refusing to rerun or overwrite E29.14b custody")
    registration = load_verified_registration(
        registration_path,
        expected_sha256=expected_registration_sha256,
        required_keys=REGISTRATION_REQUIRED_KEYS,
    )
    registration = validate_registration_payload(
        registration,
        source_directory=SOURCE_DIRECTORY,
        verify_external_files=True,
    )
    observed_registration_sha = sha256(registration_path)
    if observed_registration_sha != expected_registration_sha256:
        raise RuntimeError("E29.14b registration changed during verification")
    if registration["environment"] != runtime_environment():
        raise RuntimeError("E29.14b runtime environment differs from registration")
    execution = registration["execution"]
    if (
        batch_size != BATCH_SIZE
        or batch_size != execution["batch_size"]
        or execution["device"] != "cpu"
        or execution["model_dtype"] != "float64"
        or execution["attention_implementation"] != "eager"
    ):
        raise RuntimeError("E29.14b runtime execution differs from registration")
    panels = load_prior_bridge_masks(Path(registration["prior_protocol_masks"]))
    attempt = build_acquisition_attempt_record(
        registration=registration,
        registration_path=registration_path,
        observed_registration_sha256=observed_registration_sha,
        expected_registration_sha256=expected_registration_sha256,
    )
    attempt_sha = create_acquisition_attempt_marker(attempt, attempt_path)

    snapshot = Path(registration["model_snapshot"])
    tokenizer = _load_tokenizer(snapshot)
    model = _load_model(snapshot)
    runner = Float64ComponentPositionRunner(model, tokenizer, batch_size=batch_size)
    artifact = acquire_component_position_panel_with_runner(
        runner,
        prompt_checks=registration["prompt_checks"],
        masks=panels,
        registration_sha256=observed_registration_sha,
        expected_registration_sha256_supplied=expected_registration_sha256,
    )
    response_maximum = float(
        artifact["responses_in_registered_order"].abs().max()
    )
    atomic_torch_save(artifact, labels_path)
    custody = {
        "schema_version": "e29_14b_component_position_acquisition_custody_v1",
        "status": "all_registered_conditions_acquired_after_integrity_gates",
        "registration": str(registration_path.resolve()),
        "registration_sha256": observed_registration_sha,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "source_code_sha256_verified_before_acquisition": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_acquisition": dict(
            registration["checkpoint_sha256"]
        ),
        "prior_protocol_masks_sha256_verified_before_acquisition": registration[
            "prior_protocol_masks_sha256"
        ],
        "acquisition_attempt": str(attempt_path.resolve()),
        "acquisition_attempt_sha256": attempt_sha,
        "component_position_labels": str(labels_path.resolve()),
        "component_position_labels_sha256": sha256(labels_path),
        "prompt_panel_sha256": registration["prompt_panel_sha256"],
        "component_position_protocol_sha256": registration[
            "component_position_protocol_sha256"
        ],
        "contexts": QUERY_ACCOUNTING["contexts"],
        "patched_condition_rows_per_context": QUERY_ACCOUNTING[
            "patched_condition_rows_per_context"
        ],
        "patched_context_condition_cells": QUERY_ACCOUNTING[
            "patched_context_condition_cells"
        ],
        "clean_forward_examples": QUERY_ACCOUNTING["clean_forward_examples"],
        "batch_smoke_unique_condition_rows": PHYSICAL_QUERY_ACCOUNTING[
            "batch_smoke_unique_condition_rows"
        ],
        "batch_smoke_physical_condition_cells": PHYSICAL_QUERY_ACCOUNTING[
            "batch_smoke_physical_condition_cells"
        ],
        "maximum_logical_queries": QUERY_ACCOUNTING["maximum_logical_queries"],
        "maximum_physical_model_examples": PHYSICAL_QUERY_ACCOUNTING[
            "maximum_physical_model_examples"
        ],
        "batch_size": batch_size,
        "batch_invariance_passed": True,
        "all_clean_and_numerical_gates_passed": True,
        "device": "cpu",
        "dtype": "float64",
        "response_absolute_max": response_maximum,
        "runtime_seconds": time.perf_counter() - started,
    }
    require_exact_keys(custody, CUSTODY_KEYS, label="E29.14b acquisition custody")
    atomic_json_save(custody, custody_path)
    return {**custody, "custody_sha256": sha256(custody_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()
    print(
        json.dumps(
            acquire_component_position(
                registration_path=arguments.registration,
                expected_registration_sha256=arguments.expected_registration_sha256,
                output_directory=arguments.output_directory,
                batch_size=arguments.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CLEAN_METADATA_KEYS",
    "CUSTODY_KEYS",
    "LABEL_KEYS",
    "acquire_component_position",
    "acquire_component_position_panel_with_runner",
    "build_condition_manifest",
    "prompt_records",
]
