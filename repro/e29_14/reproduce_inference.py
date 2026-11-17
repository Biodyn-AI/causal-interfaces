#!/usr/bin/env python3
"""Run an independent CPU/float64 E29.14 inference-to-score reproduction.

This script deliberately does not create or pretend to extend the historical
prospective custody chain.  It verifies the pinned checkpoint and response-
blind artifacts, reuses the frozen acquisition functions, and writes a new
runtime manifest, label artifact, and score under a caller-selected directory.
The historical sealed artifacts remain read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from repro.e29_14.replay_scores import (
    EXPERIMENT_IDS,
    _canonical_json,
    load_manifest,
    score_labels,
    verify_hash,
)


def _atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_checkpoint(snapshot: Path, checkpoint_hashes: dict[str, str]) -> None:
    """Require exactly the pinned files used by the experiment."""
    for filename, expected in checkpoint_hashes.items():
        verify_hash(snapshot / filename, expected)


def _runtime_environment() -> dict[str, str]:
    import einops
    import numpy
    import safetensors
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "einops": einops.__version__,
        "platform": platform.platform(),
    }


def _load_registration(repository_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = repository_root / record["registration"]
    verify_hash(path, record["registration_sha256"])
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"registration is not a JSON object: {path}")
    return value


def _acquire(
    experiment_id: str,
    *,
    repository_root: Path,
    record: dict[str, Any],
    registration: dict[str, Any],
    model: Any,
    tokenizer: Any,
    batch_size: int,
) -> dict[str, Any]:
    registration_sha = record["registration_sha256"]
    if experiment_id == "a":
        from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14a_position import (
            acquire_position_panel_with_runner,
        )
        from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_runner import (
            Float64PositionDoseRunner,
        )

        runner = Float64PositionDoseRunner(model, tokenizer, batch_size=batch_size)
        return acquire_position_panel_with_runner(
            runner,
            prompt_checks=registration["prompt_checks"],
            registration_sha256=registration_sha,
            expected_registration_sha256_supplied=registration_sha,
        )

    from experiments.e29_gpt2_stable_head_interactions.e29_14b_component_position_runner import (
        Float64ComponentPositionRunner,
    )

    runner = Float64ComponentPositionRunner(model, tokenizer, batch_size=batch_size)
    if experiment_id == "b":
        from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14b_component_position import (
            acquire_component_position_panel_with_runner,
        )
        from experiments.e29_gpt2_stable_head_interactions.register_e29_14b_component_position import (
            load_prior_bridge_masks,
        )

        masks = load_prior_bridge_masks(repository_root / record["protocol_artifact"])
        return acquire_component_position_panel_with_runner(
            runner,
            prompt_checks=registration["prompt_checks"],
            masks=masks,
            registration_sha256=registration_sha,
            expected_registration_sha256_supplied=registration_sha,
        )

    if experiment_id == "c":
        from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14c_scale_scope import (
            acquire_component_position_panel_with_runner,
        )
        from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
            load_verified_protocol_masks,
        )

        masks = load_verified_protocol_masks(
            repository_root / record["protocol_artifact"],
            expected_sha256=record["protocol_artifact_sha256"],
        )
        return acquire_component_position_panel_with_runner(
            runner,
            prompt_checks=registration["prompt_checks"],
            masks=masks,
            clean_metadata_contract=registration["clean_metadata_contract"],
            registration_sha256=registration_sha,
            expected_registration_sha256_supplied=registration_sha,
        )

    if experiment_id == "d":
        from experiments.e29_gpt2_stable_head_interactions.acquire_e29_14d_direct_backup import (
            acquire_direct_backup_panel_with_runner,
        )
        from experiments.e29_gpt2_stable_head_interactions.e29_14d_direct_backup_custody import (
            load_verified_protocol_artifact,
        )

        masks = load_verified_protocol_artifact(
            repository_root / record["protocol_artifact"],
            expected_sha256=record["protocol_artifact_sha256"],
        )
        return acquire_direct_backup_panel_with_runner(
            runner,
            prompt_checks=registration["prompt_checks"],
            condition_masks=masks,
            clean_metadata_contract=registration["clean_metadata_contract"],
            registration_sha256=registration_sha,
            expected_registration_sha256_supplied=registration_sha,
        )

    raise ValueError(f"unknown E29.14 experiment: {experiment_id}")


def reproduce(
    repository_root: Path,
    snapshot: Path,
    output_directory: Path,
    experiment_ids: list[str],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Acquire and score the selected frozen protocols without overwriting."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = repository_root.resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    manifest = load_manifest()
    verify_checkpoint(snapshot, manifest["model"]["checkpoint_files"])
    output_directory = output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float64,
    ).eval()
    results = []
    for experiment_id in experiment_ids:
        started = time.perf_counter()
        record = manifest["experiments"][experiment_id]
        registration = _load_registration(root, record)
        labels = _acquire(
            experiment_id,
            repository_root=root,
            record=record,
            registration=registration,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
        score = score_labels(root, experiment_id, record, registration, labels)
        published_summary_path = root / record["published_summary"]
        verify_hash(published_summary_path, record["published_summary_sha256"])
        published = json.loads(published_summary_path.read_text(encoding="utf-8"))
        published_scientific_fields = {key: published[key] for key in score}
        exact_publication_match = _canonical_json(score) == _canonical_json(
            published_scientific_fields
        )
        experiment_directory = output_directory / f"e29_14{experiment_id}"
        labels_path = experiment_directory / "reproduced_labels.pt"
        score_path = experiment_directory / "reproduced_score.json"
        _atomic_torch_save(labels, labels_path)
        _atomic_json_save(score, score_path)
        results.append(
            {
                "experiment": f"E29.14{experiment_id}",
                "registration_sha256": record["registration_sha256"],
                "labels": str(labels_path.relative_to(output_directory)),
                "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                "score": str(score_path.relative_to(output_directory)),
                "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                "published_scientific_fields_exact_match": exact_publication_match,
                "runtime_seconds": time.perf_counter() - started,
            }
        )

    reproduction = {
        "schema_version": "e29_14_independent_inference_reproduction_v1",
        "historical_custody_extended": False,
        "scientific_protocols_and_checkpoint_hashes_verified": True,
        "model_revision": manifest["model"]["revision"],
        "runtime_environment": _runtime_environment(),
        "device": "cpu",
        "dtype": "float64",
        "attention_implementation": "eager",
        "batch_size": batch_size,
        "experiments": results,
    }
    _atomic_json_save(reproduction, output_directory / "reproduction_manifest.json")
    return reproduction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_IDS,
        default=list(EXPERIMENT_IDS),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    print(
        json.dumps(
            reproduce(
                arguments.repository_root,
                arguments.model_snapshot,
                arguments.output_directory,
                list(arguments.experiments),
                batch_size=arguments.batch_size,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

