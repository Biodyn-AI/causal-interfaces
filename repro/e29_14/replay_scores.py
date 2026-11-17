#!/usr/bin/env python3
"""Hash-verify and replay E29.14a--d scoring without model calls.

The historical custody scorers intentionally require the exact acquisition
machine and contain absolute paths. This publication wrapper leaves those
sealed files untouched. It verifies the immutable inputs by digest, invokes
the same pure scientific scorers, and compares every replayed field with the
corresponding field in the published summary under a declared strict policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


EXPERIMENT_IDS = ("a", "b", "c", "d")
MANIFEST_PATH = Path(__file__).with_name("manifest.json")
NUMERIC_RELATIVE_TOLERANCE = 1e-12
NUMERIC_ABSOLUTE_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> str:
    """Verify a file digest and return the observed value."""
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )
    return observed


def load_manifest() -> dict[str, Any]:
    """Load the publication manifest stored beside this script."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "e29_14_portable_reproduction_manifest_v1":
        raise RuntimeError("unsupported E29.14 reproduction manifest")
    if tuple(manifest.get("experiments", {})) != EXPERIMENT_IDS:
        raise RuntimeError("E29.14 experiment manifest order changed")
    return manifest


def _canonical_json(value: Any) -> str:
    """Normalize tuples and mappings using the JSON representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _scientific_values_match(expected: Any, observed: Any) -> bool:
    """Compare scientific output while allowing negligible platform rounding.

    Structure, strings, integers, and booleans must match exactly. Floating-point
    values use a deliberately tight tolerance because PyTorch reductions can differ
    in their final bits between macOS and Linux even with identical float64 inputs.
    """
    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(expected) is type(observed) and expected == observed
    if isinstance(expected, int) or isinstance(observed, int):
        return type(expected) is type(observed) and expected == observed
    if isinstance(expected, Real) and isinstance(observed, Real):
        expected_float = float(expected)
        observed_float = float(observed)
        if not math.isfinite(expected_float) or not math.isfinite(observed_float):
            return expected_float == observed_float
        return math.isclose(
            expected_float,
            observed_float,
            rel_tol=NUMERIC_RELATIVE_TOLERANCE,
            abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
        )
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        return expected.keys() == observed.keys() and all(
            _scientific_values_match(expected[key], observed[key]) for key in expected
        )
    if (
        isinstance(expected, Sequence)
        and isinstance(observed, Sequence)
        and not isinstance(expected, (str, bytes, bytearray))
        and not isinstance(observed, (str, bytes, bytearray))
    ):
        return len(expected) == len(observed) and all(
            _scientific_values_match(expected_item, observed_item)
            for expected_item, observed_item in zip(expected, observed, strict=True)
        )
    return type(expected) is type(observed) and expected == observed


def _load_inputs(
    repository_root: Path, experiment_id: str, record: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    import torch

    registration_path = repository_root / str(record["registration"])
    labels_path = repository_root / str(record["labels"])
    summary_path = repository_root / str(record["published_summary"])
    verify_hash(registration_path, str(record["registration_sha256"]))
    verify_hash(labels_path, str(record["labels_sha256"]))
    verify_hash(summary_path, str(record["published_summary_sha256"]))
    if "protocol_artifact" in record:
        verify_hash(
            repository_root / str(record["protocol_artifact"]),
            str(record["protocol_artifact_sha256"]),
        )

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    published = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(registration, dict) or not isinstance(labels, Mapping):
        raise TypeError(f"E29.14{experiment_id} inputs have unexpected types")
    if not isinstance(published, dict):
        raise TypeError(f"E29.14{experiment_id} summary is not a JSON object")
    return registration, labels, published


def score_labels(
    repository_root: Path,
    experiment_id: str,
    record: Mapping[str, Any],
    registration: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Invoke the original pure scorer for one hash-bound label artifact."""
    expected_registration_sha256 = str(record["registration_sha256"])
    if experiment_id == "a":
        from experiments.e29_gpt2_stable_head_interactions.score_e29_14a_position import (
            score_label_artifact,
        )

        return score_label_artifact(
            labels,
            expected_registration_sha256=expected_registration_sha256,
        )

    if experiment_id == "b":
        from experiments.e29_gpt2_stable_head_interactions.score_e29_14b_component_position import (
            score_label_artifact,
        )

        return score_label_artifact(
            labels,
            expected_registration_sha256=expected_registration_sha256,
            registered_protocol=registration["component_position_protocol"],
        )

    if experiment_id == "c":
        from experiments.e29_gpt2_stable_head_interactions.e29_14c_scale_scope_custody import (
            load_verified_protocol_masks,
        )
        from experiments.e29_gpt2_stable_head_interactions.score_e29_14c_scale_scope import (
            score_label_artifact,
        )

        held_masks = load_verified_protocol_masks(
            repository_root / str(record["protocol_artifact"]),
            expected_sha256=str(record["protocol_artifact_sha256"]),
        )
        return score_label_artifact(
            labels,
            expected_registration_sha256=expected_registration_sha256,
            registered_protocol=registration["scale_scope_protocol"],
            held_masks=held_masks,
            clean_metadata_contract=registration["clean_metadata_contract"],
        )

    if experiment_id == "d":
        import experiments.e29_gpt2_stable_head_interactions.score_e29_14d_direct_backup as scorer

        return scorer.score_label_artifact(
            labels,
            expected_registration_sha256=expected_registration_sha256,
            expected_label_keys=scorer.LABEL_KEYS,
            expected_condition_manifest=scorer.build_condition_manifest(),
            expected_query_manifest=scorer.build_query_manifest(),
            expected_query_accounting=scorer.QUERY_ACCOUNTING,
            expected_physical_query_accounting=scorer.PHYSICAL_QUERY_ACCOUNTING,
            clean_metadata_contract=registration["clean_metadata_contract"],
            clean_metadata_keys=scorer.CLEAN_METADATA_KEYS,
            scientific_arguments=scorer._registered_scientific_arguments(),
        )

    raise ValueError(f"unknown E29.14 experiment: {experiment_id}")


def replay_experiment(
    repository_root: Path,
    experiment_id: str,
    *,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Replay one experiment and require strict agreement with its publication."""
    manifest = load_manifest()
    record = manifest["experiments"][experiment_id]
    registration, labels, published = _load_inputs(
        repository_root, experiment_id, record
    )
    replayed = score_labels(
        repository_root, experiment_id, record, registration, labels
    )
    published_scientific_fields = {key: published[key] for key in replayed}
    exact_json_match = _canonical_json(replayed) == _canonical_json(
        published_scientific_fields
    )
    if not _scientific_values_match(replayed, published_scientific_fields):
        mismatches = [
            key
            for key, value in replayed.items()
            if not _scientific_values_match(value, published.get(key))
        ]
        raise RuntimeError(
            f"E29.14{experiment_id} replay differs from publication: {mismatches}"
        )

    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"e29_14{experiment_id}_replayed_score.json"
        output_path.write_text(
            json.dumps(replayed, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return {
        "experiment": f"E29.14{experiment_id}",
        "registration_sha256": record["registration_sha256"],
        "labels_sha256": record["labels_sha256"],
        "published_summary_sha256": record["published_summary_sha256"],
        "published_scientific_fields_match": True,
        "published_scientific_fields_exact_json_match": exact_json_match,
    }


def replay_all(
    repository_root: Path,
    experiment_ids: Sequence[str] = EXPERIMENT_IDS,
    *,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Replay a selected ordered set of E29.14 experiments."""
    root = repository_root.resolve()
    if not (root / "experiments").is_dir():
        raise FileNotFoundError(f"not a fundamental-mechinterp checkout: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    results = [
        replay_experiment(root, experiment_id, output_directory=output_directory)
        for experiment_id in experiment_ids
    ]
    return {
        "schema_version": "e29_14_portable_score_replay_v1",
        "numeric_comparison": {
            "relative_tolerance": NUMERIC_RELATIVE_TOLERANCE,
            "absolute_tolerance": NUMERIC_ABSOLUTE_TOLERANCE,
        },
        "all_replays_match_published_scientific_fields": all(
            result["published_scientific_fields_match"] for result in results
        ),
        "experiments": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_IDS,
        default=list(EXPERIMENT_IDS),
    )
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            replay_all(
                arguments.repository_root,
                arguments.experiments,
                output_directory=arguments.output_directory,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

