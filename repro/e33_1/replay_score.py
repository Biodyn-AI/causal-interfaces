"""Verify E33.1 custody and exactly replay its saved-label score."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPOSITORY = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    manifest_path = repository / "repro/e33_1/manifest.json"
    manifest = load_object(manifest_path)

    verified_hashes: dict[str, bool] = {}
    for name, artifact in manifest["artifacts"].items():
        artifact_path = repository / artifact["path"]
        verified_hashes[name] = sha256(artifact_path) == artifact["sha256"]
    if not all(verified_hashes.values()):
        failed = [name for name, passed in verified_hashes.items() if not passed]
        raise RuntimeError(f"artifact SHA-256 mismatch: {failed}")

    sys.path.insert(0, str(repository))
    from experiments.e33_ioi_scope_replication.scoring_json_repair import (  # noqa: PLC0415
        score_json_context_records,
    )

    labels = load_object(repository / manifest["artifacts"]["labels"]["path"])
    replayed_score = score_json_context_records(labels["context_records"])
    replayed_score.update(manifest["registered_scoring_provenance"])
    published_score = load_object(repository / manifest["artifacts"]["score"]["path"])
    exact_match = replayed_score == published_score
    if not exact_match:
        raise RuntimeError("replayed scientific score differs from the published score")

    json.dump(
        {
            "all_artifact_hashes_match": True,
            "all_registered_gates_passed": replayed_score[
                "all_registered_gates_passed"
            ],
            "exact_saved_label_score_match": True,
            "verified_artifacts": sorted(verified_hashes),
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
