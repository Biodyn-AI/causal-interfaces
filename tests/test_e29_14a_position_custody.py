"""Focused tests for generic E29.14a hash and write-once custody."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.e29_gpt2_stable_head_interactions.e29_14a_position_custody import (
    atomic_json_save,
    load_verified_registration,
    sha256,
    verify_checkpoint_closure,
    verify_source_closure,
)


def test_source_closure_hashes_exact_named_files(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "beta.py").write_text("beta\n", encoding="utf-8")

    hashes = verify_source_closure(tmp_path, ("alpha.py", "beta.py"))

    assert tuple(hashes) == ("alpha.py", "beta.py")
    assert hashes == {
        "alpha.py": sha256(tmp_path / "alpha.py"),
        "beta.py": sha256(tmp_path / "beta.py"),
    }


def test_source_closure_rejects_duplicates_missing_and_nested_names(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha.py").write_text("alpha\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        verify_source_closure(tmp_path, ("alpha.py", "alpha.py"))
    with pytest.raises(FileNotFoundError):
        verify_source_closure(tmp_path, ("missing.py",))
    with pytest.raises(ValueError, match="plain filenames"):
        verify_source_closure(tmp_path, ("nested/alpha.py",))


def test_checkpoint_closure_fails_closed_on_key_or_hash_drift(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    required = ("config.json", "model.safetensors")
    expected = {name: sha256(tmp_path / name) for name in required}

    assert verify_checkpoint_closure(tmp_path, expected, required) == expected
    with pytest.raises(RuntimeError, match="keys changed"):
        verify_checkpoint_closure(tmp_path, {"config.json": expected["config.json"]}, required)

    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_checkpoint_closure(tmp_path, expected, required)


def test_registration_load_and_atomic_write_are_hash_bound_and_write_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registration.json"
    record = {"schema_version": "test", "status": "sealed"}
    atomic_json_save(record, path)

    assert load_verified_registration(
        path,
        expected_sha256=sha256(path),
        required_keys=frozenset(record),
    ) == record
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_verified_registration(
            path,
            expected_sha256="0" * 64,
            required_keys=frozenset(record),
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        atomic_json_save(record, path)

    path.write_text(json.dumps({**record, "extra": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="keys changed"):
        load_verified_registration(
            path,
            expected_sha256=sha256(path),
            required_keys=frozenset(record),
        )
