from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest

import scripts.archive_entso_dataset as archive_module
from scripts.archive_entso_dataset import (
    FMV_DATA_ROOT_ENV,
    PFC_ENTSOE_DATA_ROOT_ENV,
    PFC_SHARED_DATA_ROOT_ENV,
    _parse_args,
    archive_entso_dataset,
    verify_entso_archive,
)


@pytest.fixture(autouse=True)
def _isolate_data_root_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        FMV_DATA_ROOT_ENV,
        PFC_SHARED_DATA_ROOT_ENV,
        PFC_ENTSOE_DATA_ROOT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_archive_entso_dataset_is_hash_bound_and_not_model_eligible(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    index = pd.date_range("2026-07-01", periods=4, freq="15min", tz="UTC")
    pd.DataFrame({"load_mw": [1.0] * 4}, index=index).to_parquet(
        source / "entso_15min.parquet"
    )
    pd.DataFrame({"flow_mw": [2.0] * 4}, index=index).to_parquet(
        source / "entso_border_15min.parquet"
    )
    pd.DataFrame({"price": [3.0] * 4}, index=index).to_parquet(
        source / "epex_15min.parquet"
    )

    (tmp_path / "shared" / "entsoe").mkdir(parents=True)
    destination = archive_entso_dataset(
        source_dir=source,
        data_root=tmp_path / "shared",
        import_id="ct-import-001",
        source_label="TEST_READ_ONLY",
        excluded_files={"epex_15min.parquet"},
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["calibration_eligible"] is False
    assert manifest["schema_version"] == "entso_reusable_import.v2"
    assert manifest["model_activation"] == "FORBIDDEN_UNTIL_GOVERNED_IMPORT"
    assert set(manifest["files"]) == {
        "entso_15min.parquet",
        "entso_border_15min.parquet",
    }
    assert manifest["files"]["entso_15min.parquet"]["rows"] == 4
    assert manifest["source_parquet_inventory"] == [
        "entso_15min.parquet",
        "entso_border_15min.parquet",
        "epex_15min.parquet",
    ]
    assert manifest["excluded_source_parquets"] == ["epex_15min.parquet"]
    assert verify_entso_archive(destination)["import_id"] == "ct-import-001"

    (destination / "entso_15min.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_entso_archive(destination)


@pytest.mark.parametrize("import_id", ["..", "CON", "bad/name", "name."])
def test_archive_entso_dataset_rejects_nonportable_ids(
    tmp_path: Path,
    import_id: str,
) -> None:
    (tmp_path / "shared" / "entsoe").mkdir(parents=True)
    with pytest.raises(ValueError, match="portable path-safe"):
        archive_entso_dataset(
            source_dir=tmp_path,
            data_root=tmp_path / "shared",
            import_id=import_id,
            source_label="TEST",
        )


def test_archive_cli_uses_unambiguous_shared_root_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared"
    entsoe = shared / "entsoe"
    entsoe.mkdir(parents=True)
    monkeypatch.setenv(PFC_SHARED_DATA_ROOT_ENV, str(shared))
    monkeypatch.setenv(PFC_ENTSOE_DATA_ROOT_ENV, str(entsoe))

    args = _parse_args(
        [
            "--source-dir",
            str(tmp_path / "source"),
            "--import-id",
            "test-import",
            "--source-label",
            "TEST",
        ]
    )

    assert args.data_root == shared
    assert args.entsoe_root == entsoe


def test_archive_cli_prefers_generic_fmv_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared"
    (shared / "entsoe").mkdir(parents=True)
    monkeypatch.setenv(FMV_DATA_ROOT_ENV, str(shared))
    monkeypatch.delenv(PFC_SHARED_DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(PFC_ENTSOE_DATA_ROOT_ENV, raising=False)

    args = _parse_args(
        [
            "--source-dir",
            str(tmp_path / "source"),
            "--import-id",
            "test-import",
            "--source-label",
            "TEST",
        ]
    )

    assert args.data_root == shared
    assert args.entsoe_root is None


def test_archive_verification_rejects_directory_relabelling(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    index = pd.date_range("2026-07-01", periods=2, freq="15min", tz="UTC")
    pd.DataFrame({"load_mw": [1.0, 2.0]}, index=index).to_parquet(
        source / "entso_15min.parquet"
    )
    (tmp_path / "shared" / "entsoe").mkdir(parents=True)
    original = archive_entso_dataset(
        source_dir=source,
        data_root=tmp_path / "shared",
        import_id="original-import",
        source_label="TEST",
    )
    relabelled = original.with_name("relabelled-import")
    shutil.copytree(original, relabelled)

    with pytest.raises(ValueError, match="directory name does not match"):
        verify_entso_archive(relabelled)


@pytest.mark.parametrize(
    ("source_dir", "data_root", "message"),
    [
        (Path("relative-source"), Path("C:/shared"), "source_dir must be absolute"),
        (Path("C:/source"), Path("relative-shared"), "data_root must be absolute"),
    ],
)
def test_archive_rejects_relative_roots(
    source_dir: Path,
    data_root: Path,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        archive_entso_dataset(
            source_dir=source_dir,
            data_root=data_root,
            import_id="test-import",
            source_label="TEST",
        )


def test_archive_cli_requires_shared_or_entsoe_root_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PFC_SHARED_DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(PFC_ENTSOE_DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(FMV_DATA_ROOT_ENV, raising=False)

    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--source-dir",
                str(tmp_path / "source"),
                "--import-id",
                "test-import",
                "--source-label",
                "TEST",
            ]
        )


def test_archive_accepts_direct_entsoe_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)
    pd.DataFrame({"load_mw": [1.0]}).to_parquet(source / "entso_15min.parquet")

    destination = archive_entso_dataset(
        source_dir=source,
        entsoe_root=entsoe,
        import_id="direct-root",
        source_label="TEST",
    )

    assert destination == entsoe / "imports" / "direct-root"


def test_archive_rejects_divergent_parent_and_direct_roots(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    (shared / "entsoe").mkdir(parents=True)
    other = tmp_path / "other-entsoe"
    other.mkdir()

    with pytest.raises(ValueError, match="do not identify the same store"):
        archive_entso_dataset(
            source_dir=tmp_path,
            data_root=shared,
            entsoe_root=other,
            import_id="test-import",
            source_label="TEST",
        )


def test_archive_rejects_unclassified_parquet_before_target_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0]}).to_parquet(source / "load_forecast_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)

    with pytest.raises(ValueError, match="archived or excluded explicitly"):
        archive_entso_dataset(
            source_dir=source,
            entsoe_root=entsoe,
            import_id="unknown-source",
            source_label="TEST",
        )

    assert not (entsoe / "imports").exists()


def test_archive_v2_verifier_rejects_inventory_partition_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0]}).to_parquet(source / "entso_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)
    destination = archive_entso_dataset(
        source_dir=source,
        entsoe_root=entsoe,
        import_id="inventory-bound",
        source_label="TEST",
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_parquet_inventory"].append("hidden.parquet")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = destination / "archive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="inventory partition mismatch"):
        verify_entso_archive(destination)


def test_archive_v2_verifier_recomputes_parquet_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0, 2.0]}).to_parquet(source / "entso_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)
    destination = archive_entso_dataset(
        source_dir=source,
        entsoe_root=entsoe,
        import_id="metadata-bound",
        source_label="TEST",
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["entso_15min.parquet"]["rows"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = destination / "archive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="rows mismatch"):
        verify_entso_archive(destination)


def test_archive_rejects_source_inventory_change_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0]}).to_parquet(source / "entso_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)
    original_receipts = archive_module._source_parquet_receipts
    calls = 0

    def changing_receipts(source_dir: Path) -> dict[str, dict[str, object]]:
        nonlocal calls
        calls += 1
        receipts = original_receipts(source_dir)
        if calls > 1:
            receipts["late-arrival.parquet"] = {
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
        return receipts

    monkeypatch.setattr(archive_module, "_source_parquet_receipts", changing_receipts)

    with pytest.raises(ValueError, match="source inventory changed during archival"):
        archive_entso_dataset(
            source_dir=source,
            entsoe_root=entsoe,
            import_id="changing-source",
            source_label="TEST",
        )

    assert not (entsoe / "imports" / "changing-source").exists()


def test_archive_verifies_staging_before_publishing_final_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0]}).to_parquet(source / "entso_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)

    def reject_staging(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("simulated pre-publication verification failure")

    monkeypatch.setattr(archive_module, "_verify_entso_archive", reject_staging)

    with pytest.raises(ValueError, match="pre-publication verification failure"):
        archive_entso_dataset(
            source_dir=source,
            entsoe_root=entsoe,
            import_id="not-published",
            source_label="TEST",
        )

    assert not (entsoe / "imports" / "not-published").exists()


def test_archive_v2_verifier_rejects_manifest_path_relabelling(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0]}).to_parquet(source / "entso_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)
    destination = archive_entso_dataset(
        source_dir=source,
        entsoe_root=entsoe,
        import_id="path-bound",
        source_label="TEST",
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["entso_15min.parquet"]["path"] = "other.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = destination / "archive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest path mismatch"):
        verify_entso_archive(destination)


def test_archive_v2_verifier_rejects_unsupported_coherently_rehashed_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0]}).to_parquet(source / "entso_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)
    destination = archive_entso_dataset(
        source_dir=source,
        entsoe_root=entsoe,
        import_id="schema-bound",
        source_label="TEST",
    )
    old_name = "entso_15min.parquet"
    new_name = "unknown_signal.parquet"
    (destination / old_name).rename(destination / new_name)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"].pop(old_name)
    entry["path"] = new_name
    manifest["files"][new_name] = entry
    source_receipt = manifest["source_parquet_receipts"].pop(old_name)
    manifest["source_parquet_receipts"][new_name] = source_receipt
    manifest["source_parquet_inventory"] = [new_name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = destination / "archive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt["archive_content_sha256"] = hashlib.sha256(
        json.dumps(
            {new_name: entry["sha256"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported archive files"):
        verify_entso_archive(destination)


def test_archive_rejects_imports_link_before_writing_outside_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"value": [1.0]}).to_parquet(source / "entso_15min.parquet")
    entsoe = tmp_path / "shared" / "entsoe"
    entsoe.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, entsoe / "imports", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="resolves outside entsoe_root"):
        archive_entso_dataset(
            source_dir=source,
            entsoe_root=entsoe,
            import_id="escaped-import",
            source_label="TEST",
        )

    assert list(outside.iterdir()) == []
