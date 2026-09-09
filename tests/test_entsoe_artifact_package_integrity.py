from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation import entsoe_artifact_package_integrity as module
from pfc_shaping.validation.entsoe_artifact_package_integrity import (
    EntsoeArtifactIntegrityError,
    derive_normalized_schema_sha256,
    derive_snapshot_id,
    validate_artifact_integrity_contract,
    validate_derived_integrity_receipt,
    verify_synthetic_entsoe_artifact_package,
)
from tests.test_entsoe_normalized_quality import _fixture as quality_fixture

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = (
    PHASE / "ENTSOE-ARTIFACT-PACKAGE-INTEGRITY-PREFLIGHT-CONTRACT-V1.json"
)
INTAKE_CONTRACT_PATH = PHASE / "ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json"
PROOF_ROOT = ROOT / "build" / "databricks-eex-daily" / "2026-08-05"
SOURCE_PROOF_PATHS = {
    "d233": (
        PROOF_ROOT
        / "eex-entsoe-governed-acquisition-package-proofs"
        / module.SOURCE_PROOFS["d233"]["content_id"]
        / "manifest.json"
    ),
    "d236": (
        PROOF_ROOT
        / "entsoe-bounded-execution-receipt-proofs"
        / module.SOURCE_PROOFS["d236"]["content_id"]
        / "manifest.json"
    ),
    "d238": (
        PROOF_ROOT
        / "entsoe-daily-capture-reservation-ledger-proofs"
        / module.SOURCE_PROOFS["d238"]["content_id"]
        / "manifest.json"
    ),
    "d239": (
        PROOF_ROOT
        / "entsoe-normalized-quality-profile-proofs"
        / module.SOURCE_PROOFS["d239"]["content_id"]
        / "manifest.json"
    ),
}
REFERENCE = datetime(2026, 8, 2, 3, 0, tzinfo=timezone.utc)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    if hasattr(value, "item"):
        return value.item()
    return value


def _frame_bytes(frame: pd.DataFrame, role: str) -> bytes:
    lines: list[bytes] = []
    for values in frame[list(module.ROLE_COLUMNS[role])].itertuples(
        index=False,
        name=None,
    ):
        row = {
            column: _json_value(value)
            for column, value in zip(module.ROLE_COLUMNS[role], values, strict=True)
        }
        lines.append(
            json.dumps(
                row,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
    return b"\n".join(lines) + b"\n"


@pytest.fixture
def package_factory():
    created: list[Path] = []

    def build(
        *,
        raw_overrides: dict[str, bytes] | None = None,
        row_count_overrides: dict[str, int] | None = None,
        evidence_class: str = "SYNTHETIC_TEST_ONLY",
    ) -> tuple[Path, dict[str, object]]:
        fixture = quality_fixture()
        frames = {
            "series_dimension": fixture["series_dimension"],
            "latest_values": fixture["latest_values"],
            "vintage_values": fixture["vintage_values"],
        }
        raw_overrides = {} if raw_overrides is None else raw_overrides
        row_count_overrides = (
            {} if row_count_overrides is None else row_count_overrides
        )
        payloads = {
            role: raw_overrides.get(role, _frame_bytes(frames[role], role))
            for role in module.ROLE_ORDER
        }
        declarations = []
        for role in module.ROLE_ORDER:
            payload = payloads[role]
            declarations.append(
                {
                    "role": role,
                    "format": "NDJSON",
                    "relative_path": module.ROLE_FILE_NAMES[role],
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "row_count": row_count_overrides.get(role, len(frames[role])),
                    "columns": list(module.ROLE_COLUMNS[role]),
                    "column_types": module.ROLE_TYPES[role],
                    "normalized_schema_sha256": derive_normalized_schema_sha256(
                        role
                    ),
                    "content_encoding": "UTF-8",
                    "line_ending": "LF_FINAL_LF_REQUIRED",
                    "compression": "NONE",
                }
            )
        capture_batch_id = str(uuid.uuid4())
        reservation_id = str(uuid.uuid4())
        proposal_content_id = "d08fd187e064f16b0db822bb0d4b594b54cdb8570efcaa5e3215d0a956c37928"
        derivation = {
            "capture_batch_id": capture_batch_id,
            "proposal_content_id": proposal_content_id,
            "reservation_id": reservation_id,
            "snapshot_reference_time_utc": "2026-08-02T03:00:00Z",
            "artifacts": declarations,
            "sign_semantics_by_series": fixture["sign_semantics_by_series"],
            "backfill_status_by_series": fixture["backfill_status_by_series"],
        }
        snapshot_id = derive_snapshot_id(derivation)
        manifest = {
            "schema_version": module.MANIFEST_SCHEMA,
            "evidence_class": evidence_class,
            "snapshot_id": snapshot_id,
            "capture_batch_id": capture_batch_id,
            "proposal_content_id": proposal_content_id,
            "reservation_id": reservation_id,
            "snapshot_reference_time_utc": "2026-08-02T03:00:00Z",
            "created_at_utc": "2026-08-02T03:00:00Z",
            "artifacts": declarations,
            "sign_semantics_by_series": fixture["sign_semantics_by_series"],
            "backfill_status_by_series": fixture["backfill_status_by_series"],
            "authorities": dict(module.MANIFEST_AUTHORITIES),
        }
        package_dir = module.SYNTHETIC_INTAKE_ROOT / snapshot_id
        package_dir.mkdir(parents=True, exist_ok=False)
        created.append(package_dir)
        for role, payload in payloads.items():
            (package_dir / module.ROLE_FILE_NAMES[role]).write_bytes(payload)
        manifest_path = package_dir / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        return manifest_path, manifest

    yield build

    for package_dir in reversed(created):
        if package_dir.exists():
            for entry in tuple(package_dir.iterdir()):
                if entry.is_file():
                    entry.unlink()
            package_dir.rmdir()


def _verify(manifest_path: Path):
    return verify_synthetic_entsoe_artifact_package(
        contract_path=CONTRACT_PATH,
        intake_contract_path=INTAKE_CONTRACT_PATH,
        source_proof_manifest_paths=SOURCE_PROOF_PATHS,
        package_manifest_path=manifest_path,
        reference_time_utc=REFERENCE,
    )


def test_contract_is_exact_and_has_zero_real_authority() -> None:
    result = validate_artifact_integrity_contract(
        json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    assert result["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert result["synthetic_artifact_integrity_verification_implemented"] is True
    assert result["real_artifact_admission_implemented"] is False
    assert result["databricks_connection_budget"] == 0
    assert result["warehouse_start_budget"] == 0


def test_exact_bytes_flow_through_d239_without_real_authority(package_factory) -> None:
    manifest_path, _ = package_factory()
    result = _verify(manifest_path)
    assert result.synthetic_value_row_count == 1_617
    assert result.receipt["authorities"][
        "synthetic_artifact_integrity_verified"
    ] is True
    assert result.receipt["authorities"]["real_artifact_receipt_admitted"] is False
    assert result.receipt["authorities"]["real_entsoe_values_opened"] is False
    assert result.quality_profile.summary["status"].endswith("MODEL_NO_GO")
    assert all(
        value == 0 for value in result.quality_profile.summary["execution"].values()
    )
    assert result.quality_profile.summary["authorities"][
        "model_input_authorized"
    ] is False


def test_artifact_byte_tamper_is_rejected_before_parse(package_factory) -> None:
    manifest_path, _ = package_factory()
    artifact = manifest_path.parent / module.ROLE_FILE_NAMES["latest_values"]
    artifact.write_bytes(artifact.read_bytes() + b"{}\n")
    with pytest.raises(EntsoeArtifactIntegrityError, match="artifact hash"):
        _verify(manifest_path)


@pytest.mark.parametrize(
    ("payload_mutator", "message"),
    [
        (lambda raw: b"\xef\xbb\xbf" + raw, "BOM"),
        (lambda raw: raw.rstrip(b"\n"), "line ending"),
        (lambda raw: raw.replace(b"\n", b"\r\n", 1), "line ending"),
        (lambda raw: raw.split(b"\n", 1)[0] + b"\n\n", "blank line"),
    ],
)
def test_ndjson_framing_is_strict(package_factory, payload_mutator, message) -> None:
    fixture = quality_fixture()
    original = _frame_bytes(fixture["series_dimension"], "series_dimension")
    payload = payload_mutator(original)
    manifest_path, _ = package_factory(
        raw_overrides={"series_dimension": payload},
        row_count_overrides={"series_dimension": 1},
    )
    with pytest.raises(EntsoeArtifactIntegrityError, match=message):
        _verify(manifest_path)


def test_duplicate_key_nan_wrong_order_and_wrong_type_are_rejected(package_factory) -> None:
    columns = module.ROLE_COLUMNS["series_dimension"]
    duplicate = (
        b'{"series_id":"a","series_id":"b","group_name":"actual_load",'
        b'"field_name":"load","from_zone":"CH","to_zone":"CH",'
        b'"unit":"MW","native_resolution":"PT1H",'
        b'"source_endpoint":"x","document_id":"y"}\n'
    )
    manifest_path, _ = package_factory(
        raw_overrides={"series_dimension": duplicate},
        row_count_overrides={"series_dimension": 1},
    )
    with pytest.raises(EntsoeArtifactIntegrityError, match="duplicate key"):
        _verify(manifest_path)

    fixture = quality_fixture()
    latest = json.loads(
        _frame_bytes(fixture["latest_values"].iloc[[0]], "latest_values")
    )
    latest["value"] = float("nan")
    nan_payload = (
        json.dumps(latest, separators=(",", ":"), allow_nan=True).encode() + b"\n"
    )
    manifest_path, _ = package_factory(
        raw_overrides={"latest_values": nan_payload},
        row_count_overrides={"latest_values": 1},
    )
    with pytest.raises(EntsoeArtifactIntegrityError, match="non-finite constant"):
        _verify(manifest_path)

    row = {
        column: "x"
        for column in reversed(columns)
    }
    wrong_order = json.dumps(row, separators=(",", ":")).encode() + b"\n"
    manifest_path, _ = package_factory(
        raw_overrides={"series_dimension": wrong_order},
        row_count_overrides={"series_dimension": 1},
    )
    with pytest.raises(EntsoeArtifactIntegrityError, match="column order"):
        _verify(manifest_path)

    latest = json.loads(
        _frame_bytes(fixture["latest_values"].iloc[[0]], "latest_values")
    )
    latest["revision_number"] = True
    wrong_type = json.dumps(latest, separators=(",", ":")).encode() + b"\n"
    manifest_path, _ = package_factory(
        raw_overrides={"latest_values": wrong_type},
        row_count_overrides={"latest_values": 1},
    )
    with pytest.raises(EntsoeArtifactIntegrityError, match="non-negative integer"):
        _verify(manifest_path)


def test_declared_row_size_schema_path_and_inventory_mismatch_are_rejected(
    package_factory,
) -> None:
    manifest_path, manifest = package_factory()
    manifest["artifacts"][0]["row_count"] = 0
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeArtifactIntegrityError, match="positive integer"):
        _verify(manifest_path)

    manifest_path, manifest = package_factory()
    manifest["artifacts"][0]["normalized_schema_sha256"] = "0" * 64
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeArtifactIntegrityError, match="schema hash"):
        _verify(manifest_path)

    manifest_path, manifest = package_factory()
    manifest["artifacts"][0]["relative_path"] = "../escape.ndjson"
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeArtifactIntegrityError, match="path is unsafe"):
        _verify(manifest_path)

    manifest_path, _ = package_factory()
    (manifest_path.parent / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(EntsoeArtifactIntegrityError, match="inventory differs"):
        _verify(manifest_path)


def test_snapshot_derivation_and_directory_name_are_bound(package_factory) -> None:
    manifest_path, manifest = package_factory()
    manifest["snapshot_id"] = "0" * 64
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeArtifactIntegrityError, match="snapshot ID derivation"):
        _verify(manifest_path)


def test_real_class_and_authority_escalation_are_rejected(package_factory) -> None:
    manifest_path, manifest = package_factory(evidence_class="REAL_CAPTURE")
    with pytest.raises(EntsoeArtifactIntegrityError, match="manifest evidence"):
        _verify(manifest_path)

    manifest_path, manifest = package_factory()
    manifest["authorities"]["real_artifact_receipt_admitted"] = True
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeArtifactIntegrityError, match="manifest authorities"):
        _verify(manifest_path)


def test_manifest_duplicate_key_and_source_proof_tamper_are_rejected(
    package_factory,
    tmp_path: Path,
) -> None:
    manifest_path, _ = package_factory()
    raw = manifest_path.read_bytes().rstrip(b"\n")
    manifest_path.write_bytes(raw[:-1] + b',"schema_version":"duplicate"}\n')
    with pytest.raises(EntsoeArtifactIntegrityError, match="duplicate key"):
        _verify(manifest_path)

    manifest_path, _ = package_factory()
    tampered = json.loads(SOURCE_PROOF_PATHS["d239"].read_text(encoding="utf-8"))
    tampered["execution"]["network_call_count"] = 1
    tampered_path = tmp_path / "tampered-proof.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    paths = dict(SOURCE_PROOF_PATHS)
    paths["d239"] = tampered_path
    with pytest.raises(EntsoeArtifactIntegrityError, match="proof hash differs"):
        verify_synthetic_entsoe_artifact_package(
            contract_path=CONTRACT_PATH,
            intake_contract_path=INTAKE_CONTRACT_PATH,
            source_proof_manifest_paths=paths,
            package_manifest_path=manifest_path,
            reference_time_utc=REFERENCE,
        )


def test_derived_receipt_rejects_tamper(package_factory) -> None:
    manifest_path, manifest = package_factory()
    result = _verify(manifest_path)
    receipt = copy.deepcopy(result.receipt)
    receipt["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(EntsoeArtifactIntegrityError, match="receipt series_dimension.sha256"):
        validate_derived_integrity_receipt(
            receipt,
            manifest,
            quality_profile_summary_sha256=result.receipt[
                "quality_profile_summary_sha256"
            ],
            reference_time_utc=REFERENCE,
        )


def test_package_outside_synthetic_intake_root_is_rejected(
    package_factory,
    tmp_path: Path,
) -> None:
    manifest_path, _ = package_factory()
    outside = tmp_path / manifest_path.parent.name
    outside.mkdir()
    for source in manifest_path.parent.iterdir():
        (outside / source.name).write_bytes(source.read_bytes())
    with pytest.raises(EntsoeArtifactIntegrityError, match="outside its content-addressed"):
        _verify(outside / "manifest.json")
