from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.calibration.monthly_curve_lambda_calibration as lambda_calibration
import pfc_shaping.data.acquisition_contract as acquisition_contract
from pfc_shaping.calibration.monthly_curve_lambda_calibration import (
    LambdaCalibrationSettings,
    run_verified_lambda_calibration,
    write_calibration_artifacts,
)
from pfc_shaping.data import ingest_forwards
from pfc_shaping.data.acquisition_contract import (
    TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
    TRUSTED_TIME_JOURNAL_ID_ENV,
    TRUSTED_TIME_PUBLIC_KEY_ENV,
    TRUSTED_TIME_RECEIPT_SCHEMA,
    AcquisitionAuthenticationError,
    verify_acquisition_contract,
    verify_trusted_time_receipt,
)
from pfc_shaping.data.acquisition_signing import sign_acquisition_contract
from pfc_shaping.data.eex_historical_vintage import (
    EEX_HISTORICAL_CATALOG_SCHEMA,
    EEX_HISTORICAL_QUOTE_SCHEMA,
    EexHistoricalVintageError,
    VerifiedEexHistoricalVintageCatalog,
    _replay_source_document,
    historical_vintage_revision_id,
    historical_vintage_row_hash,
    historical_vintage_snapshot_id,
    validate_eex_historical_vintage_frame,
    verify_eex_historical_vintage_catalog,
)


def test_trusted_public_key_rejects_hardlink(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    canonical = tmp_path / "trusted-public.pem"
    alias = tmp_path / "trusted-public-hardlink.pem"
    canonical.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    os.link(canonical, alias)

    with pytest.raises(AcquisitionAuthenticationError, match="exactly one link"):
        acquisition_contract._load_public_key(alias)


def test_trusted_public_key_rejects_link_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = Ed25519PrivateKey.generate()
    public_path = tmp_path / "trusted-public.pem"
    public_path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    import pfc_shaping.path_safety as path_safety

    original = path_safety.path_is_link
    monkeypatch.setattr(
        path_safety,
        "path_is_link",
        lambda path: Path(path) == public_path or original(path),
    )

    with pytest.raises(AcquisitionAuthenticationError, match="contain no links"):
        acquisition_contract._absolute_trusted_key_path(public_path)


def test_trusted_public_key_read_is_bound_to_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()
    trusted_path = tmp_path / "trusted-public.pem"
    attacker_path = tmp_path / "attacker-public.pem"
    for path, key in ((trusted_path, trusted), (attacker_path, attacker)):
        path.write_bytes(
            key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
    original_open = acquisition_contract.os.open

    def substituted_open(path: str | Path, flags: int) -> int:
        if Path(path) == trusted_path:
            return original_open(attacker_path, flags)
        return original_open(path, flags)

    monkeypatch.setattr(acquisition_contract.os, "open", substituted_open)

    with pytest.raises(AcquisitionAuthenticationError, match="changed while"):
        acquisition_contract._load_public_key(trusted_path)


def test_historical_acquisition_key_is_rejected_without_prior_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = Ed25519PrivateKey.generate()
    active = Ed25519PrivateKey.generate()
    active_public = _write_public_key(tmp_path / "active.pem", active)
    historical_private = _write_private_key(
        tmp_path / "historical-private.pem",
        historical,
    )
    unsigned = {"schema_version": "lt_input_snapshot.v1", "snapshot_id": "snapshot-001"}
    signed = sign_acquisition_contract(unsigned, private_key_path=historical_private)
    monkeypatch.setenv(TRUSTED_ACQUISITION_PUBLIC_KEY_ENV, str(active_public))

    with pytest.raises(AcquisitionAuthenticationError, match="historical replay is unsupported"):
        verify_acquisition_contract(signed)


def test_historical_trusted_time_key_is_rejected_without_prior_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = Ed25519PrivateKey.generate()
    active = Ed25519PrivateKey.generate()
    active_public = _write_public_key(tmp_path / "active-time.pem", active)
    unsigned = {
        "schema_version": TRUSTED_TIME_RECEIPT_SCHEMA,
        "journal_id": "trusted-time-journal-001",
    }
    payload = acquisition_contract._canonical_json_bytes(unsigned)
    signed = {
        **unsigned,
        "trusted_time_attestation": {
            "algorithm": acquisition_contract.SIGNATURE_ALGORITHM,
            "key_id": acquisition_contract._public_key_id(historical.public_key()),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "value_base64": base64.b64encode(historical.sign(payload)).decode("ascii"),
        },
    }
    monkeypatch.setenv(TRUSTED_TIME_PUBLIC_KEY_ENV, str(active_public))
    monkeypatch.setenv(TRUSTED_TIME_JOURNAL_ID_ENV, "trusted-time-journal-001")

    with pytest.raises(AcquisitionAuthenticationError, match="historical replay is unsupported"):
        verify_trusted_time_receipt(signed)


def _write_public_key(path: Path, key: Ed25519PrivateKey) -> Path:
    path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return path


def _write_private_key(path: Path, key: Ed25519PrivateKey) -> Path:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return path


def test_valid_vintages_preserve_real_availability_and_source_identity() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-03-01", "2027", 80.0, available_at="2026-03-02T08:00:00Z"),
            _row("2026-04-01", "2027", 81.0, available_at="2026-04-01T08:00:00Z"),
        ]
    )

    actual = validate_eex_historical_vintage_frame(frame)

    assert actual["available_at"].tolist() == [
        pd.Timestamp("2026-03-02T08:00:00Z"),
        pd.Timestamp("2026-04-01T08:00:00Z"),
    ]
    assert actual.attrs["rolling_origin_provenance"]["snapshot_count"] == 2
    assert actual.attrs["rolling_origin_provenance"]["source_document_count"] == 2


def test_legacy_latest_revision_history_is_not_rolling_origin_evidence() -> None:
    legacy = pd.DataFrame(
        {
            "date": ["2024-02-29"],
            "product": ["2027"],
            "load_type": ["BASE"],
            "product_type": ["Cal"],
            "price": [80.0],
            "market": ["CH"],
            "source": ["EEX"],
        }
    )

    with pytest.raises(EexHistoricalVintageError, match="not vintage-verifiable"):
        validate_eex_historical_vintage_frame(legacy)


@pytest.mark.parametrize("source", ["OMPEX", "BENCHMARK", "SPOT_PROXY"])
def test_non_eex_sources_are_forbidden(source: str) -> None:
    row = _row("2026-03-01", "2027", 80.0)
    row["source"] = source
    _repin(row)

    with pytest.raises(EexHistoricalVintageError, match="forbids OMPEX"):
        validate_eex_historical_vintage_frame(pd.DataFrame([row]))


def test_naive_available_at_is_rejected() -> None:
    row = _row("2026-03-01", "2027", 80.0)
    row["available_at"] = "2026-03-01 08:00:00"

    with pytest.raises(EexHistoricalVintageError, match="available_at must be timezone-aware"):
        validate_eex_historical_vintage_frame(pd.DataFrame([row]))


def test_semantic_mutation_cannot_be_hidden_by_parquet_rehash() -> None:
    row = _row("2026-03-01", "2027", 80.0)
    frame = pd.DataFrame([row])
    validate_eex_historical_vintage_frame(frame)
    frame.loc[0, "price"] = 1.0

    with pytest.raises(EexHistoricalVintageError, match="row hash mismatch"):
        validate_eex_historical_vintage_frame(frame)


def test_snapshot_metadata_cannot_diverge_under_one_snapshot_id() -> None:
    first = _row("2026-03-01", "2027", 80.0)
    second = _row("2026-03-01", "2027-Q1", 81.0)
    second["source_document_id"] = "substituted.xlsx"
    _repin(second, keep_snapshot_id=True)

    with pytest.raises(EexHistoricalVintageError, match="inconsistent source-vintage metadata"):
        validate_eex_historical_vintage_frame(pd.DataFrame([first, second]))


def test_revision_parent_must_be_present() -> None:
    row = _row("2026-03-01", "2027", 80.0)
    row["revision_sequence"] = 2
    row["supersedes_quote_id"] = "c" * 64
    _repin(row)

    with pytest.raises(EexHistoricalVintageError, match="revision parent is missing"):
        validate_eex_historical_vintage_frame(pd.DataFrame([row]))


def test_revision_cannot_change_economic_quote_identity() -> None:
    parent = _row("2026-03-01", "2027", 80.0)
    revision = _row(
        "2026-03-01",
        "2028",
        81.0,
        available_at="2026-03-02T08:00:00Z",
    )
    revision["revision_sequence"] = 2
    revision["supersedes_quote_id"] = parent["quote_id"]
    _repin(revision)

    with pytest.raises(EexHistoricalVintageError, match="changes quote identity"):
        validate_eex_historical_vintage_frame(pd.DataFrame([parent, revision]))


def test_revision_identity_has_exactly_one_root() -> None:
    first = _row("2026-03-01", "2027", 80.0)
    second = _row(
        "2026-03-01",
        "2027",
        81.0,
        available_at="2026-03-01T09:00:00Z",
    )
    second["source_document_id"] = "second-document.xlsx"
    second["source_document_sha256"] = "c" * 64
    second["acquisition_id"] = "second-acquisition"
    second["ingestion_run_id"] = "second-ingestion"
    _repin(second)

    with pytest.raises(EexHistoricalVintageError, match="one root"):
        validate_eex_historical_vintage_frame(pd.DataFrame([first, second]))


def test_distinct_documents_cannot_repeat_quote_at_same_availability() -> None:
    first = _row("2026-03-01", "2027", 80.0)
    second = dict(first)
    second["source_document_id"] = "second-document.xlsx"
    second["source_document_sha256"] = "c" * 64
    second["acquisition_id"] = "second-acquisition"
    second["ingestion_run_id"] = "second-ingestion"
    _repin(second)

    with pytest.raises(EexHistoricalVintageError, match="one availability cutoff"):
        validate_eex_historical_vintage_frame(pd.DataFrame([first, second]))


def test_vintage_catalog_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text('{"catalog_id":"first","catalog_id":"second"}', encoding="utf-8")

    with pytest.raises(EexHistoricalVintageError, match="unreadable"):
        verify_eex_historical_vintage_catalog(
            {},
            catalog_path=catalog_path,
            history_path=tmp_path / "history.parquet",
        )


def test_public_calibration_boundary_accepts_paths_not_forgeable_evidence_objects() -> None:
    parameters = inspect.signature(run_verified_lambda_calibration).parameters
    verifier_parameters = inspect.signature(
        verify_eex_historical_vintage_catalog
    ).parameters

    assert "history" not in parameters
    assert "history_bytes" not in parameters
    assert "vintage_evidence" not in parameters
    assert "allow_unverified_vintage_fixture" not in parameters
    assert {"history_path", "catalog_path"}.issubset(parameters)
    assert "history_bytes" not in verifier_parameters
    forged = VerifiedEexHistoricalVintageCatalog(
        catalog_id="forged",
        catalog_sha256="a" * 64,
        history_sha256="b" * 64,
        snapshot_count=1,
        source_document_count=1,
        data_cutoff_utc="2026-03-01T00:00:00+00:00",
    )
    assert forged.verified is False


def test_one_document_cannot_repeat_quote_under_multiple_snapshots() -> None:
    workbook = pd.DataFrame(
        [
            [None, "Y01_2027_BASE"],
            [None, "ISIN_fixture"],
            ["Date", None],
            ["01.03.2026", 80.0],
        ]
    )
    source = BytesIO()
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        workbook.to_excel(writer, sheet_name="CH", index=False, header=False)
    parser_hash = hashlib.sha256(Path(ingest_forwards.__file__).read_bytes()).hexdigest()
    parser_config = {
        "parser_version": "ingest_forwards.v1",
        "selection_mode": "latest_available",
        "markets": {"CH": "2026-03-01"},
    }
    parser_config_hash = _canonical_sha(parser_config)
    first = _row("2026-03-01", "2027", 80.0)
    second = dict(first)
    second["acquisition_id"] = "second-acquisition"
    second["ingestion_run_id"] = "second-ingestion"
    for row in (first, second):
        row["parser_code_sha256"] = parser_hash
        row["parser_config_sha256"] = parser_config_hash
        _repin(row)

    with pytest.raises(EexHistoricalVintageError, match="repeats an economic quote"):
        _replay_source_document(
            source.getvalue(),
            entry={"source_document_id": "one-document.xlsx"},
            frame=pd.DataFrame([first, second]),
            parser_code_sha256=parser_hash,
            parser_config=parser_config,
            parser_config_sha256=parser_config_hash,
        )


def test_signed_catalog_binds_history_and_every_archived_source_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    timestamp_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    timestamp_public_path = tmp_path / "timestamp-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    timestamp_public_path.write_bytes(
        timestamp_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(public_path))
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
        str(timestamp_public_path),
    )
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_JOURNAL_ID",
        "test-eex-acquisition-journal",
    )
    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "eex-2026-03-01.xlsx"
    workbook = pd.DataFrame(
        [
            [None, "Y01_2027_BASE"],
            [None, "ISIN_fixture"],
            ["Date", None],
            ["01.03.2026", 80.0],
        ]
    )
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        workbook.to_excel(writer, sheet_name="CH", index=False, header=False)
    parser_dir = tmp_path / "parser"
    parser_dir.mkdir()
    parser_code = parser_dir / "ingest_forwards.py"
    parser_code.write_bytes(Path(ingest_forwards.__file__).read_bytes())
    parser_code_hash = hashlib.sha256(parser_code.read_bytes()).hexdigest()
    parser_config = {
        "parser_version": "ingest_forwards.v1",
        "selection_mode": "latest_available",
        "markets": {"CH": "2026-03-01"},
    }
    parser_config_hash = _canonical_sha(parser_config)
    row = _row("2026-03-01", "2027", 80.0)
    row["source_document_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt_unsigned: dict[str, object] = {
        "schema_version": TRUSTED_TIME_RECEIPT_SCHEMA,
        "source_document_id": row["source_document_id"],
        "source_document_sha256": row["source_document_sha256"],
        "size_bytes": source.stat().st_size,
        "received_at_utc": pd.Timestamp(row["available_at"]).isoformat(),
        "journal_id": "test-eex-acquisition-journal",
        "journal_sequence": 1,
        "previous_receipt_id": "",
    }
    receipt_unsigned["receipt_id"] = _canonical_sha(receipt_unsigned)
    signed_receipt = _sign_trusted_time_receipt_fixture(receipt_unsigned, timestamp_key)
    row["acquisition_id"] = receipt_unsigned["receipt_id"]
    row["source_product_code"] = "Y01_2027_BASE"
    row["source_column_index"] = 2
    row["parser_code_sha256"] = parser_code_hash
    row["parser_config_sha256"] = parser_config_hash
    _repin(row)
    frame = pd.DataFrame([row])
    history_path = tmp_path / "history.parquet"
    frame.to_parquet(history_path, index=False)
    history_bytes = history_path.read_bytes()
    available = pd.Timestamp(row["available_at"]).isoformat()
    unsigned: dict[str, object] = {
        "schema_version": EEX_HISTORICAL_CATALOG_SCHEMA,
        "created_at_utc": "2026-03-01T09:00:00+00:00",
        "data_cutoff_utc": available,
        "calibration_eligible": True,
        "source_class": "IMMUTABLE_DAILY_EEX_XLSX",
        "revision_policy": "APPEND_ONLY_PRESERVE_ALL_REVISIONS",
        "ompex_used": False,
        "history_parquet": {
            "path": "history.parquet",
            "sha256": hashlib.sha256(history_bytes).hexdigest(),
            "size_bytes": len(history_bytes),
            "row_count": 1,
        },
        "source_documents": [
            {
                "snapshot_ids": [row["snapshot_id"]],
                "acquisition_id": row["acquisition_id"],
                "observed_at": pd.Timestamp(row["observed_at"]).isoformat(),
                "available_at": available,
                "source_document_id": row["source_document_id"],
                "path": "sources/eex-2026-03-01.xlsx",
                "sha256": row["source_document_sha256"],
                "size_bytes": source.stat().st_size,
                "parser_code_path": "parser/ingest_forwards.py",
                "parser_code_sha256": parser_code_hash,
                "parser_config": parser_config,
                "parser_config_sha256": parser_config_hash,
                "trusted_time_receipt": signed_receipt,
            }
        ],
    }
    unsigned["catalog_id"] = _canonical_sha(unsigned)
    signed = sign_acquisition_contract(unsigned, private_key_path=private_path)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(signed), encoding="utf-8")

    actual, evidence = verify_eex_historical_vintage_catalog(
        signed,
        catalog_path=catalog_path,
        history_path=history_path,
    )

    assert evidence.catalog_id == unsigned["catalog_id"]
    assert evidence.snapshot_count == 1
    assert actual.attrs["verified_vintage_catalog"]["status"] == (
        "VERIFIED_SIGNED_IMMUTABLE_VINTAGES"
    )
    artifacts = run_verified_lambda_calibration(
        history_path,
        catalog_path,
        grid={
            "defaults": {"constraint_tolerance": 0.01},
            "grid": {
                "lambda_smooth_month": [1.0],
                "lambda_smooth_yoy": [0.0],
                "lambda_shape": [0.0],
                "neighbor_shrinkage": [0.5],
            },
        },
        smoke=True,
        max_origins=1,
        max_configs=1,
    )
    assert artifacts.manifest["verified_vintage_catalog"]["catalog_id"] == (
        unsigned["catalog_id"]
    )
    assert artifacts.manifest["production_approved"] is False
    assert artifacts.verified is True
    assert artifacts.manifest["governance_profile"] == (
        "VERIFIED_VINTAGE_SMOKE_DIAGNOSTIC_ONLY"
    )
    assert artifacts.summary["selected_config_hash"] is None
    assert artifacts.candidate_config is None
    artifacts.summary["selection_reason"] = "mutated-after-verification"
    with pytest.raises(EexHistoricalVintageError, match="changed after strict"):
        write_calibration_artifacts(artifacts, tmp_path / "mutated-artifacts")
    assert not (tmp_path / "mutated-artifacts").exists()
    weak_grid = {
        "calibration": {
            "min_valid_origins": 1,
            "min_withheld_monthly": 1,
            "min_withheld_quarterly": 0,
            "min_history_snapshots": 1,
        },
        "defaults": {"constraint_tolerance": 0.01},
        "grid": {
            "lambda_smooth_month": [1.0],
            "lambda_smooth_yoy": [0.0],
            "lambda_shape": [0.0],
            "neighbor_shrinkage": [0.5],
        },
    }
    with pytest.raises(EexHistoricalVintageError, match="weaken tier2"):
        run_verified_lambda_calibration(
            history_path,
            catalog_path,
            grid=weak_grid,
        )
    nan_grid = json.loads(json.dumps(weak_grid))
    nan_grid["calibration"]["min_valid_origins"] = float("nan")
    with pytest.raises(EexHistoricalVintageError, match="non-finite"):
        run_verified_lambda_calibration(
            history_path,
            catalog_path,
            grid=nan_grid,
        )
    with pytest.raises(EexHistoricalVintageError, match="market='DE'"):
        run_verified_lambda_calibration(
            history_path,
            catalog_path,
            grid={
                "defaults": {"constraint_tolerance": 0.01},
                "grid": {
                    "lambda_smooth_month": [1.0],
                    "lambda_smooth_yoy": [0.0],
                    "lambda_shape": [0.0],
                    "neighbor_shrinkage": [0.5],
                },
            },
            settings=LambdaCalibrationSettings(market="DE"),
        )
    monkeypatch.setattr(lambda_calibration, "_git_worktree_dirty", lambda: True)
    with pytest.raises(EexHistoricalVintageError, match="clean, committed"):
        run_verified_lambda_calibration(
            history_path,
            catalog_path,
            grid={
                "defaults": {"constraint_tolerance": 0.01},
                "grid": {
                    "lambda_smooth_month": [1.0],
                    "lambda_smooth_yoy": [0.0],
                    "lambda_shape": [0.0],
                    "neighbor_shrinkage": [0.5],
                },
            },
        )

    forged_row = dict(row)
    forged_row["price"] = 999.0
    _repin(forged_row)
    forged_history_path = tmp_path / "forged-history.parquet"
    pd.DataFrame([forged_row]).to_parquet(forged_history_path, index=False)
    forged_history_bytes = forged_history_path.read_bytes()
    forged_unsigned = json.loads(json.dumps(unsigned))
    forged_unsigned.pop("catalog_id")
    forged_unsigned["history_parquet"] = {
        "path": "forged-history.parquet",
        "sha256": hashlib.sha256(forged_history_bytes).hexdigest(),
        "size_bytes": len(forged_history_bytes),
        "row_count": 1,
    }
    forged_unsigned["catalog_id"] = _canonical_sha(forged_unsigned)
    forged_signed = sign_acquisition_contract(
        forged_unsigned,
        private_key_path=private_path,
    )
    forged_catalog_path = tmp_path / "forged-catalog.json"
    forged_catalog_path.write_text(json.dumps(forged_signed), encoding="utf-8")

    with pytest.raises(EexHistoricalVintageError, match="prices do not match"):
        verify_eex_historical_vintage_catalog(
            forged_signed,
            catalog_path=forged_catalog_path,
            history_path=forged_history_path,
        )

    source.write_bytes(b"mutated")
    with pytest.raises(EexHistoricalVintageError, match="source document receipt mismatch"):
        verify_eex_historical_vintage_catalog(
            signed,
            catalog_path=catalog_path,
            history_path=history_path,
        )


def _row(
    date: str,
    product: str,
    price: float,
    *,
    available_at: str | None = None,
) -> dict[str, object]:
    observed_at = pd.Timestamp(available_at or f"{date}T08:00:00Z")
    available = observed_at
    product_type = "QUARTER" if "-Q" in product else "MONTH" if len(product) == 7 else "CAL"
    column = 10 + int(product[-1]) if product_type == "QUARTER" else 1
    source_hash = hashlib.sha256(f"eex-{date}".encode("ascii")).hexdigest()
    row: dict[str, object] = {
        "date": pd.Timestamp(date),
        "observed_at": observed_at,
        "available_at": available,
        "acquisition_id": f"acq-{date}",
        "product": product,
        "load_type": "BASE",
        "product_type": product_type,
        "price": float(price),
        "unit": "EUR/MWH",
        "market": "CH",
        "source": "EEX",
        "source_document_id": f"eex-{date}.xlsx",
        "source_document_sha256": source_hash,
        "source_sheet": "CH",
        "source_row_index": 4,
        "source_column_index": column,
        "source_product_code": product,
        "parser_version": "ingest_forwards.v1",
        "parser_code_sha256": "a" * 64,
        "parser_config_sha256": "b" * 64,
        "revision_sequence": 1,
        "supersedes_quote_id": "",
        "revision_timestamp": observed_at,
        "ingestion_run_id": f"ingest-{date}",
        "schema_version": EEX_HISTORICAL_QUOTE_SCHEMA,
    }
    _repin(row)
    return row


def _repin(row: dict[str, object], *, keep_snapshot_id: bool = False) -> None:
    if not keep_snapshot_id:
        row["snapshot_id"] = historical_vintage_snapshot_id(row)
    row["revision_id"] = historical_vintage_revision_id(row)
    row["row_hash"] = historical_vintage_row_hash(row)
    row["quote_id"] = row["row_hash"]


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sign_trusted_time_receipt_fixture(
    unsigned: dict[str, object],
    private_key: Ed25519PrivateKey,
) -> dict[str, object]:
    payload = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    public_raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    signed = dict(unsigned)
    signed["trusted_time_attestation"] = {
        "algorithm": "Ed25519",
        "key_id": hashlib.sha256(public_raw).hexdigest(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    return signed
