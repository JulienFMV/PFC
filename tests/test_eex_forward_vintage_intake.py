from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.cli import eex_forward_vintage_builder as vintage_builder_module
from pfc_shaping.cli import governed_acquisition_builder as acquisition_builder_module
from pfc_shaping.cli.eex_forward_vintage_builder import (
    BUILD_MANIFEST_NAME,
    SIGNING_REQUEST_NAME,
    build_eex_forward_vintage,
    validate_eex_forward_vintage_signing_request,
)
from pfc_shaping.cli.eex_forward_vintage_builder import (
    main as eex_builder_main,
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
from pfc_shaping.data.acquisition_signing import (
    sign_acquisition_contract,
    sign_eex_trusted_time_receipt,
)
from pfc_shaping.data.eex_forward_vintage_intake import (
    EEX_FORWARD_VINTAGE_INTAKE_SCHEMA,
    EexForwardVintageIntakeError,
    append_eex_forward_vintage_intake,
    canonical_json_bytes,
    compute_eex_forward_intake_id,
    load_eex_forward_intake_spec,
    validate_eex_forward_intake_spec,
)
from pfc_shaping.data.eex_historical_vintage import (
    EEX_HISTORICAL_CATALOG_SCHEMA,
    verify_eex_historical_vintage_catalog,
)
from pfc_shaping.path_safety import register_process_immutable_file_bytes
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.pipeline.strict_structured_data import load_strict_json


def _write_keypair(root: Path, name: str) -> tuple[Ed25519PrivateKey, Path, Path]:
    key = Ed25519PrivateKey.generate()
    private_path = root / f"{name}-private.pem"
    public_path = root / f"{name}-public.pem"
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return key, private_path, public_path


def test_partial_explicit_trust_never_falls_back_to_ambient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRUSTED_ACQUISITION_PUBLIC_KEY_ENV, str(tmp_path / "ambient.pem"))
    monkeypatch.setenv(TRUSTED_TIME_PUBLIC_KEY_ENV, str(tmp_path / "ambient-time.pem"))
    monkeypatch.setenv(TRUSTED_TIME_JOURNAL_ID_ENV, "ambient-journal")

    with pytest.raises(
        AcquisitionAuthenticationError,
        match="explicit acquisition trust requires trusted_public_key_path",
    ):
        verify_acquisition_contract({}, trusted_public_key_dir=tmp_path)
    with pytest.raises(
        AcquisitionAuthenticationError,
        match="explicit trusted-time trust requires trusted_public_key_path",
    ):
        verify_trusted_time_receipt({}, trusted_public_key_dir=tmp_path)
    with pytest.raises(
        AcquisitionAuthenticationError,
        match="explicit trusted-time trust requires trusted_public_key_path",
    ):
        verify_trusted_time_receipt({}, trusted_journal_id="explicit-journal")


def _write_workbook(
    path: Path,
    *,
    cal_base: float = 80.0,
    cal_peak: float = 92.0,
    quarter_base: float = 77.0,
    duplicate_cal_base: bool = False,
    later_invalid_row: bool = False,
    unknown_quoted_product: bool = False,
    include_quarter: bool = True,
) -> None:
    codes: list[object] = [
        None,
        "Y01_2027_BASE",
        "Y01_2027_PEAK",
        "Q01_2027_BASE",
    ]
    values: list[object] = ["29.07.2026", cal_base, cal_peak, quarter_base]
    if not include_quarter:
        codes.pop()
        values.pop()
    if duplicate_cal_base:
        codes.append("Y01_2027_BASE")
        values.append(cal_base)
    if unknown_quoted_product:
        codes.append("SEASON_2027_BASE")
        values.append(73.0)
    rows = [
        codes,
        [None, "ISIN_BASE", "ISIN_PEAK", "ISIN_Q1", None, None][: len(codes)],
        ["Date", *([None] * (len(codes) - 1))],
        values,
    ]
    if later_invalid_row:
        rows.append(["30.07.2026", 9_999_999.0, 0.0, 0.0])
    workbook = pd.DataFrame(
        rows
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        workbook.to_excel(writer, sheet_name="CH", index=False, header=False)


def _receipt(
    *,
    source: Path,
    private_key_path: Path,
    received_at: str,
    sequence: int = 1,
    previous_receipt_id: str = "",
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": TRUSTED_TIME_RECEIPT_SCHEMA,
        "source_document_id": source.name,
        "source_document_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
        "received_at_utc": received_at,
        "journal_id": "eex-prospective-test-journal",
        "journal_sequence": sequence,
        "previous_receipt_id": previous_receipt_id,
    }
    unsigned["receipt_id"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return sign_eex_trusted_time_receipt(
        unsigned,
        private_key_path=private_key_path,
    )


def _expected_quotes() -> list[dict[str, str]]:
    return [
        {
            "market": "CH",
            "source_product_code": "Y01_2027_BASE",
            "product": "2027",
            "load_type": "BASE",
            "product_type": "CAL",
        },
        {
            "market": "CH",
            "source_product_code": "Y01_2027_PEAK",
            "product": "2027",
            "load_type": "PEAK",
            "product_type": "CAL",
        },
        {
            "market": "CH",
            "source_product_code": "Q01_2027_BASE",
            "product": "2027-Q1",
            "load_type": "BASE",
            "product_type": "QUARTER",
        },
    ]


def _write_spec(
    path: Path,
    *,
    source: Path,
    receipt_path: Path,
    expected_quotes: list[dict[str, str]] | None = None,
) -> str:
    spec: dict[str, object] = {
        "schema_version": EEX_FORWARD_VINTAGE_INTAKE_SCHEMA,
        "source_role": "eex_forward_source",
        "source_system": "EEX_MARKET_DATA",
        "source_class": "IMMUTABLE_DAILY_EEX_XLSX",
        "source_document_id": source.name,
        "source_document_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_document_size_bytes": source.stat().st_size,
        "trusted_time_receipt_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "markets": ["CH"],
        "market_snapshot_dates": {"CH": "2026-07-29"},
        "expected_quotes": expected_quotes or _expected_quotes(),
        "quote_convention": "EEX_SETTLEMENT_EUR_MWH",
        "unit": "EUR/MWH",
        "parser_version": "ingest_forwards.v1",
        "selection_mode": "latest_available",
        "revision_policy": "APPEND_ONLY_PRESERVE_ALL_REVISIONS",
        "monthly_level_authority": (
            "SOLVER_WITH_HARD_GOVERNED_CH_EEX_FORWARD_QUOTES"
        ),
        "ompex_used": False,
        "external_catalog_signature_required": True,
        "external_cas_admission_required": True,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    spec["intake_id"] = compute_eex_forward_intake_id(spec)
    payload = canonical_json_bytes(spec)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, str, Path, Path]:
    _, timestamp_private, timestamp_public = _write_keypair(tmp_path, "time")
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH", str(timestamp_public)
    )
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_JOURNAL_ID", "eex-prospective-test-journal"
    )
    source = tmp_path / "eex-2026-07-29.xlsx"
    _write_workbook(source)
    signed_receipt = _receipt(
        source=source,
        private_key_path=timestamp_private,
        received_at="2026-07-29T16:05:00Z",
    )
    receipt_path = tmp_path / "trusted-time.json"
    receipt_path.write_bytes(canonical_json_bytes(signed_receipt))
    spec_path = tmp_path / "intake-spec.json"
    spec_hash = _write_spec(
        spec_path,
        source=source,
        receipt_path=receipt_path,
    )
    return source, receipt_path, spec_hash, spec_path, timestamp_private


def _write_signed_catalog(
    root: Path,
    *,
    source: Path,
    history: pd.DataFrame,
    result: object,
    acquisition_private: Path,
    created_at_utc: str,
) -> tuple[Path, Path]:
    sources = root / "sources"
    parser = root / "parser"
    sources.mkdir(exist_ok=True)
    parser.mkdir(exist_ok=True)
    archived_source = sources / source.name
    archived_source.write_bytes(source.read_bytes())
    archived_parser = parser / "ingest_forwards.py"
    archived_parser.write_bytes(Path(ingest_forwards.__file__).read_bytes())
    history_path = root / "history.parquet"
    history.to_parquet(history_path, index=False)
    history_payload = history_path.read_bytes()
    source_entry = result.source_catalog_entry(
        source_document_path=f"sources/{source.name}",
        parser_code_path="parser/ingest_forwards.py",
    )
    unsigned_catalog: dict[str, object] = {
        "schema_version": EEX_HISTORICAL_CATALOG_SCHEMA,
        "created_at_utc": created_at_utc,
        "data_cutoff_utc": result.available_at_utc,
        "calibration_eligible": True,
        "source_class": "IMMUTABLE_DAILY_EEX_XLSX",
        "revision_policy": "APPEND_ONLY_PRESERVE_ALL_REVISIONS",
        "ompex_used": False,
        "history_parquet": {
            "path": "history.parquet",
            "sha256": hashlib.sha256(history_payload).hexdigest(),
            "size_bytes": len(history_payload),
            "row_count": len(history),
        },
        "source_documents": [source_entry],
    }
    unsigned_catalog["catalog_id"] = hashlib.sha256(
        canonical_json_bytes(unsigned_catalog)
    ).hexdigest()
    signed_catalog = sign_acquisition_contract(
        unsigned_catalog,
        private_key_path=acquisition_private,
    )
    catalog_path = root / "catalog.json"
    catalog_path.write_bytes(canonical_json_bytes(signed_catalog))
    return catalog_path, history_path


def _sign_builder_request(
    request_bundle: Path,
    signed_bundle: Path,
    *,
    acquisition_private: Path,
) -> tuple[Path, Path]:
    signed_bundle.mkdir()
    request = load_strict_json(
        (request_bundle / SIGNING_REQUEST_NAME).read_text(encoding="utf-8")
    )
    assert isinstance(request, dict)
    proposed = request["proposed_unsigned_catalog"]
    assert isinstance(proposed, dict)
    for entry in request_bundle.iterdir():
        if entry.name.startswith(("source-", "parser-")) or entry.name == "history.parquet":
            (signed_bundle / entry.name).write_bytes(entry.read_bytes())
    signed = sign_acquisition_contract(
        proposed,
        private_key_path=acquisition_private,
    )
    catalog_path = signed_bundle / "catalog.json"
    catalog_path.write_bytes(canonical_json_bytes(signed))
    return catalog_path, signed_bundle / "history.parquet"


def test_durable_builder_emits_idempotent_unsigned_external_authority_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "eex-vintage-request"

    manifest_path = build_eex_forward_vintage(
        intake_spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
        output_directory=output,
        trusted_time_public_key_path=tmp_path / "time-public.pem",
        trusted_time_journal_id="eex-prospective-test-journal",
    )
    first_inventory = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir()
    }
    replayed = build_eex_forward_vintage(
        intake_spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
        output_directory=output,
        trusted_time_public_key_path=tmp_path / "time-public.pem",
        trusted_time_journal_id="eex-prospective-test-journal",
    )

    assert manifest_path == replayed == output / BUILD_MANIFEST_NAME
    assert first_inventory == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir()
    }
    manifest = load_strict_json(manifest_path.read_text(encoding="utf-8"))
    request = load_strict_json(
        (output / SIGNING_REQUEST_NAME).read_text(encoding="utf-8")
    )
    assert isinstance(manifest, dict)
    assert isinstance(request, dict)
    assert manifest["signing_status"].startswith("UNSIGNED_")
    assert manifest["cas_status"].startswith("NOT_ADMITTED_")
    assert manifest["calibration_eligible"] is False
    assert manifest["production_authorization"] is False
    assert request["local_authority"] == {
        "calibration_eligible": False,
        "catalog_signed": False,
        "external_cas_admitted": False,
        "production_authorization": False,
        "promotion_gate": False,
    }

    _, acquisition_private, acquisition_public = _write_keypair(tmp_path, "builder-acq")
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(acquisition_public)
    )
    catalog_path, history_path = _sign_builder_request(
        output,
        tmp_path / "signed-vintage",
        acquisition_private=acquisition_private,
    )
    catalog = load_strict_json(catalog_path.read_text(encoding="utf-8"))
    assert isinstance(catalog, dict)
    history, evidence = verify_eex_historical_vintage_catalog(
        catalog,
        catalog_path=catalog_path,
        history_path=history_path,
    )
    assert evidence.verified is True
    assert len(history) == 3


def test_durable_builder_uses_explicit_trust_without_ambient_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    _, _, wrong_public = _write_keypair(tmp_path, "wrong-time")
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH", str(wrong_public))
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_JOURNAL_ID", "wrong-journal")

    manifest = build_eex_forward_vintage(
        intake_spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
        trusted_time_public_key_path=tmp_path / "time-public.pem",
        trusted_time_journal_id="eex-prospective-test-journal",
        output_directory=tmp_path / "explicit-trust-request",
    )

    assert manifest.is_file()
    assert (
        os.environ["PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH"]
        == str(wrong_public)
    )
    assert os.environ["PFC_DATA_TIMESTAMP_JOURNAL_ID"] == "wrong-journal"


def test_explicit_trust_anchor_is_aba_safe_under_concurrent_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, _, _, _ = _prepare(tmp_path, monkeypatch)
    trust_path = tmp_path / "time-public.pem"
    frozen_bytes = trust_path.read_bytes()
    register_process_immutable_file_bytes(
        trust_path,
        frozen_bytes,
        label="test trusted-time public key",
    )
    _, _, replacement_path = _write_keypair(tmp_path, "replacement-time")
    replacement_bytes = replacement_path.read_bytes()
    trust_path.write_bytes(replacement_bytes)
    receipt = load_strict_json(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    barrier = Barrier(2)

    def verify_frozen_anchor() -> dict[str, object]:
        barrier.wait(timeout=10)
        return verify_trusted_time_receipt(
            receipt,
            trusted_public_key_path=trust_path,
            trusted_journal_id="eex-prospective-test-journal",
        )

    def attempt_rebind() -> Path:
        barrier.wait(timeout=10)
        return register_process_immutable_file_bytes(
            trust_path,
            replacement_bytes,
            label="replacement trusted-time public key",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        verify_future = pool.submit(verify_frozen_anchor)
        rebind_future = pool.submit(attempt_rebind)
        assert verify_future.result()["journal_id"] == "eex-prospective-test-journal"
        with pytest.raises(ValueError, match="immutable process binding already differs"):
            rebind_future.result()


def test_eex_builder_main_passes_explicit_runtime_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def reject_after_capture(**kwargs: object) -> None:
        observed.update(kwargs)
        raise ReleaseCliIdentityError("simulated unsealed runtime")

    monkeypatch.setattr(
        vintage_builder_module,
        "assert_installed_runtime_sealed",
        reject_after_capture,
    )
    runtime_receipt = tmp_path / "runtime-receipt.json"
    result = eex_builder_main(
        [
            "--runtime-receipt",
            str(runtime_receipt),
            "--expected-runtime-receipt-sha256",
            "a" * 64,
            "--intake-spec",
            str(tmp_path / "spec.json"),
            "--expected-spec-sha256",
            "b" * 64,
            "--source-document",
            str(tmp_path / "source.xlsx"),
            "--trusted-time-receipt",
            str(tmp_path / "time.json"),
            "--trusted-time-public-key",
            str(tmp_path / "time.pem"),
            "--trusted-time-journal-id",
            "journal-v1",
            "--output-directory",
            str(tmp_path / "output"),
        ]
    )

    assert result == 50
    assert observed == {
        "runtime_receipt_path": runtime_receipt,
        "expected_runtime_receipt_sha256": "a" * 64,
    }


def test_durable_builder_rejects_divergent_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "divergent-vintage-request"
    build_eex_forward_vintage(
        intake_spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
        output_directory=output,
        trusted_time_public_key_path=tmp_path / "time-public.pem",
        trusted_time_journal_id="eex-prospective-test-journal",
    )
    (output / "intake-spec.json").write_bytes(b"divergent")

    with pytest.raises(ValueError, match="explicit repair"):
        build_eex_forward_vintage(
            intake_spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
            output_directory=output,
            trusted_time_public_key_path=tmp_path / "time-public.pem",
            trusted_time_journal_id="eex-prospective-test-journal",
        )


def test_durable_builder_carries_verified_history_and_source_chain_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, timestamp_private = _prepare(
        tmp_path, monkeypatch
    )
    first_history, first_result = append_eex_forward_vintage_intake(
        spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
    )
    _, acquisition_private, acquisition_public = _write_keypair(
        tmp_path, "chain-acq"
    )
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(acquisition_public)
    )
    first_signed = tmp_path / "chain-first-signed"
    first_signed.mkdir()
    first_catalog, first_history_path = _write_signed_catalog(
        first_signed,
        source=source,
        history=first_history,
        result=first_result,
        acquisition_private=acquisition_private,
        created_at_utc="2026-07-29T16:06:00+00:00",
    )

    correction = tmp_path / "eex-2026-07-29-correction.xlsx"
    _write_workbook(correction, cal_base=81.0)
    correction_receipt = _receipt(
        source=correction,
        private_key_path=timestamp_private,
        received_at="2026-07-29T17:05:00Z",
        sequence=2,
        previous_receipt_id=first_result.acquisition_id,
    )
    correction_receipt_path = tmp_path / "trusted-time-correction.json"
    correction_receipt_path.write_bytes(canonical_json_bytes(correction_receipt))
    correction_spec_path = tmp_path / "intake-spec-correction.json"
    correction_spec_hash = _write_spec(
        correction_spec_path,
        source=correction,
        receipt_path=correction_receipt_path,
    )
    output = tmp_path / "chain-second-request"

    build_eex_forward_vintage(
        intake_spec_path=correction_spec_path,
        expected_spec_sha256=correction_spec_hash,
        source_document_path=correction,
        trusted_time_receipt_path=correction_receipt_path,
        output_directory=output,
        previous_catalog_path=first_catalog,
        previous_history_path=first_history_path,
        trusted_time_public_key_path=tmp_path / "time-public.pem",
        trusted_time_journal_id="eex-prospective-test-journal",
        previous_catalog_public_key_path=acquisition_public,
    )

    request = load_strict_json(
        (output / SIGNING_REQUEST_NAME).read_text(encoding="utf-8")
    )
    assert isinstance(request, dict)
    proposed = request["proposed_unsigned_catalog"]
    assert isinstance(proposed, dict)
    assert len(proposed["source_documents"]) == 2
    assert proposed["history_parquet"]["row_count"] == 6
    assert request["previous_catalog"]["catalog_id"]
    assert request["verification_trust"]["previous_catalog_public_key_sha256"] == (
        hashlib.sha256(acquisition_public.read_bytes()).hexdigest()
    )

    signed_catalog, signed_history = _sign_builder_request(
        output,
        tmp_path / "chain-second-signed",
        acquisition_private=acquisition_private,
    )
    catalog = load_strict_json(signed_catalog.read_text(encoding="utf-8"))
    assert isinstance(catalog, dict)
    combined, evidence = verify_eex_historical_vintage_catalog(
        catalog,
        catalog_path=signed_catalog,
        history_path=signed_history,
    )
    assert evidence.source_document_count == 2
    assert len(combined) == 6
    assert combined["revision_sequence"].value_counts().to_dict() == {1: 3, 2: 3}


def test_durable_builder_resumes_exact_staging_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "resumable-vintage-request"
    original_write = acquisition_builder_module._write_new
    writes = 0

    def interrupt(path: Path, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 4:
            raise OSError("simulated EEX builder interruption")
        original_write(path, payload)

    monkeypatch.setattr(acquisition_builder_module, "_write_new", interrupt)
    with pytest.raises(OSError, match="simulated EEX builder interruption"):
        build_eex_forward_vintage(
            intake_spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
            output_directory=output,
            trusted_time_public_key_path=tmp_path / "time-public.pem",
            trusted_time_journal_id="eex-prospective-test-journal",
        )
    assert not output.exists()
    assert len(list(tmp_path.glob(".resumable-vintage-request.staging-*"))) == 1

    monkeypatch.setattr(acquisition_builder_module, "_write_new", original_write)
    manifest = build_eex_forward_vintage(
        intake_spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
        output_directory=output,
        trusted_time_public_key_path=tmp_path / "time-public.pem",
        trusted_time_journal_id="eex-prospective-test-journal",
    )
    assert manifest.is_file()
    assert not list(tmp_path.glob(".resumable-vintage-request.staging-*"))


def test_durable_builder_concurrent_exact_calls_never_diverge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "concurrent-vintage-request"
    original_materialize = vintage_builder_module._materialize_output
    barrier = Barrier(2)

    def simultaneous_materialize(**kwargs: object) -> Path:
        barrier.wait(timeout=10)
        return original_materialize(**kwargs)

    monkeypatch.setattr(
        vintage_builder_module,
        "_materialize_output",
        simultaneous_materialize,
    )

    def run() -> Path:
        return build_eex_forward_vintage(
            intake_spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
            output_directory=output,
            trusted_time_public_key_path=tmp_path / "time-public.pem",
            trusted_time_journal_id="eex-prospective-test-journal",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results: list[Path] = []
        failures: list[BaseException] = []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:  # pragma: no cover - platform race outcome
                failures.append(exc)
    assert results
    assert all(isinstance(exc, (OSError, ValueError)) for exc in failures)
    monkeypatch.setattr(
        vintage_builder_module,
        "_materialize_output",
        original_materialize,
    )
    replay = run()
    assert replay == output / BUILD_MANIFEST_NAME
    manifest = load_strict_json(replay.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED_UNSIGNED_EXTERNAL_AUTHORITY_HANDOFF"


def test_signing_request_validator_rejects_local_authority_smuggling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "tamper-vintage-request"
    build_eex_forward_vintage(
        intake_spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
        output_directory=output,
        trusted_time_public_key_path=tmp_path / "time-public.pem",
        trusted_time_journal_id="eex-prospective-test-journal",
    )
    request = load_strict_json(
        (output / SIGNING_REQUEST_NAME).read_text(encoding="utf-8")
    )
    assert isinstance(request, dict)
    request["local_authority"]["production_authorization"] = True
    artifacts = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.name not in {SIGNING_REQUEST_NAME, BUILD_MANIFEST_NAME}
    }

    with pytest.raises(ValueError, match="request_id|authority"):
        validate_eex_forward_vintage_signing_request(request, artifacts=artifacts)


def test_prospective_eex_intake_builds_catalog_compatible_vintage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)

    history, result = append_eex_forward_vintage_intake(
        spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
    )

    observed_quotes = history[["product", "load_type", "price"]].sort_values(
        ["product", "load_type"], kind="mergesort"
    )
    assert observed_quotes.to_dict(orient="records") == [
        {"product": "2027", "load_type": "BASE", "price": 80.0},
        {"product": "2027", "load_type": "PEAK", "price": 92.0},
        {"product": "2027-Q1", "load_type": "BASE", "price": 77.0},
    ]
    assert result.new_row_count == 3
    assert result.calibration_eligible is False
    assert result.external_cas_admitted is False
    assert result.production_authorization is False
    assert history["available_at"].eq(pd.Timestamp("2026-07-29T16:05:00Z")).all()

    _, acquisition_private, acquisition_public = _write_keypair(tmp_path, "acq")
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(acquisition_public)
    )
    catalog_path, history_path = _write_signed_catalog(
        tmp_path,
        source=source,
        history=history,
        result=result,
        acquisition_private=acquisition_private,
        created_at_utc="2026-07-29T16:06:00+00:00",
    )
    signed_catalog = load_strict_json(catalog_path.read_text(encoding="utf-8"))
    assert isinstance(signed_catalog, dict)

    verified, evidence = verify_eex_historical_vintage_catalog(
        signed_catalog,
        catalog_path=catalog_path,
        history_path=history_path,
    )

    assert evidence.catalog_id == signed_catalog["catalog_id"]
    assert len(verified) == 3
    assert evidence.source_document_count == 1


def test_intake_rejects_changed_provider_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(EexForwardVintageIntakeError, match="document binding mismatch"):
        append_eex_forward_vintage_intake(
            spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
        )


def test_intake_rejects_quote_inventory_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, _, spec_path, _ = _prepare(tmp_path, monkeypatch)
    spec_hash = _write_spec(
        spec_path,
        source=source,
        receipt_path=receipt_path,
        expected_quotes=_expected_quotes()[:-1],
    )

    with pytest.raises(EexForwardVintageIntakeError, match="quote inventory"):
        append_eex_forward_vintage_intake(
            spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
        )


def test_intake_appends_same_date_correction_as_linked_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, timestamp_private = _prepare(
        tmp_path, monkeypatch
    )
    first, first_result = append_eex_forward_vintage_intake(
        spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
    )
    _, acquisition_private, acquisition_public = _write_keypair(tmp_path, "revision-acq")
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(acquisition_public)
    )
    first_bundle = tmp_path / "first-bundle"
    first_bundle.mkdir()
    first_catalog_path, first_history_path = _write_signed_catalog(
        first_bundle,
        source=source,
        history=first,
        result=first_result,
        acquisition_private=acquisition_private,
        created_at_utc="2026-07-29T16:06:00+00:00",
    )
    second_source = tmp_path / "eex-2026-07-29-correction.xlsx"
    _write_workbook(second_source, cal_base=81.0)
    second_receipt = _receipt(
        source=second_source,
        private_key_path=timestamp_private,
        received_at="2026-07-29T17:05:00Z",
        sequence=2,
        previous_receipt_id=first_result.acquisition_id,
    )
    second_receipt_path = tmp_path / "trusted-time-correction.json"
    second_receipt_path.write_bytes(canonical_json_bytes(second_receipt))
    second_spec_path = tmp_path / "intake-spec-correction.json"
    second_spec_hash = _write_spec(
        second_spec_path,
        source=second_source,
        receipt_path=second_receipt_path,
    )

    combined, second_result = append_eex_forward_vintage_intake(
        spec_path=second_spec_path,
        expected_spec_sha256=second_spec_hash,
        source_document_path=second_source,
        trusted_time_receipt_path=second_receipt_path,
        previous_catalog_path=first_catalog_path,
        previous_history_path=first_history_path,
    )

    corrected = combined[combined["acquisition_id"] == second_result.acquisition_id]
    assert len(combined) == 6
    assert corrected["revision_sequence"].eq(2).all()
    assert corrected["supersedes_quote_id"].isin(first["quote_id"]).all()
    assert corrected.loc[
        (corrected["product"] == "2027") & (corrected["load_type"] == "BASE"),
        "price",
    ].item() == 81.0


def test_intake_rejects_same_date_inventory_change_without_tombstones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, timestamp_private = _prepare(
        tmp_path, monkeypatch
    )
    first, first_result = append_eex_forward_vintage_intake(
        spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
    )
    _, acquisition_private, acquisition_public = _write_keypair(tmp_path, "omit-acq")
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(acquisition_public)
    )
    first_bundle = tmp_path / "omit-first-bundle"
    first_bundle.mkdir()
    first_catalog_path, first_history_path = _write_signed_catalog(
        first_bundle,
        source=source,
        history=first,
        result=first_result,
        acquisition_private=acquisition_private,
        created_at_utc="2026-07-29T16:06:00+00:00",
    )
    second_source = tmp_path / "eex-2026-07-29-omission.xlsx"
    _write_workbook(second_source, include_quarter=False)
    second_receipt = _receipt(
        source=second_source,
        private_key_path=timestamp_private,
        received_at="2026-07-29T17:05:00Z",
        sequence=2,
        previous_receipt_id=first_result.acquisition_id,
    )
    second_receipt_path = tmp_path / "trusted-time-omission.json"
    second_receipt_path.write_bytes(canonical_json_bytes(second_receipt))
    second_spec_path = tmp_path / "intake-spec-omission.json"
    second_spec_hash = _write_spec(
        second_spec_path,
        source=second_source,
        receipt_path=second_receipt_path,
        expected_quotes=_expected_quotes()[:-1],
    )

    with pytest.raises(EexForwardVintageIntakeError, match="tombstones"):
        append_eex_forward_vintage_intake(
            spec_path=second_spec_path,
            expected_spec_sha256=second_spec_hash,
            source_document_path=second_source,
            trusted_time_receipt_path=second_receipt_path,
            previous_catalog_path=first_catalog_path,
            previous_history_path=first_history_path,
        )


def test_initial_intake_rejects_non_genesis_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, _, spec_path, timestamp_private = _prepare(
        tmp_path, monkeypatch
    )
    receipt = _receipt(
        source=source,
        private_key_path=timestamp_private,
        received_at="2026-07-29T16:05:00Z",
        sequence=2,
        previous_receipt_id="a" * 64,
    )
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    spec_hash = _write_spec(spec_path, source=source, receipt_path=receipt_path)

    with pytest.raises(EexForwardVintageIntakeError, match="journal genesis"):
        append_eex_forward_vintage_intake(
            spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
        )


def test_intake_rejects_journal_gap_against_verified_previous_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, timestamp_private = _prepare(
        tmp_path, monkeypatch
    )
    first, first_result = append_eex_forward_vintage_intake(
        spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
    )
    _, acquisition_private, acquisition_public = _write_keypair(tmp_path, "gap-acq")
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(acquisition_public)
    )
    first_bundle = tmp_path / "gap-first-bundle"
    first_bundle.mkdir()
    first_catalog_path, first_history_path = _write_signed_catalog(
        first_bundle,
        source=source,
        history=first,
        result=first_result,
        acquisition_private=acquisition_private,
        created_at_utc="2026-07-29T16:06:00+00:00",
    )
    second_source = tmp_path / "eex-2026-07-29-gap.xlsx"
    _write_workbook(second_source, cal_base=81.0)
    second_receipt = _receipt(
        source=second_source,
        private_key_path=timestamp_private,
        received_at="2026-07-29T17:05:00Z",
        sequence=3,
        previous_receipt_id=first_result.acquisition_id,
    )
    second_receipt_path = tmp_path / "trusted-time-gap.json"
    second_receipt_path.write_bytes(canonical_json_bytes(second_receipt))
    second_spec_path = tmp_path / "intake-spec-gap.json"
    second_spec_hash = _write_spec(
        second_spec_path,
        source=second_source,
        receipt_path=second_receipt_path,
    )

    with pytest.raises(EexForwardVintageIntakeError, match="does not extend"):
        append_eex_forward_vintage_intake(
            spec_path=second_spec_path,
            expected_spec_sha256=second_spec_hash,
            source_document_path=second_source,
            trusted_time_receipt_path=second_receipt_path,
            previous_catalog_path=first_catalog_path,
            previous_history_path=first_history_path,
        )


def test_signed_catalog_rejects_injected_production_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, receipt_path, spec_hash, spec_path, _ = _prepare(tmp_path, monkeypatch)
    history, result = append_eex_forward_vintage_intake(
        spec_path=spec_path,
        expected_spec_sha256=spec_hash,
        source_document_path=source,
        trusted_time_receipt_path=receipt_path,
    )
    _, acquisition_private, acquisition_public = _write_keypair(
        tmp_path, "smuggling-acq"
    )
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH", str(acquisition_public)
    )
    bundle = tmp_path / "smuggling-bundle"
    bundle.mkdir()
    catalog_path, history_path = _write_signed_catalog(
        bundle,
        source=source,
        history=history,
        result=result,
        acquisition_private=acquisition_private,
        created_at_utc="2026-07-29T16:06:00+00:00",
    )
    signed_catalog = load_strict_json(catalog_path.read_text(encoding="utf-8"))
    assert isinstance(signed_catalog, dict)
    smuggled = dict(signed_catalog)
    smuggled.pop("acquisition_attestation")
    smuggled.pop("catalog_id")
    smuggled["production_authorization"] = True
    smuggled["catalog_id"] = hashlib.sha256(canonical_json_bytes(smuggled)).hexdigest()
    resigned = sign_acquisition_contract(
        smuggled,
        private_key_path=acquisition_private,
    )
    catalog_path.write_bytes(canonical_json_bytes(resigned))

    with pytest.raises(ValueError, match="fields are not exact"):
        verify_eex_historical_vintage_catalog(
            resigned,
            catalog_path=catalog_path,
            history_path=history_path,
        )


def test_spec_rejects_integer_authority_claim() -> None:
    spec = {
        "schema_version": EEX_FORWARD_VINTAGE_INTAKE_SCHEMA,
        "intake_id": "",
        "source_role": "eex_forward_source",
        "source_system": "EEX_MARKET_DATA",
        "source_class": "IMMUTABLE_DAILY_EEX_XLSX",
        "source_document_id": "fixture.xlsx",
        "source_document_sha256": "a" * 64,
        "source_document_size_bytes": 1,
        "trusted_time_receipt_sha256": "b" * 64,
        "markets": ["CH"],
        "market_snapshot_dates": {"CH": "2026-07-29"},
        "expected_quotes": [_expected_quotes()[0]],
        "quote_convention": "EEX_SETTLEMENT_EUR_MWH",
        "unit": "EUR/MWH",
        "parser_version": "ingest_forwards.v1",
        "selection_mode": "latest_available",
        "revision_policy": "APPEND_ONLY_PRESERVE_ALL_REVISIONS",
        "monthly_level_authority": "SOLVER_WITH_HARD_GOVERNED_CH_EEX_FORWARD_QUOTES",
        "ompex_used": False,
        "external_catalog_signature_required": True,
        "external_cas_admission_required": True,
        "scientific_admission": False,
        "production_authorization": 0,
        "promotion_gate": False,
    }
    spec["intake_id"] = compute_eex_forward_intake_id(spec)

    with pytest.raises(EexForwardVintageIntakeError, match="production_authorization"):
        validate_eex_forward_intake_spec(spec)


def test_oversized_spec_is_rejected_before_json_parse(tmp_path: Path) -> None:
    spec_path = tmp_path / "oversized-spec.json"
    spec_path.write_bytes(b"{" + b" " * (1024 * 1024) + b"}")

    with pytest.raises(EexForwardVintageIntakeError, match="size is invalid"):
        load_eex_forward_intake_spec(
            spec_path,
            expected_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        )


def test_intake_rejects_ambiguous_zero_quote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, timestamp_private, timestamp_public = _write_keypair(tmp_path, "zero-time")
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH", str(timestamp_public)
    )
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_JOURNAL_ID", "eex-prospective-test-journal"
    )
    source = tmp_path / "eex-zero.xlsx"
    _write_workbook(source, cal_base=0.0)
    receipt = _receipt(
        source=source,
        private_key_path=timestamp_private,
        received_at="2026-07-29T16:05:00Z",
    )
    receipt_path = tmp_path / "zero-receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    spec_path = tmp_path / "zero-spec.json"
    spec_hash = _write_spec(spec_path, source=source, receipt_path=receipt_path)

    with pytest.raises(EexForwardVintageIntakeError, match="invalid quoted price"):
        append_eex_forward_vintage_intake(
            spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
        )


def test_identical_duplicate_product_columns_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.xlsx"
    _write_workbook(source, duplicate_cal_base=True)

    with pytest.raises(ValueError, match="QUOTE_CONFLICT duplicate product"):
        ingest_forwards.load_base_prices_from_eex_report_bytes(
            source.read_bytes(),
            market="CH",
        )


def test_prospective_intake_rejects_fallback_from_later_invalid_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, timestamp_private, timestamp_public = _write_keypair(tmp_path, "fallback-time")
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH", str(timestamp_public)
    )
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_JOURNAL_ID", "eex-prospective-test-journal"
    )
    source = tmp_path / "fallback.xlsx"
    _write_workbook(source, later_invalid_row=True)
    signed_receipt = _receipt(
        source=source,
        private_key_path=timestamp_private,
        received_at="2026-07-30T16:05:00Z",
    )
    receipt_path = tmp_path / "fallback-time.json"
    receipt_path.write_bytes(canonical_json_bytes(signed_receipt))
    spec_path = tmp_path / "fallback-spec.json"
    spec_hash = _write_spec(
        spec_path,
        source=source,
        receipt_path=receipt_path,
    )

    with pytest.raises(EexForwardVintageIntakeError, match="latest physical row"):
        append_eex_forward_vintage_intake(
            spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
        )


def test_prospective_intake_rejects_unknown_quoted_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, timestamp_private, timestamp_public = _write_keypair(tmp_path, "unknown-time")
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH", str(timestamp_public)
    )
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_JOURNAL_ID", "eex-prospective-test-journal"
    )
    source = tmp_path / "unknown.xlsx"
    _write_workbook(source, unknown_quoted_product=True)
    signed_receipt = _receipt(
        source=source,
        private_key_path=timestamp_private,
        received_at="2026-07-29T16:05:00Z",
    )
    receipt_path = tmp_path / "unknown-time.json"
    receipt_path.write_bytes(canonical_json_bytes(signed_receipt))
    spec_path = tmp_path / "unknown-spec.json"
    spec_hash = _write_spec(
        spec_path,
        source=source,
        receipt_path=receipt_path,
    )

    with pytest.raises(EexForwardVintageIntakeError, match="unknown quoted product"):
        append_eex_forward_vintage_intake(
            spec_path=spec_path,
            expected_spec_sha256=spec_hash,
            source_document_path=source,
            trusted_time_receipt_path=receipt_path,
        )
