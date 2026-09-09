from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from pfc_shaping.data import eex_datasource_v2_capture as capture_module
from pfc_shaping.data.eex_datasource_v2_capture import (
    CAPTURE_STATUS,
    SPEC_SCHEMA,
    SPEC_STATUS,
    TOKEN_ENV,
    EexDatasourceV2CaptureError,
    HttpsExchange,
    canonical_json_bytes,
    capture_eex_datasource_v2,
    compute_spec_id,
    load_eex_datasource_v2_capture_spec,
)

_ISIN = "CH0000000001"


def _test_ca_bytes() -> bytes:
    key = Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PFC LT test root")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime(2026, 1, 1, tzinfo=timezone.utc))
        .not_valid_after(datetime(2027, 1, 1, tzinfo=timezone.utc))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .sign(key, algorithm=None)
    )
    return certificate.public_bytes(serialization.Encoding.PEM)


_CA_BYTES = _test_ca_bytes()
_CA_SHA256 = hashlib.sha256(_CA_BYTES).hexdigest()


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "build" / "docs").mkdir(parents=True)
    (root / ".planning").mkdir()
    guide = root / "build" / "docs" / "guide.pdf"
    openapi = root / "build" / "docs" / "openapi.yaml"
    ca_bundle = root / "build" / "docs" / "ca.pem"
    guide.write_bytes(b"official-guide-fixture")
    openapi.write_bytes(b"openapi: 3.0.3\n")
    ca_bundle.write_bytes(_CA_BYTES)
    guide_hash = hashlib.sha256(guide.read_bytes()).hexdigest()
    openapi_hash = hashlib.sha256(openapi.read_bytes()).hexdigest()
    monkeypatch.setattr(capture_module, "GUIDE_SHA256", guide_hash)
    monkeypatch.setattr(
        capture_module,
        "APPROVED_TLS_CA_SHA256S",
        frozenset({_CA_SHA256}),
    )
    return root, guide_hash, openapi_hash


def _spec_document(
    *,
    capture_id: str,
    guide_hash: str,
    openapi_hash: str,
    role: str = "REFERENCE_DATA",
    reference_capture: dict[str, str] | None = None,
) -> dict[str, object]:
    endpoint_prefix = "rd" if role == "REFERENCE_DATA" else "spr"
    document: dict[str, object] = {
        "schema_version": SPEC_SCHEMA,
        "spec_id": "0" * 64,
        "status": SPEC_STATUS,
        "capture_id": capture_id,
        "planned_at_utc": "2026-07-31T09:55:00Z",
        "request": {
            "method": "GET",
            "api_root": "https://api.eex-group.com/v2",
            "api_version": "v2",
            "data_role": role,
            "endpoint_path": (
                f"/{endpoint_prefix}/derivatives/POWER/CH/2026-07-31/CHBY/202701"
            ),
            "query": [],
            "accept": "application/json",
            "commodity": "POWER",
            "area": "CH",
            "rolling_symbols_allowed": False,
        },
        "documentation": {
            "user_guide_release": "006",
            "user_guide_published_date": "2026-07-16",
            "user_guide_path": "build/docs/guide.pdf",
            "user_guide_sha256": guide_hash,
            "openapi_path": "build/docs/openapi.yaml",
            "openapi_sha256": openapi_hash,
        },
        "tls": {
            "ca_bundle_path": "build/docs/ca.pem",
            "ca_bundle_sha256": _CA_SHA256,
            "minimum_tls_version": "TLSv1.2",
        },
        "product_identity": {
            "trade_date": "2026-07-31",
            "short_code": "CHBY",
            "maturity": "202701",
            "instrument_isin": _ISIN,
        },
        "reference_capture": reference_capture,
        "execution": {
            "access_token_env_var": TOKEN_ENV,
            "timeout_seconds": 30,
            "max_response_bytes": 1024 * 1024,
            "records_path": [],
        },
        "local_capture_execution_authorized": True,
        "scientific_admission": False,
        "monthly_solver_hard_level_eligible": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "holdout_eligible": False,
        "candidate_assembly_eligible": False,
        "publication_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }
    document["spec_id"] = compute_spec_id(document)
    return document


def _write_spec(root: Path, document: dict[str, object], name: str) -> tuple[Path, str]:
    path = root / ".planning" / name
    payload = canonical_json_bytes(document)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _reference_record() -> dict[str, object]:
    return {
        "TrdDate": "2026-07-31",
        "ShortCode": "CHBY",
        "Maturity": "202701",
        "InstrumentISIN": _ISIN,
        "LongName": "EEX Swiss Power Base Year Future",
    }


def _settlement_record(*, action: str, timestamp: str, price: float) -> dict[str, object]:
    return {
        "TrdDate": "2026-07-31",
        "ShortCode": "CHBY",
        "Maturity": "202701",
        "InstrumentISIN": _ISIN,
        "Px": price,
        "Currency": "EUR",
        "UOM": "MWh",
        "UpdtAct": action,
        "Tm": timestamp,
    }


def _exchange(
    body_value: object,
    *,
    status: int = 200,
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> HttpsExchange:
    body = json.dumps(body_value, separators=(",", ":")).encode("utf-8")
    return HttpsExchange(
        status=status,
        reason="OK" if status == 200 else "Found",
        headers=(
            ("Date", "Fri, 31 Jul 2026 10:00:00 GMT"),
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
            *extra_headers,
            ("Set-Cookie", "must-not-be-persisted=secret"),
        ),
        body=body,
        tls_version="TLSv1.3",
        tls_cipher="TLS_AES_256_GCM_SHA384",
        peer_leaf_certificate_sha256="a" * 64,
    )


def _clock() -> Callable[[], datetime]:
    values = iter(
        [
            datetime(2026, 7, 31, 9, 59, tzinfo=timezone.utc),
            datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
        ]
    )
    return lambda: next(values)


def _run_reference(
    root: Path,
    guide_hash: str,
    openapi_hash: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    capture_id: str = "eex-v2-reference-001",
) -> Path:
    document = _spec_document(
        capture_id=capture_id,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, f"{capture_id}.json")
    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(_reference_record()),
    )
    return capture_eex_datasource_v2(
        repo_root=root,
        spec_path=spec,
        expected_spec_sha256=digest,
        environment={TOKEN_ENV: "never-persist-this-token"},
        clock=_clock(),
    )


def test_reference_capture_preserves_exact_bytes_without_granting_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    manifest_path = _run_reference(root, guide_hash, openapi_hash, monkeypatch)

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    headers = json.loads((manifest_path.parent / "response-headers.json").read_bytes())
    assert manifest["status"] == CAPTURE_STATUS
    assert manifest["request"]["data_role"] == "REFERENCE_DATA"
    assert manifest["request"]["authorization_scheme"] == "BEARER_ENVIRONMENT_ONLY"
    assert manifest["request"]["access_token_persisted"] is False
    assert manifest["response"]["record_count"] == 1
    assert manifest["response"]["provider_limit_reached"] is False
    assert manifest["workstation_clock_trusted"] is False
    assert manifest["scientific_admission"] is False
    assert manifest["monthly_solver_hard_level_eligible"] is False
    assert manifest["production_authorization"] is False
    assert manifest["promotion_gate"] is False
    assert b"never-persist-this-token" not in manifest_bytes
    assert "set-cookie" not in headers["headers"]


def test_missing_token_fails_before_creating_capture_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-no-token",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "no-token.json")

    with pytest.raises(EexDatasourceV2CaptureError, match=TOKEN_ENV):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={},
        )

    assert not (root / "build" / "eex-datasource-v2-captures").exists()


@pytest.mark.parametrize(
    "query_value",
    [
        "SECRETSECRET1234+",
        "SECRETSECRET1234%2b",
        "U0VDUkVUU0VDUkVUMTIzNCs=",
    ],
)
def test_token_smuggling_in_benign_query_fails_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query_value: str,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id=f"eex-v2-query-secret-{hashlib.sha256(query_value.encode()).hexdigest()[:8]}",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    document["request"]["query"] = [{"name": "filter", "value": query_value}]
    document["spec_id"] = compute_spec_id(document)
    spec, digest = _write_spec(root, document, "query-secret.json")

    with pytest.raises(EexDatasourceV2CaptureError, match="credential material"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "SECRETSECRET1234+"},
            clock=_clock(),
        )

    assert not (root / "build" / "eex-datasource-v2-captures").exists()


def test_spec_swap_during_secret_scan_fails_hash_binding_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    token = "SECRET-SECRET-1234"
    document = _spec_document(
        capture_id=token,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "spec-swap.json")
    clean_document = dict(document)
    clean_document["capture_id"] = "clean-capture-id"
    clean_document["spec_id"] = compute_spec_id(clean_document)
    clean_payload = canonical_json_bytes(clean_document)
    real_read = capture_module.read_stable_single_link_file
    reads = 0

    def swapped_read(path: object, **kwargs: object) -> bytes:
        nonlocal reads
        if Path(path) == spec:
            reads += 1
            if reads == 2:
                return clean_payload
        return real_read(path, **kwargs)

    monkeypatch.setattr(capture_module, "read_stable_single_link_file", swapped_read)
    with pytest.raises(EexDatasourceV2CaptureError, match="changed before credential"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: token},
            clock=_clock(),
        )

    assert not (root / "build" / "eex-datasource-v2-captures").exists()


def test_settlement_capture_recursively_binds_reference_and_preserves_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    document = _spec_document(
        capture_id="eex-v2-settlement-001",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, "settlement.json")
    records = [
        _settlement_record(action="New", timestamp="2026-07-31T18:40:13Z", price=80.0),
        _settlement_record(
            action="Change", timestamp="2026-07-31T19:41:57Z", price=80.25
        ),
    ]

    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(records),
    )
    manifest_path = capture_eex_datasource_v2(
        repo_root=root,
        spec_path=spec,
        expected_spec_sha256=digest,
        environment={TOKEN_ENV: "local-test-token"},
        clock=_clock(),
    )

    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["reference_capture"] == reference
    assert manifest["response"]["record_count"] == 2
    assert manifest["response"]["corrections_preserved_without_collapse"] is True
    assert (
        manifest["response"]["settlement_revision_ordering"]
        == "TM_CHRONOLOGY_VALIDATED_RAW_PROVIDER_ORDER_PRESERVED"
    )
    assert json.loads((manifest_path.parent / "response.json").read_bytes()) == records
    assert manifest["monthly_solver_hard_level_eligible"] is False


def test_settlement_validates_tm_chronology_without_reordering_provider_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    document = _spec_document(
        capture_id="eex-v2-settlement-reversed",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, "settlement-reversed.json")
    records = [
        _settlement_record(
            action="Change", timestamp="2026-07-31T19:41:57Z", price=80.25
        ),
        _settlement_record(action="New", timestamp="2026-07-31T18:40:13Z", price=80.0),
    ]

    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(records),
    )
    manifest_path = capture_eex_datasource_v2(
        repo_root=root,
        spec_path=spec,
        expected_spec_sha256=digest,
        environment={TOKEN_ENV: "local-test-token"},
        clock=_clock(),
    )

    assert json.loads((manifest_path.parent / "response.json").read_bytes()) == records
    manifest = json.loads(manifest_path.read_bytes())
    assert (
        manifest["response"]["settlement_revision_ordering"]
        == "TM_CHRONOLOGY_VALIDATED_RAW_PROVIDER_ORDER_PRESERVED"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["request"].update(
                {"query": [{"name": "access_token", "value": "secret"}]}
            ),
            "query parameter name is unsafe",
        ),
        (
            lambda document: document.update({"scientific_admission": 0}),
            "scientific_admission must be false",
        ),
        (
            lambda document: document["request"].update(
                {
                    "endpoint_path": (
                        "/rd/derivatives/POWER/CH/2026-07-31/CHBY+1/202701"
                    )
                }
            ),
            "rolling-symbol endpoint is forbidden",
        ),
    ],
)
def test_spec_rejects_credential_smuggling_authority_and_rolling_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-invalid",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    mutation(document)
    document["spec_id"] = compute_spec_id(document)
    spec, digest = _write_spec(root, document, "invalid.json")

    with pytest.raises(EexDatasourceV2CaptureError, match=message):
        load_eex_datasource_v2_capture_spec(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
        )


def test_spec_rejects_product_identity_substrings_in_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-endpoint-substrings",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    document["request"]["endpoint_path"] = (
        "/rd/derivatives/POWER/CH/x2026-07-31x/XXCHBYXX/X202701X"
    )
    document["spec_id"] = compute_spec_id(document)
    spec, digest = _write_spec(root, document, "endpoint-substrings.json")

    with pytest.raises(EexDatasourceV2CaptureError, match="exact product identity segments"):
        load_eex_datasource_v2_capture_spec(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
        )


@pytest.mark.parametrize(
    "value",
    [
        "build//docs/guide.pdf",
        "build/./docs/guide.pdf",
        "build/docs/trailing. ",
        "CON",
        "build/AUX/file",
        "build/LPT1.txt",
    ],
)
def test_portable_relative_rejects_windows_aliases_and_noncanonical_parts(
    value: str,
) -> None:
    with pytest.raises(EexDatasourceV2CaptureError, match="portable relative"):
        capture_module._portable_relative(value, label="test")


@pytest.mark.parametrize("value", ["CON", "AUX", "LPT1.txt", "capture."])
def test_portable_id_rejects_windows_reserved_and_trailing_dot(value: str) -> None:
    with pytest.raises(EexDatasourceV2CaptureError, match="portable"):
        capture_module._portable_id(value, label="capture ID")


def test_openapi_bytes_are_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-openapi-change",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "openapi-change.json")
    (root / "build" / "docs" / "openapi.yaml").write_text(
        "openapi: 3.1.0\n", encoding="utf-8"
    )

    with pytest.raises(EexDatasourceV2CaptureError, match="OpenAPI specification SHA-256"):
        load_eex_datasource_v2_capture_spec(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
        )


def test_redirect_fails_closed_and_keeps_only_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-redirect",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "redirect.json")

    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange({}, status=302),
    )
    with pytest.raises(EexDatasourceV2CaptureError, match="HTTP 302"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "local-test-token"},
            clock=_clock(),
        )

    output = (
        root
        / "build"
        / "eex-datasource-v2-captures"
        / "_incomplete-eex-v2-redirect"
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "capture-attempt.json",
        "capture-failure.json",
    ]


def test_unicode_escaped_json_token_echo_fails_before_response_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    capture_id = "eex-v2-escaped-secret"
    document = _spec_document(
        capture_id=capture_id,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "escaped-secret.json")
    token = "SUPERSECRETBEARER123"
    escaped = "".join(f"\\u{ord(character):04x}" for character in token)
    body = (
        '{"TrdDate":"2026-07-31","ShortCode":"CHBY",'
        '"Maturity":"202701","InstrumentISIN":"CH0000000001",'
        f'"echo":"{escaped}"}}'
    ).encode("ascii")
    exchange = HttpsExchange(
        status=200,
        reason="OK",
        headers=(
            ("Date", "Fri, 31 Jul 2026 10:00:00 GMT"),
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ),
        body=body,
        tls_version="TLSv1.3",
        tls_cipher="TLS_AES_256_GCM_SHA384",
        peer_leaf_certificate_sha256="a" * 64,
    )
    monkeypatch.setattr(capture_module, "_https_transport", lambda _spec, _token: exchange)

    with pytest.raises(EexDatasourceV2CaptureError, match="credential material"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: token},
            clock=_clock(),
        )

    capture_root = root / "build" / "eex-datasource-v2-captures"
    assert not (capture_root / capture_id).exists()
    staging = capture_root / f"_incomplete-{capture_id}"
    assert sorted(path.name for path in staging.iterdir()) == [
        "capture-attempt.json",
        "capture-failure.json",
    ]


@pytest.mark.parametrize(
    "encoded_token",
    [
        "SECRETSECRET1234%2b",
        "%53%45%43%52%45%54%53%45%43%52%45%54%31%32%33%34%2B",
        "%2553%2545%2543%2552%2545%2554%2553%2545%2543%2552%2545%2554%2531%2532%2533%2534%252B",
        "%252525252553%252525252545%252525252543%252525252552%252525252545%252525252554%252525252553%252525252545%252525252543%252525252552%252525252545%252525252554%252525252531%252525252532%252525252533%252525252534%25252525252B",
        "U0VD UkVU U0VD UkVU MTIz NCs=",
    ],
)
def test_percent_encoded_token_echo_fails_before_response_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoded_token: str,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    capture_id = f"eex-v2-percent-secret-{hashlib.sha256(encoded_token.encode()).hexdigest()[:8]}"
    document = _spec_document(
        capture_id=capture_id,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, f"{capture_id}.json")
    record = _reference_record()
    record["echo"] = encoded_token
    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(record),
    )

    with pytest.raises(EexDatasourceV2CaptureError, match="credential material"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "SECRETSECRET1234+"},
            clock=_clock(),
        )

    capture_root = root / "build" / "eex-datasource-v2-captures"
    assert not (capture_root / capture_id).exists()
    assert sorted(
        path.name for path in (capture_root / f"_incomplete-{capture_id}").iterdir()
    ) == ["capture-attempt.json", "capture-failure.json"]


def test_input_mutation_during_request_prevents_finalized_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    capture_id = "eex-v2-input-toctou"
    document = _spec_document(
        capture_id=capture_id,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "input-toctou.json")

    def mutate_input(_spec: object, _token: str) -> HttpsExchange:
        (root / "build" / "docs" / "openapi.yaml").write_text(
            "openapi: 3.1.0\n", encoding="utf-8"
        )
        return _exchange(_reference_record())

    monkeypatch.setattr(capture_module, "_https_transport", mutate_input)
    with pytest.raises(EexDatasourceV2CaptureError, match="changed after request"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "local-test-token"},
            clock=_clock(),
        )

    capture_root = root / "build" / "eex-datasource-v2-captures"
    assert not (capture_root / capture_id).exists()
    assert not (
        capture_root / f"_incomplete-{capture_id}" / "capture-manifest.json"
    ).exists()


def test_http_429_writes_redacted_terminal_receipt_with_retry_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    capture_id = "eex-v2-http-429"
    document = _spec_document(
        capture_id=capture_id,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "http-429.json")
    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(
            {},
            status=429,
            extra_headers=(
                ("Retry-After", "30"),
                ("X-Correlation-ID", "eex-request-123"),
            ),
        ),
    )

    with pytest.raises(EexDatasourceV2CaptureError, match="HTTP 429"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "terminal-receipt-token"},
            clock=_clock(),
        )

    staging = (
        root
        / "build"
        / "eex-datasource-v2-captures"
        / f"_incomplete-{capture_id}"
    )
    receipt_bytes = (staging / "capture-failure.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt_bytes == canonical_json_bytes(receipt)
    assert receipt["terminal_for_capture_id"] is True
    assert receipt["failure_class"] == "PROVIDER_RATE_LIMITED"
    assert receipt["http_status"] == 429
    assert receipt["retry_after_observed"] == "30"
    assert receipt["correlation_id_observed"] == "eex-request-123"
    assert receipt["automatic_retry_attempted"] is False
    assert receipt["new_hash_bound_spec_and_capture_id_required"] is True
    assert receipt["production_authorization"] is False
    assert b"terminal-receipt-token" not in receipt_bytes


def test_unapproved_tls_root_is_rejected_by_independent_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(capture_module, "APPROVED_TLS_CA_SHA256S", frozenset())
    document = _spec_document(
        capture_id="eex-v2-unapproved-ca",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "unapproved-ca.json")

    with pytest.raises(EexDatasourceV2CaptureError, match="independent security policy"):
        load_eex_datasource_v2_capture_spec(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
        )


def test_rate_limiter_enforces_one_request_per_second_and_canonical_state(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    with capture_module._request_rate_slot(capture_root, timeout_seconds=3):
        pass
    started = time.monotonic()
    with capture_module._request_rate_slot(capture_root, timeout_seconds=3):
        pass

    assert time.monotonic() - started >= 0.9
    state = capture_root / "request-rate-limit.json"
    document = json.loads(state.read_bytes())
    assert state.read_bytes() == canonical_json_bytes(document)
    assert document["schema_version"] == "eex_datasource_v2_rate_limit.v1"


def test_rate_limiter_fails_closed_on_concurrent_lock(tmp_path: Path) -> None:
    capture_root = tmp_path / "captures"
    capture_root.mkdir()

    with capture_module._exclusive_file_lock(
        capture_root / "request-rate-limit.lock",
        deadline=time.monotonic() + 1,
    ):
        with pytest.raises(EexDatasourceV2CaptureError, match="lock is unavailable"):
            with capture_module._request_rate_slot(capture_root, timeout_seconds=0):
                pass


def test_rate_limiter_rejects_hardlinked_lock_without_touching_target(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(b"")
    os.link(target, capture_root / "request-rate-limit.lock")

    with pytest.raises(EexDatasourceV2CaptureError, match="single-link regular"):
        with capture_module._request_rate_slot(capture_root, timeout_seconds=1):
            pass

    assert target.read_bytes() == b""


def test_rate_limiter_rejects_symlinked_lock_without_touching_target(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(b"")
    link = capture_root / "request-rate-limit.lock"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable to this standard user")

    with pytest.raises(ValueError, match="link or reparse point"):
        with capture_module._request_rate_slot(capture_root, timeout_seconds=1):
            pass

    assert target.read_bytes() == b""


def test_provider_record_limit_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-limit",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    document["execution"]["max_response_bytes"] = 64 * 1024 * 1024
    document["spec_id"] = compute_spec_id(document)
    spec, digest = _write_spec(root, document, "limit.json")
    records = ["record"] * 60_000

    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(records),
    )
    with pytest.raises(EexDatasourceV2CaptureError, match="provider record limit"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "local-test-token"},
            clock=_clock(),
        )


def test_strict_json_duplicate_keys_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-duplicate-json",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    spec, digest = _write_spec(root, document, "duplicate-json.json")
    body = (
        b'{"TrdDate":"2026-07-31","ShortCode":"CHBY","ShortCode":"CHBY",'
        b'"Maturity":"202701","InstrumentISIN":"CH0000000001"}'
    )
    exchange = HttpsExchange(
        status=200,
        reason="OK",
        headers=(
            ("Date", "Fri, 31 Jul 2026 10:00:00 GMT"),
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ),
        body=body,
        tls_version="TLSv1.3",
        tls_cipher="TLS_AES_256_GCM_SHA384",
        peer_leaf_certificate_sha256="a" * 64,
    )

    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: exchange,
    )
    with pytest.raises(EexDatasourceV2CaptureError, match="strict UTF-8 JSON"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "local-test-token"},
            clock=_clock(),
        )


def test_default_transport_uses_exact_host_get_target_and_bearer_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    document = _spec_document(
        capture_id="eex-v2-default-transport",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
    )
    ca_der = x509.load_pem_x509_certificate(_CA_BYTES).public_bytes(
        serialization.Encoding.DER
    )
    (root / "build" / "docs" / "ca.pem").write_bytes(ca_der)
    ca_der_hash = hashlib.sha256(ca_der).hexdigest()
    document["tls"]["ca_bundle_sha256"] = ca_der_hash
    monkeypatch.setattr(
        capture_module,
        "APPROVED_TLS_CA_SHA256S",
        frozenset({ca_der_hash}),
    )
    document["request"]["query"] = [
        {"name": "limit", "value": "25"},
        {"name": "sort", "value": "Tm"},
    ]
    document["spec_id"] = compute_spec_id(document)
    spec_path, digest = _write_spec(root, document, "default-transport.json")
    spec = load_eex_datasource_v2_capture_spec(
        repo_root=root,
        spec_path=spec_path,
        expected_spec_sha256=digest,
    )
    observed: dict[str, object] = {}
    body = json.dumps(_reference_record(), separators=(",", ":")).encode()

    class FakeSocket:
        def getpeercert(self, *, binary_form: bool) -> bytes:
            assert binary_form is True
            return b"certificate"

        def cipher(self) -> tuple[str, str, int]:
            return ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)

        def version(self) -> str:
            return "TLSv1.3"

    class FakeResponse:
        status = 200
        reason = "OK"

        def read(self, size: int) -> bytes:
            assert size == spec.max_response_bytes + 1
            return body

        def getheaders(self) -> list[tuple[str, str]]:
            return [
                ("Date", "Fri, 31 Jul 2026 10:00:00 GMT"),
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]

    class FakeConnection:
        def __init__(
            self,
            host: str,
            port: int,
            *,
            timeout: int,
            context: object,
        ) -> None:
            observed.update(
                host=host,
                port=port,
                timeout=timeout,
                context=context,
            )
            self.sock = FakeSocket()

        def request(
            self,
            method: str,
            target: str,
            *,
            headers: dict[str, str],
        ) -> None:
            observed.update(method=method, target=target, headers=headers)

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            observed["closed"] = True

    class FakeContext:
        def __init__(self, protocol: object) -> None:
            observed["protocol"] = protocol

        minimum_version: object | None = None
        check_hostname: bool | None = None
        verify_mode: object | None = None
        keylog_filename: str | None = "ambient-keylog"

        def load_verify_locations(self, *, cadata: str) -> None:
            observed["cadata"] = cadata

    monkeypatch.setattr(capture_module.ssl, "SSLContext", FakeContext)
    monkeypatch.setattr(capture_module.http.client, "HTTPSConnection", FakeConnection)
    exchange = capture_module._https_transport(spec, "ephemeral-token")

    assert observed["host"] == "api.eex-group.com"
    assert observed["port"] == 443
    assert observed["cadata"] == _CA_BYTES.decode("ascii")
    fake_context = observed["context"]
    assert isinstance(fake_context, FakeContext)
    assert observed["protocol"] == capture_module.ssl.PROTOCOL_TLS_CLIENT
    assert fake_context.minimum_version == capture_module.ssl.TLSVersion.TLSv1_2
    assert fake_context.check_hostname is True
    assert fake_context.verify_mode == capture_module.ssl.CERT_REQUIRED
    assert fake_context.keylog_filename is None
    assert observed["method"] == "GET"
    assert observed["target"] == (
        "/rd/derivatives/POWER/CH/2026-07-31/CHBY/202701?limit=25&sort=Tm"
    )
    assert observed["headers"] == {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Authorization": "Bearer ephemeral-token",
        "Cache-Control": "no-cache",
        "User-Agent": "FMV-PFC-LT-EEX-DataSource-v2/1",
    }
    assert observed["closed"] is True
    assert exchange.peer_leaf_certificate_sha256 == hashlib.sha256(b"certificate").hexdigest()


def test_settlement_reopens_and_rehashes_reference_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    headers_path = reference_manifest.parent / "response-headers.json"
    headers_path.write_bytes(headers_path.read_bytes() + b" ")
    document = _spec_document(
        capture_id="eex-v2-tampered-reference-headers",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, "tampered-reference-headers.json")

    with pytest.raises(EexDatasourceV2CaptureError, match="headers SHA-256 mismatch"):
        load_eex_datasource_v2_capture_spec(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
        )


def test_settlement_rejects_reference_authority_smuggling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference_document["trusted_time_receipt_present"] = True
    reference_manifest.write_bytes(canonical_json_bytes(reference_document))
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    document = _spec_document(
        capture_id="eex-v2-reference-authority-smuggling",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, "reference-authority-smuggling.json")

    with pytest.raises(EexDatasourceV2CaptureError, match="not explicitly false"):
        load_eex_datasource_v2_capture_spec(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("request", "access_token_persisted", True),
        ("request", "authorization_scheme", "BEARER_PERSISTED_IN_ARTIFACT"),
        ("tls_trust", "tls_interception_security_approved", True),
        ("tls_trust", "trust_classification", "PROVIDER_DIRECT_IDENTITY_ATTESTED"),
    ],
)
def test_settlement_rejects_nested_reference_authority_smuggling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: object,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference_document[section][field] = value
    reference_manifest.write_bytes(canonical_json_bytes(reference_document))
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    document = _spec_document(
        capture_id=f"eex-v2-nested-smuggling-{section}-{field}",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, f"nested-{section}-{field}.json")

    with pytest.raises(EexDatasourceV2CaptureError, match="claims differ"):
        load_eex_datasource_v2_capture_spec(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
        )


def test_settlement_rejects_mixed_product_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    document = _spec_document(
        capture_id="eex-v2-mixed-product",
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, "mixed-product.json")
    other = _settlement_record(
        action="Change", timestamp="2026-07-31T19:41:57Z", price=81.0
    )
    other["Maturity"] = "202801"
    records = [
        _settlement_record(action="New", timestamp="2026-07-31T18:40:13Z", price=80.0),
        other,
    ]

    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(records),
    )
    with pytest.raises(EexDatasourceV2CaptureError, match="mixes or omits"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "local-test-token"},
            clock=_clock(),
        )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                _settlement_record(
                    action="Change", timestamp="2026-07-31T18:40:13Z", price=80.0
                )
            ],
            "revision lifecycle",
        ),
        (
            [
                _settlement_record(
                    action="New", timestamp="2026-07-31T18:40:13Z", price=80.0
                ),
                _settlement_record(
                    action="New", timestamp="2026-07-31T19:40:13Z", price=80.2
                ),
            ],
            "revision lifecycle",
        ),
        (
            [
                _settlement_record(
                    action="New", timestamp="2026-07-31T18:40:13Z", price=80.0
                ),
                _settlement_record(
                    action="Change", timestamp="2026-07-31T18:40:13Z", price=80.2
                ),
            ],
            "revision lifecycle",
        ),
    ],
)
def test_settlement_rejects_invalid_revision_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, object]],
    message: str,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    capture_id = f"eex-v2-bad-revisions-{len(records)}-{records[-1]['UpdtAct']}"
    document = _spec_document(
        capture_id=capture_id,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, f"{capture_id}.json")
    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(records),
    )

    with pytest.raises(EexDatasourceV2CaptureError, match=message):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "local-test-token"},
            clock=_clock(),
        )


@pytest.mark.parametrize(("field", "value"), [("Currency", "CHF"), ("UOM", "kWh")])
def test_settlement_rejects_wrong_currency_or_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    root, guide_hash, openapi_hash = _repo(tmp_path, monkeypatch)
    reference_manifest = _run_reference(root, guide_hash, openapi_hash, monkeypatch)
    reference_document = json.loads(reference_manifest.read_bytes())
    reference = {
        "manifest_path": reference_manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "response_sha256": reference_document["response"]["sha256"],
    }
    capture_id = f"eex-v2-bad-{field.lower()}"
    document = _spec_document(
        capture_id=capture_id,
        guide_hash=guide_hash,
        openapi_hash=openapi_hash,
        role="SETTLEMENT",
        reference_capture=reference,
    )
    spec, digest = _write_spec(root, document, f"{capture_id}.json")
    record = _settlement_record(
        action="New", timestamp="2026-07-31T18:40:13Z", price=80.0
    )
    record[field] = value
    monkeypatch.setattr(
        capture_module,
        "_https_transport",
        lambda _spec, _token: _exchange(record),
    )

    with pytest.raises(EexDatasourceV2CaptureError, match="currency or unit"):
        capture_eex_datasource_v2(
            repo_root=root,
            spec_path=spec,
            expected_spec_sha256=digest,
            environment={TOKEN_ENV: "local-test-token"},
            clock=_clock(),
        )
