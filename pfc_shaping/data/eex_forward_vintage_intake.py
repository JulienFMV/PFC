"""Fail-closed normalization of one prospectively captured EEX workbook.

This module deliberately does not capture bytes, mint trusted time, sign an
acquisition catalog, or publish to an external CAS.  It accepts those external
facts, verifies their bindings, and deterministically appends the exact quotes
to the bitemporal EEX vintage schema.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

from pfc_shaping.data import ingest_forwards
from pfc_shaping.data.acquisition_contract import (
    TRUSTED_TIME_RECEIPT_SCHEMA,
    verify_trusted_time_receipt,
)
from pfc_shaping.data.eex_historical_vintage import (
    EEX_HISTORICAL_QUOTE_SCHEMA,
    historical_vintage_revision_id,
    historical_vintage_row_hash,
    historical_vintage_snapshot_id,
    validate_eex_historical_vintage_frame,
    verify_eex_historical_vintage_catalog,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

EEX_FORWARD_VINTAGE_INTAKE_SCHEMA = "eex_forward_vintage_intake_spec.v1"
EEX_FORWARD_PARSER_VERSION = "ingest_forwards.v1"
EEX_FORWARD_SOURCE_ROLE = "eex_forward_source"
EEX_FORWARD_SOURCE_SYSTEM = "EEX_MARKET_DATA"
EEX_FORWARD_SOURCE_CLASS = "IMMUTABLE_DAILY_EEX_XLSX"
EEX_FORWARD_REVISION_POLICY = "APPEND_ONLY_PRESERVE_ALL_REVISIONS"
EEX_FORWARD_QUOTE_CONVENTIONS = frozenset({"EEX_SETTLEMENT_EUR_MWH"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ALLOWED_MARKETS = frozenset({"CH", "DE", "FR", "AT", "IT"})
_SPEC_KEYS = {
    "schema_version",
    "intake_id",
    "source_role",
    "source_system",
    "source_class",
    "source_document_id",
    "source_document_sha256",
    "source_document_size_bytes",
    "trusted_time_receipt_sha256",
    "markets",
    "market_snapshot_dates",
    "expected_quotes",
    "quote_convention",
    "unit",
    "parser_version",
    "selection_mode",
    "revision_policy",
    "monthly_level_authority",
    "ompex_used",
    "external_catalog_signature_required",
    "external_cas_admission_required",
    "scientific_admission",
    "production_authorization",
    "promotion_gate",
}
_EXPECTED_QUOTE_KEYS = {
    "market",
    "source_product_code",
    "product",
    "load_type",
    "product_type",
}
_TRUSTED_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "source_document_id",
    "source_document_sha256",
    "size_bytes",
    "received_at_utc",
    "journal_id",
    "journal_sequence",
    "previous_receipt_id",
    "trusted_time_attestation",
}
_ATTESTATION_KEYS = {
    "algorithm",
    "key_id",
    "payload_sha256",
    "value_base64",
}
_MAX_SPEC_BYTES = 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_CATALOG_BYTES = 4 * 1024 * 1024
_MAX_HISTORY_BYTES = 512 * 1024 * 1024
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_XLSX_ENTRIES = 4_096
_MAX_XLSX_EXPANDED_BYTES = 256 * 1024 * 1024
_MAX_XLSX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_XLSX_COMPRESSION_RATIO = 1_000


class EexForwardVintageIntakeError(ValueError):
    """Raised when prospective EEX bytes cannot enter vintage normalization."""


@dataclass(frozen=True)
class EexForwardVintageIntakeResult:
    """Verified normalization evidence, still awaiting catalog/CAS admission."""

    intake_id: str
    acquisition_id: str
    available_at_utc: str
    source_document_id: str
    source_document_sha256: str
    source_document_size_bytes: int
    trusted_time_receipt_sha256: str
    parser_code_sha256: str
    parser_config_sha256: str
    parser_config: Mapping[str, object]
    trusted_time_receipt: Mapping[str, object]
    new_row_count: int
    total_row_count: int
    snapshot_ids: tuple[str, ...]
    calibration_eligible: bool = False
    external_cas_admitted: bool = False
    production_authorization: bool = False

    def source_catalog_entry(
        self,
        *,
        source_document_path: str,
        parser_code_path: str,
    ) -> dict[str, object]:
        """Return the exact source entry for a separately signed catalog."""

        return {
            "snapshot_ids": list(self.snapshot_ids),
            "acquisition_id": self.acquisition_id,
            "observed_at": self.available_at_utc,
            "available_at": self.available_at_utc,
            "source_document_id": self.source_document_id,
            "path": _portable_relative_path(
                source_document_path, label="EEX source document path"
            ),
            "sha256": self.source_document_sha256,
            "size_bytes": self.source_document_size_bytes,
            "parser_code_path": _portable_relative_path(
                parser_code_path, label="EEX parser code path"
            ),
            "parser_code_sha256": self.parser_code_sha256,
            "parser_config": dict(self.parser_config),
            "parser_config_sha256": self.parser_config_sha256,
            "trusted_time_receipt": dict(self.trusted_time_receipt),
        }


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one identity payload without platform-dependent whitespace."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compute_eex_forward_intake_id(spec: Mapping[str, object]) -> str:
    """Compute the content identity of a spec, excluding its self identity."""

    identity = dict(spec)
    identity.pop("intake_id", None)
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def load_eex_forward_intake_spec(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], bytes]:
    """Load a canonical, caller-hash-bound intake specification."""

    expected_hash = _require_sha256(expected_sha256, label="EEX intake spec")
    payload = _read_absolute_stable_file(
        path,
        label="EEX intake spec",
        max_bytes=_MAX_SPEC_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise EexForwardVintageIntakeError("EEX intake spec SHA-256 mismatch")
    try:
        document = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EexForwardVintageIntakeError("EEX intake spec is not strict JSON") from exc
    if not isinstance(document, dict):
        raise EexForwardVintageIntakeError("EEX intake spec must be a mapping")
    if canonical_json_bytes(document) != payload:
        raise EexForwardVintageIntakeError("EEX intake spec is not canonical JSON")
    validate_eex_forward_intake_spec(document)
    return document, payload


def validate_eex_forward_intake_spec(spec: Mapping[str, object]) -> None:
    """Validate exact source, product, authority, and negative-use claims."""

    if set(spec) != _SPEC_KEYS:
        raise EexForwardVintageIntakeError("EEX intake spec fields are not exact")
    expected_constants = {
        "schema_version": EEX_FORWARD_VINTAGE_INTAKE_SCHEMA,
        "source_role": EEX_FORWARD_SOURCE_ROLE,
        "source_system": EEX_FORWARD_SOURCE_SYSTEM,
        "source_class": EEX_FORWARD_SOURCE_CLASS,
        "unit": "EUR/MWH",
        "parser_version": EEX_FORWARD_PARSER_VERSION,
        "selection_mode": "latest_available",
        "revision_policy": EEX_FORWARD_REVISION_POLICY,
        "monthly_level_authority": "SOLVER_WITH_HARD_GOVERNED_CH_EEX_FORWARD_QUOTES",
        "ompex_used": False,
        "external_catalog_signature_required": True,
        "external_cas_admission_required": True,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    for key, expected in expected_constants.items():
        actual = spec.get(key)
        if (
            (isinstance(expected, bool) and actual is not expected)
            or (not isinstance(expected, bool) and actual != expected)
        ):
            raise EexForwardVintageIntakeError(f"EEX intake spec {key} is invalid")
    if str(spec.get("quote_convention", "")) not in EEX_FORWARD_QUOTE_CONVENTIONS:
        raise EexForwardVintageIntakeError("EEX intake quote convention is unsupported")
    document_id = str(spec.get("source_document_id", ""))
    if not document_id or len(document_id) > 128 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", document_id
    ):
        raise EexForwardVintageIntakeError("EEX source_document_id is invalid")
    _require_sha256(spec.get("source_document_sha256"), label="EEX source document")
    _require_sha256(spec.get("trusted_time_receipt_sha256"), label="trusted-time receipt")
    size = spec.get("source_document_size_bytes")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
        or size > 64 * 1024 * 1024
    ):
        raise EexForwardVintageIntakeError("EEX source document size is invalid")
    markets = _string_sequence(spec.get("markets"), label="EEX intake markets")
    if markets != sorted(set(markets)) or "CH" not in markets:
        raise EexForwardVintageIntakeError(
            "EEX intake markets must be sorted, unique, and include CH"
        )
    if not set(markets).issubset(_ALLOWED_MARKETS):
        raise EexForwardVintageIntakeError("EEX intake contains an unsupported market")
    snapshot_dates = spec.get("market_snapshot_dates")
    if not isinstance(snapshot_dates, Mapping) or set(snapshot_dates) != set(markets):
        raise EexForwardVintageIntakeError("EEX market/date inventory mismatch")
    for market in markets:
        value = str(snapshot_dates[market])
        if not _DATE.fullmatch(value):
            raise EexForwardVintageIntakeError("EEX market snapshot date is invalid")
        try:
            parsed = pd.Timestamp(value)
        except ValueError as exc:
            raise EexForwardVintageIntakeError(
                "EEX market snapshot date is invalid"
            ) from exc
        if parsed.strftime("%Y-%m-%d") != value:
            raise EexForwardVintageIntakeError("EEX market snapshot date is invalid")
    expected_quotes = spec.get("expected_quotes")
    if not isinstance(expected_quotes, list) or not expected_quotes:
        raise EexForwardVintageIntakeError("EEX expected quote inventory is missing")
    normalized: list[dict[str, str]] = []
    economic_identities: set[tuple[str, str, str]] = set()
    source_identities: set[tuple[str, str]] = set()
    for value in expected_quotes:
        if not isinstance(value, Mapping) or set(value) != _EXPECTED_QUOTE_KEYS:
            raise EexForwardVintageIntakeError("EEX expected quote fields are not exact")
        quote = {key: str(value[key]) for key in _EXPECTED_QUOTE_KEYS}
        code = quote["source_product_code"]
        parsed_product = ingest_forwards.normalize_eex_product_code(code)
        if parsed_product is None:
            raise EexForwardVintageIntakeError("EEX expected source product is unsupported")
        product, load_type, product_type = parsed_product
        if product_type.upper() not in {"CAL", "QUARTER", "MONTH"}:
            raise EexForwardVintageIntakeError(
                "EEX expected source product is not LT-eligible"
            )
        expected = {
            "market": quote["market"],
            "source_product_code": code,
            "product": product,
            "load_type": load_type,
            "product_type": product_type.upper(),
        }
        if quote != expected or quote["market"] not in markets:
            raise EexForwardVintageIntakeError("EEX expected quote identity is inconsistent")
        source_identity = (quote["market"], code)
        economic_identity = (quote["market"], load_type, product)
        if source_identity in source_identities or economic_identity in economic_identities:
            raise EexForwardVintageIntakeError("EEX expected quote inventory contains duplicates")
        source_identities.add(source_identity)
        economic_identities.add(economic_identity)
        normalized.append(quote)
    if normalized != sorted(normalized, key=_quote_sort_key):
        raise EexForwardVintageIntakeError("EEX expected quote inventory is not sorted")
    if set(quote["market"] for quote in normalized) != set(markets):
        raise EexForwardVintageIntakeError("EEX expected quote inventory omits a market")
    if not any(
        quote["market"] == "CH" and quote["load_type"] == "BASE"
        for quote in normalized
    ):
        raise EexForwardVintageIntakeError(
            "EEX intake requires at least one direct CH BASE quote"
        )
    if str(spec.get("intake_id", "")) != compute_eex_forward_intake_id(spec):
        raise EexForwardVintageIntakeError("EEX intake_id mismatch")


def append_eex_forward_vintage_intake(
    *,
    spec_path: str | Path,
    expected_spec_sha256: str,
    source_document_path: str | Path,
    trusted_time_receipt_path: str | Path,
    previous_catalog_path: str | Path | None = None,
    previous_history_path: str | Path | None = None,
    trusted_time_public_key_path: str | Path | None = None,
    trusted_time_public_key_dir: str | Path | None = None,
    trusted_time_journal_id: str | None = None,
    previous_catalog_public_key_path: str | Path | None = None,
    previous_catalog_public_key_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, EexForwardVintageIntakeResult]:
    """Verify and append one workbook without granting catalog/CAS authority."""

    spec, _ = load_eex_forward_intake_spec(
        spec_path,
        expected_sha256=expected_spec_sha256,
    )
    source_bytes = _read_absolute_stable_file(
        source_document_path,
        label="EEX source document",
        max_bytes=_MAX_SOURCE_BYTES,
    )
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if (
        source_hash != spec["source_document_sha256"]
        or len(source_bytes) != spec["source_document_size_bytes"]
    ):
        raise EexForwardVintageIntakeError("EEX source document binding mismatch")
    preflight_eex_forward_workbook_bytes(source_bytes)
    receipt_bytes = _read_absolute_stable_file(
        trusted_time_receipt_path,
        label="EEX trusted-time receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != spec["trusted_time_receipt_sha256"]:
        raise EexForwardVintageIntakeError("EEX trusted-time receipt SHA-256 mismatch")
    receipt, unsigned_receipt = _verify_receipt(
        receipt_bytes,
        source_document_id=str(spec["source_document_id"]),
        source_document_sha256=source_hash,
        source_document_size_bytes=len(source_bytes),
        trusted_public_key_path=trusted_time_public_key_path,
        trusted_public_key_dir=trusted_time_public_key_dir,
        trusted_journal_id=trusted_time_journal_id,
    )
    available_at = pd.Timestamp(unsigned_receipt["received_at_utc"])
    available_iso = available_at.tz_convert("UTC").isoformat()
    parser_path = assert_absolute_path_has_no_links(Path(ingest_forwards.__file__).resolve())
    parser_size = parser_path.stat().st_size
    if parser_size < 1 or parser_size > 2 * 1024 * 1024:
        raise EexForwardVintageIntakeError("runtime EEX parser size is invalid")
    parser_payload = read_stable_single_link_file(
        parser_path,
        label="runtime EEX parser",
        max_bytes=2 * 1024 * 1024,
    )
    if len(parser_payload) != parser_size:
        raise EexForwardVintageIntakeError("runtime EEX parser changed during read")
    parser_code_hash = hashlib.sha256(parser_payload).hexdigest()
    parser_config = {
        "parser_version": EEX_FORWARD_PARSER_VERSION,
        "selection_mode": "latest_available",
        "markets": dict(spec["market_snapshot_dates"]),
        "quote_convention": str(spec["quote_convention"]),
        "expected_quotes": list(spec["expected_quotes"]),
        "intake_id": str(spec["intake_id"]),
    }
    parser_config_hash = hashlib.sha256(canonical_json_bytes(parser_config)).hexdigest()
    actual_quotes: list[dict[str, str]] = []
    parsed_rows: list[dict[str, object]] = []
    for market in spec["markets"]:
        inspected_date, inspected_quotes = inspect_eex_forward_workbook_latest_market_row(
            source_bytes,
            market=str(market),
        )
        if inspected_date != spec["market_snapshot_dates"][market]:
            raise EexForwardVintageIntakeError(
                "EEX workbook latest physical row differs from the frozen inventory"
            )
        expected_market_quotes = [
            quote for quote in spec["expected_quotes"] if quote["market"] == market
        ]
        if inspected_quotes != expected_market_quotes:
            raise EexForwardVintageIntakeError(
                "EEX latest physical row differs from the frozen quote inventory"
            )
        try:
            prices, snapshot_date, metadata = (
                ingest_forwards.load_base_prices_from_eex_report_bytes(
                    source_bytes,
                    market=str(market),
                    source_label=str(spec["source_document_id"]),
                    return_snapshot_metadata=True,
                )
            )
        except Exception as exc:
            raise EexForwardVintageIntakeError("EEX workbook replay failed") from exc
        snapshot_iso = pd.Timestamp(snapshot_date).strftime("%Y-%m-%d")
        if snapshot_iso != spec["market_snapshot_dates"][market]:
            raise EexForwardVintageIntakeError(
                "EEX workbook latest snapshot date differs from the frozen inventory"
            )
        if available_at < pd.Timestamp(snapshot_iso, tz="UTC"):
            raise EexForwardVintageIntakeError(
                "EEX trusted availability predates the workbook snapshot date"
            )
        lineage = metadata.get("quote_lineage")
        if not isinstance(lineage, Mapping) or set(lineage) != set(prices):
            raise EexForwardVintageIntakeError("EEX workbook lineage inventory mismatch")
        for price_key, price in prices.items():
            raw_lineage = lineage[price_key]
            if not isinstance(raw_lineage, Mapping):
                raise EexForwardVintageIntakeError("EEX workbook quote lineage is invalid")
            code = str(raw_lineage.get("source_product_code", ""))
            normalized = ingest_forwards.normalize_eex_product_code(code)
            if normalized is None:
                raise EexForwardVintageIntakeError("EEX workbook product identity is unsupported")
            product, load_type, product_type = normalized
            expected_key = product if load_type == "BASE" else f"{product}-Peak"
            if price_key != expected_key:
                raise EexForwardVintageIntakeError("EEX workbook price/product identity mismatch")
            actual_quotes.append(
                {
                    "market": str(market),
                    "source_product_code": code,
                    "product": product,
                    "load_type": load_type,
                    "product_type": product_type.upper(),
                }
            )
            parsed_rows.append(
                {
                    "date": pd.Timestamp(snapshot_iso),
                    "observed_at": available_iso,
                    "available_at": available_iso,
                    "acquisition_id": str(unsigned_receipt["receipt_id"]),
                    "market": str(market),
                    "load_type": load_type,
                    "product_type": product_type.upper(),
                    "product": product,
                    "price": float(price),
                    "unit": "EUR/MWH",
                    "source": "EEX",
                    "source_document_id": str(spec["source_document_id"]),
                    "source_document_sha256": source_hash,
                    "source_sheet": str(market),
                    "source_row_index": int(raw_lineage["source_row_index"]),
                    "source_column_index": int(raw_lineage["source_column_index"]),
                    "source_product_code": code,
                    "parser_version": EEX_FORWARD_PARSER_VERSION,
                    "parser_code_sha256": parser_code_hash,
                    "parser_config_sha256": parser_config_hash,
                    "revision_timestamp": available_iso,
                    "ingestion_run_id": str(spec["intake_id"]),
                    "schema_version": EEX_HISTORICAL_QUOTE_SCHEMA,
                }
            )
    actual_quotes.sort(key=_quote_sort_key)
    if actual_quotes != list(spec["expected_quotes"]):
        raise EexForwardVintageIntakeError(
            "EEX workbook quote inventory differs from the frozen inventory"
        )
    previous = _load_verified_previous_history(
        catalog_path=previous_catalog_path,
        history_path=previous_history_path,
        acquisition_public_key_path=previous_catalog_public_key_path,
        acquisition_public_key_dir=previous_catalog_public_key_dir,
        trusted_time_public_key_path=trusted_time_public_key_path,
        trusted_time_public_key_dir=trusted_time_public_key_dir,
        trusted_time_journal_id=trusted_time_journal_id,
    )
    _verify_journal_extension(
        unsigned_receipt,
        previous_catalog=previous.catalog,
    )
    if not previous.history.empty:
        for (snapshot_date, market), group in pd.DataFrame(parsed_rows).groupby(
            ["date", "market"], sort=True
        ):
            previous_same_snapshot = previous.history.loc[
                pd.to_datetime(previous.history["date"])
                .dt.normalize()
                .eq(pd.Timestamp(snapshot_date).normalize())
                & previous.history["market"].astype(str).eq(str(market))
            ]
            if previous_same_snapshot.empty:
                continue
            previous_identities = {
                (str(row.load_type), str(row.product))
                for row in previous_same_snapshot[["load_type", "product"]].itertuples(
                    index=False
                )
            }
            new_identities = {
                (str(row.load_type), str(row.product))
                for row in group[["load_type", "product"]].itertuples(index=False)
            }
            if new_identities != previous_identities:
                raise EexForwardVintageIntakeError(
                    "EEX same-date quote inventory changed; explicit governed "
                    "tombstones or a schema migration are required"
                )
    previous_by_identity: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    if not previous.history.empty:
        ordered = previous.history.sort_values(
            [
                "date",
                "market",
                "load_type",
                "product",
                "revision_sequence",
                "available_at",
            ],
            kind="mergesort",
        )
        for identity, group in ordered.groupby(
            ["date", "market", "load_type", "product"], sort=True
        ):
            previous_by_identity[
                (
                    pd.Timestamp(identity[0]).strftime("%Y-%m-%d"),
                    str(identity[1]),
                    str(identity[2]),
                    str(identity[3]),
                )
            ] = group.iloc[-1].to_dict()
    new_rows: list[dict[str, object]] = []
    for row in parsed_rows:
        identity = (
            pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            str(row["market"]),
            str(row["load_type"]),
            str(row["product"]),
        )
        parent = previous_by_identity.get(identity)
        if parent is None:
            row["revision_sequence"] = 1
            row["supersedes_quote_id"] = ""
        else:
            if available_at <= pd.Timestamp(parent["available_at"]):
                raise EexForwardVintageIntakeError(
                    "EEX revision availability is not strictly monotone"
                )
            row["revision_sequence"] = int(parent["revision_sequence"]) + 1
            row["supersedes_quote_id"] = str(parent["quote_id"])
        row["snapshot_id"] = historical_vintage_snapshot_id(row)
        row["revision_id"] = historical_vintage_revision_id(row)
        row["row_hash"] = historical_vintage_row_hash(row)
        row["quote_id"] = row["row_hash"]
        new_rows.append(row)
    new_frame = pd.DataFrame(new_rows)
    combined = (
        new_frame
        if previous.history.empty
        else pd.concat([previous.history, new_frame], ignore_index=True)
    )
    combined = validate_eex_historical_vintage_frame(combined)
    new_snapshot_ids = tuple(sorted(set(new_frame["snapshot_id"].astype(str))))
    result = EexForwardVintageIntakeResult(
        intake_id=str(spec["intake_id"]),
        acquisition_id=str(unsigned_receipt["receipt_id"]),
        available_at_utc=available_iso,
        source_document_id=str(spec["source_document_id"]),
        source_document_sha256=source_hash,
        source_document_size_bytes=len(source_bytes),
        trusted_time_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        parser_code_sha256=parser_code_hash,
        parser_config_sha256=parser_config_hash,
        parser_config=parser_config,
        trusted_time_receipt=receipt,
        new_row_count=len(new_frame),
        total_row_count=len(combined),
        snapshot_ids=new_snapshot_ids,
    )
    return combined, result


def _verify_receipt(
    payload: bytes,
    *,
    source_document_id: str,
    source_document_sha256: str,
    source_document_size_bytes: int,
    trusted_public_key_path: str | Path | None,
    trusted_public_key_dir: str | Path | None,
    trusted_journal_id: str | None,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        receipt = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EexForwardVintageIntakeError(
            "EEX trusted-time receipt is not strict JSON"
        ) from exc
    if not isinstance(receipt, dict) or set(receipt) != _TRUSTED_RECEIPT_KEYS:
        raise EexForwardVintageIntakeError("EEX trusted-time receipt fields are not exact")
    attestation = receipt.get("trusted_time_attestation")
    if not isinstance(attestation, Mapping) or set(attestation) != _ATTESTATION_KEYS:
        raise EexForwardVintageIntakeError(
            "EEX trusted-time attestation fields are not exact"
        )
    try:
        unsigned = verify_trusted_time_receipt(
            receipt,
            trusted_public_key_path=trusted_public_key_path,
            trusted_public_key_dir=trusted_public_key_dir,
            trusted_journal_id=trusted_journal_id,
        )
    except ValueError as exc:
        raise EexForwardVintageIntakeError(
            "EEX trusted-time receipt authentication failed"
        ) from exc
    if unsigned.get("schema_version") != TRUSTED_TIME_RECEIPT_SCHEMA:
        raise EexForwardVintageIntakeError("EEX trusted-time receipt schema mismatch")
    identity = dict(unsigned)
    receipt_id = str(identity.pop("receipt_id", ""))
    if receipt_id != hashlib.sha256(canonical_json_bytes(identity)).hexdigest():
        raise EexForwardVintageIntakeError("EEX trusted-time receipt_id mismatch")
    expected = {
        "source_document_id": source_document_id,
        "source_document_sha256": source_document_sha256,
        "size_bytes": source_document_size_bytes,
    }
    if any(unsigned.get(key) != value for key, value in expected.items()):
        raise EexForwardVintageIntakeError("EEX trusted-time source binding mismatch")
    sequence = unsigned.get("journal_sequence")
    previous = str(unsigned.get("previous_receipt_id", ""))
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise EexForwardVintageIntakeError("EEX trusted-time sequence is invalid")
    if (sequence == 1 and previous) or (
        sequence > 1 and not _SHA256.fullmatch(previous)
    ):
        raise EexForwardVintageIntakeError("EEX trusted-time parent is invalid")
    received = pd.Timestamp(unsigned.get("received_at_utc"))
    if received.tzinfo is None:
        raise EexForwardVintageIntakeError("EEX trusted time must be timezone-aware")
    unsigned = dict(unsigned)
    unsigned["received_at_utc"] = received.tz_convert("UTC").isoformat()
    return receipt, unsigned


def inspect_eex_forward_workbook_latest_market_row(
    source_bytes: bytes,
    *,
    market: str,
) -> tuple[str, list[dict[str, str]]]:
    """Reject silent fallback, malformed marks, and unknown quoted products."""

    try:
        raw = pd.read_excel(BytesIO(source_bytes), sheet_name=market, header=None)
    except Exception as exc:
        raise EexForwardVintageIntakeError("EEX workbook physical-row audit failed") from exc
    if raw.shape[0] < 4 or raw.shape[1] < 2:
        raise EexForwardVintageIntakeError("EEX workbook physical-row layout is invalid")
    dates = pd.to_datetime(raw.iloc[3:, 0], dayfirst=True, errors="coerce")
    valid_dates = dates.dropna().dt.normalize()
    if valid_dates.empty:
        raise EexForwardVintageIntakeError("EEX workbook has no dated market row")
    latest = pd.Timestamp(valid_dates.max()).normalize()
    positions = dates.index[dates.dt.normalize().eq(latest)].tolist()
    if len(positions) != 1:
        raise EexForwardVintageIntakeError(
            "EEX workbook repeats the latest physical market row"
        )
    row_position = int(positions[0])
    quotes: list[dict[str, str]] = []
    source_identities: set[str] = set()
    economic_identities: set[tuple[str, str]] = set()
    for column_position in range(1, raw.shape[1]):
        raw_code = raw.iloc[0, column_position]
        raw_value = raw.iloc[row_position, column_position]
        if pd.isna(raw_code) or not str(raw_code).strip():
            if not _is_blank(raw_value):
                raise EexForwardVintageIntakeError(
                    "EEX workbook has a quoted value without a product code"
                )
            continue
        code = str(raw_code).strip().upper()
        normalized = ingest_forwards.normalize_eex_product_code(code)
        if normalized is None:
            if not _is_blank(raw_value):
                raise EexForwardVintageIntakeError(
                    "EEX workbook contains an unknown quoted product"
                )
            continue
        product, load_type, product_type = normalized
        if product_type == "Week":
            continue
        if product_type.upper() not in {"CAL", "QUARTER", "MONTH"}:
            raise EexForwardVintageIntakeError(
                "EEX workbook contains an unsupported LT product type"
            )
        economic_identity = (load_type, product)
        if code in source_identities or economic_identity in economic_identities:
            raise EexForwardVintageIntakeError(
                "EEX workbook repeats a prospective product identity"
            )
        source_identities.add(code)
        economic_identities.add(economic_identity)
        price = ingest_forwards._coerce_price(raw_value)
        if price is None:
            if _is_blank(raw_value):
                continue
            raise EexForwardVintageIntakeError(
                "EEX latest physical row contains an invalid quoted price"
            )
        quotes.append(
            {
                "market": market,
                "source_product_code": code,
                "product": product,
                "load_type": load_type,
                "product_type": product_type.upper(),
            }
        )
    quotes.sort(key=_quote_sort_key)
    if not quotes:
        raise EexForwardVintageIntakeError(
            "EEX latest physical row contains no admissible LT quote"
        )
    return latest.strftime("%Y-%m-%d"), quotes


def _is_blank(value: object) -> bool:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return True
    return not str(value).strip()


@dataclass(frozen=True)
class _PreviousVintageState:
    history: pd.DataFrame
    catalog: Mapping[str, object] | None


def _load_verified_previous_history(
    *,
    catalog_path: str | Path | None,
    history_path: str | Path | None,
    acquisition_public_key_path: str | Path | None,
    acquisition_public_key_dir: str | Path | None,
    trusted_time_public_key_path: str | Path | None,
    trusted_time_public_key_dir: str | Path | None,
    trusted_time_journal_id: str | None,
) -> _PreviousVintageState:
    if catalog_path is None and history_path is None:
        return _PreviousVintageState(history=pd.DataFrame(), catalog=None)
    if catalog_path is None or history_path is None:
        raise EexForwardVintageIntakeError(
            "previous EEX catalog and history paths must be supplied together"
        )
    if not Path(history_path).is_absolute():
        raise EexForwardVintageIntakeError(
            "previous EEX history path must be absolute"
        )
    try:
        history_size = Path(history_path).stat().st_size
    except OSError as exc:
        raise EexForwardVintageIntakeError(
            "previous EEX history is unavailable"
        ) from exc
    if history_size < 1 or history_size > _MAX_HISTORY_BYTES:
        raise EexForwardVintageIntakeError("previous EEX history size is invalid")
    catalog_payload = _read_absolute_stable_file(
        catalog_path,
        label="previous EEX vintage catalog",
        max_bytes=_MAX_CATALOG_BYTES,
    )
    try:
        catalog = load_strict_json(catalog_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EexForwardVintageIntakeError(
            "previous EEX vintage catalog is not strict JSON"
        ) from exc
    if not isinstance(catalog, Mapping):
        raise EexForwardVintageIntakeError(
            "previous EEX vintage catalog must be a mapping"
        )
    try:
        history, evidence = verify_eex_historical_vintage_catalog(
            catalog,
            catalog_path=catalog_path,
            history_path=history_path,
            acquisition_public_key_path=acquisition_public_key_path,
            acquisition_public_key_dir=acquisition_public_key_dir,
            trusted_time_public_key_path=trusted_time_public_key_path,
            trusted_time_public_key_dir=trusted_time_public_key_dir,
            trusted_time_journal_id=trusted_time_journal_id,
        )
    except ValueError as exc:
        raise EexForwardVintageIntakeError(
            "previous EEX vintage catalog authentication failed"
        ) from exc
    if not evidence.verified:
        raise EexForwardVintageIntakeError(
            "previous EEX vintage catalog is not verified"
        )
    return _PreviousVintageState(history=history, catalog=dict(catalog))


def _verify_journal_extension(
    receipt: Mapping[str, object],
    *,
    previous_catalog: Mapping[str, object] | None,
) -> None:
    sequence = int(receipt["journal_sequence"])
    previous_receipt_id = str(receipt["previous_receipt_id"])
    if previous_catalog is None:
        if sequence != 1 or previous_receipt_id:
            raise EexForwardVintageIntakeError(
                "initial EEX intake must be the trusted-time journal genesis"
            )
        return
    documents = previous_catalog.get("source_documents")
    if not isinstance(documents, list) or not documents:
        raise EexForwardVintageIntakeError(
            "previous EEX catalog has no source receipt chain"
        )
    prior_receipts: list[Mapping[str, object]] = []
    prior_document_ids: set[str] = set()
    prior_document_hashes: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise EexForwardVintageIntakeError(
                "previous EEX catalog source entry is invalid"
            )
        prior_document_ids.add(str(document.get("source_document_id", "")))
        prior_document_hashes.add(str(document.get("sha256", "")))
        prior = document.get("trusted_time_receipt")
        if not isinstance(prior, Mapping):
            raise EexForwardVintageIntakeError(
                "previous EEX catalog source receipt is missing"
            )
        prior_receipts.append(prior)
    if (
        str(receipt["source_document_id"]) in prior_document_ids
        or str(receipt["source_document_sha256"]) in prior_document_hashes
    ):
        raise EexForwardVintageIntakeError(
            "EEX journal extension repeats prior source evidence"
        )
    head = max(prior_receipts, key=lambda value: int(value["journal_sequence"]))
    if (
        str(receipt["journal_id"]) != str(head["journal_id"])
        or sequence != int(head["journal_sequence"]) + 1
        or previous_receipt_id != str(head["receipt_id"])
        or pd.Timestamp(receipt["received_at_utc"])
        <= pd.Timestamp(head["received_at_utc"])
    ):
        raise EexForwardVintageIntakeError(
            "EEX trusted-time receipt does not extend the authenticated journal head"
        )


def _read_absolute_stable_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise EexForwardVintageIntakeError(f"{label} path must be absolute")
    try:
        selected = assert_absolute_path_has_no_links(candidate)
        size = selected.stat().st_size
        if size < 1 or size > max_bytes:
            raise EexForwardVintageIntakeError(f"{label} size is invalid")
        payload = read_stable_single_link_file(
            selected,
            label=label,
            max_bytes=max_bytes,
        )
        if len(payload) != size or len(payload) > max_bytes:
            raise EexForwardVintageIntakeError(f"{label} size changed during read")
        return payload
    except EexForwardVintageIntakeError:
        raise
    except (OSError, ValueError) as exc:
        raise EexForwardVintageIntakeError(f"{label} is not a stable regular file") from exc


def preflight_eex_forward_workbook_bytes(payload: bytes) -> None:
    """Reject unsafe or resource-exhausting XLSX ZIP containers."""

    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            if not members or len(members) > _MAX_XLSX_ENTRIES:
                raise EexForwardVintageIntakeError("EEX XLSX member count is invalid")
            expanded = 0
            for member in members:
                normalized = member.filename.replace("\\", "/")
                if (
                    member.is_dir()
                    or normalized.startswith("/")
                    or ":" in normalized
                    or any(part in {"", ".", ".."} for part in normalized.split("/"))
                ):
                    raise EexForwardVintageIntakeError(
                        "EEX XLSX contains an unsafe member path"
                    )
                if member.flag_bits & 0x1:
                    raise EexForwardVintageIntakeError(
                        "EEX XLSX contains an encrypted member"
                    )
                if member.file_size < 0 or member.file_size > _MAX_XLSX_MEMBER_BYTES:
                    raise EexForwardVintageIntakeError(
                        "EEX XLSX member size is invalid"
                    )
                if member.file_size and member.compress_size == 0:
                    raise EexForwardVintageIntakeError(
                        "EEX XLSX member compression metadata is invalid"
                    )
                if (
                    member.compress_size
                    and member.file_size / member.compress_size
                    > _MAX_XLSX_COMPRESSION_RATIO
                ):
                    raise EexForwardVintageIntakeError(
                        "EEX XLSX compression ratio is unsafe"
                    )
                expanded += member.file_size
                if expanded > _MAX_XLSX_EXPANDED_BYTES:
                    raise EexForwardVintageIntakeError(
                        "EEX XLSX expanded size budget is exceeded"
                    )
    except EexForwardVintageIntakeError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise EexForwardVintageIntakeError("EEX source document is not a valid XLSX") from exc


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value or "")
    if not _SHA256.fullmatch(normalized):
        raise EexForwardVintageIntakeError(f"{label} SHA-256 is invalid")
    return normalized


def _string_sequence(value: object, *, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EexForwardVintageIntakeError(f"{label} must be a list")
    result = [str(item) for item in value]
    if not result or any(not item for item in result):
        raise EexForwardVintageIntakeError(f"{label} contains an empty value")
    return result


def _quote_sort_key(value: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(value["market"]),
        str(value["product_type"]),
        str(value["product"]),
        str(value["load_type"]),
        str(value["source_product_code"]),
    )


def _portable_relative_path(value: str, *, label: str) -> str:
    raw = str(value).replace("\\", "/")
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or raw.startswith("/")
        or ":" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise EexForwardVintageIntakeError(f"{label} is not a portable relative path")
    return raw
