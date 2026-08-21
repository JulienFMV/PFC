"""Semantic contract for immutable EEX historical quote vintages."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.acquisition_contract import (
    TRUSTED_TIME_RECEIPT_SCHEMA,
    verify_acquisition_contract,
    verify_trusted_time_receipt,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
)

EEX_HISTORICAL_QUOTE_SCHEMA = "eex_historical_vintage.v1"
EEX_HISTORICAL_CATALOG_SCHEMA = "eex_historical_vintage_catalog.v1"
EEX_HISTORICAL_REQUIRED_COLUMNS = {
    "date",
    "observed_at",
    "available_at",
    "acquisition_id",
    "snapshot_id",
    "product",
    "load_type",
    "product_type",
    "price",
    "unit",
    "market",
    "source",
    "source_document_id",
    "source_document_sha256",
    "source_sheet",
    "source_row_index",
    "source_column_index",
    "source_product_code",
    "parser_version",
    "parser_code_sha256",
    "parser_config_sha256",
    "revision_id",
    "revision_sequence",
    "supersedes_quote_id",
    "revision_timestamp",
    "ingestion_run_id",
    "schema_version",
    "quote_id",
    "row_hash",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_CATALOG_TOKEN = object()
_CATALOG_UNSIGNED_KEYS = {
    "schema_version",
    "catalog_id",
    "created_at_utc",
    "data_cutoff_utc",
    "calibration_eligible",
    "source_class",
    "revision_policy",
    "ompex_used",
    "history_parquet",
    "source_documents",
}
_CATALOG_SIGNED_KEYS = {*_CATALOG_UNSIGNED_KEYS, "acquisition_attestation"}
_CATALOG_ATTESTATION_KEYS = {
    "algorithm",
    "key_id",
    "payload_sha256",
    "value_base64",
}
_HISTORY_ENTRY_KEYS = {"path", "sha256", "size_bytes", "row_count"}
_SOURCE_DOCUMENT_ENTRY_KEYS = {
    "snapshot_ids",
    "acquisition_id",
    "observed_at",
    "available_at",
    "source_document_id",
    "path",
    "sha256",
    "size_bytes",
    "parser_code_path",
    "parser_code_sha256",
    "parser_config",
    "parser_config_sha256",
    "trusted_time_receipt",
}
_TRUSTED_TIME_RECEIPT_KEYS = {
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
_TRUSTED_TIME_ATTESTATION_KEYS = _CATALOG_ATTESTATION_KEYS
_LEGACY_PARSER_CONFIG_KEYS = {"parser_version", "selection_mode", "markets"}
_PROSPECTIVE_PARSER_CONFIG_KEYS = {
    *_LEGACY_PARSER_CONFIG_KEYS,
    "quote_convention",
    "expected_quotes",
    "intake_id",
}
_MAX_CATALOG_BYTES = 4 * 1024 * 1024
_MAX_HISTORY_BYTES = 512 * 1024 * 1024
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_PARSER_BYTES = 2 * 1024 * 1024


class EexHistoricalVintageError(ValueError):
    """Raised when reconstructed history is presented as a PIT vintage set."""


@dataclass(frozen=True)
class VerifiedEexHistoricalVintageCatalog:
    catalog_id: str
    catalog_sha256: str
    history_sha256: str
    snapshot_count: int
    source_document_count: int
    data_cutoff_utc: str
    _verification_token: InitVar[object | None] = None
    verified: bool = field(init=False)

    def __post_init__(self, _verification_token: object | None) -> None:
        object.__setattr__(
            self,
            "verified",
            _verification_token is _VERIFIED_CATALOG_TOKEN,
        )


def historical_vintage_snapshot_id(row: Mapping[str, object]) -> str:
    """Return the immutable source-vintage identity for one normalized row."""

    return _sha256_json(
        {
            "schema_version": EEX_HISTORICAL_QUOTE_SCHEMA,
            "acquisition_id": str(row["acquisition_id"]),
            "source_snapshot_date": _date_iso(row["date"]),
            "observed_at": _aware_utc_iso(row["observed_at"], label="observed_at"),
            "available_at": _aware_utc_iso(row["available_at"], label="available_at"),
            "source_document_id": str(row["source_document_id"]),
            "source_document_sha256": str(row["source_document_sha256"]).lower(),
            "parser_version": str(row["parser_version"]),
            "parser_code_sha256": str(row["parser_code_sha256"]).lower(),
            "parser_config_sha256": str(row["parser_config_sha256"]).lower(),
            "ingestion_run_id": str(row["ingestion_run_id"]),
        }
    )


def historical_vintage_revision_id(row: Mapping[str, object]) -> str:
    """Return the identity of a quote revision before its row hash is known."""

    return _sha256_json(
        {
            "snapshot_id": str(row["snapshot_id"]),
            "market": str(row["market"]).upper(),
            "load_type": str(row["load_type"]).upper(),
            "product": str(row["product"]),
            "revision_sequence": int(row["revision_sequence"]),
            "supersedes_quote_id": str(row.get("supersedes_quote_id") or ""),
            "revision_timestamp": _aware_utc_iso(
                row["revision_timestamp"],
                label="revision_timestamp",
            ),
        }
    )


def historical_vintage_row_hash(row: Mapping[str, object]) -> str:
    """Hash the complete semantic quote observation, excluding self hashes."""

    return _sha256_json(
        {
            "schema_version": str(row["schema_version"]),
            "date": _date_iso(row["date"]),
            "observed_at": _aware_utc_iso(row["observed_at"], label="observed_at"),
            "available_at": _aware_utc_iso(row["available_at"], label="available_at"),
            "acquisition_id": str(row["acquisition_id"]),
            "snapshot_id": str(row["snapshot_id"]),
            "market": str(row["market"]).upper(),
            "load_type": str(row["load_type"]).upper(),
            "product_type": str(row["product_type"]),
            "product": str(row["product"]),
            "price": float(row["price"]),
            "unit": str(row["unit"]),
            "source": str(row["source"]).upper(),
            "source_document_id": str(row["source_document_id"]),
            "source_document_sha256": str(row["source_document_sha256"]).lower(),
            "source_sheet": str(row["source_sheet"]),
            "source_row_index": int(row["source_row_index"]),
            "source_column_index": int(row["source_column_index"]),
            "source_product_code": str(row["source_product_code"]),
            "parser_version": str(row["parser_version"]),
            "parser_code_sha256": str(row["parser_code_sha256"]).lower(),
            "parser_config_sha256": str(row["parser_config_sha256"]).lower(),
            "revision_id": str(row["revision_id"]),
            "revision_sequence": int(row["revision_sequence"]),
            "supersedes_quote_id": str(row.get("supersedes_quote_id") or ""),
            "revision_timestamp": _aware_utc_iso(
                row["revision_timestamp"],
                label="revision_timestamp",
            ),
            "ingestion_run_id": str(row["ingestion_run_id"]),
        }
    )


def validate_eex_historical_vintage_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Validate immutable, bitemporal EEX vintages used by rolling-origin work."""

    missing = sorted(EEX_HISTORICAL_REQUIRED_COLUMNS.difference(history.columns))
    if missing:
        raise EexHistoricalVintageError(
            f"EEX rolling-origin history is not vintage-verifiable; missing columns: {missing}"
        )
    frame = history.copy()
    if frame.empty:
        raise EexHistoricalVintageError("EEX rolling-origin history is empty")

    for column in (
        "acquisition_id",
        "snapshot_id",
        "product",
        "load_type",
        "product_type",
        "unit",
        "market",
        "source",
        "source_document_id",
        "source_document_sha256",
        "source_sheet",
        "source_product_code",
        "parser_version",
        "parser_code_sha256",
        "parser_config_sha256",
        "revision_id",
        "ingestion_run_id",
        "schema_version",
        "quote_id",
        "row_hash",
    ):
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or bool(frame[column].eq("").any()):
            raise EexHistoricalVintageError(f"EEX vintage history contains empty {column}")
    frame["supersedes_quote_id"] = (
        frame["supersedes_quote_id"].fillna("").astype("string").str.strip()
    )
    for column in ("market", "load_type", "product_type", "source"):
        frame[column] = frame[column].str.upper()
    if set(frame["source"].astype(str)) != {"EEX"}:
        raise EexHistoricalVintageError(
            "EEX vintage history forbids OMPEX, benchmark and proxy sources"
        )
    if set(frame["unit"].astype(str)) != {"EUR/MWH"}:
        raise EexHistoricalVintageError("EEX vintage history unit must be EUR/MWH")
    if set(frame["schema_version"].astype(str)) != {EEX_HISTORICAL_QUOTE_SCHEMA}:
        raise EexHistoricalVintageError("EEX vintage history schema mismatch")
    if not set(frame["market"].astype(str)).issubset({"CH", "DE", "FR", "AT", "IT"}):
        raise EexHistoricalVintageError("EEX vintage history contains unsupported markets")
    if not set(frame["load_type"].astype(str)).issubset({"BASE", "PEAK", "OFFPEAK"}):
        raise EexHistoricalVintageError("EEX vintage history contains unsupported load types")
    products = frame["product"].astype(str)
    expected_types = pd.Series("", index=frame.index, dtype="string")
    expected_types.loc[products.str.match(r"^\d{4}$")] = "CAL"
    expected_types.loc[products.str.match(r"^\d{4}-Q[1-4]$")] = "QUARTER"
    expected_types.loc[products.str.match(r"^\d{4}-(0[1-9]|1[0-2])$")] = "MONTH"
    if expected_types.eq("").any():
        raise EexHistoricalVintageError("EEX vintage history contains invalid products")
    if not frame["product_type"].astype(str).eq(expected_types.astype(str)).all():
        raise EexHistoricalVintageError("EEX vintage product/product_type mismatch")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    if frame["date"].isna().any():
        raise EexHistoricalVintageError("EEX vintage history contains invalid snapshot dates")
    for column in ("observed_at", "available_at", "revision_timestamp"):
        frame[column] = _aware_utc_series(frame[column], label=column)
    date_utc = frame["date"].dt.tz_localize("UTC")
    if bool((frame["observed_at"] < date_utc).any()):
        raise EexHistoricalVintageError("EEX vintage was observed before its source snapshot date")
    if bool((frame["available_at"] < frame["observed_at"]).any()):
        raise EexHistoricalVintageError("EEX vintage availability predates observation")
    if bool((frame["revision_timestamp"] > frame["available_at"]).any()):
        raise EexHistoricalVintageError("EEX quote revision was unavailable at vintage cutoff")

    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if not np.isfinite(frame["price"].to_numpy(dtype=float)).all():
        raise EexHistoricalVintageError("EEX vintage history contains non-finite prices")
    for column in ("source_row_index", "source_column_index", "revision_sequence"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or bool((frame[column] < 0).any()):
            raise EexHistoricalVintageError(f"EEX vintage history contains invalid {column}")
        if bool((frame[column] % 1 != 0).any()):
            raise EexHistoricalVintageError(f"EEX vintage history contains fractional {column}")
        frame[column] = frame[column].astype(int)
    if bool((frame["revision_sequence"] < 1).any()):
        raise EexHistoricalVintageError("EEX revision_sequence must start at one")

    for column in (
        "snapshot_id",
        "source_document_sha256",
        "parser_code_sha256",
        "parser_config_sha256",
        "revision_id",
        "quote_id",
        "row_hash",
    ):
        if not frame[column].astype(str).map(lambda value: bool(_SHA256_RE.fullmatch(value))).all():
            raise EexHistoricalVintageError(f"EEX vintage history contains invalid {column}")

    snapshot_fields = [
        "date",
        "observed_at",
        "available_at",
        "acquisition_id",
        "source_document_id",
        "source_document_sha256",
        "parser_version",
        "parser_code_sha256",
        "parser_config_sha256",
        "ingestion_run_id",
    ]
    for snapshot_id, group in frame.groupby("snapshot_id", sort=True):
        if any(group[column].nunique(dropna=False) != 1 for column in snapshot_fields):
            raise EexHistoricalVintageError(
                f"EEX snapshot {snapshot_id} has inconsistent source-vintage metadata"
            )
        expected = historical_vintage_snapshot_id(group.iloc[0].to_dict())
        if str(snapshot_id) != expected:
            raise EexHistoricalVintageError("EEX vintage snapshot_id mismatch")
    identity = ["snapshot_id", "market", "load_type", "product"]
    if frame.duplicated(identity, keep=False).any():
        raise EexHistoricalVintageError("EEX vintage history contains duplicate quote identities")
    lineage = ["snapshot_id", "source_sheet", "source_row_index", "source_column_index"]
    if frame.duplicated(lineage, keep=False).any():
        raise EexHistoricalVintageError("EEX vintage history contains duplicate source cells")
    origin_identity = ["available_at", "market", "load_type", "product"]
    if frame.duplicated(origin_identity, keep=False).any():
        raise EexHistoricalVintageError(
            "EEX vintage history repeats an economic quote at one availability cutoff"
        )

    preliminary_quote_rows = {
        str(row["quote_id"]): row for row in frame.to_dict(orient="records")
    }
    for row in frame.to_dict(orient="records"):
        sequence = int(row["revision_sequence"])
        parent = str(row.get("supersedes_quote_id") or "")
        if sequence > 1 and parent not in preliminary_quote_rows:
            raise EexHistoricalVintageError("EEX quote revision parent is missing")
        if sequence > 1:
            parent_row = preliminary_quote_rows[parent]
            identity_columns = ("date", "market", "load_type", "product")
            if any(str(row[column]) != str(parent_row[column]) for column in identity_columns):
                raise EexHistoricalVintageError("EEX quote revision changes quote identity")

    revision_identity = ["date", "market", "load_type", "product"]
    for _, revisions in frame.groupby(revision_identity, sort=True):
        ordered = revisions.sort_values("revision_sequence", kind="mergesort")
        sequences = ordered["revision_sequence"].astype(int).tolist()
        if sequences != list(range(1, len(ordered) + 1)):
            raise EexHistoricalVintageError(
                "EEX quote revision chain must have one root and contiguous sequence"
            )
        rows = ordered.to_dict(orient="records")
        if str(rows[0].get("supersedes_quote_id") or ""):
            raise EexHistoricalVintageError("EEX quote revision root cannot have a parent")
        for parent, child in zip(rows[:-1], rows[1:], strict=True):
            if str(child.get("supersedes_quote_id") or "") != str(parent["quote_id"]):
                raise EexHistoricalVintageError(
                    "EEX quote revision chain is not fully linked"
                )

    quote_rows = {
        str(row["quote_id"]): row
        for row in frame.to_dict(orient="records")
    }
    quote_ids = set(quote_rows)
    revision_parents: set[str] = set()
    for row in frame.to_dict(orient="records"):
        expected_revision = historical_vintage_revision_id(row)
        if str(row["revision_id"]) != expected_revision:
            raise EexHistoricalVintageError("EEX vintage revision_id mismatch")
        expected_row = historical_vintage_row_hash(row)
        if str(row["row_hash"]) != expected_row or str(row["quote_id"]) != expected_row:
            raise EexHistoricalVintageError("EEX vintage quote row hash mismatch")
        sequence = int(row["revision_sequence"])
        parent = str(row.get("supersedes_quote_id") or "")
        if sequence == 1 and parent:
            raise EexHistoricalVintageError("first EEX quote revision cannot supersede another quote")
        if sequence > 1 and parent not in quote_ids:
            raise EexHistoricalVintageError("EEX quote revision parent is missing")
        if sequence > 1:
            parent_row = quote_rows[parent]
            identity_columns = ("date", "market", "load_type", "product")
            if any(str(row[column]) != str(parent_row[column]) for column in identity_columns):
                raise EexHistoricalVintageError("EEX quote revision changes quote identity")
            if int(parent_row["revision_sequence"]) != sequence - 1:
                raise EexHistoricalVintageError("EEX quote revision sequence is not contiguous")
            if pd.Timestamp(parent_row["revision_timestamp"]) >= pd.Timestamp(
                row["revision_timestamp"]
            ):
                raise EexHistoricalVintageError("EEX quote revision timestamps are not ordered")
            if pd.Timestamp(parent_row["available_at"]) >= pd.Timestamp(row["available_at"]):
                raise EexHistoricalVintageError("EEX quote revision availability is not ordered")
            if parent in revision_parents:
                raise EexHistoricalVintageError("EEX quote revision chain contains a fork")
            revision_parents.add(parent)

    frame = frame.sort_values(
        ["available_at", "snapshot_id", "market", "load_type", "product"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.attrs["rolling_origin_provenance"] = {
        "schema_version": EEX_HISTORICAL_QUOTE_SCHEMA,
        "snapshot_count": int(frame["snapshot_id"].nunique()),
        "available_at_min": frame["available_at"].min().isoformat(),
        "available_at_max": frame["available_at"].max().isoformat(),
        "source_document_count": int(frame["source_document_sha256"].nunique()),
        "revision_count": int((frame["revision_sequence"] > 1).sum()),
    }
    return frame


def verify_eex_historical_vintage_catalog(
    catalog: Mapping[str, object],
    *,
    catalog_path: str | Path,
    history_path: str | Path,
    acquisition_public_key_path: str | Path | None = None,
    acquisition_public_key_dir: str | Path | None = None,
    trusted_time_public_key_path: str | Path | None = None,
    trusted_time_public_key_dir: str | Path | None = None,
    trusted_time_journal_id: str | None = None,
) -> tuple[pd.DataFrame, VerifiedEexHistoricalVintageCatalog]:
    """Authenticate a complete immutable vintage catalog and its exact bytes."""

    catalog_file = assert_absolute_path_has_no_links(Path(catalog_path).absolute())
    try:
        catalog_bytes = read_stable_single_link_file(
            catalog_file,
            label="EEX vintage catalog",
            max_bytes=_MAX_CATALOG_BYTES,
        )
        persisted_catalog = load_strict_json(catalog_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, StrictStructuredDataError) as exc:
        raise EexHistoricalVintageError("EEX vintage catalog file is unreadable") from exc
    if persisted_catalog != dict(catalog):
        raise EexHistoricalVintageError("EEX vintage catalog payload/path mismatch")
    if set(catalog) != _CATALOG_SIGNED_KEYS:
        raise EexHistoricalVintageError("EEX vintage catalog fields are not exact")
    catalog_attestation = catalog.get("acquisition_attestation")
    if (
        not isinstance(catalog_attestation, Mapping)
        or set(catalog_attestation) != _CATALOG_ATTESTATION_KEYS
    ):
        raise EexHistoricalVintageError(
            "EEX vintage catalog attestation fields are not exact"
        )
    try:
        unsigned = verify_acquisition_contract(
            catalog,
            trusted_public_key_path=acquisition_public_key_path,
            trusted_public_key_dir=acquisition_public_key_dir,
        )
    except ValueError as exc:
        raise EexHistoricalVintageError("EEX vintage catalog attestation is invalid") from exc
    if unsigned.get("schema_version") != EEX_HISTORICAL_CATALOG_SCHEMA:
        raise EexHistoricalVintageError("EEX vintage catalog schema mismatch")
    if set(unsigned) != _CATALOG_UNSIGNED_KEYS:
        raise EexHistoricalVintageError("EEX vintage unsigned fields are not exact")
    if unsigned.get("calibration_eligible") is not True:
        raise EexHistoricalVintageError("EEX vintage catalog is not calibration eligible")
    if unsigned.get("source_class") != "IMMUTABLE_DAILY_EEX_XLSX":
        raise EexHistoricalVintageError("EEX vintage catalog source class is not admissible")
    if unsigned.get("revision_policy") != "APPEND_ONLY_PRESERVE_ALL_REVISIONS":
        raise EexHistoricalVintageError("EEX vintage catalog revision policy is not append-only")
    if unsigned.get("ompex_used") is not False:
        raise EexHistoricalVintageError("EEX vintage catalog must exclude OMPEX")

    identity_payload = dict(unsigned)
    catalog_id = str(identity_payload.pop("catalog_id", ""))
    expected_catalog_id = _sha256_json(identity_payload)
    if catalog_id != expected_catalog_id:
        raise EexHistoricalVintageError("EEX vintage catalog_id mismatch")
    catalog_root = catalog_file.parent
    selected_history = assert_absolute_path_has_no_links(Path(history_path).absolute())
    history_entry = unsigned.get("history_parquet")
    if not isinstance(history_entry, Mapping) or set(history_entry) != _HISTORY_ENTRY_KEYS:
        raise EexHistoricalVintageError("EEX vintage catalog history entry is missing")
    expected_history = _resolve_catalog_path(
        catalog_root,
        history_entry.get("path"),
        label="history parquet",
    )
    if selected_history != expected_history:
        raise EexHistoricalVintageError("EEX vintage catalog selects a different history parquet")
    try:
        history_bytes = read_stable_single_link_file(
            selected_history,
            label="EEX vintage history parquet",
            max_bytes=_MAX_HISTORY_BYTES,
        )
    except ValueError as exc:
        raise EexHistoricalVintageError("EEX vintage history parquet is unavailable") from exc
    history_sha = hashlib.sha256(history_bytes).hexdigest()
    if (
        str(history_entry.get("sha256", "")) != history_sha
        or history_entry.get("size_bytes") != len(history_bytes)
    ):
        raise EexHistoricalVintageError("EEX vintage history byte receipt mismatch")

    try:
        parsed_history = pd.read_parquet(BytesIO(history_bytes))
    except (OSError, ValueError) as exc:
        raise EexHistoricalVintageError("EEX vintage history parquet is unreadable") from exc
    frame = validate_eex_historical_vintage_frame(parsed_history)
    if history_entry.get("row_count") != len(frame):
        raise EexHistoricalVintageError("EEX vintage history row count mismatch")
    documents = unsigned.get("source_documents")
    if not isinstance(documents, list) or not documents:
        raise EexHistoricalVintageError("EEX vintage source document inventory is missing")
    by_snapshot: dict[str, Mapping[str, object]] = {}
    document_paths: set[Path] = set()
    document_ids: set[str] = set()
    document_hashes: set[str] = set()
    parser_payload_cache: dict[Path, bytes] = {}
    trusted_receipts: list[dict[str, object]] = []
    for raw in documents:
        if not isinstance(raw, Mapping) or set(raw) != _SOURCE_DOCUMENT_ENTRY_KEYS:
            raise EexHistoricalVintageError("EEX vintage source document entry is invalid")
        snapshot_ids = raw.get("snapshot_ids")
        if (
            not isinstance(snapshot_ids, list)
            or not snapshot_ids
            or len({str(value) for value in snapshot_ids}) != len(snapshot_ids)
        ):
            raise EexHistoricalVintageError(
                "EEX vintage source document snapshot inventory is invalid"
            )
        document = _resolve_catalog_path(
            catalog_root,
            raw.get("path"),
            label="source document",
        )
        if document in document_paths:
            raise EexHistoricalVintageError("EEX vintage catalog repeats a source path")
        document_paths.add(document)
        document_id = str(raw.get("source_document_id", ""))
        document_hash = str(raw.get("sha256", ""))
        if not document_id or document_id in document_ids:
            raise EexHistoricalVintageError("EEX vintage catalog repeats a source_document_id")
        if not document_hash or document_hash in document_hashes:
            raise EexHistoricalVintageError("EEX vintage catalog repeats source document bytes")
        document_ids.add(document_id)
        document_hashes.add(document_hash)
        try:
            payload = read_stable_single_link_file(
                document,
                label="EEX vintage source document",
                max_bytes=_MAX_SOURCE_BYTES,
            )
        except ValueError as exc:
            raise EexHistoricalVintageError(
                f"EEX vintage source document is unavailable: {document}"
            ) from exc
        if (
            document_hash != hashlib.sha256(payload).hexdigest()
            or raw.get("size_bytes") != len(payload)
        ):
            raise EexHistoricalVintageError("EEX vintage source document receipt mismatch")
        try:
            from pfc_shaping.data.eex_forward_vintage_intake import (
                preflight_eex_forward_workbook_bytes,
            )

            preflight_eex_forward_workbook_bytes(payload)
        except ValueError as exc:
            raise EexHistoricalVintageError(
                "EEX vintage source document XLSX preflight failed"
            ) from exc
        receipt = raw.get("trusted_time_receipt")
        if not isinstance(receipt, Mapping) or set(receipt) != _TRUSTED_TIME_RECEIPT_KEYS:
            raise EexHistoricalVintageError("EEX trusted-time acquisition receipt is missing")
        timestamp_attestation = receipt.get("trusted_time_attestation")
        if (
            not isinstance(timestamp_attestation, Mapping)
            or set(timestamp_attestation) != _TRUSTED_TIME_ATTESTATION_KEYS
        ):
            raise EexHistoricalVintageError("EEX acquisition authority metadata is missing")
        if str(catalog_attestation.get("key_id", "")) == str(
            timestamp_attestation.get("key_id", "")
        ):
            raise EexHistoricalVintageError(
                "EEX catalog and trusted-time authorities must use distinct keys"
            )
        trusted_receipts.append(
            _verify_source_trusted_time_receipt(
                receipt,
                source_entry=raw,
                source_size=len(payload),
                trusted_public_key_path=trusted_time_public_key_path,
                trusted_public_key_dir=trusted_time_public_key_dir,
                trusted_journal_id=trusted_time_journal_id,
            )
        )
        parser_code = _resolve_catalog_path(
            catalog_root,
            raw.get("parser_code_path"),
            label="parser code",
        )
        if parser_code not in parser_payload_cache:
            try:
                parser_payload_cache[parser_code] = read_stable_single_link_file(
                    parser_code,
                    label="EEX vintage parser code",
                    max_bytes=_MAX_PARSER_BYTES,
                )
            except ValueError as exc:
                raise EexHistoricalVintageError("EEX vintage parser code is unavailable") from exc
        parser_code_hash = hashlib.sha256(parser_payload_cache[parser_code]).hexdigest()
        if str(raw.get("parser_code_sha256", "")) != parser_code_hash:
            raise EexHistoricalVintageError("EEX vintage parser code receipt mismatch")
        parser_config = raw.get("parser_config")
        if not isinstance(parser_config, Mapping):
            raise EexHistoricalVintageError("EEX vintage parser config is missing")
        if frozenset(parser_config) not in {
            frozenset(_LEGACY_PARSER_CONFIG_KEYS),
            frozenset(_PROSPECTIVE_PARSER_CONFIG_KEYS),
        }:
            raise EexHistoricalVintageError(
                "EEX vintage parser config fields are not exact"
            )
        parser_config_hash = _sha256_json(parser_config)
        if str(raw.get("parser_config_sha256", "")) != parser_config_hash:
            raise EexHistoricalVintageError("EEX vintage parser config hash mismatch")
        for snapshot_id in (str(value) for value in snapshot_ids):
            if snapshot_id in by_snapshot:
                raise EexHistoricalVintageError("EEX vintage catalog repeats a snapshot_id")
            by_snapshot[snapshot_id] = raw
        _replay_source_document(
            payload,
            entry=raw,
            frame=frame[frame["snapshot_id"].astype(str).isin([str(v) for v in snapshot_ids])],
            parser_code_sha256=parser_code_hash,
            parser_config=parser_config,
            parser_config_sha256=parser_config_hash,
        )
    _validate_trusted_time_receipt_chain(trusted_receipts)
    frame_snapshots = set(frame["snapshot_id"].astype(str))
    if set(by_snapshot) != frame_snapshots:
        raise EexHistoricalVintageError("EEX vintage catalog snapshot set mismatch")
    for snapshot_id, group in frame.groupby("snapshot_id", sort=True):
        entry = by_snapshot[str(snapshot_id)]
        first = group.iloc[0]
        expected = {
            "acquisition_id": str(first["acquisition_id"]),
            "observed_at": pd.Timestamp(first["observed_at"]).isoformat(),
            "available_at": pd.Timestamp(first["available_at"]).isoformat(),
            "source_document_id": str(first["source_document_id"]),
            "sha256": str(first["source_document_sha256"]),
        }
        if any(str(entry.get(key, "")) != str(value) for key, value in expected.items()):
            raise EexHistoricalVintageError("EEX vintage catalog/frame metadata mismatch")

    cutoff = _aware_utc_iso(unsigned.get("data_cutoff_utc"), label="data_cutoff_utc")
    max_available = frame["available_at"].max().isoformat()
    if cutoff != max_available:
        raise EexHistoricalVintageError("EEX vintage catalog data cutoff mismatch")
    created = pd.Timestamp(unsigned.get("created_at_utc"))
    if created.tzinfo is None or created.tz_convert("UTC") < frame["available_at"].max():
        raise EexHistoricalVintageError("EEX vintage catalog creation timestamp is invalid")
    catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
    evidence = VerifiedEexHistoricalVintageCatalog(
        catalog_id=catalog_id,
        catalog_sha256=catalog_sha,
        history_sha256=history_sha,
        snapshot_count=len(frame_snapshots),
        source_document_count=len(documents),
        data_cutoff_utc=cutoff,
        _verification_token=_VERIFIED_CATALOG_TOKEN,
    )
    frame.attrs["verified_vintage_catalog"] = {
        "catalog_id": evidence.catalog_id,
        "catalog_sha256": evidence.catalog_sha256,
        "history_sha256": evidence.history_sha256,
        "snapshot_count": evidence.snapshot_count,
        "source_document_count": evidence.source_document_count,
        "data_cutoff_utc": evidence.data_cutoff_utc,
        "status": "VERIFIED_SIGNED_IMMUTABLE_VINTAGES",
    }
    return frame, evidence


def materialize_eex_forward_history_as_of(
    vintages: pd.DataFrame,
    *,
    valuation_timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    """Select the latest revision available at valuation for each quote identity."""

    frame = validate_eex_historical_vintage_frame(vintages)
    cutoff = pd.Timestamp(valuation_timestamp)
    if cutoff.tzinfo is None:
        raise EexHistoricalVintageError("EEX history valuation timestamp must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    eligible = frame.loc[frame["available_at"] <= cutoff].copy()
    if eligible.empty:
        raise EexHistoricalVintageError("no EEX historical vintage was available at valuation")
    identity = ["date", "market", "load_type", "product"]
    eligible = eligible.sort_values(
        [*identity, "available_at", "revision_sequence", "revision_timestamp"],
        kind="mergesort",
    )
    selected = eligible.drop_duplicates(identity, keep="last").reset_index(drop=True)
    selected.attrs["verified_vintage_catalog"] = dict(
        frame.attrs.get("verified_vintage_catalog", {})
    )
    selected.attrs["pit_selection"] = {
        "valuation_timestamp": cutoff.isoformat(),
        "selected_rows": len(selected),
        "eligible_vintage_rows": len(eligible),
        "max_available_at": selected["available_at"].max().isoformat(),
    }
    return selected


def _replay_source_document(
    source_bytes: bytes,
    *,
    entry: Mapping[str, object],
    frame: pd.DataFrame,
    parser_code_sha256: str,
    parser_config: Mapping[str, object],
    parser_config_sha256: str,
) -> None:
    if frame.empty:
        raise EexHistoricalVintageError("EEX vintage source document has no quote rows")
    from pfc_shaping.data import ingest_forwards

    runtime_parser = Path(ingest_forwards.__file__).resolve()
    runtime_parser_payload = read_stable_single_link_file(
        runtime_parser,
        label="runtime EEX parser",
        max_bytes=_MAX_PARSER_BYTES,
    )
    if hashlib.sha256(runtime_parser_payload).hexdigest() != parser_code_sha256:
        raise EexHistoricalVintageError("runtime EEX parser differs from archived parser code")
    if not frame["parser_code_sha256"].astype(str).eq(parser_code_sha256).all():
        raise EexHistoricalVintageError("EEX quote rows do not bind archived parser code")
    if not frame["parser_config_sha256"].astype(str).eq(parser_config_sha256).all():
        raise EexHistoricalVintageError("EEX quote rows do not bind parser config")
    if not frame["parser_version"].astype(str).eq("ingest_forwards.v1").all():
        raise EexHistoricalVintageError("EEX quote rows use an unsupported parser version")
    if parser_config.get("parser_version") != "ingest_forwards.v1":
        raise EexHistoricalVintageError("EEX parser config version mismatch")
    if parser_config.get("selection_mode") != "latest_available":
        raise EexHistoricalVintageError(
            "EEX historical vintages must use latest_available source rows"
        )
    configured_markets = parser_config.get("markets")
    if not isinstance(configured_markets, Mapping):
        raise EexHistoricalVintageError("EEX parser config market inventory is missing")
    actual_market_dates = {
        str(market): pd.Timestamp(group["date"].iloc[0]).date().isoformat()
        for market, group in frame.groupby("market", sort=True)
        if group["date"].nunique() == 1
    }
    if {str(key): str(value) for key, value in configured_markets.items()} != actual_market_dates:
        raise EexHistoricalVintageError("EEX parser config market/date inventory mismatch")
    prospective_config = set(parser_config) == _PROSPECTIVE_PARSER_CONFIG_KEYS
    if prospective_config:
        if parser_config.get("quote_convention") != "EEX_SETTLEMENT_EUR_MWH":
            raise EexHistoricalVintageError(
                "prospective EEX replay requires settlement quote convention"
            )
        if not _SHA256_RE.fullmatch(str(parser_config.get("intake_id", ""))):
            raise EexHistoricalVintageError("prospective EEX intake_id is invalid")
        expected_quotes = parser_config.get("expected_quotes")
        if not isinstance(expected_quotes, list) or not expected_quotes:
            raise EexHistoricalVintageError(
                "prospective EEX quote inventory is missing"
            )
    document_identity = ["market", "load_type", "product"]
    if frame.duplicated(document_identity, keep=False).any():
        raise EexHistoricalVintageError(
            "EEX source document repeats an economic quote identity"
        )
    if not frame["source_sheet"].astype(str).eq(frame["market"].astype(str)).all():
        raise EexHistoricalVintageError("EEX source sheet does not match quote market")
    for market, market_rows in frame.groupby("market", sort=True):
        dates = market_rows["date"].drop_duplicates()
        if len(dates) != 1:
            raise EexHistoricalVintageError("EEX document market has multiple snapshot dates")
        snapshot_date = pd.Timestamp(dates.iloc[0]).date().isoformat()
        if prospective_config:
            from pfc_shaping.data.eex_forward_vintage_intake import (
                inspect_eex_forward_workbook_latest_market_row,
            )

            try:
                inspected_date, inspected_quotes = (
                    inspect_eex_forward_workbook_latest_market_row(
                        source_bytes,
                        market=str(market),
                    )
                )
            except ValueError as exc:
                raise EexHistoricalVintageError(
                    "prospective EEX physical-row replay failed"
                ) from exc
            expected_market_quotes = [
                dict(quote)
                for quote in expected_quotes
                if isinstance(quote, Mapping)
                and str(quote.get("market", "")) == str(market)
            ]
            if (
                inspected_date != snapshot_date
                or inspected_quotes != expected_market_quotes
            ):
                raise EexHistoricalVintageError(
                    "prospective EEX physical quote inventory mismatch"
                )
        try:
            prices, parsed_date, metadata = ingest_forwards.load_base_prices_from_eex_report_bytes(
                source_bytes,
                market=str(market),
                source_label=str(entry.get("source_document_id", "<catalog-document>")),
                return_snapshot_metadata=True,
            )
        except Exception as exc:
            raise EexHistoricalVintageError("archived EEX workbook replay failed") from exc
        if pd.Timestamp(parsed_date).date().isoformat() != snapshot_date:
            raise EexHistoricalVintageError("EEX workbook replay selected a different date")
        expected_prices: dict[str, float] = {}
        expected_lineage: dict[str, dict[str, object]] = {}
        for row in market_rows.to_dict(orient="records"):
            load_type = str(row["load_type"]).upper()
            key = str(row["product"])
            if load_type == "PEAK":
                key = f"{key}-Peak"
            elif load_type == "OFFPEAK":
                key = f"{key}-Offpeak"
            expected_prices[key] = float(row["price"])
            expected_lineage[key] = {
                "source_product_code": str(row["source_product_code"]),
                "source_row_index": int(row["source_row_index"]),
                "source_column_index": int(row["source_column_index"]),
                "is_direct_market_quote": True,
            }
        if dict(prices) != expected_prices:
            raise EexHistoricalVintageError("EEX workbook prices do not match vintage rows")
        if dict(metadata.get("quote_lineage", {})) != expected_lineage:
            raise EexHistoricalVintageError("EEX workbook lineage does not match vintage rows")


def _verify_source_trusted_time_receipt(
    receipt: Mapping[str, object],
    *,
    source_entry: Mapping[str, object],
    source_size: int,
    trusted_public_key_path: str | Path | None,
    trusted_public_key_dir: str | Path | None,
    trusted_journal_id: str | None,
) -> dict[str, object]:
    try:
        unsigned = verify_trusted_time_receipt(
            receipt,
            trusted_public_key_path=trusted_public_key_path,
            trusted_public_key_dir=trusted_public_key_dir,
            trusted_journal_id=trusted_journal_id,
        )
    except ValueError as exc:
        raise EexHistoricalVintageError("EEX trusted-time receipt is invalid") from exc
    if unsigned.get("schema_version") != TRUSTED_TIME_RECEIPT_SCHEMA:
        raise EexHistoricalVintageError("EEX trusted-time receipt schema mismatch")
    identity = dict(unsigned)
    receipt_id = str(identity.pop("receipt_id", ""))
    if receipt_id != _sha256_json(identity):
        raise EexHistoricalVintageError("EEX trusted-time receipt_id mismatch")
    expected = {
        "source_document_id": str(source_entry.get("source_document_id", "")),
        "source_document_sha256": str(source_entry.get("sha256", "")),
        "size_bytes": source_size,
    }
    if any(unsigned.get(key) != value for key, value in expected.items()):
        raise EexHistoricalVintageError("EEX trusted-time receipt source binding mismatch")
    received_at = _aware_utc_iso(
        unsigned.get("received_at_utc"),
        label="trusted_time.received_at_utc",
    )
    if str(source_entry.get("observed_at", "")) != received_at:
        raise EexHistoricalVintageError("EEX observed_at differs from trusted receipt time")
    if str(source_entry.get("available_at", "")) != received_at:
        raise EexHistoricalVintageError("EEX available_at differs from trusted receipt time")
    if str(source_entry.get("acquisition_id", "")) != receipt_id:
        raise EexHistoricalVintageError("EEX acquisition_id differs from trusted receipt_id")
    journal_id = str(unsigned.get("journal_id", "")).strip()
    previous = str(unsigned.get("previous_receipt_id", "")).strip()
    raw_sequence = unsigned.get("journal_sequence")
    if not isinstance(raw_sequence, int) or isinstance(raw_sequence, bool):
        raise EexHistoricalVintageError("EEX trusted-time journal sequence is invalid")
    sequence = raw_sequence
    if not journal_id or sequence < 1:
        raise EexHistoricalVintageError("EEX trusted-time journal identity is invalid")
    if previous and not _SHA256_RE.fullmatch(previous):
        raise EexHistoricalVintageError("EEX trusted-time previous receipt is invalid")
    return {
        "receipt_id": receipt_id,
        "journal_id": journal_id,
        "journal_sequence": sequence,
        "previous_receipt_id": previous,
        "received_at_utc": received_at,
    }


def _validate_trusted_time_receipt_chain(receipts: list[dict[str, object]]) -> None:
    if not receipts:
        raise EexHistoricalVintageError("EEX trusted-time receipt chain is empty")
    if len({str(item["journal_id"]) for item in receipts}) != 1:
        raise EexHistoricalVintageError("EEX trusted-time receipts mix journal identities")
    ordered = sorted(receipts, key=lambda item: int(item["journal_sequence"]))
    sequences = [int(item["journal_sequence"]) for item in ordered]
    if sequences != list(range(1, len(ordered) + 1)):
        raise EexHistoricalVintageError(
            "EEX trusted-time receipt journal is not complete and contiguous"
        )
    if str(ordered[0]["previous_receipt_id"]):
        raise EexHistoricalVintageError("EEX trusted-time journal genesis has a parent")
    for parent, child in zip(ordered[:-1], ordered[1:], strict=True):
        if str(child["previous_receipt_id"]) != str(parent["receipt_id"]):
            raise EexHistoricalVintageError("EEX trusted-time receipt chain is not linked")
        if pd.Timestamp(child["received_at_utc"]) < pd.Timestamp(parent["received_at_utc"]):
            raise EexHistoricalVintageError("EEX trusted-time receipt times are not monotonic")


def _aware_utc_series(values: pd.Series, *, label: str) -> pd.Series:
    parsed: list[pd.Timestamp] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise EexHistoricalVintageError(f"{label} contains invalid timestamps") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise EexHistoricalVintageError(f"{label} must be timezone-aware")
        parsed.append(timestamp.tz_convert("UTC"))
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]")


def _resolve_catalog_path(root: Path, value: object, *, label: str) -> Path:
    text = str(value or "")
    if not text or Path(text).is_absolute():
        raise EexHistoricalVintageError(f"EEX vintage {label} path must be relative")
    try:
        resolved = assert_absolute_path_has_no_links((root / text).absolute())
    except ValueError as exc:
        raise EexHistoricalVintageError(
            f"EEX vintage {label} path contains a link or reparse point"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EexHistoricalVintageError(
            f"EEX vintage {label} path escapes catalog root"
        ) from exc
    return resolved


def _aware_utc_iso(value: object, *, label: str) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise EexHistoricalVintageError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC").isoformat()


def _date_iso(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise EexHistoricalVintageError("source snapshot date is invalid")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize().date().isoformat()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
