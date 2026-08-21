"""Canonical LT input snapshots and byte-level source receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pfc_shaping.data.acquisition_contract import (
    GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS,
    PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
    REPLAY_GOVERNED_LT_INPUT_ROLES,
    verify_acquisition_contract,
)
from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_SOURCE_SYSTEM,
    OBSERVATION_RESOLUTION_PROVENANCE_ATTR,
    SFOE_SOURCE_SYSTEM,
    energy_price_resolution_provenance,
    validate_raw_envelope,
    verify_provider_raw_replay,
)
from pfc_shaping.data.lt_input_replay import verify_core_lt_role_replay
from pfc_shaping.data.shared_data_root import (
    FMV_DATA_ROOT_ENV,
    PFC_LT_DATA_ROOT_ENV,
    PFC_SHARED_DATA_ROOT_ENV,
    configured_lt_data_root,
    resolve_confined_path,
)
from pfc_shaping.data.snapshot_publication_contract import (
    PUBLICATION_HEAD_CHALLENGE_NONCE_ENV,
    PUBLICATION_HEAD_OBSERVATION_PATH_ENV,
)
from pfc_shaping.data.snapshot_publication_state import (
    verify_external_publication_evidence,
    verify_publication_authority_separation,
)
from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    path_is_link,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json, load_strict_yaml

CORE_LT_INPUT_ROLES = ("epex_ch", "epex_de", "entso", "hydro")
_MAX_PARQUET_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_PROVIDER_RAW_ARTIFACT_BYTES = 96 * 1024 * 1024
_MAX_SUPPORTING_ARTIFACT_BYTES = 16 * 1024 * 1024
_HYDRO_MIN_HISTORY_ROWS = 312
_HYDRO_MIN_HISTORY_DAYS = 6 * 365 - 21
_HYDRO_RECENT_SUPPORT_ROWS = 8
_QUALITY_MAX_ROWS_BY_ROLE = {
    "epex_ch": 2_000_000,
    "epex_de": 2_000_000,
    "epex_at": 2_000_000,
    "epex_fr": 2_000_000,
    "epex_it": 2_000_000,
    "entso": 2_000_000,
    "hydro": 5_000,
}
_QUALITY_MAX_COLUMNS = 128
_QUALITY_MAX_CELLS = 40_000_000
_QUALITY_MAX_ROW_GROUPS = 8_192
_QUALITY_ALLOWED_PARQUET_PHYSICAL_TYPES = {
    "BOOLEAN",
    "INT32",
    "INT64",
    "FLOAT",
    "DOUBLE",
}
PROVIDER_RAW_LT_INPUT_ROLES = frozenset(
    {*REPLAY_GOVERNED_LT_INPUT_ROLES, "eex_forwards_history"}
)
GOVERNED_SOURCE_SYSTEMS_BY_ROLE = {
    "epex_ch": frozenset({"EPEX_SPOT", ENERGY_CHARTS_SOURCE_SYSTEM}),
    "epex_de": frozenset({"EPEX_SPOT", ENERGY_CHARTS_SOURCE_SYSTEM}),
    "epex_at": frozenset({"EPEX_SPOT", ENERGY_CHARTS_SOURCE_SYSTEM}),
    "epex_fr": frozenset({"EPEX_SPOT", ENERGY_CHARTS_SOURCE_SYSTEM}),
    "epex_it": frozenset({"EPEX_SPOT", ENERGY_CHARTS_SOURCE_SYSTEM}),
    "entso": frozenset({"ENTSOE_TRANSPARENCY_PLATFORM"}),
    "outages": frozenset({"ENTSOE_TRANSPARENCY_PLATFORM"}),
    "hydro": frozenset({"FMV_HYDRO_CURATED", SFOE_SOURCE_SYSTEM}),
    "commodities": frozenset({"FMV_COMMODITIES_CURATED"}),
    "eex_forwards_history": frozenset({"EEX_MARKET_DATA"}),
    "eex_forward_source": frozenset({"EEX_MARKET_DATA"}),
}
GOVERNED_EEX_HISTORY_SOURCES = frozenset({"EEX", "HISTORIQUE2019", "YEARLY"})
GOVERNED_EEX_HISTORY_MARKETS = frozenset({"CH", "DE", "FR", "AT", "IT"})
GOVERNED_EEX_HISTORY_LOAD_TYPES = frozenset({"BASE", "PEAK"})
GOVERNED_EEX_HISTORY_PRODUCT_TYPES = frozenset({"CAL", "QUARTER", "MONTH"})
_EEX_CAL_PRODUCT = re.compile(r"^\d{4}$")
_EEX_QUARTER_PRODUCT = re.compile(r"^\d{4}-Q[1-4]$")
_EEX_MONTH_PRODUCT = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class InputSourceReceipt:
    """Identity of the exact file bytes and frame consumed by one input."""

    role: str
    path: str
    logical_path: str
    size_bytes: int
    sha256: str
    rows: int | None = None
    frame_sha256: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "logical_path": self.logical_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "rows": self.rows,
            "frame_sha256": self.frame_sha256,
        }


@dataclass(frozen=True)
class LTInputPaths:
    root: Path
    snapshot_root: Path
    layout: str
    generation_id: str | None
    calibration_eligible: bool
    available_at_utc: str | None
    files: Mapping[str, Path]
    expected_files: Mapping[str, Mapping[str, Any]]
    schema_version: str | None = None
    pointer_receipt: InputSourceReceipt | None = None
    contract_receipt: InputSourceReceipt | None = None
    publication_intent_receipt: InputSourceReceipt | None = None
    publication_anchor_receipt: InputSourceReceipt | None = None
    publication_head_observation_receipt: InputSourceReceipt | None = None

    def path_for(self, role: str) -> Path:
        try:
            return self.files[role]
        except KeyError as exc:
            raise KeyError(f"LT input role is absent from the selected snapshot: {role}") from exc

    def has_role(self, role: str) -> bool:
        return role in self.files

    @property
    def epex_ch(self) -> Path:
        return self.path_for("epex_ch")

    @property
    def epex_de(self) -> Path:
        return self.path_for("epex_de")

    @property
    def entso(self) -> Path:
        return self.path_for("entso")

    @property
    def hydro(self) -> Path:
        return self.path_for("hydro")

    @property
    def outages(self) -> Path:
        return self.path_for("outages")

    @property
    def commodities(self) -> Path:
        return self.path_for("commodities")

    @property
    def eex_forwards_history(self) -> Path:
        return self.path_for("eex_forwards_history")

    def neighbor_epex(self, market: str) -> Path:
        code = str(market).strip().lower()
        if code not in {"at", "fr", "it"}:
            raise ValueError(f"unsupported optional EPEX neighbor: {market}")
        return self.path_for(f"epex_{code}")

    def supporting_path(self, role: str, binding_name: str) -> Path:
        entry = self.expected_files.get(str(role))
        if not isinstance(entry, Mapping):
            raise KeyError(f"LT input role is unavailable: {role}")
        binding = entry.get(str(binding_name))
        if not isinstance(binding, Mapping):
            raise KeyError(f"LT input role {role} has no {binding_name} binding")
        return _safe_relative_path(self.snapshot_root, str(binding.get("path", "")))


def resolve_lt_input_paths(
    project_root: str | Path,
    *,
    data_root: str | Path | None = None,
    allow_legacy_repo: bool = False,
    expected_snapshot_contract: str | Path | None = None,
    expected_pointer_contract: str | Path | None = None,
    expected_snapshot_sha256: str | None = None,
    expected_pointer_sha256: str | None = None,
    expected_generation_id: str | None = None,
    publication_head_observation: str | Path | None = None,
    publication_head_challenge_nonce: str | None = None,
    expected_publication_head_observation_sha256: str | None = None,
) -> LTInputPaths:
    """Resolve one explicit immutable external snapshot or the legacy layout."""

    project = Path(project_root).resolve()
    selected: str | Path | None = data_root
    if selected is None:
        selected = configured_lt_data_root()

    if selected is None and not allow_legacy_repo:
        raise ValueError(
            f"explicit {FMV_DATA_ROOT_ENV} / {PFC_LT_DATA_ROOT_ENV} / "
            f"{PFC_SHARED_DATA_ROOT_ENV} / data_root is required; "
            "legacy repo inputs require allow_legacy_repo=True"
        )

    if selected is None:
        root = project / "pfc_shaping" / "data"
        files = {
            "epex_ch": root / "epex_15min.parquet",
            "epex_de": root / "epex_de_15min.parquet",
            "entso": root / "entso_15min.parquet",
            "hydro": root / "hydro_reservoir.parquet",
            "outages": root / "outages_15min.parquet",
            "commodities": project / "data" / "commodities_cache.parquet",
            "eex_forwards_history": project / "data" / "eex_forwards_history.parquet",
        }
        for code in ("at", "fr", "it"):
            candidate = root / f"epex_{code}_15min.parquet"
            if candidate.is_file():
                files[f"epex_{code}"] = candidate
        return LTInputPaths(
            root=root,
            snapshot_root=root,
            layout="legacy_repo",
            generation_id=None,
            calibration_eligible=False,
            available_at_utc=None,
            files=files,
            expected_files={},
        )

    raw_selected = str(selected).strip()
    if not raw_selected:
        raise ValueError(
            f"{FMV_DATA_ROOT_ENV} / {PFC_LT_DATA_ROOT_ENV} / "
            f"{PFC_SHARED_DATA_ROOT_ENV} / data_root cannot be empty"
        )
    root_candidate = Path(raw_selected).expanduser()
    if not root_candidate.is_absolute():
        raise ValueError(f"LT data_root must be absolute: {root_candidate}")
    root = assert_absolute_path_has_no_links(root_candidate).resolve()
    binding_values = (
        expected_snapshot_contract,
        expected_pointer_contract,
        expected_snapshot_sha256,
        expected_pointer_sha256,
        expected_generation_id,
    )
    if any(value is not None for value in binding_values) and not all(
        value is not None for value in binding_values
    ):
        raise ValueError("explicit LT snapshot binding requires contract, hashes and generation id")
    if expected_pointer_contract is None:
        pointer_path, pointer_logical_path = _resolve_lt_pointer(root)
    else:
        pointer_path = _resolve_explicit_pointer(expected_pointer_contract)
        pointer_logical_path = "views/pfc_lt/current.json"
    pointer, pointer_receipt, _ = _read_json_snapshot_payload(
        pointer_path,
        role="lt_data_pointer",
        logical_path=pointer_logical_path,
    )
    pointer_schema = pointer.get("schema_version")
    if pointer_schema not in {"lt_data_pointer.v1", "lt_data_pointer.v2"}:
        raise ValueError("LT data pointer schema_version is unsupported")
    if (
        expected_pointer_sha256 is not None
        and pointer_receipt.sha256 != str(expected_pointer_sha256).strip().lower()
    ):
        raise ValueError("LT data pointer SHA-256 does not match --input-pointer-sha256")
    generation_id = str(pointer.get("generation_id", "")).strip()
    if not generation_id:
        raise ValueError("LT data pointer generation_id is missing")
    if expected_generation_id is not None and generation_id != str(
        expected_generation_id
    ):
        raise ValueError("LT data pointer generation_id does not match --input-generation-id")
    contract_rel = str(pointer.get("contract_path", "")).strip()
    expected_contract_rel = (
        Path("snapshots") / generation_id / "lt_input_snapshot.json"
    ).as_posix()
    if contract_rel != expected_contract_rel:
        raise ValueError("LT data pointer contract_path is not canonical")
    contract_path = _safe_relative_path(root, contract_rel)
    if expected_snapshot_contract is not None:
        selected_contract = Path(expected_snapshot_contract).expanduser()
        if not selected_contract.is_absolute():
            raise ValueError("LT input snapshot contract path must be absolute")
        selected_contract = resolve_confined_path(
            root,
            selected_contract,
            label="explicit LT input snapshot contract",
            require_exists=True,
            require_file=True,
        )
        if selected_contract != contract_path:
            raise ValueError(
                "LT data pointer contract does not match --input-snapshot-contract"
            )
    contract, contract_receipt, _ = _read_json_snapshot_payload(
        contract_path,
        role="lt_input_snapshot_contract",
        logical_path=contract_rel,
    )
    if contract_receipt.sha256 != str(pointer.get("contract_sha256", "")):
        raise ValueError("LT data pointer contract_sha256 mismatch")
    if (
        expected_snapshot_sha256 is not None
        and contract_receipt.sha256 != str(expected_snapshot_sha256).strip().lower()
    ):
        raise ValueError(
            "LT input snapshot SHA-256 does not match --input-snapshot-sha256"
        )
    validate_lt_input_contract_semantics(contract)
    schema_version = str(contract.get("schema_version", ""))
    if schema_version not in {"lt_input_snapshot.v1", *GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS}:
        raise ValueError("LT input contract schema_version is unsupported")
    if contract.get("layout") != "external_v2":
        raise ValueError("LT input contract layout must be external_v2")
    if str(contract.get("generation_id", "")) != generation_id:
        raise ValueError("LT data pointer/contract generation_id mismatch")
    publication_intent_receipt = None
    publication_anchor_receipt = None
    publication_head_observation_receipt = None
    if pointer_schema == "lt_data_pointer.v2":
        intent_id = str(pointer.get("publication_intent_id", ""))
        receipt_id = str(pointer.get("anchor_receipt_id", ""))
        intent_path = _safe_relative_path(
            root,
            f"views/pfc_lt/intents/{intent_id}.json",
        )
        receipt_path = _safe_relative_path(
            root,
            f"views/pfc_lt/receipts/{receipt_id}.json",
        )
        intent_document, publication_intent_receipt, intent_payload = (
            _read_json_snapshot_payload(
            intent_path,
            role="lt_snapshot_publication_intent",
            logical_path=f"views/pfc_lt/intents/{intent_id}.json",
            )
        )
        receipt_document, publication_anchor_receipt, receipt_payload = (
            _read_json_snapshot_payload(
            receipt_path,
            role="lt_snapshot_anchor_receipt",
            logical_path=f"views/pfc_lt/receipts/{receipt_id}.json",
            )
        )
        selected_observation = publication_head_observation
        if selected_observation is None:
            selected_observation = os.environ.get(
                PUBLICATION_HEAD_OBSERVATION_PATH_ENV
            )
        if selected_observation is None or not str(selected_observation).strip():
            raise ValueError(
                "LT data pointer v2 requires a fresh external anchor HEAD observation"
            )
        selected_nonce = publication_head_challenge_nonce
        if selected_nonce is None:
            selected_nonce = os.environ.get(PUBLICATION_HEAD_CHALLENGE_NONCE_ENV)
        (
            observation_document,
            publication_head_observation_receipt,
            observation_payload,
        ) = _read_json_snapshot_payload(
            selected_observation,
            role="lt_snapshot_anchor_head_observation",
        )
        if (
            expected_publication_head_observation_sha256 is not None
            and publication_head_observation_receipt.sha256
            != str(expected_publication_head_observation_sha256).strip().lower()
        ):
            raise ValueError("external anchor HEAD observation SHA-256 mismatch")
        verify_external_publication_evidence(
            intent_payload=intent_payload,
            receipt_payload=receipt_payload,
            observation_payload=observation_payload,
            pointer=pointer,
            require_current=True,
            expected_challenge_nonce=selected_nonce,
        )
        verify_publication_authority_separation(
            contract,
            intent=intent_document,
            receipt=receipt_document,
            observation=observation_document,
        )
    elif (
        schema_version in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS
        and contract.get("calibration_eligible") is True
    ):
        raise ValueError(
            "calibration-eligible LT input requires an external CAS v2 pointer"
        )
    acquisition_id = str(contract.get("acquisition_id", "")).strip()
    if not acquisition_id:
        raise ValueError("LT input contract acquisition_id is missing")
    contract_available_at = _parse_available_at(
        contract.get("available_at_utc"),
        label="LT input contract",
    )
    snapshot_root = contract_path.parent
    roles = contract.get("files")
    if not isinstance(roles, Mapping):
        raise ValueError("LT input contract files must be a mapping")
    missing = set(CORE_LT_INPUT_ROLES).difference(roles)
    if missing:
        raise ValueError(f"LT input contract missing core roles: {sorted(missing)}")
    files: dict[str, Path] = {}
    expected: dict[str, Mapping[str, Any]] = {}
    for role, entry in roles.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"LT input contract role {role} must be a mapping")
        logical_path = str(entry.get("path", "")).strip()
        if str(entry.get("acquisition_id", "")) != acquisition_id:
            raise ValueError(f"LT input role has a divergent acquisition_id: {role}")
        _parse_available_at(
            entry.get("available_at_utc"),
            label=f"LT input role {role}",
        )
        files[str(role)] = _safe_relative_path(snapshot_root, logical_path)
        expected[str(role)] = dict(entry)
        if schema_version in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
            _verify_v2_supporting_artifacts(
                snapshot_root,
                str(role),
                entry,
                require_provider_raw=(
                    schema_version == PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA
                ),
            )
    eligible = contract.get("calibration_eligible") is True
    if eligible:
        if contract.get("source_class") != "GOVERNED_ACQUISITION":
            raise ValueError("calibration-eligible LT input must be a GOVERNED_ACQUISITION")
        if schema_version not in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
            raise ValueError(
                "calibration-eligible LT input requires a governed snapshot schema"
            )
        verify_acquisition_contract(contract)
        ineligible_roles = [
            role
            for role in CORE_LT_INPUT_ROLES
            if expected[role].get("calibration_eligible") is not True
        ]
        if ineligible_roles:
            raise ValueError(
                f"calibration-eligible LT input has ineligible core roles: {ineligible_roles}"
            )
    return LTInputPaths(
        root=root,
        snapshot_root=snapshot_root,
        layout="external_v2",
        generation_id=generation_id,
        calibration_eligible=eligible,
        available_at_utc=contract_available_at.isoformat(),
        files=files,
        expected_files=expected,
        schema_version=schema_version,
        pointer_receipt=pointer_receipt,
        contract_receipt=contract_receipt,
        publication_intent_receipt=publication_intent_receipt,
        publication_anchor_receipt=publication_anchor_receipt,
        publication_head_observation_receipt=(
            publication_head_observation_receipt
        ),
    )


def _resolve_lt_pointer(root: Path) -> tuple[Path, str]:
    """Use the consumer view, with a read-only fallback for pre-migration roots."""

    canonical = root / "views" / "pfc_lt" / "current.json"
    legacy = root / "current.json"
    if canonical.exists():
        return (
            resolve_confined_path(
                root,
                canonical,
                label="canonical LT data pointer",
                require_exists=True,
                require_file=True,
            ),
            "views/pfc_lt/current.json",
        )
    return (
        resolve_confined_path(
            root,
            legacy,
            label="legacy LT data pointer",
            require_exists=True,
            require_file=True,
        ),
        "current.json",
    )


def _resolve_explicit_pointer(value: str | Path) -> Path:
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        raise ValueError("LT input pointer contract path must be absolute")
    for component in (lexical, *lexical.parents):
        if path_is_link(component):
            raise ValueError("LT input pointer contract cannot contain a link")
    resolved = lexical.resolve()
    if not resolved.is_file():
        raise ValueError("LT input pointer contract is unavailable")
    return resolved


def validate_lt_input_contract_semantics(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Validate one authenticated LT snapshot independently of its storage root."""

    schema_version = str(contract.get("schema_version", ""))
    if schema_version not in {"lt_input_snapshot.v1", *GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS}:
        raise ValueError("LT input contract schema_version is unsupported")
    if contract.get("layout") != "external_v2":
        raise ValueError("LT input contract layout must be external_v2")
    generation_id = str(contract.get("generation_id", "")).strip()
    acquisition_id = str(contract.get("acquisition_id", "")).strip()
    if not generation_id:
        raise ValueError("LT input contract generation_id is missing")
    if not acquisition_id:
        raise ValueError("LT input contract acquisition_id is missing")
    available_at = _parse_available_at(
        contract.get("available_at_utc"),
        label="LT input contract",
    )
    files = contract.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("LT input contract files must be a mapping")
    missing = set(CORE_LT_INPUT_ROLES).difference(files)
    if missing:
        raise ValueError(f"LT input contract missing core roles: {sorted(missing)}")
    if schema_version == PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA:
        unsupported = set(str(role) for role in files).difference(
            PROVIDER_RAW_LT_INPUT_ROLES
        )
        if unsupported:
            raise ValueError(
                "provider-raw LT snapshot contains roles without exact replay: "
                f"{sorted(unsupported)}"
            )
    role_available_times: list[pd.Timestamp] = []
    for role, entry in files.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"LT input contract role {role} must be a mapping")
        if str(entry.get("acquisition_id", "")) != acquisition_id:
            raise ValueError(f"LT input role has a divergent acquisition_id: {role}")
        role_available = _parse_available_at(
            entry.get("available_at_utc"),
            label=f"LT input role {role}",
        )
        role_available_times.append(role_available)
        if (
            schema_version == "lt_input_snapshot.v1"
            and role_available != available_at
        ):
            raise ValueError(f"LT input role has a divergent acquisition cutoff: {role}")
        if (
            schema_version in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS
            and role_available > available_at
        ):
            raise ValueError(f"LT input role is newer than snapshot cutoff: {role}")
        logical_path = str(entry.get("path", "")).strip()
        if not logical_path:
            raise ValueError(f"LT input role path is missing: {role}")
        relative = Path(logical_path)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(f"LT input role path is not portable and relative: {role}")
    eligible = contract.get("calibration_eligible") is True
    if eligible:
        if schema_version not in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
            raise ValueError(
                "calibration-eligible LT input requires a governed snapshot schema"
            )
        if contract.get("source_class") != "GOVERNED_ACQUISITION":
            raise ValueError("calibration-eligible LT input must be a GOVERNED_ACQUISITION")
        verify_acquisition_contract(contract)
        ineligible = [
            role
            for role, entry in files.items()
            if entry.get("calibration_eligible") is not True
        ]
        if ineligible:
            raise ValueError(
                f"calibration-eligible LT input has ineligible roles: {sorted(ineligible)}"
            )
        for role, entry in files.items():
            validate_governed_source_system(str(role), entry)
        checkpoint = contract.get("source_journal_checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("governed LT snapshot source journal checkpoint is missing")
        checkpoint_issued_at = _parse_available_at(
            checkpoint.get("issued_at_utc"),
            label="LT source journal checkpoint",
        )
        if (
            not role_available_times
            or max(*role_available_times, checkpoint_issued_at) != available_at
        ):
            raise ValueError(
                "governed LT snapshot cutoff must equal the latest governed availability"
            )
    return {
        "generation_id": generation_id,
        "acquisition_id": acquisition_id,
        "available_at_utc": available_at.isoformat(),
        "calibration_eligible": eligible,
        "files": {str(role): dict(entry) for role, entry in files.items()},
    }


def validate_governed_lt_snapshot_bundle(
    snapshot_root: str | Path,
    contract: Mapping[str, object],
) -> None:
    """Validate a complete promotion-grade v2 bundle before publication."""

    root = Path(snapshot_root).resolve()
    validate_lt_input_contract_semantics(contract)
    schema_version = str(contract.get("schema_version", ""))
    if schema_version not in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
        raise ValueError("governed LT snapshot bundle requires a governed snapshot schema")
    if contract.get("calibration_eligible") is not True:
        raise ValueError("governed LT snapshot bundle must be calibration eligible")
    if contract.get("source_class") != "GOVERNED_ACQUISITION":
        raise ValueError("governed LT snapshot bundle must be a GOVERNED_ACQUISITION")
    verify_acquisition_contract(contract)
    roles = contract.get("files")
    if not isinstance(roles, Mapping):
        raise ValueError("LT input contract files must be a mapping")
    missing = set(CORE_LT_INPUT_ROLES).difference(roles)
    if missing:
        raise ValueError(f"LT input contract missing core roles: {sorted(missing)}")
    for role, entry in roles.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"LT input contract role {role} must be a mapping")
        _verify_bound_artifact(root, entry, role=str(role))
        _verify_v2_supporting_artifacts(
            root,
            str(role),
            entry,
            require_provider_raw=(
                schema_version == PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA
            ),
        )
    eex_entry = roles.get("eex_forwards_history")
    if eex_entry is not None:
        if not isinstance(eex_entry, Mapping):
            raise ValueError("EEX forward history contract entry is invalid")
        catalog_binding = eex_entry.get("vintage_catalog")
        if not isinstance(catalog_binding, Mapping):
            raise ValueError(
                "governed EEX forward history requires a signed vintage_catalog binding"
            )
        catalog_path, catalog_payload = _verify_bound_artifact(
            root,
            catalog_binding,
            role="eex_forwards_history.vintage_catalog",
        )
        history_path = _safe_relative_path(root, str(eex_entry.get("path", "")))
        try:
            catalog = load_strict_json(catalog_payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("EEX forward history vintage catalog is invalid") from exc
        if not isinstance(catalog, Mapping):
            raise ValueError("EEX forward history vintage catalog must be a mapping")
        from pfc_shaping.data.eex_historical_vintage import (
            verify_eex_historical_vintage_catalog,
        )

        _, evidence = verify_eex_historical_vintage_catalog(
            catalog,
            catalog_path=catalog_path,
            history_path=history_path,
        )
        if evidence.catalog_sha256 != str(catalog_binding.get("sha256", "")):
            raise ValueError("EEX vintage catalog contract hash binding mismatch")
        if evidence.history_sha256 != str(eex_entry.get("sha256", "")):
            raise ValueError("EEX vintage history contract hash binding mismatch")


def validate_governed_source_system(
    role: str,
    entry: Mapping[str, object],
) -> str:
    """Return the authenticated upstream identity or reject unknown providers."""

    allowed = GOVERNED_SOURCE_SYSTEMS_BY_ROLE.get(str(role))
    if allowed is None:
        raise ValueError(f"governed LT input role has no source-system policy: {role}")
    source_system = str(entry.get("source_system", "")).strip().upper()
    if source_system not in allowed:
        raise ValueError(
            f"governed LT input role {role} has inadmissible source_system "
            f"{source_system!r}; expected one of {sorted(allowed)}"
        )
    return source_system


def validate_lt_input_consumption(
    paths: LTInputPaths,
    *,
    roles: set[str],
    reference_timestamp: str | pd.Timestamp,
) -> None:
    """Fail before payload I/O when a consumed role is not coherent PIT data."""

    if paths.layout != "external_v2" or not paths.calibration_eligible:
        raise ValueError("LT production consumption requires a calibration-eligible snapshot")
    if paths.schema_version != PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA:
        raise ValueError(
            "LT production consumption requires lt_input_snapshot.v3 provider-raw evidence"
        )
    reference = _parse_available_at(reference_timestamp, label="LT valuation timestamp")
    contract_available = _parse_available_at(
        paths.available_at_utc,
        label="LT input contract",
    )
    if contract_available > reference:
        raise ValueError("LT input contract is not available at valuation timestamp")
    unsupported = set(roles).difference(
        REPLAY_GOVERNED_LT_INPUT_ROLES,
        {"eex_forwards_history"},
    )
    if unsupported:
        raise ValueError(
            "consumed LT input roles lack deterministic replay governance: "
            f"{sorted(unsupported)}"
        )
    for role in sorted(roles):
        entry = paths.expected_files.get(role)
        if entry is None:
            raise ValueError(f"consumed LT input role is absent from contract: {role}")
        if entry.get("calibration_eligible") is not True:
            raise ValueError(f"consumed LT input role is not calibration eligible: {role}")
        role_available = _parse_available_at(
            entry.get("available_at_utc"),
            label=f"LT input role {role}",
        )
        if (
            paths.schema_version not in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS
            and role_available != contract_available
        ):
            raise ValueError(
                f"consumed LT input role has a divergent acquisition cutoff: {role}"
            )
        if role_available > contract_available:
            raise ValueError(
                f"consumed LT input role is newer than snapshot cutoff: {role}"
            )
        if role_available > reference:
            raise ValueError(f"consumed LT input role is not PIT at valuation: {role}")


def consumed_lt_input_roles(
    config: Mapping[str, object],
    *,
    available_roles: set[str],
) -> set[str]:
    """Derive production-consumed roles with strict shared config semantics."""

    roles = set(CORE_LT_INPUT_ROLES)
    roles.update(
        role for role in ("epex_at", "epex_fr", "epex_it") if role in available_roles
    )
    forwards = _strict_config_mapping(config, "forwards", label="forwards")
    solver = _strict_config_mapping(
        forwards,
        "monthly_curve_solver",
        label="forwards.monthly_curve_solver",
    )
    solver_enabled = _strict_config_bool(
        solver,
        "enabled",
        default=False,
        label="forwards.monthly_curve_solver.enabled",
    )
    target_values = solver.get("target_markets", ("CH",))
    if isinstance(target_values, (str, bytes)) or not isinstance(target_values, (list, tuple)):
        raise ValueError("forwards.monthly_curve_solver.target_markets must be a list")
    targets = {str(value).upper() for value in target_values}
    if solver_enabled and "CH" in targets:
        roles.add("eex_forwards_history")
    if "commodities" in available_roles:
        roles.add("commodities")
    quality = _strict_config_mapping(config, "quality", label="quality")
    freshness = _strict_config_mapping(
        quality,
        "freshness",
        label="quality.freshness",
    )
    outages_enabled = _strict_config_bool(
        freshness,
        "outages_enabled",
        default=False,
        label="quality.freshness.outages_enabled",
    )
    if outages_enabled:
        roles.add("outages")
    return roles


def validate_governed_forward_history_frame(
    history: pd.DataFrame,
    *,
    reference_timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    """Validate and normalize the semantic PIT contract for EEX history."""

    required = {
        "date",
        "product",
        "load_type",
        "product_type",
        "market",
        "price",
        "source",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"governed EEX forward history missing columns: {missing}")
    frame = history.copy()
    observation_timestamps: list[pd.Timestamp] = []
    for value in frame["date"]:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            timestamp = pd.NaT
        if pd.isna(timestamp):
            observation_timestamps.append(pd.NaT)
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        observation_timestamps.append(timestamp)
    observations = pd.DatetimeIndex(observation_timestamps)
    if observations.isna().any():
        raise ValueError("governed EEX forward history contains invalid dates")
    reference = _parse_available_at(reference_timestamp, label="EEX history valuation timestamp")
    if bool((observations > reference).any()):
        raise ValueError("governed EEX forward history contains observations after valuation")
    frame["date"] = observations.tz_convert("UTC").tz_localize(None).normalize()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if not np.isfinite(frame["price"].to_numpy(dtype=float)).all():
        raise ValueError("governed EEX forward history contains non-finite prices")
    for column in ("market", "load_type", "product", "product_type", "source"):
        frame[column] = frame[column].astype("string").str.strip().str.upper()
        if frame[column].isna().any() or bool(frame[column].eq("").any()):
            raise ValueError(f"governed EEX forward history contains empty {column} values")
    unsupported_sources = sorted(
        set(frame["source"].astype(str)).difference(GOVERNED_EEX_HISTORY_SOURCES)
    )
    if unsupported_sources:
        raise ValueError(
            "governed EEX forward history contains non-EEX sources: "
            f"{unsupported_sources}"
        )
    for column, allowed in (
        ("market", GOVERNED_EEX_HISTORY_MARKETS),
        ("load_type", GOVERNED_EEX_HISTORY_LOAD_TYPES),
        ("product_type", GOVERNED_EEX_HISTORY_PRODUCT_TYPES),
    ):
        unsupported = sorted(set(frame[column].astype(str)).difference(allowed))
        if unsupported:
            raise ValueError(
                f"governed EEX forward history contains unsupported {column} values: "
                f"{unsupported}"
            )
    expected_product_types = frame["product"].map(_eex_product_type)
    if expected_product_types.isna().any():
        invalid_products = sorted(
            set(frame.loc[expected_product_types.isna(), "product"].astype(str))
        )
        raise ValueError(
            "governed EEX forward history contains invalid product values: "
            f"{invalid_products}"
        )
    inconsistent = frame["product_type"].ne(expected_product_types)
    if bool(inconsistent.any()):
        rows = frame.loc[inconsistent, ["product", "product_type"]].drop_duplicates()
        raise ValueError(
            "governed EEX forward history product/product_type mismatch: "
            f"{rows.to_dict(orient='records')}"
        )
    identity = ["date", "market", "load_type", "product"]
    if frame.duplicated(identity, keep=False).any():
        raise ValueError("governed EEX forward history contains duplicate quote identities")
    frame.attrs["eex_source_summary"] = [
        {
            "source": str(source),
            "rows": int(len(group)),
            "date_min": pd.Timestamp(group["date"].min()).date().isoformat(),
            "date_max": pd.Timestamp(group["date"].max()).date().isoformat(),
            "markets": sorted(set(group["market"].astype(str))),
        }
        for source, group in frame.groupby("source", sort=True)
    ]
    return frame


def _eex_product_type(product: object) -> str | None:
    value = str(product)
    if _EEX_CAL_PRODUCT.fullmatch(value):
        return "CAL"
    if _EEX_QUARTER_PRODUCT.fullmatch(value):
        return "QUARTER"
    if _EEX_MONTH_PRODUCT.fullmatch(value):
        return "MONTH"
    return None


def _strict_config_bool(
    mapping: Mapping[str, object],
    key: str,
    *,
    default: bool,
    label: str,
) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _strict_config_mapping(
    parent: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> Mapping[str, object]:
    if key not in parent:
        return {}
    value = parent[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def read_parquet_snapshot(
    path: str | Path,
    *,
    role: str,
    logical_path: str | None = None,
) -> tuple[pd.DataFrame, InputSourceReceipt]:
    """Read and parse one immutable in-memory copy of a parquet source."""

    source = assert_absolute_path_has_no_links(Path(path).expanduser().absolute())
    payload = read_stable_single_link_file(source, label=f"{role} parquet snapshot")
    frame = pd.read_parquet(BytesIO(payload))
    receipt = _receipt(
        source,
        payload,
        role=role,
        logical_path=logical_path,
        rows=len(frame),
        frame_sha256=dataframe_sha256(frame),
    )
    return frame, receipt


def read_yaml_snapshot(
    path: str | Path,
    *,
    role: str = "config",
    logical_path: str | None = None,
) -> tuple[dict[str, Any], InputSourceReceipt]:
    source = assert_absolute_path_has_no_links(Path(path).expanduser().absolute())
    payload = read_stable_single_link_file(source, label=f"{role} YAML snapshot")
    receipt = _receipt(source, payload, role=role, logical_path=logical_path)
    parsed = load_strict_yaml(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{role} YAML must contain a mapping: {source}")
    return parsed, receipt


def read_json_snapshot(
    path: str | Path,
    *,
    role: str,
    logical_path: str | None = None,
) -> tuple[dict[str, Any], InputSourceReceipt]:
    parsed, receipt, _ = _read_json_snapshot_payload(
        path,
        role=role,
        logical_path=logical_path,
    )
    return parsed, receipt


def _read_json_snapshot_payload(
    path: str | Path,
    *,
    role: str,
    logical_path: str | None = None,
) -> tuple[dict[str, Any], InputSourceReceipt, bytes]:
    source = assert_absolute_path_has_no_links(Path(path).expanduser().absolute())
    payload = read_stable_single_link_file(source, label=f"{role} JSON snapshot")
    receipt = _receipt(source, payload, role=role, logical_path=logical_path)
    parsed = load_strict_json(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{role} JSON must contain a mapping: {source}")
    return parsed, receipt, payload


def validate_receipt_against_contract(paths: LTInputPaths, receipt: InputSourceReceipt) -> None:
    if paths.layout != "external_v2":
        return
    expected = paths.expected_files.get(receipt.role)
    if expected is None:
        raise ValueError(f"input role is not declared by LT snapshot: {receipt.role}")
    if receipt.logical_path != str(expected.get("path", "")):
        raise ValueError(f"LT snapshot logical path mismatch: {receipt.role}")
    if receipt.sha256 != str(expected.get("sha256", "")):
        raise ValueError(f"LT snapshot source hash mismatch: {receipt.role}")
    if receipt.size_bytes != int(expected.get("size_bytes", -1)):
        raise ValueError(f"LT snapshot source size mismatch: {receipt.role}")


def verify_source_receipt(receipt: InputSourceReceipt) -> None:
    source = Path(receipt.path)
    payload = read_stable_single_link_file(source, label=f"{receipt.role} source receipt")
    actual = _receipt(
        source,
        payload,
        role=receipt.role,
        logical_path=receipt.logical_path,
        rows=receipt.rows,
        frame_sha256=receipt.frame_sha256,
    )
    if actual.sha256 != receipt.sha256 or actual.size_bytes != receipt.size_bytes:
        raise ValueError(f"input source changed after consumption: {receipt.role} ({source})")


def verify_frame_receipt(frame: pd.DataFrame, receipt: InputSourceReceipt) -> None:
    if receipt.rows != len(frame) or receipt.frame_sha256 != dataframe_sha256(frame):
        raise ValueError(f"input frame changed after consumption: {receipt.role}")


def dataframe_sha256(frame: pd.DataFrame) -> str:
    metadata = {
        "columns": [str(value) for value in frame.columns],
        "dtypes": [str(value) for value in frame.dtypes],
        "index_names": [str(value) for value in frame.index.names],
        "index_dtype": str(frame.index.dtype),
        "rows": len(frame),
    }
    if OBSERVATION_RESOLUTION_PROVENANCE_ATTR in frame.attrs:
        metadata["resolution_provenance"] = frame.attrs[
            OBSERVATION_RESOLUTION_PROVENANCE_ATTR
        ]
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _safe_relative_path(root: Path, value: str) -> Path:
    if not value:
        raise ValueError("LT snapshot relative path is missing")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"LT snapshot path must be relative: {value}")
    try:
        return resolve_confined_path(
            root,
            relative,
            label=f"LT snapshot path {value}",
            require_exists=True,
            require_file=True,
        )
    except FileNotFoundError:
        raise
    except ValueError as exc:
        raise ValueError(f"LT snapshot path escapes or links outside its root: {value}") from exc


def _verify_v2_supporting_artifacts(
    snapshot_root: Path,
    role: str,
    entry: Mapping[str, object],
    *,
    require_provider_raw: bool,
) -> None:
    raw = entry.get("raw_artifact")
    derivation = entry.get("derivation")
    quality = entry.get("quality_evidence")
    if not isinstance(raw, Mapping):
        raise ValueError(f"LT input role {role} raw_artifact is missing")
    if not isinstance(derivation, Mapping):
        raise ValueError(f"LT input role {role} derivation is missing")
    if not isinstance(quality, Mapping):
        raise ValueError(f"LT input role {role} quality_evidence is missing")
    _, raw_payload = _verify_bound_artifact(
        snapshot_root,
        raw,
        role=f"{role}.raw_artifact",
    )
    provider_raw_payload: bytes | None = None
    provider_parser_payload: bytes | None = None
    provider_parser_config_payload: bytes | None = None
    provider_derivation = entry.get("provider_derivation")
    if require_provider_raw and role in REPLAY_GOVERNED_LT_INPUT_ROLES:
        provider_raw = entry.get("provider_raw_artifact")
        if not isinstance(provider_raw, Mapping):
            raise ValueError(f"LT input role {role} provider_raw_artifact is missing")
        if not isinstance(provider_derivation, Mapping):
            raise ValueError(f"LT input role {role} provider_derivation is missing")
        _, provider_raw_payload = _verify_bound_artifact(
            snapshot_root,
            provider_raw,
            role=f"{role}.provider_raw_artifact",
        )
        _, provider_parser_payload = _verify_bound_artifact(
            snapshot_root,
            {
                "path": provider_derivation.get("parser_code_path"),
                "sha256": provider_derivation.get("parser_code_sha256"),
            },
            role=f"{role}.provider_parser_code",
            size_optional=True,
        )
        _, provider_parser_config_payload = _verify_bound_artifact(
            snapshot_root,
            {
                "path": provider_derivation.get("parser_config_path"),
                "sha256": provider_derivation.get("parser_config_sha256"),
            },
            role=f"{role}.provider_parser_config",
            size_optional=True,
        )
    _, parser_payload = _verify_bound_artifact(
        snapshot_root,
        {
            "path": derivation.get("parser_code_path"),
            "sha256": derivation.get("parser_code_sha256"),
        },
        role=f"{role}.parser_code",
        size_optional=True,
    )
    _, parser_config_payload = _verify_bound_artifact(
        snapshot_root,
        {
            "path": derivation.get("parser_config_path"),
            "sha256": derivation.get("parser_config_sha256"),
        },
        role=f"{role}.parser_config",
        size_optional=True,
    )
    report_path, report_payload = _verify_bound_artifact(
        snapshot_root,
        {
            "path": quality.get("report_path"),
            "sha256": quality.get("report_sha256"),
        },
        role=f"{role}.quality_report",
        size_optional=True,
    )
    try:
        report = load_strict_json(report_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"LT input role {role} quality report is invalid") from exc
    if not isinstance(report, Mapping):
        raise ValueError(f"LT input role {role} quality report must be a mapping")
    expected_quality_schema = (
        "lt_source_quality.v2"
        if require_provider_raw and role in REPLAY_GOVERNED_LT_INPUT_ROLES
        else "lt_source_quality.v1"
    )
    if report.get("schema_version") != expected_quality_schema:
        raise ValueError(f"LT input role {role} quality report schema is unsupported")
    if report.get("status") != "PASS" or str(report.get("role", "")) != role:
        raise ValueError(f"LT input role {role} quality report did not pass")
    if str(report.get("policy_sha256", "")) != str(quality.get("policy_sha256", "")):
        raise ValueError(f"LT input role {role} quality policy binding mismatch")
    if report_path.name == "lt_input_snapshot.json":
        raise ValueError(f"LT input role {role} quality report aliases the contract")
    if role in REPLAY_GOVERNED_LT_INPUT_ROLES:
        try:
            parser_config = load_strict_json(parser_config_payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"LT input role {role} replay config is invalid") from exc
        if not isinstance(parser_config, Mapping):
            raise ValueError(f"LT input role {role} replay config must be a mapping")
        _, derived_payload = _verify_bound_artifact(
            snapshot_root,
            entry,
            role=role,
        )
        if require_provider_raw:
            if (
                provider_raw_payload is None
                or provider_parser_payload is None
                or provider_parser_config_payload is None
                or not isinstance(provider_derivation, Mapping)
            ):
                raise ValueError(
                    f"LT input role {role} provider replay artifacts are incomplete"
                )
            try:
                provider_config = load_strict_json(
                    provider_parser_config_payload.decode("utf-8")
                )
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(
                    f"LT input role {role} provider replay config is invalid"
                ) from exc
            if not isinstance(provider_config, Mapping):
                raise ValueError(
                    f"LT input role {role} provider replay config must be a mapping"
                )
            envelope = validate_raw_envelope(provider_raw_payload)
            receipt = entry.get("source_receipt")
            if not isinstance(receipt, Mapping):
                raise ValueError(f"LT input role {role} source receipt is missing")
            expected_envelope_identity = {
                "acquisition_id": str(entry.get("acquisition_id", "")),
                "source_role": role,
                "source_system": str(entry.get("source_system", "")),
                "source_locator": str(receipt.get("source_locator", "")),
                "received_at_utc": str(receipt.get("received_at_utc", "")),
            }
            for field, expected_value in expected_envelope_identity.items():
                if str(envelope.get(field, "")) != expected_value:
                    raise ValueError(
                        f"LT input role {role} provider envelope {field} mismatch"
                    )
            verify_provider_raw_replay(
                role=role,
                source_system=str(entry.get("source_system", "")),
                envelope_payload=provider_raw_payload,
                bronze_payload=raw_payload,
                parser_payload=provider_parser_payload,
                parser_code_sha256=str(
                    provider_derivation.get("parser_code_sha256", "")
                ),
                parser_config=provider_config,
            )
            _verify_quality_v2_bindings(
                role=role,
                report=report,
                entry=entry,
                raw=raw,
                derivation=derivation,
                provider_raw=entry["provider_raw_artifact"],
                provider_derivation=provider_derivation,
                raw_payload=raw_payload,
                derived_payload=derived_payload,
            )
        verify_core_lt_role_replay(
            role=role,
            source_system=str(entry.get("source_system", "")),
            raw_payload=raw_payload,
            derived_payload=derived_payload,
            parser_payload=parser_payload,
            parser_code_sha256=str(derivation.get("parser_code_sha256", "")),
            parser_config=parser_config,
        )
    if role == "eex_forwards_history":
        catalog = entry.get("vintage_catalog")
        if not isinstance(catalog, Mapping):
            raise ValueError("EEX forward history requires a signed vintage_catalog binding")
        catalog_path, catalog_payload = _verify_bound_artifact(
            snapshot_root,
            catalog,
            role="eex_forwards_history.vintage_catalog",
        )
        try:
            catalog_mapping = load_strict_json(catalog_payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("EEX forward history vintage catalog is invalid") from exc
        if not isinstance(catalog_mapping, Mapping):
            raise ValueError("EEX forward history vintage catalog must be a mapping")
        history_path = _safe_relative_path(snapshot_root, str(entry.get("path", "")))
        from pfc_shaping.data.eex_historical_vintage import (
            verify_eex_historical_vintage_catalog,
        )

        verify_eex_historical_vintage_catalog(
            catalog_mapping,
            catalog_path=catalog_path,
            history_path=history_path,
        )


def _verify_quality_v2_bindings(
    *,
    role: str,
    report: Mapping[str, object],
    entry: Mapping[str, object],
    raw: Mapping[str, object],
    derivation: Mapping[str, object],
    provider_raw: Mapping[str, object],
    provider_derivation: Mapping[str, object],
    raw_payload: bytes,
    derived_payload: bytes,
) -> None:
    bindings = report.get("bindings")
    expected = {
        "provider_raw_sha256": str(provider_raw.get("sha256", "")),
        "provider_parser_code_sha256": str(
            provider_derivation.get("parser_code_sha256", "")
        ),
        "provider_parser_config_sha256": str(
            provider_derivation.get("parser_config_sha256", "")
        ),
        "bronze_sha256": str(raw.get("sha256", "")),
        "feature_parser_code_sha256": str(
            derivation.get("parser_code_sha256", "")
        ),
        "feature_parser_config_sha256": str(
            derivation.get("parser_config_sha256", "")
        ),
        "derived_sha256": str(entry.get("sha256", "")),
    }
    if not isinstance(bindings, Mapping) or dict(bindings) != expected:
        raise ValueError(f"LT input role {role} quality artifact bindings mismatch")
    metrics = report.get("metrics")
    expected_metrics = {
        "bronze": _quality_frame_metrics(raw_payload, role=role, label="bronze"),
        "derived": _quality_frame_metrics(
            derived_payload,
            role=role,
            label="derived",
        ),
    }
    invariant_fields = {
        "row_count",
        "start_utc",
        "end_utc",
        "duplicate_timestamp_count",
    }
    if role.startswith("epex_") and (
        "resolution_provenance" in expected_metrics["bronze"]
        or "resolution_provenance" in expected_metrics["derived"]
    ):
        invariant_fields.add("resolution_provenance")
    if any(
        expected_metrics["bronze"][field] != expected_metrics["derived"][field]
        for field in invariant_fields
    ):
        raise ValueError(f"LT input role {role} quality frame inventories diverge")
    if not isinstance(metrics, Mapping) or dict(metrics) != expected_metrics:
        raise ValueError(f"LT input role {role} quality metrics are not exact")
    if role == "hydro":
        _verify_hydro_scientific_support(
            derived_payload,
            metrics=expected_metrics["derived"],
        )


def _verify_hydro_scientific_support(
    payload: bytes,
    *,
    metrics: Mapping[str, object],
) -> None:
    """Reject hydro evidence that cannot support the causal seasonal estimate."""

    frame = pd.read_parquet(BytesIO(payload))
    if int(metrics["row_count"]) < _HYDRO_MIN_HISTORY_ROWS:
        raise ValueError(
            "LT input role hydro has insufficient scientific history: "
            f"requires at least {_HYDRO_MIN_HISTORY_ROWS} weekly observations"
        )
    history_days = (frame.index.max() - frame.index.min()) / pd.Timedelta(days=1)
    if history_days < _HYDRO_MIN_HISTORY_DAYS:
        raise ValueError(
            "LT input role hydro has insufficient scientific history span: "
            f"requires at least {_HYDRO_MIN_HISTORY_DAYS} days"
        )
    if "water_value_supported" not in frame.columns:
        raise ValueError("LT input role hydro lacks water_value_supported evidence")
    support = frame["water_value_supported"]
    if not pd.api.types.is_bool_dtype(support.dtype):
        raise ValueError("LT input role hydro water_value_supported is not boolean")
    recent = support.sort_index().tail(_HYDRO_RECENT_SUPPORT_ROWS)
    if len(recent) != _HYDRO_RECENT_SUPPORT_ROWS or not bool(recent.all()):
        raise ValueError(
            "LT input role hydro recent water-value estimate is scientifically unsupported"
        )
    for column in (
        "water_value_history_count",
        "water_value_reference_mean_pct",
        "water_value_reference_std_pct",
        "water_value_reference_se_pct",
    ):
        if column not in frame.columns:
            raise ValueError(f"LT input role hydro lacks {column} evidence")
    recent_frame = frame.sort_index().tail(_HYDRO_RECENT_SUPPORT_ROWS)
    if (
        (recent_frame["water_value_history_count"] < 5).any()
        or (recent_frame["water_value_reference_std_pct"] <= 1e-9).any()
        or not np.isfinite(
            recent_frame[
                [
                    "water_value_reference_mean_pct",
                    "water_value_reference_std_pct",
                    "water_value_reference_se_pct",
                ]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            "LT input role hydro recent water-value diagnostics are unsupported"
        )


def _quality_frame_metrics(payload: bytes, *, role: str, label: str) -> dict[str, object]:
    _validate_quality_parquet_allocation_budget(payload, role=role, label=label)
    try:
        frame = pd.read_parquet(BytesIO(payload))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"LT input role {role} quality {label} artifact is not parquet"
        ) from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"LT input role {role} quality {label} frame is empty")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(
            f"LT input role {role} quality {label} index is not timezone-aware"
        )
    metrics = {
        "row_count": len(frame),
        "start_utc": frame.index.min().tz_convert("UTC").isoformat().replace(
            "+00:00", "Z"
        ),
        "end_utc": frame.index.max().tz_convert("UTC").isoformat().replace(
            "+00:00", "Z"
        ),
        "missing_value_count": int(frame.isna().sum().sum()),
        "duplicate_timestamp_count": int(frame.index.duplicated().sum()),
    }
    if role.startswith("epex_") and OBSERVATION_RESOLUTION_PROVENANCE_ATTR in frame.attrs:
        metrics["resolution_provenance"] = energy_price_resolution_provenance(
            frame,
            role=role,
        )
    if metrics["missing_value_count"] != 0:
        raise ValueError(f"LT input role {role} quality {label} frame has missing values")
    if metrics["duplicate_timestamp_count"] != 0:
        raise ValueError(f"LT input role {role} quality {label} frame repeats timestamps")
    return metrics


def _validate_quality_parquet_allocation_budget(
    payload: bytes,
    *,
    role: str,
    label: str,
) -> None:
    """Inspect only Parquet metadata before allocating a decoded DataFrame."""

    validate_parquet_allocation_budget(
        payload,
        label=f"LT input role {role} quality {label} artifact",
        max_rows=_QUALITY_MAX_ROWS_BY_ROLE.get(role, 2_000_000),
        max_columns=_QUALITY_MAX_COLUMNS,
        max_cells=_QUALITY_MAX_CELLS,
        max_row_groups=_QUALITY_MAX_ROW_GROUPS,
        allowed_physical_types=_QUALITY_ALLOWED_PARQUET_PHYSICAL_TYPES,
    )


def _verify_bound_artifact(
    snapshot_root: Path,
    binding: Mapping[str, object],
    *,
    role: str,
    size_optional: bool = False,
) -> tuple[Path, bytes]:
    path = _safe_relative_path(snapshot_root, str(binding.get("path", "")))
    maximum = _artifact_max_bytes(role)
    expected_size = binding.get("size_bytes")
    if not size_optional and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or expected_size > maximum
    ):
        raise ValueError(f"LT snapshot supporting artifact size is invalid: {role}")
    payload = read_stable_single_link_file(
        path,
        label=f"LT snapshot artifact {role}",
        max_bytes=(maximum if size_optional else int(expected_size)),
    )
    expected_hash = str(binding.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"LT snapshot supporting artifact hash is invalid: {role}")
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"LT snapshot supporting artifact hash mismatch: {role}")
    if not size_optional:
        if expected_size != len(payload):
            raise ValueError(f"LT snapshot supporting artifact size mismatch: {role}")
    return path, payload


def _artifact_max_bytes(role: str) -> int:
    normalized = str(role).lower()
    if "provider_raw_artifact" in normalized:
        return _MAX_PROVIDER_RAW_ARTIFACT_BYTES
    if normalized in {
        "epex_ch",
        "epex_de",
        "epex_at",
        "epex_fr",
        "epex_it",
        "entso",
        "hydro",
        "eex_forwards_history",
    } or normalized.endswith(".raw_artifact"):
        return _MAX_PARQUET_ARTIFACT_BYTES
    return _MAX_SUPPORTING_ARTIFACT_BYTES


def _parse_available_at(value: object, *, label: str) -> pd.Timestamp:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} is missing available_at_utc")
    try:
        timestamp = pd.Timestamp(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid available_at_utc") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} available_at_utc must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _receipt(
    path: Path,
    payload: bytes,
    *,
    role: str,
    logical_path: str | None,
    rows: int | None = None,
    frame_sha256: str | None = None,
) -> InputSourceReceipt:
    return InputSourceReceipt(
        role=str(role),
        path=str(path.resolve()),
        logical_path=str(logical_path or path.name).replace("\\", "/"),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        rows=rows,
        frame_sha256=frame_sha256,
    )
