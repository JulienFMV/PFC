"""Canonical LT input snapshots and byte-level source receipts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pfc_shaping.data.acquisition_contract import verify_acquisition_contract
from pfc_shaping.data.shared_data_root import (
    FMV_DATA_ROOT_ENV,
    PFC_LT_DATA_ROOT_ENV,
    PFC_SHARED_DATA_ROOT_ENV,
    configured_lt_data_root,
    resolve_confined_path,
)
from pfc_shaping.path_safety import path_is_link
from pfc_shaping.pipeline.strict_structured_data import load_strict_json, load_strict_yaml

CORE_LT_INPUT_ROLES = ("epex_ch", "epex_de", "entso", "hydro")
GOVERNED_SOURCE_SYSTEMS_BY_ROLE = {
    "epex_ch": frozenset({"EPEX_SPOT"}),
    "epex_de": frozenset({"EPEX_SPOT"}),
    "epex_at": frozenset({"EPEX_SPOT"}),
    "epex_fr": frozenset({"EPEX_SPOT"}),
    "epex_it": frozenset({"EPEX_SPOT"}),
    "entso": frozenset({"ENTSOE_TRANSPARENCY_PLATFORM"}),
    "outages": frozenset({"ENTSOE_TRANSPARENCY_PLATFORM"}),
    "hydro": frozenset({"FMV_HYDRO_CURATED"}),
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
    pointer_receipt: InputSourceReceipt | None = None
    contract_receipt: InputSourceReceipt | None = None

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
    root = root_candidate.resolve()
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
    pointer, pointer_receipt = read_json_snapshot(
        pointer_path,
        role="lt_data_pointer",
        logical_path=pointer_logical_path,
    )
    if pointer.get("schema_version") != "lt_data_pointer.v1":
        raise ValueError("LT data pointer schema_version must be lt_data_pointer.v1")
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
    contract, contract_receipt = read_json_snapshot(
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
    if contract.get("schema_version") != "lt_input_snapshot.v1":
        raise ValueError("LT input contract schema_version must be lt_input_snapshot.v1")
    if contract.get("layout") != "external_v2":
        raise ValueError("LT input contract layout must be external_v2")
    if str(contract.get("generation_id", "")) != generation_id:
        raise ValueError("LT data pointer/contract generation_id mismatch")
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
    eligible = contract.get("calibration_eligible") is True
    if eligible:
        if contract.get("source_class") != "GOVERNED_ACQUISITION":
            raise ValueError("calibration-eligible LT input must be a GOVERNED_ACQUISITION")
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
        pointer_receipt=pointer_receipt,
        contract_receipt=contract_receipt,
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

    if contract.get("schema_version") != "lt_input_snapshot.v1":
        raise ValueError("LT input contract schema_version must be lt_input_snapshot.v1")
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
    for role, entry in files.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"LT input contract role {role} must be a mapping")
        if str(entry.get("acquisition_id", "")) != acquisition_id:
            raise ValueError(f"LT input role has a divergent acquisition_id: {role}")
        role_available = _parse_available_at(
            entry.get("available_at_utc"),
            label=f"LT input role {role}",
        )
        if role_available != available_at:
            raise ValueError(f"LT input role has a divergent acquisition cutoff: {role}")
        logical_path = str(entry.get("path", "")).strip()
        if not logical_path:
            raise ValueError(f"LT input role path is missing: {role}")
        relative = Path(logical_path)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(f"LT input role path is not portable and relative: {role}")
    eligible = contract.get("calibration_eligible") is True
    if eligible:
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
    return {
        "generation_id": generation_id,
        "acquisition_id": acquisition_id,
        "available_at_utc": available_at.isoformat(),
        "calibration_eligible": eligible,
        "files": {str(role): dict(entry) for role, entry in files.items()},
    }


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
    reference = _parse_available_at(reference_timestamp, label="LT valuation timestamp")
    contract_available = _parse_available_at(
        paths.available_at_utc,
        label="LT input contract",
    )
    if contract_available > reference:
        raise ValueError("LT input contract is not available at valuation timestamp")
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
        if role_available != contract_available:
            raise ValueError(
                f"consumed LT input role has a divergent acquisition cutoff: {role}"
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

    source = Path(path).resolve()
    payload = source.read_bytes()
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
    source = Path(path).resolve()
    payload = source.read_bytes()
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
    source = Path(path).resolve()
    payload = source.read_bytes()
    receipt = _receipt(source, payload, role=role, logical_path=logical_path)
    parsed = load_strict_json(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{role} JSON must contain a mapping: {source}")
    return parsed, receipt


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
    payload = source.read_bytes()
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
