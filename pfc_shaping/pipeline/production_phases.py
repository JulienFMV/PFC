from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.governed_lt_acquisition import (
    require_native_quarter_hour_price_truth,
)
from pfc_shaping.data.lt_input_sources import (
    InputSourceReceipt,
    consumed_lt_input_roles,
    read_parquet_snapshot,
    read_yaml_snapshot,
    resolve_lt_input_paths,
    validate_governed_forward_history_frame,
    validate_lt_input_consumption,
    validate_receipt_against_contract,
    verify_frame_receipt,
    verify_source_receipt,
)
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
)


# (lightgbm, torch, …) into the interpreter. The names are still exposed
@dataclass
class LoadedInputs:
    epex_ch: pd.DataFrame
    epex_de: pd.DataFrame
    neighbor_prices_15min: dict[str, pd.DataFrame]
    entso: pd.DataFrame
    hydro: pd.DataFrame
    cal_ch: pd.DataFrame
    cal_de: pd.DataFrame
    commodities: pd.DataFrame | None
    outages_all: pd.DataFrame | None
    config: dict
    sh_mode: str
    eex_report_path: str | None
    eex_report_available_at: pd.Timestamp | None = None
    reference_timestamp: pd.Timestamp | None = None
    reference_timestamp_is_explicit: bool = False
    freshness_reports: dict[str, object] = field(default_factory=dict)
    input_source_receipts: dict[str, InputSourceReceipt] = field(default_factory=dict)
    config_source_receipt: InputSourceReceipt | None = None
    data_root: str | None = None
    data_layout: str | None = None
    data_generation_id: str | None = None
    data_contract_receipt: InputSourceReceipt | None = None
    data_pointer_receipt: InputSourceReceipt | None = None
    data_publication_intent_receipt: InputSourceReceipt | None = None
    data_publication_anchor_receipt: InputSourceReceipt | None = None
    data_publication_head_observation_receipt: InputSourceReceipt | None = None
    config_semantic_sha256: str | None = None
    input_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    data_contract_bytes: bytes | None = None
    data_pointer_bytes: bytes | None = None
    data_publication_intent_bytes: bytes | None = None
    data_publication_anchor_bytes: bytes | None = None
    data_publication_head_observation_bytes: bytes | None = None
    eex_forwards_history: pd.DataFrame | None = None
    eex_historical_vintage_catalog_path: str | None = None


@dataclass
class SharedStructuralArtifacts:
    si: object
    unc: object | None
    calibrator: object | None
    entso_forecast: pd.DataFrame
    start_date: str
    horizon_days: int
    delivery_index: pd.DatetimeIndex | None = None


@dataclass
class MarketSpec:
    """Per-market description used to drive ``_build_long_term_branch``.

    Captures everything that distinguishes one country's PFC build from
    another. Adding a new market (FR / AT / IT) means assembling one
    ``MarketSpec``; the rest of the pipeline is market-agnostic.
    """

    code: str  # ISO-2 market code: 'CH', 'DE', 'AT', 'FR', 'IT'
    sheet: str  # EEX workbook sheet name (today: same as code)
    tz: str  # IANA timezone (e.g. 'Europe/Zurich')
    country: str  # holidays / EEX peak country code
    epex_df: pd.DataFrame  # market-specific EPEX spot history
    cal_df: pd.DataFrame  # market-specific calendar enrichment
    pre_fitted_sh: object | None = None  # pre-fitted ShapeHourly (CH); None → fit inside branch
    water_value: object | None = None  # only for hydro markets (CH today)
    hydro_forecast: pd.DataFrame | None = None  # only for hydro markets
    outages_forecast: pd.DataFrame | None = None  # only when REMIT outages are wired
    out_base: str = ""  # destination prefix for the {.parquet, .csv} pair


@dataclass
class MarketBranchArtifacts:
    """Output of ``_build_long_term_branch`` for one market.

    Every field is per-market; the LongTermArtifacts container keeps
    a dict keyed by market code plus convenience aliases ``swiss`` /
    ``german`` for backward compatibility.
    """

    code: str
    pfc: pd.DataFrame
    sh: object
    base_prices: dict
    cascaded_prices: dict
    fwd_source: str
    out_base: str
    forward_snapshot: dict[str, object] | None = None
    final_product_projection_report: dict[str, object] = field(default_factory=dict)
    wv: object | None = None
    hydro_forecast: pd.DataFrame | None = None
    outages_forecast: pd.DataFrame | None = None


# ── Backward-compat aliases ──────────────────────────────────────────────
# Older external code refers to ``SwissLongTermArtifacts`` /
# ``GermanLongTermArtifacts``. They are now the same object as
# ``MarketBranchArtifacts``; the per-country attribute names
# (``base_prices_ch``, ``cascaded_prices_ch`` etc.) are preserved as
# read-only properties on top of the generic dataclass for any caller
# that depends on the old shape.
SwissLongTermArtifacts = MarketBranchArtifacts
GermanLongTermArtifacts = MarketBranchArtifacts


def _legacy_alias(self: MarketBranchArtifacts, attr: str) -> object:
    """Return self.attr (used by legacy *_ch / *_de property aliases)."""
    return getattr(self, attr)


# Attach legacy attribute aliases on the dataclass so ``art.base_prices_ch``
# keeps returning ``art.base_prices`` on a CH branch (and similarly _de on
# a DE branch). This lets the dashboard pages and any leftover scripts
# read the old field names without modification.
for _legacy_name, _modern_name in [
    ("base_prices_ch", "base_prices"),
    ("cascaded_prices_ch", "cascaded_prices"),
    ("base_prices_de", "base_prices"),
    ("cascaded_prices_de", "cascaded_prices"),
]:
    setattr(
        MarketBranchArtifacts,
        _legacy_name,
        property(lambda self, _m=_modern_name: _legacy_alias(self, _m)),
    )


@dataclass
class LongTermArtifacts:
    """Top-level LT artifacts container.

    ``markets`` is the source of truth: one entry per active market, keyed
    by ISO-2 code. ``swiss`` and ``german`` properties are convenience
    accessors for the two markets currently wired in production.
    Activating FR / AT / IT only requires adding entries to ``markets`` —
    no field added on this dataclass.
    """

    shared: SharedStructuralArtifacts
    markets: dict[str, MarketBranchArtifacts]
    out_dir: str
    artifacts_dir: str
    today: str
    monthly_curve_manifests: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def swiss(self) -> MarketBranchArtifacts:
        return self.markets["CH"]

    @property
    def german(self) -> MarketBranchArtifacts:
        return self.markets["DE"]


def _first_existing_path(*paths: str | None) -> str | None:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _json_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _forward_valuation_timestamp(inputs: LoadedInputs, snapshot: object) -> pd.Timestamp:
    reference = pd.Timestamp(inputs.reference_timestamp or pd.Timestamp.now("UTC"))
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    else:
        reference = reference.tz_convert("UTC")
    available_at = getattr(snapshot, "available_at", None)
    if not inputs.reference_timestamp_is_explicit and available_at is not None:
        observed = pd.Timestamp(available_at).tz_convert("UTC")
        reference = max(reference, observed)
    return reference


def _next_delivery_start_date(
    reference_timestamp: str | pd.Timestamp,
    *,
    timezone: str,
) -> str:
    reference = pd.Timestamp(reference_timestamp)
    if reference.tzinfo is None:
        raise ValueError("delivery start reference timestamp must be timezone-aware")
    return (reference.tz_convert(timezone).normalize() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")


def _solver_delivery_quarter_hour_grid(
    months: pd.PeriodIndex,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    if months.empty:
        raise ValueError("solver delivery month grid is empty")
    expected = pd.period_range(months.min(), months.max(), freq="M")
    if not months.equals(expected):
        raise ValueError("solver delivery months must be sorted, unique, and contiguous")
    local_start = months.min().to_timestamp(how="start").tz_localize(timezone)
    local_end = (months.max() + 1).to_timestamp(how="start").tz_localize(timezone)
    return pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )


def _validate_governed_forward_history_receipt(
    history: pd.DataFrame,
    receipt: InputSourceReceipt,
    *,
    reference_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """Bind a consumed frame to its receipt and validate EEX semantics."""

    verify_source_receipt(receipt)
    verify_frame_receipt(history, receipt)
    frame = validate_governed_forward_history_frame(
        history,
        reference_timestamp=reference_timestamp,
    )
    return frame, receipt.sha256


def _prepare_governed_forward_history(
    history: pd.DataFrame,
    receipt: InputSourceReceipt,
    *,
    catalog_path: str | Path,
    reference_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, str, dict[str, object], object]:
    """Validate the exact catalog-backed PIT history consumed by the solver."""

    from pfc_shaping.data.eex_historical_vintage import (
        materialize_eex_forward_history_as_of,
        verify_eex_historical_vintage_catalog,
    )

    verify_source_receipt(receipt)
    verify_frame_receipt(history, receipt)
    source_hash = receipt.sha256
    selected_catalog = Path(catalog_path).resolve()
    try:
        catalog_payload = read_stable_single_link_file(
            selected_catalog,
            label="signed EEX historical vintage catalog",
        )
        catalog = load_strict_json(catalog_payload.decode("utf-8"))
    except (UnicodeDecodeError, StrictStructuredDataError, ValueError) as exc:
        raise ValueError("signed EEX historical vintage catalog is unreadable") from exc
    if not isinstance(catalog, dict):
        raise ValueError("signed EEX historical vintage catalog must be a mapping")
    vintages, evidence = verify_eex_historical_vintage_catalog(
        catalog,
        catalog_path=selected_catalog,
        history_path=receipt.path,
    )
    if evidence.catalog_sha256 != hashlib.sha256(catalog_payload).hexdigest():
        raise ValueError("verified EEX vintage catalog differs from consumed bytes")
    if evidence.history_sha256 != source_hash:
        raise ValueError("verified EEX vintage history differs from consumed receipt")
    selected = materialize_eex_forward_history_as_of(
        vintages,
        valuation_timestamp=reference_timestamp,
    )
    from pfc_shaping.data.forward_proxy import (
        forward_snapshot_from_verified_eex_vintage,
    )

    forward_snapshot = forward_snapshot_from_verified_eex_vintage(
        selected,
        evidence=evidence,
        catalog=catalog,
        catalog_path=selected_catalog,
        market="CH",
    )
    frame = validate_governed_forward_history_frame(
        selected,
        reference_timestamp=reference_timestamp,
    )
    vintage_evidence = {
        "catalog_id": evidence.catalog_id,
        "catalog_sha256": evidence.catalog_sha256,
        "history_sha256": evidence.history_sha256,
        "snapshot_count": evidence.snapshot_count,
        "source_document_count": evidence.source_document_count,
        "data_cutoff_utc": evidence.data_cutoff_utc,
        "status": "VERIFIED_SIGNED_IMMUTABLE_VINTAGES",
        "pit_selection": dict(selected.attrs.get("pit_selection", {})),
    }
    frame.attrs["verified_vintage_catalog"] = vintage_evidence
    return frame, source_hash, vintage_evidence, forward_snapshot


def _validate_production_input_freshness(
    *,
    epex_ch: pd.DataFrame,
    epex_de: pd.DataFrame,
    entso: pd.DataFrame,
    hydro: pd.DataFrame,
    outages: pd.DataFrame | None,
    config: dict,
    reference_timestamp: str | pd.Timestamp | None,
    neighbor_prices_15min: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    from pfc_shaping.pipeline.quality_gate import (
        QualityGateError,
        input_grid_errors,
        validate_input_frame,
        weekly_local_grid_errors,
    )

    policy = dict((config.get("quality", {}) or {}).get("freshness", {}) or {})
    _validate_strict_production_freshness_policy(policy, QualityGateError)
    fail_on_stale = True
    specs = {
        "epex_ch": (
            epex_ch,
            ["price_eur_mwh"],
            float(policy.get("epex_max_age_hours", 48.0)) / 24.0,
            float(policy.get("epex_max_gap_hours", 1.0)) / 24.0,
        ),
        "epex_de": (
            epex_de,
            ["price_eur_mwh"],
            float(policy.get("epex_max_age_hours", 48.0)) / 24.0,
            float(policy.get("epex_max_gap_hours", 1.0)) / 24.0,
        ),
        "entso": (
            entso,
            ["load_mw", "solar_mw", "wind_mw"],
            float(policy.get("entso_max_age_hours", 72.0)) / 24.0,
            float(policy.get("entso_max_gap_hours", 2.0)) / 24.0,
        ),
        "hydro": (
            hydro,
            ["fill_deviation", "water_value_supported"],
            float(policy.get("hydro_max_age_days", 10.0)),
            float(policy.get("hydro_max_gap_days", 14.0)),
        ),
    }
    for code, frame in sorted((neighbor_prices_15min or {}).items()):
        if code == "de":
            continue
        specs[f"epex_{code}"] = (
            frame,
            ["price_eur_mwh"],
            float(policy.get("epex_max_age_hours", 48.0)) / 24.0,
            float(policy.get("epex_max_gap_hours", 1.0)) / 24.0,
        )
    reports: dict[str, object] = {}
    for name, (frame, columns, max_age_days, max_gap_days) in specs.items():
        cadence_seconds = 7 * 24 * 60 * 60 if name == "hydro" else 15 * 60
        grid_errors = (
            weekly_local_grid_errors(
                frame,
                name=name,
                timezone="Europe/Zurich",
                weekday=0,
                min_grid_coverage=0.95,
            )
            if name == "hydro"
            else input_grid_errors(
                frame,
                name=name,
                expected_cadence_seconds=cadence_seconds,
                min_grid_coverage=0.95,
            )
        )
        if grid_errors:
            raise QualityGateError("; ".join(grid_errors))
        reports[name] = validate_input_frame(
            frame,
            name=name,
            required_columns=columns,
            min_rows=1,
            max_age_days=max_age_days,
            reference_timestamp=reference_timestamp,
            fail_on_stale=fail_on_stale,
            min_finite_fraction=float(policy.get("min_finite_fraction", 0.95)),
            recent_window_days=float(policy.get("recent_window_days", 7.0)),
            min_recent_finite_fraction=float(policy.get("min_recent_finite_fraction", 0.95)),
            max_finite_gap_days=max_gap_days,
        )
        if name == "hydro":
            _validate_hydro_recent_scientific_support(frame, QualityGateError)

    outages_enabled = bool(policy.get("outages_enabled", False))
    if outages_enabled:
        if outages is None:
            raise QualityGateError("outages: enabled for production but dataset is missing")
        grid_errors = input_grid_errors(
            outages,
            name="outages",
            expected_cadence_seconds=15 * 60,
            min_grid_coverage=0.95,
        )
        if grid_errors:
            raise QualityGateError("; ".join(grid_errors))
        reports["outages"] = validate_input_frame(
            outages,
            name="outages",
            required_columns=["unavailable_mw", "n_outages"],
            min_rows=1,
            max_age_days=float(policy.get("outages_max_age_hours", 48.0)) / 24.0,
            reference_timestamp=reference_timestamp,
            fail_on_stale=fail_on_stale,
            min_finite_fraction=float(policy.get("min_finite_fraction", 0.95)),
            recent_window_days=float(policy.get("recent_window_days", 7.0)),
            min_recent_finite_fraction=float(policy.get("min_recent_finite_fraction", 0.95)),
            max_finite_gap_days=float(policy.get("outages_max_gap_hours", 2.0)) / 24.0,
        )
    return reports


def _validate_hydro_recent_scientific_support(
    hydro: pd.DataFrame,
    error_type: type[RuntimeError],
) -> None:
    """Require an explicitly supported causal water-value estimate in production."""

    column = "water_value_supported"
    if column not in hydro.columns:
        raise error_type(f"hydro: missing required column {column}")
    support = hydro[column]
    if not pd.api.types.is_bool_dtype(support.dtype):
        raise error_type(f"hydro: {column} must be boolean")
    recent = support.sort_index().tail(8)
    if len(recent) != 8 or not bool(recent.all()):
        raise error_type(
            "hydro: recent water-value estimate is scientifically unsupported; "
            "eight supported weekly observations are required"
        )
    diagnostics = (
        "water_value_history_count",
        "water_value_reference_mean_pct",
        "water_value_reference_std_pct",
        "water_value_reference_se_pct",
    )
    missing = [column for column in diagnostics if column not in hydro.columns]
    if missing:
        raise error_type(f"hydro: missing water-value diagnostics {missing}")
    recent_frame = hydro.sort_index().tail(8)
    if (
        (recent_frame["water_value_history_count"] < 5).any()
        or (recent_frame["water_value_reference_std_pct"] <= 1e-9).any()
        or not np.isfinite(recent_frame[list(diagnostics[1:])].to_numpy(dtype=float)).all()
    ):
        raise error_type("hydro: recent water-value diagnostics are unsupported")


def _validate_strict_production_freshness_policy(
    policy: dict[str, object],
    error_type: type[RuntimeError],
) -> None:
    if policy.get("fail_on_stale_inputs", True) is not True:
        raise error_type("production freshness policy cannot disable fail_on_stale_inputs")
    minimums = {
        "min_finite_fraction": 0.95,
        "min_recent_finite_fraction": 0.95,
        "recent_window_days": 7.0,
    }
    maximums = {
        "epex_max_age_hours": 48.0,
        "entso_max_age_hours": 72.0,
        "hydro_max_age_days": 10.0,
        "epex_max_gap_hours": 1.0,
        "entso_max_gap_hours": 2.0,
        "hydro_max_gap_days": 14.0,
        "outages_max_age_hours": 48.0,
        "outages_max_gap_hours": 2.0,
    }
    for key, floor in minimums.items():
        value = float(policy.get(key, floor))
        if not math.isfinite(value):
            raise error_type(f"production freshness policy {key} must be finite")
        if value < floor:
            raise error_type(
                f"production freshness policy {key}={value} is below sanctioned minimum {floor}"
            )
    for key, ceiling in maximums.items():
        value = float(policy.get(key, ceiling))
        if not math.isfinite(value):
            raise error_type(f"production freshness policy {key} must be finite")
        if value > ceiling:
            raise error_type(
                f"production freshness policy {key}={value} exceeds sanctioned maximum {ceiling}"
            )


def _future_hydro_civil_weekly_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return Monday 00:00 Europe/Zurich points, represented exactly in UTC."""

    start_utc = pd.Timestamp(start)
    end_utc = pd.Timestamp(end)
    start_utc = (
        start_utc.tz_localize("UTC")
        if start_utc.tzinfo is None
        else start_utc.tz_convert("UTC")
    )
    end_utc = (
        end_utc.tz_localize("UTC")
        if end_utc.tzinfo is None
        else end_utc.tz_convert("UTC")
    )
    local = pd.date_range(
        start=start_utc.tz_convert("Europe/Zurich").normalize(),
        end=end_utc.tz_convert("Europe/Zurich").normalize(),
        freq="W-MON",
        inclusive="left",
    )
    return local.tz_convert("UTC")


def load_inputs(
    project_root: str,
    logger: logging.Logger,
    *,
    reference_timestamp: str | pd.Timestamp | None = None,
    data_root: str | Path | None = None,
    eex_report_path: str | Path | None = None,
    eex_report_available_at: str | pd.Timestamp | None = None,
    config_path: str | Path | None = None,
    expected_config_sha256: str | None = None,
    input_snapshot_contract: str | Path | None = None,
    input_pointer_contract: str | Path | None = None,
    expected_input_snapshot_sha256: str | None = None,
    expected_input_pointer_sha256: str | None = None,
    expected_input_generation_id: str | None = None,
    publication_head_observation: str | Path | None = None,
    expected_publication_head_observation_sha256: str | None = None,
    publication_head_challenge_nonce: str | None = None,
) -> LoadedInputs:
    logger.info("=" * 70)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 70)
    t0 = time.time()
    resolved_reference = (
        pd.Timestamp(reference_timestamp)
        if reference_timestamp is not None
        else pd.Timestamp.now("UTC")
    )
    if resolved_reference.tzinfo is None:
        resolved_reference = resolved_reference.tz_localize("UTC")
    else:
        resolved_reference = resolved_reference.tz_convert("UTC")

    selected_config = (
        Path(config_path)
        if config_path is not None
        else Path(project_root) / "pfc_shaping" / "config.yaml"
    )
    config, config_receipt = read_yaml_snapshot(selected_config)
    if (
        expected_config_sha256 is not None
        and config_receipt.sha256 != str(expected_config_sha256).strip().lower()
    ):
        raise ValueError("LT config SHA-256 does not match --config-sha256")
    input_paths = resolve_lt_input_paths(
        project_root,
        data_root=data_root,
        expected_snapshot_contract=input_snapshot_contract,
        expected_pointer_contract=input_pointer_contract,
        expected_snapshot_sha256=expected_input_snapshot_sha256,
        expected_pointer_sha256=expected_input_pointer_sha256,
        expected_generation_id=expected_input_generation_id,
        publication_head_observation=publication_head_observation,
        expected_publication_head_observation_sha256=(expected_publication_head_observation_sha256),
        publication_head_challenge_nonce=publication_head_challenge_nonce,
    )
    logger.info("  LT input root: %s (%s)", input_paths.root, input_paths.layout)
    if input_paths.layout != "external_v2":
        raise ValueError("LT production inputs require an explicit governed external_v2 data root")
    if not input_paths.calibration_eligible:
        raise ValueError(
            f"LT input generation {input_paths.generation_id} is not calibration eligible"
        )
    consumed_roles = consumed_lt_input_roles(
        config,
        available_roles=set(input_paths.files),
    )
    validate_lt_input_consumption(
        input_paths,
        roles=consumed_roles,
        reference_timestamp=resolved_reference,
    )
    input_receipts: dict[str, InputSourceReceipt] = {}
    input_frames: dict[str, pd.DataFrame] = {}

    def read_required(path: Path, label: str, role: str) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required {label} parquet: {path}")
        try:
            logical_path = path.relative_to(input_paths.snapshot_root).as_posix()
            frame, receipt = read_parquet_snapshot(
                path,
                role=role,
                logical_path=logical_path,
            )
            validate_receipt_against_contract(input_paths, receipt)
        except Exception:
            logger.exception("  Failed to read required %s parquet: %s", label, path)
            raise
        if frame.empty:
            raise ValueError(f"Required {label} parquet is empty: {path}")
        input_receipts[role] = receipt
        input_frames[role] = frame
        return frame

    epex_ch = read_required(input_paths.epex_ch, "EPEX CH", "epex_ch")
    epex_de = read_required(input_paths.epex_de, "EPEX DE", "epex_de")
    neighbor_prices_15min = {
        "de": epex_de,
    }
    for code in ["at", "fr", "it"]:
        role = f"epex_{code}"
        if input_paths.has_role(role):
            path = input_paths.neighbor_epex(code)
            neighbor_prices_15min[code] = read_required(
                path, f"EPEX {code.upper()}", f"epex_{code}"
            )
    entso = read_required(input_paths.entso, "ENTSO-E", "entso")
    hydro = read_required(input_paths.hydro, "Hydro reservoir", "hydro")
    eex_forwards_history = None
    eex_historical_vintage_catalog_path = None
    if "eex_forwards_history" in consumed_roles:
        if not input_paths.has_role("eex_forwards_history"):
            raise ValueError(
                "CH monthly solver requires governed external_v2 role eex_forwards_history"
            )
        eex_forwards_history = read_required(
            input_paths.eex_forwards_history,
            "EEX forwards history",
            "eex_forwards_history",
        )
        try:
            eex_historical_vintage_catalog_path = str(
                input_paths.supporting_path("eex_forwards_history", "vintage_catalog")
            )
        except KeyError as exc:
            raise ValueError(
                "CH monthly solver requires a signed EEX historical vintage catalog"
            ) from exc

    logger.info(
        "  EPEX CH:  %d rows  [%s -> %s]",
        len(epex_ch),
        epex_ch.index.min().date(),
        epex_ch.index.max().date(),
    )
    logger.info(
        "  EPEX DE:  %d rows  [%s -> %s]",
        len(epex_de),
        epex_de.index.min().date(),
        epex_de.index.max().date(),
    )
    logger.info(
        "  ENTSO-E:  %d rows  [%s -> %s]",
        len(entso),
        entso.index.min().date(),
        entso.index.max().date(),
    )
    logger.info(
        "  Hydro:    %d rows  [%s -> %s]",
        len(hydro),
        hydro.index.min().date(),
        hydro.index.max().date(),
    )
    logger.info("  Data loaded in %.1fs", time.time() - t0)

    logger.info("=" * 70)
    logger.info("STEP 2: Calendar enrichment")
    logger.info("=" * 70)
    t1 = time.time()
    from pfc_shaping.data.calendar_ch import enrich_15min_index

    cal_ch = enrich_15min_index(epex_ch.index, country="CH")
    cal_de = enrich_15min_index(epex_de.index, country="DE")

    logger.info(
        "  CH calendar: %d rows, types: %s", len(cal_ch), dict(cal_ch["type_jour"].value_counts())
    )
    logger.info("  DE calendar: %d rows", len(cal_de))
    logger.info("  Calendar enriched in %.1fs", time.time() - t1)

    logger.info("=" * 70)
    logger.info("STEP 3: Feature engineering (solar_regime, load_deviation)")
    logger.info("=" * 70)
    logger.info(
        "  solar_regime stats: mean=%.2f, std=%.2f",
        entso["solar_regime"].mean(),
        entso["solar_regime"].std(),
    )
    logger.info(
        "  load_deviation stats: mean=%.2f, std=%.2f",
        entso["load_deviation"].mean(),
        entso["load_deviation"].std(),
    )

    model_cfg = config.get("model", {})
    forwards_cfg = config.get("forwards", {})
    sh_mode = model_cfg.get("shape_hourly_mode", "table")
    selected_eex_report = None
    if eex_report_path is not None:
        explicit_report = Path(eex_report_path).expanduser()
        if not explicit_report.is_absolute() or not explicit_report.is_file():
            raise FileNotFoundError(f"explicit EEX report path is unavailable: {explicit_report}")
        selected_eex_report = str(explicit_report.resolve())
    else:
        selected_eex_report = _first_existing_path(
            forwards_cfg.get("eex_report_path"),
            forwards_cfg.get("eex_report_path_unc"),
        )
    if selected_eex_report:
        logger.info("  EEX report path selected: %s", selected_eex_report)
    else:
        raise FileNotFoundError("No explicit or configured EEX report path is available")
    selected_eex_available_at = None
    if eex_report_available_at is not None:
        selected_eex_available_at = pd.Timestamp(eex_report_available_at)
        if selected_eex_available_at.tzinfo is None:
            raise ValueError("EEX report availability must be timezone-aware")
        selected_eex_available_at = selected_eex_available_at.tz_convert("UTC")

    commodities = None
    if input_paths.has_role("commodities"):
        commodities = read_required(input_paths.commodities, "commodities", "commodities")
    outages_all = None
    if "outages" in consumed_roles:
        outages_all = read_required(input_paths.outages, "outages", "outages")

    freshness_reports = _validate_production_input_freshness(
        epex_ch=epex_ch,
        epex_de=epex_de,
        entso=entso,
        hydro=hydro,
        outages=outages_all,
        config=config,
        reference_timestamp=resolved_reference,
        neighbor_prices_15min=neighbor_prices_15min,
    )
    for name, report in freshness_reports.items():
        logger.info("  Freshness gate %s: PASS (%d checks)", name, report.checks_passed)
    if "outages" not in consumed_roles:
        outages_all = None
        logger.info(
            "  Outages feature explicitly disabled until acquisition provenance is production-ready"
        )

    return LoadedInputs(
        epex_ch=epex_ch,
        epex_de=epex_de,
        neighbor_prices_15min=neighbor_prices_15min,
        entso=entso,
        hydro=hydro,
        cal_ch=cal_ch,
        cal_de=cal_de,
        commodities=commodities,
        outages_all=outages_all,
        config=config,
        sh_mode=sh_mode,
        eex_report_path=selected_eex_report,
        eex_report_available_at=selected_eex_available_at,
        reference_timestamp=resolved_reference,
        reference_timestamp_is_explicit=reference_timestamp is not None,
        freshness_reports=freshness_reports,
        input_source_receipts=input_receipts,
        config_source_receipt=config_receipt,
        data_root=str(input_paths.root),
        data_layout=input_paths.layout,
        data_generation_id=input_paths.generation_id,
        data_contract_receipt=input_paths.contract_receipt,
        data_pointer_receipt=input_paths.pointer_receipt,
        data_publication_intent_receipt=(input_paths.publication_intent_receipt),
        data_publication_anchor_receipt=(input_paths.publication_anchor_receipt),
        data_publication_head_observation_receipt=(
            input_paths.publication_head_observation_receipt
        ),
        config_semantic_sha256=_json_sha256(config),
        input_frames=input_frames,
        data_contract_bytes=(
            read_stable_single_link_file(
                input_paths.contract_receipt.path,
                label="LT input snapshot contract",
            )
            if input_paths.contract_receipt is not None
            else None
        ),
        data_pointer_bytes=(
            read_stable_single_link_file(
                input_paths.pointer_receipt.path,
                label="LT data pointer",
            )
            if input_paths.pointer_receipt is not None
            else None
        ),
        data_publication_intent_bytes=(
            read_stable_single_link_file(
                input_paths.publication_intent_receipt.path,
                label="publication intent",
            )
            if input_paths.publication_intent_receipt is not None
            else None
        ),
        data_publication_anchor_bytes=(
            read_stable_single_link_file(
                input_paths.publication_anchor_receipt.path,
                label="publication anchor receipt",
            )
            if input_paths.publication_anchor_receipt is not None
            else None
        ),
        data_publication_head_observation_bytes=(
            read_stable_single_link_file(
                input_paths.publication_head_observation_receipt.path,
                label="publication HEAD observation",
            )
            if input_paths.publication_head_observation_receipt is not None
            else None
        ),
        eex_forwards_history=eex_forwards_history,
        eex_historical_vintage_catalog_path=eex_historical_vintage_catalog_path,
    )


def _require_production_intraday_price_truth(
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Admit QH training only after cadence and product identity are proven."""

    return require_native_quarter_hour_price_truth(frame, role="epex_de")


def run_long_term_phase(
    project_root: str,
    inputs: LoadedInputs,
    peak_source_policy: str,
    use_seasonal_hourly_shape: bool,
    logger: logging.Logger,
) -> LongTermArtifacts:
    logger.info("=" * 70)
    logger.info("STEP 4: Fitting ShapeHourly on CH EPEX (full history)")
    logger.info("=" * 70)
    t2 = time.time()

    if inputs.sh_mode == "mlp":
        from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP

        sh = ShapeHourlyMLP()
        logger.info("  Using ShapeHourlyMLP (neural)")
    else:
        from pfc_shaping.lt.model.shape_hourly import ShapeHourly

        sh = ShapeHourly(use_seasonal_hourly=use_seasonal_hourly_shape)
        logger.info("  Using ShapeHourly (table)")

    sh_fit_kwargs = {"hydro_df": inputs.hydro}
    if inputs.sh_mode == "mlp":
        # Outage features are input data, not model-local state. Passing the
        # governed frame explicitly prevents a hidden repo-cache fallback.
        sh_fit_kwargs["outages_df"] = inputs.outages_all
    sh.fit(inputs.epex_ch, inputs.cal_ch, **sh_fit_kwargs)

    if inputs.sh_mode == "mlp":
        logger.info("  MLP fitted (neural shape function)")
    else:
        logger.info("  Fitted %d (saison, type_jour) cells", len(sh.factors_))
        logger.info("  f_W ratios: %s", {k: round(v, 4) for k, v in sh.f_W_.items()})
        logger.info("  Sample Hiver/Ouvrable peak h=12: f_H=%.4f", sh.get("Hiver", "Ouvrable")[12])
    logger.info("  ShapeHourly fitted in %.1fs", time.time() - t2)

    logger.info("=" * 70)
    logger.info("STEP 5: Fitting ShapeIntraday on DE-LU post Oct 2025")
    logger.info("=" * 70)
    t3 = time.time()
    from pfc_shaping.lt.model.shape_intraday import ShapeIntraday

    _require_production_intraday_price_truth(inputs.epex_de)
    cutoff_de = pd.Timestamp("2025-10-01", tz="UTC")
    epex_de_post = inputs.epex_de[inputs.epex_de.index >= cutoff_de]
    cal_de_post = inputs.cal_de.loc[epex_de_post.index]
    entso_de_post = inputs.entso.reindex(epex_de_post.index)

    logger.info("  DE-LU post Oct 2025: %d rows", len(epex_de_post))

    si = ShapeIntraday()
    si.fit(epex_de_post, entso_de_post, cal_de_post)

    logger.info("  Fitted %d (saison, type_jour, heure) cells", len(si.base_factors_))
    logger.info("  Corrections (layer 2): %d cells", len(si.corrections_))
    logger.info("  ShapeIntraday fitted in %.1fs", time.time() - t3)

    logger.info("=" * 70)
    logger.info("STEP 6: Fitting WaterValue correction")
    logger.info("=" * 70)
    t4 = time.time()
    from pfc_shaping.lt.model.water_value import WaterValueCorrection

    wv = WaterValueCorrection()
    wv.fit(inputs.epex_ch, inputs.hydro, inputs.cal_ch)

    logger.info("  beta_WV = %.4f", wv.beta_wv_)
    logger.info(
        "  Season sensitivities: %s", {k: f"{v:.3f}" for k, v in wv.season_sensitivity_.items()}
    )
    logger.info("  Calibration obs: %d", wv.n_obs_)
    logger.info("  WaterValue fitted in %.1fs", time.time() - t4)

    logger.info("=" * 70)
    logger.info("STEP 7: Probabilistic output disabled (deterministic-only governance)")
    logger.info("=" * 70)
    # The legacy interval model is not a CH rolling-origin forecast-error model.
    # Keep production deterministic until candidate-bound calibration evidence exists.
    unc = None

    logger.info("=" * 70)
    logger.info("STEP 8: Building base_prices (EEX forward levels)")
    logger.info("=" * 70)
    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.data.forward_proxy import (
        ForwardSourceUnavailableError,
        load_forward_snapshot,
        validate_forward_snapshot,
    )
    from pfc_shaping.pipeline.monthly_curve_authority import (
        delivery_months_from_prices,
        latest_base_prices_by_market,
        monthly_solver_enabled,
        monthly_solver_settings,
        solve_monthly_level_authority,
    )

    solver_level_authority_enabled = monthly_solver_enabled(inputs.config, market="CH")
    history: pd.DataFrame | None = None
    history_sha256: str | None = None
    vintage_evidence: dict[str, object] | None = None
    settings: dict[str, object] | None = None
    if solver_level_authority_enabled:
        settings = monthly_solver_settings(inputs.config)
        history_receipt = inputs.input_source_receipts.get("eex_forwards_history")
        if inputs.eex_forwards_history is None or history_receipt is None:
            raise ValueError(
                "CH monthly solver requires the consumed governed EEX forward history receipt"
            )
        if inputs.eex_historical_vintage_catalog_path is None:
            raise ValueError("CH monthly solver requires a signed EEX vintage catalog")
        forward_valuation_timestamp_ch = _forward_valuation_timestamp(inputs, None)
        (
            history,
            history_sha256,
            vintage_evidence,
            forward_snapshot_ch,
        ) = _prepare_governed_forward_history(
            inputs.eex_forwards_history,
            history_receipt,
            catalog_path=inputs.eex_historical_vintage_catalog_path,
            reference_timestamp=forward_valuation_timestamp_ch,
        )
    else:
        forward_snapshot_ch = load_forward_snapshot(
            inputs.epex_ch,
            eex_report_path=inputs.eex_report_path,
            config=inputs.config,
            market="CH",
            allow_spot_proxy=False,
            exact_source_only=inputs.eex_report_available_at is not None,
            source_available_at=inputs.eex_report_available_at,
        )
        forward_valuation_timestamp_ch = _forward_valuation_timestamp(
            inputs, forward_snapshot_ch
        )
    freshness_policy = dict((inputs.config.get("quality", {}) or {}).get("freshness", {}) or {})
    max_forward_age = int(freshness_policy.get("eex_max_age_business_days", 1))
    forward_snapshot_report = validate_forward_snapshot(
        forward_snapshot_ch,
        reference_timestamp=forward_valuation_timestamp_ch,
        max_age_business_days=max_forward_age,
    )
    base_prices_ch = dict(forward_snapshot_ch.prices)
    fwd_source_ch = forward_snapshot_ch.source_description
    logger.info("  Forward source: %s", fwd_source_ch)
    logger.info(
        "  Forward snapshot gate: PASS (snapshot=%s, business_age=%s, sha256=%s)",
        forward_snapshot_ch.snapshot_date.date()
        if forward_snapshot_ch.snapshot_date is not None
        else None,
        forward_snapshot_report["business_age_days"],
        forward_snapshot_ch.source_sha256,
    )

    cascader_ch = ContractCascader()
    cascader_ch.fit_seasonal_ratios(inputs.epex_ch)
    # Phase 5 D-A4-2 migration (NEG-04): fit_peak_spreads calibrates spreads
    # in €/MWh (sign-invariant for negative forwards). The deprecation shim
    # still routes fit_peak_ratios() → fit_peak_spreads() but explicit migration
    # avoids the runtime DeprecationWarning.
    # Phase 5 D-A2-1 default (negative-ready): no explicit enforce_* kwargs.
    # Legacy rollback per D-A2-3: pass allow_negative_peak=False at ContractCascader construction.
    cascader_ch.fit_peak_spreads(inputs.epex_ch)
    cascaded_prices_ch = cascader_ch.cascade(base_prices_ch)
    cascaded_prices_ch = cascader_ch.synthesize_peak_prices(cascaded_prices_ch)
    quoted_keys_ch = set(base_prices_ch.keys())
    cascader_for_ch_branch: object | None = cascader_ch
    monthly_constraint_tolerance_ch = 1e-9

    monthly_authority_ch = None
    if solver_level_authority_enabled:
        assert settings is not None
        assert history is not None
        assert history_sha256 is not None
        assert vintage_evidence is not None
        forwards_cfg = inputs.config.get("forwards", {}) or {}
        solver_as_of_date = forwards_cfg.get("eex_as_of_date")
        monthly_constraint_tolerance_ch = float(settings.get("constraint_tolerance", 1e-9))
        history_source_summary = list(history.attrs.get("eex_source_summary", []))
        source_hashes = {
            "eex_forwards_history": history_sha256,
            "eex_forwards_history_source_summary": _json_sha256(history_source_summary),
            "eex_historical_vintage_catalog": str(vintage_evidence["catalog_sha256"]),
        }
        if forward_snapshot_ch.source_sha256:
            source_hashes["forward_snapshot_source"] = forward_snapshot_ch.source_sha256
        neighbor_prices: dict[str, dict[str, float]] = {}
        neighbor_as_of_date = solver_as_of_date
        if solver_as_of_date:
            raise ForwardSourceUnavailableError(
                "forwards.eex_as_of_date is research-only until EEX history carries "
                "immutable available_at/revision receipts; it cannot drive production hard levels"
            )
        neighbor_as_of_date = forward_snapshot_ch.snapshot_date
        for neighbor in settings.get("markets", ("DE", "FR", "AT", "IT")):
            try:
                _, prices = latest_base_prices_by_market(
                    history,
                    market=str(neighbor).upper(),
                    as_of_date=neighbor_as_of_date,
                )
            except ValueError:
                continue
            neighbor_prices[str(neighbor).upper()] = prices
        if forward_snapshot_ch.source_sha256:
            source_hashes["forward_snapshot_source"] = forward_snapshot_ch.source_sha256
        monthly_authority_ch = solve_monthly_level_authority(
            market="CH",
            delivery_months=delivery_months_from_prices(base_prices_ch),
            own_base_prices=base_prices_ch,
            all_market_base_prices=neighbor_prices,
            eex_history=history,
            run_timestamp=forward_valuation_timestamp_ch,
            settings=settings,
            timezone="Europe/Zurich",
            source_hashes=source_hashes,
            original_forward_prices=base_prices_ch,
            own_forward_snapshot=forward_snapshot_ch,
            forward_eligibility=forward_snapshot_report,
            expected_forward_max_age_business_days=int(
                freshness_policy.get("eex_max_age_business_days", 1)
            ),
        )
        monthly_authority_ch.manifest["eex_forwards_history_source_summary"] = (
            history_source_summary
        )
        monthly_authority_ch.manifest["verified_vintage_catalog"] = vintage_evidence
        cascaded_prices_ch = monthly_authority_ch.assembler_base_prices
        quoted_keys_ch = monthly_authority_ch.quoted_keys
        cascader_for_ch_branch = None
        logger.info(
            "Monthly curve solver enabled for CH: monthly_solution_hash=%s active_constraints_hash=%s",
            monthly_authority_ch.monthly_solution_hash,
            monthly_authority_ch.active_constraints_hash,
        )

    logger.info("  Input keys: %d", len(base_prices_ch))
    logger.info("  Cascaded keys: %d", len(cascaded_prices_ch))
    for key in sorted(cascaded_prices_ch.keys()):
        logger.info("    %s: %.2f EUR/MWh", key, cascaded_prices_ch[key])

    latest_fill_dev = inputs.hydro["fill_deviation"].iloc[-1]
    logger.info(
        "  Latest hydro fill_deviation: %.3f (as of %s)",
        latest_fill_dev,
        inputs.hydro.index[-1].date(),
    )

    delivery_index: pd.DatetimeIndex | None = None
    if monthly_authority_ch is not None:
        delivery_months = monthly_authority_ch.inputs.delivery_grid.months
        delivery_index = _solver_delivery_quarter_hour_grid(
            delivery_months,
            timezone="Europe/Zurich",
        )
        future_start = delivery_index[0]
        future_end = delivery_index[-1] + pd.Timedelta(minutes=15)
        start_date = future_start.strftime("%Y-%m-%d")
        horizon_days = int(np.ceil((future_end - future_start) / pd.Timedelta(days=1)))
        horizon_label = f"solver delivery grid {delivery_months.min()} -> {delivery_months.max()}"
    else:
        start_date = _next_delivery_start_date(
            forward_valuation_timestamp_ch,
            timezone="Europe/Zurich",
        )
        max_fwd_year = max(
            int(k[:4]) for k in cascaded_prices_ch.keys() if k[:4].isdigit() and len(k) >= 4
        )
        end_of_last_year = pd.Timestamp(f"{max_fwd_year}-12-31", tz="UTC")
        future_start = pd.Timestamp(start_date, tz="UTC")
        future_end = future_start + pd.Timedelta(days=(end_of_last_year - future_start).days + 1)
        horizon_days = int((future_end - future_start) / pd.Timedelta(days=1))
        horizon_label = f"legacy horizon through 31/12/{max_fwd_year}"
    logger.info(
        "  Horizon: %s (%s) = %d days (%.1f years)",
        future_start,
        horizon_label,
        horizon_days,
        horizon_days / 365,
    )
    hydro_idx = _future_hydro_civil_weekly_index(future_start, future_end)
    decay = np.linspace(latest_fill_dev, 0.0, len(hydro_idx))
    hydro_forecast = pd.DataFrame({"fill_deviation": decay}, index=hydro_idx)

    logger.info("  Building ENTSO-E climatology forecast for N+3 horizon...")
    future_idx = (
        delivery_index
        if delivery_index is not None
        else pd.date_range(
            future_start,
            future_end,
            freq="15min",
            inclusive="left",
            tz="UTC",
        )
    )
    future_zurich = future_idx.tz_convert("Europe/Zurich")

    entso_zurich = inputs.entso.copy()
    entso_zurich["month"] = inputs.entso.index.tz_convert("Europe/Zurich").month
    entso_zurich["hour"] = inputs.entso.index.tz_convert("Europe/Zurich").hour
    entso_zurich["qh"] = (inputs.entso.index.minute // 15) + 1

    agg_dict = {
        "solar_regime_median": ("solar_regime", "median"),
        "load_deviation_median": ("load_deviation", "median"),
    }
    if "flow_deviation" in entso_zurich.columns:
        agg_dict["flow_deviation_median"] = ("flow_deviation", "median")

    clim = entso_zurich.groupby(["month", "hour", "qh"]).agg(**agg_dict).reset_index()
    future_keys = pd.DataFrame(
        {
            "month": future_zurich.month,
            "hour": future_zurich.hour,
            "qh": (future_zurich.minute // 15) + 1,
        },
        index=future_idx,
    )
    entso_forecast = future_keys.merge(clim, on=["month", "hour", "qh"], how="left").set_index(
        future_idx
    )

    rename_map = {
        "solar_regime_median": "solar_regime",
        "load_deviation_median": "load_deviation",
    }
    keep_cols = ["solar_regime", "load_deviation"]
    if "flow_deviation_median" in entso_forecast.columns:
        rename_map["flow_deviation_median"] = "flow_deviation"
        keep_cols.append("flow_deviation")
    entso_forecast = entso_forecast.rename(columns=rename_map)[keep_cols]
    entso_forecast["solar_regime"] = entso_forecast["solar_regime"].fillna(1.0)
    entso_forecast["load_deviation"] = entso_forecast["load_deviation"].fillna(0.0)
    if "flow_deviation" in entso_forecast.columns:
        entso_forecast["flow_deviation"] = entso_forecast["flow_deviation"].fillna(0.0)

    logger.info("  ENTSO-E climatology forecast: %d rows", len(entso_forecast))
    logger.info(
        "  solar_regime: mean=%.2f  std=%.2f",
        entso_forecast["solar_regime"].mean(),
        entso_forecast["solar_regime"].std(),
    )
    logger.info(
        "  load_deviation: mean=%.3f  std=%.3f",
        entso_forecast["load_deviation"].mean(),
        entso_forecast["load_deviation"].std(),
    )

    outages_forecast = None
    if inputs.outages_all is not None:
        outages_forecast = inputs.outages_all[inputs.outages_all.index >= future_start]
        logger.info(
            "  Outages forecast: %d rows, max unavail=%.0f MW",
            len(outages_forecast),
            outages_forecast["unavailable_mw"].max() if len(outages_forecast) > 0 else 0,
        )

    out_dir = os.path.join("pfc_shaping", "output")
    artifacts_dir = os.path.join("pfc_shaping", "model", "artifacts")
    output_reference = pd.Timestamp(inputs.reference_timestamp)
    if output_reference.tzinfo is None:
        raise ValueError("LT output reference_timestamp must be timezone-aware")
    today = output_reference.tz_convert("Europe/Zurich").strftime("%Y-%m-%d")
    out_base_ch = os.path.join(out_dir, f"pfc_15min_{today}")
    out_base_de = os.path.join(out_dir, f"pfc_de_15min_{today}")

    shared = _build_shared_long_term_artifacts(
        si=si,
        unc=unc,
        entso_forecast=entso_forecast,
        start_date=start_date,
        horizon_days=horizon_days,
        delivery_index=delivery_index,
        logger=logger,
    )

    # Generic per-market builds. Adding FR / AT / IT later only requires
    # appending one more MarketSpec to this list (Phase 3).
    swiss_spec = MarketSpec(
        code="CH",
        sheet="CH",
        tz="Europe/Zurich",
        country="CH",
        epex_df=inputs.epex_ch,
        cal_df=inputs.cal_ch,
        pre_fitted_sh=sh,
        water_value=wv,
        hydro_forecast=hydro_forecast,
        outages_forecast=outages_forecast,
        out_base=out_base_ch,
    )
    swiss = _build_long_term_branch(
        spec=swiss_spec,
        inputs=inputs,
        shared=shared,
        peak_source_policy=peak_source_policy,
        use_seasonal_hourly_shape=use_seasonal_hourly_shape,
        logger=logger,
        # CH reuses the up-front computed forwards / cascader to avoid
        # parsing the EEX XLSX twice on the same run.
        pre_loaded_base_prices=base_prices_ch,
        pre_loaded_fwd_source=fwd_source_ch,
        pre_loaded_cascaded_prices=cascaded_prices_ch,
        pre_loaded_cascader=cascader_for_ch_branch,
        pre_loaded_quoted_keys=quoted_keys_ch,
        pre_loaded_forward_snapshot=forward_snapshot_ch,
        pre_loaded_forward_eligibility=forward_snapshot_report,
        monthly_level_authority="solver" if monthly_authority_ch is not None else "legacy",
        skip_legacy_level_cascade=monthly_authority_ch is not None,
        skip_legacy_base_smoothing=monthly_authority_ch is not None,
        monthly_constraint_tolerance=monthly_constraint_tolerance_ch,
    )

    german_spec = MarketSpec(
        code="DE",
        sheet="DE",
        tz="Europe/Berlin",
        country="DE",
        epex_df=inputs.epex_de,
        cal_df=inputs.cal_de,
        pre_fitted_sh=None,  # German ShapeHourly is fit inside the branch
        water_value=None,
        hydro_forecast=None,
        outages_forecast=None,
        out_base=out_base_de,
    )
    german = _build_long_term_branch(
        spec=german_spec,
        inputs=inputs,
        shared=shared,
        peak_source_policy=peak_source_policy,
        use_seasonal_hourly_shape=use_seasonal_hourly_shape,
        logger=logger,
    )

    markets: dict[str, MarketBranchArtifacts] = {"CH": swiss, "DE": german}
    monthly_curve_manifests: dict[str, dict[str, object]] = {}
    if monthly_authority_ch is not None:
        ch_manifest = dict(monthly_authority_ch.manifest)
        projection_report = dict(swiss.final_product_projection_report)
        ch_manifest["final_product_projection"] = projection_report
        ch_manifest["final_product_projection_hash"] = _json_sha256(projection_report)
        monthly_curve_manifests["CH"] = ch_manifest

    return LongTermArtifacts(
        shared=shared,
        markets=markets,
        out_dir=out_dir,
        artifacts_dir=artifacts_dir,
        today=today,
        monthly_curve_manifests=monthly_curve_manifests,
    )


def _build_shared_long_term_artifacts(
    si: object,
    unc: object,
    entso_forecast: pd.DataFrame,
    start_date: str,
    horizon_days: int,
    delivery_index: pd.DatetimeIndex | None,
    logger: logging.Logger,
) -> SharedStructuralArtifacts:
    logger.info("=" * 70)
    logger.info("STEP 9a: Finalizing shared LT structural context")
    logger.info("=" * 70)

    calibrator = None
    try:
        from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator

        calibrator = ArbitrageFreeCalibrator(smoothness_weight=1.0, tol=0.01)
        logger.info("  ArbitrageFreeCalibrator loaded OK")
    except Exception as exc:
        logger.warning("  ArbitrageFreeCalibrator unavailable: %s", exc)

    return SharedStructuralArtifacts(
        si=si,
        unc=unc,
        calibrator=calibrator,
        entso_forecast=entso_forecast,
        start_date=start_date,
        horizon_days=horizon_days,
        delivery_index=delivery_index,
    )


def _build_long_term_branch(
    spec: MarketSpec,
    inputs: LoadedInputs,
    shared: SharedStructuralArtifacts,
    peak_source_policy: str,
    use_seasonal_hourly_shape: bool,
    logger: logging.Logger,
    *,
    pre_loaded_base_prices: dict | None = None,
    pre_loaded_fwd_source: str | None = None,
    pre_loaded_cascaded_prices: dict | None = None,
    pre_loaded_cascader: object | None = None,
    pre_loaded_quoted_keys: set[str] | None = None,
    pre_loaded_forward_snapshot: object | None = None,
    pre_loaded_forward_eligibility: object | None = None,
    monthly_level_authority: str = "legacy",
    skip_legacy_level_cascade: bool = False,
    skip_legacy_base_smoothing: bool = False,
    monthly_constraint_tolerance: float = 1e-9,
) -> MarketBranchArtifacts:
    """Build one market's PFC from a MarketSpec.

    Centralises the previous Swiss / German branch logic so each new
    market only requires a MarketSpec and the up-front data wiring
    (EPEX history, calendar). The Swiss branch is the only one that
    reuses pre-fitted ShapeHourly / cascader / forward dict from
    ``run_long_term_phase`` to avoid recomputing them; other markets
    fit inside this function.

    Args:
        spec: per-market description.
        inputs: shared LoadedInputs (config, eex_report_path, etc.).
        shared: shared structural artifacts (ShapeIntraday, Uncertainty,
            ENTSO forecast, calibrator, horizon).
        peak_source_policy: see PFCAssembler.
        logger: pipeline logger.
        pre_loaded_*: optional precomputed forward dict / cascader /
            ShapeHourly. Used by the CH branch which builds them once
            in ``run_long_term_phase`` and reuses them; for other
            markets these are ``None`` and the function fits internally.

    Returns:
        MarketBranchArtifacts with the assembled 15-min PFC.
    """
    logger.info("=" * 70)
    logger.info("STEP 9.%s: Building %s PFC (LT branch, tz=%s)", spec.code, spec.code, spec.tz)
    logger.info("=" * 70)
    t0 = time.time()

    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.data.forward_proxy import (
        ForwardEligibility,
        ForwardSnapshot,
        load_forward_snapshot,
        validate_forward_snapshot,
        verify_forward_eligibility,
    )
    from pfc_shaping.lt.model.assembler import PFCAssembler

    # ── 1. ShapeHourly: reuse pre-fit if provided, else fit on the spot ──
    if spec.pre_fitted_sh is not None:
        sh = spec.pre_fitted_sh
        logger.info("  %s ShapeHourly: reusing pre-fitted instance", spec.code)
    else:
        if inputs.sh_mode == "mlp":
            from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP

            sh = ShapeHourlyMLP()
        else:
            from pfc_shaping.lt.model.shape_hourly import ShapeHourly

            sh = ShapeHourly(use_seasonal_hourly=use_seasonal_hourly_shape)
        sh.fit(spec.epex_df, spec.cal_df)
        logger.info("  %s ShapeHourly fitted (%s mode)", spec.code, inputs.sh_mode)

    # ── 2. Forwards (base_prices) ────────────────────────────────────────
    if pre_loaded_base_prices is not None:
        base_prices = dict(pre_loaded_base_prices)
        fwd_source = pre_loaded_fwd_source or "pre-loaded"
        forward_snapshot = pre_loaded_forward_snapshot
        if monthly_level_authority == "solver":
            if not isinstance(forward_snapshot, ForwardSnapshot):
                raise ValueError(
                    "solver-authority preloaded prices require a verified ForwardSnapshot"
                )
            if not isinstance(pre_loaded_forward_eligibility, ForwardEligibility):
                raise ValueError(
                    "solver-authority preloaded prices require ForwardEligibility evidence"
                )
            verify_forward_eligibility(
                forward_snapshot,
                pre_loaded_forward_eligibility,
                valuation_timestamp=_forward_valuation_timestamp(inputs, forward_snapshot),
                expected_max_age_business_days=int(
                    dict((inputs.config.get("quality", {}) or {}).get("freshness", {}) or {}).get(
                        "eex_max_age_business_days", 1
                    )
                ),
            )
            if base_prices != dict(forward_snapshot.prices):
                raise ValueError(
                    "solver-authority preloaded prices must exactly match ForwardSnapshot prices"
                )
    else:
        forward_snapshot = load_forward_snapshot(
            spec.epex_df,
            eex_report_path=inputs.eex_report_path,
            config=inputs.config,
            market=spec.sheet,
            allow_spot_proxy=False,
            exact_source_only=inputs.eex_report_available_at is not None,
            source_available_at=inputs.eex_report_available_at,
        )
        freshness_policy = dict((inputs.config.get("quality", {}) or {}).get("freshness", {}) or {})
        validate_forward_snapshot(
            forward_snapshot,
            reference_timestamp=_forward_valuation_timestamp(inputs, forward_snapshot),
            max_age_business_days=int(freshness_policy.get("eex_max_age_business_days", 1)),
        )
        base_prices = dict(forward_snapshot.prices)
        fwd_source = forward_snapshot.source_description
    logger.info("  %s forward source: %s", spec.code, fwd_source)

    # ── 3. Cascading ────────────────────────────────────────────────────
    if pre_loaded_cascader is not None and pre_loaded_cascaded_prices is not None:
        cascader = pre_loaded_cascader
        cascaded_prices = pre_loaded_cascaded_prices
    elif skip_legacy_level_cascade and pre_loaded_cascaded_prices is not None:
        cascader = None
        cascaded_prices = pre_loaded_cascaded_prices
    else:
        # Phase 5 D-A2-1 default (negative-ready): no explicit enforce_* kwargs.
        # Legacy rollback per D-A2-3: pass allow_negative_peak=False at ContractCascader construction.
        cascader = ContractCascader(tz=spec.tz)
        cascader.fit_seasonal_ratios(spec.epex_df)
        # Phase 5 D-A4-2 migration (NEG-04) — see comment at the analogous callsite above.
        cascader.fit_peak_spreads(spec.epex_df)
        cascaded_prices = cascader.cascade(base_prices)
        cascaded_prices = cascader.synthesize_peak_prices(cascaded_prices)

    logger.info("  %s cascaded keys: %d", spec.code, len(cascaded_prices))
    for key in sorted(cascaded_prices.keys()):
        logger.info("    %s %s: %.2f EUR/MWh", spec.code, key, cascaded_prices[key])

    # ── 4. Assemble ─────────────────────────────────────────────────────
    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=shared.si,
        uncertainty=None,
        water_value=spec.water_value,
        cascader=cascader,
        calibrator=shared.calibrator,
        peak_source_policy=peak_source_policy,
        monthly_level_authority=monthly_level_authority,
        skip_legacy_level_cascade=skip_legacy_level_cascade,
        skip_legacy_base_smoothing=skip_legacy_base_smoothing,
        monthly_constraint_tolerance=monthly_constraint_tolerance,
    )
    reference_date = pd.Timestamp(inputs.reference_timestamp)
    if reference_date.tzinfo is None:
        raise ValueError("LT branch reference_timestamp must be timezone-aware")
    build_kwargs = dict(
        base_prices=cascaded_prices,
        quoted_keys=set(pre_loaded_quoted_keys)
        if pre_loaded_quoted_keys is not None
        else set(base_prices.keys()),
        start_date=shared.start_date,
        horizon_days=shared.horizon_days,
        entso_forecast=shared.entso_forecast,
        hydro_forecast=spec.hydro_forecast,
        reference_date=reference_date.tz_convert("UTC"),
    )
    if monthly_level_authority == "solver":
        if shared.delivery_index is None:
            raise ValueError("solver-authority branch requires the governed delivery index")
        build_kwargs["delivery_index"] = shared.delivery_index
    # Only the Swiss branch (with country='CH' default) consumes outages today.
    # Other markets pass country=spec.country explicitly so the assembler
    # picks the right tz / holidays.
    if spec.outages_forecast is not None:
        build_kwargs["outages_forecast"] = spec.outages_forecast
    if spec.country != "CH":
        build_kwargs["country"] = spec.country

    pfc = assembler.build(**build_kwargs)
    logger.info("  %s PFC assembled: %d rows in %.1fs", spec.code, len(pfc), time.time() - t0)

    return MarketBranchArtifacts(
        code=spec.code,
        pfc=pfc,
        sh=sh,
        base_prices=base_prices,
        cascaded_prices=cascaded_prices,
        fwd_source=fwd_source,
        out_base=spec.out_base,
        forward_snapshot=(
            forward_snapshot.to_manifest()
            if forward_snapshot is not None and hasattr(forward_snapshot, "to_manifest")
            else None
        ),
        final_product_projection_report=dict(
            getattr(assembler, "final_product_projection_report_", {}) or {}
        ),
        wv=spec.water_value,
        hydro_forecast=spec.hydro_forecast,
        outages_forecast=spec.outages_forecast,
    )


# Per-market suffix used when writing artifacts. CH keeps the legacy
def _save_market_artifacts(
    art: MarketBranchArtifacts,
    artifacts_dir: str,
    logger: logging.Logger,
) -> None:
    del art, artifacts_dir, logger
    raise RuntimeError(
        "Direct LT artifact publication is disabled. Serialize an immutable candidate and "
        "use the governed REGISTER/AUDIT/PROMOTE workflow."
    )


def _save_monthly_curve_manifests(
    manifests: dict[str, dict[str, object]],
    artifacts_dir: str,
    logger: logging.Logger,
) -> None:
    for market, manifest in sorted((manifests or {}).items()):
        suffix = "" if str(market).upper() == "CH" else f"_{str(market).lower()}"
        path = os.path.join(artifacts_dir, f"production_monthly_curve_manifest{suffix}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        logger.info("  Saved monthly curve manifest: %s", path)


def save_long_term_outputs(long_term: LongTermArtifacts, logger: logging.Logger) -> None:
    del long_term, logger
    raise RuntimeError(
        "Direct LT output publication is disabled. Serialize an immutable candidate and "
        "use the governed REGISTER/AUDIT/PROMOTE workflow."
    )


def print_pipeline_summary(long_term: LongTermArtifacts, total_time: float) -> None:
    pfc = long_term.swiss.pfc
    pfc_de = long_term.german.pfc
    idx_zurich = pfc.index.tz_convert("Europe/Zurich")

    print("\n" + "=" * 80)
    print(f"  PFC 15min PRODUCTION RUN - {long_term.today}")
    print(
        "  Horizon: %s -> %s (%d days)"
        % (pfc.index.min().date(), pfc.index.max().date(), long_term.shared.horizon_days)
    )
    print("  Timestamps: %d (15min intervals)" % len(pfc))
    print("=" * 80)

    print("\n--- PRICE DISTRIBUTION (EUR/MWh) ---")
    print("  Mean:    %.2f" % pfc["price_shape"].mean())
    print("  Median:  %.2f" % pfc["price_shape"].median())
    print("  Std:     %.2f" % pfc["price_shape"].std())
    print("  Min:     %.2f" % pfc["price_shape"].min())
    print("  Max:     %.2f" % pfc["price_shape"].max())
    print("  p5:      %.2f" % pfc["price_shape"].quantile(0.05))
    print("  p95:     %.2f" % pfc["price_shape"].quantile(0.95))

    print("\n--- FACTOR RANGES ---")
    for col in ["f_S", "f_W", "f_H", "f_Q", "f_WV"]:
        series = pfc[col]
        print(
            "  %-5s: mean=%.4f  min=%.4f  max=%.4f  std=%.4f"
            % (col, series.mean(), series.min(), series.max(), series.std())
        )

    print("\n--- f_Q INTRADAY DETAIL ---")
    print("  f_Q range: [%.4f, %.4f]" % (pfc["f_Q"].min(), pfc["f_Q"].max()))
    print("  f_Q mean:  %.4f (should be ~1.0)" % pfc["f_Q"].mean())
    pfc_zurich = pfc.copy()
    pfc_zurich["hour"] = pfc.index.tz_convert("Europe/Zurich").hour
    for hour in [0, 6, 8, 12, 17, 20, 23]:
        mask = pfc_zurich["hour"] == hour
        print(
            "    h=%02d: f_Q mean=%.4f  std=%.4f"
            % (hour, pfc_zurich.loc[mask, "f_Q"].mean(), pfc_zurich.loc[mask, "f_Q"].std())
        )

    print("\n--- CONFIDENCE INTERVAL WIDTH BY HORIZON ---")
    if pfc["p10"].notna().any():
        pfc_zurich["ic_width"] = pfc["p90"] - pfc["p10"]
        for profile_type in ["M+1..M+6", "M+7..M+12", "Y+2/Y+3"]:
            mask = pfc["profile_type"] == profile_type
            if mask.any():
                width = pfc_zurich.loc[mask, "ic_width"]
                price = pfc.loc[mask, "price_shape"]
                rel_width = (width / price.abs().clip(lower=1.0)).mean() * 100
                print(
                    "  %-12s: mean_width=%.2f EUR/MWh  mean_price=%.2f  rel_width=%.1f%%"
                    % (profile_type, width.mean(), price.mean(), rel_width)
                )
    else:
        print("  (No IC computed)")

    print("\n--- ENERGY CONSISTENCY (mean PFC vs forward) ---")
    for key in sorted(long_term.swiss.cascaded_prices_ch.keys()):
        base = long_term.swiss.cascaded_prices_ch[key]
        if len(key) == 4 and key.isdigit():
            mask = idx_zurich.year == int(key)
            label = f"Cal {key}"
        elif key.endswith("-Peak"):
            continue
        elif "Q" in key:
            year = int(key[:4])
            quarter = int(key.split("Q")[1][0])
            q_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}[quarter]
            mask = (idx_zurich.year == year) & (idx_zurich.month.isin(q_months))
            label = f"  {key}"
        elif len(key) == 7 and key[4] == "-":
            year = int(key[:4])
            month = int(key[5:])
            mask = (idx_zurich.year == year) & (idx_zurich.month == month)
            label = f"    {key}"
        else:
            continue

        n_pts = mask.sum()
        if n_pts == 0:
            continue
        mean_pfc = pfc.loc[mask, "price_shape"].mean()
        dev_pct = (mean_pfc - base) / abs(base) * 100
        marker = "OK" if abs(dev_pct) < 5.0 else "WARN"
        print(
            "  %-14s: fwd=%.2f  pfc=%.2f  dev=%+.2f%%  n=%d  [%s]"
            % (label, base, mean_pfc, dev_pct, n_pts, marker)
        )

    print("\n--- CALIBRATION STATUS ---")
    if pfc["calibrated"].any():
        print("  ArbitrageFree calibration: APPLIED")
    else:
        print("  ArbitrageFree calibration: NOT APPLIED (raw shape only)")

    print("\n--- ANNUAL AVERAGES ---")
    for year in sorted(idx_zurich.year.unique()):
        mask = idx_zurich.year == year
        if mask.sum() > 0:
            price = pfc.loc[mask, "price_shape"]
            print(
                "  %d: mean=%.2f  min=%.2f  max=%.2f  n=%d"
                % (year, price.mean(), price.min(), price.max(), mask.sum())
            )

    print("\n--- PROFILE TYPE DISTRIBUTION ---")
    for profile_type, count in pfc["profile_type"].value_counts().items():
        pct = count / len(pfc) * 100
        print("  %-12s: %d rows (%.1f%%)" % (profile_type, count, pct))

    print("\n--- PEAK / OFF-PEAK ANALYSIS ---")
    hour = idx_zurich.hour
    dow = idx_zurich.dayofweek
    is_peak = (hour >= 8) & (hour < 20) & (dow < 5)
    for year in sorted(idx_zurich.year.unique()):
        yr_mask = idx_zurich.year == year
        if yr_mask.sum() == 0:
            continue
        peak = pfc.loc[yr_mask & is_peak, "price_shape"]
        offpeak = pfc.loc[yr_mask & ~is_peak, "price_shape"]
        if len(peak) > 0 and len(offpeak) > 0:
            spread = peak.mean() - offpeak.mean()
            ratio = peak.mean() / offpeak.mean() if offpeak.mean() != 0 else float("nan")
            print(
                "  %d: peak=%.2f  offpeak=%.2f  spread=%.2f  ratio=%.3f"
                % (year, peak.mean(), offpeak.mean(), spread, ratio)
            )

    print("\n" + "=" * 80)
    print("  DE PFC SUMMARY")
    print("=" * 80)
    print("  Timestamps: %d" % len(pfc_de))
    print("  Mean price: %.2f EUR/MWh" % pfc_de["price_shape"].mean())
    print("  Min: %.2f  Max: %.2f" % (pfc_de["price_shape"].min(), pfc_de["price_shape"].max()))

    idx_de_local = pfc_de.index.tz_convert("Europe/Berlin")
    print("\n--- DE ANNUAL AVERAGES ---")
    for year in sorted(idx_de_local.year.unique()):
        mask = idx_de_local.year == year
        if mask.sum() > 0:
            price = pfc_de.loc[mask, "price_shape"]
            print(
                "  %d: mean=%.2f  min=%.2f  max=%.2f  n=%d"
                % (year, price.mean(), price.min(), price.max(), mask.sum())
            )

    print("\n--- DE PEAK / OFF-PEAK ---")
    hour_de = idx_de_local.hour
    dow_de = idx_de_local.dayofweek
    is_peak_de = (hour_de >= 8) & (hour_de < 20) & (dow_de < 5)
    for year in sorted(idx_de_local.year.unique()):
        yr_mask = idx_de_local.year == year
        if yr_mask.sum() == 0:
            continue
        peak = pfc_de.loc[yr_mask & is_peak_de, "price_shape"]
        offpeak = pfc_de.loc[yr_mask & ~is_peak_de, "price_shape"]
        if len(peak) > 0 and len(offpeak) > 0:
            spread = peak.mean() - offpeak.mean()
            ratio = peak.mean() / offpeak.mean() if offpeak.mean() != 0 else float("nan")
            print(
                "  %d: peak=%.2f  offpeak=%.2f  spread=%.2f  ratio=%.3f"
                % (year, peak.mean(), offpeak.mean(), spread, ratio)
            )

    print("\n" + "=" * 80)
    print("  TOTAL EXECUTION TIME: %.1f seconds" % total_time)
    print("  Output CH: %s.parquet + .csv" % long_term.swiss.out_base)
    print("  Output DE: %s.parquet + .csv" % long_term.german.out_base)
    print("=" * 80)
