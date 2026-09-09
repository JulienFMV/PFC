"""Build a local-test CH LT PFC from the governed Phase 13 candidate inventory.

This runner is intentionally non-production. It requires the local-test
governance gate to pass, then builds one CH PFC per scenario and a weighted
structural fan chart. Production approval remains a separate human-signoff gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pfc_shaping.calibration.monthly_forward_curve import product_periods
from pfc_shaping.data.electrification_scenarios import (
    derive_hpfc_scenario_features,
    interpolate_electrification_scenario_years,
    validate_electrification_scenarios,
)
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.lt.model.electrification_shape import structural_fan_chart
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    delivery_months_for_local_window,
    delivery_months_for_window,
    monthly_solver_settings,
    solve_monthly_level_authority_from_history,
)
from pfc_shaping.validation.product_normalization import (
    build_product_normalization_gates,
)
from scripts.build_ep2050_multi_scenario_pfc import (
    _build_one_curve,
    _delivery_years,
    _df_to_md,
    _fit_components,
    _parse_csv,
    _parse_weights,
    _safe_slug,
)
from scripts.build_first_ep2050_pfc import _latest_forwards, _load_epex_hourly
from scripts.validate_scenario_governance import validate_governance, write_report

DEFAULT_INVENTORY = Path("data/electrification_scenarios_prod_candidate_neutralized_2030.parquet")
DEFAULT_MANIFEST = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml")
DEFAULT_GOVERNANCE_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-GOVERNANCE-GATE.md")
DEFAULT_SUMMARY = Path(".planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-CH-PFC.md")
DEFAULT_LOCAL_MONTHLY_CONSTRAINT_TOLERANCE = 1e-9
DEFAULT_LOCAL_FINAL_PRODUCT_REPLAY_TOLERANCE = 1e-9
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_INTRADAY_PROVENANCE_KEYS = {
    "schema_version",
    "status",
    "production_authorization",
    "promotion_gate",
    "scientific_admission",
    "training_eligible",
    "model_selection_eligible",
    "confirmatory_rolling_origin_eligible",
    "future_holdout_eligible",
    "t057_eligible",
    "candidate_assembly_eligible",
    "local_diagnostic_allowed",
    "market",
    "start_utc",
    "end_utc",
    "rows",
    "frequency_seconds",
    "resolution_provenance",
    "season_row_counts",
    "sources",
    "panel",
    "limitations",
}
_LEGACY_LOCAL_INTRADAY_PROVENANCE_KEYS = {
    "schema_version",
    "status",
    "production_authorization",
    "promotion_gate",
    "market",
    "start_utc",
    "end_utc",
    "rows",
    "frequency_seconds",
    "season_row_counts",
    "sources",
    "panel",
    "limitations",
}
_PINNED_LEGACY_INTRADAY_PROVENANCE_SHA256 = (
    "af8232e7045eb9aa685d946770a56a108e438a6a71e9cbdab47614abcc5a9e4f"
)
_PINNED_LEGACY_INTRADAY_PANEL_SHA256 = (
    "e5b0a6d6e3c6837ae1728f25ea25bdbee9278d8eefd414f4476669d7286268ab"
)


def _absolute_path(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return assert_absolute_path_has_no_links(selected)


def _output_path(path: str | Path, *, repo_root: Path) -> Path:
    selected = _absolute_path(path)
    try:
        selected.relative_to(repo_root / "build")
    except ValueError as exc:
        raise ValueError(
            f"local-test output must remain below canonical repo build/: {selected}"
        ) from exc
    return selected


def _input_path(path: str | Path, *, repo_root: Path) -> Path:
    selected = _absolute_path(path)
    try:
        selected.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"local-test input escapes the repository: {selected}") from exc
    return selected


def _safe_output_prefix(value: str) -> str:
    selected = str(value)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", selected) is None:
        raise ValueError("local-test output prefix contains unsafe path characters")
    if selected in {".", ".."}:
        raise ValueError("local-test output prefix is invalid")
    return selected


def _require_exact_monthly_constraint_tolerance(value: float) -> float:
    selected = float(value)
    if selected != DEFAULT_LOCAL_MONTHLY_CONSTRAINT_TOLERANCE:
        raise ValueError(
            "local LT runner monthly solver constraint tolerance must be exactly 1e-9"
        )
    return selected


def _delivery_index_to_local_end_exclusive(
    *,
    start_date: str | pd.Timestamp,
    local_end_exclusive: str | pd.Timestamp,
    market: str,
) -> pd.DatetimeIndex:
    timezone = "Europe/Zurich" if str(market).upper() == "CH" else "Europe/Berlin"
    start = pd.Timestamp(start_date)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(local_end_exclusive)
    end = end.tz_localize(timezone) if end.tzinfo is None else end.tz_convert(timezone)
    end_utc = end.tz_convert("UTC")
    if end_utc <= start:
        raise ValueError("delivery local end-exclusive must be after start-date")
    quarter = pd.Timedelta(minutes=15)
    if start.floor("15min") != start or end_utc.floor("15min") != end_utc:
        raise ValueError("delivery boundaries must lie on the exact 15-minute grid")
    index = pd.date_range(start, end_utc, freq="15min", inclusive="left")
    if index.empty or index[-1] + quarter != end_utc:
        raise ValueError("delivery window is not an exact end-exclusive 15-minute grid")
    return index


def _delivery_months_from_index(
    index: pd.DatetimeIndex, *, market: str
) -> pd.PeriodIndex:
    timezone = "Europe/Zurich" if str(market).upper() == "CH" else "Europe/Berlin"
    local = pd.DatetimeIndex(index).tz_convert(timezone)
    return pd.PeriodIndex(sorted(set(local.strftime("%Y-%m"))), freq="M")


def _forward_snapshot_from_payload(
    payload: bytes,
    *,
    market: str,
    forward_date: pd.Timestamp,
) -> pd.DataFrame:
    frame = pd.read_parquet(io.BytesIO(payload))
    snapshot_dates = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    selected_date = pd.Timestamp(forward_date).tz_localize(None).normalize()
    selected = frame[
        frame["market"].astype(str).str.upper().eq(str(market).upper())
        & snapshot_dates.eq(selected_date)
    ].copy()
    if selected.empty:
        raise ValueError("captured forward payload has no selected market/date snapshot")
    selected["product"] = selected["product"].astype(str)
    selected["load_type"] = selected["load_type"].astype(str).str.upper()
    selected["price"] = pd.to_numeric(selected["price"], errors="raise").astype(float)
    if selected.duplicated(["product", "load_type"]).any():
        raise ValueError("captured forward snapshot contains duplicate product/load_type rows")
    return selected.sort_values(["load_type", "product"]).reset_index(drop=True)


def _hourly_price_frame(pfc: pd.DataFrame, *, market: str) -> pd.DataFrame:
    timezone = "Europe/Zurich" if str(market).upper() == "CH" else "Europe/Berlin"
    index = pd.DatetimeIndex(pfc.index)
    hourly_price = pfc["price_shape"].astype(float).groupby(index.floor("h")).mean()
    hourly = pd.DataFrame(
        {
            "ts_utc": pd.DatetimeIndex(hourly_price.index),
            "price_shape": hourly_price.to_numpy(dtype=float),
        }
    )
    hourly["ts_ch"] = hourly["ts_utc"].dt.tz_convert(timezone)
    hourly["year"] = hourly["ts_ch"].dt.year.astype(int)
    hourly["month"] = hourly["ts_ch"].dt.month.astype(int)
    hourly["quarter"] = hourly["ts_ch"].dt.quarter.astype(int)
    return hourly


def _peak_prices_for_delivery(
    forward_snapshot: pd.DataFrame,
    *,
    delivery_months: pd.PeriodIndex,
) -> dict[str, float]:
    delivered = set(pd.PeriodIndex(delivery_months, freq="M"))
    peak_prices: dict[str, float] = {}
    peak_quotes = forward_snapshot.loc[
        forward_snapshot["load_type"].astype(str).str.upper().eq("PEAK")
    ]
    for row in peak_quotes.itertuples(index=False):
        product = str(row.product)
        product_months = set(product_periods(product))
        if product_months and product_months.issubset(delivered):
            peak_prices[f"{product}-Peak"] = float(row.price)
    return peak_prices


def _final_product_replay(
    pfc_by_scenario: dict[str, pd.DataFrame],
    *,
    forward_snapshot: pd.DataFrame,
    forward_date: pd.Timestamp,
    market: str,
    tolerance: float = DEFAULT_LOCAL_FINAL_PRODUCT_REPLAY_TOLERANCE,
) -> dict[str, object]:
    present_load_types = tuple(
        sorted(set(forward_snapshot["load_type"].astype(str).str.upper()))
    )
    if "BASE" not in present_load_types:
        raise ValueError("final product replay requires at least one BASE quote")
    scenario_results: dict[str, object] = {}
    for scenario, pfc in sorted(pfc_by_scenario.items()):
        gates = build_product_normalization_gates(
            _hourly_price_frame(pfc, market=market),
            forward_snapshot,
            forward_date=pd.Timestamp(forward_date).tz_localize(None).normalize(),
            market=market,
            price_column="price_shape",
            hard_tolerance=float(tolerance),
            peak_country=market,
            required_load_types=present_load_types,
        )
        in_scope = gates.loc[~gates["status"].eq("OUT_OF_SCOPE")].copy()
        direct = in_scope.loc[
            in_scope["gate_id"].isin(
                {
                    "hard_base_product_repricing",
                    "hard_peak_product_repricing",
                    "hard_offpeak_product_repricing",
                }
            )
        ]
        if direct.empty or not bool(direct["load_type"].eq("BASE").any()):
            raise ValueError(f"{scenario} final replay has no in-scope BASE product")
        blocking = in_scope.loc[in_scope["status"].isin({"CRITICAL", "UNSUPPORTED"})]
        if not blocking.empty:
            raise ValueError(
                f"{scenario} final product replay failed: "
                f"{blocking[['gate_id', 'load_type', 'product', 'status']].to_dict('records')}"
            )
        if not bool(direct["rows"].eq(direct["expected_rows"]).all()):
            raise ValueError(f"{scenario} final product replay contains partial coverage")
        if "PEAK" in present_load_types:
            peak_buckets = in_scope.loc[
                in_scope["gate_id"].eq("quote_aware_peak_bucket_repricing")
            ]
            if peak_buckets.empty or not bool(peak_buckets["status"].eq("PASS").all()):
                raise ValueError(f"{scenario} quote-aware PEAK buckets do not reprice")
        cal_2030 = direct.loc[
            direct["load_type"].eq("BASE") & direct["product"].eq("2030")
        ]
        if (
            len(cal_2030) != 1
            or int(cal_2030.iloc[0]["rows"]) != 8_760
            or int(cal_2030.iloc[0]["expected_rows"]) != 8_760
        ):
            raise ValueError(f"{scenario} final replay does not fully cover BASE Cal2030")
        pass_residuals = pd.to_numeric(
            in_scope.loc[in_scope["status"].eq("PASS"), "abs_residual_eur_mwh"],
            errors="coerce",
        )
        scenario_results[scenario] = {
            "status": (
                "PASS_WITH_SOURCE_QUOTE_CONFLICTS_PRODUCTION_NO_GO"
                if bool(in_scope["status"].eq("QUOTE_CONFLICT").any())
                else "PASS"
            ),
            "in_scope_gate_count": int(len(in_scope)),
            "out_of_scope_gate_count": int(gates["status"].eq("OUT_OF_SCOPE").sum()),
            "direct_product_count": int(len(direct)),
            "source_quote_conflict_count": int(
                in_scope["status"].eq("QUOTE_CONFLICT").sum()
            ),
            "maximum_supported_abs_residual_eur_mwh": float(pass_residuals.max()),
            "cal2030_base_rows": 8_760,
            "gates": json.loads(
                gates.to_json(
                    orient="records", date_format="iso", double_precision=15
                )
            ),
        }
    source_counts = {
        load_type: int(forward_snapshot["load_type"].eq(load_type).sum())
        for load_type in ("BASE", "PEAK", "OFFPEAK")
    }
    return {
        "schema_version": "local_test_ch_final_product_replay.v1",
        "status": "PASS_LOCAL_FINAL_PRODUCT_REPLAY_PRODUCTION_NO_GO",
        "production_authorization": False,
        "promotion_gate": False,
        "market": str(market).upper(),
        "forward_snapshot_date": pd.Timestamp(forward_date).date().isoformat(),
        "tolerance_eur_mwh": float(tolerance),
        "source_quote_counts": source_counts,
        "peak_offpeak_limit": (
            "NO_PEAK_OR_OFFPEAK_SOURCE_QUOTES_IN_SELECTED_TEST_FIXTURE"
            if source_counts["PEAK"] == 0 and source_counts["OFFPEAK"] == 0
            else "PRESENT_SOURCE_LOAD_TYPES_REPLAYED"
        ),
        "scenarios": scenario_results,
    }


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    assert_absolute_path_has_no_links(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_absolute_path_has_no_links(path.parent)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite local-test artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=index)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _write_json(path: Path, payload: dict[str, object], *, label: str) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    write_durable_exact_bytes(path, encoded, label=label)


def _write_governance_report(
    path: Path,
    *,
    manifest_path: Path,
    inventory_path: Path,
    vintage: str,
    issues: list[dict[str, object]],
    effective: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_report(
            temporary,
            manifest_path=manifest_path,
            inventory_path=inventory_path,
            vintage=vintage,
            mode="local-test",
            issues=issues,
            effective=effective,
        )
        payload = read_stable_single_link_file(
            temporary, label="temporary local-test governance report"
        )
    finally:
        temporary.unlink(missing_ok=True)
    write_durable_exact_bytes(path, payload, label="local-test governance report")


def _stable_file_record(
    path: str | Path,
    *,
    label: str,
    repo_root: Path,
) -> dict[str, object]:
    selected = _absolute_path(path)
    payload = read_stable_single_link_file(selected, label=label)
    return _file_record_from_payload(selected, payload, repo_root=repo_root)


def _file_record_from_payload(
    selected: Path,
    payload: bytes,
    *,
    repo_root: Path,
) -> dict[str, object]:
    try:
        display_path = selected.relative_to(repo_root).as_posix()
    except ValueError:
        display_path = str(selected)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _unique_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _admit_intraday_provenance(
    *,
    manifest_payload: bytes,
    intraday_path: Path,
    intraday_sha256: str,
) -> dict[str, object]:
    try:
        value = json.loads(
            manifest_payload.decode("utf-8"),
            object_pairs_hook=_unique_mapping,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("intraday provenance manifest is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("intraday provenance manifest must be a mapping")
    schema = value.get("schema_version")
    legacy_pinned = schema == "local_intraday_calibration_panel.v1"
    if legacy_pinned:
        if (
            hashlib.sha256(manifest_payload).hexdigest()
            != _PINNED_LEGACY_INTRADAY_PROVENANCE_SHA256
            or set(value) != _LEGACY_LOCAL_INTRADAY_PROVENANCE_KEYS
            or intraday_sha256 != _PINNED_LEGACY_INTRADAY_PANEL_SHA256
        ):
            raise ValueError(
                "legacy intraday provenance is not the exact pinned local diagnostic"
            )
    elif schema == "local_intraday_calibration_panel.v3":
        if set(value) != _LOCAL_INTRADAY_PROVENANCE_KEYS:
            raise ValueError("intraday provenance manifest fields are not exact")
    else:
        raise ValueError("intraday provenance manifest schema is unsupported")
    if (
        value.get("status") != "COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO"
        or value.get("production_authorization") is not False
        or value.get("promotion_gate") is not False
        or value.get("market") != "DE-LU_PROXY_FOR_CH_LOCAL_DIAGNOSTIC"
        or value.get("frequency_seconds") != 900
        or not isinstance(value.get("sources"), list)
        or not isinstance(value.get("limitations"), list)
    ):
        raise ValueError("intraday provenance manifest must remain non-promotional")
    if not legacy_pinned:
        if (
            value.get("scientific_admission") is not False
            or value.get("training_eligible") is not False
            or value.get("model_selection_eligible") is not False
            or value.get("confirmatory_rolling_origin_eligible") is not False
            or value.get("future_holdout_eligible") is not False
            or value.get("t057_eligible") is not False
            or value.get("candidate_assembly_eligible") is not False
            or value.get("local_diagnostic_allowed") is not True
        ):
            raise ValueError("intraday provenance must deny every scientific authority")
        resolution = value.get("resolution_provenance")
        if (
            not isinstance(resolution, dict)
            or set(resolution)
            != {
                "status",
                "native_quarter_hour_truth_eligible",
                "reason",
            }
            or resolution.get("status")
            != "MIXED_AUTHORITY_OUTPUT_GRID_NOT_NATIVE_TRUTH"
            or resolution.get("native_quarter_hour_truth_eligible") is not False
        ):
            raise ValueError(
                "local intraday provenance must explicitly deny native quarter-hour truth"
            )
    panel = value.get("panel")
    if not isinstance(panel, dict) or set(panel) != {"path", "sha256", "size_bytes"}:
        raise ValueError("intraday provenance manifest panel must be a mapping")
    if panel.get("sha256") != intraday_sha256:
        raise ValueError("intraday provenance manifest does not bind the consumed panel hash")
    manifest_panel_path = _absolute_path(str(panel.get("path", "")))
    if manifest_panel_path != intraday_path:
        raise ValueError("intraday provenance manifest does not bind the consumed panel path")
    return value


def _require_expected_sha256(record: dict[str, object], expected: str, *, label: str) -> None:
    selected = str(expected).strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"{label} expected SHA-256 is invalid")
    if record.get("sha256") != selected:
        raise ValueError(f"{label} differs from the caller-held SHA-256")


def _require_intraday_provenance_arguments(args: argparse.Namespace) -> None:
    has_panel = bool(args.intraday_epex)
    has_manifest = bool(args.intraday_provenance_manifest)
    has_hash = bool(args.expected_intraday_provenance_sha256)
    if has_panel and not (has_manifest and has_hash):
        raise ValueError(
            "an explicit intraday panel requires its provenance manifest and "
            "caller-held SHA-256"
        )
    if not has_panel and (has_manifest or has_hash):
        raise ValueError("intraday provenance requires an explicit intraday panel")
    if has_manifest != has_hash:
        raise ValueError(
            "intraday provenance manifest and caller-held SHA-256 must be supplied together"
        )


def _assert_fresh_output_set(paths: dict[str, Path]) -> None:
    values = list(paths.values())
    duplicates = {path for path in values if values.count(path) > 1}
    if duplicates:
        raise ValueError(
            f"local-test output paths are not distinct: {sorted(map(str, duplicates))}"
        )
    existing = [path for path in values if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to mix or overwrite a local-test run; existing outputs: "
            + ", ".join(str(path) for path in sorted(existing))
        )


def _completion_manifest(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    inputs: dict[str, dict[str, object]],
    outputs: dict[str, Path],
    monthly_manifest: dict[str, object],
) -> dict[str, object]:
    output_records = {
        role: _stable_file_record(path, label=f"local-test {role}", repo_root=repo_root)
        for role, path in sorted(outputs.items())
    }
    return {
        "schema_version": "local_test_ch_pfc_completion.v1",
        "status": "COMPLETE_LOCAL_TEST_NO_GO",
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "confirmatory_rolling_origin_eligible": False,
        "future_holdout_eligible": False,
        "t057_eligible": False,
        "candidate_assembly_eligible": False,
        "local_diagnostic_allowed": True,
        "completion_manifest_last_written": True,
        "monthly_level_authority": monthly_manifest.get("monthly_level_authority"),
        "forward_source_kind": monthly_manifest.get("forward_source_kind"),
        "hard_quote_eligible": monthly_manifest.get("hard_quote_eligible"),
        "monthly_manifest_promotion_eligible": monthly_manifest.get("promotion_eligible"),
        "monthly_manifest_forward_source_sha256": monthly_manifest.get(
            "forward_source_sha256"
        ),
        "code_execution_attestation": (
            "NOT_PROVIDED_SOURCE_HASHES_ARE_OBSERVED_AFTER_IMPORT_ONLY"
        ),
        "inputs": inputs,
        "outputs": output_records,
        "arguments": {key: value for key, value in sorted(vars(args).items())},
    }


def _expand_inventory(
    inventory: pd.DataFrame,
    *,
    years: list[int],
    scenarios: list[str],
    output: Path,
) -> pd.DataFrame:
    expanded = interpolate_electrification_scenario_years(inventory, years, allow_clamp=True)
    expanded = expanded[
        (expanded["country"].astype(str) == "CH")
        & expanded["scenario"].astype(str).isin(scenarios)
    ].copy()
    if expanded.empty:
        raise ValueError("expanded CH scenario inventory is empty")
    _write_parquet(expanded, output)
    return expanded


def _price_summary(frame: pd.DataFrame, *, market: str) -> dict[str, float]:
    tz = "Europe/Zurich" if market == "CH" else "Europe/Berlin"
    local = frame.index.tz_convert(tz)
    price = frame["price_shape"].astype(float)
    return {
        "mean": float(price.mean()),
        "min": float(price.min()),
        "p05": float(price.quantile(0.05)),
        "p95": float(price.quantile(0.95)),
        "max": float(price.max()),
        "midday_mean": float(price[(local.hour >= 10) & (local.hour <= 15)].mean()),
        "evening_mean": float(price[(local.hour >= 18) & (local.hour <= 21)].mean()),
        "night_mean": float(price[(local.hour <= 5) | (local.hour >= 22)].mean()),
    }


def _build_fan(pfc_by_scenario: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    curves = {scenario: frame["price_shape"].astype(float) for scenario, frame in pfc_by_scenario.items()}
    fan = structural_fan_chart(curves, weights=weights)
    out = pd.DataFrame(index=fan.index)
    for scenario, frame in pfc_by_scenario.items():
        out[f"curve_{scenario}"] = frame["price_shape"].astype(float)
    out["weighted_mean"] = fan["weighted_mean"]
    out["structural_scenario_low"] = fan["scenario_low"]
    out["structural_scenario_central"] = fan["scenario_central"]
    out["structural_scenario_high"] = fan["scenario_high"]
    out["structural_scenario_spread"] = fan["scenario_spread"]
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("fan chart contains non-finite values")
    if (out["structural_scenario_low"] > out["structural_scenario_high"]).any():
        raise ValueError("fan chart violates low <= high")
    return out


def _intraday_fit_diagnostics(
    si: object,
    frame: pd.DataFrame,
    *,
    source: str,
    market: str,
) -> dict[str, object]:
    base_factors = dict(getattr(si, "base_factors_", {}))
    observations = dict(getattr(si, "n_obs_", {}))
    direct_keys = set(observations).intersection(base_factors)
    fallback_keys = set(base_factors).difference(direct_keys)
    direct_matrix = np.asarray([base_factors[key] for key in sorted(direct_keys)], dtype=float)
    surface_matrix = np.asarray(list(base_factors.values()), dtype=float)
    seasons = ("Hiver", "Printemps", "Ete", "Automne")
    direct_by_season = {
        season: sum(1 for key in direct_keys if key[0] == season) for season in seasons
    }
    observed_counts = np.asarray([observations[key] for key in direct_keys], dtype=float)
    return {
        "source": source,
        "market": market,
        "rows": int(len(frame)),
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
        "surface_cells_after_fallback": int(len(base_factors)),
        "directly_learned_cells": int(len(direct_keys)),
        "fallback_cells": int(len(fallback_keys)),
        "fallback_fraction": (
            float(len(fallback_keys) / len(base_factors)) if base_factors else 1.0
        ),
        "direct_nonflat_cells": (
            int(np.sum(np.max(np.abs(direct_matrix - 1.0), axis=1) > 1e-12))
            if direct_matrix.size
            else 0
        ),
        "direct_cells_by_season": direct_by_season,
        "missing_direct_seasons": [
            season for season, count in direct_by_season.items() if count == 0
        ],
        "direct_n_obs_min": int(observed_counts.min()) if observed_counts.size else 0,
        "direct_n_obs_median": (
            float(np.median(observed_counts)) if observed_counts.size else 0.0
        ),
        "direct_n_obs_max": int(observed_counts.max()) if observed_counts.size else 0,
        "max_abs_deviation_from_one": (
            float(np.max(np.abs(surface_matrix - 1.0))) if surface_matrix.size else 0.0
        ),
        "sparse_support_envelope_max_abs_deviation": getattr(
            si, "sparse_support_envelope_", None
        ),
        "sparse_support_constrained_cells": int(
            getattr(si, "sparse_support_constrained_cells_", 0)
        ),
    }


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    latest_forward_date: pd.Timestamp,
    scenarios: list[str],
    weights: dict[str, float],
    expanded: pd.DataFrame,
    pfc_paths: dict[str, Path],
    fan: pd.DataFrame,
    fan_path: Path,
    governance_report: Path,
    intraday_diagnostics: dict[str, object],
    monthly_manifest: dict[str, object],
    forward_input: dict[str, object],
    completion_manifest_path: Path,
) -> None:
    metrics = pd.DataFrame(
        [
            {"scenario": scenario, **_price_summary(pd.read_parquet(pfc_paths[scenario]), market=args.market)}
            for scenario in scenarios
        ]
    )
    fan_metrics = pd.DataFrame(
        [
            {
                "rows": len(fan),
                "weighted_mean": float(fan["weighted_mean"].mean()),
                "scenario_spread_mean": float(fan["structural_scenario_spread"].mean()),
                "scenario_spread_p95": float(fan["structural_scenario_spread"].quantile(0.95)),
                "scenario_spread_max": float(fan["structural_scenario_spread"].max()),
            }
        ]
    )
    display_cols = [
        "scenario",
        "delivery_year",
        "scenario_weight",
        "quality_flag",
        "demand_twh",
        "peak_load_gw",
        "pv_gw",
        "pv_twh",
        "wind_gw",
        "wind_twh",
        "battery_power_gw",
        "battery_energy_gwh",
        "ev_twh",
        "heatpump_twh",
        "hydro_reservoir_twh",
        "net_import_twh",
        "ntc_ch_de_gw",
        "ntc_ch_fr_gw",
        "ntc_ch_it_gw",
        "ntc_ch_at_gw",
    ]
    display_cols = [col for col in display_cols if col in expanded.columns]
    lines = [
        "# Local-Test CH PFC 2030",
        "",
        "* status: `agent-approved local/test only`",
        "* production approval: `NO`",
        "* production activation allowed: `NO`",
        f"* governance report: `{governance_report}`",
        f"* source inventory: `{args.inventory}`",
        f"* expanded scenario path: `{args.expanded_output}`",
        f"* feature output: `{args.features_output}`",
        f"* fan chart: `{fan_path}`",
        f"* market: `{args.market}`",
        f"* start date UTC: `{args.start_date}`",
        f"* horizon days: `{args.horizon_days}`",
        f"* EEX forward date: `{latest_forward_date.date()}`",
        f"* monthly level authority: `{monthly_manifest.get('monthly_level_authority')}`",
        f"* forward source kind: `{monthly_manifest.get('forward_source_kind')}`",
        f"* monthly manifest forward source SHA-256: `{monthly_manifest.get('forward_source_sha256')}`",
        f"* actual forward input SHA-256: `{forward_input['sha256']}`",
        f"* hard quote eligible: `{monthly_manifest.get('hard_quote_eligible')}`",
        f"* monthly manifest promotion eligible: `{monthly_manifest.get('promotion_eligible')}`",
        f"* completion manifest (last-written): `{completion_manifest_path}`",
        f"* scenario weights: `{', '.join(f'{k}={v:.4f}' for k, v in weights.items())}`",
        f"* intraday source: `{intraday_diagnostics['source']}`",
        f"* intraday provenance manifest: `{args.intraday_provenance_manifest}`",
        f"* expected intraday provenance SHA-256: `{args.expected_intraday_provenance_sha256}`",
        f"* intraday source market: `{intraday_diagnostics['market']}`",
        f"* intraday source rows: `{intraday_diagnostics['rows']}`",
        f"* intraday calibration range: `{intraday_diagnostics['start']}` -> `{intraday_diagnostics['end']}`",
        f"* intraday surface cells after fallback: `{intraday_diagnostics['surface_cells_after_fallback']}`",
        f"* intraday directly learned cells: `{intraday_diagnostics['directly_learned_cells']}`",
        f"* intraday fallback cells: `{intraday_diagnostics['fallback_cells']}` "
        f"(`{intraday_diagnostics['fallback_fraction']:.2%}`)",
        f"* intraday direct non-flat cells: `{intraday_diagnostics['direct_nonflat_cells']}`",
        f"* intraday direct cells by season: `{intraday_diagnostics['direct_cells_by_season']}`",
        f"* intraday missing direct seasons: `{intraday_diagnostics['missing_direct_seasons']}`",
        f"* intraday direct n_obs min/median/max: `{intraday_diagnostics['direct_n_obs_min']}` / "
        f"`{intraday_diagnostics['direct_n_obs_median']}` / `{intraday_diagnostics['direct_n_obs_max']}`",
        "* intraday sparse support envelope max absolute deviation: "
        f"`{intraday_diagnostics['sparse_support_envelope_max_abs_deviation']}`",
        "* intraday sparse cells contracted to dense support: "
        f"`{intraday_diagnostics['sparse_support_constrained_cells']}`",
        f"* intraday max absolute factor deviation from 1: `{intraday_diagnostics['max_abs_deviation_from_one']:.8f}`",
        "",
        "## PFC Outputs",
        "",
        *_df_to_md(pd.DataFrame({"scenario": list(pfc_paths), "path": [str(p) for p in pfc_paths.values()]})),
        "",
        "## Price Summary",
        "",
        *_df_to_md(metrics),
        "",
        "## Structural Fan Chart",
        "",
        *_df_to_md(fan_metrics),
        "",
        "## Expanded CH Scenario Rows",
        "",
        *_df_to_md(expanded[display_cols].sort_values(["scenario", "delivery_year"])),
        "",
        "## Limitations",
        "",
        "* This curve is suitable for local validation, diagnostics and model review only.",
        "* Intraday cells without direct observations are explicit fallbacks and are not evidence of seasonal learning.",
        "* The DE-LU intraday proxy is not validated as a CH proxy by this local build.",
        "* The scenario curves do not contain calibrated probabilistic p10/p90 bands; those columns remain null.",
        "* Agent approval replaces human approval only for local/test work.",
        "* Production FMV use still requires the production governance gate with accountable human sign-off.",
        "* Proxy/partial/internal quality flags remain visible and are not relabelled as production-governed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_durable_exact_bytes(
        path,
        ("\n".join(lines) + "\n").encode("utf-8"),
        label="local-test PFC summary",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--governance-report", default=str(DEFAULT_GOVERNANCE_REPORT))
    parser.add_argument("--vintage", default="2026-06-12")
    parser.add_argument("--scenarios", default="slow,central,fast")
    parser.add_argument("--weights", default="0.25,0.50,0.25")
    parser.add_argument("--market", default="CH")
    parser.add_argument("--start-date", default="2030-01-01")
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument(
        "--delivery-local-end-exclusive",
        default=None,
        help=(
            "Exact end-exclusive local-market delivery timestamp. When provided, "
            "it is authoritative over the day-count approximation."
        ),
    )
    parser.add_argument("--epex-hourly", default="data/epex_hourly.parquet")
    parser.add_argument(
        "--intraday-epex",
        default=None,
        help=(
            "Optional distinct 15-minute EPEX source for ShapeIntraday. "
            "Use DE-LU post-2025-10-01 to mirror the canonical LT pipeline."
        ),
    )
    parser.add_argument("--intraday-market", default="DE")
    parser.add_argument("--intraday-cutoff", default="2025-10-01T00:00:00Z")
    parser.add_argument("--intraday-provenance-manifest", default=None)
    parser.add_argument("--expected-intraday-provenance-sha256", default=None)
    parser.add_argument(
        "--require-nonflat-intraday-shape",
        action="store_true",
        help="Fail if the fitted 15-minute factor surface is entirely flat.",
    )
    parser.add_argument(
        "--require-direct-intraday-seasons",
        default="",
        help=(
            "Comma-separated seasons that must contain directly fitted cells "
            "(Hiver,Printemps,Ete,Automne)."
        ),
    )
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--expanded-output", default="output/local_test_ch_pfc_2030_scenario_expanded.parquet")
    parser.add_argument(
        "--features-output",
        default="output/local_test_ch_pfc_2030_hpfc_scenario_features.parquet",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output-prefix", default="local_test_ch_pfc_2030")
    parser.add_argument("--fan-chart-output", default="output/local_test_ch_pfc_2030_structural_fan_chart.parquet")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument(
        "--completion-manifest",
        default=None,
        help=(
            "Last-written immutable completion manifest. Defaults beside the fan chart."
        ),
    )
    parser.add_argument(
        "--enable-monthly-forward-curve-solver",
        action="store_true",
        default=True,
        help=(
            "Compatibility flag; the governed monthly BASE solver is always enabled "
            "and is the LT level authority."
        ),
    )
    parser.add_argument(
        "--monthly-solver-constraint-tolerance",
        type=float,
        default=DEFAULT_LOCAL_MONTHLY_CONSTRAINT_TOLERANCE,
    )
    parser.add_argument("--monthly-solver-lambda-smooth-month", type=float, default=1.0)
    parser.add_argument("--monthly-solver-lambda-smooth-yoy", type=float, default=0.25)
    parser.add_argument("--monthly-solver-lambda-shape", type=float, default=1.0)
    parser.add_argument("--monthly-solver-neighbor-shrinkage", type=float, default=0.5)
    parser.add_argument("--monthly-solver-delivery-local-start-date", default=None)
    parser.add_argument("--monthly-solver-delivery-local-end-date", default=None)
    parser.add_argument(
        "--monthly-solver-allow-template-structural-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--monthly-solver-structural-amplitude", type=float, default=110.0)
    parser.add_argument(
        "--disable-cascade-trend-for-annual-only",
        action="store_true",
        help=(
            "Local/test guard: do not extrapolate fitted seasonal trends when a delivery "
            "year has only a Cal quote. Quoted months/quarters are still preserved."
        ),
    )
    parser.add_argument(
        "--enable-experimental-sparse-intraday-regularization",
        action="store_true",
        help=(
            "Local diagnostic only: use the rolling-origin candidate sparse-cell "
            "price-space fit and dense-support envelope. This candidate is not a "
            "production default or promotion gate."
        ),
    )
    args = parser.parse_args(argv)
    _require_intraday_provenance_arguments(args)
    args.monthly_solver_constraint_tolerance = (
        _require_exact_monthly_constraint_tolerance(
            args.monthly_solver_constraint_tolerance
        )
    )
    delivery_index = None
    if args.delivery_local_end_exclusive:
        if (
            args.monthly_solver_delivery_local_start_date
            or args.monthly_solver_delivery_local_end_date
        ):
            raise ValueError(
                "exact delivery end-exclusive cannot be combined with monthly solver "
                "delivery overrides"
            )
        delivery_index = _delivery_index_to_local_end_exclusive(
            start_date=args.start_date,
            local_end_exclusive=args.delivery_local_end_exclusive,
            market=args.market,
        )

    repo_root = _absolute_path(Path.cwd())
    scenarios = _parse_csv(args.scenarios)
    weights = _parse_weights(args.weights, scenarios)
    inventory_path = _input_path(args.inventory, repo_root=repo_root)
    manifest_path = _input_path(args.manifest, repo_root=repo_root)
    governance_report_path = _output_path(args.governance_report, repo_root=repo_root)
    expanded_path = _output_path(args.expanded_output, repo_root=repo_root)
    features_path = _output_path(args.features_output, repo_root=repo_root)
    output_dir = _output_path(args.output_dir, repo_root=repo_root)
    output_prefix = _safe_output_prefix(args.output_prefix)
    pfc_paths = {
        scenario: _output_path(
            output_dir / f"{output_prefix}_{_safe_slug(scenario)}.parquet",
            repo_root=repo_root,
        )
        for scenario in scenarios
    }
    fan_path = _output_path(args.fan_chart_output, repo_root=repo_root)
    monthly_manifest_path = fan_path.with_suffix(".monthly_curve_manifest.json")
    final_product_replay_path = fan_path.with_suffix(".final_product_replay.json")
    summary_path = _output_path(args.summary, repo_root=repo_root)
    completion_manifest_path = _output_path(
        args.completion_manifest
        or fan_path.with_suffix(".local_test_completion_manifest.json"),
        repo_root=repo_root,
    )
    artifact_paths = {
        "governance_report": governance_report_path,
        "expanded_scenarios": expanded_path,
        "scenario_features": features_path,
        **{f"pfc_{scenario}": path for scenario, path in pfc_paths.items()},
        "structural_fan_chart": fan_path,
        "monthly_curve_manifest": monthly_manifest_path,
        "final_product_replay": final_product_replay_path,
        "quality_summary": summary_path,
        "completion_manifest": completion_manifest_path,
    }
    _assert_fresh_output_set(artifact_paths)
    input_paths = {
        "scenario_inventory": inventory_path,
        "scenario_governance_manifest": manifest_path,
        "hourly_epex": _input_path(args.epex_hourly, repo_root=repo_root),
        "forward_history": _input_path(args.forwards, repo_root=repo_root),
        "local_runner_source_observed_after_import": _input_path(
            __file__, repo_root=repo_root
        ),
        "multi_scenario_builder_source_observed_after_import": _input_path(
            _build_one_curve.__code__.co_filename, repo_root=repo_root
        ),
        "assembler_source_observed_after_import": _input_path(
            "pfc_shaping/lt/model/assembler.py", repo_root=repo_root
        ),
        "intraday_model_source_observed_after_import": _input_path(
            "pfc_shaping/lt/model/shape_intraday.py", repo_root=repo_root
        ),
    }
    if args.intraday_epex:
        input_paths["intraday_epex"] = _input_path(
            args.intraday_epex, repo_root=repo_root
        )
    if args.intraday_provenance_manifest:
        input_paths["intraday_provenance_manifest"] = _input_path(
            args.intraday_provenance_manifest, repo_root=repo_root
        )
    consumed_roles = {
        "scenario_inventory",
        "scenario_governance_manifest",
        "hourly_epex",
        "forward_history",
        "intraday_epex",
        "intraday_provenance_manifest",
    }
    captured_payloads: dict[str, bytes] = {}
    input_records: dict[str, dict[str, object]] = {}
    for role, path in sorted(input_paths.items()):
        if role in consumed_roles:
            payload = read_stable_single_link_file(path, label=f"local-test captured {role}")
            captured_payloads[role] = payload
            input_records[role] = _file_record_from_payload(
                path,
                payload,
                repo_root=repo_root,
            )
        else:
            input_records[role] = _stable_file_record(
                path,
                label=f"local-test {role}",
                repo_root=repo_root,
            )
    if args.intraday_provenance_manifest:
        _require_expected_sha256(
            input_records["intraday_provenance_manifest"],
            args.expected_intraday_provenance_sha256,
            label="intraday provenance manifest",
        )
    intraday_payload = captured_payloads.get("intraday_epex")
    if args.intraday_provenance_manifest:
        provenance_payload = captured_payloads["intraday_provenance_manifest"]
        _require_expected_sha256(
            input_records["intraday_provenance_manifest"],
            args.expected_intraday_provenance_sha256,
            label="intraday provenance manifest",
        )
        _admit_intraday_provenance(
            manifest_payload=provenance_payload,
            intraday_path=input_paths["intraday_epex"],
            intraday_sha256=str(input_records["intraday_epex"]["sha256"]),
        )
    inventory = validate_electrification_scenarios(
        pd.read_parquet(io.BytesIO(captured_payloads["scenario_inventory"])),
        require_recommended=True,
    )
    manifest = yaml.safe_load(
        captured_payloads["scenario_governance_manifest"].decode("utf-8")
    ) or {}

    issues, effective = validate_governance(
        inventory=inventory,
        manifest=manifest,
        vintage=args.vintage,
        countries=["CH", "DE", "FR", "IT", "AT"],
        scenarios=scenarios,
        years=[2030],
        mode="local-test",
    )
    _write_governance_report(
        governance_report_path,
        manifest_path=manifest_path,
        inventory_path=inventory_path,
        vintage=args.vintage,
        issues=issues,
        effective=effective,
    )
    if issues:
        raise ValueError(f"local-test governance gate failed with {len(issues)} issues")

    if delivery_index is None:
        years = _delivery_years(args.start_date, args.horizon_days, args.market)
    else:
        timezone = "Europe/Zurich" if args.market == "CH" else "Europe/Berlin"
        years = sorted(
            delivery_index.tz_convert(timezone).year.unique().astype(int).tolist()
        )
    expanded = _expand_inventory(
        inventory,
        years=years,
        scenarios=scenarios,
        output=expanded_path,
    )
    _write_parquet(derive_hpfc_scenario_features(expanded), features_path)

    epex = _load_epex_hourly(io.BytesIO(captured_payloads["hourly_epex"]))
    intraday_epex = (
        _load_epex_hourly(io.BytesIO(intraday_payload))
        if intraday_payload is not None
        else None
    )
    latest, base_prices = _latest_forwards(
        io.BytesIO(captured_payloads["forward_history"]),
        market=args.market,
    )
    reference_date = pd.Timestamp(latest, tz="UTC")
    monthly_authority = None
    if args.enable_monthly_forward_curve_solver:
        settings = monthly_solver_settings(
            {
                "forwards": {
                    "monthly_curve_solver": {
                        "enabled": True,
                        "eex_history_path": args.forwards,
                        "constraint_tolerance": args.monthly_solver_constraint_tolerance,
                        "lambda_smooth_month": args.monthly_solver_lambda_smooth_month,
                        "lambda_smooth_yoy": args.monthly_solver_lambda_smooth_yoy,
                        "lambda_shape": args.monthly_solver_lambda_shape,
                        "neighbor_shrinkage": args.monthly_solver_neighbor_shrinkage,
                        "allow_template_structural_fallback": args.monthly_solver_allow_template_structural_fallback,
                        "structural_amplitude_eur_mwh": args.monthly_solver_structural_amplitude,
                    }
                }
            }
        )
        if delivery_index is not None:
            delivery_months = _delivery_months_from_index(
                delivery_index,
                market=args.market,
            )
        elif args.monthly_solver_delivery_local_start_date or args.monthly_solver_delivery_local_end_date:
            if not (
                args.monthly_solver_delivery_local_start_date
                and args.monthly_solver_delivery_local_end_date
            ):
                raise ValueError(
                    "monthly solver local delivery override requires both start and end dates"
                )
            delivery_months = delivery_months_for_local_window(
                start_date=args.monthly_solver_delivery_local_start_date,
                end_date=args.monthly_solver_delivery_local_end_date,
                timezone="Europe/Zurich" if args.market == "CH" else "Europe/Berlin",
            )
        else:
            delivery_months = delivery_months_for_window(
                start_date=args.start_date,
                horizon_days=args.horizon_days,
                timezone="Europe/Zurich" if args.market == "CH" else "Europe/Berlin",
            )
        monthly_authority = solve_monthly_level_authority_from_history(
            forwards_path=args.forwards,
            market=args.market,
            delivery_months=delivery_months,
            settings=settings,
            timezone="Europe/Zurich" if args.market == "CH" else "Europe/Berlin",
            original_forward_prices=base_prices,
            forwards_payload=captured_payloads["forward_history"],
        )
    if monthly_authority is None:
        raise RuntimeError("monthly solver authority is mandatory for the local LT runner")
    sh, si, cascader, calibrator = _fit_components(
        epex,
        args.market,
        intraday_epex=intraday_epex,
        intraday_market=args.intraday_market if intraday_epex is not None else args.market,
        intraday_cutoff=args.intraday_cutoff if intraday_epex is not None else None,
        disable_cascade_trend_for_annual_only=bool(args.disable_cascade_trend_for_annual_only),
        enable_sparse_intraday_regularization=bool(
            args.enable_experimental_sparse_intraday_regularization
        ),
    )
    intraday_frame = epex if intraday_epex is None else intraday_epex
    if intraday_epex is not None:
        cutoff = pd.Timestamp(args.intraday_cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        intraday_frame = intraday_frame.loc[intraday_frame.index >= cutoff]
    intraday_diagnostics = _intraday_fit_diagnostics(
        si,
        intraday_frame,
        source=str(args.intraday_epex or args.epex_hourly),
        market=str(args.intraday_market if intraday_epex is not None else args.market),
    )
    if (
        args.require_nonflat_intraday_shape
        and intraday_diagnostics["direct_nonflat_cells"] == 0
    ):
        raise ValueError(
            "intraday shape has no directly learned non-flat cell despite the quality requirement"
        )
    required_seasons = {
        value.strip() for value in args.require_direct_intraday_seasons.split(",") if value.strip()
    }
    allowed_seasons = {"Hiver", "Printemps", "Ete", "Automne"}
    if not required_seasons.issubset(allowed_seasons):
        raise ValueError("required intraday seasons contain an unsupported value")
    direct_by_season = dict(intraday_diagnostics["direct_cells_by_season"])
    missing_required = sorted(
        season for season in required_seasons if direct_by_season.get(season, 0) == 0
    )
    if missing_required:
        raise ValueError(
            f"intraday direct-fit coverage is missing required seasons: {missing_required}"
        )

    forward_snapshot = _forward_snapshot_from_payload(
        captured_payloads["forward_history"],
        market=args.market,
        forward_date=latest,
    )
    eligible_peak_prices = _peak_prices_for_delivery(
        forward_snapshot,
        delivery_months=delivery_months,
    )
    pfc_by_scenario: dict[str, pd.DataFrame] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        pfc = _build_one_curve(
            scenario=scenario,
            scenario_path=expanded_path,
            base_prices=base_prices,
            reference_date=reference_date,
            market=args.market,
            start_date=args.start_date,
            horizon_days=args.horizon_days,
            sh=sh,
            si=si,
            cascader=cascader,
            calibrator=calibrator,
            monthly_authority=monthly_authority,
            delivery_index=delivery_index,
            peak_prices=eligible_peak_prices,
        )
        pfc_by_scenario[scenario] = pfc

    final_product_replay = _final_product_replay(
        pfc_by_scenario,
        forward_snapshot=forward_snapshot,
        forward_date=latest,
        market=args.market,
    )
    for scenario, pfc in sorted(pfc_by_scenario.items()):
        _write_parquet(pfc, pfc_paths[scenario], index=True)

    fan = _build_fan(pfc_by_scenario, weights)
    _write_parquet(fan, fan_path, index=True)
    monthly_manifest = dict(monthly_authority.manifest)
    _write_json(
        monthly_manifest_path,
        monthly_manifest,
        label="local-test monthly curve manifest",
    )
    _write_json(
        final_product_replay_path,
        final_product_replay,
        label="local-test final product replay",
    )
    _write_summary(
        summary_path,
        args=args,
        latest_forward_date=latest,
        scenarios=scenarios,
        weights=weights,
        expanded=expanded,
        pfc_paths=pfc_paths,
        fan=fan,
        fan_path=fan_path,
        governance_report=governance_report_path,
        intraday_diagnostics=intraday_diagnostics,
        monthly_manifest=monthly_manifest,
        forward_input=input_records["forward_history"],
        completion_manifest_path=completion_manifest_path,
    )
    final_input_records = {
        role: _stable_file_record(path, label=f"local-test {role}", repo_root=repo_root)
        for role, path in sorted(input_paths.items())
    }
    if final_input_records != input_records:
        raise ValueError("local-test inputs changed while the model was running")
    completed_outputs = {
        role: path for role, path in artifact_paths.items() if role != "completion_manifest"
    }
    completion = _completion_manifest(
        args=args,
        repo_root=repo_root,
        inputs=input_records,
        outputs=completed_outputs,
        monthly_manifest=monthly_manifest,
    )
    _write_json(
        completion_manifest_path,
        completion,
        label="local-test PFC completion manifest",
    )
    print(f"[local-test-pfc] governance -> {governance_report_path}")
    for scenario, path in pfc_paths.items():
        print(f"[local-test-pfc] {scenario} -> {path}")
    print(f"[local-test-pfc] fan chart -> {fan_path}")
    print(f"[local-test-pfc] monthly curve manifest -> {monthly_manifest_path}")
    print(f"[local-test-pfc] final product replay -> {final_product_replay_path}")
    print(f"[local-test-pfc] summary -> {summary_path}")
    print(f"[local-test-pfc] completion manifest -> {completion_manifest_path}")
    print(
        f"[local-test-pfc] weighted_mean={fan['weighted_mean'].mean():.2f} "
        f"scenario_spread_mean={fan['structural_scenario_spread'].mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
