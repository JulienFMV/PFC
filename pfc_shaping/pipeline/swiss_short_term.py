from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from pfc_shaping.model.lear_forecaster import LEARForecaster
from pfc_shaping.storage.local_duckdb import init_db, upsert_lear_backtest, upsert_lear_forecast

DEFAULT_PRICEFM_PYTHON = r"C:\Users\jbattaglia\.conda\pricefm_tf\python.exe"


def _foundation_enabled() -> bool:
    return os.getenv("PFC_CT_ENABLE_FOUNDATION", "1").strip() == "1"


def _governed_forecast_features_enabled() -> bool:
    return os.getenv("PFC_CT_ENABLE_GOVERNED_FORECAST_FEATURES", "0").strip() == "1"


@dataclass
class SwissShortTermInputs:
    epex_ch: pd.DataFrame
    epex_de: pd.DataFrame
    neighbor_prices_15min: dict[str, pd.DataFrame] | None
    entso: pd.DataFrame
    hydro: pd.DataFrame
    commodities: pd.DataFrame | None
    outages_all: pd.DataFrame | None
    de_renewable_forecast: pd.DataFrame | None
    multi_country_forecast: pd.DataFrame | None
    fr_nuclear_forecast: pd.DataFrame | None
    weather_forecast: pd.DataFrame | None
    base_pfc_ch: pd.DataFrame
    require_de_exogenous: bool = True
    required_neighbor_codes: tuple[str, ...] = ("de",)


@dataclass
class SwissShortTermInputHealth:
    epex_ch_rows: int
    epex_de_rows: int
    entso_rows: int
    hydro_rows: int
    outages_rows: int
    commodities_rows: int
    de_renewable_forecast_rows: int
    multi_country_forecast_rows: int
    weather_forecast_rows: int
    fr_nuclear_forecast_rows: int
    ch_de_overlap_hours: int
    entso_ch_overlap_hours: int
    has_de_price_support: bool
    neighbor_overlap_hours: dict[str, int]
    has_required_neighbor_support: bool
    governed_forecast_features_requested: bool = False
    governed_forecast_features_enabled: bool = False
    governed_forecast_coverage_ratio: float = 0.0
    governed_forecast_min_coverage_ratio: float = 0.0
    governed_forecast_max_supported_horizon_days: int = 0
    governed_forecast_applied_horizon_days: int = 0
    governed_forecast_missing_sources: list[str] | None = None
    governed_forecast_source_coverage: dict[str, float] | None = None
    governed_forecast_vintage_schema_verified: bool | None = None


@dataclass
class SwissShortTermArtifacts:
    pfc_ch: pd.DataFrame
    lear_forecast: pd.DataFrame
    lear_run_id: str | None


def run_swiss_short_term_overlay(
    project_root: str,
    inputs: SwissShortTermInputs,
    logger: logging.Logger,
) -> SwissShortTermArtifacts:
    logger.info("=" * 70)
    logger.info("STEP 10: LEAR short-term forecast (Swiss CT branch)")
    logger.info("=" * 70)
    t_lear = time.time()

    input_health = _validate_swiss_short_term_inputs(inputs, logger)

    use_foundation_model = _foundation_enabled()
    use_governed_forecast_features = _governed_forecast_features_enabled()
    min_governed_forecast_coverage = float(os.getenv("PFC_CT_MIN_GOVERNED_FORECAST_COVERAGE", "0.98"))
    logger.info(
        "  Swiss CT model policy: LEAR CH+DE baseline, foundation=%s, governed_forecast_features=%s, required_neighbors=%s",
        "enabled" if use_foundation_model else "disabled",
        "enabled" if use_governed_forecast_features else "disabled",
        inputs.required_neighbor_codes,
    )

    primary_lear = _fit_lear_model(
        inputs=inputs,
        use_foundation_model=use_foundation_model,
        use_governed_forecast_features=False,
    )

    governed_lear: LEARForecaster | None = None
    forecast_horizon_days = 10
    governed_applied_horizon_days = 0
    if use_governed_forecast_features:
        governed_lear = _fit_lear_model(
            inputs=inputs,
            use_foundation_model=False,
            use_governed_forecast_features=True,
        )
        governed_health = governed_lear.assess_governed_forecast_feature_health(
            horizon_days=forecast_horizon_days,
            min_coverage_ratio=min_governed_forecast_coverage,
        )
    else:
        governed_health = primary_lear.assess_governed_forecast_feature_health(
            horizon_days=forecast_horizon_days,
            min_coverage_ratio=min_governed_forecast_coverage,
        )

    max_supported_horizon_days = int(governed_health.get("max_supported_horizon_days", 0))
    if use_governed_forecast_features and max_supported_horizon_days <= 0:
        logger.warning(
            "  Disabling governed forecast features for this run: coverage=%.3f < %.3f, max_supported_horizon_days=%s, missing sources=%s",
            governed_health.get("coverage_ratio", 0.0),
            governed_health.get("min_coverage_ratio", min_governed_forecast_coverage),
            max_supported_horizon_days,
            governed_health.get("missing_sources", []),
        )
        governed_lear = None
    elif use_governed_forecast_features and max_supported_horizon_days < forecast_horizon_days:
        logger.warning(
            "  Governed forecast features only supported through J+%d; using baseline fallback for J+%d..J+%d",
            max_supported_horizon_days,
            max_supported_horizon_days + 1,
            forecast_horizon_days,
        )
        governed_applied_horizon_days = max_supported_horizon_days
    elif use_governed_forecast_features:
        governed_applied_horizon_days = forecast_horizon_days

    governed_enabled_final = bool(use_governed_forecast_features and governed_applied_horizon_days > 0)
    input_health = replace(
        input_health,
        governed_forecast_features_requested=use_governed_forecast_features,
        governed_forecast_features_enabled=governed_enabled_final,
        governed_forecast_coverage_ratio=float(governed_health.get("coverage_ratio", 0.0)),
        governed_forecast_min_coverage_ratio=float(governed_health.get("min_coverage_ratio", 0.0)),
        governed_forecast_max_supported_horizon_days=max_supported_horizon_days,
        governed_forecast_applied_horizon_days=governed_applied_horizon_days,
        governed_forecast_missing_sources=list(governed_health.get("missing_sources", [])),
        governed_forecast_source_coverage=dict(governed_health.get("source_coverage", {})),
        governed_forecast_vintage_schema_verified=bool(governed_health.get("vintage_schema_verified", False)),
    )
    _save_input_health(input_health, logger)

    primary_forecast = primary_lear.predict(horizon_days=forecast_horizon_days)
    lear_forecast = primary_forecast.copy()
    if governed_lear is not None and governed_applied_horizon_days > 0:
        governed_forecast = governed_lear.predict(horizon_days=forecast_horizon_days)
        lear_forecast = _merge_governed_and_baseline_forecasts(
            governed_forecast=governed_forecast,
            baseline_forecast=primary_forecast,
            governed_horizon_days=governed_applied_horizon_days,
        )
        if use_foundation_model:
            lear_forecast = _overwrite_routed_day_slice(
                base_forecast=lear_forecast,
                replacement_forecast=primary_forecast,
                day_values=[1],
            )
    logger.info("  LEAR forecast: %d hours, mean=%.1f EUR/MWh", len(lear_forecast), lear_forecast["price_lear"].mean())

    lear_forecast = _maybe_apply_experimental_pricefm(project_root, lear_forecast, logger)
    pfc_ch = lear.blend_with_pfc(inputs.base_pfc_ch.copy(), lear_forecast)

    lear_base = os.path.join("pfc_shaping", "output", f"lear_forecast_{pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lear_forecast.to_parquet(f"{lear_base}.parquet", index=False)
    lear_forecast.to_csv(f"{lear_base}.csv", index=False)
    lear_forecast.to_parquet(os.path.join("pfc_shaping", "output", "lear_forecast_latest.parquet"), index=False)
    lear_forecast.to_csv(os.path.join("pfc_shaping", "output", "lear_forecast_latest.csv"), index=False)
    logger.info("  LEAR standalone saved: %s.parquet", lear_base)

    lear_run_id = _persist_lear_forecast(lear_forecast, logger)
    _maybe_run_lear_backtest(primary_lear, lear_run_id, logger)

    logger.info("  LEAR completed in %.1fs", time.time() - t_lear)
    return SwissShortTermArtifacts(pfc_ch=pfc_ch, lear_forecast=lear_forecast, lear_run_id=lear_run_id)


def _fit_lear_model(
    inputs: SwissShortTermInputs,
    use_foundation_model: bool,
    use_governed_forecast_features: bool,
) -> LEARForecaster:
    lear = LEARForecaster(
        tz="Europe/Zurich",
        use_foundation_model=use_foundation_model,
        use_governed_forecast_features=use_governed_forecast_features,
    )
    lear.fit(
        epex_15min=inputs.epex_ch,
        entso_15min=inputs.entso,
        outages_15min=inputs.outages_all,
        commodities=inputs.commodities,
        hydro=inputs.hydro,
        epex_de_15min=inputs.epex_de,
        neighbor_price_15min=inputs.neighbor_prices_15min,
        de_renewable_forecast=inputs.de_renewable_forecast,
        multi_country_forecast=inputs.multi_country_forecast,
        fr_nuclear_forecast=inputs.fr_nuclear_forecast,
        weather_forecast=inputs.weather_forecast,
    )
    return lear


def _merge_governed_and_baseline_forecasts(
    governed_forecast: pd.DataFrame,
    baseline_forecast: pd.DataFrame,
    governed_horizon_days: int,
) -> pd.DataFrame:
    if governed_horizon_days <= 0:
        return baseline_forecast.copy()
    if governed_horizon_days >= int(baseline_forecast["days_ahead"].max()):
        return governed_forecast.copy()

    join_cols = ["timestamp"]
    if "hour" in governed_forecast.columns and "hour" in baseline_forecast.columns:
        join_cols.append("hour")
    cols_keep = [c for c in ["date", "days_ahead", "n_windows", "mlp_used", "fm_used", "fm_raw_price"] if c in baseline_forecast.columns or c in governed_forecast.columns]

    gov = governed_forecast.copy().rename(
        columns={
            "price_lear": "price_lear_gov",
            "price_p10": "price_p10_gov",
            "price_p90": "price_p90_gov",
        }
    )
    base = baseline_forecast.copy().rename(
        columns={
            "price_lear": "price_lear_base",
            "price_p10": "price_p10_base",
            "price_p90": "price_p90_base",
        }
    )
    merged = base.merge(
        gov[[c for c in gov.columns if c in join_cols or c.endswith("_gov")]],
        on=join_cols,
        how="left",
    )
    merged = merged.sort_values(["timestamp", "hour"] if "hour" in merged.columns else ["timestamp"]).reset_index(drop=True)

    def weight(days_ahead: int) -> float:
        d = float(days_ahead)
        if d <= governed_horizon_days:
            return 1.0
        if d <= governed_horizon_days + 2:
            return float(np.clip(1.0 - (d - governed_horizon_days) / 2.0, 0.0, 1.0))
        return 0.0

    merged["_w_governed"] = merged["days_ahead"].astype(float).map(weight)
    for col in ["price_lear", "price_p10", "price_p90"]:
        gov_col = f"{col}_gov"
        base_col = f"{col}_base"
        if gov_col not in merged.columns or base_col not in merged.columns:
            continue
        merged[col] = (
            merged["_w_governed"] * merged[gov_col].fillna(merged[base_col])
            + (1.0 - merged["_w_governed"]) * merged[base_col]
        )

    if all(col in merged.columns for col in ["price_p10_gov", "price_p90_gov", "price_p10_base", "price_p90_base"]):
        mid_g = (merged["price_p10_gov"].fillna(merged["price_p10_base"]) + merged["price_p90_gov"].fillna(merged["price_p90_base"])) / 2.0
        mid_b = (merged["price_p10_base"] + merged["price_p90_base"]) / 2.0
        half_g = (merged["price_p90_gov"].fillna(merged["price_p90_base"]) - merged["price_p10_gov"].fillna(merged["price_p10_base"])) / 2.0
        half_b = (merged["price_p90_base"] - merged["price_p10_base"]) / 2.0
        q_mid = merged["_w_governed"] * mid_g + (1.0 - merged["_w_governed"]) * mid_b
        q_half = merged["_w_governed"] * half_g + (1.0 - merged["_w_governed"]) * half_b
        merged["price_p10"] = q_mid - q_half
        merged["price_p90"] = q_mid + q_half

    merged = _apply_transition_continuity_guard(merged, governed_horizon_days=governed_horizon_days)

    keep_cols = [c for c in join_cols + cols_keep + ["price_lear", "price_p10", "price_p90"] if c in merged.columns]
    merged = merged[keep_cols].copy()
    for bool_col in ["mlp_used", "fm_used"]:
        if bool_col in merged.columns:
            merged[bool_col] = merged[bool_col].astype(bool)
    return merged


def _overwrite_routed_day_slice(
    base_forecast: pd.DataFrame,
    replacement_forecast: pd.DataFrame,
    day_values: list[int],
    max_jump_eur_mwh: float = 1.5,
) -> pd.DataFrame:
    merged = base_forecast.copy()
    join_cols = ["timestamp"]
    if "hour" in merged.columns and "hour" in replacement_forecast.columns:
        join_cols.append("hour")

    repl = replacement_forecast.copy()
    for col in [c for c in ["price_lear", "price_p10", "price_p90", "n_windows", "mlp_used", "fm_used", "fm_raw_price"] if c in repl.columns]:
        repl = repl.rename(columns={col: f"{col}_repl"})

    merged = merged.merge(
        repl[[c for c in repl.columns if c in join_cols or c.endswith("_repl")]],
        on=join_cols,
        how="left",
    )
    mask = merged["days_ahead"].isin(day_values)
    for col in ["price_lear", "price_p10", "price_p90", "n_windows", "mlp_used", "fm_used", "fm_raw_price"]:
        repl_col = f"{col}_repl"
        if repl_col in merged.columns and col in merged.columns:
            merged.loc[mask, col] = merged.loc[mask, repl_col].fillna(merged.loc[mask, col])

    keep_cols = [c for c in base_forecast.columns if c in merged.columns]
    merged = merged[keep_cols].copy()
    if "days_ahead" in merged.columns and day_values:
        merged = _apply_local_boundary_continuity_guard(
            merged,
            boundary_days=[int(max(day_values))],
            max_jump_eur_mwh=max_jump_eur_mwh,
        )
    return merged


def _apply_transition_continuity_guard(
    forecast: pd.DataFrame,
    governed_horizon_days: int,
    max_jump_eur_mwh: float = 1.5,
) -> pd.DataFrame:
    guarded = forecast.copy()
    if "days_ahead" not in guarded.columns or "price_lear" not in guarded.columns:
        return guarded
    sort_cols = ["days_ahead"]
    if "hour" in guarded.columns:
        group_cols = ["hour"]
    else:
        group_cols = []
    transition_mask = guarded["days_ahead"].between(governed_horizon_days, governed_horizon_days + 2, inclusive="both")
    for _, idx in guarded.loc[transition_mask].groupby(group_cols or (lambda _: 0)).groups.items():
        rows = guarded.loc[list(idx)].sort_values(sort_cols).index.tolist()
        for col in [c for c in ["price_lear", "price_p10", "price_p90"] if c in guarded.columns]:
            for pos in range(1, len(rows)):
                prev_idx = rows[pos - 1]
                curr_idx = rows[pos]
                delta = float(guarded.at[curr_idx, col] - guarded.at[prev_idx, col])
                if abs(delta) > max_jump_eur_mwh:
                    guarded.at[curr_idx, col] = float(
                        guarded.at[prev_idx, col] + np.sign(delta) * max_jump_eur_mwh
                    )
    if all(col in guarded.columns for col in ["price_p10", "price_lear", "price_p90"]):
        guarded["price_p10"] = np.minimum(guarded["price_p10"], guarded["price_lear"])
        guarded["price_p90"] = np.maximum(guarded["price_p90"], guarded["price_lear"])
    return guarded


def _apply_local_boundary_continuity_guard(
    forecast: pd.DataFrame,
    boundary_days: list[int],
    max_jump_eur_mwh: float = 1.5,
) -> pd.DataFrame:
    guarded = forecast.copy()
    if "days_ahead" not in guarded.columns or "price_lear" not in guarded.columns:
        return guarded
    group_cols = ["hour"] if "hour" in guarded.columns else []
    for boundary_day in sorted(set(int(v) for v in boundary_days)):
        boundary_mask = guarded["days_ahead"].isin([boundary_day, boundary_day + 1])
        for _, idx in guarded.loc[boundary_mask].groupby(group_cols or (lambda _: 0)).groups.items():
            rows = guarded.loc[list(idx)].sort_values(["days_ahead"]).index.tolist()
            if len(rows) != 2:
                continue
            prev_idx, curr_idx = rows
            for col in [c for c in ["price_lear", "price_p10", "price_p90"] if c in guarded.columns]:
                delta = float(guarded.at[curr_idx, col] - guarded.at[prev_idx, col])
                if abs(delta) > max_jump_eur_mwh:
                    guarded.at[curr_idx, col] = float(
                        guarded.at[prev_idx, col] + np.sign(delta) * max_jump_eur_mwh
                    )
    if all(col in guarded.columns for col in ["price_p10", "price_lear", "price_p90"]):
        guarded["price_p10"] = np.minimum(guarded["price_p10"], guarded["price_lear"])
        guarded["price_p90"] = np.maximum(guarded["price_p90"], guarded["price_lear"])
    return guarded


def _validate_swiss_short_term_inputs(
    inputs: SwissShortTermInputs,
    logger: logging.Logger,
) -> SwissShortTermInputHealth:
    if inputs.epex_ch.empty:
        raise ValueError("Swiss CT requires non-empty CH EPEX input.")
    if inputs.entso.empty:
        raise ValueError("Swiss CT requires non-empty ENTSO input.")
    if inputs.hydro.empty:
        raise ValueError("Swiss CT requires non-empty hydro input.")
    if inputs.base_pfc_ch.empty:
        raise ValueError("Swiss CT requires non-empty Swiss LT base PFC.")

    ch_hourly_idx = inputs.epex_ch.index.to_series().resample("h").mean().dropna().index
    de_hourly_idx = inputs.epex_de.index.to_series().resample("h").mean().dropna().index if not inputs.epex_de.empty else pd.DatetimeIndex([])
    entso_hourly_idx = inputs.entso.index.to_series().resample("h").mean().dropna().index

    ch_de_overlap_hours = len(ch_hourly_idx.intersection(de_hourly_idx))
    entso_ch_overlap_hours = len(ch_hourly_idx.intersection(entso_hourly_idx))
    has_de_price_support = ch_de_overlap_hours > 0
    neighbor_overlap_hours: dict[str, int] = {}
    for code, df in (inputs.neighbor_prices_15min or {}).items():
        if df is None or df.empty:
            neighbor_overlap_hours[code] = 0
            continue
        nb_hourly_idx = df.index.to_series().resample("h").mean().dropna().index
        neighbor_overlap_hours[code] = len(ch_hourly_idx.intersection(nb_hourly_idx))

    has_required_neighbor_support = all(
        neighbor_overlap_hours.get(code, 0) > 0 for code in inputs.required_neighbor_codes
    )

    if inputs.require_de_exogenous and not has_required_neighbor_support:
        raise ValueError(
            f"Swiss CT is configured to require neighbor exogenous prices {inputs.required_neighbor_codes}, "
            f"but overlap was insufficient: {neighbor_overlap_hours}"
        )

    health = SwissShortTermInputHealth(
        epex_ch_rows=len(inputs.epex_ch),
        epex_de_rows=len(inputs.epex_de),
        entso_rows=len(inputs.entso),
        hydro_rows=len(inputs.hydro),
        outages_rows=0 if inputs.outages_all is None else len(inputs.outages_all),
        commodities_rows=0 if inputs.commodities is None else len(inputs.commodities),
        de_renewable_forecast_rows=0 if inputs.de_renewable_forecast is None else len(inputs.de_renewable_forecast),
        multi_country_forecast_rows=0 if inputs.multi_country_forecast is None else len(inputs.multi_country_forecast),
        fr_nuclear_forecast_rows=0 if inputs.fr_nuclear_forecast is None else len(inputs.fr_nuclear_forecast),
        weather_forecast_rows=0 if inputs.weather_forecast is None else len(inputs.weather_forecast),
        ch_de_overlap_hours=ch_de_overlap_hours,
        entso_ch_overlap_hours=entso_ch_overlap_hours,
        has_de_price_support=has_de_price_support,
        neighbor_overlap_hours=neighbor_overlap_hours,
        has_required_neighbor_support=has_required_neighbor_support,
    )
    logger.info(
        "  Swiss CT input health: CH=%d rows, DE=%d rows, ENTSO=%d rows, hydro=%d rows, DEfc=%d rows, MCFc=%d rows, FRnucFc=%d rows, WX=%d rows, CH/DE overlap=%d h, CH/ENTSO overlap=%d h, neighbors=%s",
        health.epex_ch_rows,
        health.epex_de_rows,
        health.entso_rows,
        health.hydro_rows,
        health.de_renewable_forecast_rows,
        health.multi_country_forecast_rows,
        health.fr_nuclear_forecast_rows,
        health.weather_forecast_rows,
        health.ch_de_overlap_hours,
        health.entso_ch_overlap_hours,
        health.neighbor_overlap_hours,
    )
    if not has_required_neighbor_support:
        logger.warning("  Swiss CT running without full required neighbor price overlap support.")
    return health


def _save_input_health(health: SwissShortTermInputHealth, logger: logging.Logger) -> None:
    path = os.path.join("pfc_shaping", "output", "swiss_ct_input_health_latest.json")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(health), handle, indent=2)
        logger.info("  Swiss CT input health saved: %s", path)
    except Exception as exc:
        logger.warning("  Failed to persist Swiss CT input health: %s", exc)


def _maybe_apply_experimental_pricefm(
    project_root: str,
    lear_forecast: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    if os.getenv("PFC_ENABLE_PRICEFM_EXPERIMENT", "0") != "1":
        return lear_forecast

    try:
        from pfc_shaping.model.futureboost_experimental import (
            DEFAULT_FUTUREBOOST_EXPERIMENT,
            apply_futureboost_experimental,
        )
        from pfc_shaping.model.pricefm_experimental import (
            BEST_PRICEFM_EXPERIMENT,
            blend_lear_with_pricefm,
            load_pricefm_forecast,
        )

        pricefm_path = BEST_PRICEFM_EXPERIMENT.forecast_latest_path
        pricefm_meta_path = os.path.join("pfc_shaping", "output", "pricefm_forecast_latest_meta.json")
        generate_pricefm = os.getenv("PFC_GENERATE_PRICEFM_EXPERIMENT", "1") == "1"
        pricefm_blend_mode = os.getenv("PFC_PRICEFM_BLEND_MODE", "regime").strip().lower()

        if generate_pricefm:
            try:
                pricefm_python = os.getenv("PFC_PRICEFM_PYTHON", DEFAULT_PRICEFM_PYTHON)
                if not os.path.exists(pricefm_python):
                    pricefm_python = sys.executable
                pricefm_cmd = [
                    pricefm_python,
                    os.path.join(project_root, "scripts", "generate_pricefm_experimental_forecast.py"),
                    "--horizon-days",
                    "10",
                ]
                logger.info("  Generating experimental PriceFM forecast...")
                completed = subprocess.run(
                    pricefm_cmd,
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if completed.stdout:
                    logger.info("  PriceFM generation stdout: %s", completed.stdout.strip())
            except Exception as gen_exc:
                if isinstance(gen_exc, subprocess.CalledProcessError) and gen_exc.stderr:
                    logger.warning("  Experimental PriceFM stderr: %s", gen_exc.stderr.strip())
                logger.warning("  Experimental PriceFM generation failed: %s", gen_exc)

        if not os.path.exists(pricefm_path):
            logger.info("  Experimental PriceFM forecast not found at %s", pricefm_path)
            return lear_forecast

        pricefm_forecast = load_pricefm_forecast(pricefm_path)
        if pricefm_blend_mode == "futureboost":
            use_cin = os.getenv("PFC_FUTUREBOOST_USE_CIN", "0") == "1"
            use_qra = os.getenv("PFC_FUTUREBOOST_USE_QRA", "1") == "1"
            futureboost_cfg = replace(
                DEFAULT_FUTUREBOOST_EXPERIMENT,
                use_causal_instance_norm=use_cin,
                use_qra_quantiles=use_qra,
            )
            lear_forecast_pricefm, futureboost_meta = apply_futureboost_experimental(
                lear_forecast,
                pricefm_forecast,
                config=futureboost_cfg,
            )
            logger.info(
                "  Experimental FutureBoost applied: %.0f rows, alpha=%s, train_rows=%d, use_cin=%s, use_qra=%s, qra_enabled=%s",
                lear_forecast_pricefm["futureboost_used"].sum(),
                futureboost_meta.get("alpha"),
                futureboost_meta.get("train_rows"),
                "1" if use_cin else "0",
                "1" if use_qra else "0",
                futureboost_meta.get("qra_enabled"),
            )
        else:
            lear_forecast_pricefm = blend_lear_with_pricefm(
                lear_forecast,
                pricefm_forecast,
                weight_pricefm=None if pricefm_blend_mode == "regime" else BEST_PRICEFM_EXPERIMENT.blend_weight,
            )

        logger.info(
            "  Experimental PriceFM blend applied: %.0f rows, mode=%s",
            (
                lear_forecast_pricefm["futureboost_used"].sum()
                if "futureboost_used" in lear_forecast_pricefm.columns
                else lear_forecast_pricefm["pricefm_used"].sum()
            ),
            pricefm_blend_mode,
        )
        if os.path.exists(pricefm_meta_path):
            logger.info("  Experimental PriceFM metadata: %s", pricefm_meta_path)

        pricefm_base = os.path.join("pfc_shaping", "output", f"lear_forecast_pricefm_experimental_{pd.Timestamp.now().strftime('%Y-%m-%d')}")
        lear_forecast_pricefm.to_parquet(f"{pricefm_base}.parquet", index=False)
        lear_forecast_pricefm.to_csv(f"{pricefm_base}.csv", index=False)
        lear_forecast_pricefm.to_parquet(
            os.path.join("pfc_shaping", "output", "lear_forecast_pricefm_experimental_latest.parquet"),
            index=False,
        )
        lear_forecast_pricefm.to_csv(
            os.path.join("pfc_shaping", "output", "lear_forecast_pricefm_experimental_latest.csv"),
            index=False,
        )

        if os.getenv("PFC_APPLY_PRICEFM_EXPERIMENT_TO_PFC", "0") == "1":
            logger.info("  Experimental PriceFM blend promoted into PFC overlay")
            return lear_forecast_pricefm.copy()
        return lear_forecast
    except Exception as pricefm_exc:
        logger.warning("  Experimental PriceFM blend failed: %s", pricefm_exc)
        return lear_forecast


def _persist_lear_forecast(lear_forecast: pd.DataFrame, logger: logging.Logger) -> str | None:
    lear_run_id = None
    try:
        db_path = init_db(os.path.join("pfc_shaping", "data", "pfc_local.duckdb"))
        lear_run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S_LEAR")
        upsert_lear_forecast(db_path, lear_run_id, lear_forecast)
        logger.info("  LEAR forecast persisted to DuckDB: %s", lear_run_id)
    except Exception as db_exc:
        logger.warning("  LEAR DuckDB persistence failed: %s", db_exc)
    return lear_run_id


def _maybe_run_lear_backtest(lear: Any, lear_run_id: str | None, logger: logging.Logger) -> None:
    backtest_mode = os.getenv("PFC_LEAR_BACKTEST_MODE", "full").strip().lower()
    if backtest_mode not in {"full", "fast", "skip"}:
        logger.warning("  Unknown PFC_LEAR_BACKTEST_MODE=%s, falling back to 'full'", backtest_mode)
        backtest_mode = "full"

    if backtest_mode == "skip":
        logger.info("  LEAR backtest skipped (PFC_LEAR_BACKTEST_MODE=skip)")
        return

    default_horizons = [1, 2, 3, 5, 7, 10] if backtest_mode == "full" else [1, 3, 7]
    default_n_days = 30 if backtest_mode == "full" else 10
    horizons_env = os.getenv("PFC_LEAR_BACKTEST_HORIZONS", "").strip()
    n_days_env = os.getenv("PFC_LEAR_BACKTEST_DAYS", "").strip()
    horizons = [int(x) for x in horizons_env.split(",") if x.strip()] if horizons_env else default_horizons
    n_days = int(n_days_env) if n_days_env else default_n_days

    logger.info(
        "  Running LEAR backtest multi-horizon: mode=%s, n_days=%d, horizons=%s",
        backtest_mode,
        n_days,
        horizons,
    )
    t_bt = time.time()
    try:
        bt_frames = []
        for horizon in horizons:
            bt_h = lear.backtest(n_days=n_days, horizon=horizon)
            bt_h["horizon"] = horizon
            bt_frames.append(bt_h)
        bt = pd.concat(bt_frames, ignore_index=True)
        bt_path = os.path.join("pfc_shaping", "output", f"lear_backtest_{pd.Timestamp.now().strftime('%Y-%m-%d')}.parquet")
        bt.to_parquet(bt_path, index=False)
        bt.to_parquet(os.path.join("pfc_shaping", "output", "lear_backtest_latest.parquet"), index=False)
        summary = (
            bt.groupby("horizon")
            .agg(
                mae=("abs_error", "mean"),
                rmse=("error", lambda series: float(np.sqrt(np.mean(np.square(series))))),
                bias=("error", "mean"),
                corr=("forecast", lambda series: float(series.corr(bt.loc[series.index, "actual"]))),
            )
            .reset_index()
        )
        summary_path = os.path.join("pfc_shaping", "output", f"lear_backtest_summary_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv")
        summary.to_csv(summary_path, index=False)
        summary.to_csv(os.path.join("pfc_shaping", "output", "lear_backtest_summary_latest.csv"), index=False)
        try:
            db_path = init_db(os.path.join("pfc_shaping", "data", "pfc_local.duckdb"))
            if lear_run_id is not None:
                upsert_lear_backtest(db_path, lear_run_id, bt)
                logger.info("  LEAR backtest persisted to DuckDB: %s", lear_run_id)
        except Exception as db_exc:
            logger.warning("  LEAR backtest DuckDB persistence failed: %s", db_exc)
        logger.info("  Backtest saved: %s (%.1fs)", bt_path, time.time() - t_bt)
    except Exception as bt_exc:
        logger.warning("  Backtest failed: %s", bt_exc)
