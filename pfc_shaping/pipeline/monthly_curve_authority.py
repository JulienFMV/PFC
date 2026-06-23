"""Integration helpers for the monthly BASE curve solver.

The functions in this module are deliberately thin orchestration glue.  They
turn current EEX quotes into the contract-based monthly solver inputs and then
return the exact ``base_prices`` / ``quoted_keys`` pair that the LT assembler
must consume when the monthly solver is enabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.calibration.monthly_curve_priors import (
    MonthlyShapePrior,
    build_fused_shape_prior,
    build_history_shape_prior,
    build_neighbor_panel_shape_prior,
    build_structural_monthly_shape_prior_from_history,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyConstraintSystem,
    MonthlyCurveConfig,
    MonthlyCurveInputs,
    MonthlyCurveResult,
    build_delivery_grid,
    build_monthly_constraint_system,
    product_periods,
    solve_monthly_forward_curve_from_constraints,
)

DEFAULT_MONTHLY_SOLVER_CONFIG: dict[str, object] = {
    "enabled": False,
    "markets": ["DE", "FR", "AT", "IT"],
    "monthly_level_authority": "solver",
    "skip_legacy_level_cascade": True,
    "skip_legacy_base_smoothing": True,
    "lambda_prior": 1e-6,
    "lambda_smooth_month": 1.0,
    "lambda_smooth_yoy": 0.25,
    "lambda_shape": 1.0,
    "neighbor_shrinkage": 0.5,
    "robust_panel_quantile": 0.5,
    "min_history_snapshots": 24,
    "history_lookback_years": 6,
    "min_structural_snapshots": 24,
    "allow_template_structural_fallback": True,
    "structural_amplitude_eur_mwh": 110.0,
    "panel_weight": 1.0,
    "history_weight": 0.5,
    "structural_weight": 1.0,
    "constraint_tolerance": 1e-9,
    "stationarity_tolerance": 1e-7,
    "eex_history_path": "data/eex_forwards_history.parquet",
}


@dataclass(frozen=True)
class MonthlyLevelAuthority:
    inputs: MonthlyCurveInputs
    constraints: MonthlyConstraintSystem
    result: MonthlyCurveResult
    shape_prior: MonthlyShapePrior
    assembler_base_prices: dict[str, float]
    quoted_keys: set[str]
    synthetic_monthly_keys: set[str]
    manifest: dict[str, object]

    @property
    def monthly_solution_hash(self) -> str:
        return str(self.manifest["monthly_solution_hash"])

    @property
    def active_constraints_hash(self) -> str:
        return str(self.manifest["active_constraints_hash"])


def monthly_solver_settings(config: Mapping[str, object] | None) -> dict[str, object]:
    forwards = dict((config or {}).get("forwards", {}) if isinstance(config, Mapping) else {})
    raw = dict(forwards.get("monthly_curve_solver", {}) or {})
    out = dict(DEFAULT_MONTHLY_SOLVER_CONFIG)
    out.update(raw)
    return out


def monthly_solver_enabled(config: Mapping[str, object] | None, *, market: str = "CH") -> bool:
    settings = monthly_solver_settings(config)
    if not bool(settings.get("enabled", False)):
        return False
    markets = [str(m).upper() for m in settings.get("target_markets", ["CH"])]
    return str(market).upper() in markets


def monthly_curve_config_from_settings(settings: Mapping[str, object]) -> MonthlyCurveConfig:
    return MonthlyCurveConfig(
        lambda_prior=float(settings.get("lambda_prior", 1e-6)),
        lambda_smooth_month=float(settings.get("lambda_smooth_month", 1.0)),
        lambda_smooth_yoy=float(settings.get("lambda_smooth_yoy", 0.25)),
        lambda_shape=float(settings.get("lambda_shape", 1.0)),
        neighbor_shrinkage=float(settings.get("neighbor_shrinkage", 0.5)),
        robust_panel_quantile=float(settings.get("robust_panel_quantile", 0.5)),
        min_history_snapshots=int(settings.get("min_history_snapshots", 24)),
        constraint_tolerance=float(settings.get("constraint_tolerance", 1e-9)),
        stationarity_tolerance=float(settings.get("stationarity_tolerance", 1e-7)),
    )


def delivery_months_for_window(
    *,
    start_date: str | pd.Timestamp,
    horizon_days: int,
    timezone: str = "Europe/Zurich",
) -> pd.PeriodIndex:
    start = pd.Timestamp(start_date)
    if start.tz is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    end_exclusive = start + pd.Timedelta(days=max(int(horizon_days), 1))
    last_included = end_exclusive - pd.Timedelta(nanoseconds=1)
    start_local = start.tz_convert(timezone)
    last_local = last_included.tz_convert(timezone)
    first_month = pd.Timestamp(start_local.year, start_local.month, 1, tz=timezone)
    last_month = pd.Timestamp(last_local.year, last_local.month, 1, tz=timezone)
    month_starts = pd.date_range(first_month, last_month, freq="MS", tz=timezone)
    months = pd.PeriodIndex(month_starts.strftime("%Y-%m"), freq="M")
    return pd.PeriodIndex(sorted(months.unique()), freq="M")


def delivery_months_from_prices(prices: Mapping[str, float]) -> pd.PeriodIndex:
    months: set[pd.Period] = set()
    for product in prices:
        if not _is_base_delivery_product(str(product)):
            continue
        months.update(product_periods(str(product)).tolist())
    if not months:
        raise ValueError("no delivery months can be inferred from forward prices")
    return pd.PeriodIndex(sorted(months), freq="M")


def latest_base_prices_by_market(history: pd.DataFrame, *, market: str) -> tuple[pd.Timestamp, dict[str, float]]:
    return _latest_base_prices(history, market=market)


def solve_monthly_level_authority(
    *,
    market: str,
    delivery_months: pd.PeriodIndex,
    own_base_prices: Mapping[str, float],
    all_market_base_prices: Mapping[str, Mapping[str, float]] | None = None,
    eex_history: pd.DataFrame | None = None,
    run_timestamp: pd.Timestamp | None = None,
    settings: Mapping[str, object] | None = None,
    timezone: str = "Europe/Zurich",
    source_hashes: Mapping[str, str] | None = None,
    original_forward_prices: Mapping[str, float] | None = None,
) -> MonthlyLevelAuthority:
    """Solve the monthly level and return assembler-ready inputs."""

    cfg_settings = dict(DEFAULT_MONTHLY_SOLVER_CONFIG)
    cfg_settings.update(dict(settings or {}))
    cfg = monthly_curve_config_from_settings(cfg_settings)
    market = str(market).upper()
    eex_history = eex_history.copy() if eex_history is not None else pd.DataFrame()
    run_ts = _resolve_run_timestamp(run_timestamp, eex_history)
    own_quotes = _quotes_from_prices(
        market=market,
        prices=own_base_prices,
        load_type="BASE",
        run_timestamp=run_ts,
        source="monthly_solver_current_snapshot",
    )
    constraints = build_monthly_constraint_system(
        delivery_months,
        own_quotes,
        timezone=timezone,
        market=market,
        load_type="BASE",
        constraint_tolerance=cfg.constraint_tolerance,
    )
    neighbor_markets = tuple(str(m).upper() for m in cfg_settings.get("markets", ("DE", "FR", "AT", "IT")))
    neighbor_prices = {
        str(neighbor).upper(): dict(prices)
        for neighbor, prices in dict(all_market_base_prices or {}).items()
        if str(neighbor).upper() in neighbor_markets
    }
    panel = build_neighbor_panel_shape_prior(
        constraints,
        neighbor_prices,
        neighbor_markets=neighbor_markets,
        neighbor_shrinkage=cfg.neighbor_shrinkage,
        run_timestamp=run_ts,
    )
    historical = build_history_shape_prior(
        constraints,
        eex_history,
        market=market,
        load_type="BASE",
        run_timestamp=run_ts,
        min_snapshots=cfg.min_history_snapshots,
        lookback_years=int(cfg_settings.get("history_lookback_years", 6)),
    )
    structural = build_structural_monthly_shape_prior_from_history(
        constraints,
        eex_history,
        market=market,
        load_type="BASE",
        run_timestamp=run_ts,
        min_snapshots=int(cfg_settings.get("min_structural_snapshots", 24)),
        lookback_years=int(cfg_settings.get("history_lookback_years", 6)),
        fallback_to_template=bool(cfg_settings.get("allow_template_structural_fallback", False)),
        fallback_amplitude_eur_mwh=float(cfg_settings.get("structural_amplitude_eur_mwh", 110.0)),
    )
    fused = build_fused_shape_prior(
        constraints,
        panel_prior=panel,
        history_prior=historical,
        structural_prior=structural,
        weights={
            "panel": float(cfg_settings.get("panel_weight", 1.0)),
            "history": float(cfg_settings.get("history_weight", 0.5)),
            "structural": float(cfg_settings.get("structural_weight", 1.0)),
        },
    )
    result = solve_monthly_forward_curve_from_constraints(constraints, config=cfg, shape_prior=fused)
    assembler_base_prices = dict(original_forward_prices or own_base_prices)
    synthetic_monthly_keys: set[str] = set()
    for month, value in result.monthly_curve.items():
        key = str(month)
        assembler_base_prices[key] = float(value)
        synthetic_monthly_keys.add(key)
    quoted_keys = set(str(key) for key in dict(original_forward_prices or own_base_prices))
    inputs = MonthlyCurveInputs(
        delivery_grid=build_delivery_grid(delivery_months, timezone=timezone, calendar=market),
        own_quotes=own_quotes,
        neighbor_quotes=tuple(
            quote
            for neighbor, prices in neighbor_prices.items()
            for quote in _quotes_from_prices(
                market=neighbor,
                prices=prices,
                load_type="BASE",
                run_timestamp=run_ts,
                source="monthly_solver_neighbor_snapshot",
            )
        ),
        eex_history=eex_history,
        run_timestamp=run_ts,
        config=cfg,
        source_hashes=dict(source_hashes or {}),
    )
    manifest = _manifest(
        market=market,
        run_timestamp=run_ts,
        settings=cfg_settings,
        result=result,
        constraints=constraints,
        panel_status=panel.status,
        history_status=historical.status,
        structural_status=structural.status,
        structural_prior_summary=_prior_diagnostics_summary(structural),
        fused_status=fused.status,
        source_hashes=source_hashes or {},
    )
    return MonthlyLevelAuthority(
        inputs=inputs,
        constraints=constraints,
        result=result,
        shape_prior=fused,
        assembler_base_prices=assembler_base_prices,
        quoted_keys=quoted_keys,
        synthetic_monthly_keys=synthetic_monthly_keys,
        manifest=manifest,
    )


def solve_monthly_level_authority_from_history(
    *,
    forwards_path: str | Path,
    market: str,
    delivery_months: pd.PeriodIndex,
    settings: Mapping[str, object] | None = None,
    timezone: str = "Europe/Zurich",
    original_forward_prices: Mapping[str, float] | None = None,
) -> MonthlyLevelAuthority:
    history = pd.read_parquet(forwards_path)
    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None).dt.normalize()
    market = str(market).upper()
    run_timestamp, own = _latest_base_prices(history, market=market)
    settings = dict(DEFAULT_MONTHLY_SOLVER_CONFIG) | dict(settings or {})
    neighbor_markets = tuple(str(m).upper() for m in settings.get("markets", ("DE", "FR", "AT", "IT")))
    neighbors: dict[str, dict[str, float]] = {}
    for neighbor in neighbor_markets:
        try:
            _, prices = _latest_base_prices(history, market=neighbor)
        except ValueError:
            continue
        neighbors[neighbor] = prices
    return solve_monthly_level_authority(
        market=market,
        delivery_months=delivery_months,
        own_base_prices=own,
        all_market_base_prices=neighbors,
        eex_history=history,
        run_timestamp=run_timestamp,
        settings=settings,
        timezone=timezone,
        source_hashes={"forwards_path": _file_sha256(forwards_path)},
        original_forward_prices=original_forward_prices or own,
    )


def _latest_base_prices(history: pd.DataFrame, *, market: str) -> tuple[pd.Timestamp, dict[str, float]]:
    sub = history[
        history["market"].astype(str).str.upper().eq(str(market).upper())
        & history["load_type"].astype(str).str.upper().eq("BASE")
    ].copy()
    if sub.empty:
        raise ValueError(f"no BASE forward history for market={market}")
    latest = pd.Timestamp(sub["date"].max()).tz_localize(None).normalize()
    snap = sub[sub["date"].eq(latest)]
    return latest, dict(zip(snap["product"].astype(str), snap["price"].astype(float)))


def _quotes_from_prices(
    *,
    market: str,
    prices: Mapping[str, float],
    load_type: str,
    run_timestamp: pd.Timestamp,
    source: str,
) -> tuple[MarketQuote, ...]:
    out: list[MarketQuote] = []
    for product, price in sorted(prices.items()):
        if not _is_base_delivery_product(str(product)):
            continue
        out.append(
            MarketQuote(
                market=str(market).upper(),
                product=str(product),
                load_type=str(load_type).upper(),
                price=float(price),
                snapshot_date=run_timestamp,
                source=source,
                available_at=run_timestamp,
            )
        )
    return tuple(out)


def _is_base_delivery_product(product: str) -> bool:
    if "-" in product and product.rsplit("-", 1)[-1].lower() in {"peak", "offpeak"}:
        return False
    try:
        product_periods(product)
    except ValueError:
        return False
    return True


def _resolve_run_timestamp(run_timestamp: pd.Timestamp | None, history: pd.DataFrame) -> pd.Timestamp:
    if run_timestamp is not None:
        return pd.Timestamp(run_timestamp).tz_localize(None).normalize()
    if not history.empty and "date" in history.columns:
        return pd.Timestamp(history["date"].max()).tz_localize(None).normalize()
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def _manifest(
    *,
    market: str,
    run_timestamp: pd.Timestamp,
    settings: Mapping[str, object],
    result: MonthlyCurveResult,
    constraints: MonthlyConstraintSystem,
    panel_status: str,
    history_status: str,
    structural_status: str,
    structural_prior_summary: Mapping[str, object],
    fused_status: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    monthly_payload = {str(k): round(float(v), 10) for k, v in result.monthly_curve.sort_index().items()}
    return {
        "monthly_curve_schema_version": result.monthly_curve_schema_version,
        "market": market,
        "run_timestamp": str(pd.Timestamp(run_timestamp)),
        "solver_config": dict(settings),
        "solver_config_hash": _sha256_json(dict(settings)),
        "active_config_hash": config_hash(settings),
        "source_hashes": dict(source_hashes),
        "forward_snapshot_date": str(pd.Timestamp(run_timestamp).date()),
        "active_constraints_hash": _hash_frame(constraints.rows),
        "monthly_solution_hash": _sha256_json(monthly_payload),
        "panel_status": panel_status,
        "history_status": history_status,
        "structural_status": structural_status,
        "structural_prior_summary": dict(structural_prior_summary),
        "fused_status": fused_status,
        "solver_kkt": dict(result.kkt),
        "monthly_level_authority": settings.get("monthly_level_authority", "solver"),
        "skip_legacy_level_cascade": bool(settings.get("skip_legacy_level_cascade", True)),
        "skip_legacy_base_smoothing": bool(settings.get("skip_legacy_base_smoothing", True)),
    }


def _prior_diagnostics_summary(prior: MonthlyShapePrior) -> dict[str, object]:
    diagnostics = prior.diagnostics.copy()
    summary: dict[str, object] = {
        "status": prior.status,
        "diagnostic_rows": int(len(diagnostics)),
    }
    if diagnostics.empty:
        return summary
    if "source" in diagnostics.columns:
        summary["sources"] = sorted(diagnostics["source"].dropna().astype(str).unique().tolist())
    if "fallback_reason" in diagnostics.columns:
        summary["fallback_reasons"] = sorted(diagnostics["fallback_reason"].dropna().astype(str).unique().tolist())
    for column in ("amplitude_eur_mwh", "max_abs_parent_mean_residual", "n_history"):
        if column not in diagnostics.columns:
            if column == "n_history" and prior.status == "STRUCTURAL_TEMPLATE":
                summary[f"{column}_min"] = 0.0
                summary[f"{column}_max"] = 0.0
            continue
        values = pd.to_numeric(diagnostics[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[f"{column}_min"] = float(values.min())
        summary[f"{column}_max"] = float(values.max())
    if "zero_mean_parent_space" in diagnostics.columns:
        summary["zero_mean_parent_space_all"] = bool(diagnostics["zero_mean_parent_space"].fillna(False).all())
    return summary


def _hash_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return _sha256_json([])
    payload = frame.copy()
    payload = payload.reindex(sorted(payload.columns), axis=1)
    payload = payload.sort_values(list(payload.columns)).reset_index(drop=True)
    return _sha256_json(payload.to_dict(orient="records"))


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
