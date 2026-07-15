"""Integration helpers for the monthly BASE curve solver.

The functions in this module are deliberately thin orchestration glue.  They
turn current EEX quotes into the contract-based monthly solver inputs and then
return the exact ``base_prices`` / ``quoted_keys`` pair that the LT assembler
must consume when the monthly solver is enabled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pfc_shaping.calibration.monthly_curve_config_identity import config_hash
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
from pfc_shaping.data.forward_proxy import (
    ForwardEligibility,
    ForwardSnapshot,
    verify_forward_eligibility,
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
    "min_history_snapshots": 24,
    "history_lookback_years": 6,
    "min_structural_snapshots": 24,
    "allow_template_structural_fallback": True,
    "structural_amplitude_eur_mwh": 110.0,
    "panel_weight": 1.0,
    "history_weight": 0.5,
    "structural_weight": 1.0,
    "constraint_tolerance": 1e-9,
    "quote_conflict_tolerance": 0.01,
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
        min_history_snapshots=int(settings.get("min_history_snapshots", 24)),
        constraint_tolerance=float(settings.get("constraint_tolerance", 1e-9)),
        quote_conflict_tolerance=float(settings.get("quote_conflict_tolerance", 0.01)),
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


def delivery_months_for_local_window(
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    timezone: str = "Europe/Zurich",
) -> pd.PeriodIndex:
    """Return inclusive delivery months for the intended local artifact window."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start.tz is None:
        start = start.tz_localize(timezone)
    else:
        start = start.tz_convert(timezone)
    if end.tz is None:
        end = end.tz_localize(timezone)
    else:
        end = end.tz_convert(timezone)
    if end < start:
        raise ValueError("end_date must be >= start_date")
    first_month = pd.Timestamp(start.year, start.month, 1, tz=timezone)
    last_month = pd.Timestamp(end.year, end.month, 1, tz=timezone)
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


def latest_base_prices_by_market(
    history: pd.DataFrame,
    *,
    market: str,
    as_of_date: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, dict[str, float]]:
    return _latest_base_prices(history, market=market, as_of_date=as_of_date)


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
    own_forward_snapshot: ForwardSnapshot | None = None,
    forward_eligibility: ForwardEligibility | None = None,
    expected_forward_max_age_business_days: int = 1,
    allow_unverified_inputs: bool = False,
) -> MonthlyLevelAuthority:
    """Solve the monthly level and return assembler-ready inputs."""

    cfg_settings = dict(DEFAULT_MONTHLY_SOLVER_CONFIG)
    cfg_settings.update(dict(settings or {}))
    cfg = monthly_curve_config_from_settings(cfg_settings)
    market = str(market).upper()
    if own_forward_snapshot is None and not allow_unverified_inputs:
        raise ValueError(
            "a verified ForwardSnapshot is required; "
            "allow_unverified_inputs is restricted to non-promotional tests and research"
        )
    eex_history = eex_history.copy() if eex_history is not None else pd.DataFrame()
    valuation_ts = _resolve_valuation_timestamp(run_timestamp)
    run_ts = _resolve_run_timestamp(run_timestamp, eex_history)
    quote_provenance = _normalize_quote_provenance(
        own_forward_snapshot.to_manifest() if own_forward_snapshot is not None else None,
        run_timestamp=run_ts,
    )
    provenance_market = str(quote_provenance.get("market") or market).upper()
    if provenance_market != market:
        raise ValueError(
            f"forward provenance market mismatch: expected {market}, got {provenance_market}"
        )
    if own_forward_snapshot is not None and dict(own_base_prices) != dict(own_forward_snapshot.prices):
        raise ValueError("own_base_prices must exactly match the validated ForwardSnapshot prices")
    if (
        own_forward_snapshot is not None
        and original_forward_prices is not None
        and dict(original_forward_prices) != dict(own_forward_snapshot.prices)
    ):
        raise ValueError("original_forward_prices must exactly match the validated ForwardSnapshot prices")
    if own_forward_snapshot is not None:
        assert own_forward_snapshot.snapshot_date is not None
        assert own_forward_snapshot.available_at is not None
        if own_forward_snapshot.snapshot_date > valuation_ts:
            raise ValueError("forward snapshot_date is after the valuation timestamp")
        if own_forward_snapshot.available_at > valuation_ts:
            raise ValueError("forward available_at is after the valuation timestamp")
        if forward_eligibility is None:
            raise ValueError("matching forward_eligibility.v1 evidence is required")
        if not isinstance(forward_eligibility, ForwardEligibility):
            raise TypeError("forward_eligibility must be a verified ForwardEligibility object")
        verify_forward_eligibility(
            own_forward_snapshot,
            forward_eligibility,
            valuation_timestamp=valuation_ts,
            expected_max_age_business_days=int(expected_forward_max_age_business_days),
        )
    quote_source_ids = {
        str(quote["product"]): str(quote["quote_id"])
        for quote in quote_provenance.get("quotes", [])
        if isinstance(quote, Mapping) and quote.get("product") and quote.get("quote_id")
    }
    own_quotes = _quotes_from_prices(
        market=market,
        prices=own_base_prices,
        load_type="BASE",
        snapshot_date=pd.Timestamp(quote_provenance["snapshot_date"]),
        available_at=pd.Timestamp(quote_provenance["available_at"]),
        source=str(quote_provenance["source_kind"]),
        source_by_product=quote_source_ids,
        snapshot_id=str(quote_provenance.get("snapshot_id") or ""),
        observation_id=str(quote_provenance.get("observation_id") or ""),
        source_kind=str(quote_provenance.get("source_kind") or ""),
        source_sha256=str(quote_provenance.get("source_sha256") or ""),
    )
    constraints = build_monthly_constraint_system(
        delivery_months,
        own_quotes,
        timezone=timezone,
        market=market,
        load_type="BASE",
        constraint_tolerance=cfg.constraint_tolerance,
        quote_conflict_tolerance=cfg.quote_conflict_tolerance,
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
                snapshot_date=run_ts,
                available_at=run_ts,
                source="NEIGHBOR_PRIOR_UNVERIFIED",
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
        fused_status=fused.status,
        structural_prior_summary=_prior_diagnostics_summary(structural),
        source_hashes=source_hashes or {},
        quote_provenance=quote_provenance,
        valuation_timestamp=valuation_ts,
        forward_eligibility=(
            forward_eligibility.to_manifest() if forward_eligibility is not None else {}
        ),
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


def active_monthly_curve_config_payload(
    settings: Mapping[str, object],
) -> dict[str, object]:
    """Return the canonical lambda-selection payload consumed by production."""

    return _active_config_payload(settings)


def solve_monthly_level_authority_from_history(
    *,
    forwards_path: str | Path,
    market: str,
    delivery_months: pd.PeriodIndex,
    settings: Mapping[str, object] | None = None,
    timezone: str = "Europe/Zurich",
    original_forward_prices: Mapping[str, float] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> MonthlyLevelAuthority:
    history = pd.read_parquet(forwards_path)
    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None).dt.normalize()
    market = str(market).upper()
    run_timestamp, own = _latest_base_prices(history, market=market, as_of_date=as_of_date)
    settings = dict(DEFAULT_MONTHLY_SOLVER_CONFIG) | dict(settings or {})
    neighbor_markets = tuple(str(m).upper() for m in settings.get("markets", ("DE", "FR", "AT", "IT")))
    neighbors: dict[str, dict[str, float]] = {}
    for neighbor in neighbor_markets:
        try:
            _, prices = _latest_base_prices(history, market=neighbor, as_of_date=run_timestamp)
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
        allow_unverified_inputs=True,
    )


def _latest_base_prices(
    history: pd.DataFrame,
    *,
    market: str,
    as_of_date: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, dict[str, float]]:
    sub = history[
        history["market"].astype(str).str.upper().eq(str(market).upper())
        & history["load_type"].astype(str).str.upper().eq("BASE")
    ].copy()
    if sub.empty:
        raise ValueError(f"no BASE forward history for market={market}")
    if as_of_date is None:
        snapshot = pd.Timestamp(sub["date"].max()).tz_localize(None).normalize()
    else:
        snapshot = pd.Timestamp(as_of_date).tz_localize(None).normalize()
    snap = sub[sub["date"].eq(snapshot)]
    if snap.empty:
        raise ValueError(f"no BASE forward history for market={market} at date={snapshot.date()}")
    return snapshot, dict(zip(snap["product"].astype(str), snap["price"].astype(float)))


def _quotes_from_prices(
    *,
    market: str,
    prices: Mapping[str, float],
    load_type: str,
    snapshot_date: pd.Timestamp,
    available_at: pd.Timestamp,
    source: str,
    source_by_product: Mapping[str, str] | None = None,
    snapshot_id: str = "",
    observation_id: str = "",
    source_kind: str = "",
    source_sha256: str = "",
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
                snapshot_date=snapshot_date,
                source=str((source_by_product or {}).get(str(product), source)),
                available_at=available_at,
                quote_id=str((source_by_product or {}).get(str(product), "")),
                snapshot_id=snapshot_id,
                observation_id=observation_id,
                source_kind=source_kind,
                source_sha256=source_sha256,
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


def _resolve_valuation_timestamp(run_timestamp: pd.Timestamp | None) -> pd.Timestamp:
    timestamp = pd.Timestamp(run_timestamp) if run_timestamp is not None else pd.Timestamp.now("UTC")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _normalize_quote_provenance(
    provenance: Mapping[str, object] | None,
    *,
    run_timestamp: pd.Timestamp,
) -> dict[str, object]:
    if provenance is None:
        fixture_timestamp = pd.Timestamp(run_timestamp).isoformat()
        return {
            "schema_version": "forward_snapshot.v1",
            "market": None,
            "source_kind": "TEST_FIXTURE",
            "source_description": "Implicit non-promotional test fixture",
            "snapshot_date": fixture_timestamp,
            "available_at": fixture_timestamp,
            "source_path": None,
            "source_sha256": None,
            "hard_quote_eligible": False,
            "quote_count": None,
            "promotion_eligible": False,
        }

    payload = dict(provenance)
    source_kind = str(payload.get("source_kind") or "UNKNOWN").upper()
    if source_kind not in {"EEX_XLSX", "EEX_UNC"}:
        raise ValueError(f"forward source kind {source_kind} is forbidden for hard constraints")
    if not bool(payload.get("hard_quote_eligible", False)):
        raise ValueError(f"forward source {source_kind} is not hard-quote eligible")
    for field in ("snapshot_date", "available_at", "source_path", "source_sha256"):
        if not payload.get(field):
            raise ValueError(f"forward provenance field {field} is required")
    source_sha256 = str(payload["source_sha256"]).lower()
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("forward provenance source_sha256 must be a lowercase SHA-256 digest")
    source_path = Path(str(payload["source_path"]))
    if not source_path.is_file():
        raise ValueError(f"forward provenance source_path does not exist: {source_path}")
    if _file_sha256(source_path) != source_sha256:
        raise ValueError("forward provenance source_sha256 does not match source_path bytes")
    snapshot_date = pd.Timestamp(payload["snapshot_date"])
    available_at = pd.Timestamp(payload["available_at"])
    if available_at.tzinfo is None:
        available_at = available_at.tz_localize("UTC")
    else:
        available_at = available_at.tz_convert("UTC")
    if snapshot_date.tzinfo is None:
        snapshot_date = snapshot_date.tz_localize("UTC")
    else:
        snapshot_date = snapshot_date.tz_convert("UTC")
    if available_at.date() < snapshot_date.date():
        raise ValueError("forward source available_at precedes snapshot_date")
    payload.update(
        {
            "schema_version": str(payload.get("schema_version") or "forward_snapshot.v1"),
            "source_kind": source_kind,
            "snapshot_date": snapshot_date.isoformat(),
            "available_at": available_at.isoformat(),
            "source_path": str(Path(str(payload["source_path"])).resolve()),
            "source_sha256": source_sha256,
            "hard_quote_eligible": True,
            "promotion_eligible": False,
        }
    )
    return payload


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
    fused_status: str,
    structural_prior_summary: Mapping[str, object],
    source_hashes: Mapping[str, str],
    quote_provenance: Mapping[str, object],
    valuation_timestamp: pd.Timestamp,
    forward_eligibility: Mapping[str, object],
) -> dict[str, object]:
    monthly_payload = {str(k): round(float(v), 10) for k, v in result.monthly_curve.sort_index().items()}
    active_config_payload = _active_config_payload(settings)
    product_hierarchy_policy = {
        "schema_version": "monthly_quote_hierarchy_policy.v1",
        "priority": ["MONTH", "QUARTER", "CALENDAR"],
        "quote_conflict_tolerance_eur_mwh": float(
            settings.get("quote_conflict_tolerance", 0.01)
        ),
        "hard_repricing_tolerance_eur_mwh": float(
            settings.get("constraint_tolerance", 1e-9)
        ),
        "stationarity_tolerance": float(settings.get("stationarity_tolerance", 1e-7)),
        "residual_lineage_required": True,
    }
    hard_quotes = [
        dict(quote)
        for quote in quote_provenance.get("quotes", [])
        if isinstance(quote, Mapping) and str(quote.get("load_type", "")).upper() == "BASE"
    ]
    constraint_provenance_frame = constraints.rows[
        [
            "constraint_name",
            "product",
            "parent_product",
            "source_quote_ids",
            "lineage_formula",
            "lineage_sha256",
            "lineage_payload",
            "target",
            "hours",
            "n_months",
            "is_residual",
            "load_type",
        ]
    ]
    constraint_provenance_rows = _canonical_frame_records(constraint_provenance_frame)
    return {
        "monthly_curve_schema_version": result.monthly_curve_schema_version,
        "delivery_months": [str(month) for month in result.monthly_curve.sort_index().index],
        "market": market,
        "run_timestamp": str(pd.Timestamp(run_timestamp)),
        "valuation_timestamp": pd.Timestamp(valuation_timestamp).isoformat(),
        "solver_config": dict(settings),
        "solver_config_hash": _sha256_json(dict(settings)),
        "active_config_hash": config_hash(active_config_payload),
        "source_hashes": dict(source_hashes),
        "forward_snapshot": dict(quote_provenance),
        "forward_eligibility": dict(forward_eligibility),
        "forward_snapshot_date": str(pd.Timestamp(quote_provenance["snapshot_date"]).date()),
        "forward_source_kind": str(quote_provenance["source_kind"]),
        "forward_source_sha256": quote_provenance.get("source_sha256"),
        "hard_quote_eligible": bool(quote_provenance.get("hard_quote_eligible", False)),
        "promotion_eligible": bool(
            quote_provenance.get("hard_quote_eligible", False)
            and forward_eligibility.get("status") == "PASS"
            and forward_eligibility.get("requested_role") == "HARD_LEVEL"
        ),
        "active_constraints_hash": _hash_frame(constraints.rows),
        "constraint_provenance_rows": constraint_provenance_rows,
        "constraint_provenance_hash": _sha256_json(constraint_provenance_rows),
        "quote_diagnostics_hash": _hash_frame(constraints.quote_diagnostics),
        "hard_quote_set_hash": _sha256_json(hard_quotes),
        "hard_quotes": hard_quotes,
        "product_hierarchy_policy": product_hierarchy_policy,
        "product_hierarchy_policy_sha256": _sha256_json(product_hierarchy_policy),
        "monthly_solution": monthly_payload,
        "monthly_solution_hash": _sha256_json(monthly_payload),
        "panel_status": panel_status,
        "history_status": history_status,
        "structural_status": structural_status,
        "fused_status": fused_status,
        "structural_prior_summary": dict(structural_prior_summary),
        "solver_kkt": dict(result.kkt),
        "monthly_level_authority": settings.get("monthly_level_authority", "solver"),
        "skip_legacy_level_cascade": bool(settings.get("skip_legacy_level_cascade", True)),
        "skip_legacy_base_smoothing": bool(settings.get("skip_legacy_base_smoothing", True)),
    }


def _active_config_payload(settings: Mapping[str, object]) -> dict[str, object]:
    raw = dict(settings)
    payload = dict(monthly_curve_config_from_settings(raw).__dict__)
    payload.update(
        {
            "markets": sorted(str(m).upper() for m in raw.get("markets", [])),
            "history_lookback_years": raw.get("history_lookback_years"),
            "min_structural_snapshots": raw.get("min_structural_snapshots"),
            "allow_template_structural_fallback": bool(
                raw.get("allow_template_structural_fallback", False)
            ),
            "structural_amplitude_eur_mwh": float(
                raw.get("structural_amplitude_eur_mwh", 110.0)
            ),
            "panel_weight": float(raw.get("panel_weight", 1.0)),
            "history_weight": float(raw.get("history_weight", 0.5)),
            "structural_weight": float(raw.get("structural_weight", 1.0)),
        }
    )
    return payload


def _prior_diagnostics_summary(prior: MonthlyShapePrior) -> dict[str, object]:
    diagnostics = prior.diagnostics.copy()
    summary: dict[str, object] = {
        "status": prior.status,
        "diagnostic_rows": int(len(diagnostics)),
    }
    if diagnostics.empty:
        return summary

    for column in ("source", "fallback_reason"):
        if column in diagnostics.columns:
            values = sorted(
                {
                    str(value)
                    for value in diagnostics[column].dropna().unique().tolist()
                    if str(value) != ""
                }
            )
            summary[f"{column}s"] = values
    for column in ("amplitude_eur_mwh", "max_abs_parent_mean_residual", "n_history"):
        if column in diagnostics.columns:
            numeric = pd.to_numeric(diagnostics[column], errors="coerce")
            if numeric.notna().any():
                summary[f"{column}_min"] = float(numeric.min())
                summary[f"{column}_max"] = float(numeric.max())
    if "zero_mean_parent_space" in diagnostics.columns:
        summary["zero_mean_parent_space_all"] = bool(
            diagnostics["zero_mean_parent_space"].fillna(False).astype(bool).all()
        )
    return summary


def _hash_frame(frame: pd.DataFrame) -> str:
    return _sha256_json(_canonical_frame_records(frame))


def _canonical_frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    payload = frame.copy()
    payload = payload.reindex(sorted(payload.columns), axis=1)
    payload = payload.sort_values(list(payload.columns)).reset_index(drop=True)
    return payload.to_dict(orient="records")


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
