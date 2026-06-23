"""Numerical audit gates for monthly forward curve candidates.

These gates are evidence generation only.  They never mutate the solved curve.
Shape gates are fail-closed: when calibrated historical thresholds are missing
or undersampled, the row is ``UNSUPPORTED`` rather than ``PASS``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyConstraintSystem,
    month_delivery_hours,
)


THRESHOLD_COLUMNS = [
    "gate_id",
    "metric",
    "market",
    "delivery_bucket",
    "parent_type_pair",
    "lookback_start",
    "lookback_end",
    "n_snapshots",
    "min_required_n",
    "p50",
    "p90",
    "p975",
    "max_observed",
    "regime_filter",
    "status",
]

_SAME_MONTH_METRIC = "same_month_shape_delta_abs_eur_mwh"
_COMPARABLE_BLOCK_METRIC = "comparable_block_shape_delta_abs_eur_mwh"
_CALENDAR_SPREAD_DECOMPOSITION_METRIC = "calendar_spread_decomposition_residual_abs_eur_mwh"


def audit_monthly_curve_shape(
    monthly_curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_pairs: Sequence[tuple[int, int]] | None = None,
    historical_thresholds: pd.DataFrame | None = None,
    neighbor_level_leakage_max_abs: float | None = None,
    repricing_tolerance: float = 1e-8,
    leakage_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return machine-readable monthly shape audit rows.

    ``historical_thresholds`` follows the Phase F threshold schema.  Shape
    gates look up P90/P97.5 by ``gate_id`` and metric name, preferring a
    month-specific ``delivery_bucket`` such as ``month_12`` and then ``all``.
    """

    curve = monthly_curve.reindex(constraints.delivery_grid.months).astype(float)
    if curve.isna().any():
        missing = [str(month) for month in curve[curve.isna()].index]
        raise ValueError(f"monthly curve is missing delivery months: {missing}")
    pairs = list(year_pairs or _adjacent_year_pairs(constraints.delivery_grid.months))
    rows: list[dict[str, object]] = []
    rows.extend(_active_repricing_rows(curve, constraints, repricing_tolerance))
    if neighbor_level_leakage_max_abs is not None:
        rows.append(
            _neighbor_level_leakage_row(
                leakage_max_abs=float(neighbor_level_leakage_max_abs),
                tolerance=float(leakage_tolerance),
            )
        )
    for year_a, year_b in pairs:
        rows.extend(
            _same_month_rows(
                curve,
                constraints,
                year_a=year_a,
                year_b=year_b,
                historical_thresholds=historical_thresholds,
            )
        )
        rows.extend(
            _residual_comparable_block_rows(
                curve,
                constraints,
                year_a=year_a,
                year_b=year_b,
                historical_thresholds=historical_thresholds,
            )
        )
        rows.append(
            _calendar_spread_seasonal_decomposition_row(
                curve,
                constraints,
                year_a=year_a,
                year_b=year_b,
                tolerance=repricing_tolerance,
            )
        )
    rows.append(_monthly_shape_regression_row(rows))
    return pd.DataFrame(rows)


def build_monthly_curve_governance_gates(
    *,
    run_timestamp: pd.Timestamp,
    own_quotes: Sequence[MarketQuote] = (),
    neighbor_quotes: Sequence[MarketQuote] = (),
    eex_history: pd.DataFrame | None = None,
    active_config_hash: str | None = None,
    selected_config_hash: str | None = None,
    production_monthly_solution_hash: str | None = None,
    export_monthly_solution_hash: str | None = None,
    production_active_constraints_hash: str | None = None,
    export_active_constraints_hash: str | None = None,
    require_lambda_artifact: bool = False,
    require_path_parity: bool = False,
) -> pd.DataFrame:
    """Build machine-readable governance gate rows.

    These are metadata/evidence gates rather than curve-shape gates.  They use
    the same Phase F row schema as ``audit_monthly_curve_shape``.
    """

    rows = [
        _point_in_time_data_contract_row(
            run_timestamp=run_timestamp,
            own_quotes=own_quotes,
            neighbor_quotes=neighbor_quotes,
            eex_history=eex_history,
        )
    ]
    if require_lambda_artifact or active_config_hash is not None or selected_config_hash is not None:
        rows.append(
            _lambda_calibration_artifact_row(
                active_config_hash=active_config_hash,
                selected_config_hash=selected_config_hash,
            )
        )
    if (
        require_path_parity
        or production_monthly_solution_hash is not None
        or export_monthly_solution_hash is not None
        or production_active_constraints_hash is not None
        or export_active_constraints_hash is not None
    ):
        rows.append(
            _production_export_path_parity_row(
                production_monthly_solution_hash=production_monthly_solution_hash,
                export_monthly_solution_hash=export_monthly_solution_hash,
                production_active_constraints_hash=production_active_constraints_hash,
                export_active_constraints_hash=export_active_constraints_hash,
            )
        )
    return pd.DataFrame(rows)


def build_monthly_curve_historical_thresholds(
    eex_history: pd.DataFrame,
    *,
    market: str = "CH",
    load_type: str = "BASE",
    run_timestamp: pd.Timestamp | None = None,
    lookback_years: int | None = 6,
    min_required_n: int = 24,
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Build Phase F historical P90/P97.5 threshold rows.

    Thresholds are estimated from traded monthly quotes only.  The function is
    point-in-time: rows after ``run_timestamp`` are excluded before metrics are
    computed.  Insufficient samples emit schema-valid ``UNSUPPORTED`` rows.
    """

    history = _prepare_threshold_history(
        eex_history,
        market=market,
        load_type=load_type,
        run_timestamp=run_timestamp,
        lookback_years=lookback_years,
    )
    if history.empty:
        lookback_start = ""
        lookback_end = ""
        observations = pd.DataFrame(
            columns=["gate_id", "metric", "delivery_bucket", "parent_type_pair", "date", "metric_value"]
        )
    else:
        lookback_start = str(pd.Timestamp(history["date"].min()).date())
        lookback_end = str(pd.Timestamp(history["date"].max()).date())
        observations = _historical_threshold_observations(history, timezone=timezone)

    rows: list[dict[str, object]] = []
    specs = [
        ("same_month_rank_consistency", _SAME_MONTH_METRIC, ("all",)),
        ("residual_vs_implied_comparable_block", _COMPARABLE_BLOCK_METRIC, ("residual|calendar", "quarter|calendar")),
    ]
    buckets = ["all"] + [f"month_{month:02d}" for month in range(1, 13)]
    for gate_id, metric, parent_type_pairs in specs:
        for parent_type_pair in parent_type_pairs:
            for bucket in buckets:
                values = observations[
                    observations["gate_id"].astype(str).eq(gate_id)
                    & observations["metric"].astype(str).eq(metric)
                    & observations["delivery_bucket"].astype(str).eq(bucket)
                    & observations["parent_type_pair"].astype(str).eq(parent_type_pair)
                ]
                rows.append(
                    _threshold_row(
                        values["metric_value"].astype(float) if not values.empty else pd.Series(dtype=float),
                        dates=values["date"] if not values.empty else pd.Series(dtype="datetime64[ns]"),
                        gate_id=gate_id,
                        metric=metric,
                        market=str(market).upper(),
                        delivery_bucket=bucket,
                        parent_type_pair=parent_type_pair,
                        lookback_start=lookback_start,
                        lookback_end=lookback_end,
                        min_required_n=int(min_required_n),
                        regime_filter=_threshold_regime_filter(bucket, parent_type_pair=parent_type_pair),
                    )
                )
    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)


def _active_repricing_rows(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    residuals = constraints.residuals(curve.to_numpy(dtype=float))
    for row in residuals.itertuples(index=False):
        product = str(row.name).split(":", 1)[-1]
        error = float(row.achieved) - float(row.target)
        status = "PASS" if abs(error) <= tolerance else "CRITICAL"
        first_month = _first_month_for_bucket(constraints, product)
        rows.append(
            _gate_row(
                gate_id="hard_monthly_curve_repricing",
                status=status,
                severity="P0" if status == "CRITICAL" else "INFO",
                year=int(first_month.year) if first_month is not None else 0,
                month=None if first_month is None else int(first_month.month),
                product=product,
                parent_block_id=product,
                parent_block_type=_active_product_type(product),
                parent_hours=_parent_hours_for_bucket(constraints, product),
                parent_mean=float(row.target),
                month_price=np.nan,
                month_deviation=np.nan,
                metric_name="active_constraint_abs_error_eur_mwh",
                metric_value=abs(error),
                threshold_warning=tolerance,
                threshold_critical=tolerance,
                threshold_source="hard_constraint",
                n_history=np.nan,
                n_neighbors=np.nan,
                evidence=f"achieved={float(row.achieved):.12g}, target={float(row.target):.12g}",
                remediation_hint="Fix monthly solver constraints before inspecting shape.",
            )
        )
    return rows


def _prepare_threshold_history(
    eex_history: pd.DataFrame,
    *,
    market: str,
    load_type: str,
    run_timestamp: pd.Timestamp | None,
    lookback_years: int | None,
) -> pd.DataFrame:
    required = {"date", "product", "load_type", "market", "price"}
    missing = sorted(required - set(eex_history.columns))
    if missing:
        raise ValueError(f"missing required EEX history columns: {missing}")
    df = eex_history.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["product"] = df["product"].astype(str)
    df["load_type"] = df["load_type"].astype(str).str.upper()
    df["market"] = df["market"].astype(str).str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[
        df["market"].eq(str(market).upper())
        & df["load_type"].eq(str(load_type).upper())
        & df["price"].notna()
    ].copy()
    if run_timestamp is not None:
        run_date = pd.Timestamp(run_timestamp).tz_localize(None).normalize()
        df = df[df["date"] <= run_date]
        if lookback_years is not None:
            df = df[df["date"] >= run_date - pd.DateOffset(years=int(lookback_years))]
    return df.sort_values(["date", "product"]).reset_index(drop=True)


def _historical_threshold_observations(history: pd.DataFrame, *, timezone: str) -> pd.DataFrame:
    same_snapshot_rows: list[dict[str, object]] = []
    deviation_frames: list[pd.DataFrame] = []
    for date, group in history.groupby("date", sort=True):
        prices = {
            str(row.product): float(row.price)
            for row in group.itertuples(index=False)
            if np.isfinite(float(row.price))
        }
        deviations = _historical_month_deviations(prices, timezone=timezone)
        if not deviations.empty:
            deviations = deviations.copy()
            deviations["date"] = pd.Timestamp(date)
            deviation_frames.append(deviations)
        for month, month_devs in deviations.groupby("month", sort=True):
            records = month_devs.to_dict("records")
            for left_pos in range(len(records)):
                for right_pos in range(left_pos + 1, len(records)):
                    left = records[left_pos]
                    right = records[right_pos]
                    same_month_value = abs(float(left["deviation"]) - float(right["deviation"]))
                    parent_type_pair = _parent_type_pair(str(left["parent_type"]), str(right["parent_type"]))
                    if _is_calendar_vs_seasonal_subblock(parent_type_pair):
                        same_snapshot_rows.extend(
                            _observation_rows(
                                date=date,
                                gate_id="residual_vs_implied_comparable_block",
                                metric=_COMPARABLE_BLOCK_METRIC,
                                month=int(month),
                                parent_type_pair=parent_type_pair,
                                metric_value=same_month_value,
                            )
                        )
    rows = _cross_snapshot_same_month_observations(deviation_frames)
    rows.extend(same_snapshot_rows)
    return pd.DataFrame(
        rows,
        columns=["date", "gate_id", "metric", "delivery_bucket", "parent_type_pair", "metric_value"],
    )


def _cross_snapshot_same_month_observations(deviation_frames: list[pd.DataFrame]) -> list[dict[str, object]]:
    if not deviation_frames:
        return []
    deviations = pd.concat(deviation_frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for month, month_devs in deviations.groupby("month", sort=True):
        records = month_devs[["date", "deviation"]].to_dict("records")
        for left_pos in range(len(records)):
            for right_pos in range(left_pos + 1, len(records)):
                left = records[left_pos]
                right = records[right_pos]
                metric_value = abs(float(left["deviation"]) - float(right["deviation"]))
                rows.extend(
                    _observation_rows(
                        date=max(pd.Timestamp(left["date"]), pd.Timestamp(right["date"])),
                        gate_id="same_month_rank_consistency",
                        metric=_SAME_MONTH_METRIC,
                        month=int(month),
                        metric_value=metric_value,
                    )
                )
    return rows


def _historical_month_deviations(prices: dict[str, float], *, timezone: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in _historical_years(prices):
        for month in range(1, 13):
            month_key = f"{int(year)}-{month:02d}"
            if month_key not in prices:
                continue
            parent = _historical_parent_value_for_threshold(
                prices,
                year=int(year),
                month=month,
                timezone=timezone,
            )
            if parent is None:
                continue
            rows.append(
                {
                    "year": int(year),
                    "month": int(month),
                    "parent_type": str(parent["type"]),
                    "parent_product": str(parent["product"]),
                    "deviation": float(prices[month_key]) - float(parent["target"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["year", "month", "parent_type", "parent_product", "deviation"],
    )


def _observation_rows(
    *,
    date: pd.Timestamp,
    gate_id: str,
    metric: str,
    month: int,
    metric_value: float,
    parent_type_pair: str = "all",
) -> list[dict[str, object]]:
    if not np.isfinite(float(metric_value)):
        return []
    return [
        {
            "date": pd.Timestamp(date),
            "gate_id": gate_id,
            "metric": metric,
            "delivery_bucket": "all",
            "parent_type_pair": parent_type_pair,
            "metric_value": abs(float(metric_value)),
        },
        {
            "date": pd.Timestamp(date),
            "gate_id": gate_id,
            "metric": metric,
            "delivery_bucket": f"month_{int(month):02d}",
            "parent_type_pair": parent_type_pair,
            "metric_value": abs(float(metric_value)),
        },
    ]


def _threshold_row(
    values: pd.Series,
    *,
    dates: pd.Series,
    gate_id: str,
    metric: str,
    market: str,
    delivery_bucket: str,
    parent_type_pair: str,
    lookback_start: str,
    lookback_end: str,
    min_required_n: int,
    regime_filter: str,
) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n_snapshots = int(pd.to_datetime(dates).dropna().nunique()) if not dates.empty else 0
    if n_snapshots >= int(min_required_n) and not clean.empty:
        status = "PASS"
        p50 = float(clean.quantile(0.50))
        p90 = float(clean.quantile(0.90))
        p975 = float(clean.quantile(0.975))
        max_observed = float(clean.max())
    else:
        status = "UNSUPPORTED"
        p50 = np.nan
        p90 = np.nan
        p975 = np.nan
        max_observed = np.nan
    return {
        "gate_id": gate_id,
        "metric": metric,
        "market": market,
        "delivery_bucket": delivery_bucket,
        "parent_type_pair": parent_type_pair,
        "lookback_start": lookback_start,
        "lookback_end": lookback_end,
        "n_snapshots": n_snapshots,
        "min_required_n": int(min_required_n),
        "p50": p50,
        "p90": p90,
        "p975": p975,
        "max_observed": max_observed,
        "regime_filter": regime_filter,
        "status": status,
    }


def _threshold_regime_filter(delivery_bucket: str, *, parent_type_pair: str) -> str:
    if delivery_bucket == "all":
        bucket_text = "all_months"
    else:
        bucket_text = f"shape_{delivery_bucket}"
    if parent_type_pair == "all":
        return f"monthly_forward_{bucket_text}"
    return f"monthly_forward_{bucket_text}_{parent_type_pair.replace('|', '_vs_')}"


def _historical_years(prices: dict[str, float]) -> list[int]:
    years: set[int] = set()
    for product in prices:
        text = str(product)
        if len(text) >= 4 and text[:4].isdigit():
            years.add(int(text[:4]))
    return sorted(years)


def _historical_parent_value_for_threshold(
    prices: dict[str, float],
    *,
    year: int,
    month: int,
    timezone: str,
) -> dict[str, object] | None:
    quarter = ((int(month) - 1) // 3) + 1
    quarter_key = f"{int(year)}-Q{quarter}"
    cal_key = str(int(year))
    if quarter_key in prices:
        return {"type": "quarter", "product": quarter_key, "target": float(prices[quarter_key])}
    if int(month) >= 4 and cal_key in prices and f"{int(year)}-Q1" in prices:
        target = _historical_apr_dec_residual_target(
            year=int(year),
            cal_price=float(prices[cal_key]),
            q1_price=float(prices[f"{int(year)}-Q1"]),
            timezone=timezone,
        )
        return {"type": "residual", "product": f"{int(year)}-RESIDUAL", "target": target}
    if cal_key in prices:
        return {"type": "calendar", "product": cal_key, "target": float(prices[cal_key])}
    return None


def _historical_apr_dec_residual_target(
    *,
    year: int,
    cal_price: float,
    q1_price: float,
    timezone: str,
) -> float:
    months = pd.period_range(f"{int(year)}-01", f"{int(year)}-12", freq="M")
    hours = month_delivery_hours(months, timezone=timezone)
    q1 = pd.period_range(f"{int(year)}-01", f"{int(year)}-03", freq="M")
    q1_hours = float(hours.loc[q1].sum())
    cal_hours = float(hours.sum())
    residual_hours = cal_hours - q1_hours
    return float((float(cal_price) * cal_hours - float(q1_price) * q1_hours) / residual_hours)


def _neighbor_level_leakage_row(
    *,
    leakage_max_abs: float,
    tolerance: float,
) -> dict[str, object]:
    status = "PASS" if abs(float(leakage_max_abs)) <= float(tolerance) else "CRITICAL"
    return _gate_row(
        gate_id="neighbor_level_leakage",
        status=status,
        severity="P0" if status == "CRITICAL" else "INFO",
        year=0,
        month=None,
        product="neighbor_shift_invariance",
        parent_block_id="all",
        parent_block_type="all",
        parent_hours=np.nan,
        parent_mean=np.nan,
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name="neighbor_shift_solution_delta_max_abs_eur_mwh",
        metric_value=float(abs(leakage_max_abs)),
        threshold_warning=float(tolerance),
        threshold_critical=float(tolerance),
        threshold_source="hard_invariance",
        n_history=np.nan,
        n_neighbors=np.nan,
        evidence=f"max_abs_solution_delta={float(leakage_max_abs):.12g}",
        remediation_hint="External market levels leaked into CH monthly solution; recenter neighbor priors.",
    )


def _point_in_time_data_contract_row(
    *,
    run_timestamp: pd.Timestamp,
    own_quotes: Sequence[MarketQuote],
    neighbor_quotes: Sequence[MarketQuote],
    eex_history: pd.DataFrame | None,
) -> dict[str, object]:
    run_ts = pd.Timestamp(run_timestamp).tz_localize(None)
    checked = 0
    violations: list[str] = []
    for quote in list(own_quotes) + list(neighbor_quotes):
        checked += 1
        available_at = getattr(quote, "available_at", None) or getattr(quote, "snapshot_date", None)
        if available_at is None:
            violations.append(f"{quote.market}:{quote.load_type}:{quote.product}:missing_available_at")
            continue
        available_ts = pd.Timestamp(available_at).tz_localize(None)
        if available_ts > run_ts:
            violations.append(f"{quote.market}:{quote.load_type}:{quote.product}:{available_ts}")
    if eex_history is not None and not eex_history.empty:
        date_col = "available_at" if "available_at" in eex_history.columns else "date" if "date" in eex_history.columns else ""
        if date_col:
            dates = pd.to_datetime(eex_history[date_col], errors="coerce").dt.tz_localize(None)
            checked += int(dates.notna().sum())
            future = eex_history[dates > run_ts]
            for row in future.head(5).itertuples(index=False):
                market = getattr(row, "market", "")
                load_type = getattr(row, "load_type", "")
                product = getattr(row, "product", "")
                date_value = getattr(row, date_col)
                violations.append(f"{market}:{load_type}:{product}:{date_value}")
        else:
            violations.append("eex_history:missing_date_or_available_at")
    if checked == 0:
        status = "UNSUPPORTED"
        severity = "P1"
        evidence = "no point-in-time inputs supplied"
    elif violations:
        status = "CRITICAL"
        severity = "P0"
        evidence = f"future_or_unverifiable_inputs={violations[:5]}, checked={checked}"
    else:
        status = "PASS"
        severity = "INFO"
        evidence = f"all inputs available_at <= run_timestamp; checked={checked}"
    return _gate_row(
        gate_id="point_in_time_data_contract",
        status=status,
        severity=severity,
        year=int(run_ts.year),
        month=int(run_ts.month),
        product="monthly_curve_inputs",
        parent_block_id="input_contract",
        parent_block_type="governance",
        parent_hours=np.nan,
        parent_mean=np.nan,
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name="future_input_count",
        metric_value=float(len(violations)),
        threshold_warning=0.0,
        threshold_critical=0.0,
        threshold_source="hard_point_in_time_contract",
        n_history=float(checked),
        n_neighbors=np.nan,
        evidence=evidence,
        remediation_hint="Remove or mask inputs with available_at after run_timestamp.",
    )


def _lambda_calibration_artifact_row(
    *,
    active_config_hash: str | None,
    selected_config_hash: str | None,
) -> dict[str, object]:
    active = str(active_config_hash or "")
    selected = str(selected_config_hash or "")
    if not active or not selected:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = f"missing active_config_hash or selected_config_hash; active={active}, selected={selected}"
    elif active != selected:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = f"active_config_hash={active} selected_config_hash={selected}"
    else:
        status = "PASS"
        severity = "INFO"
        metric_value = 0.0
        evidence = f"active_config_hash matches selected_config_hash={active}"
    return _gate_row(
        gate_id="lambda_calibration_artifact_present",
        status=status,
        severity=severity,
        year=0,
        month=None,
        product="monthly_curve_selected_config",
        parent_block_id="lambda_config",
        parent_block_type="governance",
        parent_hours=np.nan,
        parent_mean=np.nan,
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name="selected_config_hash_mismatch",
        metric_value=metric_value,
        threshold_warning=0.0,
        threshold_critical=0.0,
        threshold_source="selected_config_artifact",
        n_history=np.nan,
        n_neighbors=np.nan,
        evidence=evidence,
        remediation_hint="Regenerate lambda calibration artifact or align active monthly solver config.",
    )


def _production_export_path_parity_row(
    *,
    production_monthly_solution_hash: str | None,
    export_monthly_solution_hash: str | None,
    production_active_constraints_hash: str | None,
    export_active_constraints_hash: str | None,
) -> dict[str, object]:
    prod_solution = str(production_monthly_solution_hash or "")
    export_solution = str(export_monthly_solution_hash or "")
    prod_constraints = str(production_active_constraints_hash or "")
    export_constraints = str(export_active_constraints_hash or "")
    missing = [
        name
        for name, value in {
            "production_monthly_solution_hash": prod_solution,
            "export_monthly_solution_hash": export_solution,
            "production_active_constraints_hash": prod_constraints,
            "export_active_constraints_hash": export_constraints,
        }.items()
        if not value
    ]
    mismatch = (
        bool(prod_solution and export_solution and prod_solution != export_solution)
        or bool(prod_constraints and export_constraints and prod_constraints != export_constraints)
    )
    if missing:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = f"missing parity hashes: {missing}"
    elif mismatch:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = (
            f"production_solution={prod_solution}, export_solution={export_solution}, "
            f"production_constraints={prod_constraints}, export_constraints={export_constraints}"
        )
    else:
        status = "PASS"
        severity = "INFO"
        metric_value = 0.0
        evidence = "production/export monthly_solution_hash and active_constraints_hash match"
    return _gate_row(
        gate_id="production_export_path_parity",
        status=status,
        severity=severity,
        year=0,
        month=None,
        product="monthly_level_authority",
        parent_block_id="prod_export_paths",
        parent_block_type="governance",
        parent_hours=np.nan,
        parent_mean=np.nan,
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name="prod_export_hash_mismatch",
        metric_value=metric_value,
        threshold_warning=0.0,
        threshold_critical=0.0,
        threshold_source="monthly_authority_hash_parity",
        n_history=np.nan,
        n_neighbors=np.nan,
        evidence=evidence,
        remediation_hint="Route production and local export through the same MonthlyCurveInputs and solver config.",
    )


def _same_month_rows(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_a: int,
    year_b: int,
    historical_thresholds: pd.DataFrame | None,
) -> list[dict[str, object]]:
    cal_a = _quote_target(constraints, str(year_a))
    cal_b = _quote_target(constraints, str(year_b))
    if cal_a is None or cal_b is None:
        return []
    calendar_spread = float(cal_a) - float(cal_b)
    rows: list[dict[str, object]] = []
    for month_number in range(1, 13):
        month_a = pd.Period(f"{year_a}-{month_number:02d}", freq="M")
        month_b = pd.Period(f"{year_b}-{month_number:02d}", freq="M")
        if month_a not in curve.index or month_b not in curve.index:
            continue
        price_a = float(curve.loc[month_a])
        price_b = float(curve.loc[month_b])
        parent_a = _parent_info(constraints, month_a)
        parent_b = _parent_info(constraints, month_b)
        if parent_a is None or parent_b is None:
            continue
        parent_spread = parent_a["target"] - parent_b["target"]
        month_spread = price_a - price_b
        shape_delta = (price_a - parent_a["target"]) - (price_b - parent_b["target"])
        direct_support = parent_a["type"] == "month" or parent_b["type"] == "month"
        metric = abs(float(shape_delta))
        threshold = _threshold_lookup(
            historical_thresholds,
            gate_id="same_month_rank_consistency",
            metric_name="same_month_shape_delta_abs_eur_mwh",
            month=month_number,
        )
        status, severity, threshold_source, warning, critical, n_history, reason = _status_from_threshold(
            metric_value=metric,
            threshold=threshold,
            direct_quote_support=direct_support,
            quote_support_reason="direct monthly quote support",
        )
        sign_reason = _sign_reason(
            calendar_spread=calendar_spread,
            parent_spread=parent_spread,
            month_spread=month_spread,
        )
        rows.append(
            _gate_row(
                gate_id="same_month_rank_consistency",
                status=status,
                severity=severity,
                year=year_a,
                month=month_number,
                product=f"{year_a}-{month_number:02d}_vs_{year_b}-{month_number:02d}",
                parent_block_id=f"{parent_a['bucket']}|{parent_b['bucket']}",
                parent_block_type=f"{parent_a['type']}|{parent_b['type']}",
                parent_hours=parent_a["hours"],
                parent_mean=parent_a["target"],
                month_price=price_a,
                month_deviation=price_a - parent_a["target"],
                metric_name="same_month_shape_delta_abs_eur_mwh",
                metric_value=metric,
                threshold_warning=warning,
                threshold_critical=critical,
                threshold_source=threshold_source,
                n_history=n_history,
                n_neighbors=np.nan,
                evidence=(
                    f"calendar_spread={calendar_spread:.4f}, "
                    f"parent_spread={parent_spread:.4f}, month_spread={month_spread:.4f}, "
                    f"shape_delta={shape_delta:.4f}, {sign_reason}, {reason}"
                ),
                remediation_hint="Review comparable-block shape prior or active quote support.",
                year_b=year_b,
                price_b=price_b,
                parent_mean_b=parent_b["target"],
                parent_hours_b=parent_b["hours"],
                parent_mix_adjustment=parent_spread - calendar_spread,
                expected_sign=float(np.sign(parent_spread)),
                actual_sign=float(np.sign(month_spread)),
                z_score_or_quantile=np.nan,
                quote_support_type="DIRECT_MONTH" if direct_support else "NONE",
                supporting_quote_keys="|".join(parent_a["source_quote_keys"] + parent_b["source_quote_keys"])
                if direct_support
                else "",
                quote_support_value=month_spread if direct_support else np.nan,
                quote_support_rule="active monthly hard constraint" if direct_support else "none",
            )
        )
    return rows


def _residual_comparable_block_rows(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_a: int,
    year_b: int,
    historical_thresholds: pd.DataFrame | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month_number in range(1, 12 + 1):
        month_a = pd.Period(f"{year_a}-{month_number:02d}", freq="M")
        month_b = pd.Period(f"{year_b}-{month_number:02d}", freq="M")
        if month_a not in curve.index or month_b not in curve.index:
            continue
        parent_a = _parent_info(constraints, month_a)
        parent_b = _parent_info(constraints, month_b)
        if parent_a is None or parent_b is None:
            continue
        parent_type_pair = _parent_type_pair(str(parent_a["type"]), str(parent_b["type"]))
        if not _is_calendar_vs_seasonal_subblock(parent_type_pair):
            continue
        price_a = float(curve.loc[month_a])
        price_b = float(curve.loc[month_b])
        dev_a = price_a - float(parent_a["target"])
        dev_b = price_b - float(parent_b["target"])
        metric = abs(float(dev_a - dev_b))
        threshold = _threshold_lookup(
            historical_thresholds,
            gate_id="residual_vs_implied_comparable_block",
            metric_name="comparable_block_shape_delta_abs_eur_mwh",
            month=month_number,
            parent_type_pair=parent_type_pair,
        )
        status, severity, threshold_source, warning, critical, n_history, reason = _status_from_threshold(
            metric_value=metric,
            threshold=threshold,
            direct_quote_support=False,
            quote_support_reason="",
        )
        parent_spread = float(parent_a["target"]) - float(parent_b["target"])
        rows.append(
            _gate_row(
                gate_id="residual_vs_implied_comparable_block",
                status=status,
                severity=severity,
                year=year_a,
                month=month_number,
                product=f"{year_a}-{month_number:02d}_vs_{year_b}-{month_number:02d}",
                parent_block_id=f"{parent_a['bucket']}|{parent_b['bucket']}",
                parent_block_type=f"{parent_a['type']}|{parent_b['type']}",
                parent_hours=parent_a["hours"],
                parent_mean=float(parent_a["target"]),
                month_price=price_a,
                month_deviation=dev_a,
                metric_name="comparable_block_shape_delta_abs_eur_mwh",
                metric_value=metric,
                threshold_warning=warning,
                threshold_critical=critical,
                threshold_source=threshold_source,
                n_history=n_history,
                n_neighbors=np.nan,
                evidence=(
                    f"parent_block_a={parent_a['bucket']}, parent_block_b={parent_b['bucket']}, "
                    f"parent_spread={parent_spread:.4f}, dev_a={dev_a:.4f}, dev_b={dev_b:.4f}, "
                    f"comparable_parent_types={parent_type_pair}, {reason}"
                ),
                remediation_hint=(
                    "Compare same-month deviations from comparable parent blocks; do not compare a quoted "
                    "seasonal sub-block level directly with a full calendar level."
                ),
                year_b=year_b,
                price_b=price_b,
                parent_mean_b=float(parent_b["target"]),
                parent_hours_b=parent_b["hours"],
                parent_mix_adjustment=parent_spread - _calendar_spread(constraints, year_a, year_b),
                parent_type_pair=parent_type_pair,
            )
        )
    return rows


def _calendar_spread_seasonal_decomposition_row(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_a: int,
    year_b: int,
    tolerance: float,
) -> dict[str, object]:
    cal_a = _quote_target(constraints, str(year_a))
    cal_b = _quote_target(constraints, str(year_b))
    calendar_spread = np.nan if cal_a is None or cal_b is None else float(cal_a) - float(cal_b)
    months_a = pd.period_range(f"{year_a}-01", f"{year_a}-12", freq="M")
    months_b = pd.period_range(f"{year_b}-01", f"{year_b}-12", freq="M")
    missing = [str(month) for month in list(months_a) + list(months_b) if month not in curve.index]
    if cal_a is None or cal_b is None or missing:
        reason = "calendar quote missing" if cal_a is None or cal_b is None else f"missing months={missing}"
        return _gate_row(
            gate_id="calendar_spread_seasonal_decomposition",
            status="UNSUPPORTED",
            severity="P2",
            year=year_a,
            month=None,
            product=f"{year_a}_vs_{year_b}",
            parent_block_id=f"{year_a}|{year_b}",
            parent_block_type="calendar_spread_decomposition",
            parent_hours=np.nan,
            parent_mean=float(cal_a) if cal_a is not None else np.nan,
            month_price=np.nan,
            month_deviation=np.nan,
            metric_name=_CALENDAR_SPREAD_DECOMPOSITION_METRIC,
            metric_value=np.nan,
            threshold_warning=float(tolerance),
            threshold_critical=float(tolerance),
            threshold_source="hard_calendar_spread_identity",
            n_history=np.nan,
            n_neighbors=np.nan,
            evidence=reason,
            remediation_hint="Provide calendar targets and a complete monthly curve for both years.",
            year_b=year_b,
            calendar_spread=calendar_spread,
            weighted_month_spread=np.nan,
        )

    weights_a = month_delivery_hours(months_a, timezone=constraints.delivery_grid.timezone).to_numpy()
    weights_b = month_delivery_hours(months_b, timezone=constraints.delivery_grid.timezone).to_numpy()
    curve_mean_a = float(np.average(curve.loc[months_a].astype(float).to_numpy(), weights=weights_a))
    curve_mean_b = float(np.average(curve.loc[months_b].astype(float).to_numpy(), weights=weights_b))
    weighted_month_spread = curve_mean_a - curve_mean_b
    residual = weighted_month_spread - float(calendar_spread)
    metric = abs(float(residual))
    status = "PASS" if metric <= float(tolerance) else "CRITICAL"
    severity = "INFO" if status == "PASS" else "P1"
    return _gate_row(
        gate_id="calendar_spread_seasonal_decomposition",
        status=status,
        severity=severity,
        year=year_a,
        month=None,
        product=f"{year_a}_vs_{year_b}",
        parent_block_id=f"{year_a}|{year_b}",
        parent_block_type="calendar_spread_decomposition",
        parent_hours=float(weights_a.sum()),
        parent_mean=float(cal_a),
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name=_CALENDAR_SPREAD_DECOMPOSITION_METRIC,
        metric_value=metric,
        threshold_warning=float(tolerance),
        threshold_critical=float(tolerance),
        threshold_source="hard_calendar_spread_identity",
        n_history=np.nan,
        n_neighbors=np.nan,
        evidence=(
            f"calendar_spread={float(calendar_spread):.10f}, "
            f"weighted_month_spread={weighted_month_spread:.10f}, residual={residual:.3e}"
        ),
        remediation_hint="Investigate calendar repricing or downstream monthly level mutation.",
        year_b=year_b,
        parent_hours_b=float(weights_b.sum()),
        parent_mean_b=float(cal_b),
        calendar_spread=float(calendar_spread),
        weighted_month_spread=weighted_month_spread,
        calendar_curve_mean_a=curve_mean_a,
        calendar_curve_mean_b=curve_mean_b,
        calendar_target_a=float(cal_a),
        calendar_target_b=float(cal_b),
        parent_mix_adjustment=0.0,
    )


def _monthly_shape_regression_row(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    targeted = [
        row
        for row in rows
        if row.get("gate_id")
        in {
            "same_month_rank_consistency",
            "residual_vs_implied_comparable_block",
            "calendar_spread_seasonal_decomposition",
        }
        and 2028 <= _safe_int(row.get("year"), default=0) <= 2030
    ]
    statuses = [str(row.get("status", "")) for row in targeted]
    if any(status == "CRITICAL" for status in statuses):
        status = "CRITICAL"
        severity = "P1"
        reason = "at least one targeted 2028-2030 monthly shape gate is CRITICAL"
    elif not targeted or any(status == "UNSUPPORTED" for status in statuses):
        status = "UNSUPPORTED"
        severity = "P2"
        reason = "targeted 2028-2030 monthly shape evidence is missing or undersampled"
    elif any(status == "WARNING" for status in statuses):
        status = "WARNING"
        severity = "P2"
        reason = "at least one targeted 2028-2030 monthly shape gate is WARNING"
    else:
        status = "PASS"
        severity = "INFO"
        reason = "targeted 2028-2030 monthly shape gates passed"
    return _gate_row(
        gate_id="monthly_shape_regression_2028_2030",
        status=status,
        severity=severity,
        year=2028,
        month=None,
        product="2028_2030_focus_population",
        parent_block_id="2028_2030",
        parent_block_type="focus_population",
        parent_hours=np.nan,
        parent_mean=np.nan,
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name="targeted_gate_max_status",
        metric_value=float(_status_rank(status)),
        threshold_warning=float(_status_rank("WARNING")),
        threshold_critical=float(_status_rank("CRITICAL")),
        threshold_source="aggregate_targeted_gates",
        n_history=np.nan,
        n_neighbors=np.nan,
        evidence=f"{reason}; counts={dict(pd.Series(statuses, dtype=object).value_counts()) if statuses else {}}",
        remediation_hint="Investigate targeted same-month and comparable-block gate rows.",
    )


def _parent_info(constraints: MonthlyConstraintSystem, month: pd.Period) -> dict[str, object] | None:
    bucket = constraints.month_buckets.loc[month]
    if pd.isna(bucket):
        return None
    bucket_text = str(bucket)
    row = constraints.rows[constraints.rows["product"].astype(str).eq(bucket_text)]
    parent_product = bucket_text if row.empty else str(row["parent_product"].iloc[0])
    source_quote_keys: tuple[str, ...] = tuple()
    hours = np.nan
    if not row.empty:
        hours = float(row["hours"].iloc[0]) if "hours" in row else np.nan
        raw_sources = str(row["source_quote_keys"].iloc[0]) if "source_quote_keys" in row else ""
        source_quote_keys = tuple(part for part in raw_sources.split("|") if part)
    return {
        "bucket": bucket_text,
        "product": parent_product,
        "type": _parent_type(bucket_text, parent_product),
        "target": float(constraints.bucket_targets[bucket_text]),
        "hours": hours,
        "source_quote_keys": source_quote_keys,
    }


def _parent_hours_for_bucket(constraints: MonthlyConstraintSystem, bucket: str) -> float:
    row = constraints.rows[constraints.rows["product"].astype(str).eq(str(bucket))]
    if row.empty or "hours" not in row:
        return np.nan
    return float(row["hours"].iloc[0])


def _parent_type(bucket: str, parent_product: str) -> str:
    if bucket.endswith("-RESIDUAL"):
        return "residual"
    if "-Q" in parent_product:
        return "quarter"
    if parent_product.isdigit():
        return "calendar"
    if len(parent_product) == 7 and parent_product[4] == "-":
        return "month"
    return "unknown"


def _first_month_for_bucket(constraints: MonthlyConstraintSystem, bucket: str) -> pd.Period | None:
    mask = constraints.month_buckets.astype(str).eq(str(bucket))
    if not bool(mask.any()):
        return None
    return constraints.month_buckets.index[mask][0]


def _active_product_type(product: str) -> str:
    if product.endswith("-RESIDUAL"):
        return "residual"
    if "-Q" in product:
        return "quarter"
    if product.isdigit():
        return "calendar"
    if len(product) == 7 and product[4] == "-":
        return "month"
    return "unknown"


def _quote_target(constraints: MonthlyConstraintSystem, product: str) -> float | None:
    diag = constraints.quote_diagnostics
    if diag.empty:
        return None
    row = diag[diag["product"].astype(str).eq(str(product))]
    if row.empty:
        return None
    return float(row["target"].iloc[0])


def _calendar_spread(constraints: MonthlyConstraintSystem, year_a: int, year_b: int) -> float:
    cal_a = _quote_target(constraints, str(year_a))
    cal_b = _quote_target(constraints, str(year_b))
    if cal_a is None or cal_b is None:
        return np.nan
    return float(cal_a) - float(cal_b)


def _parent_type_pair(left: str, right: str) -> str:
    types = {str(left), str(right)}
    if "calendar" in types:
        if "quarter" in types:
            return "quarter|calendar"
        if "residual" in types:
            return "residual|calendar"
    return "|".join(sorted(types))


def _is_calendar_vs_seasonal_subblock(parent_type_pair: str) -> bool:
    return str(parent_type_pair) in {"residual|calendar", "quarter|calendar"}


def _adjacent_year_pairs(months: pd.PeriodIndex) -> list[tuple[int, int]]:
    years = sorted({int(month.year) for month in months})
    return list(zip(years, years[1:]))


def _gate_row(
    *,
    gate_id: str,
    status: str,
    severity: str,
    year: int,
    month: int | None,
    product: str,
    parent_block_id: str,
    parent_block_type: str,
    parent_hours: float,
    parent_mean: float,
    month_price: float,
    month_deviation: float,
    metric_name: str,
    metric_value: float,
    threshold_warning: float,
    threshold_critical: float,
    threshold_source: str,
    n_history: float,
    n_neighbors: float,
    evidence: str,
    remediation_hint: str,
    **extra: object,
) -> dict[str, object]:
    row = {
        "gate_id": gate_id,
        "status": status,
        "severity": severity,
        "market": "CH",
        "load_type": "BASE",
        "year": year,
        "month": month,
        "product": product,
        "parent_block_id": parent_block_id,
        "parent_block_type": parent_block_type,
        "parent_hours": parent_hours,
        "parent_mean": parent_mean,
        "month_price": month_price,
        "month_deviation_from_parent": month_deviation,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold_warning": threshold_warning,
        "threshold_critical": threshold_critical,
        "threshold_source": threshold_source,
        "n_history": n_history,
        "n_neighbors": n_neighbors,
        "evidence": evidence,
        "remediation_hint": remediation_hint,
    }
    row.update(extra)
    return row


def _threshold_lookup(
    thresholds: pd.DataFrame | None,
    *,
    gate_id: str,
    metric_name: str,
    month: int,
    parent_type_pair: str | None = None,
) -> dict[str, object] | None:
    if thresholds is None or thresholds.empty:
        return None
    required = {"gate_id", "metric", "market", "p90", "p975", "n_snapshots", "min_required_n", "status"}
    if not required <= set(thresholds.columns):
        return None
    frame = thresholds[
        thresholds["gate_id"].astype(str).eq(gate_id)
        & thresholds["metric"].astype(str).eq(metric_name)
        & thresholds["market"].astype(str).str.upper().eq("CH")
    ].copy()
    if frame.empty:
        return None
    if parent_type_pair is not None and "parent_type_pair" in frame.columns:
        frame = frame[frame["parent_type_pair"].astype(str).eq(str(parent_type_pair))]
        if frame.empty:
            return None
    delivery_bucket = frame.get("delivery_bucket", pd.Series("", index=frame.index)).astype(str)
    specific = frame[delivery_bucket.eq(f"month_{int(month):02d}")]
    if not specific.empty:
        return specific.iloc[0].to_dict()
    fallback = frame[delivery_bucket.eq("all")]
    if not fallback.empty:
        return fallback.iloc[0].to_dict()
    return None


def _status_from_threshold(
    *,
    metric_value: float,
    threshold: dict[str, object] | None,
    direct_quote_support: bool,
    quote_support_reason: str,
) -> tuple[str, str, str, float, float, float, str]:
    if direct_quote_support:
        return ("PASS", "INFO", "active_quote_support", np.nan, np.nan, np.nan, quote_support_reason)
    if threshold is None:
        return (
            "UNSUPPORTED",
            "P2",
            "missing_historical_threshold",
            np.nan,
            np.nan,
            0.0,
            "historical threshold row missing",
        )
    n_history = float(pd.to_numeric(pd.Series([threshold.get("n_snapshots")]), errors="coerce").iloc[0])
    min_required = float(pd.to_numeric(pd.Series([threshold.get("min_required_n")]), errors="coerce").iloc[0])
    warning = float(pd.to_numeric(pd.Series([threshold.get("p90")]), errors="coerce").iloc[0])
    critical = float(pd.to_numeric(pd.Series([threshold.get("p975")]), errors="coerce").iloc[0])
    threshold_status = str(threshold.get("status", "")).upper()
    if (
        not np.isfinite(n_history)
        or not np.isfinite(min_required)
        or n_history < min_required
        or threshold_status not in {"PASS", "OK", "USED"}
        or not np.isfinite(warning)
        or not np.isfinite(critical)
    ):
        return (
            "UNSUPPORTED",
            "P2",
            "insufficient_historical_threshold",
            warning,
            critical,
            n_history if np.isfinite(n_history) else 0.0,
            f"insufficient historical sample n={n_history}, min_required={min_required}",
        )
    value = abs(float(metric_value))
    if value > critical:
        return ("CRITICAL", "P1", "historical_p90_p975", warning, critical, n_history, "above historical P97.5")
    if value > warning:
        return ("WARNING", "P2", "historical_p90_p975", warning, critical, n_history, "above historical P90")
    return ("PASS", "INFO", "historical_p90_p975", warning, critical, n_history, "within historical envelope")


def _sign_reason(*, calendar_spread: float, parent_spread: float, month_spread: float) -> str:
    cal_sign = np.sign(calendar_spread)
    parent_sign = np.sign(parent_spread)
    month_sign = np.sign(month_spread)
    if parent_sign != 0.0 and month_sign not in (0.0, parent_sign):
        return "month sign contradicts comparable parent spread"
    if cal_sign != 0.0 and month_sign not in (0.0, cal_sign):
        return "month sign contradicts calendar spread after comparable-block check"
    return "month sign is compatible with comparable-block decomposition"


def _safe_int(value: object, *, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARNING": 1, "UNSUPPORTED": 2, "CRITICAL": 3}.get(str(status), 2)
