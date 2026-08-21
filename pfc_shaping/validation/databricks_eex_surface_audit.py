"""Offline quality audit for normalized Databricks EEX quote surfaces.

The audit separates structural data integrity from observable market quote
conflicts.  A redundant CAL or quarter settlement can differ from the
hour-weighted finer-product strip without making either source row corrupt.
Those conflicts must be measured and routed through solver policy, not hidden
or silently forced into one simultaneously exact constraint system.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Mapping

import pandas as pd

from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.calibration.monthly_forward_curve import product_periods


SURFACE_AUDIT_SCHEMA_VERSION = "fmv_databricks_eex_surface_audit.v2"
DEFAULT_NESTING_TOLERANCE_EUR_MWH = 0.01
DEFAULT_RECOMPOSITION_TOLERANCE_EUR_MWH = 1e-9

EXPECTED_COLUMNS = (
    "date",
    "product",
    "load_type",
    "product_type",
    "market",
    "price",
    "unit",
    "source",
    "source_system",
    "product_id",
    "delivery_period_id",
    "fact_load_timestamp_utc",
    "source_snapshot_sha256",
    "point_in_time_available_at_proven",
    "rolling_origin_selection_authorized",
    "production_promotion_authorized",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MONTH = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")
_QUARTER = re.compile(r"^\d{4}-Q[1-4]$")
_CAL = re.compile(r"^\d{4}$")
_PRODUCT_TYPE_RANK = {"Month": 0, "Quarter": 1, "Cal": 2}


class DatabricksEexSurfaceAuditError(ValueError):
    """Raised when normalized EEX history is not safe to audit."""


@dataclass(frozen=True)
class DatabricksEexSurfaceAuditResult:
    """Inspectable local diagnostics with a non-promotional summary."""

    daily_coverage: pd.DataFrame
    nesting_diagnostics: pd.DataFrame
    offpeak_diagnostics: pd.DataFrame
    summary: Mapping[str, object]


def audit_databricks_eex_surfaces(
    history: pd.DataFrame,
    *,
    source_snapshot_sha256: str,
    source_normalization_content_id: str,
    nesting_tolerance_eur_mwh: float = DEFAULT_NESTING_TOLERANCE_EUR_MWH,
    recomposition_tolerance_eur_mwh: float = (
        DEFAULT_RECOMPOSITION_TOLERANCE_EUR_MWH
    ),
) -> DatabricksEexSurfaceAuditResult:
    """Audit quote coverage, nested-product conflicts and OFFPEAK identities."""

    snapshot_hash = _require_sha256(
        source_snapshot_sha256,
        label="source snapshot",
    )
    normalization_id = _require_sha256(
        source_normalization_content_id,
        label="source normalization content ID",
    )
    nesting_tolerance = _positive_finite(
        nesting_tolerance_eur_mwh,
        label="nesting tolerance",
    )
    recomposition_tolerance = _positive_finite(
        recomposition_tolerance_eur_mwh,
        label="recomposition tolerance",
    )
    frame = _validate_history(history, source_snapshot_sha256=snapshot_hash)

    nesting = _build_nesting_diagnostics(
        frame,
        tolerance_eur_mwh=nesting_tolerance,
    )
    offpeak = _build_offpeak_diagnostics(
        frame,
        tolerance_eur_mwh=recomposition_tolerance,
    )
    daily = _build_daily_coverage(frame, nesting=nesting, offpeak=offpeak)
    summary = _build_summary(
        frame,
        daily=daily,
        nesting=nesting,
        offpeak=offpeak,
        source_snapshot_sha256=snapshot_hash,
        source_normalization_content_id=normalization_id,
        nesting_tolerance_eur_mwh=nesting_tolerance,
        recomposition_tolerance_eur_mwh=recomposition_tolerance,
    )
    return DatabricksEexSurfaceAuditResult(
        daily_coverage=daily,
        nesting_diagnostics=nesting,
        offpeak_diagnostics=offpeak,
        summary=summary,
    )


def _validate_history(
    history: pd.DataFrame,
    *,
    source_snapshot_sha256: str,
) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise DatabricksEexSurfaceAuditError("normalized EEX history is empty")
    if tuple(history.columns) != EXPECTED_COLUMNS:
        raise DatabricksEexSurfaceAuditError(
            "normalized EEX history columns are not exact"
        )
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].dt.tz is not None:
        raise DatabricksEexSurfaceAuditError(
            "normalized EEX quotation dates are invalid"
        )
    frame["date"] = frame["date"].dt.normalize()
    frame["fact_load_timestamp_utc"] = pd.to_datetime(
        frame["fact_load_timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    if frame["fact_load_timestamp_utc"].isna().any():
        raise DatabricksEexSurfaceAuditError(
            "normalized EEX load timestamps are invalid"
        )
    load_dates = (
        frame["fact_load_timestamp_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if bool((frame["date"] > load_dates).any()):
        raise DatabricksEexSurfaceAuditError(
            "normalized EEX quotation date follows its load timestamp"
        )

    for column in (
        "product",
        "load_type",
        "product_type",
        "market",
        "unit",
        "source",
        "source_system",
        "product_id",
        "delivery_period_id",
        "source_snapshot_sha256",
    ):
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or bool(values.eq("").any()):
            raise DatabricksEexSurfaceAuditError(
                f"normalized EEX history has an empty {column}"
            )
        frame[column] = values

    _require_exact_values(frame, "market", {"CH"})
    _require_exact_values(frame, "unit", {"EUR/MWH"})
    _require_exact_values(frame, "source", {"EEX"})
    _require_exact_values(frame, "source_system", {"DATABRICKS_PRD_GOLD"})
    _require_exact_values(
        frame,
        "source_snapshot_sha256",
        {source_snapshot_sha256},
    )
    _require_allowed_values(frame, "load_type", {"BASE", "PEAK"})
    _require_allowed_values(frame, "product_type", set(_PRODUCT_TYPE_RANK))

    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if not all(math.isfinite(value) for value in frame["price"].to_numpy(float)):
        raise DatabricksEexSurfaceAuditError(
            "normalized EEX history contains a non-finite price"
        )
    for column in (
        "point_in_time_available_at_proven",
        "rolling_origin_selection_authorized",
        "production_promotion_authorized",
    ):
        values = frame[column].astype("boolean")
        if values.isna().any() or bool(values.any()):
            raise DatabricksEexSurfaceAuditError(
                f"normalized EEX history overclaims {column}"
            )
        frame[column] = values

    identity = ["date", "market", "load_type", "product"]
    if bool(frame.duplicated(identity).any()):
        raise DatabricksEexSurfaceAuditError(
            "normalized EEX history repeats a quote identity"
        )
    for row in frame[["product", "product_type"]].drop_duplicates().itertuples(
        index=False
    ):
        expected = _expected_product_type(str(row.product))
        if str(row.product_type) != expected:
            raise DatabricksEexSurfaceAuditError(
                "normalized EEX product/product_type mismatch"
            )
        _product_period_tuple(str(row.product))
    return frame.sort_values(identity, kind="mergesort").reset_index(drop=True)


def _build_nesting_diagnostics(
    history: pd.DataFrame,
    *,
    tolerance_eur_mwh: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (date, load_type), surface in history.groupby(
        ["date", "load_type"], sort=True
    ):
        quotes = {
            str(row.product): (float(row.price), str(row.product_type))
            for row in surface.itertuples(index=False)
        }
        periods = {
            product: _product_period_tuple(product) for product in quotes
        }
        for parent_product, (parent_price, parent_type) in quotes.items():
            if parent_type not in {"Cal", "Quarter"}:
                continue
            parent_periods = periods[parent_product]
            assignments = _complete_non_overlapping_child_partition(
                parent_product=parent_product,
                parent_type=parent_type,
                parent_periods=parent_periods,
                quotes=quotes,
            )
            if not assignments:
                continue
            parent_hours = sum(
                _month_hours(int(month.year), int(month.month), str(load_type))
                for month in parent_periods
            )
            implied_energy = sum(
                price
                * _month_hours(int(month.year), int(month.month), str(load_type))
                for month, _, price in assignments
            )
            implied_price = implied_energy / parent_hours
            residual = parent_price - implied_price
            child_products = tuple(
                dict.fromkeys(child for _, child, _ in assignments)
            )
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "market": "CH",
                    "load_type": str(load_type),
                    "parent_product": parent_product,
                    "parent_product_type": parent_type,
                    "parent_price_eur_mwh": parent_price,
                    "nested_implied_price_eur_mwh": implied_price,
                    "residual_eur_mwh": residual,
                    "abs_residual_eur_mwh": abs(residual),
                    "tolerance_eur_mwh": tolerance_eur_mwh,
                    "status": (
                        "CONSISTENT"
                        if abs(residual) <= tolerance_eur_mwh
                        else "QUOTE_CONFLICT"
                    ),
                    "covered_month_count": len(assignments),
                    "parent_delivery_hours": parent_hours,
                    "child_product_count": len(child_products),
                    "child_products": "|".join(child_products),
                    "comparison_policy": (
                        "FINEST_COMPLETE_NON_OVERLAPPING_CHILD_PARTITION"
                    ),
                }
            )
    columns = (
        "date",
        "market",
        "load_type",
        "parent_product",
        "parent_product_type",
        "parent_price_eur_mwh",
        "nested_implied_price_eur_mwh",
        "residual_eur_mwh",
        "abs_residual_eur_mwh",
        "tolerance_eur_mwh",
        "status",
        "covered_month_count",
        "parent_delivery_hours",
        "child_product_count",
        "child_products",
        "comparison_policy",
    )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["date", "load_type", "parent_product"], kind="mergesort"
    ).reset_index(drop=True)


def _complete_non_overlapping_child_partition(
    *,
    parent_product: str,
    parent_type: str,
    parent_periods: tuple[pd.Period, ...],
    quotes: Mapping[str, tuple[float, str]],
) -> list[tuple[pd.Period, str, float]]:
    """Return a strict child partition without splicing overlapping quotes.

    A quarter is comparable only with all three direct month quotes.  A
    calendar is partitioned quarter by quarter: use all three months when the
    complete monthly strip exists, otherwise use the direct quarter for the
    whole quarter.  One or two months must never be combined with the same
    quarter quote as if that quote represented only the uncovered remainder.
    """

    if parent_type == "Quarter":
        month_products = tuple(str(month) for month in parent_periods)
        if not all(product in quotes for product in month_products):
            return []
        return [
            (month, str(month), float(quotes[str(month)][0]))
            for month in parent_periods
        ]

    if parent_type != "Cal":
        raise DatabricksEexSurfaceAuditError(
            f"unsupported nesting parent type: {parent_type}"
        )

    assignments: list[tuple[pd.Period, str, float]] = []
    year = int(parent_product)
    for quarter in range(1, 5):
        quarter_months = tuple(
            month
            for month in parent_periods
            if int(month.quarter) == quarter
        )
        month_products = tuple(str(month) for month in quarter_months)
        if all(product in quotes for product in month_products):
            assignments.extend(
                (month, str(month), float(quotes[str(month)][0]))
                for month in quarter_months
            )
            continue
        quarter_product = f"{year}-Q{quarter}"
        if quarter_product not in quotes:
            return []
        quarter_price = float(quotes[quarter_product][0])
        assignments.extend(
            (month, quarter_product, quarter_price)
            for month in quarter_months
        )
    return assignments


def _build_offpeak_diagnostics(
    history: pd.DataFrame,
    *,
    tolerance_eur_mwh: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pivot = history.pivot(
        index=["date", "product", "product_type"],
        columns="load_type",
        values="price",
    )
    if not {"BASE", "PEAK"}.issubset(pivot.columns):
        paired = pivot.iloc[0:0]
    else:
        paired = pivot.dropna(subset=["BASE", "PEAK"])
    for row in paired.reset_index().itertuples(index=False):
        date = pd.Timestamp(row.date)
        product = str(row.product)
        product_type = str(row.product_type)
        base_price = float(row.BASE)
        peak_price = float(row.PEAK)
        total_hours, peak_hours, offpeak_hours = _product_hours(product)
        if total_hours <= 0 or peak_hours <= 0 or offpeak_hours <= 0:
            raise DatabricksEexSurfaceAuditError(
                "BASE/PEAK product has invalid delivery hours"
            )
        implied_offpeak = (
            base_price * total_hours - peak_price * peak_hours
        ) / offpeak_hours
        recomposed_base = (
            peak_price * peak_hours + implied_offpeak * offpeak_hours
        ) / total_hours
        residual = recomposed_base - base_price
        if not all(
            math.isfinite(value)
            for value in (implied_offpeak, recomposed_base, residual)
        ):
            raise DatabricksEexSurfaceAuditError(
                "BASE/PEAK product implies a non-finite OFFPEAK value"
            )
        rows.append(
            {
                "date": date,
                "market": "CH",
                "product": product,
                "product_type": product_type,
                "base_price_eur_mwh": base_price,
                "peak_price_eur_mwh": peak_price,
                "implied_offpeak_price_eur_mwh": implied_offpeak,
                "implied_offpeak_negative": implied_offpeak < 0.0,
                "total_hours": total_hours,
                "peak_hours": peak_hours,
                "offpeak_hours": offpeak_hours,
                "recomposition_residual_eur_mwh": residual,
                "abs_recomposition_residual_eur_mwh": abs(residual),
                "tolerance_eur_mwh": tolerance_eur_mwh,
                "status": (
                    "PASS"
                    if abs(residual) <= tolerance_eur_mwh
                    else "RECOMPOSITION_FAILURE"
                ),
            }
        )
    columns = (
        "date",
        "market",
        "product",
        "product_type",
        "base_price_eur_mwh",
        "peak_price_eur_mwh",
        "implied_offpeak_price_eur_mwh",
        "implied_offpeak_negative",
        "total_hours",
        "peak_hours",
        "offpeak_hours",
        "recomposition_residual_eur_mwh",
        "abs_recomposition_residual_eur_mwh",
        "tolerance_eur_mwh",
        "status",
    )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["date", "product"], kind="mergesort"
    ).reset_index(drop=True)


def _build_daily_coverage(
    history: pd.DataFrame,
    *,
    nesting: pd.DataFrame,
    offpeak: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    nesting_by_date = {
        pd.Timestamp(date): group
        for date, group in nesting.groupby("date", sort=False)
    }
    offpeak_by_date = {
        pd.Timestamp(date): group
        for date, group in offpeak.groupby("date", sort=False)
    }
    empty_nesting = nesting.iloc[0:0]
    empty_offpeak = offpeak.iloc[0:0]
    for date, surface in history.groupby("date", sort=True):
        load_sets = surface.groupby("product")["load_type"].agg(
            lambda values: frozenset(str(value) for value in values)
        )
        paired = int(load_sets.eq(frozenset({"BASE", "PEAK"})).sum())
        distinct_products = len(load_sets)
        date_nesting = nesting_by_date.get(pd.Timestamp(date), empty_nesting)
        date_offpeak = offpeak_by_date.get(pd.Timestamp(date), empty_offpeak)
        counts = surface.groupby(["load_type", "product_type"]).size()
        rows.append(
            {
                "date": pd.Timestamp(date),
                "row_count": len(surface),
                "distinct_product_count": distinct_products,
                "base_row_count": int(surface["load_type"].eq("BASE").sum()),
                "peak_row_count": int(surface["load_type"].eq("PEAK").sum()),
                "base_peak_pair_count": paired,
                "unpaired_product_count": distinct_products - paired,
                "base_peak_pair_coverage_rate": (
                    paired / distinct_products if distinct_products else 0.0
                ),
                "base_month_count": int(counts.get(("BASE", "Month"), 0)),
                "base_quarter_count": int(counts.get(("BASE", "Quarter"), 0)),
                "base_cal_count": int(counts.get(("BASE", "Cal"), 0)),
                "peak_month_count": int(counts.get(("PEAK", "Month"), 0)),
                "peak_quarter_count": int(counts.get(("PEAK", "Quarter"), 0)),
                "peak_cal_count": int(counts.get(("PEAK", "Cal"), 0)),
                "fully_nested_parent_count": len(date_nesting),
                "nesting_conflict_count": int(
                    date_nesting["status"].eq("QUOTE_CONFLICT").sum()
                ),
                "nesting_max_abs_residual_eur_mwh": (
                    float(date_nesting["abs_residual_eur_mwh"].max())
                    if not date_nesting.empty
                    else 0.0
                ),
                "offpeak_identity_count": len(date_offpeak),
                "negative_implied_offpeak_count": int(
                    date_offpeak["implied_offpeak_negative"].sum()
                ),
                "offpeak_max_abs_recomposition_residual_eur_mwh": (
                    float(
                        date_offpeak[
                            "abs_recomposition_residual_eur_mwh"
                        ].max()
                    )
                    if not date_offpeak.empty
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("date", kind="mergesort").reset_index(
        drop=True
    )


def _build_summary(
    history: pd.DataFrame,
    *,
    daily: pd.DataFrame,
    nesting: pd.DataFrame,
    offpeak: pd.DataFrame,
    source_snapshot_sha256: str,
    source_normalization_content_id: str,
    nesting_tolerance_eur_mwh: float,
    recomposition_tolerance_eur_mwh: float,
) -> dict[str, object]:
    conflicts = nesting["status"].eq("QUOTE_CONFLICT")
    conflict_count = int(conflicts.sum())
    pair_count = int(daily["base_peak_pair_count"].sum())
    distinct_date_products = int(daily["distinct_product_count"].sum())
    latest = daily.iloc[-1]
    first_peak = daily.loc[daily["peak_row_count"].gt(0), "date"]
    latest_conflicts = nesting.loc[
        nesting["date"].eq(latest["date"])
        & nesting["status"].eq("QUOTE_CONFLICT")
    ]
    dates_with_both_loads = daily["base_row_count"].gt(0) & daily[
        "peak_row_count"
    ].gt(0)
    return {
        "schema_version": SURFACE_AUDIT_SCHEMA_VERSION,
        "status": (
            "PASS_LOCAL_INTEGRITY_WITH_MARKET_QUOTE_CONFLICTS"
            if conflict_count
            else "PASS_LOCAL_INTEGRITY_NO_MARKET_QUOTE_CONFLICTS"
        ),
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_normalization_content_id": source_normalization_content_id,
        "grain": {
            "history": "quotation_date_market_load_type_product",
            "daily_coverage": "quotation_date",
            "nesting": "quotation_date_load_type_parent_product",
            "offpeak": "quotation_date_product",
        },
        "history_row_count": len(history),
        "quotation_date_count": len(daily),
        "quotation_date_min": daily["date"].min().date().isoformat(),
        "quotation_date_max": daily["date"].max().date().isoformat(),
        "first_peak_quotation_date": (
            first_peak.min().date().isoformat() if not first_peak.empty else None
        ),
        "date_count_with_base_and_peak": int(dates_with_both_loads.sum()),
        "date_share_with_base_and_peak": float(dates_with_both_loads.mean()),
        "daily_row_count_quantiles": _quantiles(daily["row_count"]),
        "coverage_by_quotation_year": _coverage_year_groups(daily),
        "base_peak_pair_count": pair_count,
        "base_peak_pair_coverage_rate": (
            pair_count / distinct_date_products if distinct_date_products else 0.0
        ),
        "unpaired_date_product_count": distinct_date_products - pair_count,
        "fully_nested_parent_count": len(nesting),
        "nesting_conflict_count": conflict_count,
        "nesting_conflict_rate": (
            conflict_count / len(nesting) if len(nesting) else 0.0
        ),
        "nesting_tolerance_eur_mwh": nesting_tolerance_eur_mwh,
        "nesting_abs_residual_quantiles_eur_mwh": _quantiles(
            nesting["abs_residual_eur_mwh"]
        ),
        "nesting_max_abs_residual_eur_mwh": (
            float(nesting["abs_residual_eur_mwh"].max())
            if not nesting.empty
            else 0.0
        ),
        "nesting_by_load_type": _status_groups(nesting, "load_type"),
        "nesting_by_parent_product_type": _status_groups(
            nesting, "parent_product_type"
        ),
        "nesting_by_quotation_year": _year_status_groups(nesting),
        "offpeak_identity_count": len(offpeak),
        "negative_implied_offpeak_count": int(
            offpeak["implied_offpeak_negative"].sum()
        ),
        "offpeak_recomposition_failure_count": int(
            offpeak["status"].eq("RECOMPOSITION_FAILURE").sum()
        ),
        "offpeak_recomposition_tolerance_eur_mwh": (
            recomposition_tolerance_eur_mwh
        ),
        "offpeak_max_abs_recomposition_residual_eur_mwh": (
            float(offpeak["abs_recomposition_residual_eur_mwh"].max())
            if not offpeak.empty
            else 0.0
        ),
        "latest_surface": {
            "date": latest["date"].date().isoformat(),
            "row_count": int(latest["row_count"]),
            "base_peak_pair_count": int(latest["base_peak_pair_count"]),
            "base_peak_pair_coverage_rate": float(
                latest["base_peak_pair_coverage_rate"]
            ),
            "fully_nested_parent_count": int(
                latest["fully_nested_parent_count"]
            ),
            "nesting_conflict_count": int(latest["nesting_conflict_count"]),
            "nesting_max_abs_residual_eur_mwh": float(
                latest["nesting_max_abs_residual_eur_mwh"]
            ),
            "offpeak_identity_count": int(latest["offpeak_identity_count"]),
            "negative_implied_offpeak_count": int(
                latest["negative_implied_offpeak_count"]
            ),
            "nesting_conflicts": [
                {
                    "load_type": str(row.load_type),
                    "parent_product": str(row.parent_product),
                    "parent_product_type": str(row.parent_product_type),
                    "residual_eur_mwh": float(row.residual_eur_mwh),
                    "abs_residual_eur_mwh": float(row.abs_residual_eur_mwh),
                }
                for row in latest_conflicts.itertuples(index=False)
            ],
        },
        "interpretation": {
            "quote_conflict_is_source_corruption": False,
            "quote_conflict_requires_solver_policy": True,
            "negative_implied_offpeak_is_automatically_invalid": False,
            "per_product_fill_forward_used": False,
            "settlement_price_used": True,
            "last_price_used": False,
        },
        "authority": {
            "local_research_and_pipeline_development": True,
            "point_in_time_availability_proven": False,
            "rolling_origin_selection_authorized": False,
            "model_selection_authorized": False,
            "production_promotion_authorized": False,
        },
    }


@lru_cache(maxsize=None)
def _month_hour_counts(year: int, month: int) -> tuple[int, int, int]:
    return count_hours(year, month, month, country="CH")


def _month_hours(year: int, month: int, load_type: str) -> int:
    total, peak, _ = _month_hour_counts(year, month)
    if load_type == "BASE":
        return total
    if load_type == "PEAK":
        return peak
    raise DatabricksEexSurfaceAuditError(f"unsupported load type: {load_type}")


@lru_cache(maxsize=None)
def _product_hours(product: str) -> tuple[int, int, int]:
    total = peak = offpeak = 0
    for month in _product_period_tuple(product):
        month_total, month_peak, month_offpeak = _month_hour_counts(
            int(month.year), int(month.month)
        )
        total += month_total
        peak += month_peak
        offpeak += month_offpeak
    return total, peak, offpeak


@lru_cache(maxsize=None)
def _product_period_tuple(product: str) -> tuple[pd.Period, ...]:
    return tuple(product_periods(product))


def _expected_product_type(product: str) -> str:
    if _MONTH.fullmatch(product):
        return "Month"
    if _QUARTER.fullmatch(product):
        return "Quarter"
    if _CAL.fullmatch(product):
        return "Cal"
    raise DatabricksEexSurfaceAuditError(
        f"normalized EEX product is invalid: {product}"
    )


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise DatabricksEexSurfaceAuditError(f"{label} SHA-256 is invalid")
    return normalized


def _positive_finite(value: object, *, label: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise DatabricksEexSurfaceAuditError(f"{label} is invalid") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise DatabricksEexSurfaceAuditError(f"{label} is invalid")
    return normalized


def _require_exact_values(
    frame: pd.DataFrame,
    column: str,
    expected: set[str],
) -> None:
    actual = set(frame[column].astype(str))
    if actual != expected:
        raise DatabricksEexSurfaceAuditError(
            f"normalized EEX {column} values are invalid"
        )


def _require_allowed_values(
    frame: pd.DataFrame,
    column: str,
    allowed: set[str],
) -> None:
    actual = set(frame[column].astype(str))
    if not actual or not actual.issubset(allowed):
        raise DatabricksEexSurfaceAuditError(
            f"normalized EEX {column} values are invalid"
        )


def _quantiles(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    return {
        str(quantile): float(value)
        for quantile, value in values.quantile(
            [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
        ).items()
    }


def _status_groups(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(column, sort=True):
        conflicts = int(group["status"].eq("QUOTE_CONFLICT").sum())
        rows.append(
            {
                column: str(value),
                "comparison_count": len(group),
                "conflict_count": conflicts,
                "conflict_rate": conflicts / len(group),
                "max_abs_residual_eur_mwh": float(
                    group["abs_residual_eur_mwh"].max()
                ),
            }
        )
    return rows


def _year_status_groups(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    working = frame.assign(quotation_year=frame["date"].dt.year)
    rows = _status_groups(working, "quotation_year")
    for row in rows:
        row["quotation_year"] = int(row["quotation_year"])
    return rows


def _coverage_year_groups(daily: pd.DataFrame) -> list[dict[str, object]]:
    working = daily.assign(quotation_year=daily["date"].dt.year)
    rows: list[dict[str, object]] = []
    for year, group in working.groupby("quotation_year", sort=True):
        both = group["base_row_count"].gt(0) & group["peak_row_count"].gt(0)
        pairs = int(group["base_peak_pair_count"].sum())
        products = int(group["distinct_product_count"].sum())
        rows.append(
            {
                "quotation_year": int(year),
                "date_count": len(group),
                "date_count_with_base_and_peak": int(both.sum()),
                "date_share_with_base_and_peak": float(both.mean()),
                "median_daily_row_count": float(group["row_count"].median()),
                "max_daily_row_count": int(group["row_count"].max()),
                "base_peak_pair_count": pairs,
                "base_peak_pair_coverage_rate": (
                    pairs / products if products else 0.0
                ),
            }
        )
    return rows


__all__ = [
    "DEFAULT_NESTING_TOLERANCE_EUR_MWH",
    "DEFAULT_RECOMPOSITION_TOLERANCE_EUR_MWH",
    "DatabricksEexSurfaceAuditError",
    "DatabricksEexSurfaceAuditResult",
    "EXPECTED_COLUMNS",
    "SURFACE_AUDIT_SCHEMA_VERSION",
    "audit_databricks_eex_surfaces",
]
