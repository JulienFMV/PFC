"""Offline quality audit for live EEX DAY/WEEK/WEEKEND quote history.

The audit is deliberately descriptive.  It measures observable quotation
horizons, BASE/PEAK pairing, product lifecycles and same-vintage nesting of
WEEK/WEEKEND against complete DAY strips.  It neither fills missing quotes nor
grants point-in-time, model-selection or production authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as calendar_date, timedelta
import math
import re
from typing import Mapping

import pandas as pd

from pfc_shaping.data.databricks_eex_daily_snapshot import (
    EEX_PUBLIC_CONTRACT_DETAILS_AS_OF,
    EEX_PUBLIC_CONTRACT_DETAILS_SHA256,
    EEX_PUBLIC_CONTRACT_DETAILS_URL,
)


SHORT_TENOR_AUDIT_SCHEMA_VERSION = "fmv_databricks_eex_short_tenor_audit.v1"
DEFAULT_NESTING_TOLERANCE_EUR_MWH = 0.01
DEFAULT_RECOMPOSITION_TOLERANCE_EUR_MWH = 1e-9
LOCAL_TIMEZONE = "Europe/Zurich"
EEX_PUBLIC_HOLIDAY_CALENDAR_AS_OF = "2025-06-19"
EEX_PUBLIC_HOLIDAY_CALENDAR_URL = (
    "https://www.eex.com/fileadmin/EEX/Downloads/Trading/Calendar/"
    "Holiday_Calendar/EEX_Trading_Calendar_Emissions_Spot_Derivatives.pdf"
)
EEX_PUBLIC_HOLIDAY_CALENDAR_SHA256 = (
    "17f170941ad6765b2e53d06733a49afcc40895acaf28510b39c528677f9fed8d"
)
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
_DAY = re.compile(r"^D:(\d{4}-\d{2}-\d{2})$")
_WEEK = re.compile(r"^W:(\d{4})-W(\d{2})$")
_WEEKEND = re.compile(r"^WE:(\d{4})-W(\d{2})$")
_PRODUCT_TYPES = ("Day", "Week", "Weekend")
_LOAD_TYPES = ("BASE", "PEAK")


class DatabricksEexShortTenorAuditError(ValueError):
    """Raised when the normalized short-tenor layer is unsafe to audit."""


@dataclass(frozen=True)
class DatabricksEexShortTenorAuditResult:
    """Inspectable short-tenor diagnostics with a non-promotional summary."""

    daily_coverage: pd.DataFrame
    horizon_profile: pd.DataFrame
    product_lifecycle: pd.DataFrame
    gap_diagnostics: pd.DataFrame
    nesting_diagnostics: pd.DataFrame
    offpeak_diagnostics: pd.DataFrame
    summary: Mapping[str, object]


def audit_databricks_eex_short_tenors(
    history: pd.DataFrame,
    *,
    source_snapshot_sha256: str,
    source_normalization_content_id: str,
    quarantine: pd.DataFrame | None = None,
    nesting_tolerance_eur_mwh: float = DEFAULT_NESTING_TOLERANCE_EUR_MWH,
    recomposition_tolerance_eur_mwh: float = (
        DEFAULT_RECOMPOSITION_TOLERANCE_EUR_MWH
    ),
) -> DatabricksEexShortTenorAuditResult:
    """Audit the live DAY/WEEK/WEEKEND layer without predictive claims."""

    snapshot_hash = _require_sha256(source_snapshot_sha256, label="source snapshot")
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
    frame, mapping_diagnostics = _validate_history(
        history,
        source_snapshot_sha256=snapshot_hash,
    )
    quarantine_keys = _validate_quarantine(quarantine)
    nesting = _build_nesting_diagnostics(
        frame,
        tolerance_eur_mwh=nesting_tolerance,
    )
    offpeak = _build_offpeak_diagnostics(
        frame,
        tolerance_eur_mwh=recomposition_tolerance,
    )
    daily = _build_daily_coverage(frame, nesting=nesting)
    horizon = _build_horizon_profile(frame)
    lifecycle = _build_product_lifecycle(frame)
    gaps = _build_gap_diagnostics(frame, quarantine_keys=quarantine_keys)
    summary = _build_summary(
        frame,
        daily=daily,
        lifecycle=lifecycle,
        gaps=gaps,
        nesting=nesting,
        offpeak=offpeak,
        mapping_diagnostics=mapping_diagnostics,
        source_snapshot_sha256=snapshot_hash,
        source_normalization_content_id=normalization_id,
        nesting_tolerance_eur_mwh=nesting_tolerance,
        recomposition_tolerance_eur_mwh=recomposition_tolerance,
    )
    return DatabricksEexShortTenorAuditResult(
        daily_coverage=daily,
        horizon_profile=horizon,
        product_lifecycle=lifecycle,
        gap_diagnostics=gaps,
        nesting_diagnostics=nesting,
        offpeak_diagnostics=offpeak,
        summary=summary,
    )


def _validate_history(
    history: pd.DataFrame,
    *,
    source_snapshot_sha256: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX short-tenor history is empty"
        )
    if tuple(history.columns) != EXPECTED_COLUMNS:
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX short-tenor history columns are not exact"
        )
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].dt.tz is not None:
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quotation dates are invalid"
        )
    frame["date"] = frame["date"].dt.normalize()
    frame["fact_load_timestamp_utc"] = pd.to_datetime(
        frame["fact_load_timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    if frame["fact_load_timestamp_utc"].isna().any():
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX load timestamps are invalid"
        )
    load_dates = (
        frame["fact_load_timestamp_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if bool(frame["date"].gt(load_dates).any()):
        raise DatabricksEexShortTenorAuditError(
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
            raise DatabricksEexShortTenorAuditError(
                f"normalized EEX short-tenor history has an empty {column}"
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
    _require_allowed_values(frame, "load_type", set(_LOAD_TYPES))
    _require_allowed_values(frame, "product_type", set(_PRODUCT_TYPES))

    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if not all(math.isfinite(value) for value in frame["price"].to_numpy(float)):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX short-tenor history has a non-finite price"
        )
    for column in (
        "point_in_time_available_at_proven",
        "rolling_origin_selection_authorized",
        "production_promotion_authorized",
    ):
        values = frame[column].astype("boolean")
        if values.isna().any() or bool(values.any()):
            raise DatabricksEexShortTenorAuditError(
                f"normalized EEX short-tenor history overclaims {column}"
            )
        frame[column] = values

    identity = ["date", "market", "load_type", "product"]
    if bool(frame.duplicated(identity).any()):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX short-tenor history repeats a quote identity"
        )

    metadata: list[tuple[pd.Timestamp, pd.Timestamp, int, int, int]] = []
    for row in frame[["product", "product_type"]].itertuples(index=False):
        metadata.append(_delivery_metadata(str(row.product), str(row.product_type)))
    frame["delivery_start"] = pd.to_datetime([item[0] for item in metadata])
    frame["delivery_end"] = pd.to_datetime([item[1] for item in metadata])
    frame["total_hours"] = pd.array([item[2] for item in metadata], dtype="Int64")
    frame["peak_hours"] = pd.array([item[3] for item in metadata], dtype="Int64")
    frame["offpeak_hours"] = pd.array([item[4] for item in metadata], dtype="Int64")
    frame["horizon_days"] = pd.array(
        (frame["delivery_start"] - frame["date"]).dt.days,
        dtype="Int64",
    )
    if bool(frame["horizon_days"].le(0).any()):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX short-tenor history contains a non-live quote"
        )

    delivery_ids_per_identity = frame.groupby(
        ["load_type", "product"], sort=False
    )["delivery_period_id"].nunique()
    if bool(delivery_ids_per_identity.gt(1).any()):
        raise DatabricksEexShortTenorAuditError(
            "one normalized short-tenor identity maps to multiple delivery periods"
        )
    products_per_delivery_id = frame.groupby("delivery_period_id", sort=False)[
        "product"
    ].nunique()
    if bool(products_per_delivery_id.gt(1).any()):
        raise DatabricksEexShortTenorAuditError(
            "one delivery period maps to multiple normalized products"
        )
    identities_per_product_id = frame.groupby("product_id", sort=False).apply(
        lambda group: len(
            group[["load_type", "product"]].drop_duplicates(ignore_index=True)
        ),
        include_groups=False,
    )
    mapping_diagnostics = {
        "product_id_count": int(frame["product_id"].nunique()),
        "product_id_reused_across_identity_count": int(
            identities_per_product_id.gt(1).sum()
        ),
        "delivery_period_id_count": int(frame["delivery_period_id"].nunique()),
        "delivery_period_cross_product_count": 0,
        "identity_cross_delivery_period_count": 0,
    }
    return frame.sort_values(identity, kind="mergesort").reset_index(drop=True), (
        mapping_diagnostics
    )


def _delivery_metadata(
    product: str,
    product_type: str,
) -> tuple[pd.Timestamp, pd.Timestamp, int, int, int]:
    day_match = _DAY.fullmatch(product)
    week_match = _WEEK.fullmatch(product)
    weekend_match = _WEEKEND.fullmatch(product)
    if day_match is not None:
        expected_type = "Day"
        start = pd.Timestamp(day_match.group(1))
        end = start
        peak_hours = 12
    elif week_match is not None:
        expected_type = "Week"
        start = _iso_week_date(week_match.group(1), week_match.group(2), weekday=1)
        end = start + pd.Timedelta(days=6)
        peak_hours = 60
    elif weekend_match is not None:
        expected_type = "Weekend"
        start = _iso_week_date(
            weekend_match.group(1),
            weekend_match.group(2),
            weekday=6,
        )
        end = start + pd.Timedelta(days=1)
        peak_hours = 24
    else:
        raise DatabricksEexShortTenorAuditError(
            f"normalized EEX short-tenor product is invalid: {product}"
        )
    if product_type != expected_type:
        raise DatabricksEexShortTenorAuditError(
            f"normalized EEX product type does not match {product}"
        )
    total_hours = _local_delivery_hours(start, end)
    offpeak_hours = total_hours - peak_hours
    if offpeak_hours <= 0:
        raise DatabricksEexShortTenorAuditError(
            f"normalized EEX product hours are invalid: {product}"
        )
    return start, end, total_hours, peak_hours, offpeak_hours


def _validate_quarantine(
    quarantine: pd.DataFrame | None,
) -> dict[tuple[pd.Timestamp, str, str, str], str]:
    if quarantine is None:
        return {}
    expected_columns = (
        "ProductID",
        "DeliveryPeriodID",
        "QuotationDateID",
        "SettlementPriceEurMWh",
        "LastPriceEurMWh",
        "FactLoadTimestampUtc",
        "Country",
        "Commodity",
        "ProductType",
        "DeliveryPeriodType",
        "DeliveryStartDate",
        "DeliveryEndDate",
        "canonical_product",
        "quarantine_reason",
    )
    if not isinstance(quarantine, pd.DataFrame) or tuple(quarantine.columns) != (
        expected_columns
    ):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quarantine columns are not exact"
        )
    working = quarantine.copy()
    for column in (
        "Country",
        "Commodity",
        "ProductType",
        "DeliveryPeriodType",
        "canonical_product",
        "quarantine_reason",
    ):
        values = working[column].astype("string").str.strip()
        if values.isna().any() or bool(values.eq("").any()):
            raise DatabricksEexShortTenorAuditError(
                f"normalized EEX quarantine has an empty {column}"
            )
        working[column] = values
    short = working.loc[
        working["DeliveryPeriodType"].isin({"DAY", "WEEK", "WEEKEND"})
    ].copy()
    if short.empty:
        return {}
    short["QuotationDateID"] = pd.to_datetime(
        short["QuotationDateID"], errors="coerce"
    )
    if (
        short["QuotationDateID"].isna().any()
        or short["QuotationDateID"].dt.tz is not None
    ):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quarantine quotation dates are invalid"
        )
    short["QuotationDateID"] = short["QuotationDateID"].dt.normalize()
    short["FactLoadTimestampUtc"] = pd.to_datetime(
        short["FactLoadTimestampUtc"], errors="coerce", utc=True
    )
    if short["FactLoadTimestampUtc"].isna().any():
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quarantine load timestamps are invalid"
        )
    load_dates = (
        short["FactLoadTimestampUtc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if bool(short["QuotationDateID"].gt(load_dates).any()):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quarantine quotation date follows its load timestamp"
        )
    short["SettlementPriceEurMWh"] = pd.to_numeric(
        short["SettlementPriceEurMWh"], errors="coerce"
    )
    if not all(
        math.isfinite(value)
        for value in short["SettlementPriceEurMWh"].to_numpy(float)
    ):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quarantine settlement price is invalid"
        )
    for column in ("DeliveryStartDate", "DeliveryEndDate"):
        short[column] = pd.to_datetime(short[column], errors="coerce")
        if short[column].isna().any() or short[column].dt.tz is not None:
            raise DatabricksEexShortTenorAuditError(
                f"normalized EEX quarantine has an invalid {column}"
            )
        short[column] = short[column].dt.normalize()
    if bool(short["DeliveryEndDate"].lt(short["DeliveryStartDate"]).any()):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quarantine delivery bounds are invalid"
        )
    _require_exact_values(short, "Country", {"CH"})
    _require_exact_values(short, "Commodity", {"POWER"})
    _require_allowed_values(short, "ProductType", set(_LOAD_TYPES))
    allowed_reasons = {
        "DELIVERY_BOUNDARY_MISMATCH",
        "QUOTATION_NOT_BEFORE_DELIVERY_START",
    }
    actual_reasons = set(short["quarantine_reason"].astype(str))
    if not actual_reasons.issubset(allowed_reasons):
        raise DatabricksEexShortTenorAuditError(
            "normalized EEX quarantine reason is invalid"
        )
    keys: dict[tuple[pd.Timestamp, str, str, str], str] = {}
    labels = {"DAY": "Day", "WEEK": "Week", "WEEKEND": "Weekend"}
    for row in short.itertuples(index=False):
        product_type = labels[str(row.DeliveryPeriodType)]
        start, end, *_ = _delivery_metadata(
            str(row.canonical_product),
            product_type,
        )
        if str(row.ProductType) == "PEAK" and product_type == "Week":
            end = start + pd.Timedelta(days=4)
        if str(row.quarantine_reason) != "DELIVERY_BOUNDARY_MISMATCH" and (
            pd.Timestamp(row.DeliveryStartDate) != start
            or pd.Timestamp(row.DeliveryEndDate) != end
        ):
            raise DatabricksEexShortTenorAuditError(
                "normalized EEX quarantine delivery metadata is inconsistent"
            )
        key = (
            pd.Timestamp(row.QuotationDateID),
            product_type,
            str(row.ProductType),
            str(row.canonical_product),
        )
        if key in keys:
            raise DatabricksEexShortTenorAuditError(
                "normalized EEX quarantine repeats a short-tenor quote key"
            )
        keys[key] = str(row.quarantine_reason)
    return keys


def _iso_week_date(year: str, week: str, *, weekday: int) -> pd.Timestamp:
    try:
        value = calendar_date.fromisocalendar(int(year), int(week), weekday)
    except ValueError as exc:
        raise DatabricksEexShortTenorAuditError(
            f"normalized EEX ISO week is invalid: {year}-W{week}"
        ) from exc
    return pd.Timestamp(value)


def _local_delivery_hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    start_local = pd.Timestamp(start.date(), tz=LOCAL_TIMEZONE)
    end_exclusive_local = pd.Timestamp(
        (end + pd.Timedelta(days=1)).date(),
        tz=LOCAL_TIMEZONE,
    )
    duration = end_exclusive_local.tz_convert("UTC") - start_local.tz_convert("UTC")
    hours = duration / pd.Timedelta(hours=1)
    if not float(hours).is_integer():
        raise DatabricksEexShortTenorAuditError("delivery hours are not integral")
    return int(hours)


def _build_nesting_diagnostics(
    frame: pd.DataFrame,
    *,
    tolerance_eur_mwh: float,
) -> pd.DataFrame:
    lookup = {
        (row.date, row.load_type, row.product): row
        for row in frame.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    parents = frame.loc[frame["product_type"].isin({"Week", "Weekend"})]
    for parent in parents.itertuples(index=False):
        dates = _expected_child_dates(
            pd.Timestamp(parent.delivery_start),
            str(parent.product_type),
            str(parent.load_type),
        )
        children = [
            lookup.get(
                (
                    pd.Timestamp(parent.date),
                    str(parent.load_type),
                    f"D:{delivery_date.date().isoformat()}",
                )
            )
            for delivery_date in dates
        ]
        if any(child is None for child in children):
            continue
        complete_children = [child for child in children if child is not None]
        weight_column = "total_hours" if parent.load_type == "BASE" else "peak_hours"
        weights = [float(getattr(child, weight_column)) for child in complete_children]
        child_price = sum(
            float(child.price) * weight
            for child, weight in zip(complete_children, weights, strict=True)
        ) / sum(weights)
        residual = float(parent.price) - child_price
        rows.append(
            {
                "date": pd.Timestamp(parent.date),
                "parent_product_type": str(parent.product_type),
                "parent_product": str(parent.product),
                "load_type": str(parent.load_type),
                "parent_horizon_days": int(parent.horizon_days),
                "expected_child_count": len(dates),
                "child_products": "|".join(str(child.product) for child in complete_children),
                "parent_price_eur_mwh": float(parent.price),
                "child_strip_price_eur_mwh": child_price,
                "residual_eur_mwh": residual,
                "abs_residual_eur_mwh": abs(residual),
                "tolerance_eur_mwh": tolerance_eur_mwh,
                "status": (
                    "CONSISTENT"
                    if abs(residual) <= tolerance_eur_mwh
                    else "QUOTE_CONFLICT"
                ),
            }
        )
    columns = (
        "date",
        "parent_product_type",
        "parent_product",
        "load_type",
        "parent_horizon_days",
        "expected_child_count",
        "child_products",
        "parent_price_eur_mwh",
        "child_strip_price_eur_mwh",
        "residual_eur_mwh",
        "abs_residual_eur_mwh",
        "tolerance_eur_mwh",
        "status",
    )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["date", "parent_product_type", "parent_product", "load_type"],
        kind="mergesort",
        ignore_index=True,
    )


def _expected_child_dates(
    start: pd.Timestamp,
    product_type: str,
    load_type: str,
) -> tuple[pd.Timestamp, ...]:
    if product_type == "Weekend":
        count = 2
    elif product_type == "Week":
        count = 5 if load_type == "PEAK" else 7
    else:
        raise DatabricksEexShortTenorAuditError(
            f"unsupported short-tenor parent type: {product_type}"
        )
    return tuple(start + pd.Timedelta(days=offset) for offset in range(count))


def _build_offpeak_diagnostics(
    frame: pd.DataFrame,
    *,
    tolerance_eur_mwh: float,
) -> pd.DataFrame:
    pivot = frame.pivot(
        index=["date", "product_type", "product"],
        columns="load_type",
        values="price",
    )
    for load_type in _LOAD_TYPES:
        if load_type not in pivot.columns:
            pivot[load_type] = float("nan")
    pairs = pivot.dropna(subset=["BASE", "PEAK"]).reset_index()
    metadata = frame.drop_duplicates(["product_type", "product"]).set_index(
        ["product_type", "product"]
    )
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        product_meta = metadata.loc[(pair.product_type, pair.product)]
        total_hours = int(product_meta["total_hours"])
        peak_hours = int(product_meta["peak_hours"])
        offpeak_hours = int(product_meta["offpeak_hours"])
        implied_offpeak = (
            float(pair.BASE) * total_hours - float(pair.PEAK) * peak_hours
        ) / offpeak_hours
        recomposed = (
            float(pair.PEAK) * peak_hours + implied_offpeak * offpeak_hours
        ) / total_hours
        residual = recomposed - float(pair.BASE)
        rows.append(
            {
                "date": pd.Timestamp(pair.date),
                "product_type": str(pair.product_type),
                "product": str(pair.product),
                "total_hours": total_hours,
                "peak_hours": peak_hours,
                "offpeak_hours": offpeak_hours,
                "base_price_eur_mwh": float(pair.BASE),
                "peak_price_eur_mwh": float(pair.PEAK),
                "implied_offpeak_price_eur_mwh": implied_offpeak,
                "recomposition_residual_eur_mwh": residual,
                "abs_recomposition_residual_eur_mwh": abs(residual),
                "tolerance_eur_mwh": tolerance_eur_mwh,
                "status": (
                    "PASS" if abs(residual) <= tolerance_eur_mwh else "FAIL"
                ),
            }
        )
    columns = (
        "date",
        "product_type",
        "product",
        "total_hours",
        "peak_hours",
        "offpeak_hours",
        "base_price_eur_mwh",
        "peak_price_eur_mwh",
        "implied_offpeak_price_eur_mwh",
        "recomposition_residual_eur_mwh",
        "abs_recomposition_residual_eur_mwh",
        "tolerance_eur_mwh",
        "status",
    )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["date", "product_type", "product"],
        kind="mergesort",
        ignore_index=True,
    )


def _build_daily_coverage(
    frame: pd.DataFrame,
    *,
    nesting: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (quotation_date, product_type), group in frame.groupby(
        ["date", "product_type"], sort=True
    ):
        products = group.pivot_table(
            index="product",
            columns="load_type",
            values="price",
            aggfunc="size",
            fill_value=0,
        )
        for load_type in _LOAD_TYPES:
            if load_type not in products:
                products[load_type] = 0
        paired = products["BASE"].eq(1) & products["PEAK"].eq(1)
        base_only = products["BASE"].eq(1) & products["PEAK"].eq(0)
        peak_only = products["BASE"].eq(0) & products["PEAK"].eq(1)
        date_nesting = nesting.loc[
            nesting["date"].eq(quotation_date)
            & nesting["parent_product_type"].eq(product_type)
        ]
        rows.append(
            {
                "date": pd.Timestamp(quotation_date),
                "product_type": str(product_type),
                "row_count": len(group),
                "distinct_product_count": len(products),
                "base_row_count": int(group["load_type"].eq("BASE").sum()),
                "peak_row_count": int(group["load_type"].eq("PEAK").sum()),
                "base_peak_pair_count": int(paired.sum()),
                "base_only_product_count": int(base_only.sum()),
                "peak_only_product_count": int(peak_only.sum()),
                "base_peak_pair_coverage_rate": float(paired.mean()),
                "min_horizon_days": int(group["horizon_days"].min()),
                "median_horizon_days": float(group["horizon_days"].median()),
                "max_horizon_days": int(group["horizon_days"].max()),
                "complete_day_strip_parent_count": len(date_nesting),
                "nesting_conflict_count": int(
                    date_nesting["status"].eq("QUOTE_CONFLICT").sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["date", "product_type"], kind="mergesort", ignore_index=True
    )


def _build_horizon_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segment_sizes = frame.groupby(["product_type", "load_type"], sort=False).size()
    for (product_type, load_type, horizon_days), group in frame.groupby(
        ["product_type", "load_type", "horizon_days"], sort=True
    ):
        segment_count = int(segment_sizes.loc[(product_type, load_type)])
        rows.append(
            {
                "product_type": str(product_type),
                "load_type": str(load_type),
                "horizon_days": int(horizon_days),
                "row_count": len(group),
                "row_share_within_segment": len(group) / segment_count,
                "distinct_product_count": int(group["product"].nunique()),
                "quotation_date_count": int(group["date"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["product_type", "load_type", "horizon_days"],
        kind="mergesort",
        ignore_index=True,
    )


def _build_product_lifecycle(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    calendar_years = range(
        int(frame["date"].dt.year.min()),
        int(frame["delivery_start"].dt.year.max()) + 1,
    )
    standard_closures = _standard_eex_closure_dates(calendar_years)
    for (product_type, load_type, product), group in frame.groupby(
        ["product_type", "load_type", "product"], sort=True
    ):
        dates = pd.DatetimeIndex(group["date"].sort_values().unique())
        calendar_gaps = dates.to_series(index=range(len(dates))).diff().dt.days.dropna()
        intervening_weekdays = 0
        intervening_candidate_exchange_days = 0
        for previous, current in zip(dates[:-1], dates[1:], strict=True):
            for intervening_date in pd.date_range(
                previous + pd.Timedelta(days=1),
                current - pd.Timedelta(days=1),
            ):
                if intervening_date.weekday() >= 5:
                    continue
                intervening_weekdays += 1
                if intervening_date.date() not in standard_closures:
                    intervening_candidate_exchange_days += 1
        rows.append(
            {
                "product_type": str(product_type),
                "load_type": str(load_type),
                "product": str(product),
                "delivery_start": pd.Timestamp(group["delivery_start"].iloc[0]),
                "delivery_end": pd.Timestamp(group["delivery_end"].iloc[0]),
                "first_quotation_date": pd.Timestamp(dates.min()),
                "last_quotation_date": pd.Timestamp(dates.max()),
                "first_horizon_days": int(
                    (group["delivery_start"].iloc[0] - dates.min()).days
                ),
                "last_horizon_days": int(
                    (group["delivery_start"].iloc[0] - dates.max()).days
                ),
                "observation_count": len(dates),
                "max_calendar_gap_days": (
                    int(calendar_gaps.max()) if not calendar_gaps.empty else 0
                ),
                "intervening_weekday_count_without_exchange_calendar": int(
                    intervening_weekdays
                ),
                "intervening_candidate_exchange_day_count": int(
                    intervening_candidate_exchange_days
                ),
                "product_id": str(group["product_id"].iloc[0]),
                "delivery_period_id": str(group["delivery_period_id"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["product_type", "load_type", "delivery_start", "product"],
        kind="mergesort",
        ignore_index=True,
    )


def _build_gap_diagnostics(
    frame: pd.DataFrame,
    *,
    quarantine_keys: Mapping[tuple[pd.Timestamp, str, str, str], str],
) -> pd.DataFrame:
    calendar_years = range(
        int(frame["date"].dt.year.min()),
        int(frame["delivery_start"].dt.year.max()) + 1,
    )
    standard_closures = _standard_eex_closure_dates(calendar_years)
    globally_observed_dates = set(frame["date"].dt.date)
    rows: list[dict[str, object]] = []
    for (product_type, load_type, product), group in frame.groupby(
        ["product_type", "load_type", "product"], sort=True
    ):
        dates = pd.DatetimeIndex(group["date"].sort_values().unique())
        delivery_start = pd.Timestamp(group["delivery_start"].iloc[0])
        for previous, current in zip(dates[:-1], dates[1:], strict=True):
            for missing_date in pd.date_range(
                previous + pd.Timedelta(days=1),
                current - pd.Timedelta(days=1),
            ):
                if missing_date.weekday() >= 5:
                    continue
                if missing_date.date() in standard_closures:
                    continue
                source_wide = missing_date.date() not in globally_observed_dates
                quarantine_reason = quarantine_keys.get(
                    (
                        pd.Timestamp(missing_date),
                        str(product_type),
                        str(load_type),
                        str(product),
                    )
                )
                rows.append(
                    {
                        "missing_quotation_date": pd.Timestamp(missing_date),
                        "product_type": str(product_type),
                        "load_type": str(load_type),
                        "product": str(product),
                        "previous_quotation_date": pd.Timestamp(previous),
                        "next_quotation_date": pd.Timestamp(current),
                        "missing_horizon_days": int(
                            (delivery_start - missing_date).days
                        ),
                        "source_wide_date_absent": source_wide,
                        "classification": (
                            "EXPLAINED_NORMALIZATION_QUARANTINE"
                            if quarantine_reason is not None
                            else (
                                "CANDIDATE_SOURCE_WIDE_GAP"
                                if source_wide
                                else "CANDIDATE_IDENTITY_GAP"
                            )
                        ),
                        "quarantine_reason": quarantine_reason,
                    }
                )
    columns = (
        "missing_quotation_date",
        "product_type",
        "load_type",
        "product",
        "previous_quotation_date",
        "next_quotation_date",
        "missing_horizon_days",
        "source_wide_date_absent",
        "classification",
        "quarantine_reason",
    )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["missing_quotation_date", "product_type", "load_type", "product"],
        kind="mergesort",
        ignore_index=True,
    )


def _build_summary(
    frame: pd.DataFrame,
    *,
    daily: pd.DataFrame,
    lifecycle: pd.DataFrame,
    gaps: pd.DataFrame,
    nesting: pd.DataFrame,
    offpeak: pd.DataFrame,
    mapping_diagnostics: Mapping[str, int],
    source_snapshot_sha256: str,
    source_normalization_content_id: str,
    nesting_tolerance_eur_mwh: float,
    recomposition_tolerance_eur_mwh: float,
) -> dict[str, object]:
    conflicts = nesting["status"].eq("QUOTE_CONFLICT")
    recomposition_failures = offpeak["status"].eq("FAIL")
    pair_count = int(daily["base_peak_pair_count"].sum())
    distinct_date_products = int(daily["distinct_product_count"].sum())
    peak_rows = frame.loc[frame["load_type"].eq("PEAK")]
    unexplained_gaps = gaps.loc[
        ~gaps["classification"].eq("EXPLAINED_NORMALIZATION_QUARANTINE")
    ]
    if bool(conflicts.any()) and not unexplained_gaps.empty:
        status = (
            "PASS_LOCAL_INTEGRITY_WITH_MARKET_QUOTE_CONFLICTS_"
            "AND_LOCALIZED_TEMPORAL_GAPS"
        )
    elif bool(conflicts.any()):
        status = "PASS_LOCAL_INTEGRITY_WITH_MARKET_QUOTE_CONFLICTS"
    elif not unexplained_gaps.empty:
        status = "PASS_LOCAL_INTEGRITY_WITH_LOCALIZED_TEMPORAL_GAPS"
    else:
        status = "PASS_LOCAL_INTEGRITY_NO_MARKET_QUOTE_CONFLICTS"
    return {
        "schema_version": SHORT_TENOR_AUDIT_SCHEMA_VERSION,
        "status": status,
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_normalization_content_id": source_normalization_content_id,
        "grain": {
            "history": "quotation_date_market_load_type_product",
            "daily_coverage": "quotation_date_product_type",
            "horizon_profile": "product_type_load_type_horizon_days",
            "product_lifecycle": "product_type_load_type_product",
            "gap_diagnostics": "missing_quotation_date_product_type_load_type_product",
            "nesting": "quotation_date_parent_product_load_type",
            "offpeak": "quotation_date_product_type_product",
        },
        "history_row_count": len(frame),
        "quotation_date_count": int(frame["date"].nunique()),
        "quotation_date_min": frame["date"].min().date().isoformat(),
        "quotation_date_max": frame["date"].max().date().isoformat(),
        "delivery_start_min": frame["delivery_start"].min().date().isoformat(),
        "delivery_start_max": frame["delivery_start"].max().date().isoformat(),
        "first_peak_quotation_date": (
            peak_rows["date"].min().date().isoformat() if not peak_rows.empty else None
        ),
        "non_positive_horizon_count": 0,
        "base_peak_pair_count": pair_count,
        "base_peak_pair_coverage_rate": (
            pair_count / distinct_date_products if distinct_date_products else 0.0
        ),
        "base_only_date_product_count": int(
            daily["base_only_product_count"].sum()
        ),
        "peak_only_date_product_count": int(
            daily["peak_only_product_count"].sum()
        ),
        "segments": _segment_summary(frame),
        "coverage_by_quotation_year": _coverage_year_summary(daily),
        "lifecycle": {
            "identity_count": len(lifecycle),
            "identity_with_unadjusted_intervening_weekday_count": int(
                lifecycle[
                    "intervening_weekday_count_without_exchange_calendar"
                ].gt(0).sum()
            ),
            "max_calendar_gap_days": int(lifecycle["max_calendar_gap_days"].max()),
            "identity_with_candidate_exchange_day_gap_count": int(
                lifecycle["intervening_candidate_exchange_day_count"].gt(0).sum()
            ),
            "candidate_exchange_day_gap_count": int(
                lifecycle["intervening_candidate_exchange_day_count"].sum()
            ),
            "max_candidate_exchange_day_gap_count_per_identity": int(
                lifecycle["intervening_candidate_exchange_day_count"].max()
            ),
            "observation_count_quantiles": _quantiles(
                lifecycle["observation_count"]
            ),
        },
        "mapping_diagnostics": dict(mapping_diagnostics),
        "temporal_gap_diagnostic_count": len(gaps),
        "explained_normalization_quarantine_gap_count": int(
            gaps["classification"].eq("EXPLAINED_NORMALIZATION_QUARANTINE").sum()
        ),
        "unexplained_candidate_gap_count": len(unexplained_gaps),
        "temporal_gap_date_count": int(
            gaps["missing_quotation_date"].nunique()
        ),
        "unexplained_candidate_gap_date_count": int(
            unexplained_gaps["missing_quotation_date"].nunique()
        ),
        "source_wide_gap_date_count": int(
            unexplained_gaps.loc[
                unexplained_gaps["source_wide_date_absent"],
                "missing_quotation_date",
            ].nunique()
        ),
        "temporal_gaps_by_date": _gap_date_summary(gaps),
        "temporal_gaps_by_segment": _gap_segment_summary(gaps),
        "nesting_comparison_count": len(nesting),
        "nesting_conflict_count": int(conflicts.sum()),
        "nesting_conflict_rate": float(conflicts.mean()) if len(nesting) else 0.0,
        "nesting_max_abs_residual_eur_mwh": (
            float(nesting["abs_residual_eur_mwh"].max()) if len(nesting) else None
        ),
        "nesting_tolerance_eur_mwh": nesting_tolerance_eur_mwh,
        "nesting_by_parent_product_type": _status_groups(
            nesting,
            "parent_product_type",
        ),
        "nesting_by_load_type": _status_groups(nesting, "load_type"),
        "offpeak_identity_count": len(offpeak),
        "negative_implied_offpeak_count": int(
            offpeak["implied_offpeak_price_eur_mwh"].lt(0.0).sum()
        ),
        "offpeak_recomposition_failure_count": int(recomposition_failures.sum()),
        "offpeak_max_abs_recomposition_residual_eur_mwh": (
            float(offpeak["abs_recomposition_residual_eur_mwh"].max())
            if len(offpeak)
            else None
        ),
        "offpeak_recomposition_tolerance_eur_mwh": (
            recomposition_tolerance_eur_mwh
        ),
        "public_contract_details": {
            "as_of": EEX_PUBLIC_CONTRACT_DETAILS_AS_OF,
            "url": EEX_PUBLIC_CONTRACT_DETAILS_URL,
            "sha256": EEX_PUBLIC_CONTRACT_DETAILS_SHA256,
        },
        "public_holiday_calendar": {
            "as_of": EEX_PUBLIC_HOLIDAY_CALENDAR_AS_OF,
            "url": EEX_PUBLIC_HOLIDAY_CALENDAR_URL,
            "sha256": EEX_PUBLIC_HOLIDAY_CALENDAR_SHA256,
            "applied_standard_power_closures": (
                "JAN_01|GOOD_FRIDAY|EASTER_MONDAY|MAY_01|"
                "DEC_24|DEC_25|DEC_26|DEC_31"
            ),
        },
        "interpretation": {
            "settlement_price_used": True,
            "last_price_used": False,
            "per_product_fill_forward_used": False,
            "quotation_must_precede_delivery_start": True,
            "short_tenors_are_intraday_or_15_minute_evidence": False,
            "unadjusted_weekday_gap_counts_adjusted_for_exchange_holidays": False,
            "candidate_exchange_day_gap_counts_adjusted_for_standard_holidays": (
                True
            ),
            "weekday_gap_is_automatically_missing_data": False,
            "standard_eex_holiday_calendar_applied": True,
            "remaining_candidate_exchange_day_gap_is_automatically_missing_data": (
                False
            ),
            "localized_temporal_gap_is_automatically_source_corruption": False,
            "quote_conflict_is_source_corruption": False,
            "negative_implied_offpeak_is_automatically_invalid": False,
        },
        "authority": {
            "local_research_and_pipeline_development": True,
            "point_in_time_availability_proven": False,
            "rolling_origin_selection_authorized": False,
            "model_selection_authorized": False,
            "production_promotion_authorized": False,
        },
    }


def _segment_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (product_type, load_type), group in frame.groupby(
        ["product_type", "load_type"], sort=True
    ):
        rows.append(
            {
                "product_type": str(product_type),
                "load_type": str(load_type),
                "row_count": len(group),
                "product_count": int(group["product"].nunique()),
                "quotation_date_count": int(group["date"].nunique()),
                "quotation_date_min": group["date"].min().date().isoformat(),
                "quotation_date_max": group["date"].max().date().isoformat(),
                "delivery_start_min": (
                    group["delivery_start"].min().date().isoformat()
                ),
                "delivery_start_max": (
                    group["delivery_start"].max().date().isoformat()
                ),
                "horizon_days_quantiles": _quantiles(group["horizon_days"]),
            }
        )
    return rows


def _standard_eex_closure_dates(years: range) -> set[calendar_date]:
    closures: set[calendar_date] = set()
    for year in years:
        easter = _gregorian_easter(year)
        closures.update(
            {
                calendar_date(year, 1, 1),
                easter - timedelta(days=2),
                easter + timedelta(days=1),
                calendar_date(year, 5, 1),
                calendar_date(year, 12, 24),
                calendar_date(year, 12, 25),
                calendar_date(year, 12, 26),
                calendar_date(year, 12, 31),
            }
        )
    return closures


def _gregorian_easter(year: int) -> calendar_date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return calendar_date(year, month, day)


def _coverage_year_summary(daily: pd.DataFrame) -> list[dict[str, object]]:
    working = daily.assign(quotation_year=daily["date"].dt.year)
    rows: list[dict[str, object]] = []
    for (year, product_type), group in working.groupby(
        ["quotation_year", "product_type"], sort=True
    ):
        pair_count = int(group["base_peak_pair_count"].sum())
        product_count = int(group["distinct_product_count"].sum())
        rows.append(
            {
                "quotation_year": int(year),
                "product_type": str(product_type),
                "quotation_date_count": len(group),
                "row_count": int(group["row_count"].sum()),
                "base_peak_pair_count": pair_count,
                "base_peak_pair_coverage_rate": (
                    pair_count / product_count if product_count else 0.0
                ),
                "base_only_product_count": int(
                    group["base_only_product_count"].sum()
                ),
                "peak_only_product_count": int(
                    group["peak_only_product_count"].sum()
                ),
            }
        )
    return rows


def _gap_date_summary(gaps: pd.DataFrame) -> list[dict[str, object]]:
    if gaps.empty:
        return []
    return [
        {
            "missing_quotation_date": date.date().isoformat(),
            "diagnostic_count": len(group),
            "explained_quarantine_count": int(
                group["classification"].eq(
                    "EXPLAINED_NORMALIZATION_QUARANTINE"
                ).sum()
            ),
            "unexplained_candidate_count": int(
                (~group["classification"].eq(
                    "EXPLAINED_NORMALIZATION_QUARANTINE"
                )).sum()
            ),
            "source_wide_date_absent": bool(group["source_wide_date_absent"].all()),
        }
        for date, group in gaps.groupby("missing_quotation_date", sort=True)
    ]


def _gap_segment_summary(gaps: pd.DataFrame) -> list[dict[str, object]]:
    if gaps.empty:
        return []
    return [
        {
            "product_type": str(product_type),
            "load_type": str(load_type),
            "diagnostic_count": len(group),
            "affected_product_count": int(group["product"].nunique()),
            "affected_date_count": int(group["missing_quotation_date"].nunique()),
        }
        for (product_type, load_type), group in gaps.groupby(
            ["product_type", "load_type"], sort=True
        )
    ]


def _status_groups(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
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


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        str(quantile): float(value)
        for quantile, value in values.astype(float).quantile(
            [0.0, 0.05, 0.5, 0.95, 1.0]
        ).items()
    }


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise DatabricksEexShortTenorAuditError(f"{label} SHA-256 is invalid")
    return normalized


def _positive_finite(value: object, *, label: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise DatabricksEexShortTenorAuditError(f"{label} is invalid") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise DatabricksEexShortTenorAuditError(f"{label} is invalid")
    return normalized


def _require_exact_values(
    frame: pd.DataFrame,
    column: str,
    expected: set[str],
) -> None:
    actual = set(frame[column].astype(str))
    if actual != expected:
        raise DatabricksEexShortTenorAuditError(
            f"normalized EEX {column} values are invalid"
        )


def _require_allowed_values(
    frame: pd.DataFrame,
    column: str,
    allowed: set[str],
) -> None:
    actual = set(frame[column].astype(str))
    if not actual or not actual.issubset(allowed):
        raise DatabricksEexShortTenorAuditError(
            f"normalized EEX {column} values are invalid"
        )


__all__ = [
    "DEFAULT_NESTING_TOLERANCE_EUR_MWH",
    "DEFAULT_RECOMPOSITION_TOLERANCE_EUR_MWH",
    "DatabricksEexShortTenorAuditError",
    "DatabricksEexShortTenorAuditResult",
    "EEX_PUBLIC_HOLIDAY_CALENDAR_AS_OF",
    "EEX_PUBLIC_HOLIDAY_CALENDAR_SHA256",
    "EEX_PUBLIC_HOLIDAY_CALENDAR_URL",
    "EXPECTED_COLUMNS",
    "SHORT_TENOR_AUDIT_SCHEMA_VERSION",
    "audit_databricks_eex_short_tenors",
]
