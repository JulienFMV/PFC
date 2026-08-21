"""Offline normalization of one local Databricks EEX daily snapshot.

This module deliberately has no Databricks connector or network path.  It
turns the already captured ``prd.gold`` Swiss power rows into two explicit
layers: all live DAY/WEEK/WEEKEND/MONTH/QUARTER/YEAR quotes for research, and
the CAL/Q/M-only view consumed by the LT monthly solver.  Rows observed on or
after delivery start and rows with invalid delivery boundaries are quarantined
rather than silently admitted or deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

import pandas as pd


NORMALIZATION_SCHEMA_VERSION = "fmv_databricks_eex_daily_normalization.v1"

EXPECTED_COLUMNS = (
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
)
SOURCE_KEY_COLUMNS = ("ProductID", "DeliveryPeriodID", "QuotationDateID")
SOLVER_PERIOD_TYPES = ("MONTH", "QUARTER", "YEAR")
SHORT_PERIOD_TYPES = ("DAY", "WEEK", "WEEKEND")
ALLOWED_DELIVERY_PERIOD_TYPES = (
    "DAY",
    "WEEK",
    "WEEKEND",
    *SOLVER_PERIOD_TYPES,
)
ALLOWED_PRODUCT_TYPES = ("BASE", "PEAK")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRODUCT_TYPE_LABEL = {
    "DAY": "Day",
    "WEEK": "Week",
    "WEEKEND": "Weekend",
    "MONTH": "Month",
    "QUARTER": "Quarter",
    "YEAR": "Cal",
}
EEX_PUBLIC_CONTRACT_DETAILS_AS_OF = "2026-07-22"
EEX_PUBLIC_CONTRACT_DETAILS_SHA256 = (
    "e03e51125b5e0b76668bc66bd736351799323abcb57b85e940ea574cd9ff1232"
)
EEX_PUBLIC_CONTRACT_DETAILS_URL = (
    "https://www.eex.com/fileadmin/EEX/Downloads/Trading/Specifications/"
    "Contract_Details/eex-contract-details-xls-data_20260722.xlsx"
)
SHORT_TENOR_CONTRACT_SEMANTICS = {
    "DAY": {
        "BASE": {
            "delivery_days": 1,
            "contract_size_mwh_per_mw": [23, 24, 25],
            "delivery_day_rule": "ALL_WEEKDAYS",
        },
        "PEAK": {
            "delivery_days": 1,
            "contract_size_mwh_per_mw": [12],
            "delivery_day_rule": "ALL_WEEKDAYS",
        },
    },
    "WEEK": {
        "BASE": {
            "delivery_days": 7,
            "contract_size_mwh_per_mw": [167, 168, 169],
            "delivery_day_rule": "MONDAY_THROUGH_SUNDAY",
        },
        "PEAK": {
            "delivery_days": 5,
            "contract_size_mwh_per_mw": [60],
            "delivery_day_rule": "MONDAY_THROUGH_FRIDAY",
        },
    },
    "WEEKEND": {
        "BASE": {
            "delivery_days": 2,
            "contract_size_mwh_per_mw": [47, 48, 49],
            "delivery_day_rule": "SATURDAY_THROUGH_SUNDAY",
        },
        "PEAK": {
            "delivery_days": 2,
            "contract_size_mwh_per_mw": [24],
            "delivery_day_rule": "SATURDAY_THROUGH_SUNDAY",
        },
    },
}


class DatabricksEexDailyNormalizationError(ValueError):
    """Raised when a local snapshot cannot be normalized without ambiguity."""


@dataclass(frozen=True)
class NormalizedDatabricksEexDailySnapshot:
    """Complete live research layer, solver view, quarantine, and audit."""

    all_products_history: pd.DataFrame
    history: pd.DataFrame
    latest_all_products_surface: pd.DataFrame
    latest_surface: pd.DataFrame
    quarantine: pd.DataFrame
    audit: Mapping[str, object]


def normalize_databricks_eex_daily_snapshot(
    frame: pd.DataFrame,
    *,
    source_snapshot_sha256: str,
    as_of_date: str | pd.Timestamp | None = None,
) -> NormalizedDatabricksEexDailySnapshot:
    """Normalize captured CH EEX daily rows without granting PIT authority.

    DAY, WEEK and WEEKEND are retained in ``all_products_history`` but remain
    outside ``history``, the explicit CAL/Q/M monthly-solver view. Settlement
    prices are authoritative within this local transformation; sparse last
    prices are never used as a fallback. A quotation must predate delivery
    start, consistent with the public EEX contract-details file bound below.
    """

    source_hash = str(source_snapshot_sha256).strip().lower()
    if _SHA256.fullmatch(source_hash) is None:
        raise DatabricksEexDailyNormalizationError(
            "source snapshot SHA-256 is invalid"
        )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DatabricksEexDailyNormalizationError("EEX daily snapshot is empty")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise DatabricksEexDailyNormalizationError(
            "EEX daily snapshot columns are not exact"
        )

    source = frame.copy()
    for column in ("ProductID", "DeliveryPeriodID"):
        source[column] = source[column].astype("string").str.strip()
        if source[column].isna().any() or bool(source[column].eq("").any()):
            raise DatabricksEexDailyNormalizationError(
                f"EEX daily snapshot has an empty {column}"
            )

    for column in ("QuotationDateID", "DeliveryStartDate", "DeliveryEndDate"):
        source[column] = pd.to_datetime(source[column], errors="coerce")
        if source[column].isna().any():
            raise DatabricksEexDailyNormalizationError(
                f"EEX daily snapshot has an invalid {column}"
            )
        if source[column].dt.tz is not None:
            raise DatabricksEexDailyNormalizationError(
                f"EEX daily snapshot {column} must be a calendar date"
            )
        source[column] = source[column].dt.normalize()

    source["FactLoadTimestampUtc"] = pd.to_datetime(
        source["FactLoadTimestampUtc"], errors="coerce", utc=True
    )
    if source["FactLoadTimestampUtc"].isna().any():
        raise DatabricksEexDailyNormalizationError(
            "EEX daily snapshot has an invalid FactLoadTimestampUtc"
        )

    for column in ("SettlementPriceEurMWh", "LastPriceEurMWh"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    settlement = source["SettlementPriceEurMWh"].to_numpy(dtype=float)
    if not all(math.isfinite(value) for value in settlement):
        raise DatabricksEexDailyNormalizationError(
            "EEX daily snapshot has a missing or non-finite settlement price"
        )

    _require_exact_values(source, "Country", {"CH"})
    _require_exact_values(source, "Commodity", {"POWER"})
    _require_allowed_values(source, "ProductType", set(ALLOWED_PRODUCT_TYPES))
    _require_allowed_values(
        source,
        "DeliveryPeriodType",
        set(ALLOWED_DELIVERY_PERIOD_TYPES),
    )
    if bool((source["DeliveryEndDate"] < source["DeliveryStartDate"]).any()):
        raise DatabricksEexDailyNormalizationError(
            "EEX delivery period ends before it starts"
        )
    if bool(source.duplicated(list(SOURCE_KEY_COLUMNS)).any()):
        raise DatabricksEexDailyNormalizationError(
            "EEX daily snapshot repeats a source quote key"
        )
    load_dates = (
        source["FactLoadTimestampUtc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    )
    if bool((source["QuotationDateID"] > load_dates).any()):
        raise DatabricksEexDailyNormalizationError(
            "EEX quotation date is later than its fact load timestamp"
        )

    canonical_products: list[str] = []
    quarantine_reasons: list[str] = []
    for row in source.itertuples(index=False):
        start = pd.Timestamp(row.DeliveryStartDate)
        end = pd.Timestamp(row.DeliveryEndDate)
        period_type = str(row.DeliveryPeriodType)
        load_type = str(row.ProductType)
        product, expected_start, expected_end = _canonical_product_and_bounds(
            start,
            period_type,
            load_type,
        )
        reasons: list[str] = []
        if start != expected_start or end != expected_end:
            reasons.append("DELIVERY_BOUNDARY_MISMATCH")
        if pd.Timestamp(row.QuotationDateID) >= start:
            reasons.append("QUOTATION_NOT_BEFORE_DELIVERY_START")
        canonical_products.append(product)
        quarantine_reasons.append("|".join(reasons))

    source["canonical_product"] = pd.array(canonical_products, dtype="string")
    source["quarantine_reason"] = pd.array(quarantine_reasons, dtype="string")
    quarantine = source.loc[source["quarantine_reason"].ne("")].copy()
    admitted = source.loc[source["quarantine_reason"].eq("")].copy()
    if admitted.empty:
        raise DatabricksEexDailyNormalizationError(
            "EEX daily snapshot has no live quote rows after quarantine"
        )

    all_products_history = _build_normalized_history(
        admitted,
        source_hash=source_hash,
    )
    history = all_products_history.loc[
        all_products_history["product_type"].isin({"Month", "Quarter", "Cal"})
    ].copy().reset_index(drop=True)
    if history.empty:
        raise DatabricksEexDailyNormalizationError(
            "EEX daily snapshot has no live CAL/Q/M rows after quarantine"
        )

    latest_all_products = select_latest_quote_surface(
        all_products_history,
        as_of_date=as_of_date,
    )
    latest = select_latest_quote_surface(history, as_of_date=as_of_date)

    audit = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "status": (
            "PASS_LOCAL_ALL_PRODUCTS_NORMALIZATION_WITH_QUARANTINE_"
            "NOT_PIT_OR_PROMOTION_AUTHORITY"
        ),
        "source_snapshot_sha256": source_hash,
        "input_row_count": len(source),
        "all_products_live_row_count": len(all_products_history),
        "normalized_row_count": len(history),
        "non_solver_live_row_count": len(all_products_history) - len(history),
        "quarantined_row_count": len(quarantine),
        "quarantined_not_before_delivery_count": int(
            quarantine["quarantine_reason"]
            .str.contains("QUOTATION_NOT_BEFORE_DELIVERY_START", regex=False)
            .sum()
        ),
        "quarantined_boundary_mismatch_count": int(
            quarantine["quarantine_reason"]
            .str.contains("DELIVERY_BOUNDARY_MISMATCH", regex=False)
            .sum()
        ),
        "quotation_date_min": history["date"].min().date().isoformat(),
        "quotation_date_max": history["date"].max().date().isoformat(),
        "selected_surface_date": latest["date"].iloc[0].date().isoformat(),
        "selected_surface_row_count": len(latest),
        "selected_all_products_surface_date": (
            latest_all_products["date"].iloc[0].date().isoformat()
        ),
        "selected_all_products_surface_row_count": len(latest_all_products),
        "normalized_identity_duplicate_count": 0,
        "settlement_price_used": True,
        "last_price_used": False,
        "zero_and_negative_settlement_prices_preserved": True,
        "quotation_must_precede_delivery_start": True,
        "public_contract_details": {
            "as_of": EEX_PUBLIC_CONTRACT_DETAILS_AS_OF,
            "url": EEX_PUBLIC_CONTRACT_DETAILS_URL,
            "sha256": EEX_PUBLIC_CONTRACT_DETAILS_SHA256,
        },
        "short_tenor_contract_semantics": {
            period_type: {
                load_type: {
                    key: list(value) if isinstance(value, list) else value
                    for key, value in semantics.items()
                }
                for load_type, semantics in load_semantics.items()
            }
            for period_type, load_semantics in SHORT_TENOR_CONTRACT_SEMANTICS.items()
        },
        "rows_by_load_and_product_type_all_products": _segment_counts(
            all_products_history
        ),
        "rows_by_load_and_product_type": _segment_counts(history),
        "authority": {
            "local_research_and_pipeline_development": True,
            "point_in_time_availability_proven": False,
            "rolling_origin_selection_authorized": False,
            "production_promotion_authorized": False,
        },
    }
    return NormalizedDatabricksEexDailySnapshot(
        all_products_history=all_products_history,
        history=history,
        latest_all_products_surface=latest_all_products,
        latest_surface=latest,
        quarantine=quarantine.reset_index(drop=True),
        audit=audit,
    )


def select_latest_quote_surface(
    history: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Select one complete observed quotation date without per-product fill-forward."""

    required = {"date", "market", "load_type", "product", "price"}
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise DatabricksEexDailyNormalizationError("normalized EEX history is empty")
    if not required.issubset(history.columns):
        raise DatabricksEexDailyNormalizationError(
            "normalized EEX history is missing quote columns"
        )
    dates = pd.to_datetime(history["date"], errors="coerce")
    if dates.isna().any() or dates.dt.tz is not None:
        raise DatabricksEexDailyNormalizationError(
            "normalized EEX history dates are invalid"
        )
    if as_of_date is None:
        cutoff = dates.max()
    else:
        cutoff = pd.Timestamp(as_of_date)
        if cutoff.tzinfo is not None:
            raise DatabricksEexDailyNormalizationError(
                "EEX as-of quotation date must be timezone-naive"
            )
        cutoff = cutoff.normalize()
    available = history.loc[dates.le(cutoff)].copy()
    if available.empty:
        raise DatabricksEexDailyNormalizationError(
            "no EEX quotation surface exists at or before the requested date"
        )
    selected_date = pd.to_datetime(available["date"]).max().normalize()
    surface = available.loc[pd.to_datetime(available["date"]).eq(selected_date)].copy()
    identity = ["market", "load_type", "product"]
    if bool(surface.duplicated(identity).any()):
        raise DatabricksEexDailyNormalizationError(
            "selected EEX quote surface repeats a product identity"
        )
    return surface.sort_values(identity, kind="mergesort").reset_index(drop=True)


def split_solver_quote_maps(surface: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return explicit BASE and PEAK product mappings for local solver research."""

    required = {"market", "load_type", "product", "price"}
    if not isinstance(surface, pd.DataFrame) or surface.empty:
        raise DatabricksEexDailyNormalizationError("EEX quote surface is empty")
    if not required.issubset(surface.columns):
        raise DatabricksEexDailyNormalizationError(
            "EEX quote surface is missing solver columns"
        )
    working = surface.copy()
    _require_exact_values(working, "market", {"CH"})
    _require_allowed_values(working, "load_type", set(ALLOWED_PRODUCT_TYPES))
    if "product_type" not in working.columns:
        raise DatabricksEexDailyNormalizationError(
            "EEX quote surface is missing product_type"
        )
    _require_allowed_values(
        working,
        "product_type",
        {"MONTH", "QUARTER", "CAL"},
    )
    if bool(working.duplicated(["load_type", "product"]).any()):
        raise DatabricksEexDailyNormalizationError(
            "EEX quote surface repeats a solver product"
        )
    result: dict[str, dict[str, float]] = {"BASE": {}, "PEAK": {}}
    for row in working.itertuples(index=False):
        price = float(row.price)
        if not math.isfinite(price):
            raise DatabricksEexDailyNormalizationError(
                "EEX quote surface has a non-finite solver price"
            )
        result[str(row.load_type)][str(row.product)] = price
    return result


def _build_normalized_history(
    eligible: pd.DataFrame,
    *,
    source_hash: str,
) -> pd.DataFrame:
    history = pd.DataFrame(
        {
            "date": eligible["QuotationDateID"].to_numpy(),
            "product": pd.array(eligible["canonical_product"], dtype="string"),
            "load_type": pd.array(eligible["ProductType"], dtype="string"),
            "product_type": pd.array(
                eligible["DeliveryPeriodType"].map(_PRODUCT_TYPE_LABEL),
                dtype="string",
            ),
            "market": pd.array(["CH"] * len(eligible), dtype="string"),
            "price": eligible["SettlementPriceEurMWh"].astype(float).to_numpy(),
            "unit": pd.array(["EUR/MWH"] * len(eligible), dtype="string"),
            "source": pd.array(["EEX"] * len(eligible), dtype="string"),
            "source_system": pd.array(
                ["DATABRICKS_PRD_GOLD"] * len(eligible), dtype="string"
            ),
            "product_id": pd.array(eligible["ProductID"], dtype="string"),
            "delivery_period_id": pd.array(
                eligible["DeliveryPeriodID"], dtype="string"
            ),
            "fact_load_timestamp_utc": pd.array(
                eligible["FactLoadTimestampUtc"], dtype="datetime64[ns, UTC]"
            ),
            "source_snapshot_sha256": pd.array(
                [source_hash] * len(eligible), dtype="string"
            ),
            "point_in_time_available_at_proven": pd.array(
                [False] * len(eligible), dtype="boolean"
            ),
            "rolling_origin_selection_authorized": pd.array(
                [False] * len(eligible), dtype="boolean"
            ),
            "production_promotion_authorized": pd.array(
                [False] * len(eligible), dtype="boolean"
            ),
        }
    )
    identity = ["date", "market", "load_type", "product"]
    if bool(history.duplicated(identity).any()):
        raise DatabricksEexDailyNormalizationError(
            "normalized EEX daily history repeats a quote identity"
        )
    return history.sort_values(identity, kind="mergesort").reset_index(drop=True)


def _canonical_product_and_bounds(
    start: pd.Timestamp,
    period_type: str,
    load_type: str,
) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    if period_type == "DAY":
        product = f"D:{start.date().isoformat()}"
        calendar_start = calendar_end = start
    elif period_type == "WEEK":
        iso_year, iso_week, _ = start.isocalendar()
        product = f"W:{iso_year}-W{iso_week:02d}"
        calendar_start = start
        calendar_end = start + pd.Timedelta(days=6)
    elif period_type == "WEEKEND":
        iso_year, iso_week, _ = start.isocalendar()
        product = f"WE:{iso_year}-W{iso_week:02d}"
        calendar_start = start
        calendar_end = start + pd.Timedelta(days=1)
    elif period_type == "MONTH":
        period = start.to_period("M")
        product = period.strftime("%Y-%m")
        calendar_start, calendar_end = (
            period.start_time.normalize(),
            period.end_time.normalize(),
        )
    elif period_type == "QUARTER":
        period = start.to_period("Q-DEC")
        product = f"{period.year}-Q{period.quarter}"
        calendar_start, calendar_end = (
            period.start_time.normalize(),
            period.end_time.normalize(),
        )
    elif period_type == "YEAR":
        period = start.to_period("Y-DEC")
        product = f"{period.year}"
        calendar_start, calendar_end = (
            period.start_time.normalize(),
            period.end_time.normalize(),
        )
    else:
        raise DatabricksEexDailyNormalizationError(
            f"unsupported delivery period type: {period_type}"
        )
    if period_type == "WEEK":
        if start.weekday() != 0:
            return product, start - pd.Timedelta(days=start.weekday()), calendar_end
        if load_type == "PEAK":
            calendar_end = start + pd.Timedelta(days=4)
    elif period_type == "WEEKEND":
        if start.weekday() != 5:
            return product, start + pd.Timedelta(days=(5 - start.weekday()) % 7), calendar_end
    elif load_type == "PEAK" and period_type in SOLVER_PERIOD_TYPES:
        calendar_start = _first_weekday(calendar_start)
        calendar_end = _last_weekday(calendar_end)
    return product, calendar_start, calendar_end


def _first_weekday(value: pd.Timestamp) -> pd.Timestamp:
    result = value
    while result.weekday() >= 5:
        result += pd.Timedelta(days=1)
    return result


def _last_weekday(value: pd.Timestamp) -> pd.Timestamp:
    result = value
    while result.weekday() >= 5:
        result -= pd.Timedelta(days=1)
    return result


def _require_exact_values(
    frame: pd.DataFrame,
    column: str,
    expected: set[str],
) -> None:
    values = frame[column].astype("string").str.strip().str.upper()
    if values.isna().any() or set(values.astype(str)) != expected:
        raise DatabricksEexDailyNormalizationError(
            f"EEX daily snapshot {column} values are invalid"
        )
    frame[column] = values


def _require_allowed_values(
    frame: pd.DataFrame,
    column: str,
    allowed: set[str],
) -> None:
    values = frame[column].astype("string").str.strip().str.upper()
    actual = set(values.dropna().astype(str))
    if values.isna().any() or not actual or not actual.issubset(allowed):
        raise DatabricksEexDailyNormalizationError(
            f"EEX daily snapshot {column} values are invalid"
        )
    frame[column] = values


def _segment_counts(history: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "load_type": str(load_type),
            "product_type": str(product_type),
            "row_count": int(len(group)),
        }
        for (load_type, product_type), group in history.groupby(
            ["load_type", "product_type"], sort=True
        )
    ]


__all__ = [
    "ALLOWED_DELIVERY_PERIOD_TYPES",
    "ALLOWED_PRODUCT_TYPES",
    "DatabricksEexDailyNormalizationError",
    "EEX_PUBLIC_CONTRACT_DETAILS_AS_OF",
    "EEX_PUBLIC_CONTRACT_DETAILS_SHA256",
    "EEX_PUBLIC_CONTRACT_DETAILS_URL",
    "EXPECTED_COLUMNS",
    "NORMALIZATION_SCHEMA_VERSION",
    "NormalizedDatabricksEexDailySnapshot",
    "SHORT_PERIOD_TYPES",
    "SHORT_TENOR_CONTRACT_SEMANTICS",
    "SOLVER_PERIOD_TYPES",
    "SOURCE_KEY_COLUMNS",
    "normalize_databricks_eex_daily_snapshot",
    "select_latest_quote_surface",
    "split_solver_quote_maps",
]
