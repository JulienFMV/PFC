"""Offline descriptive horizon coverage for normalized live EEX quotes.

Quotation dates in the local Databricks snapshot are useful for describing
which tenors are observable at which lead times.  They are not authenticated
historical availability timestamps.  This module therefore measures coverage,
lifecycles and quotation gaps without granting point-in-time, rolling-origin,
selection or promotion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_forward_curve import product_periods
from pfc_shaping.validation.databricks_eex_surface_audit import (
    EXPECTED_COLUMNS,
)


HORIZON_AUDIT_SCHEMA_VERSION = "fmv_databricks_eex_horizon_audit.v1"
HORIZON_BUCKETS = (
    "D01_30",
    "D31_90",
    "D91_180",
    "D181_365",
    "D366_730",
    "D731_PLUS",
)
PRODUCT_TYPES = ("Month", "Quarter", "Cal")
LOAD_TYPES = ("BASE", "PEAK")
LONG_GAP_PROXY_THRESHOLD_WEEKDAYS = 20

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MONTH = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")
_QUARTER = re.compile(r"^\d{4}-Q[1-4]$")
_CAL = re.compile(r"^\d{4}$")


class DatabricksEexHorizonAuditError(ValueError):
    """Raised when live normalized EEX history is unsafe to profile."""


@dataclass(frozen=True)
class DatabricksEexHorizonAuditResult:
    """Inspectable descriptive horizon evidence with explicit non-authority."""

    daily_horizon_coverage: pd.DataFrame
    product_lifecycles: pd.DataFrame
    horizon_summary: pd.DataFrame
    summary: Mapping[str, object]


def audit_databricks_eex_horizon_coverage(
    history: pd.DataFrame,
    *,
    source_snapshot_sha256: str,
    source_normalization_content_id: str,
) -> DatabricksEexHorizonAuditResult:
    """Profile lead-time coverage without treating quotation dates as PIT."""

    snapshot_hash = _require_sha256(
        source_snapshot_sha256,
        label="source snapshot",
    )
    normalization_id = _require_sha256(
        source_normalization_content_id,
        label="source normalization content ID",
    )
    observations = _validate_and_enrich_history(
        history,
        source_snapshot_sha256=snapshot_hash,
    )
    daily = _build_daily_horizon_coverage(observations)
    lifecycles = _build_product_lifecycles(observations)
    horizon_summary = _build_horizon_summary(observations, daily=daily)
    summary = _build_summary(
        observations,
        daily=daily,
        lifecycles=lifecycles,
        horizon_summary=horizon_summary,
        source_snapshot_sha256=snapshot_hash,
        source_normalization_content_id=normalization_id,
    )
    return DatabricksEexHorizonAuditResult(
        daily_horizon_coverage=daily,
        product_lifecycles=lifecycles,
        horizon_summary=horizon_summary,
        summary=summary,
    )


def _validate_and_enrich_history(
    history: pd.DataFrame,
    *,
    source_snapshot_sha256: str,
) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise DatabricksEexHorizonAuditError("normalized EEX history is empty")
    if tuple(history.columns) != EXPECTED_COLUMNS:
        raise DatabricksEexHorizonAuditError(
            "normalized EEX history columns are not exact"
        )
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].dt.tz is not None:
        raise DatabricksEexHorizonAuditError(
            "normalized EEX quotation dates are invalid"
        )
    frame["date"] = frame["date"].dt.normalize()
    frame["fact_load_timestamp_utc"] = pd.to_datetime(
        frame["fact_load_timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    if frame["fact_load_timestamp_utc"].isna().any():
        raise DatabricksEexHorizonAuditError(
            "normalized EEX load timestamps are invalid"
        )
    load_dates = (
        frame["fact_load_timestamp_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if bool((frame["date"] > load_dates).any()):
        raise DatabricksEexHorizonAuditError(
            "normalized EEX quotation date follows its load timestamp"
        )
    frame["fact_load_date_utc"] = load_dates
    frame["quotation_to_fact_load_lag_days"] = (
        frame["fact_load_date_utc"] - frame["date"]
    ).dt.days.astype(int)

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
            raise DatabricksEexHorizonAuditError(
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
    _require_allowed_values(frame, "load_type", set(LOAD_TYPES))
    _require_allowed_values(frame, "product_type", set(PRODUCT_TYPES))

    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if not all(math.isfinite(value) for value in frame["price"].to_numpy(float)):
        raise DatabricksEexHorizonAuditError(
            "normalized EEX history contains a non-finite price"
        )
    for column in (
        "point_in_time_available_at_proven",
        "rolling_origin_selection_authorized",
        "production_promotion_authorized",
    ):
        values = frame[column].astype("boolean")
        if values.isna().any() or bool(values.any()):
            raise DatabricksEexHorizonAuditError(
                f"normalized EEX history overclaims {column}"
            )
        frame[column] = values

    identity = ["date", "market", "load_type", "product"]
    if bool(frame.duplicated(identity).any()):
        raise DatabricksEexHorizonAuditError(
            "normalized EEX history repeats a quote identity"
        )

    delivery_starts: list[pd.Timestamp] = []
    for row in frame[["product", "product_type", "load_type"]].itertuples(
        index=False
    ):
        expected_type = _expected_product_type(str(row.product))
        if str(row.product_type) != expected_type:
            raise DatabricksEexHorizonAuditError(
                "normalized EEX product/product_type mismatch"
            )
        delivery_starts.append(
            _delivery_start(str(row.product), str(row.load_type))
        )
    frame["delivery_start_date"] = pd.to_datetime(delivery_starts)
    frame["lead_days"] = (
        frame["delivery_start_date"] - frame["date"]
    ).dt.days.astype(int)
    if bool(frame["lead_days"].le(0).any()):
        raise DatabricksEexHorizonAuditError(
            "normalized live EEX quote does not predate delivery start"
        )
    frame["horizon_bucket"] = frame["lead_days"].map(_horizon_bucket)
    return frame.sort_values(identity, kind="mergesort").reset_index(drop=True)


def _build_daily_horizon_coverage(observations: pd.DataFrame) -> pd.DataFrame:
    observed = (
        observations.groupby(
            ["date", "load_type", "product_type", "horizon_bucket"],
            sort=True,
        )
        .agg(
            quote_count=("product", "size"),
            distinct_product_count=("product", "nunique"),
            min_lead_days=("lead_days", "min"),
            max_lead_days=("lead_days", "max"),
        )
        .reset_index()
    )
    all_dates = pd.Index(
        sorted(observations["date"].drop_duplicates()),
        name="date",
    )
    rows: list[pd.DataFrame] = []
    for load_type, load_frame in observations.groupby("load_type", sort=True):
        load_dates = all_dates[
            (all_dates >= load_frame["date"].min())
            & (all_dates <= load_frame["date"].max())
        ]
        index = pd.MultiIndex.from_product(
            [
                load_dates,
                [str(load_type)],
                PRODUCT_TYPES,
                HORIZON_BUCKETS,
            ],
            names=["date", "load_type", "product_type", "horizon_bucket"],
        )
        rows.append(pd.DataFrame(index=index).reset_index())
    grid = pd.concat(rows, ignore_index=True)
    daily = grid.merge(
        observed,
        how="left",
        on=["date", "load_type", "product_type", "horizon_bucket"],
        validate="one_to_one",
    )
    for column in ("quote_count", "distinct_product_count"):
        daily[column] = daily[column].fillna(0).astype(int)
    for column in ("min_lead_days", "max_lead_days"):
        daily[column] = daily[column].astype("Int64")
    return _sort_horizon_frame(daily, prefix=["date"])


def _build_product_lifecycles(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (load_type, product), group in observations.groupby(
        ["load_type", "product"],
        sort=True,
    ):
        dates = pd.DatetimeIndex(sorted(group["date"].unique()))
        first = pd.Timestamp(dates[0])
        last = pd.Timestamp(dates[-1])
        weekday_quotes = int(sum(date.weekday() < 5 for date in dates))
        weekend_quotes = len(dates) - weekday_quotes
        weekday_denominator = int(
            np.busday_count(
                first.date(),
                (last + pd.Timedelta(days=1)).date(),
            )
        )
        calendar_gaps = [
            int((current - previous).days)
            for previous, current in zip(dates[:-1], dates[1:])
        ]
        gap_records = [
            (
                int(
                    np.busday_count(
                        (previous + pd.Timedelta(days=1)).date(),
                        current.date(),
                    )
                ),
                previous,
                current,
                position,
            )
            for position, (previous, current) in enumerate(
                zip(dates[:-1], dates[1:]),
                start=1,
            )
        ]
        largest_gap = max(
            gap_records,
            default=(0, pd.NaT, pd.NaT, 0),
            key=lambda value: value[0],
        )
        rows.append(
            {
                "market": "CH",
                "load_type": str(load_type),
                "product": str(product),
                "product_type": str(group["product_type"].iloc[0]),
                "delivery_start_date": pd.Timestamp(
                    group["delivery_start_date"].iloc[0]
                ),
                "first_quotation_date": first,
                "last_quotation_date": last,
                "quotation_date_count": len(dates),
                "first_quote_lead_days": int(group["lead_days"].max()),
                "last_quote_lead_days": int(group["lead_days"].min()),
                "median_lead_days": float(group["lead_days"].median()),
                "calendar_span_days": int((last - first).days) + 1,
                "weekday_denominator_proxy": weekday_denominator,
                "weekday_quotation_count": weekday_quotes,
                "weekend_quotation_count": weekend_quotes,
                "weekday_coverage_proxy": (
                    weekday_quotes / weekday_denominator
                    if weekday_denominator
                    else 0.0
                ),
                "median_calendar_gap_days": (
                    float(np.median(calendar_gaps)) if calendar_gaps else 0.0
                ),
                "max_calendar_gap_days": max(calendar_gaps, default=0),
                "max_missing_weekdays_between_quotes_proxy": int(
                    largest_gap[0]
                ),
                "long_gap_proxy_count": int(
                    sum(
                        gap[0] > LONG_GAP_PROXY_THRESHOLD_WEEKDAYS
                        for gap in gap_records
                    )
                ),
                "largest_gap_after_date": largest_gap[1],
                "largest_gap_before_date": largest_gap[2],
                "quotation_count_before_largest_gap": int(largest_gap[3]),
                "quotation_count_after_largest_gap": int(
                    len(dates) - largest_gap[3]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["load_type", "product_type", "delivery_start_date", "product"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_horizon_summary(
    observations: pd.DataFrame,
    *,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (load_type, product_type, horizon_bucket), group in daily.groupby(
        ["load_type", "product_type", "horizon_bucket"],
        sort=False,
    ):
        mask = (
            observations["load_type"].eq(load_type)
            & observations["product_type"].eq(product_type)
            & observations["horizon_bucket"].eq(horizon_bucket)
        )
        matching = observations.loc[mask]
        positive = group["quote_count"].gt(0)
        rows.append(
            {
                "market": "CH",
                "load_type": str(load_type),
                "product_type": str(product_type),
                "horizon_bucket": str(horizon_bucket),
                "active_quotation_date_count": len(group),
                "date_count_with_quotes": int(positive.sum()),
                "date_share_with_quotes": float(positive.mean()),
                "total_quote_count": int(group["quote_count"].sum()),
                "distinct_product_count": int(matching["product"].nunique()),
                "median_daily_quote_count_including_zero": float(
                    group["quote_count"].median()
                ),
                "p90_daily_quote_count_including_zero": float(
                    group["quote_count"].quantile(0.9)
                ),
                "first_quotation_date": (
                    matching["date"].min()
                    if not matching.empty
                    else pd.NaT
                ),
                "last_quotation_date": (
                    matching["date"].max()
                    if not matching.empty
                    else pd.NaT
                ),
                "min_lead_days": (
                    int(matching["lead_days"].min())
                    if not matching.empty
                    else pd.NA
                ),
                "max_lead_days": (
                    int(matching["lead_days"].max())
                    if not matching.empty
                    else pd.NA
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["min_lead_days"] = result["min_lead_days"].astype("Int64")
    result["max_lead_days"] = result["max_lead_days"].astype("Int64")
    return _sort_horizon_frame(result)


def _build_summary(
    observations: pd.DataFrame,
    *,
    daily: pd.DataFrame,
    lifecycles: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    source_snapshot_sha256: str,
    source_normalization_content_id: str,
) -> dict[str, object]:
    latest_date = pd.Timestamp(observations["date"].max())
    latest_daily = daily.loc[daily["date"].eq(latest_date)]
    monthly = observations.loc[observations["product_type"].eq("Month")]
    first_peak = observations.loc[
        observations["load_type"].eq("PEAK"),
        "date",
    ]
    long_gap_lifecycles = lifecycles.loc[
        lifecycles["long_gap_proxy_count"].gt(0)
    ]
    load_timestamps = observations["fact_load_timestamp_utc"]
    load_timestamp_count = int(load_timestamps.nunique())
    load_lags = observations["quotation_to_fact_load_lag_days"]
    delayed_load_share_30d = float(load_lags.gt(30).mean())
    return {
        "schema_version": HORIZON_AUDIT_SCHEMA_VERSION,
        "status": (
            "PASS_LOCAL_DESCRIPTIVE_HORIZON_COVERAGE_WITH_GAP_ANOMALIES_NOT_PIT"
            if not long_gap_lifecycles.empty
            else "PASS_LOCAL_DESCRIPTIVE_HORIZON_COVERAGE_NOT_PIT"
        ),
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_normalization_content_id": source_normalization_content_id,
        "grain": {
            "input_history": "quotation_date_market_load_type_product",
            "daily_horizon_coverage": (
                "quotation_date_load_type_product_type_horizon_bucket"
            ),
            "product_lifecycles": "market_load_type_product",
            "horizon_summary": "market_load_type_product_type_horizon_bucket",
        },
        "horizon_bucket_definition_days": {
            "D01_30": [1, 30],
            "D31_90": [31, 90],
            "D91_180": [91, 180],
            "D181_365": [181, 365],
            "D366_730": [366, 730],
            "D731_PLUS": [731, None],
        },
        "history_row_count": len(observations),
        "quotation_date_count": int(observations["date"].nunique()),
        "quotation_date_min": observations["date"].min().date().isoformat(),
        "quotation_date_max": latest_date.date().isoformat(),
        "first_peak_quotation_date": (
            first_peak.min().date().isoformat()
            if not first_peak.empty
            else None
        ),
        "product_lifecycle_count": len(lifecycles),
        "long_gap_proxy_threshold_weekdays": (
            LONG_GAP_PROXY_THRESHOLD_WEEKDAYS
        ),
        "long_gap_lifecycle_count": len(long_gap_lifecycles),
        "long_gap_lifecycles": _json_records(
            long_gap_lifecycles.loc[
                :,
                [
                    "load_type",
                    "product_type",
                    "product",
                    "quotation_date_count",
                    "max_missing_weekdays_between_quotes_proxy",
                    "largest_gap_after_date",
                    "largest_gap_before_date",
                    "quotation_count_before_largest_gap",
                    "quotation_count_after_largest_gap",
                ],
            ]
        ),
        "lineage_timing": {
            "distinct_fact_load_timestamp_count": load_timestamp_count,
            "fact_load_timestamp_min_utc": load_timestamps.min().isoformat(),
            "fact_load_timestamp_max_utc": load_timestamps.max().isoformat(),
            "single_batch_backfill_signature": load_timestamp_count == 1,
            "quotation_to_fact_load_lag_days_p00": float(
                load_lags.quantile(0.0)
            ),
            "quotation_to_fact_load_lag_days_p50": float(
                load_lags.quantile(0.5)
            ),
            "quotation_to_fact_load_lag_days_p90": float(
                load_lags.quantile(0.9)
            ),
            "quotation_to_fact_load_lag_days_p100": float(
                load_lags.quantile(1.0)
            ),
            "row_share_loaded_more_than_30_days_after_quotation": (
                delayed_load_share_30d
            ),
            "row_share_loaded_more_than_365_days_after_quotation": float(
                load_lags.gt(365).mean()
            ),
            "delayed_load_backfill_or_restatement_signature": (
                delayed_load_share_30d > 0.0
            ),
            "fact_load_timestamp_is_provider_availability": False,
            "historical_pit_inference_blocked": True,
        },
        "lead_days_profile_by_load_and_product_type": _lead_profiles(
            observations
        ),
        "horizon_coverage": _json_records(horizon_summary),
        "direct_monthly_support": {
            "base_max_lead_days": _max_lead(monthly, "BASE"),
            "peak_max_lead_days": _max_lead(monthly, "PEAK"),
            "quote_count_over_180_days": int(monthly["lead_days"].gt(180).sum()),
            "quote_count_over_365_days": int(monthly["lead_days"].gt(365).sum()),
            "direct_monthly_shape_beyond_one_year_observed": bool(
                monthly["lead_days"].gt(365).any()
            ),
        },
        "latest_surface": {
            "date": latest_date.date().isoformat(),
            "quote_count": int(
                observations["date"].eq(latest_date).sum()
            ),
            "coverage": _json_records(
                latest_daily.loc[
                    latest_daily["quote_count"].gt(0),
                    [
                        "load_type",
                        "product_type",
                        "horizon_bucket",
                        "quote_count",
                        "min_lead_days",
                        "max_lead_days",
                    ],
                ]
            ),
        },
        "interpretation": {
            "quotation_date_is_authenticated_availability": False,
            "point_in_time_history_proven": False,
            "descriptive_horizon_coverage_only": True,
            "per_product_fill_forward_used": False,
            "prices_used_in_coverage_metrics": False,
            "weekday_gap_metric_uses_eex_holiday_calendar": False,
            "weekday_gap_metric_is_proxy_only": True,
            "long_gap_proves_market_inactivity": False,
            "long_gap_requires_upstream_explanation": bool(
                not long_gap_lifecycles.empty
            ),
        },
        "authority": {
            "local_research_and_pipeline_development": True,
            "point_in_time_availability_proven": False,
            "rolling_origin_selection_authorized": False,
            "model_selection_authorized": False,
            "production_promotion_authorized": False,
        },
    }


def _lead_profiles(observations: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (load_type, product_type), group in observations.groupby(
        ["load_type", "product_type"],
        sort=True,
    ):
        quantiles = group["lead_days"].quantile([0.0, 0.1, 0.5, 0.9, 1.0])
        rows.append(
            {
                "load_type": str(load_type),
                "product_type": str(product_type),
                "quote_count": len(group),
                "distinct_product_count": int(group["product"].nunique()),
                "lead_days_p00": float(quantiles.loc[0.0]),
                "lead_days_p10": float(quantiles.loc[0.1]),
                "lead_days_p50": float(quantiles.loc[0.5]),
                "lead_days_p90": float(quantiles.loc[0.9]),
                "lead_days_p100": float(quantiles.loc[1.0]),
            }
        )
    return rows


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in frame.to_dict(orient="records"):
        record: dict[str, object] = {}
        for key, value in raw.items():
            if value is pd.NA or pd.isna(value):
                record[str(key)] = None
            elif isinstance(value, pd.Timestamp):
                record[str(key)] = value.date().isoformat()
            elif isinstance(value, (np.integer,)):
                record[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                record[str(key)] = float(value)
            else:
                record[str(key)] = value
        records.append(record)
    return records


def _max_lead(frame: pd.DataFrame, load_type: str) -> int | None:
    subset = frame.loc[frame["load_type"].eq(load_type), "lead_days"]
    return int(subset.max()) if not subset.empty else None


def _sort_horizon_frame(
    frame: pd.DataFrame,
    *,
    prefix: list[str] | None = None,
) -> pd.DataFrame:
    working = frame.assign(
        _load_rank=frame["load_type"].map(
            {value: rank for rank, value in enumerate(LOAD_TYPES)}
        ),
        _product_rank=frame["product_type"].map(
            {value: rank for rank, value in enumerate(PRODUCT_TYPES)}
        ),
        _horizon_rank=frame["horizon_bucket"].map(
            {value: rank for rank, value in enumerate(HORIZON_BUCKETS)}
        ),
    )
    return working.sort_values(
        [
            *(prefix or []),
            "_load_rank",
            "_product_rank",
            "_horizon_rank",
        ],
        kind="mergesort",
    ).drop(columns=["_load_rank", "_product_rank", "_horizon_rank"]).reset_index(
        drop=True
    )


@lru_cache(maxsize=None)
def _delivery_start(product: str, load_type: str) -> pd.Timestamp:
    start = product_periods(product)[0].start_time.normalize()
    if load_type == "PEAK":
        while start.weekday() >= 5:
            start += pd.Timedelta(days=1)
    return start


def _horizon_bucket(lead_days: int) -> str:
    value = int(lead_days)
    if value <= 0:
        raise DatabricksEexHorizonAuditError(
            "live EEX lead days must be strictly positive"
        )
    if value <= 30:
        return "D01_30"
    if value <= 90:
        return "D31_90"
    if value <= 180:
        return "D91_180"
    if value <= 365:
        return "D181_365"
    if value <= 730:
        return "D366_730"
    return "D731_PLUS"


def _expected_product_type(product: str) -> str:
    if _MONTH.fullmatch(product):
        return "Month"
    if _QUARTER.fullmatch(product):
        return "Quarter"
    if _CAL.fullmatch(product):
        return "Cal"
    raise DatabricksEexHorizonAuditError(
        f"normalized EEX product is invalid: {product}"
    )


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise DatabricksEexHorizonAuditError(f"{label} SHA-256 is invalid")
    return normalized


def _require_exact_values(
    frame: pd.DataFrame,
    column: str,
    expected: set[str],
) -> None:
    actual = set(frame[column].astype(str))
    if actual != expected:
        raise DatabricksEexHorizonAuditError(
            f"normalized EEX {column} values are invalid"
        )


def _require_allowed_values(
    frame: pd.DataFrame,
    column: str,
    allowed: set[str],
) -> None:
    actual = set(frame[column].astype(str))
    if not actual or not actual.issubset(allowed):
        raise DatabricksEexHorizonAuditError(
            f"normalized EEX {column} values are invalid"
        )


__all__ = [
    "DatabricksEexHorizonAuditError",
    "DatabricksEexHorizonAuditResult",
    "EXPECTED_COLUMNS",
    "HORIZON_AUDIT_SCHEMA_VERSION",
    "HORIZON_BUCKETS",
    "LONG_GAP_PROXY_THRESHOLD_WEEKDAYS",
    "LOAD_TYPES",
    "PRODUCT_TYPES",
    "audit_databricks_eex_horizon_coverage",
]
