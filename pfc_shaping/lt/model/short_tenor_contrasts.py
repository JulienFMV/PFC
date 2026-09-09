"""Pure construction of dormant EEX DAY/WEEK/WEEKEND contrast observations.

The output keeps every contrast family separate and at quoted native grain.
It performs no interpolation, amplitude fitting, component blending, time-series
forecasting, solver assembly, or production activation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date as calendar_date
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.shape_constraints import (
    interval_hours,
    validate_utc_index,
)

SHORT_TENOR_CONTRAST_CONTRACT_VERSION = (
    "fmv-eex-ch-short-tenor-contrast-construction-v1"
)
SHORT_TENOR_CONTRAST_SCHEMA_VERSION = "fmv_eex_ch_short_tenor_contrast_panel.v1"
LOCAL_TIMEZONE = "Europe/Zurich"
PARENT_CHILD_TOLERANCE_EUR_MWH = 0.01
MAX_ZERO_PARENT_MEAN_RESIDUAL_EUR_MWH = 1e-10

NORMALIZED_HISTORY_COLUMNS = (
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
CONTRAST_COLUMNS = (
    "quotation_date",
    "contrast_id",
    "family",
    "parent_product_type",
    "parent_product",
    "source_load_type",
    "source_product",
    "source_role",
    "delivery_day",
    "short_block",
    "priced_segment",
    "delivery_hours",
    "delta_eur_mwh",
)
DIAGNOSTIC_COLUMNS = (
    "quotation_date",
    "contrast_id",
    "family",
    "parent_product_type",
    "parent_product",
    "load_type",
    "expected_component_count",
    "observed_component_count",
    "parent_delivery_hours",
    "recomposition_residual_eur_mwh",
    "abs_recomposition_residual_eur_mwh",
    "negative_implied_offpeak",
    "status",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DAY = re.compile(r"^D:(\d{4}-\d{2}-\d{2})$")
_WEEK = re.compile(r"^W:(\d{4})-W(\d{2})$")
_WEEKEND = re.compile(r"^WE:(\d{4})-W(\d{2})$")
_SHORT_TYPES = {"Day", "Week", "Weekend"}
_ALL_TYPES = {*_SHORT_TYPES, "Month", "Quarter", "Cal"}
_LOAD_TYPES = {"BASE", "PEAK"}


@dataclass(frozen=True)
class ShortTenorContrastPanel:
    """Separate native-grain contrast components and audit diagnostics."""

    contrasts: pd.DataFrame
    diagnostics: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class _ProductMetadata:
    product_type: str
    product: str
    start: pd.Timestamp
    end: pd.Timestamp
    delivery_days: tuple[pd.Timestamp, ...]
    peak_delivery_days: tuple[pd.Timestamp, ...]
    total_hours: int
    peak_hours: int
    offpeak_hours: int


def build_short_tenor_contrast_panel(
    normalized_history: pd.DataFrame,
    *,
    source_normalization_content_id: str,
    source_audit_content_id: str,
) -> ShortTenorContrastPanel:
    """Construct same-vintage, zero-parent-mean short-tenor observations.

    Parent/child quote inconsistencies and incomplete strips are retained as
    diagnostics but never emitted as contrast rows.
    """

    _validate_content_id(
        source_normalization_content_id,
        "source_normalization_content_id",
    )
    _validate_content_id(source_audit_content_id, "source_audit_content_id")
    short, source_snapshot_sha256 = _validate_and_select_short_history(
        normalized_history
    )
    lookup = {
        (pd.Timestamp(row.date), str(row.load_type), str(row.product)): row
        for row in short.itertuples(index=False)
    }
    metadata = {
        (str(row.product_type), str(row.product)): _parse_product(
            str(row.product), str(row.product_type)
        )
        for row in short.drop_duplicates(["product_type", "product"])
        .sort_values(["product_type", "product"], kind="mergesort")
        .itertuples(index=False)
    }

    contrast_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    _construct_day_within_parent_families(
        short,
        lookup=lookup,
        metadata=metadata,
        contrast_rows=contrast_rows,
        diagnostic_rows=diagnostic_rows,
    )
    _construct_weekend_vs_weekday_base(
        short,
        lookup=lookup,
        metadata=metadata,
        contrast_rows=contrast_rows,
        diagnostic_rows=diagnostic_rows,
    )
    negative_implied_offpeak_count = _construct_base_peak_offpeak(
        short,
        lookup=lookup,
        metadata=metadata,
        contrast_rows=contrast_rows,
        diagnostic_rows=diagnostic_rows,
    )

    contrasts = pd.DataFrame(contrast_rows, columns=CONTRAST_COLUMNS)
    diagnostics = pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS)
    if not contrasts.empty:
        contrasts = contrasts.sort_values(
            ["quotation_date", "family", "contrast_id", "delivery_day", "short_block"],
            kind="mergesort",
            ignore_index=True,
        )
        _validate_contrast_output(contrasts)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(
            ["quotation_date", "family", "contrast_id"],
            kind="mergesort",
            ignore_index=True,
        )

    zero_means = _contrast_weighted_means(contrasts)
    max_zero_mean_residual = (
        float(zero_means["abs_weighted_mean_eur_mwh"].max())
        if not zero_means.empty
        else 0.0
    )
    if max_zero_mean_residual > MAX_ZERO_PARENT_MEAN_RESIDUAL_EUR_MWH:
        raise RuntimeError(
            "constructed short-tenor contrast breaches zero-parent-mean tolerance: "
            f"{max_zero_mean_residual:.12g} EUR/MWh"
        )

    accepted = diagnostics.loc[diagnostics["status"].eq("ACCEPTED")]
    accepted_strip = accepted.loc[
        accepted["family"].isin({"DAY_WITHIN_WEEK", "DAY_WITHIN_WEEKEND"})
    ]
    max_accepted_strip_residual = (
        float(accepted_strip["abs_recomposition_residual_eur_mwh"].max())
        if not accepted_strip.empty
        else 0.0
    )
    audit: dict[str, object] = {
        "schema_version": SHORT_TENOR_CONTRAST_SCHEMA_VERSION,
        "contract_version": SHORT_TENOR_CONTRAST_CONTRACT_VERSION,
        "status": "PASS_LOCAL_DESCRIPTIVE_CONSTRUCTION_ONLY_NO_MODEL_AUTHORITY",
        "source_normalization_content_id": source_normalization_content_id,
        "source_audit_content_id": source_audit_content_id,
        "source_snapshot_sha256": source_snapshot_sha256,
        "input_short_row_count": int(len(short)),
        "quotation_date_count": int(short["date"].nunique()),
        "quotation_date_min": short["date"].min().date().isoformat(),
        "quotation_date_max": short["date"].max().date().isoformat(),
        "contrast_count": int(contrasts["contrast_id"].nunique()),
        "contrast_row_count": int(len(contrasts)),
        "contrast_count_by_family": _counts_by_value(
            contrasts.drop_duplicates("contrast_id"), "family"
        ),
        "contrast_row_count_by_family": _counts_by_value(contrasts, "family"),
        "diagnostic_count": int(len(diagnostics)),
        "diagnostic_count_by_status": _counts_by_value(diagnostics, "status"),
        "max_abs_zero_parent_mean_residual_eur_mwh": max_zero_mean_residual,
        "max_abs_accepted_parent_child_residual_eur_mwh": max_accepted_strip_residual,
        "parent_child_tolerance_eur_mwh": PARENT_CHILD_TOLERANCE_EUR_MWH,
        "negative_implied_offpeak_contrast_count": int(
            negative_implied_offpeak_count
        ),
        "same_vintage_required": True,
        "complete_non_overlapping_strips_required": True,
        "per_product_fill_forward_used": False,
        "synthetic_quote_created": False,
        "multiplicative_ratio_used": False,
        "components_combined": False,
        "amplitude_fitted": False,
        "point_in_time_availability_proven": False,
        "rolling_origin_selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "production_authorized": False,
        "ompex_used": False,
        "afry_numeric_values_used": False,
    }
    return ShortTenorContrastPanel(
        contrasts=contrasts,
        diagnostics=diagnostics,
        audit=audit,
    )


def contrast_weighted_means(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Return one hour-weighted mean per separate contrast component."""

    if not isinstance(contrasts, pd.DataFrame):
        raise TypeError("contrasts must be a pandas DataFrame")
    if tuple(contrasts.columns) != CONTRAST_COLUMNS:
        raise ValueError("contrast columns are not exact")
    return _contrast_weighted_means(contrasts)


def materialize_short_tenor_contrast_signal(
    contrasts: pd.DataFrame,
    *,
    contrast_id: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Embed one separate native-grain contrast in complete local months.

    Intervals outside the quoted component support receive the additive
    identity value zero.  This is a sparse component embedding, not a quote
    fill, forecast, interpolation, component blend, or activation decision.
    """

    if not isinstance(contrasts, pd.DataFrame):
        raise TypeError("contrasts must be a pandas DataFrame")
    if tuple(contrasts.columns) != CONTRAST_COLUMNS:
        raise ValueError("contrast columns are not exact")
    if not isinstance(contrast_id, str) or not contrast_id.strip():
        raise ValueError("contrast_id must be a non-empty string")
    _validate_contrast_output(contrasts)
    selected = contrasts.loc[contrasts["contrast_id"].eq(contrast_id)].copy()
    if selected.empty:
        raise ValueError(f"contrast_id is absent from the panel: {contrast_id}")
    if len(_contrast_weighted_means(selected)) != 1:
        raise ValueError("contrast_id does not identify exactly one component")
    weighted_mean = float(
        _contrast_weighted_means(selected).iloc[0]["abs_weighted_mean_eur_mwh"]
    )
    if weighted_mean > MAX_ZERO_PARENT_MEAN_RESIDUAL_EUR_MWH:
        raise ValueError(
            "contrast component is not hour-weighted zero-parent-mean: "
            f"{weighted_mean:.12g} EUR/MWh"
        )

    utc = validate_utc_index(index)
    hours = interval_hours(utc)
    _require_complete_local_month_index(utc, hours=hours)
    local = utc.tz_convert(LOCAL_TIMEZONE)
    local_days = local.strftime("%Y-%m-%d")
    local_is_peak = (local.hour >= 8) & (local.hour < 20)
    values = np.zeros(len(utc), dtype=float)
    assigned = np.zeros(len(utc), dtype=bool)

    delivery_days = pd.to_datetime(selected["delivery_day"], errors="coerce")
    if delivery_days.isna().any() or delivery_days.dt.tz is not None:
        raise ValueError("contrast delivery_day values are invalid")
    if not delivery_days.eq(delivery_days.dt.normalize()).all():
        raise ValueError("contrast delivery_day values must be calendar dates")
    selected["delivery_day"] = delivery_days

    for row in selected.itertuples(index=False):
        day_mask = local_days == pd.Timestamp(row.delivery_day).date().isoformat()
        block_mask = local_is_peak if row.short_block == "SHORT_PEAK" else ~local_is_peak
        mask = np.asarray(day_mask & block_mask, dtype=bool)
        if not mask.any():
            raise ValueError(
                "complete-month delivery index does not contain contrast support "
                f"for {row.delivery_day} {row.short_block}"
            )
        observed_hours = float(hours[mask].sum())
        if not np.isclose(
            observed_hours,
            float(row.delivery_hours),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "contrast delivery hours do not match the target index: "
                f"{observed_hours:.12g} != {float(row.delivery_hours):.12g}"
            )
        if assigned[mask].any():
            raise ValueError("contrast component has overlapping native-grain cells")
        values[mask] = float(row.delta_eur_mwh)
        assigned[mask] = True

    return pd.Series(
        values,
        index=utc,
        name="short_tenor_raw_delta_eur_mwh",
        dtype=float,
    )


def _require_complete_local_month_index(
    utc: pd.DatetimeIndex,
    *,
    hours: np.ndarray,
) -> None:
    if len(utc) < 2 or len(hours) != len(utc):
        raise ValueError("delivery index must cover at least one complete local month")
    cadence_hours = float(hours[0])
    frequency = "1h" if np.isclose(cadence_hours, 1.0) else "15min"
    local = utc.tz_convert(LOCAL_TIMEZONE)
    first_month = str(local[0].strftime("%Y-%m"))
    last_month = str(local[-1].strftime("%Y-%m"))
    first_local = pd.Timestamp(f"{first_month}-01", tz=LOCAL_TIMEZONE)
    after_last_local = (
        pd.Timestamp(f"{last_month}-01", tz=LOCAL_TIMEZONE)
        + pd.offsets.MonthBegin(1)
    )
    expected = pd.date_range(
        first_local.tz_convert("UTC"),
        after_last_local.tz_convert("UTC"),
        freq=frequency,
        inclusive="left",
    )
    if not utc.equals(expected):
        raise ValueError("delivery index must contain contiguous complete local calendar months")


def _validate_and_select_short_history(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("normalized EEX history is empty")
    if tuple(frame.columns) != NORMALIZED_HISTORY_COLUMNS:
        raise ValueError("normalized EEX history columns are not exact")
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    if working["date"].isna().any() or working["date"].dt.tz is not None:
        raise ValueError("normalized EEX quotation dates are invalid")
    if not working["date"].eq(working["date"].dt.normalize()).all():
        raise ValueError("normalized EEX quotation dates must be calendar dates")
    working["fact_load_timestamp_utc"] = pd.to_datetime(
        working["fact_load_timestamp_utc"], errors="coerce", utc=True
    )
    if working["fact_load_timestamp_utc"].isna().any():
        raise ValueError("normalized EEX fact load timestamps are invalid")
    load_dates = (
        working["fact_load_timestamp_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if working["date"].gt(load_dates).any():
        raise ValueError("normalized EEX quotation date follows its load timestamp")

    string_columns = (
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
    )
    for column in string_columns:
        values = working[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"normalized EEX history has an empty {column}")
        working[column] = values
    _require_exact(working, "market", {"CH"})
    _require_exact(working, "unit", {"EUR/MWH"})
    _require_exact(working, "source", {"EEX"})
    _require_exact(working, "source_system", {"DATABRICKS_PRD_GOLD"})
    _require_subset(working, "load_type", _LOAD_TYPES)
    _require_subset(working, "product_type", _ALL_TYPES)

    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    if not all(math.isfinite(value) for value in working["price"].to_numpy(float)):
        raise ValueError("normalized EEX settlement prices are invalid")
    for column in (
        "point_in_time_available_at_proven",
        "rolling_origin_selection_authorized",
        "production_promotion_authorized",
    ):
        flags = working[column].astype("boolean")
        if flags.isna().any() or flags.any():
            raise ValueError(f"normalized EEX authority flag {column} must remain false")
        working[column] = flags

    snapshot_hashes = set(working["source_snapshot_sha256"].astype(str))
    if len(snapshot_hashes) != 1:
        raise ValueError("normalized EEX history mixes source snapshot hashes")
    source_snapshot_sha256 = next(iter(snapshot_hashes))
    _validate_content_id(source_snapshot_sha256, "source_snapshot_sha256")
    identity = ["date", "market", "load_type", "product"]
    if working.duplicated(identity).any():
        raise ValueError("normalized EEX history repeats a quote identity")

    short = working.loc[working["product_type"].isin(_SHORT_TYPES)].copy()
    if short.empty:
        raise ValueError("normalized EEX history has no short-tenor rows")
    for row in short[["date", "product", "product_type"]].itertuples(index=False):
        meta = _parse_product(str(row.product), str(row.product_type))
        if pd.Timestamp(row.date) >= meta.start:
            raise ValueError("normalized EEX short-tenor history contains a non-live quote")
    delivery_ids = short.groupby(["load_type", "product"], sort=False)[
        "delivery_period_id"
    ].nunique()
    if delivery_ids.gt(1).any():
        raise ValueError("one short-tenor identity maps to multiple delivery periods")
    products = short.groupby("delivery_period_id", sort=False)["product"].nunique()
    if products.gt(1).any():
        raise ValueError("one short-tenor delivery period maps to multiple products")
    return (
        short.sort_values(identity, kind="mergesort", ignore_index=True),
        source_snapshot_sha256,
    )


def _construct_day_within_parent_families(
    short: pd.DataFrame,
    *,
    lookup: Mapping[tuple[pd.Timestamp, str, str], object],
    metadata: Mapping[tuple[str, str], _ProductMetadata],
    contrast_rows: list[dict[str, object]],
    diagnostic_rows: list[dict[str, object]],
) -> None:
    parents = short.loc[short["product_type"].isin({"Week", "Weekend"})]
    for parent in parents.itertuples(index=False):
        quotation_date = pd.Timestamp(parent.date)
        load_type = str(parent.load_type)
        parent_type = str(parent.product_type)
        parent_product = str(parent.product)
        family = "DAY_WITHIN_WEEK" if parent_type == "Week" else "DAY_WITHIN_WEEKEND"
        contrast_id = _contrast_id(
            quotation_date,
            family,
            load_type,
            parent_product,
        )
        parent_meta = metadata[(parent_type, parent_product)]
        child_days = (
            parent_meta.peak_delivery_days
            if load_type == "PEAK"
            else parent_meta.delivery_days
        )
        children = [
            lookup.get(
                (
                    quotation_date,
                    load_type,
                    f"D:{day.date().isoformat()}",
                )
            )
            for day in child_days
        ]
        complete_children = [child for child in children if child is not None]
        parent_hours = (
            parent_meta.peak_hours if load_type == "PEAK" else parent_meta.total_hours
        )
        if len(complete_children) != len(child_days):
            diagnostic_rows.append(
                _diagnostic_row(
                    quotation_date=quotation_date,
                    contrast_id=contrast_id,
                    family=family,
                    parent_type=parent_type,
                    parent_product=parent_product,
                    load_type=load_type,
                    expected=len(child_days),
                    observed=len(complete_children),
                    parent_hours=parent_hours,
                    residual=float("nan"),
                    status="REJECT_INCOMPLETE_CHILD_STRIP",
                )
            )
            continue
        child_weights = np.array(
            [
                12.0
                if load_type == "PEAK"
                else float(_local_day_hours(pd.Timestamp(child.product[2:])))
                for child in complete_children
            ],
            dtype=float,
        )
        if not np.isclose(child_weights.sum(), parent_hours, rtol=0.0, atol=1e-12):
            raise RuntimeError("complete DAY strip hours do not match parent contract")
        child_prices = np.array(
            [float(child.price) for child in complete_children], dtype=float
        )
        strip_mean = float(np.average(child_prices, weights=child_weights))
        residual = float(parent.price) - strip_mean
        if abs(residual) > PARENT_CHILD_TOLERANCE_EUR_MWH:
            diagnostic_rows.append(
                _diagnostic_row(
                    quotation_date=quotation_date,
                    contrast_id=contrast_id,
                    family=family,
                    parent_type=parent_type,
                    parent_product=parent_product,
                    load_type=load_type,
                    expected=len(child_days),
                    observed=len(complete_children),
                    parent_hours=parent_hours,
                    residual=residual,
                    status="REJECT_PARENT_CHILD_QUOTE_CONFLICT",
                )
            )
            continue
        diagnostic_rows.append(
            _diagnostic_row(
                quotation_date=quotation_date,
                contrast_id=contrast_id,
                family=family,
                parent_type=parent_type,
                parent_product=parent_product,
                load_type=load_type,
                expected=len(child_days),
                observed=len(complete_children),
                parent_hours=parent_hours,
                residual=residual,
                status="ACCEPTED",
            )
        )
        for child, child_price in zip(complete_children, child_prices, strict=True):
            delivery_day = pd.Timestamp(str(child.product)[2:])
            delta = float(child_price - strip_mean)
            if load_type == "PEAK":
                _append_contrast_cell(
                    contrast_rows,
                    quotation_date=quotation_date,
                    contrast_id=contrast_id,
                    family=family,
                    parent_type=parent_type,
                    parent_product=parent_product,
                    source_load_type=load_type,
                    source_product=str(child.product),
                    source_role="DAY_CHILD",
                    delivery_day=delivery_day,
                    short_block="SHORT_PEAK",
                    priced_segment="PEAK",
                    delivery_hours=12,
                    delta=delta,
                )
            else:
                for short_block, hours in _day_blocks(delivery_day):
                    _append_contrast_cell(
                        contrast_rows,
                        quotation_date=quotation_date,
                        contrast_id=contrast_id,
                        family=family,
                        parent_type=parent_type,
                        parent_product=parent_product,
                        source_load_type=load_type,
                        source_product=str(child.product),
                        source_role="DAY_CHILD",
                        delivery_day=delivery_day,
                        short_block=short_block,
                        priced_segment="BASE",
                        delivery_hours=hours,
                        delta=delta,
                    )


def _construct_weekend_vs_weekday_base(
    short: pd.DataFrame,
    *,
    lookup: Mapping[tuple[pd.Timestamp, str, str], object],
    metadata: Mapping[tuple[str, str], _ProductMetadata],
    contrast_rows: list[dict[str, object]],
    diagnostic_rows: list[dict[str, object]],
) -> None:
    weeks = short.loc[
        short["product_type"].eq("Week") & short["load_type"].eq("BASE")
    ]
    for week in weeks.itertuples(index=False):
        quotation_date = pd.Timestamp(week.date)
        week_product = str(week.product)
        weekend_product = f"WE:{week_product[2:]}"
        family = "WEEKEND_VS_WEEKDAY_BASE"
        contrast_id = _contrast_id(
            quotation_date,
            family,
            "BASE",
            week_product,
        )
        weekend = lookup.get((quotation_date, "BASE", weekend_product))
        week_meta = metadata[("Week", week_product)]
        if weekend is None:
            diagnostic_rows.append(
                _diagnostic_row(
                    quotation_date=quotation_date,
                    contrast_id=contrast_id,
                    family=family,
                    parent_type="Week",
                    parent_product=week_product,
                    load_type="BASE",
                    expected=2,
                    observed=1,
                    parent_hours=week_meta.total_hours,
                    residual=float("nan"),
                    status="REJECT_MISSING_WEEKEND_PARENT",
                )
            )
            continue
        weekend_meta = metadata[("Weekend", weekend_product)]
        if (
            weekend_meta.start != week_meta.start + pd.Timedelta(days=5)
            or weekend_meta.end != week_meta.end
        ):
            raise RuntimeError("WEEKEND parent is not the exact nested WEEK partition")
        weekend_hours = weekend_meta.total_hours
        weekday_hours = week_meta.total_hours - weekend_hours
        if weekday_hours <= 0:
            raise RuntimeError("WEEK weekday partition has non-positive hours")
        implied_weekday = (
            float(week.price) * week_meta.total_hours
            - float(weekend.price) * weekend_hours
        ) / weekday_hours
        weekday_delta = implied_weekday - float(week.price)
        weekend_delta = float(weekend.price) - float(week.price)
        recomposed = (
            implied_weekday * weekday_hours + float(weekend.price) * weekend_hours
        ) / week_meta.total_hours
        residual = recomposed - float(week.price)
        diagnostic_rows.append(
            _diagnostic_row(
                quotation_date=quotation_date,
                contrast_id=contrast_id,
                family=family,
                parent_type="Week",
                parent_product=week_product,
                load_type="BASE",
                expected=2,
                observed=2,
                parent_hours=week_meta.total_hours,
                residual=residual,
                status="ACCEPTED",
            )
        )
        for delivery_day in week_meta.delivery_days:
            is_weekend = delivery_day >= weekend_meta.start
            delta = weekend_delta if is_weekend else weekday_delta
            source_product = weekend_product if is_weekend else week_product
            source_role = "WEEKEND_QUOTE" if is_weekend else "IMPLIED_WEEKDAY"
            for short_block, hours in _day_blocks(delivery_day):
                _append_contrast_cell(
                    contrast_rows,
                    quotation_date=quotation_date,
                    contrast_id=contrast_id,
                    family=family,
                    parent_type="Week",
                    parent_product=week_product,
                    source_load_type="BASE",
                    source_product=source_product,
                    source_role=source_role,
                    delivery_day=delivery_day,
                    short_block=short_block,
                    priced_segment="BASE",
                    delivery_hours=hours,
                    delta=delta,
                )


def _construct_base_peak_offpeak(
    short: pd.DataFrame,
    *,
    lookup: Mapping[tuple[pd.Timestamp, str, str], object],
    metadata: Mapping[tuple[str, str], _ProductMetadata],
    contrast_rows: list[dict[str, object]],
    diagnostic_rows: list[dict[str, object]],
) -> int:
    family = "BASE_PEAK_IMPLIED_OFFPEAK"
    negative_count = 0
    bases = short.loc[short["load_type"].eq("BASE")]
    peaks = short.loc[short["load_type"].eq("PEAK")]
    for base in bases.itertuples(index=False):
        quotation_date = pd.Timestamp(base.date)
        product_type = str(base.product_type)
        product = str(base.product)
        contrast_id = _contrast_id(
            quotation_date,
            family,
            "BASE_PEAK",
            product,
        )
        peak = lookup.get((quotation_date, "PEAK", product))
        meta = metadata[(product_type, product)]
        if peak is None:
            diagnostic_rows.append(
                _diagnostic_row(
                    quotation_date=quotation_date,
                    contrast_id=contrast_id,
                    family=family,
                    parent_type=product_type,
                    parent_product=product,
                    load_type="BASE_PEAK",
                    expected=2,
                    observed=1,
                    parent_hours=meta.total_hours,
                    residual=float("nan"),
                    status="REJECT_MISSING_PEAK_PAIR",
                )
            )
            continue
        implied_offpeak = (
            float(base.price) * meta.total_hours - float(peak.price) * meta.peak_hours
        ) / meta.offpeak_hours
        negative = implied_offpeak < 0.0
        negative_count += int(negative)
        peak_delta = float(peak.price) - float(base.price)
        offpeak_delta = implied_offpeak - float(base.price)
        recomposed = (
            float(peak.price) * meta.peak_hours
            + implied_offpeak * meta.offpeak_hours
        ) / meta.total_hours
        residual = recomposed - float(base.price)
        diagnostic_rows.append(
            _diagnostic_row(
                quotation_date=quotation_date,
                contrast_id=contrast_id,
                family=family,
                parent_type=product_type,
                parent_product=product,
                load_type="BASE_PEAK",
                expected=2,
                observed=2,
                parent_hours=meta.total_hours,
                residual=residual,
                negative_implied_offpeak=negative,
                status="ACCEPTED",
            )
        )
        peak_days = set(meta.peak_delivery_days)
        for delivery_day in meta.delivery_days:
            for short_block, hours in _day_blocks(delivery_day):
                is_contract_peak = (
                    short_block == "SHORT_PEAK" and delivery_day in peak_days
                )
                _append_contrast_cell(
                    contrast_rows,
                    quotation_date=quotation_date,
                    contrast_id=contrast_id,
                    family=family,
                    parent_type=product_type,
                    parent_product=product,
                    source_load_type="BASE_PEAK",
                    source_product=product,
                    source_role="BASE_PEAK_PAIR",
                    delivery_day=delivery_day,
                    short_block=short_block,
                    priced_segment="PEAK" if is_contract_peak else "OFFPEAK",
                    delivery_hours=hours,
                    delta=peak_delta if is_contract_peak else offpeak_delta,
                )
    base_keys = {
        (pd.Timestamp(row.date), str(row.product))
        for row in bases.itertuples(index=False)
    }
    for peak in peaks.itertuples(index=False):
        quotation_date = pd.Timestamp(peak.date)
        product = str(peak.product)
        if (quotation_date, product) in base_keys:
            continue
        product_type = str(peak.product_type)
        meta = metadata[(product_type, product)]
        contrast_id = _contrast_id(
            quotation_date,
            family,
            "BASE_PEAK",
            product,
        )
        diagnostic_rows.append(
            _diagnostic_row(
                quotation_date=quotation_date,
                contrast_id=contrast_id,
                family=family,
                parent_type=product_type,
                parent_product=product,
                load_type="BASE_PEAK",
                expected=2,
                observed=1,
                parent_hours=meta.total_hours,
                residual=float("nan"),
                status="REJECT_MISSING_BASE_PAIR",
            )
        )
    return negative_count


def _parse_product(product: str, product_type: str) -> _ProductMetadata:
    day_match = _DAY.fullmatch(product)
    week_match = _WEEK.fullmatch(product)
    weekend_match = _WEEKEND.fullmatch(product)
    if day_match is not None:
        expected_type = "Day"
        start = pd.Timestamp(day_match.group(1))
        end = start
        peak_days = (start,)
    elif week_match is not None:
        expected_type = "Week"
        start = _iso_week_date(week_match.group(1), week_match.group(2), weekday=1)
        end = start + pd.Timedelta(days=6)
        peak_days = tuple(start + pd.Timedelta(days=offset) for offset in range(5))
    elif weekend_match is not None:
        expected_type = "Weekend"
        start = _iso_week_date(
            weekend_match.group(1), weekend_match.group(2), weekday=6
        )
        end = start + pd.Timedelta(days=1)
        peak_days = (start, end)
    else:
        raise ValueError(f"normalized EEX short-tenor product is invalid: {product}")
    if product_type != expected_type:
        raise ValueError(f"normalized EEX product type does not match {product}")
    delivery_days = tuple(
        start + pd.Timedelta(days=offset) for offset in range((end - start).days + 1)
    )
    total_hours = sum(_local_day_hours(day) for day in delivery_days)
    peak_hours = 12 * len(peak_days)
    offpeak_hours = total_hours - peak_hours
    if offpeak_hours <= 0:
        raise ValueError(f"normalized EEX product hours are invalid: {product}")
    return _ProductMetadata(
        product_type=product_type,
        product=product,
        start=start,
        end=end,
        delivery_days=delivery_days,
        peak_delivery_days=peak_days,
        total_hours=total_hours,
        peak_hours=peak_hours,
        offpeak_hours=offpeak_hours,
    )


def _local_day_hours(day: pd.Timestamp) -> int:
    start = pd.Timestamp(day.date(), tz=LOCAL_TIMEZONE)
    end = pd.Timestamp((day + pd.Timedelta(days=1)).date(), tz=LOCAL_TIMEZONE)
    value = (end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1)
    if not float(value).is_integer():
        raise RuntimeError("local delivery day does not contain integral hours")
    return int(value)


def _day_blocks(day: pd.Timestamp) -> tuple[tuple[str, int], tuple[str, int]]:
    offpeak_hours = _local_day_hours(day) - 12
    if offpeak_hours <= 0:
        raise RuntimeError("short OFFPEAK block has non-positive hours")
    return (("SHORT_OFFPEAK", offpeak_hours), ("SHORT_PEAK", 12))


def _iso_week_date(year: str, week: str, *, weekday: int) -> pd.Timestamp:
    try:
        value = calendar_date.fromisocalendar(int(year), int(week), weekday)
    except ValueError as exc:
        raise ValueError(f"normalized EEX ISO week is invalid: {year}-W{week}") from exc
    return pd.Timestamp(value)


def _contrast_id(
    quotation_date: pd.Timestamp,
    family: str,
    load_type: str,
    product: str,
) -> str:
    return "|".join(
        [quotation_date.date().isoformat(), family, load_type, product]
    )


def _append_contrast_cell(
    rows: list[dict[str, object]],
    *,
    quotation_date: pd.Timestamp,
    contrast_id: str,
    family: str,
    parent_type: str,
    parent_product: str,
    source_load_type: str,
    source_product: str,
    source_role: str,
    delivery_day: pd.Timestamp,
    short_block: str,
    priced_segment: str,
    delivery_hours: int,
    delta: float,
) -> None:
    rows.append(
        {
            "quotation_date": quotation_date,
            "contrast_id": contrast_id,
            "family": family,
            "parent_product_type": parent_type,
            "parent_product": parent_product,
            "source_load_type": source_load_type,
            "source_product": source_product,
            "source_role": source_role,
            "delivery_day": delivery_day,
            "short_block": short_block,
            "priced_segment": priced_segment,
            "delivery_hours": int(delivery_hours),
            "delta_eur_mwh": float(delta),
        }
    )


def _diagnostic_row(
    *,
    quotation_date: pd.Timestamp,
    contrast_id: str,
    family: str,
    parent_type: str,
    parent_product: str,
    load_type: str,
    expected: int,
    observed: int,
    parent_hours: int,
    residual: float,
    status: str,
    negative_implied_offpeak: bool = False,
) -> dict[str, object]:
    return {
        "quotation_date": quotation_date,
        "contrast_id": contrast_id,
        "family": family,
        "parent_product_type": parent_type,
        "parent_product": parent_product,
        "load_type": load_type,
        "expected_component_count": int(expected),
        "observed_component_count": int(observed),
        "parent_delivery_hours": int(parent_hours),
        "recomposition_residual_eur_mwh": float(residual),
        "abs_recomposition_residual_eur_mwh": abs(float(residual)),
        "negative_implied_offpeak": bool(negative_implied_offpeak),
        "status": status,
    }


def _validate_contrast_output(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != CONTRAST_COLUMNS:
        raise RuntimeError("constructed contrast columns are not exact")
    if frame.duplicated(["contrast_id", "delivery_day", "short_block"]).any():
        raise RuntimeError("constructed contrast repeats a native-grain cell")
    if not frame["delivery_hours"].gt(0).all():
        raise RuntimeError("constructed contrast has non-positive delivery hours")
    if not all(math.isfinite(value) for value in frame["delta_eur_mwh"].to_numpy(float)):
        raise RuntimeError("constructed contrast has non-finite additive values")
    if not set(frame["short_block"]).issubset({"SHORT_PEAK", "SHORT_OFFPEAK"}):
        raise RuntimeError("constructed contrast has an invalid short block")
    if not set(frame["priced_segment"]).issubset({"BASE", "PEAK", "OFFPEAK"}):
        raise RuntimeError("constructed contrast has an invalid priced segment")


def _contrast_weighted_means(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "contrast_id",
        "family",
        "delivery_hours",
        "weighted_mean_eur_mwh",
        "abs_weighted_mean_eur_mwh",
    )
    if frame.empty:
        return pd.DataFrame(columns=columns)
    working = frame[["contrast_id", "family", "delivery_hours", "delta_eur_mwh"]].copy()
    working["weighted_value"] = (
        working["delivery_hours"].astype(float) * working["delta_eur_mwh"].astype(float)
    )
    grouped = working.groupby(["contrast_id", "family"], sort=True).agg(
        delivery_hours=("delivery_hours", "sum"),
        weighted_value=("weighted_value", "sum"),
    )
    grouped["weighted_mean_eur_mwh"] = (
        grouped["weighted_value"] / grouped["delivery_hours"]
    )
    grouped["abs_weighted_mean_eur_mwh"] = grouped[
        "weighted_mean_eur_mwh"
    ].abs()
    return grouped.drop(columns="weighted_value").reset_index()[list(columns)]


def _counts_by_value(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in frame.groupby(column, sort=True).size().items()
    }


def _require_exact(frame: pd.DataFrame, column: str, expected: set[str]) -> None:
    actual = set(frame[column].astype(str))
    if actual != expected:
        raise ValueError(f"normalized EEX {column} values are invalid: {sorted(actual)}")


def _require_subset(frame: pd.DataFrame, column: str, allowed: set[str]) -> None:
    actual = set(frame[column].astype(str))
    if not actual.issubset(allowed):
        raise ValueError(f"normalized EEX {column} values are invalid: {sorted(actual)}")


def _validate_content_id(value: str, field: str) -> None:
    if _SHA256.fullmatch(str(value)) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 content id")


__all__ = [
    "CONTRAST_COLUMNS",
    "DIAGNOSTIC_COLUMNS",
    "LOCAL_TIMEZONE",
    "MAX_ZERO_PARENT_MEAN_RESIDUAL_EUR_MWH",
    "NORMALIZED_HISTORY_COLUMNS",
    "PARENT_CHILD_TOLERANCE_EUR_MWH",
    "SHORT_TENOR_CONTRAST_CONTRACT_VERSION",
    "SHORT_TENOR_CONTRAST_SCHEMA_VERSION",
    "ShortTenorContrastPanel",
    "build_short_tenor_contrast_panel",
    "contrast_weighted_means",
    "materialize_short_tenor_contrast_signal",
]
