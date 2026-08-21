from __future__ import annotations

import inspect
from datetime import date as calendar_date

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.short_tenor_contrasts import (
    CONTRAST_COLUMNS,
    NORMALIZED_HISTORY_COLUMNS,
    build_short_tenor_contrast_panel,
    contrast_weighted_means,
    materialize_short_tenor_contrast_signal,
)
from pfc_shaping.lt.model.short_tenor_shape_contract import (
    project_short_tenor_shape_delta,
)
from pfc_shaping.pipeline import production_phases

NORMALIZATION_ID = "2" * 64
AUDIT_ID = "b" * 64
SNAPSHOT_ID = "5" * 64
TZ = "Europe/Zurich"


def _week_start(iso_year: int, iso_week: int) -> pd.Timestamp:
    return pd.Timestamp(calendar_date.fromisocalendar(iso_year, iso_week, 1))


def _day_hours(day: pd.Timestamp) -> int:
    start = pd.Timestamp(day.date(), tz=TZ)
    end = pd.Timestamp((day + pd.Timedelta(days=1)).date(), tz=TZ)
    return int((end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1))


def _row(
    quotation_date: str | pd.Timestamp,
    product: str,
    product_type: str,
    load_type: str,
    price: float,
    *,
    snapshot: str = SNAPSHOT_ID,
) -> dict[str, object]:
    quote = pd.Timestamp(quotation_date).normalize()
    return {
        "date": quote,
        "product": product,
        "load_type": load_type,
        "product_type": product_type,
        "market": "CH",
        "price": float(price),
        "unit": "EUR/MWH",
        "source": "EEX",
        "source_system": "DATABRICKS_PRD_GOLD",
        "product_id": f"PID:{load_type}:{product}",
        "delivery_period_id": f"DP:{product}",
        "fact_load_timestamp_utc": quote.tz_localize("UTC") + pd.Timedelta(hours=12),
        "source_snapshot_sha256": snapshot,
        "point_in_time_available_at_proven": False,
        "rolling_origin_selection_authorized": False,
        "production_promotion_authorized": False,
    }


def _complete_week(
    iso_year: int,
    iso_week: int,
    *,
    quotation_date: str,
    base_prices: tuple[float, ...] = (70.0, 72.0, 74.0, 76.0, 78.0, 60.0, 62.0),
    peak_prices: tuple[float, ...] = (84.0, 86.0, 88.0, 90.0, 92.0, 74.0, 76.0),
) -> pd.DataFrame:
    start = _week_start(iso_year, iso_week)
    days = tuple(start + pd.Timedelta(days=offset) for offset in range(7))
    week_product = f"W:{iso_year}-W{iso_week:02d}"
    weekend_product = f"WE:{iso_year}-W{iso_week:02d}"
    rows: list[dict[str, object]] = []
    for day, base, peak in zip(days, base_prices, peak_prices, strict=True):
        product = f"D:{day.date().isoformat()}"
        rows.append(_row(quotation_date, product, "Day", "BASE", base))
        rows.append(_row(quotation_date, product, "Day", "PEAK", peak))
    day_weights = np.array([_day_hours(day) for day in days], dtype=float)
    week_base = float(np.average(np.asarray(base_prices), weights=day_weights))
    week_peak = float(np.mean(peak_prices[:5]))
    weekend_base = float(
        np.average(np.asarray(base_prices[5:]), weights=day_weights[5:])
    )
    weekend_peak = float(np.mean(peak_prices[5:]))
    rows.extend(
        [
            _row(quotation_date, week_product, "Week", "BASE", week_base),
            _row(quotation_date, week_product, "Week", "PEAK", week_peak),
            _row(quotation_date, weekend_product, "Weekend", "BASE", weekend_base),
            _row(quotation_date, weekend_product, "Weekend", "PEAK", weekend_peak),
        ]
    )
    return pd.DataFrame(rows, columns=NORMALIZED_HISTORY_COLUMNS)


def _build(frame: pd.DataFrame):
    return build_short_tenor_contrast_panel(
        frame,
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
    )


def _complete_local_month(month: str, *, freq: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{month}-01", tz=TZ)
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq=freq,
        inclusive="left",
    )


def test_complete_spring_dst_week_builds_separate_zero_mean_families() -> None:
    frame = _complete_week(2027, 12, quotation_date="2027-03-19")
    panel = _build(frame)
    means = contrast_weighted_means(panel.contrasts)

    assert panel.audit["contrast_count_by_family"] == {
        "BASE_PEAK_IMPLIED_OFFPEAK": 9,
        "DAY_WITHIN_WEEK": 2,
        "DAY_WITHIN_WEEKEND": 2,
        "WEEKEND_VS_WEEKDAY_BASE": 1,
    }
    assert panel.audit["contrast_count"] == 14
    assert panel.audit["diagnostic_count_by_status"] == {"ACCEPTED": 14}
    assert means["abs_weighted_mean_eur_mwh"].max() <= 1e-10
    assert panel.audit["components_combined"] is False
    assert panel.audit["amplitude_fitted"] is False
    assert panel.audit["model_input_authorized"] is False
    assert panel.audit["ompex_used"] is False
    assert panel.audit["afry_numeric_values_used"] is False

    dst_day = panel.contrasts.loc[
        panel.contrasts["delivery_day"].eq(pd.Timestamp("2027-03-28"))
        & panel.contrasts["short_block"].eq("SHORT_OFFPEAK")
    ]
    assert set(dst_day["delivery_hours"]) == {11}
    week_pair = panel.contrasts.loc[
        panel.contrasts["family"].eq("BASE_PEAK_IMPLIED_OFFPEAK")
        & panel.contrasts["parent_product"].eq("W:2027-W12")
        & panel.contrasts["delivery_day"].eq(pd.Timestamp("2027-03-27"))
        & panel.contrasts["short_block"].eq("SHORT_PEAK")
    ]
    assert week_pair["priced_segment"].tolist() == ["OFFPEAK"]


def test_autumn_dst_contract_hours_are_169_for_week_and_49_for_weekend() -> None:
    panel = _build(_complete_week(2027, 43, quotation_date="2027-10-22"))
    diagnostics = panel.diagnostics

    week_hours = diagnostics.loc[
        diagnostics["parent_product_type"].eq("Week"),
        ["load_type", "parent_delivery_hours"],
    ]
    assert set(
        week_hours.loc[week_hours["load_type"].eq("PEAK"), "parent_delivery_hours"]
    ) == {60}
    assert set(
        week_hours.loc[~week_hours["load_type"].eq("PEAK"), "parent_delivery_hours"]
    ) == {169}
    weekend_hours = diagnostics.loc[
        diagnostics["parent_product_type"].eq("Weekend"),
        ["load_type", "parent_delivery_hours"],
    ]
    assert set(
        weekend_hours.loc[weekend_hours["load_type"].eq("PEAK"), "parent_delivery_hours"]
    ) == {24}
    assert set(
        weekend_hours.loc[~weekend_hours["load_type"].eq("PEAK"), "parent_delivery_hours"]
    ) == {49}
    sunday = panel.contrasts.loc[
        panel.contrasts["delivery_day"].eq(pd.Timestamp("2027-10-31"))
        & panel.contrasts["short_block"].eq("SHORT_OFFPEAK")
    ]
    assert set(sunday["delivery_hours"]) == {13}


def test_missing_same_vintage_child_is_not_filled_from_another_date() -> None:
    frame = _complete_week(2027, 20, quotation_date="2027-05-14")
    missing_product = "D:2027-05-23"
    missing = frame.loc[
        frame["product"].eq(missing_product) & frame["load_type"].eq("BASE")
    ].iloc[0]
    frame = frame.loc[
        ~(frame["product"].eq(missing_product) & frame["load_type"].eq("BASE"))
    ].copy()
    prior = missing.copy()
    prior["date"] = pd.Timestamp("2027-05-13")
    prior["fact_load_timestamp_utc"] = pd.Timestamp("2027-05-13T12:00:00Z")
    frame = pd.concat([frame, prior.to_frame().T], ignore_index=True)[
        list(NORMALIZED_HISTORY_COLUMNS)
    ]
    panel = _build(frame)

    rejected = panel.diagnostics.loc[
        panel.diagnostics["family"].isin({"DAY_WITHIN_WEEK", "DAY_WITHIN_WEEKEND"})
        & panel.diagnostics["load_type"].eq("BASE")
        & panel.diagnostics["quotation_date"].eq(pd.Timestamp("2027-05-14"))
    ]
    assert set(rejected["status"]) == {"REJECT_INCOMPLETE_CHILD_STRIP"}
    assert not panel.contrasts["contrast_id"].isin(rejected["contrast_id"]).any()
    assert panel.audit["per_product_fill_forward_used"] is False


def test_parent_child_conflict_is_diagnostic_and_not_a_contrast() -> None:
    frame = _complete_week(2027, 30, quotation_date="2027-07-23")
    mask = frame["product"].eq("W:2027-W30") & frame["load_type"].eq("BASE")
    frame.loc[mask, "price"] += 0.02
    panel = _build(frame)
    rejected = panel.diagnostics.loc[
        panel.diagnostics["family"].eq("DAY_WITHIN_WEEK")
        & panel.diagnostics["load_type"].eq("BASE")
    ]

    assert rejected["status"].tolist() == ["REJECT_PARENT_CHILD_QUOTE_CONFLICT"]
    assert not panel.contrasts["contrast_id"].isin(rejected["contrast_id"]).any()


def test_negative_implied_offpeak_is_retained_in_additive_space() -> None:
    product = "D:2027-08-03"
    frame = pd.DataFrame(
        [
            _row("2027-08-01", product, "Day", "BASE", -5.0),
            _row("2027-08-01", product, "Day", "PEAK", 20.0),
        ],
        columns=NORMALIZED_HISTORY_COLUMNS,
    )
    panel = _build(frame)
    means = contrast_weighted_means(panel.contrasts)

    assert panel.audit["negative_implied_offpeak_contrast_count"] == 1
    assert panel.audit["contrast_count_by_family"] == {
        "BASE_PEAK_IMPLIED_OFFPEAK": 1
    }
    assert means["abs_weighted_mean_eur_mwh"].max() <= 1e-10
    assert set(panel.contrasts["priced_segment"]) == {"PEAK", "OFFPEAK"}


def test_peak_only_product_is_diagnostic_not_synthetic_base() -> None:
    frame = pd.DataFrame(
        [_row("2027-08-01", "D:2027-08-03", "Day", "PEAK", 20.0)],
        columns=NORMALIZED_HISTORY_COLUMNS,
    )
    panel = _build(frame)

    assert panel.contrasts.empty
    assert panel.diagnostics["status"].tolist() == ["REJECT_MISSING_BASE_PAIR"]
    assert panel.audit["synthetic_quote_created"] is False


def test_weekend_weekday_partition_is_exact_and_kept_separate() -> None:
    panel = _build(_complete_week(2027, 35, quotation_date="2027-08-27"))
    family = panel.contrasts.loc[
        panel.contrasts["family"].eq("WEEKEND_VS_WEEKDAY_BASE")
    ]
    means = contrast_weighted_means(family.reset_index(drop=True))

    assert family["contrast_id"].nunique() == 1
    assert set(family["source_role"]) == {"IMPLIED_WEEKDAY", "WEEKEND_QUOTE"}
    assert means["abs_weighted_mean_eur_mwh"].max() <= 1e-10
    assert panel.audit["components_combined"] is False


def test_construction_is_deterministic_under_input_row_order() -> None:
    frame = _complete_week(2027, 40, quotation_date="2027-10-01")
    first = _build(frame)
    second = _build(frame.sample(frac=1.0, random_state=42).reset_index(drop=True))

    pd.testing.assert_frame_equal(first.contrasts, second.contrasts)
    pd.testing.assert_frame_equal(first.diagnostics, second.diagnostics)
    assert first.audit == second.audit


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="unit"), "columns are not exact"),
        (
            lambda frame: frame.assign(point_in_time_available_at_proven=True),
            "must remain false",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "repeats a quote identity",
        ),
        (
            lambda frame: frame.assign(
                source_snapshot_sha256=[SNAPSHOT_ID] * (len(frame) - 1)
                + ["6" * 64]
            ),
            "mixes source snapshot hashes",
        ),
        (
            lambda frame: frame.assign(
                price=np.where(
                    np.arange(len(frame)) == 0,
                    np.nan,
                    frame["price"],
                )
            ),
            "settlement prices are invalid",
        ),
    ],
)
def test_rejects_ungoverned_normalized_inputs(mutator, message: str) -> None:
    frame = _complete_week(2027, 45, quotation_date="2027-11-05")
    with pytest.raises(ValueError, match=message):
        _build(mutator(frame.copy()))


def test_rejects_non_live_quote_and_bad_content_id() -> None:
    frame = pd.DataFrame(
        [_row("2027-08-03", "D:2027-08-03", "Day", "BASE", 10.0)],
        columns=NORMALIZED_HISTORY_COLUMNS,
    )
    with pytest.raises(ValueError, match="non-live quote"):
        _build(frame)
    valid = pd.DataFrame(
        [_row("2027-08-02", "D:2027-08-03", "Day", "BASE", 10.0)],
        columns=NORMALIZED_HISTORY_COLUMNS,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        build_short_tenor_contrast_panel(
            valid,
            source_normalization_content_id="bad",
            source_audit_content_id=AUDIT_ID,
        )


def test_weighted_mean_public_helper_requires_exact_schema() -> None:
    with pytest.raises(ValueError, match="columns are not exact"):
        contrast_weighted_means(pd.DataFrame({"contrast_id": []}))
    empty = pd.DataFrame(columns=CONTRAST_COLUMNS)
    assert contrast_weighted_means(empty).empty


def test_each_hourly_spring_component_is_accepted_by_solver_neutral_contract() -> None:
    panel = _build(_complete_week(2027, 12, quotation_date="2027-03-19"))
    index = _complete_local_month("2027-03", freq="1h")

    for contrast_id in panel.contrasts["contrast_id"].drop_duplicates():
        raw = materialize_short_tenor_contrast_signal(
            panel.contrasts,
            contrast_id=str(contrast_id),
            index=index,
        )
        projected = project_short_tenor_shape_delta(
            raw,
            monthly_base_levels={"2027-03": 80.0},
            peak_forward_prices={"2027-03": 95.0},
            source_normalization_content_id=NORMALIZATION_ID,
            source_audit_content_id=AUDIT_ID,
            valuation_timestamp_utc="2027-02-28T22:00:00Z",
        )

        assert projected.audit["max_constraint_residual_eur_mwh"] <= 1e-9
        assert projected.audit["max_monthly_mean_residual_eur_mwh"] <= 1e-9
        assert projected.audit["model_input_authorized"] is False


def test_quarter_hour_autumn_component_materialization_preserves_native_blocks() -> None:
    panel = _build(_complete_week(2027, 43, quotation_date="2027-10-22"))
    index = _complete_local_month("2027-10", freq="15min")
    contrast_id = panel.contrasts.loc[
        panel.contrasts["family"].eq("WEEKEND_VS_WEEKDAY_BASE"),
        "contrast_id",
    ].iloc[0]
    raw = materialize_short_tenor_contrast_signal(
        panel.contrasts,
        contrast_id=str(contrast_id),
        index=index,
    )
    local = raw.index.tz_convert(TZ)
    frame = pd.DataFrame(
        {
            "value": raw.to_numpy(),
            "day": local.strftime("%Y-%m-%d"),
            "block": np.where(
                (local.hour >= 8) & (local.hour < 20),
                "SHORT_PEAK",
                "SHORT_OFFPEAK",
            ),
        }
    )
    dispersion = frame.groupby(["day", "block"])["value"].agg(
        lambda values: values.max() - values.min()
    )

    assert dispersion.max() == pytest.approx(0.0, abs=1e-12)
    assert raw.loc[raw.index.tz_convert(TZ).date == calendar_date(2027, 10, 31)].nunique() == 1


def test_materialization_rejects_partial_month_missing_support_and_nonzero_component() -> None:
    panel = _build(_complete_week(2027, 12, quotation_date="2027-03-19"))
    contrast_id = str(panel.contrasts["contrast_id"].iloc[0])
    march = _complete_local_month("2027-03", freq="1h")
    with pytest.raises(ValueError, match="complete local calendar months"):
        materialize_short_tenor_contrast_signal(
            panel.contrasts,
            contrast_id=contrast_id,
            index=march[1:],
        )

    with pytest.raises(ValueError, match="does not contain contrast support"):
        materialize_short_tenor_contrast_signal(
            panel.contrasts,
            contrast_id=contrast_id,
            index=_complete_local_month("2027-04", freq="1h"),
        )

    tampered = panel.contrasts.copy()
    first_row = tampered["contrast_id"].eq(contrast_id).idxmax()
    tampered.loc[first_row, "delta_eur_mwh"] += 1.0
    with pytest.raises(ValueError, match="zero-parent-mean"):
        materialize_short_tenor_contrast_signal(
            tampered,
            contrast_id=contrast_id,
            index=march,
        )


def test_contrast_constructor_and_materializer_remain_unwired() -> None:
    source = inspect.getsource(production_phases)
    assert "build_short_tenor_contrast_panel" not in source
    assert "materialize_short_tenor_contrast_signal" not in source
