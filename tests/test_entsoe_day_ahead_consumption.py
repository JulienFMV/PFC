from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_day_ahead_consumption import (
    OUTPUT_COLUMNS,
    SOURCE_COLUMNS,
    AvailabilityBasis,
    DayAheadConsumptionError,
    EffectiveDatedSeriesRule,
    IndependentControl,
    SpotUsage,
    build_monthly_zero_mean_spot_shape,
    materialize_day_ahead_consumption,
)


def _key(field: str, sequence: str | None = None) -> str:
    base = f"day_ahead_prices||{field}"
    return f"{base}||{sequence}" if sequence is not None else base


def _row(
    start: str,
    end: str,
    *,
    field: str = "ch_price",
    zone: str = "CH",
    timezone: str = "Europe/Zurich",
    sequence: str | None = None,
    resolution: str = "PT60M",
    price: float = 50.0,
    publication: str | None = "2025-12-31T12:00:00Z",
    first_seen: str | None = "2025-12-31T12:05:00Z",
    availability_basis: str = AvailabilityBasis.FMV_FIRST_SEEN.value,
    original_publication_proven: bool = False,
    is_historical: bool = True,
    is_final: bool = True,
    revision: int = 1,
) -> dict[str, object]:
    return {
        "field_name": field,
        "market_zone": zone,
        "market_timezone": timezone,
        "series_key": _key(field, sequence),
        "classification_sequence": sequence,
        "interval_start_utc": start,
        "interval_end_utc": end,
        "native_resolution": resolution,
        "price_eur_per_mwh": price,
        "publication_timestamp_utc": publication,
        "first_seen_pull_ts_utc": first_seen,
        "availability_basis": availability_basis,
        "original_publication_proven": original_publication_proven,
        "is_historical": is_historical,
        "is_final": is_final,
        "dq_failed": False,
        "quality_status": "PASSED",
        "source_time_series_id": f"TS-{zone}-{sequence or 'BASE'}",
        "source_document_mrid": f"DOC-{zone}",
        "source_document_revision_number": revision,
        "source_snapshot_id": f"SNAP-{zone}-{revision}",
        "source_file_path": f"prd/entsoe/{zone}/{revision}.xml",
    }


def _frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _rule(
    start: str,
    end: str,
    *,
    field: str = "ch_price",
    zone: str = "CH",
    sequence: str | None = None,
) -> EffectiveDatedSeriesRule:
    control = (
        IndependentControl.ENTSOE_ONLY_NO_LSEG_CURVE
        if zone == "IT_NORD"
        else IndependentControl.LSEG_RECONCILIATION_PASSED
    )
    return EffectiveDatedSeriesRule(
        field_name=field,
        market_zone=zone,
        series_key=_key(field, sequence),
        classification_sequence=sequence,
        effective_start_utc=start,
        effective_end_utc=end,
        source_semantics="A44 coupled day-ahead auction selected by effective-dated source evidence",
        selection_evidence_id=f"RULE-{zone}-{sequence or 'BASE'}",
        selection_evidence_sha256="a" * 64,
        independent_control=control,
    )


def _materialize(
    source: pd.DataFrame,
    start: str,
    end: str,
    *,
    usage: SpotUsage = SpotUsage.REALIZED_FINAL,
    rules: list[EffectiveDatedSeriesRule] | None = None,
    as_of: str | None = None,
    required_fields: tuple[str, ...] = ("ch_price",),
):
    return materialize_day_ahead_consumption(
        source,
        usage=usage,
        series_rules=rules or [_rule(start, end)],
        window_start_utc=start,
        window_end_utc=end,
        as_of_utc=as_of,
        required_fields=required_fields,
    )


def test_single_cadence_silver_interval_is_preserved_at_native_grain() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"
    result = _materialize(_frame(_row(start, end)), start, end)

    assert tuple(result.frame.columns) == OUTPUT_COLUMNS
    assert len(result.frame) == 1
    assert "curve_type" not in result.frame.columns
    assert result.frame.loc[0, "source_interval_end_utc"] == pd.Timestamp(end)


def test_normalized_blocks_expand_deterministically_and_keep_source_right_edge() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z"
    source = _frame(
        _row(start, "2026-01-01T04:00:00Z", price=40.0),
        _row("2026-01-01T04:00:00Z", end, price=60.0, revision=2),
    )

    result = _materialize(source, start, end)

    assert len(result.frame) == 8
    assert result.frame["price_eur_per_mwh"].tolist() == [40.0] * 4 + [60.0] * 4
    assert result.frame.iloc[-1]["interval_end_utc"] == pd.Timestamp(end)
    assert result.frame.iloc[-1]["source_interval_end_utc"] == pd.Timestamp(end)


def test_non_multiple_duration_is_rejected() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T01:30:00Z"
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(_frame(_row(start, end)), start, end)
    assert caught.value.code == "INTERVAL_NOT_RESOLUTION_MULTIPLE"


def test_native_resolution_alignment_is_required() -> None:
    start, end = "2026-01-01T00:15:00Z", "2026-01-01T01:15:00Z"
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(_frame(_row(start, end)), start, end)
    assert caught.value.code == "INTERVAL_OFF_NATIVE_GRID"


def test_overlap_is_rejected_after_expansion() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z"
    source = _frame(
        _row(start, "2026-01-01T02:00:00Z"),
        _row("2026-01-01T01:00:00Z", end, revision=2),
    )
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(source, start, end)
    assert caught.value.code == "INTERVAL_OVERLAP"


def test_true_gap_is_not_hidden_by_a03_expansion() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"
    source = _frame(
        _row(start, "2026-01-01T02:00:00Z"),
        _row("2026-01-01T03:00:00Z", end, revision=2),
    )
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(source, start, end)
    assert caught.value.code == "INTERVAL_GAP"


@pytest.mark.parametrize(
    "end",
    ["2026-01-01T00:00:00Z", "2025-12-31T23:00:00Z"],
)
def test_zero_or_negative_interval_is_rejected(end: str) -> None:
    start = "2026-01-01T00:00:00Z"
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(_frame(_row(start, end)), start, "2026-01-01T01:00:00Z")
    assert caught.value.code == "INTERVAL_NONPOSITIVE"


@pytest.mark.parametrize("column", ["interval_start_utc", "interval_end_utc"])
def test_missing_interval_bound_is_rejected(column: str) -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"
    source = _frame(_row(start, end))
    source.loc[0, column] = None
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(source, start, end)
    assert caught.value.code == "INTERVAL_BOUND_MISSING"


@pytest.mark.parametrize(
    ("start", "end", "expected_hours", "expected_offsets"),
    [
        (
            "2026-03-28T23:00:00Z",
            "2026-03-29T22:00:00Z",
            23,
            {"+01:00", "+02:00"},
        ),
        (
            "2026-10-24T22:00:00Z",
            "2026-10-25T23:00:00Z",
            25,
            {"+02:00", "+01:00"},
        ),
    ],
)
def test_dst_days_follow_utc_intervals_not_a_24_hour_assumption(
    start: str, end: str, expected_hours: int, expected_offsets: set[str]
) -> None:
    result = _materialize(_frame(_row(start, end)), start, end)

    assert len(result.frame) == expected_hours
    observed_offsets = {value[-6:] for value in result.frame["interval_start_market_time"].tolist()}
    assert observed_offsets == expected_offsets


def test_late_created_datetime_backfill_is_accepted_only_as_realized_final() -> None:
    start, end = "2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"
    source = _frame(
        _row(
            start,
            end,
            publication="2026-08-07T10:00:00Z",
            first_seen="2026-08-07T10:00:30Z",
            availability_basis=AvailabilityBasis.SOURCE_DOCUMENT_CREATED.value,
            original_publication_proven=False,
        )
    )

    result = _materialize(source, start, end)

    assert result.audit["usage"] == "realized_final"
    assert result.audit["authorities"]["point_in_time_filter_validated"] is False


def test_realized_final_refuses_a_nonfinal_observation() -> None:
    start, end = "2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(_frame(_row(start, end, is_final=False)), start, end)
    assert caught.value.code == "REALIZED_FINAL_UNPROVEN"


def test_causal_asof_refuses_unproven_created_datetime_without_fallback() -> None:
    start, end = "2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"
    source = _frame(
        _row(
            start,
            end,
            publication="2026-08-07T10:00:00Z",
            first_seen="2026-08-07T10:00:30Z",
            availability_basis=AvailabilityBasis.SOURCE_DOCUMENT_CREATED.value,
            original_publication_proven=False,
        )
    )
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(
            source,
            start,
            end,
            usage=SpotUsage.CAUSAL_ASOF,
            as_of="2026-09-01T00:00:00Z",
        )
    assert caught.value.code == "ORIGINAL_PUBLICATION_UNPROVEN"


def test_causal_asof_accepts_explicit_first_seen_only_after_it_was_observed() -> None:
    start, end = "2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"
    source = _frame(
        _row(
            start,
            end,
            publication="2026-08-07T10:00:00Z",
            first_seen="2026-08-07T10:00:30Z",
            availability_basis=AvailabilityBasis.FMV_FIRST_SEEN.value,
        )
    )

    result = _materialize(
        source,
        start,
        end,
        usage=SpotUsage.CAUSAL_ASOF,
        as_of="2026-08-07T10:00:30Z",
    )

    assert result.frame.loc[0, "availability_timestamp_utc"] == pd.Timestamp("2026-08-07T10:00:30Z")
    assert result.frame.loc[0, "availability_basis"] == "FMV_FIRST_SEEN"


def test_causal_asof_accepts_proven_original_document_time_before_delivery() -> None:
    start, end = "2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"
    source = _frame(
        _row(
            start,
            end,
            publication="2026-06-30T12:00:00Z",
            availability_basis=AvailabilityBasis.SOURCE_DOCUMENT_CREATED.value,
            original_publication_proven=True,
        )
    )

    result = _materialize(
        source,
        start,
        end,
        usage=SpotUsage.CAUSAL_ASOF,
        as_of="2026-06-30T12:00:00Z",
    )

    assert result.frame.loc[0, "availability_timestamp_utc"] == pd.Timestamp("2026-06-30T12:00:00Z")


def test_unknown_backfill_can_never_become_causal_asof() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"
    source = _frame(
        _row(
            start,
            end,
            publication=None,
            first_seen=None,
            availability_basis=AvailabilityBasis.UNKNOWN_BACKFILL.value,
        )
    )
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(
            source,
            start,
            end,
            usage=SpotUsage.CAUSAL_ASOF,
            as_of="2026-02-01T00:00:00Z",
        )
    assert caught.value.code == "CAUSAL_AVAILABILITY_UNPROVEN"


def test_at_inventory_keeps_both_sequences_but_rule_selects_only_proven_sequence() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"
    source = _frame(
        _row(start, end, field="at_price", zone="AT", timezone="Europe/Vienna", sequence="1"),
        _row(
            start,
            end,
            field="at_price",
            zone="AT",
            timezone="Europe/Vienna",
            sequence="2",
            price=55.0,
            revision=2,
        ),
    )
    result = _materialize(
        source,
        start,
        end,
        rules=[_rule(start, end, field="at_price", zone="AT", sequence="2")],
        required_fields=("at_price",),
    )

    assert result.frame["series_key"].unique().tolist() == [_key("at_price", "2")]
    assert result.frame["price_eur_per_mwh"].tolist() == [55.0]


def test_overlapping_multi_auction_rules_fail_as_ambiguous_selection() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"
    rules = [
        _rule(start, end, field="de_lu_price", zone="DE_LU", sequence="1"),
        _rule(start, end, field="de_lu_price", zone="DE_LU", sequence="2"),
    ]
    source = _frame(
        _row(
            start,
            end,
            field="de_lu_price",
            zone="DE_LU",
            timezone="Europe/Berlin",
            sequence="1",
        ),
        _row(
            start,
            end,
            field="de_lu_price",
            zone="DE_LU",
            timezone="Europe/Berlin",
            sequence="2",
            revision=2,
        ),
    )
    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(
            source,
            start,
            end,
            rules=rules,
            required_fields=("de_lu_price",),
        )
    assert caught.value.code == "SERIES_RULE_OVERLAP"


def test_source_block_cannot_cross_an_effective_dated_series_transition() -> None:
    start, transition, end = (
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
        "2026-01-01T02:00:00Z",
    )
    source = _frame(
        _row(
            start,
            end,
            field="at_price",
            zone="AT",
            timezone="Europe/Vienna",
            sequence="1",
        )
    )
    rules = [
        _rule(start, transition, field="at_price", zone="AT", sequence="1"),
        _rule(transition, end, field="at_price", zone="AT", sequence="2"),
    ]

    with pytest.raises(DayAheadConsumptionError) as caught:
        _materialize(
            source,
            start,
            end,
            rules=rules,
            required_fields=("at_price",),
        )
    assert caught.value.code == "SERIES_INTERVAL_CROSSES_RULE_BOUNDARY"


def test_spot_shape_cannot_change_solver_monthly_mean() -> None:
    start, end = "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"
    source = _frame(
        _row(start, "2026-01-01T02:00:00Z", price=30.0),
        _row("2026-01-01T02:00:00Z", end, price=70.0, revision=2),
    )
    consumption = _materialize(source, start, end)

    shape = build_monthly_zero_mean_spot_shape(consumption, monthly_level_authority="solver")
    solver_level = 82.5
    delivered = solver_level + shape["shape_eur_per_mwh"]
    weights = shape["duration_seconds"]

    assert (shape["shape_eur_per_mwh"] * weights).sum() / weights.sum() == pytest.approx(0.0)
    assert (delivered * weights).sum() / weights.sum() == pytest.approx(solver_level)


def test_consumption_module_has_no_ct_import() -> None:
    path = Path("pfc_shaping/validation/entsoe_day_ahead_consumption.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert not any(name.startswith("pfc_shaping.ct") for name in imports)
