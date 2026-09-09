from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.data.governed_lt_acquisition import dataframe_semantic_sha256
from pfc_shaping.validation.entsoe_day_ahead_consumption import (
    AvailabilityBasis,
    EffectiveDatedSeriesRule,
    IndependentControl,
    SpotUsage,
    materialize_day_ahead_consumption,
)
from pfc_shaping.validation.entsoe_day_ahead_export import (
    CAUSAL_SQL_PATH,
    CAUSAL_SQL_SHA256,
    RAW_COLUMNS,
    REALIZED_LATEST_CANDIDATE_USAGE,
    REALIZED_SQL_PATH,
    REALIZED_SQL_SHA256,
    DayAheadMarketUse,
    EntsoeDayAheadExportError,
    EvidenceAuthority,
    EvidenceKind,
    ScopedDayAheadEvidence,
    assess_export_cost_preflight,
    build_causal_export_parameters,
    build_export_replay_package,
    build_realized_export_parameters,
    validate_causal_export,
    validate_realized_export,
    validate_realized_latest_candidate,
    verify_export_replay_package,
    verify_export_sql_bindings,
)
from pfc_shaping.validation.entsoe_day_ahead_prd import PIT_SQL_SHA256
from pfc_shaping.validation.entsoe_day_ahead_prd import (
    verify_sql_bindings as verify_historical_sql_bindings,
)

START = pd.Timestamp("2026-07-01T00:00:00Z")
END = pd.Timestamp("2026-07-01T01:00:00Z")
AS_OF = pd.Timestamp("2026-06-30T18:00:00Z")
ASSESSED = pd.Timestamp("2026-09-02T08:00:00Z")

SELECTION = {
    "ch_price": "day_ahead_prices||ch_price",
    "de_lu_price": "day_ahead_prices||de_lu_price||1",
}
MARKET_USES = {
    "ch_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE,
    "de_lu_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE,
}


def _row(
    field: str,
    *,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
    basis: AvailabilityBasis = AvailabilityBasis.FMV_FIRST_SEEN,
    availability: pd.Timestamp | None = AS_OF,
    publication: pd.Timestamp | None = pd.Timestamp("2026-06-30T12:00:00Z"),
    first_seen: pd.Timestamp | None = AS_OF,
    availability_known: bool = True,
    historical: bool = False,
    revision: int = 1,
) -> dict[str, object]:
    keys = {
        "ch_price": "day_ahead_prices||ch_price",
        "de_lu_price": "day_ahead_prices||de_lu_price||1",
        "it_nord_price": "day_ahead_prices||it_nord_price",
    }
    sequences = {"ch_price": None, "de_lu_price": "1", "it_nord_price": None}
    zones = {"ch_price": "CH", "de_lu_price": "DE", "it_nord_price": "IT"}
    zone = zones[field]
    return {
        "field_name": field,
        "series_key": keys[field],
        "classification_sequence": sequences[field],
        "interval_start_utc": start,
        "date_time_utc": end,
        "interval_end_utc": end,
        "resolution": "PT60M",
        "price_eur_per_mwh": 80.0 if field == "ch_price" else 70.0,
        "publication_timestamp_utc": publication,
        "first_seen_pull_ts_utc": first_seen,
        "availability_basis": basis.value,
        "availability_known": availability_known,
        "availability_timestamp_utc": availability,
        "is_historical": historical,
        "dq_failed": False,
        "source_time_series_id": f"TS-{zone}",
        "source_document_mrid": f"DOC-{zone}",
        "source_document_revision_number": revision,
        "source_snapshot_id": f"SNAP-{zone}-{revision}",
        "source_file_path": f"prd/entsoe/{zone}/{revision}.xml",
        "vintage_id": f"VINTAGE-{zone}-{revision}-{start.isoformat()}",
    }


def _frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _evidence(
    *,
    kind: EvidenceKind,
    authority: EvidenceAuthority,
    keys: tuple[str, ...],
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
    asserted: pd.Timestamp = ASSESSED,
    evidence_id: str = "EVIDENCE-1",
    covered_frame: pd.DataFrame | None = None,
) -> ScopedDayAheadEvidence:
    if covered_frame is None:
        key_to_field = {
            "day_ahead_prices||ch_price": "ch_price",
            "day_ahead_prices||de_lu_price||1": "de_lu_price",
            "day_ahead_prices||it_nord_price": "it_nord_price",
        }
        covered_frame = _frame(*(_row(key_to_field[key]) for key in keys))
    return ScopedDayAheadEvidence(
        kind=kind,
        authority=authority,
        evidence_id=evidence_id,
        evidence_document_sha256="a" * 64,
        covered_frame_semantic_sha256=dataframe_semantic_sha256(covered_frame),
        asserted_at_utc=asserted,
        window_start_utc=start,
        window_end_utc=end,
        series_keys=keys,
    )


def _finality_evidence(
    keys: tuple[str, ...] = tuple(SELECTION.values()),
) -> tuple[ScopedDayAheadEvidence, ...]:
    return (
        _evidence(
            kind=EvidenceKind.REALIZED_FINALITY,
            authority=EvidenceAuthority.INDEPENDENT_SETTLEMENT_RECONCILIATION,
            keys=keys,
        ),
    )


def test_v2_sql_bindings_are_exact_and_historical_pit_hash_is_unchanged() -> None:
    assert verify_export_sql_bindings() == {
        "causal_sql_sha256": CAUSAL_SQL_SHA256,
        "realized_sql_sha256": REALIZED_SQL_SHA256,
    }
    assert verify_historical_sql_bindings()["pit_sql_sha256"] == PIT_SQL_SHA256
    for path, expected in (
        (CAUSAL_SQL_PATH, CAUSAL_SQL_SHA256),
        (REALIZED_SQL_PATH, REALIZED_SQL_SHA256),
    ):
        payload = Path(path).read_bytes()
        sql = payload.decode("utf-8")
        assert hashlib.sha256(payload).hexdigest() == expected
        assert "SELECT *" not in sql
        assert "curve_type" not in sql.lower()
        assert "LIMIT 20001" in sql
        assert "prd.silver.ge_power_entsoe_time_series_vintages" in sql


def test_market_month_parameters_span_two_explicit_utc_partitions() -> None:
    params = build_causal_export_parameters(
        series_selection={"ch_price": SELECTION["ch_price"]},
        window_start_utc="2026-02-28T23:00:00Z",
        window_end_utc="2026-03-31T22:00:00Z",
        as_of_utc="2026-02-28T12:00:00Z",
    )
    assert params["partition_start_year"] == 2026
    assert params["partition_start_month"] == 2
    assert params["partition_end_year"] == 2026
    assert params["partition_end_month"] == 3
    assert params["ch_series_key"] == SELECTION["ch_price"]
    assert params["de_lu_series_key"] is None
    assert params["fr_series_key"] is None


def test_realized_parameters_require_delivery_to_be_complete() -> None:
    with pytest.raises(EntsoeDayAheadExportError, match="must not precede"):
        build_realized_export_parameters(
            series_selection={"ch_price": SELECTION["ch_price"]},
            window_start_utc=START,
            window_end_utc=END,
            assessed_at_utc=START,
        )


def test_causal_fmv_first_seen_export_integrates_with_consumer() -> None:
    exported = validate_causal_export(
        _frame(_row("ch_price"), _row("de_lu_price")),
        series_selection=SELECTION,
        market_uses=MARKET_USES,
        window_start_utc=START,
        window_end_utc=END,
        as_of_utc=AS_OF,
        query_sha256=CAUSAL_SQL_SHA256,
    )
    assert tuple(exported.frame.columns) != RAW_COLUMNS
    assert not exported.frame["original_publication_proven"].any()
    assert not exported.frame["is_final"].any()
    assert exported.audit["market_uses"] == {
        "ch_price": "VALUATION_HEDGE_SCOPE",
        "de_lu_price": "VALUATION_HEDGE_SCOPE",
    }
    rules = [
        EffectiveDatedSeriesRule(
            field_name=field,
            market_zone="CH" if field == "ch_price" else "DE_LU",
            series_key=key,
            classification_sequence=None if field == "ch_price" else "1",
            effective_start_utc=START,
            effective_end_utc=END,
            source_semantics="ENTSO-E A44 coupled day-ahead auction",
            selection_evidence_id=f"SERIES-{field}",
            selection_evidence_sha256="b" * 64,
            independent_control=IndependentControl.LSEG_RECONCILIATION_PASSED,
        )
        for field, key in SELECTION.items()
    ]
    consumed = materialize_day_ahead_consumption(
        exported.frame,
        usage=SpotUsage.CAUSAL_ASOF,
        series_rules=rules,
        window_start_utc=START,
        window_end_utc=END,
        as_of_utc=AS_OF,
        required_fields=tuple(SELECTION),
    )
    assert len(consumed.frame) == 2
    assert consumed.audit["authorities"]["monthly_level_authorized"] is False


def test_source_document_created_requires_exact_original_publication_evidence() -> None:
    row = _row(
        "ch_price",
        basis=AvailabilityBasis.SOURCE_DOCUMENT_CREATED,
        availability=pd.Timestamp("2026-06-30T12:00:00Z"),
        publication=pd.Timestamp("2026-06-30T12:00:00Z"),
        first_seen=pd.Timestamp("2026-09-01T00:00:00Z"),
        historical=True,
    )
    selection = {"ch_price": SELECTION["ch_price"]}
    uses = {"ch_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE}
    with pytest.raises(EntsoeDayAheadExportError, match="exactly cover"):
        validate_causal_export(
            _frame(row),
            series_selection=selection,
            market_uses=uses,
            window_start_utc=START,
            window_end_utc=END,
            as_of_utc=AS_OF,
            query_sha256=CAUSAL_SQL_SHA256,
        )
    evidence = _evidence(
        kind=EvidenceKind.ORIGINAL_PUBLICATION,
        authority=EvidenceAuthority.PLATFORM_SIGNED_ORIGINAL_PUBLICATION_RECEIPT,
        keys=(SELECTION["ch_price"],),
        covered_frame=_frame(row),
    )
    exported = validate_causal_export(
        _frame(row),
        series_selection=selection,
        market_uses=uses,
        window_start_utc=START,
        window_end_utc=END,
        as_of_utc=AS_OF,
        query_sha256=CAUSAL_SQL_SHA256,
        publication_evidence=(evidence,),
    )
    assert exported.frame["original_publication_proven"].all()

    after_delivery = dict(row)
    after_delivery["publication_timestamp_utc"] = END
    after_delivery["availability_timestamp_utc"] = END
    after_evidence = _evidence(
        kind=EvidenceKind.ORIGINAL_PUBLICATION,
        authority=EvidenceAuthority.PLATFORM_SIGNED_ORIGINAL_PUBLICATION_RECEIPT,
        keys=(SELECTION["ch_price"],),
        covered_frame=_frame(after_delivery),
    )
    with pytest.raises(EntsoeDayAheadExportError, match="must precede delivery"):
        validate_causal_export(
            _frame(after_delivery),
            series_selection=selection,
            market_uses=uses,
            window_start_utc=START,
            window_end_utc=END,
            as_of_utc=END,
            query_sha256=CAUSAL_SQL_SHA256,
            publication_evidence=(after_evidence,),
        )


def test_causal_unknown_backfill_and_post_asof_value_fail_closed() -> None:
    unknown = _row(
        "ch_price",
        basis=AvailabilityBasis.UNKNOWN_BACKFILL,
        availability=None,
        publication=None,
        first_seen=None,
        availability_known=False,
        historical=True,
    )
    kwargs = {
        "series_selection": {"ch_price": SELECTION["ch_price"]},
        "market_uses": {"ch_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE},
        "window_start_utc": START,
        "window_end_utc": END,
        "as_of_utc": AS_OF,
        "query_sha256": CAUSAL_SQL_SHA256,
    }
    with pytest.raises(EntsoeDayAheadExportError, match="unknown availability"):
        validate_causal_export(_frame(unknown), **kwargs)
    late = _row(
        "ch_price",
        availability=pd.Timestamp("2026-06-30T19:00:00Z"),
        first_seen=pd.Timestamp("2026-06-30T19:00:00Z"),
    )
    with pytest.raises(EntsoeDayAheadExportError, match="after the as-of"):
        validate_causal_export(_frame(late), **kwargs)


def test_realized_latest_revision_is_not_final_without_scoped_evidence() -> None:
    kwargs = {
        "series_selection": SELECTION,
        "market_uses": MARKET_USES,
        "window_start_utc": START,
        "window_end_utc": END,
        "assessed_at_utc": ASSESSED,
        "query_sha256": REALIZED_SQL_SHA256,
    }
    with pytest.raises(EntsoeDayAheadExportError, match="exactly cover"):
        validate_realized_export(
            _frame(_row("ch_price"), _row("de_lu_price")),
            finality_evidence=(),
            **kwargs,
        )
    exported = validate_realized_export(
        _frame(_row("ch_price"), _row("de_lu_price")),
        finality_evidence=_finality_evidence(),
        **kwargs,
    )
    assert exported.frame["is_final"].all()
    assert not exported.frame["original_publication_proven"].any()
    assert exported.audit["authorities"]["realized_finality_evidence_validated"] is True

    altered = _frame(_row("ch_price"), _row("de_lu_price"))
    altered.loc[altered["field_name"].eq("ch_price"), "price_eur_per_mwh"] = 81.0
    with pytest.raises(EntsoeDayAheadExportError, match="exact covered export rows"):
        validate_realized_export(
            altered,
            finality_evidence=_finality_evidence(),
            **kwargs,
        )

    premature = _evidence(
        kind=EvidenceKind.REALIZED_FINALITY,
        authority=EvidenceAuthority.INDEPENDENT_SETTLEMENT_RECONCILIATION,
        keys=tuple(SELECTION.values()),
        asserted=START,
    )
    with pytest.raises(EntsoeDayAheadExportError, match="predates delivery"):
        validate_realized_export(
            _frame(_row("ch_price"), _row("de_lu_price")),
            finality_evidence=(premature,),
            **kwargs,
        )


def test_latest_revision_candidate_is_replayable_but_not_final_authority() -> None:
    raw = _frame(_row("ch_price"), _row("de_lu_price"))
    for column in (
        "interval_start_utc",
        "date_time_utc",
        "interval_end_utc",
        "publication_timestamp_utc",
        "first_seen_pull_ts_utc",
        "availability_timestamp_utc",
    ):
        raw[column] = raw[column].astype(str)
    candidate = validate_realized_latest_candidate(
        raw,
        series_selection=SELECTION,
        market_uses=MARKET_USES,
        window_start_utc=START,
        window_end_utc=END,
        assessed_at_utc=ASSESSED,
        query_sha256=REALIZED_SQL_SHA256,
    )

    assert candidate.audit["usage"] == REALIZED_LATEST_CANDIDATE_USAGE
    assert candidate.audit["authorities"]["consumer_contract_authorized"] is False
    assert candidate.audit["authorities"]["realized_finality_evidence_validated"] is False
    assert not candidate.frame["is_final"].any()

    package = build_export_replay_package(raw_frame=raw, export=candidate)
    report = verify_export_replay_package(
        artifacts=package.artifacts,
        manifest_payload=package.manifest_payload,
    )
    assert report["usage"] == REALIZED_LATEST_CANDIDATE_USAGE
    assert report["status"] == "VERIFIED_SELF_CONTAINED_DAY_AHEAD_EXPORT_REPLAY"
    assert report["authorities"]["model_input_authorized"] is False


def test_it_north_requires_platform_finality_not_lseg_reconciliation() -> None:
    selection = {"it_nord_price": "day_ahead_prices||it_nord_price"}
    uses = {"it_nord_price": DayAheadMarketUse.OBSERVATION_RISK}
    row = _row("it_nord_price")
    evidence = _evidence(
        kind=EvidenceKind.REALIZED_FINALITY,
        authority=EvidenceAuthority.INDEPENDENT_SETTLEMENT_RECONCILIATION,
        keys=(selection["it_nord_price"],),
    )
    kwargs = {
        "series_selection": selection,
        "market_uses": uses,
        "window_start_utc": START,
        "window_end_utc": END,
        "assessed_at_utc": ASSESSED,
        "query_sha256": REALIZED_SQL_SHA256,
    }
    with pytest.raises(EntsoeDayAheadExportError, match="cannot cover IT-North"):
        validate_realized_export(_frame(row), finality_evidence=(evidence,), **kwargs)
    platform = _evidence(
        kind=EvidenceKind.REALIZED_FINALITY,
        authority=EvidenceAuthority.PLATFORM_SIGNED_FINALITY_RECEIPT,
        keys=(selection["it_nord_price"],),
    )
    assert (
        validate_realized_export(_frame(row), finality_evidence=(platform,), **kwargs)
        .frame["is_final"]
        .all()
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.assign(date_time_utc=pd.Timestamp("2026-07-01T00:45:00Z")),
            "Date_Time_UTC differs",
        ),
        (
            lambda frame: frame.assign(
                interval_end_utc=pd.Timestamp("2026-07-01T00:50:00Z"),
                date_time_utc=pd.Timestamp("2026-07-01T00:50:00Z"),
            ),
            "native-resolution multiple",
        ),
        (lambda frame: frame.assign(dq_failed=True), "failed DQ"),
    ],
)
def test_normalized_interval_and_quality_failures_are_rejected(mutate, message: str) -> None:
    frame = mutate(_frame(_row("ch_price")))
    with pytest.raises(EntsoeDayAheadExportError, match=message):
        validate_realized_export(
            frame,
            series_selection={"ch_price": SELECTION["ch_price"]},
            market_uses={"ch_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE},
            window_start_utc=START,
            window_end_utc=END,
            assessed_at_utc=ASSESSED,
            query_sha256=REALIZED_SQL_SHA256,
            finality_evidence=_finality_evidence((SELECTION["ch_price"],)),
        )


def test_overlap_duplicate_and_market_use_scope_fail_closed() -> None:
    first = _row("ch_price", start=START, end=START + pd.Timedelta(minutes=30))
    first["resolution"] = "PT15M"
    second = _row(
        "ch_price",
        start=START + pd.Timedelta(minutes=15),
        end=START + pd.Timedelta(minutes=45),
        revision=2,
    )
    second["resolution"] = "PT15M"
    selection = {"ch_price": SELECTION["ch_price"]}
    evidence = _finality_evidence((SELECTION["ch_price"],))
    kwargs = {
        "series_selection": selection,
        "market_uses": {"ch_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE},
        "window_start_utc": START,
        "window_end_utc": END,
        "assessed_at_utc": ASSESSED,
        "query_sha256": REALIZED_SQL_SHA256,
        "finality_evidence": evidence,
    }
    with pytest.raises(EntsoeDayAheadExportError, match="overlap"):
        validate_realized_export(_frame(first, second), **kwargs)
    duplicate = _frame(_row("ch_price"), _row("ch_price"))
    with pytest.raises(EntsoeDayAheadExportError, match="duplicated"):
        validate_realized_export(duplicate, **kwargs)
    with pytest.raises(EntsoeDayAheadExportError, match="exactly cover"):
        validate_realized_export(
            _frame(_row("ch_price")),
            **{
                **kwargs,
                "market_uses": {
                    "ch_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE,
                    "fr_price": DayAheadMarketUse.OBSERVATION_RISK,
                },
            },
        )


def test_offline_cost_preflight_stops_and_never_authorizes_execution() -> None:
    stopped = assess_export_cost_preflight(
        usage=SpotUsage.CAUSAL_ASOF,
        warehouse_state="STOPPED",
        warehouse_started_for_request=False,
        partition_pruning_proven=False,
        scan_upper_bound_is_hard=False,
        estimated_scan_upper_bound_bytes=None,
        maximum_scan_bytes=100_000_000,
        maximum_runtime_seconds=300,
    )
    assert stopped.assessment["status"] == "STOP_NO_ACTIVE_WAREHOUSE"
    assert set(stopped.assessment["execution"].values()) == {0}
    assert stopped.assessment["execution_authorized"] is False

    uncapped = assess_export_cost_preflight(
        usage=SpotUsage.REALIZED_FINAL,
        warehouse_state="RUNNING_ALREADY_FOR_SEPARATE_AUTHORIZED_WORKLOAD",
        warehouse_started_for_request=False,
        partition_pruning_proven=True,
        scan_upper_bound_is_hard=False,
        estimated_scan_upper_bound_bytes=50_000_000,
        maximum_scan_bytes=100_000_000,
        maximum_runtime_seconds=300,
    )
    assert uncapped.assessment["status"] == "STOP_SCAN_BOUND_UNPROVEN"

    over = assess_export_cost_preflight(
        usage=SpotUsage.REALIZED_FINAL,
        warehouse_state="RUNNING_ALREADY_FOR_SEPARATE_AUTHORIZED_WORKLOAD",
        warehouse_started_for_request=False,
        partition_pruning_proven=True,
        scan_upper_bound_is_hard=True,
        estimated_scan_upper_bound_bytes=100_000_001,
        maximum_scan_bytes=100_000_000,
        maximum_runtime_seconds=300,
    )
    assert over.assessment["status"] == "STOP_SCAN_CAP_EXCEEDED"

    ready = assess_export_cost_preflight(
        usage=SpotUsage.REALIZED_FINAL,
        warehouse_state="RUNNING_ALREADY_FOR_SEPARATE_AUTHORIZED_WORKLOAD",
        warehouse_started_for_request=False,
        partition_pruning_proven=True,
        scan_upper_bound_is_hard=True,
        estimated_scan_upper_bound_bytes=50_000_000,
        maximum_scan_bytes=100_000_000,
        maximum_runtime_seconds=300,
    )
    assert ready.assessment["status"] == ("READY_FOR_HUMAN_COST_REVIEW_NO_EXECUTION_AUTHORITY")
    assert ready.assessment["execution_authorized"] is False


def test_self_contained_replay_reproduces_causal_export_and_rejects_tamper() -> None:
    raw = _frame(_row("ch_price"), _row("de_lu_price"))
    exported = validate_causal_export(
        raw,
        series_selection=SELECTION,
        market_uses=MARKET_USES,
        window_start_utc=START,
        window_end_utc=END,
        as_of_utc=AS_OF,
        query_sha256=CAUSAL_SQL_SHA256,
    )
    package = build_export_replay_package(raw_frame=raw, export=exported)
    report = verify_export_replay_package(
        artifacts=package.artifacts,
        manifest_payload=package.manifest_payload,
    )
    assert report["status"] == "VERIFIED_SELF_CONTAINED_DAY_AHEAD_EXPORT_REPLAY"
    assert report["artifact_count"] == 3
    assert report["databricks_statement_count"] == 0
    assert report["authorities"]["model_input_authorized"] is False

    tampered = dict(package.artifacts)
    path = "source/day-ahead-export.parquet"
    tampered[path] = tampered[path] + b"tamper"
    with pytest.raises(EntsoeDayAheadExportError, match="artifact changed"):
        verify_export_replay_package(
            artifacts=tampered,
            manifest_payload=package.manifest_payload,
        )
