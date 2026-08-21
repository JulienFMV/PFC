from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_ROOT = ROOT / "docs" / "data" / "sql"
DATA_ENGINEER_REQUEST = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "DATABRICKS-SPOT-WEATHER-SWISSGRID-DATA-ENGINEER-REQUEST-20260806.md"
)
MUTATION_TOKENS = {
    "alter",
    "copy",
    "create",
    "delete",
    "drop",
    "grant",
    "insert",
    "merge",
    "optimize",
    "replace",
    "revoke",
    "truncate",
    "update",
    "vacuum",
}


def _executable_sql(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))


def _executable_tokens(sql: str) -> set[str]:
    return set(re.findall(r"[a-z_]+", _executable_sql(sql).lower()))


def _assert_not_execution_authorized(sql: str) -> None:
    assert "NON AUTORISÉ EN L'ÉTAT" in sql
    assert "LIMIT borne les lignes retournées, pas les octets lus" in sql
    assert "taille Delta, partitions et plan estimé" in sql


def test_spot_profile_is_bounded_read_only_and_uses_only_candidate_table() -> None:
    sql = (SQL_ROOT / "databricks_dev_spot_profile.sql").read_text(encoding="utf-8")
    tokens = _executable_tokens(sql)

    _assert_not_execution_authorized(sql)
    assert not MUTATION_TOKENS & tokens
    assert "`dev`.`silver`.`ge_market_euler_spot`" in sql
    assert "LIMIT 10001" in sql
    assert "interval_duration_mismatch_count" in sql
    assert "invalid_frequency_count" in sql
    assert "TIMESTAMPADD(MINUTE, `frequency_minutes`, `ts_start_utc`)" in sql
    assert "INTERVAL 1 MINUTE *" not in sql
    assert "price_unit" in sql
    assert "source_row_id" in sql
    assert _executable_sql(sql).count(";") == 0


def test_weather_profile_is_bounded_read_only_and_does_not_invent_issue_time() -> None:
    sql = (SQL_ROOT / "databricks_prd_weather_profile.sql").read_text(encoding="utf-8")
    tokens = _executable_tokens(sql)

    _assert_not_execution_authorized(sql)
    assert not MUTATION_TOKENS & tokens
    assert "`prd`.`silver`.`weather_meteoswiss_measurement`" in sql
    assert "`prd`.`silver`.`weather_meteoswiss_forecast`" in sql
    assert "`prd`.`silver`.`weather_openmeteo_forecast`" in sql
    assert "LIMIT 20001" in sql
    assert "forecast_issue_utc" not in "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    assert "lead_time" in sql
    assert "meta_load_timestamp_utc" in sql
    assert _executable_sql(sql).count(";") == 0


def test_swissgrid_balancing_profile_is_bounded_read_only_and_preserves_semantics() -> None:
    sql = (SQL_ROOT / "databricks_prd_swissgrid_balancing_profile.sql").read_text(encoding="utf-8")
    tokens = _executable_tokens(sql)

    _assert_not_execution_authorized(sql)
    assert not MUTATION_TOKENS & tokens
    assert "`prd`.`silver`.`ge_power_swissgrid_cab`" in sql
    assert "LIMIT 1" in sql
    assert "duplicated_target_count" in sql
    assert "invalid_quarter_hour_alignment_count" in sql
    assert "PMOD(UNIX_MICROS(`ts_start_utc`), 900000000) <> 0" in sql
    assert "MINUTE(`ts_start_utc`)" not in sql
    assert "SECOND(`ts_start_utc`)" not in sql
    assert "COALESCE(MAX(duplicate_summary.`duplicated_target_count`), 0)" in sql
    assert "afrr_plus_observed_count" in sql
    assert "direct_mfrr_plus_observed_count" in sql
    assert "one_price_observed_count" in sql
    assert _executable_sql(sql).count(";") == 0


def test_swissgrid_tender_profile_is_bounded_read_only_and_keeps_units() -> None:
    sql = (SQL_ROOT / "databricks_prd_swissgrid_tender_profile.sql").read_text(encoding="utf-8")
    tokens = _executable_tokens(sql)

    _assert_not_execution_authorized(sql)
    assert not MUTATION_TOKENS & tokens
    assert "`prd`.`silver`.`ge_market_swissgrid_sdl_tenders`" in sql
    assert "LIMIT 20001" in sql
    assert "capacity_price_unit" in sql
    assert "awarded_volume_unit" in sql
    assert "source_snapshot_ts" in sql
    assert "dq_missing_business_fields_count" in sql
    assert _executable_sql(sql).count(";") == 0


def test_data_engineer_request_remains_profile_first_and_bounded() -> None:
    request = DATA_ENGINEER_REQUEST.read_text(encoding="utf-8")

    assert "## Phase A — profils uniquement" in request
    assert "## Phase B — publication Gold, puis export local par notre équipe" in request
    assert request.count("docs/data/sql/databricks_") == 4
    assert "ne pas lancer de backfill" in request
    assert "aucune création ou modification de table/vue" in request
    assert "d'enrichir les tables Gold demandées en Phase B" in request
    assert "source_semantics.md" in request
    assert "manifest.json" in request
    assert "DATABRICKS_TOKEN" not in request
