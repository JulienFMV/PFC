from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUEST = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "DATABRICKS-SPOT-WEATHER-SWISSGRID-DATA-ENGINEER-REQUEST-20260806.md"
)


def _request() -> str:
    return REQUEST.read_text(encoding="utf-8")


def test_cost_preflight_precedes_every_business_profile() -> None:
    request = _request()

    assert request.index("## Phase 0") < request.index("## Phase A")
    assert "sans lecture métier" in request
    assert "non autorisés en\nl'état" in request
    assert "LIMIT` borne le résultat retourné, pas les octets lus" in request
    assert "notre GO" in request


def test_preflight_requires_scan_and_warehouse_cost_evidence() -> None:
    request = _request()

    for requirement in (
        "taille Delta totale en octets",
        "nombre de fichiers",
        "colonnes de partition",
        "volume estimé lu",
        "déjà actif",
        "timeout et plafond de coût",
    ):
        assert requirement in request
    assert "ne doit ouvrir aucune ligne métier" in request
    assert "ne doit pas démarrer un\nWarehouse" in request


def test_request_binds_all_four_profiles_and_results() -> None:
    request = _request()

    for query in (
        "databricks_dev_spot_profile.sql",
        "databricks_prd_weather_profile.sql",
        "databricks_prd_swissgrid_balancing_profile.sql",
        "databricks_prd_swissgrid_tender_profile.sql",
    ):
        assert query in request
    for artifact in (
        "spot_profile.parquet",
        "weather_profile.parquet",
        "swissgrid_balancing_profile.parquet",
        "swissgrid_tender_profile.parquet",
        "source_semantics.md",
        "cost_receipt.json",
        "manifest.json",
    ):
        assert artifact in request


def test_request_preserves_source_semantics_and_decides_cross_border_families() -> None:
    request = _request()

    for requirement in (
        "droit/licence",
        "forecast",
        "date d'émission",
        "convention de signe",
        "caractère indicatif ou final",
        "FCR, aFRR, mFRR, RR",
        "NTC_MONTH_AHEAD",
        "NTC_DAY_AHEAD_D2",
        "NTC_INTRADAY",
        "SCHEDULED_EXCHANGE",
        "PHYSICAL_FLOW",
        "source_document_id",
    ):
        assert requirement in request
    assert "Merci d'indiquer `présent / absent / autre table`" not in request


def test_request_requires_gold_surfaces_and_one_local_snapshot() -> None:
    request = _request()

    for requirement in (
        "prd.gold.dimspotproduct",
        "prd.gold.factspotpriceinterval",
        "prd.gold.factweatherforecasthistms",
        "prd.gold.factweatherforecasthistom",
        "prd.gold.factswissgridbalancingquarterhourly",
        "prd.gold.factswissgridtenderofferresult",
        "prd.gold` les trois surfaces",
        "factentsoetimeseriesvintages",
        "build\\databricks-exports\\<snapshot_id>",
        "notre équipe** réalisera elle-même un snapshot",
        "Le data engineer ne doit donc pas produire ni déposer de fichier",
    ):
        assert requirement in request
    assert "La Gold est le contrat de consommation" in request
    assert "La Silver reste une preuve de" in request


def test_request_is_read_only_no_retry_and_non_authoritative() -> None:
    request = _request()

    assert "lecture seule" in request
    assert "aucun démarrage de Warehouse" in request
    assert "au maximum quatre statements" in request
    assert "sans retry\n  automatique" in request
    assert "aucune autorité PIT, modèle, sélection, promotion ou production" in request
    assert "aucune création ou modification de table/vue" in request


def test_full_export_remains_blocked_until_separate_go() -> None:
    request = _request()

    assert request.index("## Phase B") > request.index("## Phase A")
    assert "Phase B — publication Gold, puis export local par notre équipe" in request
    assert "ne pas lancer de backfill ni de dump des faits avant notre GO" in request


def test_data_engineer_publishes_gold_but_local_team_owns_snapshot() -> None:
    request = _request()

    assert "la livraison du data\nengineer s'arrête" in request
    assert "Notre équipe\neffectuera ensuite une lecture sélective" in request
    assert "Les Gold doivent permettre à notre exporteur" in request
    assert "déposer de fichier sur notre\n`C:`" in request
