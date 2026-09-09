from __future__ import annotations

from scripts.audit_bfe_opendata_catalog import build_catalog_audit, main


def _packages():
    return [
        {
            "name": "fullungsgrad-der-speicherseen",
            "title": {"en": "Reservoir filling levels"},
            "description": {"en": "weekly reservoir content in GWh"},
            "keywords": {"en": ["hydro"]},
            "resources": [{"format": "CSV", "download_url": "https://example.test/reservoir.csv"}],
        },
        {
            "name": "unrelated",
            "title": {"en": "Unrelated"},
            "description": {"en": "not energy modelling"},
            "resources": [],
        },
    ]


def test_build_catalog_audit_classifies_relevant_packages():
    audit = build_catalog_audit(_packages())

    reservoir = audit[audit["dataset_name"] == "fullungsgrad-der-speicherseen"].iloc[0]
    assert reservoir["priority"] == "P0"
    assert "hydro_reservoir" in reservoir["family"]
    other = audit[audit["dataset_name"] == "unrelated"].iloc[0]
    assert other["priority"] == "P3"


def test_audit_bfe_opendata_catalog_writes_outputs(monkeypatch, tmp_path):
    import scripts.audit_bfe_opendata_catalog as mod

    monkeypatch.setattr(mod, "fetch_bfe_packages", lambda rows=200: _packages())
    output = tmp_path / "audit.csv"
    report = tmp_path / "audit.md"

    rc = main(["--output", str(output), "--report", str(report)])

    assert rc == 0
    assert output.exists()
    assert "BFE opendata.swiss Catalogue Audit" in report.read_text(encoding="utf-8")
