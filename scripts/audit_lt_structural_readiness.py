"""Audit retained public scenario components and signed historical CH shapes.

Run from the canonical workspace with all runtime caches below build/. No
network, estimation, dispatch, new assembly adapter or operational authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pfc_shaping.lt.curve_products import AuthorityNegative
from pfc_shaping.lt.structural_readiness import audit_annual_inventory, center_signed_hourly_shape


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "electrification_scenarios_tyndp2024_supply.parquet": "09d042b6fe1ccc77c005a6b60dcd3d9a1a9b9c519886543d7a80dd36fec1907e",
    "electrification_scenarios_tyndp2024_demand.parquet": "9c63cf1da7ea862e8d5e116d1c1212e5ed36ae0e97f9ef4f0f1e553934aaaa3b",
    "electrification_scenarios_ep2050.parquet": "319105a4143544c4ce43082d6d7ed6f7a82bab87585b847f7da13d23966427e9",
    "electrification_scenarios_composed_p0_public_sources_2030.parquet": "f4a06bc8b8372744331de8e0465ab5de9b27a8da904a5d12a087e3471b0b11b7",
    "electrification_scenarios_prod_candidate_neutralized_2030.parquet": "4819bbf2c5bb486a1991218bb22924c78aa1ed24a3862a829d72a19e251d6a33",
}
D300 = ROOT / "build/local-pfc-source-preflight-20260907/prepared-inputs"
D300_MANIFEST_SHA = "36f27a705bdf56ff723d778d226ed3b5f4ae594f20e54b402cc3d6385088b3ef"
D301_PLAN_SHA = "d0ea1b8fc21725e74da9cdf1234e4e473c3f877f0b9c39c8a1856028a74f56e1"


def _verified_bytes(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"source hash mismatch: {path.name}")
    return raw


def _signed_history(output: Path, origin: pd.Timestamp) -> dict[str, object]:
    manifest = json.loads(_verified_bytes(D300 / "manifest.json", D300_MANIFEST_SHA))
    raw = _verified_bytes(D300 / "epex-ch.parquet", manifest["files"]["epex-ch.parquet"])
    prices = pd.read_parquet(io.BytesIO(raw))["price_eur_mwh"]
    if prices.index.tz is None or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("invalid historical CH delivery index")
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite historical CH prices")
    utc = prices.index.tz_convert("UTC").as_unit("ns")
    if ((utc.minute % 15 != 0) | (utc.second != 0) | (utc.microsecond != 0) | (utc.nanosecond != 0)).any():
        raise ValueError("invalid transport quarter-hour alignment")
    prices = pd.Series(prices.to_numpy(), index=utc)
    hourly_key = utc.floor("h")
    transport = prices.groupby(hourly_key).agg(["count", "nunique", "first"])
    if not transport["count"].eq(4).all() or not transport["nunique"].eq(1).all():
        raise ValueError("CH transport quarter-hours do not prove native hourly identity")
    hourly = transport["first"]
    records, shapes = [], []
    for month, values in hourly.groupby(hourly.index.tz_convert("Europe/Zurich").strftime("%Y-%m")):
        period = pd.Period(month, freq="M")
        start = period.start_time.tz_localize("Europe/Zurich")
        end = (period + 1).start_time.tz_localize("Europe/Zurich")
        expected = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
        row = {"month": month, "observed_hours": len(values), "expected_hours": len(expected)}
        if end.tz_convert("UTC") > origin:
            records.append({**row, "status": "EXCLUDED_NOT_CLOSED_AT_ORIGIN"})
            continue
        if not values.index.equals(expected):
            records.append({**row, "status": "EXCLUDED_INCOMPLETE_MONTH"})
            continue
        shape = center_signed_hourly_shape(values)
        # Independent arithmetic with a scalar mean, rather than groupby transform.
        manual = values.to_numpy() - np.mean(values.to_numpy())
        error = float(np.max(np.abs(manual - shape.to_numpy())))
        if error > 1e-9:
            raise ValueError("independent signed-target check failed")
        records.append({
            **row, "status": "PASS_SIGNED_TARGET_ARITHMETIC",
            "negative_raw_hours_retained": int(values.lt(0).sum()),
            "monthly_shape_mean_eur_mwh": float(shape.mean()),
            "independent_max_error_eur_mwh": error,
        })
        shapes.append(shape)
    if not shapes:
        raise ValueError("no complete CH month for arithmetic verification")
    pd.DataFrame(records).to_csv(output / "signed-history-months.csv", index=False)
    pd.concat(shapes).to_frame().to_parquet(output / "signed-historical-targets.parquet")
    accepted = [r for r in records if r["status"] == "PASS_SIGNED_TARGET_ARITHMETIC"]
    return {
        "purpose": "HISTORICAL_TARGET_ARITHMETIC_NOT_FORECAST_OR_TRAINING",
        "source_manifest_sha256": D300_MANIFEST_SHA,
        "source_prices_sha256": manifest["files"]["epex-ch.parquet"],
        "complete_months": len(accepted),
        "excluded_months": [r["month"] for r in records if r not in accepted],
        "hours": sum(r["observed_hours"] for r in accepted),
        "negative_raw_hours_retained": sum(r["negative_raw_hours_retained"] for r in accepted),
        "max_monthly_mean_abs_eur_mwh": max(abs(r["monthly_shape_mean_eur_mwh"]) for r in accepted),
        "independent_max_error_eur_mwh": max(r["independent_max_error_eur_mwh"] for r in accepted),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT or ROOT != Path(r"C:\Users\jbattaglia\PFC_LT"):
        raise ValueError("canonical workspace cwd required")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "build").resolve()) or output == ROOT / "build":
        raise ValueError("output must be a fresh directory below workspace build")
    origin = pd.Timestamp(args.as_of)
    if pd.isna(origin) or origin.tz is None:
        raise ValueError("as-of must have an explicit timezone")
    # Bind known sources before writing any output or parsing numerical values.
    captured = {name: _verified_bytes(ROOT / "data" / name, sha) for name, sha in SOURCES.items()}
    _verified_bytes(ROOT / "build/local-lt-benchmark-20260907/plan.json", D301_PLAN_SHA)
    output.mkdir(parents=True, exist_ok=False)
    reports, matrices = {}, []
    for name, raw in captured.items():
        table = pq.ParquetFile(io.BytesIO(raw))
        metadata_fields = [c for c in ("source", "component_ids", "quality_flag") if c in table.schema_arrow.names]
        metadata = table.read(columns=metadata_fields).to_pandas().astype(str)
        if metadata.apply(lambda c: c.str.contains("afry", case=False, regex=False)).any().any():
            raise ValueError("restricted source requires its separate verified catalog interface")
        frame = table.read().to_pandas()
        matrix, report = audit_annual_inventory(
            frame, as_of=origin, countries=("CH", "DE", "FR", "IT", "AT"),
            years=tuple(range(2030, 2036)),
        )
        reports[name] = {"sha256": SOURCES[name], "bytes": len(raw), **report}
        matrix.insert(0, "source_file", name)
        matrices.append(matrix)
    fields = pd.concat(matrices, ignore_index=True)
    fields.to_csv(output / "field-matrix.csv", index=False)
    identities = ["source_file", "scenario", "country", "delivery_year"]
    coverage = fields[[*identities, "row_status", "quality_flag"]].drop_duplicates()
    coverage.to_csv(output / "annual-coverage.csv", index=False)
    groups = fields.groupby([*identities, "group", "field_status"]).size().rename("field_count").reset_index()
    groups.to_csv(output / "mechanism-coverage.csv", index=False)
    history = _signed_history(output, origin)
    source_summary = []
    for name, report in reports.items():
        cells = coverage.loc[coverage["source_file"].eq(name)]
        source_summary.append({
            "source_file": name, "rows_read": report["rows_read"],
            "exact_cells": int(cells["row_status"].eq("PRESENT").sum()),
            "requested_cells": len(cells),
            "missing_cells": int(cells["row_status"].eq("NO_EXACT_ROW").sum()),
        })
    pd.DataFrame(source_summary).to_csv(output / "source-summary.csv", index=False)
    report = {
        "schema": "fmv-lt-structural-readiness.v1",
        "status": "LOCAL_CONTRACT_VERIFIED_STRUCTURAL_FORECAST_INPUTS_INCOMPLETE",
        "as_of": origin.isoformat(), "sources": reports, "signed_history": history,
        "models_fitted": 0, "forecast_curves_generated": 0, "source_values_imputed": 0,
        "d301_plan_sha256": D301_PLAN_SHA,
        "authority": AuthorityNegative().to_manifest(),
        "source_code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                "scripts/audit_lt_structural_readiness.py",
                "pfc_shaping/lt/structural_readiness.py",
                "docs/model/LT-STRUCTURAL-SHAPING-CONTRACT.md",
            )
        },
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Shaping 2030–2035 : inventaire exécuté et forme signée vérifiée", "",
        "Le contrat numérique fonctionne. Les trajectoires physiques et leurs chronologies restent incomplètes.",
        "Aucune nouvelle PFC, simulation de dispatch ou performance prédictive n'est annoncée.", "",
        "## Inventaires existants", "",
        "Sources auditables séparément ; leurs étiquettes ne désignent pas automatiquement le même scénario FMV.",
        "Une cellule = source/scénario/pays/année. Aucun remplissage entre années ni fusion entre sources.", "",
        "| Source locale | Lignes lues | Cellules exactes / demandées |",
        "| --- | ---: | ---: |",
    ]
    for row in source_summary:
        lines.append(f"| {row['source_file']} | {row['rows_read']} | {row['exact_cells']} / {row['requested_cells']} |")
    lines += [
        "", "TYNDP supply contient les millésimes 2030/2040 ; TYNDP demand les millésimes 2040/2050 et des noms de scénario différents.",
        "EP2050 couvre CH et plusieurs millésimes, mais n'établit pas la trajectoire complète du système voisin.",
        "Les deux assemblages anciens de 2030 conservent leurs labels partial/proxy/neutralized. Des colonnes remplies ne prouvent pas leur validité.",
        "", "## Contrôle de la forme signée sur les prix CH PRD retenus", "",
        f"- {history['complete_months']} mois complets, {history['hours']} heures natives.",
        f"- {history['negative_raw_hours_retained']} heures de prix négatifs conservées dans les cibles.",
        f"- Mois incomplets exclus : {', '.join(history['excluded_months'])}.",
        f"- Résidu moyen mensuel maximal : {history['max_monthly_mean_abs_eur_mwh']:.3e} EUR/MWh.",
        f"- Écart maximal du recalcul indépendant : {history['independent_max_error_eur_mwh']:.3e} EUR/MWh.",
        "- Ces valeurs vérifient une transformation arithmétique, pas la qualité d'une prévision.",
        "", "## Prochain lot utile", "",
        "Choisir et documenter les trajectoires physiques réconciliées, puis fournir les chronologies météo/demande/hydro et les contraintes de flexibilité.",
        "Le data ingénieur apporte les définitions, disponibilités et révisions ; l'équipe modèle définit le fonctionnement horaire et les hypothèses FMV.",
        "Le premier moteur réduit devra conserver l'énergie stockée/déplacée sur des heures successives, avec états initiaux et terminaux explicites.",
        "Le raccord à l'assembleur existant sera testé séparément avec les contraintes BASE/PEAK ; le solveur mensuel garde le niveau.",
        "", "## Fichiers", "",
        "`audit.json`, `source-summary.csv`, `annual-coverage.csv`, `field-matrix.csv`, `mechanism-coverage.csv`,",
        "`signed-history-months.csv`, `signed-historical-targets.parquet`, `manifest.json`.",
        "", "Toutes les autorités restent false. Aucun accès Warehouse, GPU, AFRY ou T057.",
    ]
    (output / "RAPPORT-SHAPING-2030.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "sources": source_summary, "signed_history": history}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
