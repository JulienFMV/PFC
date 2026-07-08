"""Stage an LT-only EPEX lab adjusted candidate from audited LT outputs.

The runner is deliberately off-by-default and non-promotional.  It starts from
either an LT fan parquet or an already exported hourly CH CSV, applies the
EPEX-only shape lab with explicit parameters, and writes hash-bound staging
evidence.  It does not read OMPEX and it does not mark anything production
approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_epex_lab_adjusted_production_manifest import build_manifest as build_production_contract
from scripts.export_local_test_ch_hourly_csv import to_hourly_csv_frame
from scripts.run_epex_shape_lab_ab import run_ab


def stage_candidate(
    *,
    output_dir: Path,
    spot_parquet: Path,
    valuation_timestamp: str,
    fan_parquet: Path | None = None,
    candidate_csv: Path | None = None,
    local_start_date: str | None = None,
    local_end_date: str | None = None,
    weekend_intensity: float = 0.5,
    low_tail_intensity: float = 0.25,
    peak_subshape_intensity: float = 0.75,
    max_abs_delta_eur_mwh: float = 3.0,
    negative_price_floor: float = -10.0,
    max_weighted_negative_hours: int = 0,
    lookback_years: int = 5,
    baseline_monthly_manifest: Path | None = None,
    product_summary: Path | None = None,
    powerbi_summary: Path | None = None,
    source_hierarchy_policy: Path | None = None,
    independent_summary: Path | None = None,
    governance_audit: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_candidate_csv = output_dir / "lt_hourly_candidate.csv"
    if (fan_parquet is None) == (candidate_csv is None):
        raise ValueError("provide exactly one of fan_parquet or candidate_csv")
    if fan_parquet is not None:
        if local_start_date is None or local_end_date is None:
            raise ValueError("fan_parquet staging requires local_start_date and local_end_date")
        fan = pd.read_parquet(fan_parquet)
        hourly = to_hourly_csv_frame(
            fan,
            local_start_date=local_start_date,
            local_end_date=local_end_date,
        )
        hourly.to_csv(staged_candidate_csv, index=False)
        source_kind = "fan_parquet"
        source_path = fan_parquet
    else:
        source_kind = "candidate_csv"
        source_path = candidate_csv
        hourly = pd.read_csv(candidate_csv)
        hourly.to_csv(staged_candidate_csv, index=False)

    lab_dir = output_dir / "epex_lab"
    lab_manifest = run_ab(
        candidate_csv=staged_candidate_csv,
        spot_parquet=spot_parquet,
        output_dir=lab_dir,
        valuation_timestamp=valuation_timestamp,
        weekend_intensity=weekend_intensity,
        low_tail_intensity=low_tail_intensity,
        peak_subshape_intensity=peak_subshape_intensity,
        max_abs_delta_eur_mwh=max_abs_delta_eur_mwh,
        negative_price_floor=negative_price_floor,
        max_weighted_negative_hours=max_weighted_negative_hours,
        lookback_years=lookback_years,
        preserve_peak=True,
    )
    lab_manifest_path = lab_dir / "ab_lab_manifest.json"

    staging_manifest_path = output_dir / "staged_lt_epex_lab_candidate_manifest.json"
    staging_manifest: dict[str, Any] = {
        "schema_version": "epex_lab_adjusted_lt_candidate_stage.v1",
        "activation_status": "staged_lab_only",
        "production_approved": False,
        "production_promotion_approved": False,
        "promotion_scope": "LT_EPEX_LAB_STAGING_NO_GO",
        "source_kind": source_kind,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "staged_candidate_csv": str(staged_candidate_csv),
        "staged_candidate_csv_sha256": _sha256(staged_candidate_csv),
        "spot_parquet": str(spot_parquet),
        "spot_parquet_sha256": _sha256(spot_parquet),
        "lab_manifest": str(lab_manifest_path),
        "lab_manifest_sha256": _sha256(lab_manifest_path),
        "adjusted_csv": str((lab_manifest.get("outputs") or {}).get("adjusted_csv")),
        "adjusted_csv_sha256": _sha256(Path(str((lab_manifest.get("outputs") or {}).get("adjusted_csv")))),
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "epex_lab_config": {
            "valuation_timestamp": valuation_timestamp,
            "weekend_intensity": float(weekend_intensity),
            "low_tail_intensity": float(low_tail_intensity),
            "peak_subshape_intensity": float(peak_subshape_intensity),
            "max_abs_delta_eur_mwh": float(max_abs_delta_eur_mwh),
            "negative_price_floor": float(negative_price_floor),
            "max_weighted_negative_hours": int(max_weighted_negative_hours),
            "lookback_years": int(lookback_years),
            "preserve_peak": True,
        },
        "note": (
            "This staging manifest proves reproducible LT-to-EPEX-lab candidate "
            "construction only. It is not production promotion evidence."
        ),
    }
    production_contract_inputs = {
        "baseline_monthly_manifest": baseline_monthly_manifest,
        "product_summary": product_summary,
        "powerbi_summary": powerbi_summary,
        "source_hierarchy_policy": source_hierarchy_policy,
        "independent_summary": independent_summary,
        "governance_audit": governance_audit,
    }
    missing_contract_inputs = [
        name for name, value in production_contract_inputs.items() if value is None or not value.exists()
    ]
    staging_manifest["missing_production_contract_inputs"] = missing_contract_inputs
    if not missing_contract_inputs:
        contract_path = output_dir / "adjusted_production_manifest_no_go.json"
        contract = build_production_contract(
            lab_manifest=lab_manifest_path,
            baseline_monthly_manifest=baseline_monthly_manifest,
            product_summary=product_summary,
            powerbi_summary=powerbi_summary,
            source_hierarchy_policy=source_hierarchy_policy,
            independent_summary=independent_summary,
            governance_audit=governance_audit,
            production_run_id="LT_EPEX_LAB_STAGING_NO_GO",
            production_entrypoint="scripts/stage_epex_lab_adjusted_lt_candidate.py",
            git_commit=None,
            output=contract_path,
        )
        staging_manifest["adjusted_production_manifest_no_go"] = str(contract_path)
        staging_manifest["adjusted_production_manifest_no_go_sha256"] = _sha256(contract_path)
        staging_manifest["adjusted_production_contract_pass"] = bool(contract.get("contract_pass"))
    staging_manifest_path.write_text(
        json.dumps(staging_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return staging_manifest


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fan-parquet", type=Path)
    source.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spot-parquet", type=Path, required=True)
    parser.add_argument("--valuation-timestamp", required=True)
    parser.add_argument("--local-start-date")
    parser.add_argument("--local-end-date")
    parser.add_argument("--weekend-intensity", type=float, default=0.5)
    parser.add_argument("--low-tail-intensity", type=float, default=0.25)
    parser.add_argument("--peak-subshape-intensity", type=float, default=0.75)
    parser.add_argument("--max-abs-delta-eur-mwh", type=float, default=3.0)
    parser.add_argument("--negative-price-floor", type=float, default=-10.0)
    parser.add_argument("--max-weighted-negative-hours", type=int, default=0)
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--baseline-monthly-manifest", type=Path)
    parser.add_argument("--product-summary", type=Path)
    parser.add_argument("--powerbi-summary", type=Path)
    parser.add_argument("--source-hierarchy-policy", type=Path)
    parser.add_argument("--independent-summary", type=Path)
    parser.add_argument("--governance-audit", type=Path)
    args = parser.parse_args(argv)
    manifest = stage_candidate(
        fan_parquet=args.fan_parquet,
        candidate_csv=args.candidate_csv,
        output_dir=args.output_dir,
        spot_parquet=args.spot_parquet,
        valuation_timestamp=args.valuation_timestamp,
        local_start_date=args.local_start_date,
        local_end_date=args.local_end_date,
        weekend_intensity=args.weekend_intensity,
        low_tail_intensity=args.low_tail_intensity,
        peak_subshape_intensity=args.peak_subshape_intensity,
        max_abs_delta_eur_mwh=args.max_abs_delta_eur_mwh,
        negative_price_floor=args.negative_price_floor,
        max_weighted_negative_hours=args.max_weighted_negative_hours,
        lookback_years=args.lookback_years,
        baseline_monthly_manifest=args.baseline_monthly_manifest,
        product_summary=args.product_summary,
        powerbi_summary=args.powerbi_summary,
        source_hierarchy_policy=args.source_hierarchy_policy,
        independent_summary=args.independent_summary,
        governance_audit=args.governance_audit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
