"""Run an off-production EPEX-only A/B shape lab on an hourly CH HPFC CSV.

The runner is deliberately not a production export path.  It derives monthly
BASE/PEAK preservation targets from the input candidate itself, applies the
experimental EPEX shape lab, and writes a lab-only manifest plus residual
audits.  OMPEX/HFC benchmark data is not read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.lt.model.epex_shape_lab import (  # noqa: E402
    ABShapeLabConfig,
    apply_epex_ab_shape_lab,
    build_ab_lab_manifest,
    fit_epex_shape_templates,
)
from pfc_shaping.lt.model.shape_constraints import (  # noqa: E402
    build_base_peak_offpeak_constraint_system,
    eex_peak_mask,
)
from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch  # noqa: E402


PRICE = "price_weighted_mean_eur_mwh"
REQUIRED_PRICE_COLUMNS = [
    "price_slow_eur_mwh",
    "price_central_eur_mwh",
    "price_fast_eur_mwh",
    PRICE,
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]


def run_ab(
    *,
    candidate_csv: Path,
    spot_parquet: Path,
    output_dir: Path,
    valuation_timestamp: str,
    weekend_intensity: float,
    low_tail_intensity: float,
    peak_subshape_intensity: float,
    max_abs_delta_eur_mwh: float,
    evening_recovery_intensity: float = 0.0,
    night_intensity: float = 0.0,
    ramp_intensity: float = 0.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
    lookback_years: int = 5,
    preserve_peak: bool = True,
) -> dict[str, Any]:
    """Run the lab and return the JSON-serializable manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate, ts_ch = _load_candidate_csv(candidate_csv)
    spot = pd.read_parquet(spot_parquet)
    base_constraints, peak_constraints = _derive_candidate_monthly_constraints(
        candidate,
        ts_ch,
        preserve_peak=preserve_peak,
    )

    config = ABShapeLabConfig(
        valuation_timestamp=valuation_timestamp,
        weekend_intensity=float(weekend_intensity),
        low_tail_intensity=float(low_tail_intensity),
        peak_subshape_intensity=float(peak_subshape_intensity),
        evening_recovery_intensity=float(evening_recovery_intensity),
        night_intensity=float(night_intensity),
        ramp_intensity=float(ramp_intensity),
        max_abs_delta_eur_mwh=float(max_abs_delta_eur_mwh),
        negative_price_floor=float(negative_price_floor),
        max_weighted_negative_hours=int(max_weighted_negative_hours),
        lookback_years=int(lookback_years),
        require_monthly_base_constraints=True,
    )
    templates, fit_info = fit_epex_shape_templates(spot, config)
    adjusted, audit = apply_epex_ab_shape_lab(
        candidate,
        templates,
        ts_ch=ts_ch,
        base_forward_prices=base_constraints,
        peak_forward_prices=peak_constraints,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        config=config,
    )

    before_residuals = _constraint_residuals(
        candidate,
        ts_ch,
        base_forward_prices=base_constraints,
        peak_forward_prices=peak_constraints,
        stage="before",
    )
    after_residuals = _constraint_residuals(
        adjusted,
        ts_ch,
        base_forward_prices=base_constraints,
        peak_forward_prices=peak_constraints,
        stage="after",
    )
    residuals = pd.concat([before_residuals, after_residuals], ignore_index=True)

    adjusted_path = output_dir / "candidate_epex_shape_lab_adjusted.csv"
    audit_path = output_dir / "ab_lab_audit.csv"
    residual_path = output_dir / "constraint_residuals_before_after.csv"
    template_path = output_dir / "epex_shape_templates.csv"
    pre_registration = _pre_registration(
        candidate_csv=candidate_csv,
        spot_parquet=spot_parquet,
        valuation_timestamp=valuation_timestamp,
        config=config,
        preserve_peak=preserve_peak,
    )
    pre_registration_path = output_dir / "pre_registered_ab_plan.json"
    manifest_path = output_dir / "ab_lab_manifest.json"

    adjusted.to_csv(adjusted_path, index=False)
    audit.to_csv(audit_path, index=False)
    residuals.to_csv(residual_path, index=False)
    templates.to_csv(template_path, index=False)
    pre_registration_path.write_text(json.dumps(pre_registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = build_ab_lab_manifest(
        config,
        fit_info=fit_info,
        source_hashes={
            "candidate_csv": _sha256(candidate_csv),
            "spot_parquet": _sha256(spot_parquet),
        },
        audit=audit,
    )
    manifest.update(
        {
            "runner": "scripts/run_epex_shape_lab_ab.py",
            "candidate_csv": str(candidate_csv),
            "spot_parquet": str(spot_parquet),
            "preserve_peak_constraints": bool(preserve_peak),
            "outputs": {
                "adjusted_csv": str(adjusted_path),
                "audit_csv": str(audit_path),
                "constraint_residuals_csv": str(residual_path),
                "templates_csv": str(template_path),
                "pre_registered_plan_json": str(pre_registration_path),
            },
            "constraint_summary": {
                "base_monthly_constraints": len(base_constraints),
                "peak_monthly_constraints": len(peak_constraints),
                "max_before_abs_error_eur_mwh": _max_abs_error(before_residuals),
                "max_after_abs_error_eur_mwh": _max_abs_error(after_residuals),
            },
            "candidate_delta_summary": _delta_summary(candidate, adjusted, ts_ch),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _load_candidate_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_PRICE_COLUMNS + ["timestamp_ch"]) - set(frame.columns))
    if missing:
        raise ValueError(f"candidate CSV missing required columns: {missing}")
    ts = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    out = frame.copy()
    for column in REQUIRED_PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    return out, pd.Series(ts, index=out.index)


def _derive_candidate_monthly_constraints(
    candidate: pd.DataFrame,
    ts_ch: pd.Series,
    *,
    preserve_peak: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    months = pd.PeriodIndex(ts_ch.dt.strftime("%Y-%m"), freq="M")
    base = {
        str(month): float(candidate.loc[months == month, PRICE].mean())
        for month in sorted(months.unique())
    }
    if not preserve_peak:
        return base, {}
    peak_mask = pd.Series(eex_peak_mask(pd.DatetimeIndex(ts_ch).tz_convert("UTC"), country="CH"), index=candidate.index)
    peak: dict[str, float] = {}
    for month in sorted(months.unique()):
        mask = (months == month) & peak_mask.to_numpy(dtype=bool)
        if bool(mask.any()):
            peak[str(month)] = float(candidate.loc[mask, PRICE].mean())
    return base, peak


def _constraint_residuals(
    frame: pd.DataFrame,
    ts_ch: pd.Series,
    *,
    base_forward_prices: dict[str, float],
    peak_forward_prices: dict[str, float],
    stage: str,
) -> pd.DataFrame:
    system = build_base_peak_offpeak_constraint_system(
        pd.DatetimeIndex(ts_ch).tz_convert("UTC"),
        base_forward_prices,
        peak_forward_prices,
        country="CH",
    )
    residuals = system.residuals(frame[PRICE].to_numpy(dtype=float))
    residuals.insert(0, "stage", stage)
    return residuals


def _pre_registration(
    *,
    candidate_csv: Path,
    spot_parquet: Path,
    valuation_timestamp: str,
    config: ABShapeLabConfig,
    preserve_peak: bool,
) -> dict[str, Any]:
    return {
        "ab_plan": "epex_shape_lab_candidate_ab",
        "activation_status": "lab_only",
        "production_approved": False,
        "candidate_csv": str(candidate_csv),
        "spot_parquet": str(spot_parquet),
        "candidate_csv_sha256": _sha256(candidate_csv),
        "spot_parquet_sha256": _sha256(spot_parquet),
        "valuation_timestamp": valuation_timestamp,
        "preserve_peak_constraints": bool(preserve_peak),
        "hypothesis": (
            "EPEX residual templates can improve hourly shape diagnostics while "
            "preserving candidate monthly BASE/PEAK levels."
        ),
        "primary_independent_checks": [
            "finite_prices",
            "quantile_order",
            "monthly_base_constraint_residuals",
            "monthly_peak_constraint_residuals",
            "negative_price_guard",
        ],
        "forbidden_checks": [
            "OMPEX as input",
            "OMPEX as target",
            "OMPEX as loss",
            "OMPEX as promotion gate",
        ],
        "config": {
            "weekend_intensity": float(config.weekend_intensity),
            "low_tail_intensity": float(config.low_tail_intensity),
            "peak_subshape_intensity": float(config.peak_subshape_intensity),
            "evening_recovery_intensity": float(config.evening_recovery_intensity),
            "night_intensity": float(config.night_intensity),
            "ramp_intensity": float(config.ramp_intensity),
            "max_abs_delta_eur_mwh": float(config.max_abs_delta_eur_mwh),
            "lookback_years": int(config.lookback_years),
            "require_monthly_base_constraints": bool(config.require_monthly_base_constraints),
        },
    }


def _delta_summary(before: pd.DataFrame, after: pd.DataFrame, ts_ch: pd.Series) -> dict[str, float | int]:
    delta = after[PRICE].astype(float).to_numpy(dtype=float) - before[PRICE].astype(float).to_numpy(dtype=float)
    weekend = ts_ch.dt.weekday >= 5
    solar_tail = ts_ch.dt.month.between(3, 10) & ts_ch.dt.hour.between(10, 16)
    night = ts_ch.dt.hour.between(0, 5)
    ramp = pd.Series(delta, index=ts_ch.index).diff().abs().dropna()
    return {
        "n_hours": int(len(delta)),
        "max_abs_delta_eur_mwh": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "mean_delta_eur_mwh": float(np.mean(delta)) if len(delta) else 0.0,
        "weekend_mean_delta_eur_mwh": float(np.mean(delta[weekend.to_numpy(dtype=bool)])) if bool(weekend.any()) else 0.0,
        "weekday_mean_delta_eur_mwh": float(np.mean(delta[(~weekend).to_numpy(dtype=bool)])) if bool((~weekend).any()) else 0.0,
        "solar_tail_mean_delta_eur_mwh": float(np.mean(delta[solar_tail.to_numpy(dtype=bool)])) if bool(solar_tail.any()) else 0.0,
        "night_mean_delta_eur_mwh": float(np.mean(delta[night.to_numpy(dtype=bool)])) if bool(night.any()) else 0.0,
        "delta_ramp_abs_p99_eur_mwh": float(ramp.quantile(0.99)) if not ramp.empty else 0.0,
    }


def _max_abs_error(residuals: pd.DataFrame) -> float:
    return float(residuals["abs_error"].max()) if not residuals.empty else 0.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an off-production EPEX-only A/B shape lab.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--spot-parquet", type=Path, default=Path("data/epex_hourly.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valuation-timestamp", required=True)
    parser.add_argument("--weekend-intensity", type=float, default=0.0)
    parser.add_argument("--low-tail-intensity", type=float, default=0.0)
    parser.add_argument("--peak-subshape-intensity", type=float, default=0.0)
    parser.add_argument("--evening-recovery-intensity", type=float, default=0.0)
    parser.add_argument("--night-intensity", type=float, default=0.0)
    parser.add_argument("--ramp-intensity", type=float, default=0.0)
    parser.add_argument("--max-abs-delta-eur-mwh", type=float, default=8.0)
    parser.add_argument("--negative-price-floor", type=float, default=-30.0)
    parser.add_argument("--max-weighted-negative-hours", type=int, default=0)
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--no-preserve-peak", action="store_true")
    args = parser.parse_args(argv)

    manifest = run_ab(
        candidate_csv=args.candidate_csv,
        spot_parquet=args.spot_parquet,
        output_dir=args.output_dir,
        valuation_timestamp=args.valuation_timestamp,
        weekend_intensity=args.weekend_intensity,
        low_tail_intensity=args.low_tail_intensity,
        peak_subshape_intensity=args.peak_subshape_intensity,
        evening_recovery_intensity=args.evening_recovery_intensity,
        night_intensity=args.night_intensity,
        ramp_intensity=args.ramp_intensity,
        max_abs_delta_eur_mwh=args.max_abs_delta_eur_mwh,
        negative_price_floor=args.negative_price_floor,
        max_weighted_negative_hours=args.max_weighted_negative_hours,
        lookback_years=args.lookback_years,
        preserve_peak=not args.no_preserve_peak,
    )
    print(json.dumps(manifest["constraint_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
