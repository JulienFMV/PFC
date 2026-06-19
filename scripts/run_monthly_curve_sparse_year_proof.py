"""Run a diagnostic sparse-year monthly BASE curve proof.

This script is deliberately not a production path.  It exercises the new
monthly constraint system, zero-mean shape priors and KKT solver on the local
EEX history parquet, then emits machine-readable diagnostics for 2027-2030.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.calibration.monthly_curve_audit import audit_monthly_curve_shape
from pfc_shaping.calibration.monthly_curve_priors import (
    build_fused_shape_prior,
    build_history_shape_prior,
    build_neighbor_panel_shape_prior,
    build_structural_monthly_shape_prior,
    quote_coverage_by_horizon,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    solve_monthly_forward_curve_from_constraints,
)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_parquet(args.forwards)
    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None).dt.normalize()
    months = pd.period_range(args.start, args.end, freq="M")
    neighbor_markets = tuple(
        market.strip().upper()
        for market in str(args.neighbor_markets).split(",")
        if market.strip()
    )

    own_quotes = _latest_quotes(history, args.market, "BASE")
    constraints = build_monthly_constraint_system(
        months,
        own_quotes,
        market=args.market,
        timezone=args.timezone,
        constraint_tolerance=float(args.quote_consistency_tolerance),
    )

    neighbor_prices = {
        market: {quote.product: quote.price for quote in _latest_quotes(history, market, "BASE")}
        for market in neighbor_markets
    }
    panel = build_neighbor_panel_shape_prior(
        constraints,
        neighbor_prices,
        neighbor_markets=neighbor_markets,
        neighbor_shrinkage=float(args.neighbor_shrinkage),
    )
    structural = build_structural_monthly_shape_prior(
        constraints,
        amplitude_eur_mwh=float(args.structural_amplitude_eur_mwh),
    )
    historical = build_history_shape_prior(
        constraints,
        history,
        market=args.market,
        load_type="BASE",
        run_timestamp=max(q.snapshot_date for q in own_quotes if q.snapshot_date is not None),
        min_snapshots=int(args.min_history_snapshots),
        lookback_years=int(args.history_lookback_years),
    )
    fused = build_fused_shape_prior(
        constraints,
        panel_prior=panel,
        history_prior=historical,
        structural_prior=structural,
        weights={
            "panel": float(args.panel_weight),
            "history": float(args.history_weight),
            "structural": float(args.structural_weight),
        },
    )
    config = MonthlyCurveConfig(
        lambda_prior=float(args.lambda_prior),
        lambda_smooth_month=float(args.lambda_smooth_month),
        lambda_smooth_yoy=float(args.lambda_smooth_yoy),
        lambda_shape=float(args.lambda_shape),
        neighbor_shrinkage=float(args.neighbor_shrinkage),
        min_history_snapshots=int(args.min_history_snapshots),
        constraint_tolerance=float(args.quote_consistency_tolerance),
    )
    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=config,
        shape_prior=fused,
    )

    monthly = _monthly_curve_frame(result.monthly_curve, constraints)
    checks = _sparse_year_checks(monthly, constraints, year_a=2028, year_b=2029)
    audit_gates = audit_monthly_curve_shape(result.monthly_curve, constraints, year_pairs=[(2028, 2029)])
    coverage = quote_coverage_by_horizon(history)
    _write_outputs(
        output_dir=output_dir,
        monthly=monthly,
        constraints=constraints.rows,
        quote_diagnostics=constraints.quote_diagnostics,
        residuals=result.residuals,
        priors=result.priors,
        diagnostics=result.diagnostics,
        panel_diagnostics=panel.diagnostics,
        history_diagnostics=historical.diagnostics,
        fused_diagnostics=fused.diagnostics,
        checks=checks,
        audit_gates=audit_gates,
        coverage=coverage,
        manifest={
            "script": Path(__file__).as_posix(),
            "market": args.market,
            "start": args.start,
            "end": args.end,
            "forwards": str(args.forwards),
            "latest_snapshot": str(max(q.snapshot_date for q in own_quotes if q.snapshot_date is not None)),
            "neighbor_markets": neighbor_markets,
            "panel_status": panel.status,
            "history_status": historical.status,
            "fused_status": fused.status,
            "solver_kkt": result.kkt,
            "gate_summary": audit_gates["status"].value_counts().to_dict(),
            "config": config.__dict__,
            "production_approved": False,
        },
    )
    if not args.no_plot:
        _plot_monthly(output_dir / "monthly_curve_2027_2030.png", monthly)

    pivot = monthly.pivot(index="month", columns="year", values="price_eur_mwh").round(2)
    print(pivot.to_string())
    print(f"max_abs_constraint_residual={result.kkt['max_abs_constraint_residual']:.3e}")
    print(f"gate_summary={audit_gates['status'].value_counts().to_dict()}")
    print(f"panel_status={panel.status} history_status={historical.status} fused_status={fused.status}")
    print(f"wrote {output_dir}")


def _latest_quotes(history: pd.DataFrame, market: str, load_type: str) -> tuple[MarketQuote, ...]:
    market = str(market).upper()
    load_type = str(load_type).upper()
    sub = history[
        (history["market"].astype(str).str.upper() == market)
        & (history["load_type"].astype(str).str.upper() == load_type)
    ].copy()
    if sub.empty:
        raise ValueError(f"no EEX quotes for market={market}, load_type={load_type}")
    latest = sub["date"].max()
    snap = sub[sub["date"].eq(latest)]
    return tuple(
        MarketQuote(
            market=market,
            product=str(row.product),
            load_type=load_type,
            price=float(row.price),
            snapshot_date=pd.Timestamp(row.date),
            source=str(row.source),
            available_at=pd.Timestamp(row.date),
        )
        for row in snap.itertuples(index=False)
    )


def _monthly_curve_frame(curve: pd.Series, constraints) -> pd.DataFrame:
    rows = []
    parent_map = {
        str(row["product"]): str(row["parent_product"])
        for _, row in constraints.rows.iterrows()
    }
    for month, price in curve.items():
        bucket = constraints.month_buckets.loc[month]
        bucket_text = "" if pd.isna(bucket) else str(bucket)
        rows.append(
            {
                "month_key": str(month),
                "year": int(month.year),
                "month": int(month.month),
                "price_eur_mwh": float(price),
                "parent_block_id": bucket_text,
                "parent_product": parent_map.get(bucket_text, bucket_text),
                "parent_target_eur_mwh": constraints.bucket_targets.get(bucket_text),
            }
        )
    return pd.DataFrame(rows)


def _sparse_year_checks(monthly: pd.DataFrame, constraints, *, year_a: int, year_b: int) -> pd.DataFrame:
    rows = []
    prices = {
        (int(row.year), int(row.month)): float(row.price_eur_mwh)
        for row in monthly.itertuples(index=False)
    }
    parent = {
        (int(row.year), int(row.month)): str(row.parent_block_id)
        for row in monthly.itertuples(index=False)
    }
    cal_a = _quote_target(constraints, str(year_a))
    cal_b = _quote_target(constraints, str(year_b))
    for month in range(1, 13):
        if (year_a, month) not in prices or (year_b, month) not in prices:
            continue
        rows.append(
            {
                "year_a": year_a,
                "year_b": year_b,
                "month": month,
                "calendar_spread_eur_mwh": None if cal_a is None or cal_b is None else cal_a - cal_b,
                "month_spread_eur_mwh": prices[(year_a, month)] - prices[(year_b, month)],
                "price_a_eur_mwh": prices[(year_a, month)],
                "price_b_eur_mwh": prices[(year_b, month)],
                "parent_block_a": parent[(year_a, month)],
                "parent_block_b": parent[(year_b, month)],
                "status": "INFO",
            }
        )
    return pd.DataFrame(rows)


def _quote_target(constraints, product: str) -> float | None:
    diag = constraints.quote_diagnostics
    if diag.empty:
        return None
    row = diag[diag["product"].astype(str).eq(str(product))]
    if row.empty:
        return None
    return float(row["target"].iloc[0])


def _write_outputs(
    *,
    output_dir: Path,
    monthly: pd.DataFrame,
    constraints: pd.DataFrame,
    quote_diagnostics: pd.DataFrame,
    residuals: pd.DataFrame,
    priors: pd.DataFrame,
    diagnostics: pd.DataFrame,
    panel_diagnostics: pd.DataFrame,
    history_diagnostics: pd.DataFrame,
    fused_diagnostics: pd.DataFrame,
    checks: pd.DataFrame,
    audit_gates: pd.DataFrame,
    coverage: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    monthly.to_csv(output_dir / "monthly_curve.csv", index=False)
    constraints.to_csv(output_dir / "monthly_curve_constraints.csv", index=False)
    quote_diagnostics.to_csv(output_dir / "quote_diagnostics.csv", index=False)
    residuals.to_csv(output_dir / "monthly_curve_residuals.csv", index=False)
    priors.to_csv(output_dir / "monthly_curve_priors.csv", index=False)
    diagnostics.to_csv(output_dir / "monthly_curve_diagnostics.csv", index=False)
    panel_diagnostics.to_csv(output_dir / "panel_prior_diagnostics.csv", index=False)
    history_diagnostics.to_csv(output_dir / "history_prior_diagnostics.csv", index=False)
    fused_diagnostics.to_csv(output_dir / "fused_prior_diagnostics.csv", index=False)
    checks.to_csv(output_dir / "sparse_year_checks.csv", index=False)
    audit_gates.to_csv(output_dir / "audit_gates.csv", index=False)
    coverage.to_csv(output_dir / "monthly_quote_coverage_by_horizon.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _plot_monthly(path: Path, monthly: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for year, group in monthly.groupby("year", sort=True):
        ax.plot(group["month"], group["price_eur_mwh"], marker="o", linewidth=1.8, label=str(year))
    ax.set_title("Diagnostic monthly BASE curve")
    ax.set_xlabel("Month")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Year")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forwards", type=Path, default=Path("data/eex_forwards_history.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/monthly_curve_sparse_year_proof"))
    parser.add_argument("--market", default="CH")
    parser.add_argument("--start", default="2027-01")
    parser.add_argument("--end", default="2030-12")
    parser.add_argument("--timezone", default="Europe/Zurich")
    parser.add_argument("--neighbor-markets", default="DE,FR,AT,IT")
    parser.add_argument("--quote-consistency-tolerance", type=float, default=0.01)
    parser.add_argument("--lambda-prior", type=float, default=1e-6)
    parser.add_argument("--lambda-smooth-month", type=float, default=1.0)
    parser.add_argument("--lambda-smooth-yoy", type=float, default=0.25)
    parser.add_argument("--lambda-shape", type=float, default=4.0)
    parser.add_argument("--neighbor-shrinkage", type=float, default=0.5)
    parser.add_argument("--structural-amplitude-eur-mwh", type=float, default=20.0)
    parser.add_argument("--panel-weight", type=float, default=1.0)
    parser.add_argument("--history-weight", type=float, default=0.25)
    parser.add_argument("--structural-weight", type=float, default=0.5)
    parser.add_argument("--min-history-snapshots", type=int, default=24)
    parser.add_argument("--history-lookback-years", type=int, default=6)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
