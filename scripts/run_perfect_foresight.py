#!/usr/bin/env python3
"""One-command runner for the Perfect-Foresight Shaping diagnostic (Phase 10).

Produces, for a fully-realized delivery year (default Cal 2025):
  * a markdown report decomposing seasonal-shape error into forecast vs shaping,
    across a granularity ladder (Cal -> Cal+Quarter -> market) and a training-
    maturity sweep;
  * the de-levelled intra-day shape scores (cosine + demeaned RMSE);
  * Swiss-physical sub-KPIs (winter/summer ratio, solar-bowl depth, peak/off-peak
    spread) model vs realized;
  * publication-quality figures.

This is strictly additive and never touches the reproducibility-guarded pillars.

Run (canonical pinned env):
    PYTHONPATH=. /tmp/venv233/bin/python scripts/run_perfect_foresight.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Scoped warning filters — silence only the noise the diagnostic produces
# (pandas FutureWarnings on tz-aware datetime ops, scipy RuntimeWarnings on
# empty slices). Do NOT use a blanket `filterwarnings("ignore")`: it would also
# hide real DeprecationWarnings from pandas/numpy upgrades and the project's own
# `Phase10ReproducibilityWarning`.
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

_REPO_ROOT = Path(__file__).resolve().parents[1]
EPEX_REL = "data/epex_hourly.parquet"
FORWARDS_REL = "data/forwards_history_phase10.parquet"


def _granularity_ladder(
    epex: pd.Series,
    forwards: pd.DataFrame,
    vintage: pd.Timestamp,
    target_year: int,
    config_name: str,
) -> dict[str, object]:
    """Compute monthly-signature corr for Cal / Cal+Quarter / market anchors.

    Returns the three correlations plus the anchor key sets, so the report can
    document apples-to-apples vs apples-to-oranges comparisons explicitly (the
    market quote set varies by vintage; printing it defuses any misreading).
    """
    from scipy.stats import pearsonr

    from pfc_shaping.validation.perfect_foresight import (
        build_curve,
        monthly_signature,
        realized_window_mean,
    )
    from pfc_shaping.validation.scorecard import _forwards_for_vintage

    real_sig = monthly_signature(epex, target_year)

    def corr(anchors: dict) -> float:
        if not anchors:
            return float("nan")
        sig = monthly_signature(build_curve(config_name, vintage, epex, anchors), target_year)
        both = pd.concat([sig, real_sig], axis=1, join="inner").dropna()
        if len(both) < 3:
            return float("nan")
        return float(pearsonr(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())[0])

    cal_anchor = {str(target_year): realized_window_mean(epex, target_year)}
    calq_anchor = dict(cal_anchor)
    for q in range(1, 5):
        val = realized_window_mean(epex, target_year, quarter=q)
        if val is not None:
            calq_anchor[f"{target_year}-Q{q}"] = val
    market = _forwards_for_vintage(forwards, vintage, epex)

    return {
        "pf_cal": {"corr": corr(cal_anchor), "keys": sorted(cal_anchor)},
        "pf_cal_quarter": {"corr": corr(calq_anchor), "keys": sorted(calq_anchor)},
        "market": {"corr": corr(market), "keys": sorted(market) if market else []},
    }


def _make_figures(epex, fwds, res, target_year, config_name, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pfc_shaping.validation.perfect_foresight import (
        build_curve,
        monthly_signature,
        realized_window_mean,
    )
    from pfc_shaping.validation.scorecard import _forwards_for_vintage, list_vintages_2024_2025

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    TZ = "Europe/Zurich"

    # Fig 1 — maturity sweep: market vs PF-cal correlation vs training years.
    s = res.sweep
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s["train_years"], s["market_corr"], "o-", label="market anchors (traded granularity)")
    ax.plot(s["train_years"], s["pf_cal_corr"], "s-", label="perfect-foresight Cal (cascader profiling only)")
    ax.axhline(0.85, ls="--", c="grey", lw=1, label="SC#1 threshold 0.85")
    ax.set_xlabel("training history available at vintage (years)")
    ax.set_ylabel(f"monthly-shape correlation vs realized Cal {target_year}")
    ax.set_title(f"Seasonal-shape skill vs training maturity (delivery {target_year})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = out_dir / "pf_maturity_sweep.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    best = max((v for v in list_vintages_2024_2025() if v.year < target_year),
               key=lambda v: (epex[epex.index < v].index.max() - epex[epex.index < v].index.min()).days)

    # Fig 2 — monthly signature: realized vs PF-cal vs market.
    real_sig = monthly_signature(epex, target_year)
    pf_sig = monthly_signature(build_curve(config_name, best, epex, {str(target_year): realized_window_mean(epex, target_year)}), target_year)
    mkt_sig = monthly_signature(build_curve(config_name, best, epex, _forwards_for_vintage(fwds, best, epex)), target_year)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(real_sig.index, real_sig.values, "k-o", label="realized")
    ax.plot(pf_sig.index, pf_sig.values, "s--", label="perfect-foresight Cal")
    ax.plot(mkt_sig.index, mkt_sig.values, "^--", label="market anchors")
    ax.set_xlabel("month"); ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Monthly signature Cal {target_year} (best-trained vintage {best.date()})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = out_dir / "pf_monthly_signature.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # Fig 3 — CH-physical sub-KPIs model vs realized.
    k = res.subkpis
    labels = ["winter/summer\nratio", "solar-bowl\ndepth", "peak-offpeak\nspread (EUR/MWh)"]
    model = [k["winter_summer_ratio"]["model"], k["solar_bowl_depth"]["model"], k["peak_offpeak_spread"]["model"]]
    real = [k["winter_summer_ratio"]["realized"], k["solar_bowl_depth"]["realized"], k["peak_offpeak_spread"]["realized"]]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, model, w, label="model")
    ax.bar(x + w / 2, real, w, label="realized")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"Swiss-physical shape sub-KPIs (delivery {target_year}, vintage {k['vintage']})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    p = out_dir / "pf_ch_subkpis.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # Fig 4 — de-levelled diurnal profile, realized vs model (summer).
    pf_curve = build_curve(config_name, best, epex, {str(target_year): realized_window_mean(epex, target_year)})
    m = pf_curve.tz_convert(TZ); r = epex.tz_convert(TZ)
    summer = (np.isin(m.index.month, (6, 7, 8))) & (m.index.year == target_year)
    summer_r = (np.isin(r.index.month, (6, 7, 8))) & (r.index.year == target_year)
    pm = m[summer].groupby(m[summer].index.hour).mean(); pm = pm - pm.mean()
    pr = r[summer_r].groupby(r[summer_r].index.hour).mean(); pr = pr - pr.mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pr.index, pr.values, "k-o", label="realized (de-levelled)")
    ax.plot(pm.index, pm.values, "s--", label="model (de-levelled)")
    ax.set_xlabel("hour of day (local)"); ax.set_ylabel("EUR/MWh vs daily mean")
    ax.set_title(f"Summer diurnal shape Cal {target_year} (de-levelled) — solar bowl")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = out_dir / "pf_diurnal_summer.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    return paths


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table writer — avoids the optional `tabulate` dep."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def _fmt(x, spec: str = ".3f") -> str:
    """Format helper that gracefully handles None/NaN."""
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and not np.isfinite(x):
            return "nan"
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def _write_report(res, ladder_df, fig_paths, target_year, out_path: Path) -> None:
    s, d, k, agg = res.sweep, res.diurnal, res.subkpis, res.aggregates
    L = []
    L.append(f"# Perfect-Foresight Shaping Diagnostic — Delivery Cal {target_year}\n")
    L.append(f"**Config** : `{res.config_name}`\n")
    L.append("Additive, non-gating. Isolates **shaping** quality from **forward-forecast** "
             "error by re-anchoring on realized ex-post settlements (perfect foresight) and "
             "de-levelling intra-period profiles. Methodology: Fleten-Lemming (2003), "
             "Benth et al. (2007), Kiesel et al. (2019), Lago et al. (2021); CH-physical "
             "decomposition after Bevilacqua et al. (2022).\n")

    L.append("## 1. Granularity ladder (best-trained vintage)\n")
    L.append("Monthly-shape correlation vs realized, as the anchor granularity coarsens from "
             "the full market quotes down to a single realized annual level. The anchor key "
             "sets are listed alongside so the comparison is transparent — note in particular "
             "that at late-vintage delivery the market may already quote at month granularity "
             "for the near horizon, so `market` is not always at coarser granularity than "
             "`pf_cal_quarter`.\n")
    L.append(ladder_df.pipe(_df_to_md) + "\n")

    L.append("## 2. Training-maturity sweep (seasonal-shape decomposition)\n")
    L.append("`pf_cal_corr` = monthly-shape correlation under **perfect annual-level "
             "foresight**. `market_corr` = with real traded quotes. `shaping_residual` = "
             "1 − pf_cal_corr. **Important caveat**: `fit_seasonal_ratios` only counts full "
             "calendar years, so many adjacent vintages share identical seasonal_ratios → the "
             f"sweep's effective sample size is **{agg.get('n_distinct_pf_corr_regimes', 'n/a')} "
             "distinct seasonal-ratio regimes**, not the row count. The median/p10/p90 "
             "aggregates below should be read with that collapse in mind.\n")
    L.append(s.pipe(_df_to_md) + "\n")
    pf_agg = agg.get("pf_cal_corr") or {}
    mkt_agg = agg.get("market_corr") or {}
    L.append(f"\n**Aggregates** — pf_cal_corr median **{_fmt(pf_agg.get('median'))}** "
             f"[min {_fmt(pf_agg.get('min'))}, max {_fmt(pf_agg.get('max'))}]; "
             f"market_corr median **{_fmt(mkt_agg.get('median'))}** "
             f"(distinct seasonal-ratio regimes: {agg.get('n_distinct_pf_corr_regimes', 'n/a')}).\n")

    L.append("## 3. De-levelled intra-day shape (cosine + demeaned RMSE)\n")
    L.append(f"Computed on the **delivery-year window only** (year == {target_year}), so the "
             "score is not contaminated by extrapolated hours outside the anchored window. "
             "Cosine = pattern fidelity (additive- AND multiplicative-invariant on the profile); "
             "demeaned RMSE = amplitude fidelity (additive-invariant only — responds to "
             "multiplicative rescale by design, since amplitude should).\n")
    L.append(d.pipe(_df_to_md) + "\n")

    L.append("## 4. Swiss-physical sub-KPIs (model vs realized)\n")
    L.append(f"Best-trained vintage `{k.get('vintage')}`.\n")
    rows = [
        ("winter/summer ratio", k["winter_summer_ratio"]["model"], k["winter_summer_ratio"]["realized"]),
        ("solar-bowl depth", k["solar_bowl_depth"]["model"], k["solar_bowl_depth"]["realized"]),
        ("peak/off-peak spread (EUR/MWh)", k["peak_offpeak_spread"]["model"], k["peak_offpeak_spread"]["realized"]),
    ]
    sub = pd.DataFrame(rows, columns=["sub-KPI", "model", "realized"])
    sub["model"] = sub["model"].round(3); sub["realized"] = sub["realized"].round(3)
    L.append(sub.pipe(_df_to_md) + "\n")

    L.append("## 5. Figures\n")
    for p in fig_paths:
        L.append(f"![{p.stem}](figures/{p.name})\n")

    out_path.write_text("\n".join(L))


def _ab_benchmark(epex, fwds, target_year: int, config_name: str, out_dir: Path) -> dict:
    """Run baseline + SOTA, compute paired-test gain, return summary dict.

    Saves the per-vintage A/B parquet and a matplotlib figure to `out_dir`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    from pfc_shaping.validation.perfect_foresight import run_perfect_foresight

    # CRITICAL: ensure the figure subdir exists even when the main run was
    # called with --no-figures (so the A/B figure can still be saved).
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[pf] A/B benchmark: baseline vs SOTA for Cal {target_year} ...")
    base = run_perfect_foresight(epex, fwds, target_year=target_year,
                                 config_name=config_name, estimator="baseline")
    sota = run_perfect_foresight(epex, fwds, target_year=target_year,
                                 config_name=config_name, estimator="sota")
    cmp = (base.sweep[["vintage", "train_years", "pf_cal_corr"]]
           .rename(columns={"pf_cal_corr": "baseline"})
           .merge(sota.sweep[["vintage", "pf_cal_corr"]]
                  .rename(columns={"pf_cal_corr": "sota"}), on="vintage"))
    cmp["gain"] = (cmp["sota"] - cmp["baseline"]).round(4)
    cmp.to_parquet(out_dir / "pf_ab_benchmark.parquet")

    # Wilcoxon signed-rank, one-sided (H1: SOTA > baseline). Guard against the
    # degenerate "all-zero differences" case (which would crash scipy).
    if (cmp["gain"] != 0).any():
        stat, p = wilcoxon(cmp["sota"].to_numpy(), cmp["baseline"].to_numpy(),
                           alternative="greater")
        stat, p = float(stat), float(p)
    else:
        stat, p = float("nan"), float("nan")

    # Figure: per-vintage comparison.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = cmp["train_years"].to_numpy()
    ax.plot(x, cmp["baseline"], "o-", label=f"baseline (LS + full-year)")
    ax.plot(x, cmp["sota"], "s-", label="SOTA (regime-aware + CH prior)")
    ax.axhline(0.85, ls="--", c="grey", lw=1, label="SC#1 gate 0.85")
    ax.set_xlabel("training history available at vintage (years)")
    ax.set_ylabel(f"monthly-shape correlation vs realized Cal {target_year}")
    ax.set_title(f"A/B benchmark — seasonal-ratio estimators (delivery {target_year})\n"
                 f"median gain = {cmp['gain'].median():+.3f}, "
                 f"Wilcoxon stat = {stat:.0f}, p = {p:.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "figures" / "pf_ab_benchmark.png"
    fig.savefig(fig_path, dpi=130); plt.close(fig)

    return {
        "cmp": cmp, "wilcoxon_stat": float(stat), "wilcoxon_p": float(p),
        "median_baseline": float(cmp["baseline"].median()),
        "median_sota": float(cmp["sota"].median()),
        "median_gain": float(cmp["gain"].median()),
        "n_better": int((cmp["gain"] > 0).sum()),
        "n_total": int(len(cmp)),
        "fig": fig_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--config", default="bowl_on_floors_off")
    parser.add_argument("--estimator", choices=("baseline", "sota"), default="baseline",
                        help="Seasonal-ratios estimator for the single-run mode "
                             "(ignored when --ab is set).")
    parser.add_argument("--ab", action="store_true",
                        help="Run A/B benchmark (baseline vs SOTA) with Wilcoxon "
                             "signed-rank test and side-by-side report.")
    parser.add_argument("--output-dir",
                        default=".planning/phases/10-pfc-fmv-quality-scorecard/perfect_foresight")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    from pfc_shaping.validation.perfect_foresight import run_perfect_foresight
    from pfc_shaping.validation.scorecard import ABLATION_GRID, list_vintages_2024_2025

    # Pre-validate inputs so we fail fast with friendly messages.
    valid_configs = {c.name for c in ABLATION_GRID}
    if args.config not in valid_configs:
        parser.error(f"--config={args.config!r} must be one of {sorted(valid_configs)}")
    # --ab runs both estimators internally, so a single-run --estimator value
    # is redundant. Force baseline for the main report to keep the header and
    # the A/B section self-consistent (the A/B section runs SOTA itself).
    if args.ab and args.estimator != "baseline":
        print(f"[pf] note: --ab overrides --estimator={args.estimator}; "
              f"main run uses 'baseline', A/B section runs both.")
        args.estimator = "baseline"
    candidate_vintages = [v for v in list_vintages_2024_2025() if v.year < args.target_year]
    if not candidate_vintages:
        parser.error(
            f"--target-year={args.target_year} has no vintages strictly before it in "
            f"list_vintages_2024_2025(); pick a year > {min(v.year for v in list_vintages_2024_2025())}."
        )

    epex = pd.read_parquet(_REPO_ROOT / EPEX_REL)["price_eur_mwh"]
    fwds = pd.read_parquet(_REPO_ROOT / FORWARDS_REL)

    print(f"[pf] running perfect-foresight diagnostic for Cal {args.target_year} "
          f"({args.estimator} estimator) ...")
    res = run_perfect_foresight(
        epex, fwds, target_year=args.target_year, config_name=args.config,
        estimator=args.estimator,
    )

    best = max(candidate_vintages,
               key=lambda v: (epex[epex.index < v].index.max() - epex[epex.index < v].index.min()).days)
    ladder = _granularity_ladder(epex, fwds, best, args.target_year, args.config)
    ladder_df = pd.DataFrame([
        {"anchor": k, "monthly_corr": round(v["corr"], 4), "n_keys": len(v["keys"]),
         "keys": ", ".join(v["keys"]) if v["keys"] else "—"}
        for k, v in ladder.items()
    ])

    out_dir = _REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: list[Path] = []
    if not args.no_figures:
        print("[pf] rendering figures ...")
        fig_paths = _make_figures(epex, fwds, res, args.target_year, args.config, out_dir / "figures")

    report = out_dir / "PERFECT-FORESIGHT-REPORT.md"
    _write_report(res, ladder_df, fig_paths, args.target_year, report)
    res.sweep.to_parquet(out_dir / "pf_sweep.parquet")
    res.diurnal.to_parquet(out_dir / "pf_diurnal.parquet")
    print(f"[pf] report  -> {report}")
    print(f"[pf] figures -> {len(fig_paths)} written to {out_dir / 'figures'}")
    print("\n=== granularity ladder ===")
    print(ladder_df.to_string(index=False))

    # A/B benchmark mode: run both estimators, paired-test, write report block.
    if args.ab:
        ab = _ab_benchmark(epex, fwds, args.target_year, args.config, out_dir)
        print("\n=== A/B benchmark (baseline vs SOTA seasonal_ratios) ===")
        print(ab["cmp"].to_string(index=False))
        print(f"\nmedian baseline: {ab['median_baseline']:.4f}   "
              f"median SOTA: {ab['median_sota']:.4f}   "
              f"median gain: {ab['median_gain']:+.4f}")
        print(f"vintages improved: {ab['n_better']}/{ab['n_total']}   "
              f"Wilcoxon stat = {ab['wilcoxon_stat']:.0f}, "
              f"p = {ab['wilcoxon_p']:.4f}")
        # Append A/B section to the markdown report.
        with open(report, "a") as fh:
            fh.write("\n## 6. SOTA A/B benchmark — seasonal-ratios estimator\n\n")
            fh.write("Paired comparison of baseline (`LS` over full calendar years, "
                     "in-tree `ContractCascader.fit_seasonal_ratios`) vs SOTA "
                     "(regime-aware weighted mean + Bayesian shrinkage to CH-physical "
                     "prior, `RegimeAwareSeasonalRatios`). Wilcoxon signed-rank "
                     "(one-sided, H1: SOTA > baseline).\n\n")
            fh.write(_df_to_md(ab["cmp"]) + "\n\n")
            fh.write(f"**Median**: baseline `{ab['median_baseline']:.4f}` → "
                     f"SOTA `{ab['median_sota']:.4f}` ({ab['median_gain']:+.4f}). "
                     f"**Vintages improved**: {ab['n_better']}/{ab['n_total']}. "
                     f"**Wilcoxon**: stat={ab['wilcoxon_stat']:.0f}, "
                     f"p={ab['wilcoxon_p']:.4f}.\n\n")
            fh.write(f"![A/B benchmark](figures/{ab['fig'].name})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
