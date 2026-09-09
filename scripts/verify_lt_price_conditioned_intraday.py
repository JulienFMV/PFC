"""Independently check saved D306 arithmetic, populations, factors and exports."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference
from pfc_shaping.lt.signed_intraday import intraday_cell_reference
from pfc_shaping.validation.product_normalization import build_product_normalization_gates
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, sha, write_json

PREVIOUS = ROOT / "build/lt-signed-composition-20260907/run-v4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    task = ROOT / "build/lt-price-conditioned-20260907"
    if Path.cwd() != ROOT or not run.is_relative_to(task) or not out.is_relative_to(task) or out == task:
        raise ValueError("canonical task paths required")
    out.mkdir(exist_ok=False)
    plan = json.loads((run / "plan.json").read_text())
    complete = json.loads((run / "complete.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert sha(run / name) == digest, name
    for name, digest in {**plan["inputs_sha256"], **plan["code_sha256"]}.items():
        assert sha(ROOT / name) == digest, name
    assert not any(plan["authority"].values()) and not any(complete["authority"].values())
    metrics = pd.read_csv(run / "metrics.csv")
    de = pd.read_parquet(SOURCE / "prepared-inputs/epex-de.parquet").price_eur_mwh
    errors, replay = [], []
    for i, origin_month in enumerate(plan["months"]):
        origin = pd.Timestamp(origin_month + "-01", tz="Europe/Berlin")
        training = de.loc[de.index < origin]
        for j, delivery in enumerate(plan["months"][i:], start=i):
            frame = pd.read_parquet(run / f"de-{origin_month}-{delivery}/predictions.parquet")
            start = pd.Timestamp(delivery + "-01", tz="Europe/Berlin")
            end = (pd.Period(delivery, freq="M") + 1).start_time.tz_localize("Europe/Berlin")
            expected = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
            assert frame.index.equals(expected)
            truth = de.reindex(expected)
            parent = truth.to_numpy().reshape(-1, 4).mean(axis=1).repeat(4)
            np.testing.assert_array_equal(truth, frame.actual)
            np.testing.assert_allclose(parent, frame.parent, atol=1e-12, rtol=0)
            predicted, _ = calendar_cell_reference(training.resample("h").mean(), expected[::4])
            np.testing.assert_array_equal(predicted.repeat(4), frame.forecast)
            additive, _ = intraday_cell_reference(training, expected, country="DE")
            np.testing.assert_array_equal(additive, frame.additive)
            calendar = enrich_15min_index(expected, country="DE")
            for lane in ("conditional", "forecast"):
                levels = frame.parent.to_numpy() if lane == "conditional" else frame.forecast.to_numpy()
                for candidate in plan["candidates"]:
                    if candidate == "flat": delta = np.zeros(len(frame))
                    elif candidate == "signed-additive": delta = additive.to_numpy()
                    else:
                        model = ShapeIntraday.load(PREVIOUS / f"de-{origin_month}/{candidate}.parquet")
                        factor = model.apply(expected, calendar, reference_date=origin).to_numpy()
                        raw = (levels * (factor - 1)).reshape(-1, 4)
                        delta = (raw - raw.mean(axis=1)[:, None]).ravel()
                    saved = frame[f"{lane}/{candidate}"].to_numpy()
                    difference = float(np.max(np.abs(saved - levels - delta)))
                    replay.append(difference)
                    assert difference < 1e-9
                    np.testing.assert_allclose(saved.reshape(-1, 4).mean(axis=1), levels[::4], atol=1e-9, rtol=0)
                    # Regime boundaries use the saved common pandas parent after
                    # independent numerical verification above. Different sum
                    # orders can straddle exact zero by machine epsilon.
                    regime_parent = frame.parent.to_numpy()
                    masks = {"ALL": np.ones(len(frame), bool), "NEGATIVE_PARENT": regime_parent < 0,
                        "NEAR_ZERO_PARENT": np.abs(regime_parent) <= 5, "POSITIVE_PARENT": regime_parent > 5,
                        "NEGATIVE_FORECAST": frame.forecast.to_numpy() < 0,
                        "NEAR_ZERO_FORECAST": np.abs(frame.forecast.to_numpy()) <= 5}
                    for season in ("Hiver", "Printemps", "Ete", "Automne"):
                        masks["SEASON_" + season] = calendar.saison.eq(season).to_numpy()
                    for segment, mask in masks.items():
                        e = (saved - truth.to_numpy())[mask]
                        row = metrics.loc[(metrics.origin == origin_month) & (metrics.delivery == delivery)
                            & (metrics.lane == lane) & (metrics.candidate == candidate) & (metrics.segment == segment)]
                        assert len(row) == 1 and int(row.iloc[0].rows) == len(e), (origin_month, delivery, lane, candidate, segment, len(e), row.to_dict())
                        if len(e):
                            np.testing.assert_allclose(row.iloc[0][["mae", "rmse", "bias"]].to_numpy(dtype=float),
                                [np.abs(e).mean(), np.sqrt(np.square(e).mean()), e.mean()], atol=1e-10, rtol=0)
                        else:
                            assert row.iloc[0][["mae", "rmse", "bias"]].isna().all()
                        errors.append(dict(origin=origin_month, delivery=delivery, lead=j-i,
                            horizon="0" if j==i else "1-2" if j-i<=2 else "3-5" if j-i<=5 else "6-7",
                            lane=lane, candidate=candidate, segment=segment, rows=len(e),
                            abs_sum=float(np.abs(e).sum()), sq_sum=float(np.square(e).sum()),
                            mae=float(np.abs(e).mean()) if len(e) else np.nan,
                            rmse=float(np.sqrt(np.square(e).mean())) if len(e) else np.nan))
    recomputed = pd.DataFrame(errors)
    recomputed.to_csv(out / "recomputed-fold-metrics.csv", index=False)
    summaries = []
    for horizon, sample in [("ALL", recomputed)] + list(recomputed.groupby("horizon")):
        for keys, group in sample.groupby(["lane", "candidate", "segment"]):
            n = int(group.rows.sum())
            summaries.append(dict(horizon=horizon, lane=keys[0], candidate=keys[1], segment=keys[2], rows=n,
                supported_folds=int(group.rows.gt(0).sum()), equal_fold_mae=group.mae.mean(),
                equal_fold_rmse=group.rmse.mean(), pooled_mae=group.abs_sum.sum()/n if n else np.nan,
                pooled_rmse=np.sqrt(group.sq_sum.sum()/n) if n else np.nan))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "comparison.csv", index=False)
    lead_zero = recomputed.loc[recomputed.lead.eq(0)].groupby(["lane", "candidate", "segment"], as_index=False).agg(
        equal_month_mae=("mae", "mean"), equal_month_rmse=("rmse", "mean"), rows=("rows", "sum"))
    lead_zero.to_csv(out / "d305-same-population.csv", index=False)
    gates = []
    for row in summary.itertuples():
        if row.candidate == "flat" or row.segment in ("NEGATIVE_FORECAST", "NEAR_ZERO_FORECAST"):
            continue
        for control in (["flat", "native"] if row.candidate == "native-regularized" else ["flat"]):
            ref = summary.loc[(summary.horizon == row.horizon) & (summary.lane == row.lane)
                & (summary.segment == row.segment) & (summary.candidate == control)].iloc[0]
            mae_ratio = row.pooled_mae / ref.pooled_mae if ref.pooled_mae > 0 else np.nan
            rmse_ratio = row.pooled_rmse / ref.pooled_rmse if ref.pooled_rmse > 0 else np.nan
            status = "UNSUPPORTED" if row.rows < 96 else "REGRESSION" if max(mae_ratio, rmse_ratio) > 1.05 else "PASS_LOCAL_SCREEN"
            gates.append(dict(lane=row.lane, candidate=row.candidate, control=control,
                horizon=row.horizon, segment=row.segment, rows=row.rows,
                mae_ratio=mae_ratio, rmse_ratio=rmse_ratio, status=status))
    gates = pd.DataFrame(gates)
    gates.to_csv(out / "regime-gates.csv", index=False)
    # Every fold remains visible; no favorable pooling can hide these regressions.
    pivot = recomputed.pivot(index=["origin", "delivery", "lane", "segment"], columns="candidate", values="mae")
    for candidate in ("native", "native-regularized", "signed-additive"):
        pivot[candidate + "_versus_flat_pct"] = 100 * (pivot[candidate] / pivot.flat - 1)
    pivot.to_csv(out / "fold-comparison.csv")

    annual, hourly_drifts, monthly_errors, ch_scores = [], [], [], []
    ch_truth = pd.read_parquet(SOURCE / "prepared-inputs/epex-ch.parquet").price_eur_mwh
    assert (ch_truth.resample("h").max() - ch_truth.resample("h").min()).eq(0).all()
    for receipt in complete["receipts"]:
        label, candidate = receipt["origin"], receipt["candidate"]
        frame = pd.read_parquet(run / label / candidate / "curve.parquet")
        baseline = pd.read_parquet(PREVIOUS / ("current/signed/curve.parquet" if label == "current" else f"{label}/signed/curve.parquet"))
        assert frame.index.equals(baseline.index)
        grid = frame.index
        fit = next(f for f in complete["fits"] if f["origin"] == label)
        origin = pd.Timestamp(fit["cutoff"])
        if fit["rows"]:
            assert pd.Timestamp(fit["last_training"]) < origin
            model = ShapeIntraday.load(run / label / f"{candidate}.parquet")
        else:
            assert de.loc[de.index < origin].empty
            model = ShapeIntraday()
        f = model.apply(grid, enrich_15min_index(grid), reference_date=origin).to_numpy()
        level = baseline.price_shape.to_numpy().reshape(-1, 4).mean(axis=1).repeat(4)
        delta = (level * (f-1)).reshape(-1, 4)
        delta = (delta - delta.mean(axis=1)[:, None]).ravel()
        np.testing.assert_allclose(frame.price_shape-baseline.price_shape, delta, atol=1e-9, rtol=0)
        drift = float(np.max(np.abs(frame.price_shape.to_numpy().reshape(-1, 4).mean(axis=1)-level[::4])))
        hourly_drifts.append(drift)
        assert drift < 1e-9
        monthly = frame.price_shape.groupby(grid.tz_convert("Europe/Zurich").strftime("%Y-%m")).mean()
        solver = json.loads((SOURCE / "monthly-solver/result.json" if label == "current" else OLD / f"{label}/solver.json").read_text())
        levels = solver["assembler_base_prices"] if label == "current" else solver["base_prices"]
        difference = float((monthly-pd.Series(levels).reindex(monthly.index)).abs().max())
        assert difference < 1e-9
        monthly_errors.append(difference)
        gates_saved = pd.read_csv(run / label / candidate / "product-gates.csv")
        assert not gates_saved.status.eq("CRITICAL").any()
        # Source surface is saved in the historical inputs; current selection is
        # the same frozen retained latest surface, independently repriced here.
        if label == "current":
            from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
            surface = select_latest_quote_surface(pd.read_parquet(SOURCE / "eex-replay/eex-normalized-history.parquet"))
        else:
            surface = pd.read_parquet(OLD / f"{label}/eex-surface.parquet")
        hourly = frame.price_shape.resample("h").mean().to_frame("price_eur_mwh")
        hourly["ts_ch"] = hourly.index.tz_convert("Europe/Zurich")
        for field in ("year", "month", "quarter"):
            hourly[field] = getattr(hourly.ts_ch.dt, field)
        market = build_product_normalization_gates(hourly, surface, forward_date=pd.Timestamp(surface.date.iloc[0]),
            price_column="price_eur_mwh", hard_tolerance=1e-6, peak_country="CH")
        for column in market.select_dtypes(include="object"):
            market[column] = market[column].fillna("")
            gates_saved[column] = gates_saved[column].fillna("")
        pd.testing.assert_frame_equal(market.reset_index(drop=True), gates_saved, check_dtype=False, atol=1e-9, rtol=1e-10)
        if label != "current":
            actual = ch_truth.resample("h").mean().reindex(hourly.index)
            valid = actual.notna()
            a, b = hourly.price_eur_mwh.loc[valid], baseline.price_shape.resample("h").mean().loc[valid]
            groups = a.index.tz_convert("Europe/Zurich").strftime("%Y-%m")
            truth = actual.loc[valid]
            truth = truth - truth.groupby(groups).transform("mean")
            ea = a-a.groupby(groups).transform("mean")-truth
            eb = b-b.groupby(groups).transform("mean")-truth
            np.testing.assert_allclose(ea, eb, atol=1e-9, rtol=0)
            ch_scores.append(dict(origin=label, candidate=candidate, hours=len(ea),
                mae=ea.abs().mean(), signed_mae=eb.abs().mean(), rmse=np.sqrt(np.mean(ea**2))))
        else:
            export = pd.read_csv(run / label / candidate / "pfc-fmv-ch-15min.csv", sep=";")
            main = frame.loc[grid < pd.Timestamp("2030-01-01", tz="Europe/Zurich")]
            assert pd.DatetimeIndex(pd.to_datetime(export.timestamp_utc, utc=True)).equals(main.index)
            assert export.timestamp_ch.tolist() == main.index.tz_convert("Europe/Zurich").map(lambda t: t.isoformat()).tolist()
            np.testing.assert_allclose(export.price_eur_mwh, main.price_shape, atol=5.1e-11, rtol=0)
            for year in sorted(set(grid.tz_convert("Europe/Zurich").year)):
                mask = grid.tz_convert("Europe/Zurich").year == year
                for segment, selected in (("ALL", mask), ("NEGATIVE_PARENT", mask & (level<0)),
                                          ("NEAR_ZERO_PARENT", mask & (np.abs(level)<=5))):
                    p = frame.price_shape.to_numpy()[selected]
                    annual.append(dict(candidate=candidate, year=year, segment=segment, rows=len(p),
                        negative_quarters=int((p<0).sum()), minimum=float(p.min()) if len(p) else np.nan,
                        maximum=float(p.max()) if len(p) else np.nan,
                        mean_abs_intraday=float(np.abs(delta[selected]).mean()) if len(p) else np.nan))
    pd.DataFrame(ch_scores).to_csv(out / "ch-hourly-invariance.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "current-annual-regimes.csv", index=False)
    verification = dict(status="VERIFIED", inputs=len(plan["inputs_sha256"]), source_files=len(plan["code_sha256"]),
        outputs=len(manifest), recomputed_metric_rows=len(recomputed), component_replays=len(replay),
        max_replay_error=max(replay), max_hourly_drift=max(hourly_drifts), max_monthly_error=max(monthly_errors),
        verifier_sha256=sha(Path(__file__)), authority=plan["authority"])
    write_json(out / "verification.json", verification)
    report = ["# D306 — décomposition conditionnée par le prix horaire", "",
        "Benchmark CPU local vérifié. Références D304/D305 conservées ; aucune adoption ni autorité accordée.",
        "Le composant existant est réutilisé sans nouvel estimateur. Les facteurs sont multipliés par le prix horaire signé et recentrés dans chaque heure.",
        "", "## Même population que D305 : huit mois DE, horizon zéro", "",
        "| Voie | Candidat | MAE | RMSE |", "|---|---|---:|---:|"]
    for row in lead_zero.loc[lead_zero.segment.eq("ALL")].itertuples():
        report.append(f"| {row.lane} | {row.candidate} | {row.equal_month_mae:.6f} | {row.equal_month_rmse:.6f} |")
    report += ["", "## Contrôles de régime sur les 36 couples origine/mois", "",
        "| Voie | Candidat | Régime | MAE pondérée | RMSE pondérée | QH origine |", "|---|---|---|---:|---:|---:|"]
    selected = summary.loc[summary.horizon.eq("ALL") & summary.segment.isin(["ALL", "NEGATIVE_PARENT", "NEAR_ZERO_PARENT"])]
    for row in selected.itertuples():
        report.append(f"| {row.lane} | {row.candidate} | {row.segment} | {row.pooled_mae:.6f} | {row.pooled_rmse:.6f} | {row.rows} |")
    report += ["", "Gates préfixés : régression >5% MAE ou RMSE sur au moins96 QH versus flat ; contrôle additionnel versus native pour regularized. Les échantillons par origine se recouvrent.", ""]
    for (lane, candidate), group in gates.groupby(["lane", "candidate"]):
        failures = group.loc[group.status.eq("REGRESSION")]
        report.append(f"- {lane}/{candidate} : {len(failures)} contrôles en régression ; aucune qualification globale si non nul.")
    report += ["", "Les résultats détaillés (saisons, horizons0/1–2/3–5/6–7 mois, populations, biais, régressions par fold) sont dans comparison.csv, recomputed-fold-metrics.csv, fold-comparison.csv et regime-gates.csv.",
        "L'automne DE et les horizons au-delà de7 mois ne sont pas observés dans ce test. Les segments vides restent UNSUPPORTED.",
        "", "## Portée et limites", "",
        "La voie conditional utilise le vrai prix horaire : ce n'est pas une prévision. La voie forecast utilise uniquement une moyenne calendaire DE pré-origine (calendrier du composant horaire existant, CH) ; c'est une prévision effective transparente, pas une PFC DE calibrée EEX. Ni niveaux mensuels réalisés ni prix futurs ne sont injectés dans cette voie.",
        "À prix horaire nul, le composant multiplicatif reste plat ; il ne sait pas apprendre une dispersion non nulle autour de zéro. La conditionnalité testée est une mise à l'échelle, pas un apprentissage spécifique des régimes de prix.",
        "Les observations CH disponibles restent horaires répétées. Leur score horaire est conservé ; aucune précision CH à15min n'est démontrée. Les origines sans DE historique sont explicitement plates et non qualifiées. Aucun ajout hydro : D305 n'a pas démontré son intérêt.",
        "Les années2026–2032 des exports sont descriptives, sans preuve de précision à ces horizons ni scénarios structurels2030. Pas de bandes d'incertitude validées. Historique révisé, populations exposées et origines dépendantes : aucune preuve scientifique de promotion.",
        "", "## PFC candidates et conservation", "",
        "Deux exports candidats sous ../run-v1/current/{native,native-regularized}/pfc-fmv-ch-15min.csv (Oct2026–Dec2029). Chaque curve.parquet conserve le plein horizon jusqu'à2032. Les références MLP/signed/additive D305 restent dans leurs dossiers antérieurs intacts.",
        f"Erreur maximale de moyenne horaire : {max(hourly_drifts):.6g} EUR/MWh ; niveau mensuel : {max(monthly_errors):.6g}. Projection EEX recalculée et CSV/Parquet vérifiés.",
        "Les neuf QUOTE_CONFLICT existants restent visibles sur les exports courants ; aucune correction manuelle de mois ni CRITICAL introduit.",
        "", "## Vérification indépendante", "", json.dumps(verification, indent=2), ""]
    (out / "RAPPORT-COMPARATIF.md").write_text("\n".join(report), encoding="utf-8")
    write_json(out / "manifest.json", {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()})
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
