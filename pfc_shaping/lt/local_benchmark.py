"""Local retrospective CPU benchmark mathematics, separate from scientific v6.

No synthetic containers, registry slots, source admissions or authority grants
are created here. The caller supplies already captured data and records their
latest-observed (not historical point-in-time) provenance. Frozen numerical
kernels and the existing assembly injection seam are reused without changing
their source or pretending this is the prospective scientific runner.
"""

from __future__ import annotations

from importlib.metadata import version
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.evaluation_challengers import _WeightedMLPRegressor, recency_weights
from pfc_shaping.lt.evaluation_engine import _monthly_center, _weighted_quantile
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.model.shape_hourly_mlp import _encode_features
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP

SCHEMA = "fmv-lt-local-retrospective-hourly-benchmark.v1"
REFERENCE = "seasonal-reference-local-v1"
CORRECTED = "hydro-aligned-unweighted-mlp-local-v1"
FROZEN = "frozen-unweighted-mlp-diagnostic"
CHALLENGERS = (
    "recency-weighted-mlp", "ridge-linear", "spline-ridge-gam",
    "lightgbm-deterministic-cpu",
)
FEATURES = (
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_holiday", "hydro_fill", "years_ahead",
)
AUTHORITIES = dict(production=False, promotion=False, scientific_admission=False,
                   trading=False, externally_registered=False, countable_origin=False)


def utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    index = index.tz_convert("UTC")
    if index.has_duplicates or not index.is_monotonic_increasing or index.hasnans:
        raise ValueError("timestamps must be unique, ordered and non-null")
    if np.any(index.as_unit("ns").asi8 % pd.Timedelta(minutes=15).value):
        raise ValueError("timestamps must be aligned to quarter-hours")
    return index


def complete_training_days(prices: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Keep complete delivered Swiss days strictly before the simulated origin."""
    index = utc_index(prices.index)
    if origin.tzinfo is None:
        raise ValueError("origin must be timezone-aware")
    if not np.isfinite(prices["price_eur_mwh"].to_numpy(dtype=float)).all():
        raise ValueError("source prices must be finite; gaps must remain explicit")
    frame = prices.loc[index < origin, ["price_eur_mwh"]].copy()
    local_day = frame.index.tz_convert("Europe/Zurich").normalize()
    counts = pd.Series(1, index=local_day).groupby(level=0).sum()
    next_day = counts.index + pd.DateOffset(days=1)
    expected = (next_day - counts.index).total_seconds() / 900
    accepted = counts.index[(counts.to_numpy() == expected) & (next_day <= origin)]
    result = frame.loc[local_day.isin(accepted)]
    if result.empty:
        raise ValueError("no complete training day before origin")
    return result


def training_matrix(prices: pd.DataFrame, hydro: pd.DataFrame, origin: pd.Timestamp):
    """Vectorized native target/feature aggregation, including autumn hour merge.

    Lexicographic local-hour ordering matches native MLP.fit. Observation times
    are returned separately so recency weighting can be sorted chronologically.
    Historical hydro fractions come from the corrected local native component.
    """
    index = utc_index(prices.index)
    if bool((index >= origin).any()) or bool((hydro.index >= origin).any()):
        raise ValueError("training prices and hydro must precede origin")
    if hydro.empty or not np.isfinite(hydro["fill_pct"]).all():
        raise ValueError("hydro history must be nonempty and finite")
    if not hydro["fill_pct"].between(0, 100).all():
        raise ValueError("hydro fill_pct must be in [0, 100]")
    cal = enrich_15min_index(index)
    local = index.tz_convert("Europe/Zurich")
    day = local.strftime("%Y-%m-%d").to_numpy()
    price = prices["price_eur_mwh"].to_numpy(dtype=float)
    if not np.isfinite(price).all():
        raise ValueError("training prices must be finite")
    means = pd.Series(price).groupby(day).transform("mean").to_numpy()
    valid = means > 5.0
    if not valid.any():
        raise ValueError("no native positive-day training target")
    mapper = HydroAlignedShapeHourlyMLP()
    mapper._setup_hydro(hydro[["fill_pct"]])
    fill = mapper._map_hydro_fill(index)
    features = _encode_features(
        local.hour.to_numpy(), local.month.to_numpy(), local.dayofweek.to_numpy(),
        cal["type_jour"].isin(["Ferie_CH", "Ferie_DE"]).to_numpy(),
        fill, np.zeros(len(index)),
    )[:, :9]
    keys = day[valid].astype(str) + "-" + local.hour.to_numpy()[valid].astype(str)
    _, inverse = np.unique(keys, return_inverse=True)
    weight = np.exp2(-(index.max() - index[valid]).total_seconds().to_numpy() / 86400 / 180)
    denom = np.bincount(inverse, weights=weight)
    x = np.column_stack([
        np.bincount(inverse, weights=features[valid, j] * weight) / denom
        for j in range(9)
    ])
    y = np.bincount(inverse, weights=np.clip(price[valid] / means[valid], .2, 3) * weight) / denom
    first = np.full(len(denom), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, inverse, index[valid].as_unit("ns").asi8)
    timestamps = pd.to_datetime(first, utc=True)
    return x, y, timestamps, mapper


def prediction_matrix(index: pd.DatetimeIndex, origin: pd.Timestamp,
                      mapper: HydroAlignedShapeHourlyMLP) -> np.ndarray:
    index = utc_index(index)
    if bool((index < origin).any()):
        raise ValueError("prediction cannot precede origin")
    local = index.tz_convert("Europe/Zurich")
    cal = enrich_15min_index(index)
    weeks = local.isocalendar().week.to_numpy(dtype=int)
    fill = np.array([mapper.get_climatological_fill(int(w)) for w in weeks])
    years = np.maximum((index - origin).total_seconds().to_numpy() / (365.25 * 86400), 0)
    return _encode_features(
        local.hour.to_numpy(), local.month.to_numpy(), local.dayofweek.to_numpy(),
        cal["type_jour"].isin(["Ferie_CH", "Ferie_DE"]).to_numpy(), fill, years,
    )[:, :9]


def postprocess(raw: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """Native floor/day-normalization/clip once; no origin identity fabrication."""
    utc_index(index)
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (len(index),) or not np.isfinite(raw).all():
        raise ValueError("one finite raw factor per timestamp required")
    floored = np.maximum(raw, .1)
    day = index.tz_convert("Europe/Zurich").strftime("%Y-%m-%d")
    means = pd.Series(floored).groupby(day).transform("mean").to_numpy()
    return np.clip(floored / means, .4, 2)


def parameter_grid(candidate: str) -> list[dict[str, str]]:
    if candidate not in CHALLENGERS:
        raise ValueError("unknown local challenger")
    spec = next(s for s in default_evaluation_protocol().candidates if s.candidate_id == candidate)
    names = [name for name, _ in spec.tuning_grid]
    return [dict(zip(names, values)) for values in product(*(v for _, v in spec.tuning_grid))]


def fit_challenger(candidate: str, x: np.ndarray, y: np.ndarray,
                   timestamps: pd.DatetimeIndex, origin: pd.Timestamp, parameters: dict):
    """Execute actual local arrays; synthetic APIs and their manifests stay unused."""
    if parameters not in parameter_grid(candidate):
        raise ValueError("parameters outside the declared grid")
    if x.ndim != 2 or x.shape != (len(y), 9) or len(timestamps) != len(y):
        raise ValueError("common nine-column training matrix required")
    if len(y) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training matrix must be finite and nonempty")
    order = np.argsort(timestamps.as_unit("ns").asi8, kind="stable")
    ordered = utc_index(timestamps[order])
    if bool((ordered >= origin).any()):
        raise ValueError("challenger training delivery must precede origin")
    if candidate == "recency-weighted-mlp":
        model = _WeightedMLPRegressor()
        # Exact existing numerical algorithm; failure to converge remains failure.
        model.fit(x[order], y[order], sample_weight=recency_weights(ordered, origin))
    elif candidate == "ridge-linear":
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(parameters["alpha"])))
        model.fit(x, y)
    elif candidate == "spline-ridge-gam":
        model = make_pipeline(SplineTransformer(degree=3, n_knots=int(parameters["n_knots"]),
                                               include_bias=False), StandardScaler(),
                              Ridge(alpha=float(parameters["alpha"])))
        model.fit(x, y)
    else:
        if version("lightgbm") != "4.6.0":
            raise RuntimeError("local LightGBM requires exactly version 4.6.0")
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            objective="regression_l1", device_type="cpu", deterministic=True,
            force_col_wise=True, n_jobs=1, random_state=42, bagging_seed=42,
            data_random_seed=42, feature_fraction_seed=42, verbosity=-1,
            learning_rate=float(parameters["learning_rate"]),
            min_data_in_leaf=int(parameters["min_data_in_leaf"]),
            n_estimators=int(parameters["n_estimators"]), num_leaves=int(parameters["num_leaves"]),
        )
        model.fit(pd.DataFrame(x, columns=FEATURES), y)
    return model


def predict_challenger(candidate: str, model, x: np.ndarray) -> np.ndarray:
    data = pd.DataFrame(x, columns=FEATURES) if candidate == CHALLENGERS[-1] else x
    values = np.asarray(model.predict(data), dtype=float)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise ValueError("challenger returned invalid prediction")
    return values


def seasonal_reference(target: np.ndarray, training_index: pd.DatetimeIndex,
                       delivery_index: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
    """Untuned arithmetic mean native f_H by season/day-type/hour.

    A missing cell falls back to season/hour, then hour, then global mean.
    The fallback order is frozen locally and its actual counts are returned.
    Common f_W and EEX product constraints are applied later by PFCAssembler.
    """
    train = enrich_15min_index(training_index).copy()
    train["target"] = target
    future = enrich_15min_index(delivery_index)
    raw = np.full(len(future), np.nan)
    counts = {}
    for keys in [("saison", "type_jour", "heure_hce"), ("saison", "heure_hce"), ("heure_hce",)]:
        table = train.groupby(list(keys))["target"].mean()
        query = (pd.MultiIndex.from_frame(future[list(keys)]) if len(keys) > 1
                 else pd.Index(future[keys[0]]))
        values = table.reindex(query).to_numpy()
        use = np.isnan(raw) & np.isfinite(values)
        raw[use] = values[use]
        counts["/".join(keys)] = int(use.sum())
    counts["global"] = int(np.isnan(raw).sum())
    raw[np.isnan(raw)] = float(np.mean(target))
    return postprocess(raw, delivery_index), counts


def score_curves(predictions: pd.DataFrame, truth: pd.Series, origin: pd.Timestamp):
    """Score full prices on common complete local months, at native hourly grain.

    Replicated CH quarter-hours are transport, not four independent observations.
    Center each complete Swiss month first; segment the same errors afterwards.
    Bias after monthly centering is an algebraic check, not a predictive success.
    """
    index = utc_index(predictions.index)
    if len(index) < 2 or not np.all(np.diff(index.as_unit("ns").asi8) == 900_000_000_000):
        raise ValueError("predictions require complete quarter-hour delivery grid")
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("failed candidates cannot silently change the scoring mask")
    hourly = predictions.resample("h").mean()
    actual_qh = truth.reindex(index)
    actual = actual_qh.resample("h").mean().where(actual_qh.resample("h").count() == 4)
    periods = hourly.index.tz_convert("Europe/Zurich").tz_localize(None).to_period("M")
    accepted, coverage = [], []
    for month in periods.unique():
        mask = periods == month
        start = month.start_time.tz_localize("Europe/Zurich")
        end = (month + 1).start_time.tz_localize("Europe/Zurich")
        expected = int((end - start).total_seconds() / 3600)
        observed = int(actual.iloc[np.flatnonzero(mask)].notna().sum())
        ok = int(mask.sum()) == expected and observed == expected
        coverage.append(dict(month=str(month), expected_hours=expected,
                             observed_hours=observed, eligible=bool(ok)))
        if ok:
            accepted.append(month)
    eligible = periods.isin(accepted)
    if not eligible.any():
        raise ValueError("no common complete truth month")
    hourly = hourly.loc[eligible]
    actual = actual.loc[eligible]
    index = hourly.index
    local = index.tz_convert("Europe/Zurich")
    cal = enrich_15min_index(index)
    weights = np.ones(len(index))  # One observed hour = one MWh at constant 1 MW.
    centered_truth = _monthly_center(actual.to_numpy(), index, weights)
    errors = pd.DataFrame({name: _monthly_center(hourly[name].to_numpy(), index, weights)
                           - centered_truth for name in hourly}, index=index)
    local_origin = origin.tz_convert("Europe/Zurich")
    leads = local.year * 12 + local.month - (local_origin.year * 12 + local_origin.month)
    peak = (local.dayofweek < 5) & (local.hour >= 8) & (local.hour < 20)
    masks = {"ALL": np.ones(len(index), dtype=bool), "PEAK_CLOCK": peak,
             "OFFPEAK_CLOCK": ~peak, "NEGATIVE_TRUTH": actual.to_numpy() < 0}
    for lo, hi in [(1, 6), (7, 12), (13, 24), (25, 36)]:
        masks[f"M{lo:02d}_M{hi:02d}"] = (leads >= lo) & (leads <= hi)
    for season in ["Hiver", "Printemps", "Ete", "Automne"]:
        masks[season] = cal["saison"].to_numpy() == season
    masks["WEEKEND_OR_HOLIDAY"] = cal["type_jour"].to_numpy() != "Ouvrable"
    rows = []
    for segment, mask in masks.items():
        for candidate in predictions:
            error = errors[candidate].to_numpy()[mask]
            row = dict(candidate=candidate, segment=segment, hours=len(error),
                       status="COMPUTED_LOCAL_RETROSPECTIVE" if len(error) else "UNSUPPORTED_NO_ROWS",
                       mae=None, rmse=None, bias=None, p95=None)
            if len(error):
                row.update(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                           bias=float(error.mean()),
                           p95=_weighted_quantile(np.abs(error), np.ones(len(error)), .95))
            rows.append(row)
    return pd.DataFrame(rows), errors, coverage
