"""Rolling-origin comparison of dense-ratio and sparse price-space QH shaping.

This is a local, non-promotional diagnostic.  It binds the exact calibration
panel and its provenance manifest by caller-held SHA-256 values and emits the
fold table before a summary manifest written last.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.lt.model.shape_intraday import (
    MIN_OBS_COUCHE1,
    MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR,
    SAISONS,
)
from pfc_shaping.path_safety import read_stable_single_link_file

SCHEMA_VERSION = "intraday_shape_estimator_rolling_origin.v2"
_MAX_INPUT_BYTES = 512 * 1024 * 1024
_FALLBACK_DAY_TYPE = {
    "Ferie_DE": "Ferie_CH",
    "Ferie_CH": "Dimanche",
    "Dimanche": "Samedi",
    "Samedi": "Ouvrable",
}


def run_backtest(
    *,
    panel_path: Path,
    expected_panel_sha256: str,
    panel_manifest_path: Path,
    expected_panel_manifest_sha256: str,
    output_dir: Path,
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp,
    fold_days: int = 14,
    minimum_train_days: int = 30,
) -> dict[str, object]:
    panel_payload = _read_bound(panel_path, expected_panel_sha256, label="panel")
    manifest_payload = _read_bound(
        panel_manifest_path,
        expected_panel_manifest_sha256,
        label="panel manifest",
    )
    manifest = _strict_mapping(manifest_payload, label="panel manifest")
    if manifest.get("schema_version") != "local_intraday_calibration_panel.v3":
        raise ValueError("panel manifest schema is unsupported")
    if (
        manifest.get("status") != "COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO"
        or manifest.get("production_authorization") is not False
        or manifest.get("promotion_gate") is not False
        or manifest.get("scientific_admission") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("model_selection_eligible") is not False
        or manifest.get("confirmatory_rolling_origin_eligible") is not False
        or manifest.get("future_holdout_eligible") is not False
        or manifest.get("t057_eligible") is not False
        or manifest.get("candidate_assembly_eligible") is not False
        or manifest.get("local_diagnostic_allowed") is not True
    ):
        raise ValueError("rolling-origin input is not an explicit local diagnostic NO_GO")
    resolution = manifest.get("resolution_provenance")
    if (
        not isinstance(resolution, dict)
        or resolution.get("native_quarter_hour_truth_eligible") is not False
    ):
        raise ValueError(
            "local mixed-authority panel must explicitly deny native quarter-hour truth"
        )
    manifest_panel = _mapping(manifest.get("panel"), label="manifest panel")
    if manifest_panel.get("sha256") != expected_panel_sha256:
        raise ValueError("panel hash is not bound by the provenance manifest")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(io.BytesIO(panel_payload))
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("panel must have a DatetimeIndex")
    panel.index = pd.to_datetime(panel.index, utc=True)
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be unique and monotonic")
    if "price_eur_mwh" not in panel.columns:
        raise ValueError("panel is missing price_eur_mwh")
    calendar = enrich_15min_index(panel.index, country="DE")
    data = panel[["price_eur_mwh"]].join(
        calendar[["saison", "type_jour", "heure_hce", "quart"]]
    )
    counts = data.groupby(data.index.floor("h")).size()
    if not bool((counts == 4).all()):
        raise ValueError("panel contains incomplete UTC parent hours")

    starts = pd.date_range(origin_start, origin_end, freq=f"{fold_days}D", tz="UTC")
    fold_rows: list[dict[str, object]] = []
    for origin in starts:
        train = data.loc[data.index < origin]
        test_end = min(origin + pd.Timedelta(days=fold_days), data.index.max() + pd.Timedelta(minutes=15))
        test = data.loc[(data.index >= origin) & (data.index < test_end)]
        if len(test) == 0:
            continue
        if train.index.max() - train.index.min() < pd.Timedelta(days=minimum_train_days):
            continue
        profiles, parent_hours = _fit_profiles(train)
        candidate, dense_support_envelope = _candidate_profiles(
            profiles=profiles,
            parent_hours=parent_hours,
        )
        models = {
            "incumbent_ratio": profiles["ratio"],
            "candidate_sparse_price_space_24_dense_envelope": candidate,
        }
        for model_name, fitted in models.items():
            fold_rows.append(
                _score_fold(
                    model_name=model_name,
                    origin=origin,
                    test_end=test_end,
                    test=test,
                    profiles=fitted,
                    parent_hours=parent_hours,
                    dense_support_envelope=dense_support_envelope,
                )
            )
    if not fold_rows:
        raise ValueError("rolling-origin schedule produced no folds")

    folds = pd.DataFrame(fold_rows).sort_values(["origin_utc", "model"], kind="stable")
    summary_metrics = {
        model: _aggregate(group)
        for model, group in folds.groupby("model", sort=True)
    }
    baseline = summary_metrics["incumbent_ratio"]
    candidate_name = "candidate_sparse_price_space_24_dense_envelope"
    candidate = summary_metrics[candidate_name]
    paired = folds.pivot(index="origin_utc", columns="model", values="mae_eur_mwh")
    candidate_wins = int(
        (
            paired[candidate_name]
            < paired["incumbent_ratio"]
        ).sum()
    )
    tolerance = 1e-12
    candidate_losses = int(
        (
            paired[candidate_name]
            > paired["incumbent_ratio"] + tolerance
        ).sum()
    )
    candidate_ties = int(len(paired) - candidate_wins - candidate_losses)
    candidate_rows = folds[folds["model"] == candidate_name].sort_values("origin_utc")
    evaluable = candidate_rows[candidate_rows["n_sparse_quarter_hours"] > 0]
    latest_evaluable_origin = str(evaluable["origin_utc"].iloc[-1]) if len(evaluable) else None
    latest_evaluable_noninferior = False
    if latest_evaluable_origin is not None:
        latest_evaluable_noninferior = bool(
            paired.loc[latest_evaluable_origin, candidate_name]
            <= paired.loc[latest_evaluable_origin, "incumbent_ratio"] + tolerance
        )
    gates = {
        "aggregate_mae_improves": candidate["mae_eur_mwh"] < baseline["mae_eur_mwh"],
        "aggregate_rmse_improves": candidate["rmse_eur_mwh"] < baseline["rmse_eur_mwh"],
        "sparse_mae_improves": candidate["sparse_mae_eur_mwh"]
        < baseline["sparse_mae_eur_mwh"],
        "learned_surface_factor_tail_reduces": candidate[
            "surface_max_factor_abs_deviation"
        ]
        < baseline["surface_max_factor_abs_deviation"],
        "sparse_effect_is_evaluable": len(evaluable) >= 8,
        "no_mae_fold_regression": candidate_losses == 0,
        "latest_evaluable_fold_noninferior": latest_evaluable_noninferior,
    }

    csv_path = output_dir / "fold_metrics.csv"
    csv_payload = folds.to_csv(index=False, lineterminator="\n").encode("utf-8")
    write_durable_exact_bytes(csv_path, csv_payload, label="intraday rolling-origin folds")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "LOCAL_DIAGNOSTIC_CANDIDATE_PASS_NOT_PRODUCTION"
            if all(gates.values())
            else "LOCAL_DIAGNOSTIC_CANDIDATE_REJECTED_NOT_PRODUCTION"
        ),
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "confirmatory_rolling_origin_eligible": False,
        "future_holdout_eligible": False,
        "t057_eligible": False,
        "candidate_assembly_eligible": False,
        "local_diagnostic_allowed": True,
        "ompex_used": False,
        "input": {
            "panel_sha256": expected_panel_sha256,
            "panel_manifest_sha256": expected_panel_manifest_sha256,
            "panel_rows": len(data),
            "panel_start_utc": data.index.min().isoformat(),
            "panel_end_utc": data.index.max().isoformat(),
            "authority_status": manifest.get("status"),
            "limitations": manifest.get("limitations", []),
        },
        "design": {
            "origin_start_utc": origin_start.isoformat(),
            "origin_end_utc": origin_end.isoformat(),
            "fold_days": fold_days,
            "minimum_train_days": minimum_train_days,
            "minimum_parent_hours": MIN_OBS_COUCHE1,
            "sparse_price_space_max_parent_hours": MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR,
            "sparse_amplitude_policy": "contract_to_max_dense_cell_abs_deviation",
            "target": "quarter_hour_price_given_realized_parent_hour_mean",
        },
        "fold_count": int(len(paired)),
        "candidate_mae_fold_wins": candidate_wins,
        "candidate_mae_fold_losses": candidate_losses,
        "candidate_mae_fold_ties": candidate_ties,
        "folds_with_direct_test_rows": int(
            (candidate_rows["n_direct_quarter_hours"] > 0).sum()
        ),
        "folds_with_sparse_test_rows": int(len(evaluable)),
        "latest_evaluable_origin_utc": latest_evaluable_origin,
        "metrics": summary_metrics,
        "gates": gates,
        "artifacts": {
            "fold_metrics_csv": str(csv_path),
            "fold_metrics_csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
        },
        "limitations": [
            "The panel is local mixed-authority evidence and is not production-admissible.",
            "DE-LU quarter-hour shaping transfer to CH is not validated by this backtest.",
            "Fold regressions and the latest evaluable fold are blocking gates; aggregate improvements remain regime-sensitive.",
            "This isolates quarter-hour shaping using the realized parent-hour mean.",
        ],
    }
    summary_payload = _canonical_json(summary)
    write_durable_exact_bytes(
        output_dir / "summary.json",
        summary_payload,
        label="intraday rolling-origin summary",
    )
    return summary


def _fit_profiles(
    train: pd.DataFrame,
) -> tuple[dict[str, dict[tuple[str, str, int], np.ndarray]], dict[tuple[str, str, int], int]]:
    profiles: dict[str, dict[tuple[str, str, int], np.ndarray]] = {
        "ratio": {},
        "price_space": {},
    }
    parent_hours: dict[tuple[str, str, int], int] = {}
    t_max = train.index.max()
    for raw_key, group in train.groupby(["saison", "type_jour", "heure_hce"], sort=True):
        key = (str(raw_key[0]), str(raw_key[1]), int(raw_key[2]))
        count = int(group.index.floor("h").nunique())
        if count < MIN_OBS_COUCHE1:
            continue
        ratio = _ratio_profile(group, t_max=t_max)
        price_space = _price_space_profile(group, t_max=t_max)
        if ratio is None or price_space is None:
            continue
        profiles["ratio"][key] = ratio
        profiles["price_space"][key] = price_space
        parent_hours[key] = count
    return profiles, parent_hours


def _candidate_profiles(
    *,
    profiles: Mapping[str, Mapping[tuple[str, str, int], np.ndarray]],
    parent_hours: Mapping[tuple[str, str, int], int],
) -> tuple[dict[tuple[str, str, int], np.ndarray], float]:
    dense_deviations = [
        float(np.max(np.abs(profile - 1.0)))
        for key, profile in profiles["ratio"].items()
        if parent_hours[key] > MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR
    ]
    if dense_deviations:
        envelope = max(dense_deviations)
    else:
        envelope = max(
            float(np.max(np.abs(profile - 1.0)))
            for profile in profiles["ratio"].values()
        )
    candidate: dict[tuple[str, str, int], np.ndarray] = {}
    for key, ratio_profile in profiles["ratio"].items():
        if parent_hours[key] > MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR:
            candidate[key] = ratio_profile.copy()
            continue
        selected = profiles["price_space"][key].copy()
        deviation = float(np.max(np.abs(selected - 1.0)))
        if deviation > envelope and deviation > np.finfo(float).eps:
            selected = 1.0 + (selected - 1.0) * (envelope / deviation)
            selected = selected / selected.mean()
        candidate[key] = selected
    return candidate, envelope


def _ratio_profile(group: pd.DataFrame, *, t_max: pd.Timestamp) -> np.ndarray | None:
    hour_means = group.groupby(group.index.floor("h"))["price_eur_mwh"].mean()
    values: list[float] = []
    for quarter in range(1, 5):
        selected = group[group["quart"] == quarter]
        x = selected.index.floor("h").map(hour_means).to_numpy(dtype=float)
        y = selected["price_eur_mwh"].to_numpy(dtype=float)
        weights = _decay_weights(selected.index, t_max=t_max)
        admitted = np.abs(x) > 0.1
        if int(admitted.sum()) < MIN_OBS_COUCHE1:
            return None
        model = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
        model.fit(
            np.ones((int(admitted.sum()), 1)),
            y[admitted] / x[admitted],
            sample_weight=weights[admitted],
        )
        values.append(float(model.coef_[0]))
    return _normalize(values)


def _price_space_profile(group: pd.DataFrame, *, t_max: pd.Timestamp) -> np.ndarray | None:
    hour_means = group.groupby(group.index.floor("h"))["price_eur_mwh"].mean()
    values: list[float] = []
    for quarter in range(1, 5):
        selected = group[group["quart"] == quarter]
        x = selected.index.floor("h").map(hour_means).to_numpy(dtype=float)
        y = selected["price_eur_mwh"].to_numpy(dtype=float)
        weights = _decay_weights(selected.index, t_max=t_max)
        denominator = float(np.sum(weights * x * x))
        if len(x) < MIN_OBS_COUCHE1 or denominator <= np.finfo(float).eps:
            return None
        values.append(float(np.sum(weights * x * y) / denominator))
    return _normalize(values)


def _normalize(values: list[float]) -> np.ndarray | None:
    selected = np.asarray(values, dtype=float)
    mean = float(selected.mean())
    if not np.isfinite(selected).all() or abs(mean) <= np.finfo(float).eps:
        return None
    return selected / mean


def _decay_weights(index: pd.DatetimeIndex, *, t_max: pd.Timestamp) -> np.ndarray:
    days_ago = (t_max - index).total_seconds() / 86400.0
    return np.exp(-(math.log(2.0) / 180.0) * days_ago)


def _fallback_key(
    profiles: Mapping[tuple[str, str, int], np.ndarray],
    key: tuple[str, str, int],
) -> tuple[str, str, int] | None:
    season, day_type, hour = key
    selected = day_type
    while selected in _FALLBACK_DAY_TYPE:
        selected = _FALLBACK_DAY_TYPE[selected]
        fallback = (season, selected, hour)
        if fallback in profiles:
            return fallback
    for fallback_season in SAISONS:
        fallback = (fallback_season, "Ouvrable", hour)
        if fallback in profiles:
            return fallback
    return None


def _score_fold(
    *,
    model_name: str,
    origin: pd.Timestamp,
    test_end: pd.Timestamp,
    test: pd.DataFrame,
    profiles: Mapping[tuple[str, str, int], np.ndarray],
    parent_hours: Mapping[tuple[str, str, int], int],
    dense_support_envelope: float,
) -> dict[str, object]:
    factors: list[float] = []
    direct: list[bool] = []
    sparse: list[bool] = []
    for row in test.itertuples():
        key = (str(row.saison), str(row.type_jour), int(row.heure_hce))
        selected = key if key in profiles else _fallback_key(profiles, key)
        profile = profiles[selected] if selected is not None else np.ones(4)
        factors.append(float(profile[int(row.quart) - 1]))
        is_direct = key in profiles
        direct.append(is_direct)
        sparse.append(
            is_direct
            and parent_hours[key] <= MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR
        )
    factor_values = np.asarray(factors)
    y = test["price_eur_mwh"].to_numpy(dtype=float)
    hour_mean = (
        test.groupby(test.index.floor("h"))["price_eur_mwh"]
        .transform("mean")
        .to_numpy(dtype=float)
    )
    error = y - hour_mean * factor_values
    abs_error = np.abs(error)
    direct_mask = np.asarray(direct)
    sparse_mask = np.asarray(sparse)
    return {
        "origin_utc": origin.isoformat(),
        "test_end_utc": test_end.isoformat(),
        "model": model_name,
        "n_quarter_hours": len(test),
        "n_direct_quarter_hours": int(direct_mask.sum()),
        "n_sparse_quarter_hours": int(sparse_mask.sum()),
        "mae_eur_mwh": float(abs_error.mean()),
        "rmse_eur_mwh": float(np.sqrt(np.mean(error * error))),
        "direct_mae_eur_mwh": float(abs_error[direct_mask].mean()) if direct_mask.any() else math.nan,
        "sparse_mae_eur_mwh": float(abs_error[sparse_mask].mean()) if sparse_mask.any() else math.nan,
        "max_factor_abs_deviation": float(np.max(np.abs(factor_values - 1.0))),
        "surface_max_factor_abs_deviation": float(
            max(np.max(np.abs(profile - 1.0)) for profile in profiles.values())
        ),
        "dense_support_envelope": dense_support_envelope,
    }


def _aggregate(folds: pd.DataFrame) -> dict[str, float | int]:
    weights = folds["n_quarter_hours"].to_numpy(dtype=float)
    sparse = folds.dropna(subset=["sparse_mae_eur_mwh"])
    sparse_weights = sparse["n_sparse_quarter_hours"].to_numpy(dtype=float)
    return {
        "fold_count": len(folds),
        "quarter_hours": int(weights.sum()),
        "mae_eur_mwh": float(np.average(folds["mae_eur_mwh"], weights=weights)),
        "rmse_eur_mwh": float(
            np.sqrt(np.average(folds["rmse_eur_mwh"] ** 2, weights=weights))
        ),
        "sparse_mae_eur_mwh": float(
            np.average(sparse["sparse_mae_eur_mwh"], weights=sparse_weights)
        ),
        "max_factor_abs_deviation": float(folds["max_factor_abs_deviation"].max()),
        "surface_max_factor_abs_deviation": float(
            folds["surface_max_factor_abs_deviation"].max()
        ),
    }


def _read_bound(path: Path, expected_sha256: str, *, label: str) -> bytes:
    if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    payload = read_stable_single_link_file(path, label=label, max_bytes=_MAX_INPUT_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return payload


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not UTF-8 JSON") from exc
    return _mapping(value, label=label)


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _timestamp(value: str) -> pd.Timestamp:
    selected = pd.Timestamp(value)
    if selected.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamps must include a timezone")
    return selected.tz_convert("UTC")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--expected-panel-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--origin-start", type=_timestamp, required=True)
    parser.add_argument("--origin-end", type=_timestamp, required=True)
    parser.add_argument("--fold-days", type=int, default=14)
    parser.add_argument("--minimum-train-days", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_backtest(
        panel_path=args.panel,
        expected_panel_sha256=args.expected_panel_sha256,
        panel_manifest_path=args.panel_manifest,
        expected_panel_manifest_sha256=args.expected_panel_manifest_sha256,
        output_dir=args.output_dir,
        origin_start=args.origin_start,
        origin_end=args.origin_end,
        fold_days=args.fold_days,
        minimum_train_days=args.minimum_train_days,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
