"""Offline lambda calibration harness for the monthly forward curve solver.

The harness is intentionally research-only.  It evaluates candidate monthly
curve configurations on rolling-origin snapshots by withholding monthly or
quarterly products, solving from a degraded quote set, and scoring the solved
curve against the withheld same-snapshot target.  All solver inputs and priors
are built from the visible quote set plus history strictly before the origin
date.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import yaml

from pfc_shaping.calibration.monthly_curve_audit import audit_monthly_curve_shape
from pfc_shaping.calibration.monthly_curve_priors import (
    MonthlyShapePrior,
    build_fused_shape_prior,
    build_history_shape_prior,
    build_neighbor_panel_shape_prior,
    build_structural_monthly_shape_prior_from_history,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    product_periods,
    solve_monthly_forward_curve_from_constraints,
)


SCORING_COLUMNS = [
    "config_hash",
    "origin_date",
    "snapshot_id",
    "source_file_hash",
    "market",
    "withheld_product",
    "withheld_load_type",
    "withheld_tenor",
    "withheld_horizon_years",
    "withheld_horizon_bucket",
    "product_start",
    "product_end",
    "target_price",
    "predicted_price",
    "abs_error",
    "signed_error",
    "constraint_residual_max",
    "curvature_score",
    "same_month_rank_score",
    "historical_outlier_score",
    "neighbor_disagreement_score",
    "unsupported_gate_count",
    "critical_gate_count",
    "status",
    "unsupported_reason",
    "sample_size",
]

FINAL_STATUSES = {
    "PASS_IMPLEMENTATION_ONLY",
    "PASS_CALIBRATION_CANDIDATE_NOT_PRODUCTION_APPROVED",
    "UNSUPPORTED_INSUFFICIENT_HISTORY",
    "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS",
    "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA",
    "FAIL_LEAKAGE_DETECTED",
    "FAIL_HARD_CONSTRAINT_VIOLATION",
    "FAIL_NON_REPRODUCIBLE_HASH",
    "FAIL_ARTIFACT_INCOMPLETE",
}

FAIL_STATUSES = {
    "FAIL_LEAKAGE_DETECTED",
    "FAIL_HARD_CONSTRAINT_VIOLATION",
    "FAIL_NON_REPRODUCIBLE_HASH",
    "FAIL_ARTIFACT_INCOMPLETE",
}

_MONTH_RE = r"^\d{4}-\d{2}$"
_QUARTER_RE = r"^\d{4}-Q[1-4]$"
_YEAR_RE = r"^\d{4}$"


@dataclass(frozen=True)
class LambdaCalibrationSettings:
    market: str = "CH"
    load_type: str = "BASE"
    timezone: str = "Europe/Zurich"
    neighbor_markets: tuple[str, ...] = ("DE", "FR", "AT", "IT")
    min_valid_origins: int = 3
    min_withheld_monthly: int = 3
    min_withheld_quarterly: int = 2
    min_history_snapshots: int = 24
    max_withheld_per_origin: int = 4
    quote_consistency_tolerance: float = 0.01
    hard_constraint_tolerance: float = 1e-8
    history_lookback_years: int | None = 6
    structural_weight: float = 0.5
    panel_weight: float = 1.0
    history_weight: float = 0.25
    allow_template_structural_fallback: bool = False
    structural_amplitude_eur_mwh: float = 20.0
    min_structural_snapshots: int = 24
    identifiability_min_abs_error_improvement: float = 0.05
    identifiability_min_rel_error_improvement: float = 0.01


@dataclass(frozen=True)
class WithheldProduct:
    market: str
    product: str
    load_type: str
    price: float
    origin_date: pd.Timestamp
    source: str = ""

    @property
    def tenor(self) -> str:
        return product_tenor(self.product)

    @property
    def key(self) -> str:
        return quote_key(self.market, self.load_type, self.product)


@dataclass(frozen=True)
class LambdaCandidateConfig:
    monthly_config: MonthlyCurveConfig
    history_lookback_years: int | None


@dataclass(frozen=True)
class CalibrationArtifacts:
    scoring: pd.DataFrame
    manifest: dict[str, object]
    summary: dict[str, object]
    candidate_config: dict[str, object] | None


def run_lambda_calibration(
    history: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    settings: LambdaCalibrationSettings | None = None,
    source_file_hash: str = "",
    max_origins: int | None = None,
    max_configs: int | None = None,
    smoke: bool = False,
    command_line: Sequence[str] | None = None,
    input_parquet_path: str = "",
) -> CalibrationArtifacts:
    """Run a point-in-time rolling-origin lambda calibration."""

    settings = _settings_from_grid(grid, settings=settings, smoke=smoke)
    history = normalize_history(history)
    configs = list(iter_candidate_configs(grid))
    if not configs:
        raise ValueError("lambda grid did not produce any candidate configuration")
    if smoke:
        configs = configs[: max_configs or 2]
        max_origins = max_origins or 2
    elif max_configs is not None:
        configs = configs[: int(max_configs)]

    origins = select_origins(history, settings=settings, max_origins=max_origins)
    scoring_rows: list[dict[str, object]] = []
    excluded_reasons: Counter[str] = Counter()
    withheld_counts: Counter[tuple[str, str, str]] = Counter()

    for origin in origins:
        full_quote_set = quote_set_for_origin(history, origin)
        withheld = select_withheld_products(
            full_quote_set,
            settings=settings,
            max_products=settings.max_withheld_per_origin if not smoke else min(2, settings.max_withheld_per_origin),
        )
        if not withheld:
            excluded_reasons["no_withheld_products"] += 1
            continue
        for withheld_product in withheld:
            withheld_counts[
                (
                    withheld_product.market,
                    withheld_product.load_type,
                    withheld_product.tenor,
                )
            ] += 1
            masked = mask_quote_sets(
                full_quote_set,
                withheld_product,
                neighbor_markets=settings.neighbor_markets,
            )
            try:
                validate_masked_inputs(masked, withheld_product)
            except ValueError:
                excluded_reasons["mask_leakage_detected"] += 1
                raise
            for cfg in configs:
                history_view = history_before_origin(
                    history,
                    origin,
                    lookback_years=_candidate_history_lookback(cfg, settings),
                )
                validate_history_point_in_time(history_view, origin)
                scoring_rows.append(
                    _score_one_case(
                        masked=masked,
                        withheld=withheld_product,
                        config=cfg,
                        settings=settings,
                        history_view=history_view,
                        source_file_hash=source_file_hash,
                    )
                )

    scoring = pd.DataFrame(scoring_rows, columns=SCORING_COLUMNS)
    summary = summarize_calibration(
        scoring,
        configs=configs,
        settings=settings,
        withheld_counts=withheld_counts,
        excluded_reasons=excluded_reasons,
    )
    manifest = build_calibration_manifest(
        history=history,
        scoring=scoring,
        summary=summary,
        settings=settings,
        grid=grid,
        source_file_hash=source_file_hash,
        command_line=command_line or (),
        input_parquet_path=input_parquet_path,
        withheld_counts=withheld_counts,
        excluded_reasons=excluded_reasons,
    )
    candidate = build_candidate_config(
        scoring,
        summary=summary,
        configs=configs,
        grid_hash=lambda_grid_hash(grid),
        source_data_hash=source_file_hash,
        withheld_counts=withheld_counts,
        excluded_reasons=excluded_reasons,
    )
    _validate_artifacts(scoring, manifest, summary, candidate)
    return CalibrationArtifacts(scoring, manifest, summary, candidate)


def write_calibration_artifacts(artifacts: CalibrationArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.scoring.to_csv(output_dir / "scoring.csv", index=False)
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(artifacts.manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(artifacts.summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    if artifacts.candidate_config is not None:
        (output_dir / "candidate_config.yaml").write_text(
            yaml.safe_dump(artifacts.candidate_config, sort_keys=True),
            encoding="utf-8",
        )


def load_lambda_grid(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("lambda grid YAML must contain a mapping")
    return data


def iter_candidate_configs(grid: Mapping[str, object]) -> Iterable[LambdaCandidateConfig]:
    config_grid = dict(grid.get("grid", grid))
    defaults = dict(grid.get("defaults", {}))
    keys = ["lambda_smooth_month", "lambda_smooth_yoy", "lambda_shape", "neighbor_shrinkage"]
    value_lists: list[list[object]] = []
    for key in keys:
        raw = config_grid.get(key, defaults.get(key))
        if raw is None:
            raise ValueError(f"missing lambda grid key {key!r}")
        value_lists.append(list(raw if isinstance(raw, list) else [raw]))

    lookbacks = config_grid.get("history_lookback_years", defaults.get("history_lookback_years", 6))
    lookback_values = list(lookbacks if isinstance(lookbacks, list) else [lookbacks])

    baseline = LambdaCandidateConfig(
        monthly_config=MonthlyCurveConfig(
            lambda_prior=float(defaults.get("lambda_prior", 1e-6)),
            lambda_smooth_month=1.0,
            lambda_smooth_yoy=0.0,
            lambda_shape=0.0,
            neighbor_shrinkage=0.5,
            min_history_snapshots=int(defaults.get("min_history_snapshots", 24)),
            constraint_tolerance=float(defaults.get("constraint_tolerance", 0.01)),
            stationarity_tolerance=float(defaults.get("stationarity_tolerance", 1e-7)),
        ),
        history_lookback_years=int(lookback_values[0]) if lookback_values[0] is not None else None,
    )
    yield baseline

    for smooth_month in value_lists[0]:
        for smooth_yoy in value_lists[1]:
            for shape in value_lists[2]:
                for shrinkage in value_lists[3]:
                    for lookback in lookback_values:
                        yield LambdaCandidateConfig(
                            monthly_config=MonthlyCurveConfig(
                                lambda_prior=float(defaults.get("lambda_prior", 1e-6)),
                                lambda_smooth_month=float(smooth_month),
                                lambda_smooth_yoy=float(smooth_yoy),
                                lambda_shape=float(shape),
                                neighbor_shrinkage=float(shrinkage),
                                min_history_snapshots=int(defaults.get("min_history_snapshots", 24)),
                                constraint_tolerance=float(defaults.get("constraint_tolerance", 0.01)),
                                stationarity_tolerance=float(defaults.get("stationarity_tolerance", 1e-7)),
                            ),
                            history_lookback_years=int(lookback) if lookback is not None else None,
                        )


def config_hash(config: MonthlyCurveConfig | LambdaCandidateConfig | Mapping[str, object]) -> str:
    payload = _config_payload(config)
    first = _sha256_text(canonical_json(payload))
    second = _sha256_text(canonical_json(dict(reversed(list(payload.items())))))
    if first != second:
        raise ValueError("non-reproducible config hash")
    return first


def lambda_grid_hash(grid: Mapping[str, object]) -> str:
    return _sha256_text(canonical_json(grid))


def canonical_json(value: object) -> str:
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "product", "load_type", "market", "price"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"missing required EEX history columns: {missing}")
    df = history.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["product"] = df["product"].astype(str)
    df["load_type"] = df["load_type"].astype(str).str.upper()
    df["market"] = df["market"].astype(str).str.upper()
    df["price"] = df["price"].astype(float)
    if "source" not in df.columns:
        df["source"] = ""
    return df.sort_values(["date", "market", "load_type", "product"]).reset_index(drop=True)


def select_origins(
    history: pd.DataFrame,
    *,
    settings: LambdaCalibrationSettings,
    max_origins: int | None = None,
) -> list[pd.Timestamp]:
    sub = history[
        (history["market"].eq(settings.market.upper()))
        & (history["load_type"].eq(settings.load_type.upper()))
        & (history["product"].astype(str).str.match(f"{_MONTH_RE}|{_QUARTER_RE}"))
    ]
    dates = sorted(pd.Timestamp(d) for d in sub["date"].dropna().unique())
    if max_origins is not None:
        dates = dates[-int(max_origins) :]
    return dates


def quote_set_for_origin(history: pd.DataFrame, origin: pd.Timestamp) -> tuple[MarketQuote, ...]:
    origin = pd.Timestamp(origin).tz_localize(None).normalize()
    snap = history[history["date"].eq(origin)]
    return tuple(
        MarketQuote(
            market=str(row.market).upper(),
            product=str(row.product),
            load_type=str(row.load_type).upper(),
            price=float(row.price),
            snapshot_date=origin,
            source=str(getattr(row, "source", "")),
            available_at=origin,
        )
        for row in snap.itertuples(index=False)
        if np.isfinite(float(row.price))
    )


def select_withheld_products(
    full_quote_set: Sequence[MarketQuote],
    *,
    settings: LambdaCalibrationSettings,
    max_products: int,
) -> tuple[WithheldProduct, ...]:
    candidates: list[WithheldProduct] = []
    for quote in full_quote_set:
        if quote.market.upper() != settings.market.upper():
            continue
        if quote.load_type.upper() != settings.load_type.upper():
            continue
        tenor = product_tenor(quote.product)
        if tenor not in {"monthly", "quarterly"}:
            continue
        if quote.snapshot_date is None:
            continue
        candidates.append(
            WithheldProduct(
                market=quote.market.upper(),
                product=quote.product,
                load_type=quote.load_type.upper(),
                price=float(quote.price),
                origin_date=pd.Timestamp(quote.snapshot_date).tz_localize(None).normalize(),
                source=quote.source,
            )
        )
    candidates.sort(key=lambda item: (item.tenor, item.product))
    monthly = [item for item in candidates if item.tenor == "monthly"]
    quarterly = [item for item in candidates if item.tenor == "quarterly"]
    selected: list[WithheldProduct] = []
    selected.extend(monthly[: max(0, max_products // 2)])
    selected.extend(quarterly[: max(0, max_products - len(selected))])
    if len(selected) < max_products:
        selected.extend([item for item in candidates if item not in selected][: max_products - len(selected)])
    return tuple(selected[:max_products])


@dataclass(frozen=True)
class MaskedQuoteSets:
    origin_date: pd.Timestamp
    full_quote_set: tuple[MarketQuote, ...]
    withheld_set: tuple[WithheldProduct, ...]
    visible_quote_set: tuple[MarketQuote, ...]
    own_quotes: tuple[MarketQuote, ...]
    neighbor_quotes: tuple[MarketQuote, ...]
    removed_quote_keys: tuple[str, ...]
    removed_reasons: Mapping[str, str]


def mask_quote_sets(
    full_quote_set: Sequence[MarketQuote],
    withheld: WithheldProduct,
    *,
    neighbor_markets: Sequence[str],
) -> MaskedQuoteSets:
    forbidden = revealing_quote_keys(withheld)
    neighbor_set = {m.upper() for m in neighbor_markets}
    visible: list[MarketQuote] = []
    removed: dict[str, str] = {}
    for quote in full_quote_set:
        key = quote_key(quote.market, quote.load_type, quote.product)
        same_load = quote.load_type.upper() == withheld.load_type.upper()
        same_market = quote.market.upper() == withheld.market.upper()
        is_neighbor = quote.market.upper() in neighbor_set
        if same_load and (same_market or is_neighbor) and key in forbidden:
            removed[key] = "withheld_or_revealing_product"
            continue
        visible.append(quote)
    own = tuple(
        q for q in visible
        if q.market.upper() == withheld.market.upper() and q.load_type.upper() == withheld.load_type.upper()
    )
    neighbors = tuple(q for q in visible if q.market.upper() in neighbor_set and q.load_type.upper() == withheld.load_type.upper())
    return MaskedQuoteSets(
        origin_date=withheld.origin_date,
        full_quote_set=tuple(full_quote_set),
        withheld_set=(withheld,),
        visible_quote_set=tuple(visible),
        own_quotes=own,
        neighbor_quotes=neighbors,
        removed_quote_keys=tuple(sorted(removed)),
        removed_reasons=removed,
    )


def revealing_quote_keys(withheld: WithheldProduct) -> set[str]:
    markets = {withheld.market.upper(), "DE", "FR", "AT", "IT"}
    products = revealing_products(withheld.product)
    return {
        quote_key(market, withheld.load_type, product)
        for market in markets
        for product in products
    }


def revealing_products(product: str) -> set[str]:
    product = str(product)
    tenor = product_tenor(product)
    months = product_periods(product)
    year = int(months[0].year)
    products = {product, str(year)}
    quarters = {f"{month.year}-Q{((month.month - 1) // 3) + 1}" for month in months}
    products.update(quarters)
    if tenor == "quarterly":
        products.update(str(month) for month in months)
    return products


def validate_masked_inputs(masked: MaskedQuoteSets, withheld: WithheldProduct) -> None:
    forbidden = revealing_quote_keys(withheld)
    for quote in masked.own_quotes + masked.neighbor_quotes:
        key = quote_key(quote.market, quote.load_type, quote.product)
        if key in forbidden:
            raise ValueError(f"masked quote leakage detected for {key}")
        if quote.available_at is not None and pd.Timestamp(quote.available_at).tz_localize(None) > withheld.origin_date:
            raise ValueError(f"future quote leaked into calibration inputs: {key}")


def history_before_origin(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    lookback_years: int | None,
) -> pd.DataFrame:
    origin = pd.Timestamp(origin).tz_localize(None).normalize()
    hist = history[history["date"] < origin].copy()
    if lookback_years is not None:
        hist = hist[hist["date"] >= origin - pd.DateOffset(years=int(lookback_years))]
    return hist


def validate_history_point_in_time(history_view: pd.DataFrame, origin: pd.Timestamp) -> None:
    if history_view.empty:
        return
    origin = pd.Timestamp(origin).tz_localize(None).normalize()
    max_date = pd.to_datetime(history_view["date"]).dt.tz_localize(None).max()
    if max_date >= origin:
        raise ValueError(f"history feature leakage: max history date {max_date} >= origin {origin}")


def validate_feature_frame_point_in_time(
    features: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    withheld: WithheldProduct,
) -> None:
    if features.empty:
        return
    if "date" in features.columns:
        dates = pd.to_datetime(features["date"]).dt.tz_localize(None)
        if bool((dates >= pd.Timestamp(origin).tz_localize(None).normalize()).any()):
            raise ValueError("pre-mask feature cache includes same-origin or future rows")
    if "product" in features.columns and bool(features["product"].astype(str).eq(withheld.product).any()):
        raise ValueError("pre-mask feature cache includes the withheld product")


def summarize_calibration(
    scoring: pd.DataFrame,
    *,
    configs: Sequence[MonthlyCurveConfig | LambdaCandidateConfig],
    settings: LambdaCalibrationSettings,
    withheld_counts: Counter[tuple[str, str, str]],
    excluded_reasons: Counter[str],
) -> dict[str, object]:
    final_status = _final_status(scoring, settings=settings)
    by_config = _config_metric_table(scoring)
    baseline_hash = config_hash(configs[0])
    best_hash = _best_config_hash(scoring, baseline_hash=baseline_hash)
    baseline_mae = _mae_for_config(scoring, baseline_hash)
    best_mae = _mae_for_config(scoring, best_hash) if best_hash else np.nan
    selection_reason = _selection_reason(
        final_status=final_status,
        baseline_mae=baseline_mae,
        best_mae=best_mae,
        baseline_hash=baseline_hash,
        best_hash=best_hash,
    )
    return {
        "final_status": final_status,
        "production_approved": False,
        "n_scoring_rows": int(len(scoring)),
        "n_origins_evaluated": int(scoring["origin_date"].nunique()) if not scoring.empty else 0,
        "n_configs_evaluated": int(scoring["config_hash"].nunique()) if not scoring.empty else 0,
        "sample_counts": _counter_to_nested_counts(withheld_counts),
        "excluded_case_counts": dict(excluded_reasons),
        "baseline_config_hash": baseline_hash,
        "selected_config_hash": best_hash,
        "baseline_mae": None if not np.isfinite(baseline_mae) else float(baseline_mae),
        "selected_mae": None if not np.isfinite(best_mae) else float(best_mae),
        "selection_reason": selection_reason,
        "train_deploy_gap": _train_deploy_gap_summary(scoring),
        "by_tenor_horizon": _tenor_horizon_metric_table(scoring),
        "by_config": by_config,
    }


def build_calibration_manifest(
    *,
    history: pd.DataFrame,
    scoring: pd.DataFrame,
    summary: Mapping[str, object],
    settings: LambdaCalibrationSettings,
    grid: Mapping[str, object],
    source_file_hash: str,
    command_line: Sequence[str],
    input_parquet_path: str,
    withheld_counts: Counter[tuple[str, str, str]],
    excluded_reasons: Counter[str],
) -> dict[str, object]:
    row_counts = {
        str(date.date() if hasattr(date, "date") else date): int(count)
        for date, count in history.groupby("date", sort=True).size().items()
    }
    return {
        "git_commit": _git_commit(),
        "command_line": list(command_line),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_parquet_path": input_parquet_path,
        "input_parquet_sha256": source_file_hash,
        "lambda_grid_hash": lambda_grid_hash(grid),
        "run_config_hash": _sha256_text(canonical_json(_settings_payload(settings))),
        "python_version": platform.python_version(),
        "row_counts_by_snapshot": row_counts,
        "origins_evaluated": int(scoring["origin_date"].nunique()) if not scoring.empty else 0,
        "withheld_products_by_market_load_type_tenor": _counter_to_nested_counts(withheld_counts),
        "withheld_products_by_tenor_horizon": _scoring_count_table(
            scoring,
            keys=("withheld_tenor", "withheld_horizon_bucket"),
        ),
        "excluded_cases_by_reason": dict(excluded_reasons),
        "final_status": str(summary["final_status"]),
        "production_approved": False,
    }


def build_candidate_config(
    scoring: pd.DataFrame,
    *,
    summary: Mapping[str, object],
    configs: Sequence[MonthlyCurveConfig | LambdaCandidateConfig],
    grid_hash: str,
    source_data_hash: str,
    withheld_counts: Counter[tuple[str, str, str]],
    excluded_reasons: Counter[str],
) -> dict[str, object] | None:
    final_status = str(summary["final_status"])
    if final_status in FAIL_STATUSES:
        return None
    selected_hash = str(summary.get("selected_config_hash") or config_hash(configs[0]))
    selected = next((cfg for cfg in configs if config_hash(cfg) == selected_hash), configs[0])
    selection_status = "SELECTED" if final_status.startswith("PASS_CALIBRATION_CANDIDATE") else "UNSUPPORTED"
    if final_status == "PASS_IMPLEMENTATION_ONLY":
        selection_status = "NOT_PRODUCTION_APPROVED"
    return {
        "config_hash": config_hash(selected),
        "canonical_config": _config_payload(selected),
        "grid_file_hash": grid_hash,
        "source_data_hash": source_data_hash,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_status": selection_status,
        "production_approved": False,
        "selection_reason": str(summary.get("selection_reason", "")),
        "sample_counts": _counter_to_nested_counts(withheld_counts),
        "excluded_config_counts": dict(excluded_reasons),
        "baseline_comparison": {
            "baseline_config_hash": summary.get("baseline_config_hash"),
            "baseline_mae": summary.get("baseline_mae"),
            "selected_mae": summary.get("selected_mae"),
            "by_tenor_horizon": summary.get("by_tenor_horizon", []),
        },
        "known_bad_known_coherent_gate_summary": {
            "available": False,
            "reason": "not part of offline lambda calibration harness",
        },
    }


def quote_key(market: str, load_type: str, product: str) -> str:
    return f"{str(market).upper()}:{str(load_type).upper()}:{str(product)}"


def product_tenor(product: str) -> str:
    text = str(product)
    if pd.Series([text]).str.match(_MONTH_RE).iloc[0]:
        return "monthly"
    if pd.Series([text]).str.match(_QUARTER_RE).iloc[0]:
        return "quarterly"
    if pd.Series([text]).str.match(_YEAR_RE).iloc[0]:
        return "calendar"
    return "unsupported"


def product_horizon_years(origin_date: pd.Timestamp, product: str) -> int:
    periods = product_periods(product)
    origin = pd.Timestamp(origin_date).tz_localize(None).normalize()
    return int(periods[0].year) - int(origin.year)


def horizon_bucket(horizon_years: int) -> str:
    if horizon_years <= 0:
        return "h+0"
    if horizon_years == 1:
        return "h+1"
    if horizon_years == 2:
        return "h+2"
    return "h+3+"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_one_case(
    *,
    masked: MaskedQuoteSets,
    withheld: WithheldProduct,
    config: MonthlyCurveConfig | LambdaCandidateConfig,
    settings: LambdaCalibrationSettings,
    history_view: pd.DataFrame,
    source_file_hash: str,
) -> dict[str, object]:
    monthly_config = _monthly_config(config)
    months = _delivery_months_for_withheld(withheld)
    horizon_years = product_horizon_years(withheld.origin_date, withheld.product)
    horizon = horizon_bucket(horizon_years)
    try:
        constraints = build_monthly_constraint_system(
            months,
            masked.own_quotes,
            timezone=settings.timezone,
            market=settings.market,
            load_type=settings.load_type,
            constraint_tolerance=monthly_config.constraint_tolerance,
        )
        shape_prior = _build_shape_prior(
            constraints=constraints,
            masked=masked,
            config=monthly_config,
            settings=settings,
            history_view=history_view,
        )
        result = solve_monthly_forward_curve_from_constraints(
            constraints,
            config=monthly_config,
            shape_prior=shape_prior,
        )
        predicted = _product_average(result.monthly_curve, withheld.product, constraints.delivery_grid.month_hours)
        signed_error = float(predicted) - float(withheld.price)
        residual_max = float(result.kkt["max_abs_constraint_residual"])
        audit = audit_monthly_curve_shape(result.monthly_curve, constraints)
        critical_count = int(audit["status"].astype(str).eq("CRITICAL").sum()) if not audit.empty else 0
        unsupported_count = int(audit["status"].astype(str).eq("UNSUPPORTED").sum()) if not audit.empty else 0
        row_status = "PASS"
        reason = ""
        if residual_max > settings.hard_constraint_tolerance:
            row_status = "FAIL_HARD_CONSTRAINT_VIOLATION"
            reason = "hard constraint residual exceeds tolerance"
        elif critical_count > 0:
            row_status = "CRITICAL_GATES"
            reason = "critical audit gates present"
        return {
            "config_hash": config_hash(config),
            "origin_date": withheld.origin_date.date().isoformat(),
            "snapshot_id": withheld.origin_date.date().isoformat(),
            "source_file_hash": source_file_hash,
            "market": withheld.market,
            "withheld_product": withheld.product,
            "withheld_load_type": withheld.load_type,
            "withheld_tenor": withheld.tenor,
            "withheld_horizon_years": horizon_years,
            "withheld_horizon_bucket": horizon,
            "product_start": str(product_periods(withheld.product)[0]),
            "product_end": str(product_periods(withheld.product)[-1]),
            "target_price": float(withheld.price),
            "predicted_price": float(predicted),
            "abs_error": abs(signed_error),
            "signed_error": signed_error,
            "constraint_residual_max": residual_max,
            "curvature_score": _curvature_score(result.monthly_curve),
            "same_month_rank_score": _same_month_rank_score(audit),
            "historical_outlier_score": _historical_outlier_score(shape_prior),
            "neighbor_disagreement_score": _neighbor_disagreement_score(shape_prior),
            "unsupported_gate_count": unsupported_count,
            "critical_gate_count": critical_count,
            "status": row_status,
            "unsupported_reason": reason,
            "sample_size": int(len(history_view["date"].unique())) if not history_view.empty else 0,
        }
    except Exception as exc:
        return {
            "config_hash": config_hash(config),
            "origin_date": withheld.origin_date.date().isoformat(),
            "snapshot_id": withheld.origin_date.date().isoformat(),
            "source_file_hash": source_file_hash,
            "market": withheld.market,
            "withheld_product": withheld.product,
            "withheld_load_type": withheld.load_type,
            "withheld_tenor": withheld.tenor,
            "withheld_horizon_years": horizon_years,
            "withheld_horizon_bucket": horizon,
            "product_start": str(product_periods(withheld.product)[0]),
            "product_end": str(product_periods(withheld.product)[-1]),
            "target_price": float(withheld.price),
            "predicted_price": np.nan,
            "abs_error": np.nan,
            "signed_error": np.nan,
            "constraint_residual_max": np.nan,
            "curvature_score": np.nan,
            "same_month_rank_score": np.nan,
            "historical_outlier_score": np.nan,
            "neighbor_disagreement_score": np.nan,
            "unsupported_gate_count": 1,
            "critical_gate_count": 0,
            "status": "UNSUPPORTED",
            "unsupported_reason": str(exc),
            "sample_size": int(len(history_view["date"].unique())) if not history_view.empty else 0,
        }


def _build_shape_prior(
    *,
    constraints,
    masked: MaskedQuoteSets,
    config: MonthlyCurveConfig,
    settings: LambdaCalibrationSettings,
    history_view: pd.DataFrame,
) -> MonthlyShapePrior:
    neighbor_prices: dict[str, dict[str, float]] = {}
    for quote in masked.neighbor_quotes:
        neighbor_prices.setdefault(quote.market.upper(), {})[quote.product] = float(quote.price)
    panel = build_neighbor_panel_shape_prior(
        constraints,
        neighbor_prices,
        neighbor_markets=settings.neighbor_markets,
        neighbor_shrinkage=config.neighbor_shrinkage,
    )
    history = build_history_shape_prior(
        constraints,
        history_view,
        market=settings.market,
        load_type=settings.load_type,
        run_timestamp=masked.origin_date - pd.Timedelta(days=1),
        min_snapshots=config.min_history_snapshots,
        lookback_years=settings.history_lookback_years,
    )
    structural = build_structural_monthly_shape_prior_from_history(
        constraints,
        history_view,
        market=settings.market,
        load_type=settings.load_type,
        run_timestamp=masked.origin_date - pd.Timedelta(days=1),
        min_snapshots=settings.min_structural_snapshots,
        lookback_years=settings.history_lookback_years,
        fallback_to_template=settings.allow_template_structural_fallback,
        fallback_amplitude_eur_mwh=settings.structural_amplitude_eur_mwh,
    )
    return build_fused_shape_prior(
        constraints,
        panel_prior=panel,
        history_prior=history,
        structural_prior=structural,
        weights={
            "panel": settings.panel_weight,
            "history": settings.history_weight,
            "structural": settings.structural_weight,
        },
    )


def _delivery_months_for_withheld(withheld: WithheldProduct) -> pd.PeriodIndex:
    periods = product_periods(withheld.product)
    year = int(periods[0].year)
    return pd.period_range(f"{year}-01", f"{year}-12", freq="M")


def _product_average(curve: pd.Series, product: str, hours: pd.Series) -> float:
    months = product_periods(product)
    values = curve.reindex(months).astype(float)
    weights = hours.reindex(months).astype(float)
    if values.isna().any() or weights.isna().any():
        raise ValueError(f"curve does not cover withheld product {product}")
    return float((values.to_numpy() * weights.to_numpy()).sum() / weights.sum())


def _curvature_score(curve: pd.Series) -> float:
    arr = curve.to_numpy(dtype=float)
    if len(arr) < 3:
        return 0.0
    return float(np.mean(np.diff(arr, n=2) ** 2))


def _same_month_rank_score(audit: pd.DataFrame) -> float:
    if audit.empty or "metric_value" not in audit.columns:
        return 0.0
    sub = audit[audit["gate_id"].astype(str).eq("same_month_rank_consistency")]
    if sub.empty:
        return 0.0
    return float(sub["metric_value"].astype(float).abs().max())


def _historical_outlier_score(prior: MonthlyShapePrior) -> float:
    diag = prior.diagnostics
    if diag.empty or "mad_deviation" not in diag.columns:
        return 0.0 if prior.status != "UNSUPPORTED" else np.nan
    mad = pd.to_numeric(diag["mad_deviation"], errors="coerce").replace(0.0, np.nan)
    med = pd.to_numeric(diag.get("median_deviation", pd.Series(dtype=float)), errors="coerce")
    if mad.dropna().empty:
        return 0.0
    return float((med.abs() / mad).replace([np.inf, -np.inf], np.nan).max(skipna=True))


def _neighbor_disagreement_score(prior: MonthlyShapePrior) -> float:
    contrib = prior.contributions
    if contrib.empty or contrib.shape[1] < 2:
        return 0.0
    return float(contrib.std(axis=1, skipna=True).mean(skipna=True))


def _final_status(scoring: pd.DataFrame, *, settings: LambdaCalibrationSettings) -> str:
    if scoring.empty:
        return "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS"
    if bool(scoring["status"].astype(str).eq("FAIL_HARD_CONSTRAINT_VIOLATION").any()):
        return "FAIL_HARD_CONSTRAINT_VIOLATION"
    if bool((pd.to_numeric(scoring["constraint_residual_max"], errors="coerce") > settings.hard_constraint_tolerance).any()):
        return "FAIL_HARD_CONSTRAINT_VIOLATION"
    if bool((pd.to_numeric(scoring["critical_gate_count"], errors="coerce").fillna(0) > 0).any()):
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    origins = int(scoring["origin_date"].nunique())
    monthly = int(scoring[scoring["withheld_tenor"].eq("monthly")]["withheld_product"].nunique())
    quarterly = int(scoring[scoring["withheld_tenor"].eq("quarterly")]["withheld_product"].nunique())
    if origins < settings.min_valid_origins:
        return "UNSUPPORTED_INSUFFICIENT_HISTORY"
    if monthly < settings.min_withheld_monthly or quarterly < settings.min_withheld_quarterly:
        return "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS"
    baseline = str(scoring["config_hash"].iloc[0])
    best = _best_config_hash(scoring, baseline_hash=baseline)
    if best is None:
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    baseline_mae = _mae_for_config(scoring, baseline)
    best_mae = _mae_for_config(scoring, best)
    if not np.isfinite(baseline_mae) or not np.isfinite(best_mae):
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    abs_improvement = baseline_mae - best_mae
    rel_improvement = abs_improvement / max(abs(baseline_mae), 1e-9)
    if best == baseline or (
        abs_improvement < settings.identifiability_min_abs_error_improvement
        and rel_improvement < settings.identifiability_min_rel_error_improvement
    ):
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    return "PASS_CALIBRATION_CANDIDATE_NOT_PRODUCTION_APPROVED"


def _best_config_hash(scoring: pd.DataFrame, *, baseline_hash: str) -> str | None:
    if scoring.empty:
        return None
    valid = scoring[scoring["abs_error"].notna() & scoring["status"].astype(str).isin(["PASS", "CRITICAL_GATES"])]
    if valid.empty:
        return None
    grouped = valid.groupby("config_hash", sort=True)["abs_error"].mean()
    if grouped.empty:
        return baseline_hash
    return str(grouped.sort_values(kind="mergesort").index[0])


def _mae_for_config(scoring: pd.DataFrame, cfg_hash: str | None) -> float:
    if not cfg_hash or scoring.empty:
        return float("nan")
    sub = scoring[scoring["config_hash"].astype(str).eq(str(cfg_hash))]
    return float(pd.to_numeric(sub["abs_error"], errors="coerce").mean()) if not sub.empty else float("nan")


def _config_metric_table(scoring: pd.DataFrame) -> list[dict[str, object]]:
    if scoring.empty:
        return []
    rows: list[dict[str, object]] = []
    for cfg_hash, group in scoring.groupby("config_hash", sort=True):
        rows.append(
            {
                "config_hash": str(cfg_hash),
                "mean_abs_error": float(pd.to_numeric(group["abs_error"], errors="coerce").mean()),
                "median_abs_error": float(pd.to_numeric(group["abs_error"], errors="coerce").median()),
                "max_constraint_residual": float(pd.to_numeric(group["constraint_residual_max"], errors="coerce").max()),
                "critical_gate_count": int(pd.to_numeric(group["critical_gate_count"], errors="coerce").fillna(0).sum()),
                "n_rows": int(len(group)),
            }
        )
    return rows


def _tenor_horizon_metric_table(scoring: pd.DataFrame) -> list[dict[str, object]]:
    required = {"withheld_tenor", "withheld_horizon_bucket", "abs_error", "config_hash"}
    if scoring.empty or not required <= set(scoring.columns):
        return []
    rows: list[dict[str, object]] = []
    for (tenor, horizon), group in scoring.groupby(["withheld_tenor", "withheld_horizon_bucket"], sort=True):
        valid = pd.to_numeric(group["abs_error"], errors="coerce").dropna()
        rows.append(
            {
                "withheld_tenor": str(tenor),
                "withheld_horizon_bucket": str(horizon),
                "n_rows": int(len(group)),
                "n_origins": int(group["origin_date"].nunique()) if "origin_date" in group.columns else 0,
                "n_products": int(group["withheld_product"].nunique()) if "withheld_product" in group.columns else 0,
                "mean_abs_error": None if valid.empty else float(valid.mean()),
                "median_abs_error": None if valid.empty else float(valid.median()),
                "min_abs_error": None if valid.empty else float(valid.min()),
                "max_abs_error": None if valid.empty else float(valid.max()),
                "n_configs": int(group["config_hash"].nunique()),
            }
        )
    return rows


def _train_deploy_gap_summary(scoring: pd.DataFrame) -> dict[str, object]:
    if scoring.empty or "withheld_horizon_bucket" not in scoring.columns:
        return {
            "status": "UNSUPPORTED_NO_SCORING",
            "message": "No withheld-product scoring is available.",
        }
    counts = _scoring_count_table(scoring, keys=("withheld_tenor", "withheld_horizon_bucket"))
    far_rows = scoring[scoring["withheld_horizon_bucket"].astype(str).isin(["h+2", "h+3+"])]
    far_monthly_rows = far_rows[far_rows["withheld_tenor"].astype(str).eq("monthly")]
    if far_monthly_rows.empty:
        return {
            "status": "UNSUPPORTED_NO_FAR_HORIZON_MONTHLY_TRUTH",
            "message": (
                "Withheld monthly/quarterly quote tests validate near-tenor reconstruction only; "
                "the sparse h+2/h+3 deployment zone has no direct withheld monthly truth in this run."
            ),
            "counts": counts,
            "far_horizon_non_monthly_rows": int(len(far_rows)),
        }
    return {
        "status": "PARTIAL_FAR_HORIZON_EVIDENCE",
        "message": "Some far-horizon withheld monthly products were scored; inspect by_tenor_horizon before promotion.",
        "counts": counts,
        "far_horizon_monthly_rows": int(len(far_monthly_rows)),
    }


def _scoring_count_table(scoring: pd.DataFrame, *, keys: tuple[str, ...]) -> dict[str, int]:
    if scoring.empty or not set(keys) <= set(scoring.columns):
        return {}
    return {
        "|".join(str(part) for part in idx if str(part) != ""): int(count)
        for idx, count in scoring.groupby(list(keys), sort=True).size().items()
    }


def _selection_reason(
    *,
    final_status: str,
    baseline_mae: float,
    best_mae: float,
    baseline_hash: str,
    best_hash: str | None,
) -> str:
    if final_status.startswith("PASS_CALIBRATION_CANDIDATE"):
        return (
            "candidate minimizes withheld-product MAE versus baseline while "
            "remaining research-only and production_approved=false"
        )
    if final_status == "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA":
        return (
            f"lambda grid did not clearly beat baseline {baseline_hash}; "
            f"baseline_mae={baseline_mae}, best_hash={best_hash}, best_mae={best_mae}"
        )
    return f"{final_status}: no production-approved lambda selected"


def _settings_from_grid(
    grid: Mapping[str, object],
    *,
    settings: LambdaCalibrationSettings | None,
    smoke: bool,
) -> LambdaCalibrationSettings:
    base = settings or LambdaCalibrationSettings()
    raw = dict(grid.get("calibration", {}))
    if smoke:
        raw.update(dict(grid.get("smoke", {})))
    if not raw:
        return base
    payload = _settings_payload(base)
    payload.update(raw)
    if isinstance(payload.get("neighbor_markets"), str):
        payload["neighbor_markets"] = tuple(m.strip().upper() for m in str(payload["neighbor_markets"]).split(",") if m.strip())
    elif "neighbor_markets" in payload:
        payload["neighbor_markets"] = tuple(str(m).upper() for m in payload["neighbor_markets"])
    return LambdaCalibrationSettings(**payload)


def _config_payload(config: MonthlyCurveConfig | LambdaCandidateConfig | Mapping[str, object]) -> dict[str, object]:
    history_lookback_years: int | None = None
    if isinstance(config, LambdaCandidateConfig):
        history_lookback_years = config.history_lookback_years
        raw = dict(config.monthly_config.__dict__)
    elif isinstance(config, Mapping):
        raw = dict(config)
        history_lookback_years = raw.get("history_lookback_years")  # type: ignore[assignment]
    else:
        raw = dict(config.__dict__)
    keys = [
        "lambda_prior",
        "lambda_smooth_month",
        "lambda_smooth_yoy",
        "lambda_shape",
        "neighbor_shrinkage",
        "robust_panel_quantile",
        "min_history_snapshots",
        "max_prior_residual_eur_mwh",
        "constraint_tolerance",
        "stationarity_tolerance",
    ]
    payload = {key: raw.get(key) for key in keys}
    payload["history_lookback_years"] = history_lookback_years
    return payload


def _monthly_config(config: MonthlyCurveConfig | LambdaCandidateConfig) -> MonthlyCurveConfig:
    if isinstance(config, LambdaCandidateConfig):
        return config.monthly_config
    return config


def _candidate_history_lookback(
    config: MonthlyCurveConfig | LambdaCandidateConfig,
    settings: LambdaCalibrationSettings,
) -> int | None:
    if isinstance(config, LambdaCandidateConfig):
        return config.history_lookback_years
    return settings.history_lookback_years


def _settings_payload(settings: LambdaCalibrationSettings) -> dict[str, object]:
    payload = dict(settings.__dict__)
    payload["neighbor_markets"] = tuple(settings.neighbor_markets)
    return payload


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return float(f"{value:.12g}")
    if isinstance(value, (np.floating,)):
        return _canonicalize(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _counter_to_nested_counts(counter: Counter[tuple[str, str, str]]) -> dict[str, int]:
    return {"|".join(map(str, key)): int(value) for key, value in sorted(counter.items())}


def _validate_artifacts(
    scoring: pd.DataFrame,
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    candidate: Mapping[str, object] | None,
) -> None:
    final_status = str(summary.get("final_status"))
    if final_status not in FINAL_STATUSES:
        raise ValueError(f"unknown final_status {final_status!r}")
    if bool(summary.get("production_approved")) or bool(manifest.get("production_approved")):
        raise ValueError("lambda calibration artifacts must never be production-approved")
    missing = [col for col in SCORING_COLUMNS if col not in scoring.columns]
    if missing:
        raise ValueError(f"scoring artifact missing required columns: {missing}")
    if final_status in FAIL_STATUSES and candidate is not None:
        raise ValueError("FAIL status cannot produce a candidate config")
    if candidate is not None and bool(candidate.get("production_approved")):
        raise ValueError("candidate config must keep production_approved=false")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"
