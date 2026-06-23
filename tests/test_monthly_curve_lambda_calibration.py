from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from pfc_shaping.calibration.monthly_curve_lambda_calibration import (
    LambdaCalibrationSettings,
    WithheldProduct,
    build_calibration_manifest,
    build_candidate_config,
    config_hash,
    history_before_origin,
    horizon_bucket,
    load_lambda_grid,
    mask_quote_sets,
    normalize_history,
    product_horizon_years,
    quote_key,
    quote_set_for_origin,
    run_lambda_calibration,
    summarize_calibration,
    validate_feature_frame_point_in_time,
    validate_history_point_in_time,
    validate_masked_inputs,
)
from pfc_shaping.calibration.monthly_forward_curve import MarketQuote, MonthlyCurveConfig


def test_masking_removes_withheld_and_revealing_own_and_neighbor_quotes():
    origin = pd.Timestamp("2026-03-01")
    full = (
        _quote("CH", "2027-01", 80.0, origin),
        _quote("CH", "2027-Q1", 80.0, origin),
        _quote("CH", "2027", 80.0, origin),
        _quote("CH", "2027-02", 80.0, origin),
        _quote("DE", "2027-01", 81.0, origin),
        _quote("DE", "2027-Q1", 81.0, origin),
        _quote("DE", "2027-02", 81.0, origin),
    )
    withheld = WithheldProduct("CH", "2027-01", "BASE", 80.0, origin)

    masked = mask_quote_sets(full, withheld, neighbor_markets=("DE",))

    visible = {quote_key(q.market, q.load_type, q.product) for q in masked.visible_quote_set}
    assert quote_key("CH", "BASE", "2027-01") not in visible
    assert quote_key("CH", "BASE", "2027-Q1") not in visible
    assert quote_key("CH", "BASE", "2027") not in visible
    assert quote_key("DE", "BASE", "2027-01") not in visible
    assert quote_key("DE", "BASE", "2027-Q1") not in visible
    assert quote_key("CH", "BASE", "2027-02") in visible
    assert quote_key("DE", "BASE", "2027-02") in visible
    validate_masked_inputs(masked, withheld)


def test_validate_masked_inputs_fails_if_withheld_quote_reappears_in_own_quotes():
    origin = pd.Timestamp("2026-03-01")
    withheld = WithheldProduct("CH", "2027-01", "BASE", 80.0, origin)
    masked = mask_quote_sets((_quote("CH", "2027-02", 80.0, origin),), withheld, neighbor_markets=("DE",))
    leaked = masked.__class__(
        origin_date=masked.origin_date,
        full_quote_set=masked.full_quote_set,
        withheld_set=masked.withheld_set,
        visible_quote_set=masked.visible_quote_set,
        own_quotes=masked.own_quotes + (_quote("CH", "2027-01", 80.0, origin),),
        neighbor_quotes=masked.neighbor_quotes,
        removed_quote_keys=masked.removed_quote_keys,
        removed_reasons=masked.removed_reasons,
    )

    with pytest.raises(ValueError, match="masked quote leakage"):
        validate_masked_inputs(leaked, withheld)


def test_validate_masked_inputs_fails_if_withheld_quote_reappears_in_neighbor_quotes():
    origin = pd.Timestamp("2026-03-01")
    withheld = WithheldProduct("CH", "2027-01", "BASE", 80.0, origin)
    masked = mask_quote_sets((_quote("CH", "2027-02", 80.0, origin),), withheld, neighbor_markets=("DE",))
    leaked = masked.__class__(
        origin_date=masked.origin_date,
        full_quote_set=masked.full_quote_set,
        withheld_set=masked.withheld_set,
        visible_quote_set=masked.visible_quote_set,
        own_quotes=masked.own_quotes,
        neighbor_quotes=masked.neighbor_quotes + (_quote("DE", "2027-01", 81.0, origin),),
        removed_quote_keys=masked.removed_quote_keys,
        removed_reasons=masked.removed_reasons,
    )

    with pytest.raises(ValueError, match="masked quote leakage"):
        validate_masked_inputs(leaked, withheld)


def test_history_point_in_time_is_strictly_before_origin():
    history = normalize_history(pd.DataFrame([_row("2026-02-01", "2027", 80.0), _row("2026-03-01", "2027", 81.0)]))

    view = history_before_origin(history, pd.Timestamp("2026-03-01"), lookback_years=None)

    assert view["date"].max() == pd.Timestamp("2026-02-01")
    validate_history_point_in_time(view, pd.Timestamp("2026-03-01"))
    with pytest.raises(ValueError, match="history feature leakage"):
        validate_history_point_in_time(history, pd.Timestamp("2026-03-01"))


def test_pre_mask_feature_cache_with_withheld_or_future_data_fails():
    withheld = WithheldProduct("CH", "2027-01", "BASE", 80.0, pd.Timestamp("2026-03-01"))

    with pytest.raises(ValueError, match="withheld product"):
        validate_feature_frame_point_in_time(
            pd.DataFrame({"date": [pd.Timestamp("2026-02-01")], "product": ["2027-01"]}),
            origin=pd.Timestamp("2026-03-01"),
            withheld=withheld,
        )
    with pytest.raises(ValueError, match="same-origin or future"):
        validate_feature_frame_point_in_time(
            pd.DataFrame({"date": [pd.Timestamp("2026-03-01")], "product": ["2027-02"]}),
            origin=pd.Timestamp("2026-03-01"),
            withheld=withheld,
        )


def test_config_hash_is_reproducible_and_independent_of_yaml_key_order():
    left = {
        "lambda_shape": 1.0,
        "lambda_smooth_month": 0.1,
        "lambda_smooth_yoy": 0.0,
        "neighbor_shrinkage": 0.5,
        "lambda_prior": 1e-6,
        "robust_panel_quantile": 0.5,
        "min_history_snapshots": 24,
        "max_prior_residual_eur_mwh": None,
        "constraint_tolerance": 0.01,
        "stationarity_tolerance": 1e-7,
    }
    right = dict(reversed(list(left.items())))

    assert config_hash(left) == config_hash(right)


def test_config_hash_includes_structural_prior_knobs():
    base = {
        "lambda_shape": 1.0,
        "lambda_smooth_month": 0.1,
        "lambda_smooth_yoy": 0.0,
        "neighbor_shrinkage": 0.5,
        "lambda_prior": 1e-6,
        "min_history_snapshots": 24,
        "constraint_tolerance": 0.01,
        "stationarity_tolerance": 1e-7,
        "markets": ["DE"],
        "history_lookback_years": 6,
        "min_structural_snapshots": 24,
        "allow_template_structural_fallback": True,
        "structural_amplitude_eur_mwh": 110.0,
        "panel_weight": 1.0,
        "history_weight": 0.5,
        "structural_weight": 1.0,
    }

    assert config_hash(base) != config_hash(base | {"structural_amplitude_eur_mwh": 90.0})
    assert config_hash(base) != config_hash(base | {"structural_weight": 0.5})
    assert config_hash(base) != config_hash(base | {"allow_template_structural_fallback": False})


def test_scoring_of_synthetic_withheld_product_is_point_in_time_and_near_exact():
    history = _flat_history()
    grid = _small_grid()

    artifacts = run_lambda_calibration(
        history,
        grid=grid,
        smoke=True,
        max_origins=1,
        max_configs=1,
        source_file_hash="fixture-hash",
    )

    assert set(artifacts.scoring.columns) >= {"target_price", "predicted_price", "abs_error"}
    assert artifacts.scoring["abs_error"].max() < 0.002
    assert set(artifacts.scoring["withheld_horizon_bucket"]) == {"h+1"}
    assert artifacts.scoring["sample_size"].min() == 2
    assert artifacts.summary["production_approved"] is False
    assert artifacts.manifest["production_approved"] is False
    assert artifacts.summary["by_tenor_horizon"]
    assert artifacts.summary["train_deploy_gap"]["status"] == "UNSUPPORTED_NO_FAR_HORIZON_MONTHLY_TRUTH"


def test_fail_closed_if_history_is_insufficient():
    history = normalize_history(pd.DataFrame([_row("2026-03-01", "2027-01", 80.0)]))
    grid = _small_grid()

    artifacts = run_lambda_calibration(
        history,
        grid=grid,
        smoke=True,
        max_origins=1,
        max_configs=1,
        source_file_hash="fixture-hash",
    )

    assert artifacts.summary["final_status"] in {
        "UNSUPPORTED_INSUFFICIENT_HISTORY",
        "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS",
        "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA",
    }
    assert artifacts.candidate_config is not None
    assert artifacts.candidate_config["production_approved"] is False


def test_summary_does_not_pass_when_critical_gate_count_is_present():
    scoring = pd.DataFrame(
        {
            "config_hash": ["abc"],
            "origin_date": ["2026-03-01"],
            "withheld_tenor": ["monthly"],
            "withheld_product": ["2027-01"],
            "constraint_residual_max": [0.0],
            "critical_gate_count": [1],
            "abs_error": [1.0],
            "status": ["PASS"],
        }
    )

    summary = summarize_calibration(
        scoring,
        configs=[MonthlyCurveConfig()],
        settings=LambdaCalibrationSettings(min_valid_origins=1, min_withheld_monthly=1, min_withheld_quarterly=0),
        withheld_counts=Counter({("CH", "BASE", "monthly"): 1}),
        excluded_reasons=Counter(),
    )

    assert not str(summary["final_status"]).startswith("PASS")


def test_train_deploy_gap_requires_far_horizon_monthly_truth_not_only_quarters():
    scoring = pd.DataFrame(
        {
            "config_hash": ["abc"],
            "origin_date": ["2026-03-01"],
            "withheld_tenor": ["quarterly"],
            "withheld_horizon_bucket": ["h+2"],
            "withheld_product": ["2028-Q1"],
            "constraint_residual_max": [0.0],
            "critical_gate_count": [0],
            "abs_error": [1.0],
            "status": ["PASS"],
        }
    )

    summary = summarize_calibration(
        scoring,
        configs=[MonthlyCurveConfig()],
        settings=LambdaCalibrationSettings(min_valid_origins=1, min_withheld_monthly=0, min_withheld_quarterly=1),
        withheld_counts=Counter({("CH", "BASE", "quarterly"): 1}),
        excluded_reasons=Counter(),
    )

    assert summary["train_deploy_gap"]["status"] == "UNSUPPORTED_NO_FAR_HORIZON_MONTHLY_TRUTH"
    assert summary["train_deploy_gap"]["far_horizon_non_monthly_rows"] == 1


def test_summary_fails_on_hard_constraint_violation():
    scoring = pd.DataFrame(
        {
            "config_hash": ["abc"],
            "origin_date": ["2026-03-01"],
            "withheld_tenor": ["monthly"],
            "withheld_product": ["2027-01"],
            "constraint_residual_max": [1e-3],
            "critical_gate_count": [0],
            "abs_error": [1.0],
            "status": ["PASS"],
        }
    )

    summary = summarize_calibration(
        scoring,
        configs=[MonthlyCurveConfig()],
        settings=LambdaCalibrationSettings(
            min_valid_origins=1,
            min_withheld_monthly=1,
            min_withheld_quarterly=0,
            hard_constraint_tolerance=1e-8,
        ),
        withheld_counts=Counter({("CH", "BASE", "monthly"): 1}),
        excluded_reasons=Counter(),
    )

    assert summary["final_status"] == "FAIL_HARD_CONSTRAINT_VIOLATION"


def test_no_real_exploitable_data_cannot_select_lambda():
    scoring = pd.DataFrame(columns=["origin_date", "withheld_tenor", "withheld_product", "constraint_residual_max"])

    summary = summarize_calibration(
        scoring,
        configs=[MonthlyCurveConfig()],
        settings=LambdaCalibrationSettings(),
        withheld_counts=Counter(),
        excluded_reasons=Counter(),
    )

    assert summary["final_status"] == "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS"


def test_candidate_config_is_never_production_approved(tmp_path):
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        """
defaults:
  constraint_tolerance: 0.01
grid:
  lambda_smooth_month: [1.0]
  lambda_smooth_yoy: [0.0]
  lambda_shape: [0.0]
  neighbor_shrinkage: [0.5]
""",
        encoding="utf-8",
    )
    grid = load_lambda_grid(grid_path)
    history = _flat_history()
    artifacts = run_lambda_calibration(history, grid=grid, smoke=True, max_origins=1, max_configs=1)

    assert artifacts.candidate_config is not None
    assert artifacts.candidate_config["production_approved"] is False
    assert artifacts.candidate_config["baseline_comparison"]["by_tenor_horizon"]


def test_candidate_config_hash_uses_canonical_structural_payload(tmp_path):
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        """
defaults:
  constraint_tolerance: 0.01
calibration:
  neighbor_markets: [DE]
  min_history_snapshots: 1
  min_structural_snapshots: 1
  panel_weight: 1.0
  history_weight: 0.5
  structural_weight: 1.0
  allow_template_structural_fallback: true
  structural_amplitude_eur_mwh: 110.0
smoke:
  min_valid_origins: 1
  min_withheld_monthly: 1
  min_withheld_quarterly: 1
  max_withheld_per_origin: 2
grid:
  lambda_smooth_month: [1.0]
  lambda_smooth_yoy: [0.0]
  lambda_shape: [0.0]
  neighbor_shrinkage: [0.5]
  history_lookback_years: [6]
""",
        encoding="utf-8",
    )
    grid = load_lambda_grid(grid_path)
    artifacts = run_lambda_calibration(
        _flat_history(),
        grid=grid,
        smoke=True,
        max_origins=1,
        max_configs=1,
    )
    assert artifacts.candidate_config is not None
    canonical = artifacts.candidate_config["canonical_config"]

    assert artifacts.candidate_config["config_hash"] == config_hash(canonical)
    assert canonical["markets"] == ["DE"]
    assert canonical["allow_template_structural_fallback"] is True
    assert canonical["structural_amplitude_eur_mwh"] == 110.0
    assert canonical["structural_weight"] == 1.0


def test_product_horizon_bucket_helpers_expose_train_deploy_gap():
    assert product_horizon_years(pd.Timestamp("2026-06-17"), "2026-07") == 0
    assert horizon_bucket(product_horizon_years(pd.Timestamp("2026-06-17"), "2027-Q1")) == "h+1"
    assert horizon_bucket(product_horizon_years(pd.Timestamp("2026-06-17"), "2028-Q1")) == "h+2"
    assert horizon_bucket(product_horizon_years(pd.Timestamp("2026-06-17"), "2030")) == "h+3+"


def test_build_candidate_config_returns_none_for_fail_status():
    scoring = pd.DataFrame()
    summary = {"final_status": "FAIL_HARD_CONSTRAINT_VIOLATION", "production_approved": False}

    candidate = build_candidate_config(
        scoring,
        summary=summary,
        configs=[MonthlyCurveConfig()],
        grid_hash="grid",
        source_data_hash="source",
        withheld_counts=Counter(),
        excluded_reasons=Counter(),
    )

    assert candidate is None


def test_manifest_contains_required_fields_and_false_production_flag():
    history = _flat_history()
    scoring = pd.DataFrame({"origin_date": ["2026-03-01"]})
    summary = {"final_status": "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"}

    manifest = build_calibration_manifest(
        history=history,
        scoring=scoring,
        summary=summary,
        settings=LambdaCalibrationSettings(),
        grid=_small_grid(),
        source_file_hash="abc",
        command_line=("script.py", "--smoke"),
        input_parquet_path="data/eex_forwards_history.parquet",
        withheld_counts=Counter({("CH", "BASE", "monthly"): 1}),
        excluded_reasons=Counter(),
    )

    assert manifest["input_parquet_sha256"] == "abc"
    assert manifest["lambda_grid_hash"]
    assert manifest["production_approved"] is False


def test_masking_result_is_unchanged_by_same_origin_hidden_target_price():
    history = _flat_history()
    origin = pd.Timestamp("2026-03-01")
    full_a = quote_set_for_origin(history, origin)
    history_b = history.copy()
    idx = history_b["date"].eq(origin) & history_b["product"].eq("2027-01") & history_b["market"].eq("CH")
    history_b.loc[idx, "price"] = 999.0
    full_b = quote_set_for_origin(history_b, origin)
    withheld_a = WithheldProduct("CH", "2027-01", "BASE", 80.0, origin)
    withheld_b = WithheldProduct("CH", "2027-01", "BASE", 999.0, origin)

    masked_a = mask_quote_sets(full_a, withheld_a, neighbor_markets=("DE",))
    masked_b = mask_quote_sets(full_b, withheld_b, neighbor_markets=("DE",))

    visible_a = {(q.market, q.product, q.price) for q in masked_a.visible_quote_set}
    visible_b = {(q.market, q.product, q.price) for q in masked_b.visible_quote_set}
    assert visible_a == visible_b


def _quote(market: str, product: str, price: float, origin: pd.Timestamp) -> MarketQuote:
    return MarketQuote(
        market=market,
        product=product,
        load_type="BASE",
        price=price,
        snapshot_date=origin,
        available_at=origin,
        source="TEST",
    )


def _row(date: str, product: str, price: float, *, market: str = "CH") -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "product": product,
        "load_type": "BASE",
        "product_type": "Month" if len(product) == 7 and product[4] == "-" else "Cal",
        "price": float(price),
        "market": market,
        "source": "TEST",
    }


def _flat_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in ("2026-01-01", "2026-02-01", "2026-03-01"):
        for market in ("CH", "DE"):
            rows.append(_row(date, "2027", 80.0, market=market))
            for quarter in range(1, 5):
                rows.append(_row(date, f"2027-Q{quarter}", 80.0, market=market))
            for month in range(1, 13):
                rows.append(_row(date, f"2027-{month:02d}", 80.0, market=market))
    return normalize_history(pd.DataFrame(rows))


def _small_grid() -> dict[str, object]:
    return {
        "defaults": {"constraint_tolerance": 0.01, "min_history_snapshots": 1},
        "smoke": {
            "min_valid_origins": 1,
            "min_withheld_monthly": 1,
            "min_withheld_quarterly": 1,
            "min_history_snapshots": 1,
            "min_structural_snapshots": 1,
            "max_withheld_per_origin": 2,
        },
        "grid": {
            "lambda_smooth_month": [1.0],
            "lambda_smooth_yoy": [0.0],
            "lambda_shape": [0.0],
            "neighbor_shrinkage": [0.5],
        },
    }
