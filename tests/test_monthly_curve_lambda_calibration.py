from __future__ import annotations

from collections import Counter
import hashlib
import json

import pandas as pd
import pytest

import pfc_shaping.calibration.monthly_curve_lambda_calibration as calibration_module
from pfc_shaping.calibration.monthly_curve_lambda_calibration import (
    LambdaCandidateConfig,
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
    _write_calibration_artifacts_core,
    _run_lambda_calibration_core,
    summarize_calibration,
    validate_feature_frame_point_in_time,
    validate_history_point_in_time,
    validate_masked_inputs,
    verify_calibration_artifact_set,
    write_calibration_artifacts,
)
from pfc_shaping.calibration.monthly_forward_curve import MarketQuote, MonthlyCurveConfig
from pfc_shaping.data.eex_historical_vintage import (
    EEX_HISTORICAL_QUOTE_SCHEMA,
    EexHistoricalVintageError,
    historical_vintage_revision_id,
    historical_vintage_row_hash,
    historical_vintage_snapshot_id,
)
from pfc_shaping.pipeline.monthly_curve_authority import solve_monthly_level_authority


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


def test_candidate_config_hash_includes_active_prior_stack_knobs():
    config = LambdaCandidateConfig(
        monthly_config=MonthlyCurveConfig(),
        history_lookback_years=6,
    )
    summary = {
        "final_status": "PASS_CALIBRATION_CANDIDATE_NOT_PRODUCTION_APPROVED",
        "selected_config_hash": config_hash(config),
    }

    base = build_candidate_config(
        pd.DataFrame(),
        summary=summary,
        configs=[config],
        settings=LambdaCalibrationSettings(),
        grid_hash="grid",
        source_data_hash="source",
        withheld_counts=Counter(),
        excluded_reasons=Counter(),
    )
    changed_weight = build_candidate_config(
        pd.DataFrame(),
        summary=summary,
        configs=[config],
        settings=LambdaCalibrationSettings(structural_weight=0.25),
        grid_hash="grid",
        source_data_hash="source",
        withheld_counts=Counter(),
        excluded_reasons=Counter(),
    )
    changed_fallback = build_candidate_config(
        pd.DataFrame(),
        summary=summary,
        configs=[config],
        settings=LambdaCalibrationSettings(
            allow_template_structural_fallback=False,
            structural_amplitude_eur_mwh=40.0,
            min_structural_snapshots=12,
        ),
        grid_hash="grid",
        source_data_hash="source",
        withheld_counts=Counter(),
        excluded_reasons=Counter(),
    )

    assert base is not None
    assert changed_weight is not None
    assert changed_fallback is not None
    assert base["config_hash"] != changed_weight["config_hash"]
    assert base["config_hash"] != changed_fallback["config_hash"]
    assert base["canonical_config"]["structural_weight"] == 1.0
    assert base["canonical_config"]["history_weight"] == 0.5


def test_candidate_config_hash_matches_monthly_authority_active_config_hash():
    settings = LambdaCalibrationSettings()
    config = LambdaCandidateConfig(
        monthly_config=MonthlyCurveConfig(),
        history_lookback_years=settings.history_lookback_years,
    )
    summary = {
        "final_status": "PASS_CALIBRATION_CANDIDATE_NOT_PRODUCTION_APPROVED",
        "selected_config_hash": config_hash(config),
    }

    candidate = build_candidate_config(
        pd.DataFrame(),
        summary=summary,
        configs=[config],
        settings=settings,
        grid_hash="grid",
        source_data_hash="source",
        withheld_counts=Counter(),
        excluded_reasons=Counter(),
    )

    authority = solve_monthly_level_authority(
        market="CH",
        delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
        own_base_prices={"2028": 80.0},
        all_market_base_prices={},
        eex_history=pd.DataFrame(columns=["date", "market", "load_type", "product", "price"]),
        run_timestamp=pd.Timestamp("2026-06-17"),
        settings={
            "enabled": True,
            "markets": list(settings.neighbor_markets),
            "history_lookback_years": settings.history_lookback_years,
            "min_structural_snapshots": settings.min_structural_snapshots,
            "allow_template_structural_fallback": settings.allow_template_structural_fallback,
            "structural_amplitude_eur_mwh": settings.structural_amplitude_eur_mwh,
            "panel_weight": settings.panel_weight,
            "history_weight": settings.history_weight,
            "structural_weight": settings.structural_weight,
        },
        allow_unverified_inputs=True,
    )

    assert candidate is not None
    assert candidate["config_hash"] == authority.manifest["active_config_hash"]


def test_scoring_of_synthetic_withheld_product_is_point_in_time_and_near_exact():
    history = _flat_history()
    grid = _small_grid()

    artifacts = _run_lambda_calibration_core(
        history,
        grid=grid,
        smoke=True,
        max_origins=1,
        max_configs=1,
        source_file_hash="fixture-hash",
        allow_unverified_vintage_fixture=True,
    )

    assert set(artifacts.scoring.columns) >= {"target_price", "predicted_price", "abs_error"}
    assert artifacts.scoring["abs_error"].max() < 0.002
    assert set(artifacts.scoring["withheld_horizon_bucket"]) == {"h+1"}
    assert artifacts.scoring["sample_size"].min() == 2
    assert artifacts.summary["production_approved"] is False
    assert artifacts.manifest["production_approved"] is False
    assert artifacts.summary["by_tenor_horizon"]
    assert artifacts.summary["train_deploy_gap"]["status"] == "UNSUPPORTED_NO_FAR_HORIZON_MONTHLY_TRUTH"


def test_calibration_api_requires_signed_vintage_catalog_by_default() -> None:
    with pytest.raises(EexHistoricalVintageError, match="signed eex_historical"):
        _run_lambda_calibration_core(
            _flat_history(),
            grid=_small_grid(),
            smoke=True,
            max_origins=1,
            max_configs=1,
            source_file_hash="fixture-hash",
        )


def test_fail_closed_if_history_is_insufficient():
    history = normalize_history(pd.DataFrame([_row("2026-03-01", "2027-01", 80.0)]))
    grid = _small_grid()

    artifacts = _run_lambda_calibration_core(
        history,
        grid=grid,
        smoke=True,
        max_origins=1,
        max_configs=1,
        source_file_hash="fixture-hash",
        allow_unverified_vintage_fixture=True,
    )

    assert artifacts.summary["final_status"] in {
        "UNSUPPORTED_INSUFFICIENT_HISTORY",
        "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS",
        "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA",
    }
    assert artifacts.summary["selected_config_hash"] is None
    assert artifacts.candidate_config is None


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


def test_partial_coverage_config_cannot_win_by_dropping_hard_cases() -> None:
    baseline = LambdaCandidateConfig(MonthlyCurveConfig(lambda_shape=0.0), 6)
    challenger = LambdaCandidateConfig(MonthlyCurveConfig(lambda_shape=1.0), 6)
    baseline_hash = config_hash(baseline)
    challenger_hash = config_hash(challenger)
    common = {
        "market": "CH",
        "withheld_load_type": "BASE",
        "withheld_tenor": "monthly",
        "withheld_horizon_bucket": "h+1",
        "constraint_residual_max": 0.0,
        "critical_gate_count": 0,
    }
    scoring = pd.DataFrame(
        [
            {**common, "config_hash": baseline_hash, "origin_date": "2026-01-01T00:00:00Z", "snapshot_id": "s1", "withheld_product": "2027-01", "abs_error": 1.0, "status": "PASS"},
            {**common, "config_hash": baseline_hash, "origin_date": "2026-02-01T00:00:00Z", "snapshot_id": "s2", "withheld_product": "2027-02", "abs_error": 1.0, "status": "PASS"},
            {**common, "config_hash": challenger_hash, "origin_date": "2026-01-01T00:00:00Z", "snapshot_id": "s1", "withheld_product": "2027-01", "abs_error": 0.1, "status": "PASS"},
            {**common, "config_hash": challenger_hash, "origin_date": "2026-02-01T00:00:00Z", "snapshot_id": "s2", "withheld_product": "2027-02", "abs_error": float("nan"), "status": "UNSUPPORTED"},
        ]
    )

    summary = summarize_calibration(
        scoring,
        configs=[baseline, challenger],
        settings=LambdaCalibrationSettings(
            min_valid_origins=2,
            min_withheld_monthly=2,
            min_withheld_quarterly=0,
        ),
        withheld_counts=Counter({("CH", "BASE", "monthly"): 2}),
        excluded_reasons=Counter(),
    )

    assert summary["selected_config_hash"] == baseline_hash
    assert summary["final_status"] == "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    eligibility = {
        row["config_hash"]: row["complete_case_eligible"]
        for row in summary["by_config"]
    }
    assert eligibility == {baseline_hash: True, challenger_hash: False}


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


def test_smoke_run_cannot_emit_candidate_config(tmp_path):
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
    artifacts = _run_lambda_calibration_core(
        history,
        grid=grid,
        smoke=True,
        max_origins=1,
        max_configs=1,
        allow_unverified_vintage_fixture=True,
    )

    assert artifacts.summary["selected_config_hash"] is None
    assert artifacts.summary["selected_mae"] is None
    assert artifacts.candidate_config is None
    assert artifacts.verified is False

    fresh = tmp_path / "fresh-output"
    with pytest.raises(EexHistoricalVintageError, match="only run_verified"):
        write_calibration_artifacts(artifacts, fresh)
    _write_calibration_artifacts_core(artifacts, fresh)
    inventory = json.loads(
        (fresh / "artifact_set_manifest.json").read_text(encoding="utf-8")
    )
    assert inventory["candidate_config_present"] is False
    assert not (fresh / "candidate_config.yaml").exists()
    (fresh / "calibration_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt mismatch"):
        verify_calibration_artifact_set(fresh)

    extra_dir = tmp_path / "extra-directory-output"
    _write_calibration_artifacts_core(artifacts, extra_dir)
    (extra_dir / "hidden").mkdir()
    with pytest.raises(ValueError, match="non-regular entries"):
        verify_calibration_artifact_set(extra_dir)

    incomplete = tmp_path / "incomplete-output"
    incomplete.mkdir()
    scoring_payload = ",".join(calibration_module.SCORING_COLUMNS).encode("utf-8")
    (incomplete / "scoring.csv").write_bytes(scoring_payload)
    (incomplete / "artifact_set_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "monthly_curve_calibration_artifact_set.v1",
                "final_status": "PASS_CALIBRATION_CANDIDATE_NOT_PRODUCTION_APPROVED",
                "candidate_config_present": False,
                "files": {
                    "scoring.csv": {
                        "sha256": hashlib.sha256(scoring_payload).hexdigest(),
                        "size_bytes": len(scoring_payload),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required role mismatch"):
        verify_calibration_artifact_set(incomplete)

    occupied = tmp_path / "occupied-output"
    occupied.mkdir()
    stale_candidate = occupied / "candidate_config.yaml"
    stale_candidate.write_text("stale: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _write_calibration_artifacts_core(artifacts, occupied)
    assert stale_candidate.read_text(encoding="utf-8") == "stale: true\n"


def test_governed_run_rejects_execution_environment_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {"git_worktree_dirty": False, "source_bundle_sha256": "before"}
    after = {"git_worktree_dirty": False, "source_bundle_sha256": "after"}
    monkeypatch.setattr(
        calibration_module,
        "_execution_environment_receipt",
        lambda: after,
    )

    with pytest.raises(EexHistoricalVintageError, match="changed during the run"):
        _run_lambda_calibration_core(
            _flat_history(),
            grid=_small_grid(),
            smoke=True,
            max_origins=1,
            max_configs=1,
            allow_unverified_vintage_fixture=True,
            expected_execution_environment=before,
        )


def test_failed_post_publish_replay_removes_canonical_output(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _run_lambda_calibration_core(
        _flat_history(),
        grid=_small_grid(),
        smoke=True,
        max_origins=1,
        max_configs=1,
        allow_unverified_vintage_fixture=True,
    )
    output = tmp_path / "failed-publication"
    monkeypatch.setattr(
        calibration_module,
        "verify_calibration_artifact_set",
        lambda _: (_ for _ in ()).throw(ValueError("forced replay failure")),
    )

    with pytest.raises(ValueError, match="forced replay failure"):
        _write_calibration_artifacts_core(artifacts, output)
    assert not output.exists()


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


def test_build_candidate_config_returns_none_for_unsupported_status():
    candidate = build_candidate_config(
        pd.DataFrame(),
        summary={
            "final_status": "UNSUPPORTED_INSUFFICIENT_HISTORY",
            "production_approved": False,
        },
        configs=[MonthlyCurveConfig()],
        settings=LambdaCalibrationSettings(),
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
    assert manifest["execution_environment"]["source_bundle_sha256"]
    assert isinstance(manifest["git_worktree_dirty"], bool)


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
    timestamp = pd.Timestamp(date, tz="UTC")
    source_document_sha256 = hashlib.sha256(date.encode("ascii")).hexdigest()
    product_type = (
        "MONTH"
        if len(product) == 7 and product[4] == "-" and product[5] != "Q"
        else "QUARTER"
        if "-Q" in product
        else "CAL"
    )
    source_column = (
        int(product[-2:]) + 20
        if product_type == "MONTH"
        else int(product[-1]) + 10
        if product_type == "QUARTER"
        else 1
    )
    row: dict[str, object] = {
        "date": pd.Timestamp(date),
        "observed_at": timestamp,
        "available_at": timestamp,
        "acquisition_id": f"fixture-{date}",
        "product": product,
        "load_type": "BASE",
        "product_type": product_type,
        "price": float(price),
        "market": market,
        "unit": "EUR/MWH",
        "source": "EEX",
        "source_document_id": f"eex-{date}.xlsx",
        "source_document_sha256": source_document_sha256,
        "source_sheet": market,
        "source_row_index": 4,
        "source_column_index": source_column,
        "source_product_code": product,
        "parser_version": "ingest_forwards.v1",
        "parser_code_sha256": "a" * 64,
        "parser_config_sha256": "b" * 64,
        "revision_sequence": 1,
        "supersedes_quote_id": "",
        "revision_timestamp": timestamp,
        "ingestion_run_id": f"fixture-{date}",
        "schema_version": EEX_HISTORICAL_QUOTE_SCHEMA,
    }
    row["snapshot_id"] = historical_vintage_snapshot_id(row)
    row["revision_id"] = historical_vintage_revision_id(row)
    row["row_hash"] = historical_vintage_row_hash(row)
    row["quote_id"] = row["row_hash"]
    return row


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
