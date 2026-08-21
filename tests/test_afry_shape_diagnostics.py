from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pfc_shaping.data.afry_scenarios import canonical_json_bytes, file_sha256
from pfc_shaping.validation import afry_shape_diagnostics as diagnostics_module
from pfc_shaping.validation.afry_shape_diagnostics import (
    DECISION_STATE,
    DIAGNOSTIC_STATUS,
    AfryShapeDiagnosticError,
    build_shape_diagnostic_bundle,
    build_shape_diagnostic_frames,
    verify_shape_diagnostic_bundle,
)


@pytest.fixture(autouse=True)
def _bounded_test_runtime_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostics_module,
        "runtime_tree_receipt",
        lambda _prefix: {
            "schema_version": "fmv-afry-runtime-tree-receipt-v1",
            "tree_sha256": "a" * 64,
            "file_count": 1,
            "total_bytes": 1,
            "plain_file_tree": True,
            "stable_during_materialization": True,
        },
    )


def _slot_values(spread: float) -> list[float]:
    values = [0.0] * 24
    for slot in diagnostics_module.CENTRAL_BLOCK_SLOTS:
        values[slot] = -spread / 2.0
    for slot in diagnostics_module.LATE_BLOCK_SLOTS:
        values[slot] = spread / 2.0
    offset = sum(values) / 24.0
    return [value - offset for value in values]


def _inputs(
    *,
    scenarios: tuple[str, ...] = ("Central",),
    delivery_years: tuple[int, ...] = (2027, 2030),
    weather_years: tuple[int, ...] = (2012, 2014),
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    profile_sheets = []
    for scenario_index, scenario in enumerate(scenarios):
        groups = []
        for delivery_index, delivery_year in enumerate(delivery_years):
            for weather_index, weather_year in enumerate(weather_years):
                spread = 10.0 + scenario_index * 20.0 + delivery_index * 6.0 + weather_index * 2.0
                values = _slot_values(spread)
                for slot, value in enumerate(values):
                    rows.append(
                        {
                            "release_id": "AFRY_CH_2026_Q2_V100",
                            "scenario_vendor": scenario,
                            "delivery_year": delivery_year,
                            "weather_year": weather_year,
                            "representative_hour_slot": slot,
                            "annual_zero_mean_additive_shape_eur_mwh": value,
                            "model_input_authorized": False,
                            "monthly_level_authority": False,
                            "production_authorized": False,
                        }
                    )
                negative = 100 + delivery_index * 10 + weather_index
                strict_negative = negative + 2
                groups.append(
                    {
                        "delivery_year": delivery_year,
                        "weather_year": weather_year,
                        "rows": 8760,
                        "negative_prices": negative,
                        "strict_negative_prices": strict_negative,
                        "near_zero_negative_noise_count": 2,
                    }
                )
        profile_sheets.append(
            {
                "scenario_vendor": scenario,
                "groups": groups,
            }
        )
    return pd.DataFrame(rows), {
        "release_id": "AFRY_CH_2026_Q2_V100",
        "weather_years": list(weather_years),
        "representative_hours_per_group": 8760,
        "sheets": profile_sheets,
    }


def test_diagnostics_preserve_scenarios_and_measure_term_change_without_calendar() -> None:
    shape, profile = _inputs(scenarios=("Central", "High"))

    group, weather, term, audit = build_shape_diagnostic_frames(shape, profile)

    assert len(group) == 8
    assert len(weather) == 4
    assert len(term) == 4
    central_2027 = weather[
        (weather["scenario_vendor"] == "Central") & (weather["delivery_year"] == 2027)
    ].iloc[0]
    assert central_2027["duck_spread_min_eur_mwh"] == pytest.approx(10.0)
    assert central_2027["duck_spread_median_eur_mwh"] == pytest.approx(11.0)
    assert central_2027["duck_spread_max_eur_mwh"] == pytest.approx(12.0)
    central_term = term[
        (term["scenario_vendor"] == "Central") & (term["weather_year"] == 2012)
    ].iloc[0]
    assert central_term["delivery_year_gap"] == 3
    assert central_term["duck_spread_change_eur_mwh"] == pytest.approx(6.0)
    assert central_term["duck_spread_change_per_year_eur_mwh"] == pytest.approx(2.0)
    assert set(group["scenario_vendor"]) == {"Central", "High"}
    assert not group["calendar_mapping_authorized"].any()
    assert not weather["probability_weighting_authorized"].any()
    assert audit["slot_labels_are_clock_hours"] is False
    assert audit["external_inputs_consumed"] == []
    assert (
        audit["qualitative_vendor_output"]["interpretation"]
        == "DESCRIPTIVE_VENDOR_OUTPUT_NOT_EMPIRICAL_OR_CAUSAL_VALIDATION"
    )
    assert not audit["qualitative_vendor_output"]["by_scenario"]["Central"][
        "deepening_then_recompression_all_weather_patterns"
    ]
    assert (
        audit["empirical_validation_status"]
        == "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS"
    )


def test_diagnostics_reject_missing_representative_slot() -> None:
    shape, profile = _inputs()
    shape = shape.drop(index=shape.index[shape["representative_hour_slot"] == 23][0])

    with pytest.raises(AfryShapeDiagnosticError, match="exact slots 0..23"):
        build_shape_diagnostic_frames(shape, profile)


def test_diagnostics_reject_input_authority_flip() -> None:
    shape, profile = _inputs()
    shape.loc[0, "model_input_authorized"] = True

    with pytest.raises(AfryShapeDiagnosticError, match="model_input_authorized"):
        build_shape_diagnostic_frames(shape, profile)


def test_diagnostics_reject_profile_group_mismatch() -> None:
    shape, profile = _inputs()
    profile["sheets"][0]["groups"].pop()

    with pytest.raises(AfryShapeDiagnosticError, match="coverage does not match"):
        build_shape_diagnostic_frames(shape, profile)


def test_diagnostics_reject_inconsistent_negative_noise_count() -> None:
    shape, profile = _inputs()
    profile["sheets"][0]["groups"][0]["near_zero_negative_noise_count"] = 3

    with pytest.raises(AfryShapeDiagnosticError, match="negative-price counts"):
        build_shape_diagnostic_frames(shape, profile)


def test_diagnostics_are_invariant_to_unused_annual_price_level() -> None:
    shape, profile = _inputs()
    shape["annual_mean_eur_mwh_real_release_basis"] = 50.0
    first = build_shape_diagnostic_frames(shape, profile)[0]
    shape["annual_mean_eur_mwh_real_release_basis"] = 5_000.0
    second = build_shape_diagnostic_frames(shape, profile)[0]

    pd.testing.assert_frame_equal(first, second)


def test_diagnostics_require_every_declared_weather_pattern() -> None:
    shape, profile = _inputs()
    key = (
        (shape["delivery_year"] == 2030)
        & (shape["weather_year"] == 2014)
        & (shape["scenario_vendor"] == "Central")
    )
    shape = shape.loc[~key].copy()
    profile["sheets"][0]["groups"] = [
        group
        for group in profile["sheets"][0]["groups"]
        if not (group["delivery_year"] == 2030 and group["weather_year"] == 2014)
    ]

    with pytest.raises(AfryShapeDiagnosticError, match="every declared weather pattern"):
        build_shape_diagnostic_frames(shape, profile)


def _write_fake_source_catalog(
    source_bundle: Path,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    source_bundle.mkdir()
    shape, profile = _inputs()
    shape_path = source_bundle / "hour_slot_shape_benchmark.parquet"
    profile_path = source_bundle / "hourly_profile.json"
    pq.write_table(pa.Table.from_pandas(shape, preserve_index=False), shape_path)
    profile_path.write_bytes(canonical_json_bytes(profile))
    (source_bundle / "catalog.json").write_bytes(b"{}\n")
    artifacts = []
    for path in (shape_path, profile_path):
        artifacts.append(
            {
                "name": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    catalog = {
        "bundle_id": "f" * 64,
        "integrity_verdict": "PASS_LOCAL_INTEGRITY_ONLY",
        "decision_state": DECISION_STATE,
        "admission": {"model_input_authorized": False},
        "artifacts": artifacts,
    }
    return catalog, shape, profile


def test_bundle_is_deterministic_closed_and_keeps_external_data_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    output_root = tmp_path / "diagnostics"

    first = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=output_root,
    )
    second = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=output_root,
    )
    manifest = verify_shape_diagnostic_bundle(first, source_catalog_bundle=source_bundle)

    assert first == second
    assert manifest["status"] == DIAGNOSTIC_STATUS
    assert manifest["decision_state"] == DECISION_STATE
    assert manifest["admission"]["model_input_authorized"] is False
    assert manifest["admission"]["production_authorized"] is False
    assert manifest["contract"]["path"].endswith("SHAPE-DIAGNOSTIC-CONTRACT-V1.json")
    assert manifest["contract"]["stable_during_build"] is True
    assert {item["role"] for item in manifest["producer"]["files"]} == {
        "diagnostic_module",
        "diagnostic_entrypoint",
        "source_catalog_module",
    }
    assert manifest["producer"]["runtime"]["dependency_closure"]["tree_sha256"] == (
        "a" * 64
    )
    assert (
        manifest["admission"]["empirical_validation_status"]
        == "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS"
    )
    assert {path.name for path in first.iterdir()} == {
        "group_shape_diagnostics.parquet",
        "term_shape_changes.parquet",
        "weather_shape_sensitivity.parquet",
        "diagnostic-audit.json",
        "diagnostic-manifest.json",
        "diagnostic-manifest.sha256",
    }
    with pytest.raises(AfryShapeDiagnosticError, match="caller anchor is required"):
        verify_shape_diagnostic_bundle(first)


def test_bundle_verifier_rejects_self_consistent_model_authority_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    bundle = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=tmp_path / "diagnostics",
    )
    manifest_path = bundle / "diagnostic-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["admission"]["model_input_authorized"] = True
    body = dict(manifest)
    body.pop("bundle_id")
    new_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    manifest["bundle_id"] = new_id
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (bundle / "diagnostic-manifest.sha256").write_bytes(
        f"{file_sha256(manifest_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryShapeDiagnosticError, match="admission invariants"):
        verify_shape_diagnostic_bundle(rewritten, source_catalog_bundle=source_bundle)


def test_bundle_verifier_rejects_self_consistent_parquet_row_count_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    bundle = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=tmp_path / "diagnostics",
    )
    manifest_path = bundle / "diagnostic-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group_artifact = next(
        item
        for item in manifest["artifacts"]
        if item["name"] == "group_shape_diagnostics.parquet"
    )
    group_artifact["rows"] += 1
    body = dict(manifest)
    body.pop("bundle_id")
    new_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    manifest["bundle_id"] = new_id
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (bundle / "diagnostic-manifest.sha256").write_bytes(
        f"{file_sha256(manifest_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryShapeDiagnosticError, match="Parquet row count mismatch"):
        verify_shape_diagnostic_bundle(rewritten, source_catalog_bundle=source_bundle)


def test_builder_rejects_source_output_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)

    with pytest.raises(AfryShapeDiagnosticError, match="must not overlap"):
        build_shape_diagnostic_bundle(
            workspace=Path.cwd(),
            source_catalog_bundle=source_bundle,
            output_root=source_bundle / "diagnostics",
        )


def test_builder_rejects_noncanonical_workspace(tmp_path: Path) -> None:
    with pytest.raises(AfryShapeDiagnosticError, match="workspace must be exactly"):
        build_shape_diagnostic_bundle(
            workspace=tmp_path,
            source_catalog_bundle=tmp_path / "source",
            output_root=tmp_path / "output",
        )


def test_workspace_guard_rejects_observed_cwd_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics_module, "_current_directory", lambda: tmp_path)

    with pytest.raises(AfryShapeDiagnosticError, match="diagnostic cwd"):
        diagnostics_module._require_canonical_workspace(Path.cwd())


def test_workspace_guard_rejects_observed_git_root_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics_module, "_git_top_level", lambda _workspace: tmp_path)

    with pytest.raises(AfryShapeDiagnosticError, match="diagnostic Git root"):
        diagnostics_module._require_canonical_workspace(Path.cwd())


def test_bundle_verifier_rejects_manifest_contract_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    bundle = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=tmp_path / "diagnostics",
    )
    manifest_path = bundle / "diagnostic-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract"]["sha256"] = "0" * 64
    body = dict(manifest)
    body.pop("bundle_id")
    new_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    manifest["bundle_id"] = new_id
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (bundle / "diagnostic-manifest.sha256").write_bytes(
        f"{file_sha256(manifest_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryShapeDiagnosticError, match="canonical workspace"):
        verify_shape_diagnostic_bundle(rewritten, source_catalog_bundle=source_bundle)


def test_bundle_verifier_rejects_self_consistent_out_of_range_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    bundle = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=tmp_path / "diagnostics",
    )
    parquet_path = bundle / "group_shape_diagnostics.parquet"
    frame = pq.read_table(parquet_path).to_pandas()
    frame.loc[0, "negative_price_share"] = 1.5
    diagnostics_module._write_frame(frame, parquet_path, diagnostics_module.GROUP_SCHEMA)
    manifest_path = bundle / "diagnostic-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["name"] == "group_shape_diagnostics.parquet"
    )
    artifact["bytes"] = parquet_path.stat().st_size
    artifact["sha256"] = file_sha256(parquet_path)
    body = dict(manifest)
    body.pop("bundle_id")
    new_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    manifest["bundle_id"] = new_id
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (bundle / "diagnostic-manifest.sha256").write_bytes(
        f"{file_sha256(manifest_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryShapeDiagnosticError, match=r"outside \[0, 1\]"):
        verify_shape_diagnostic_bundle(rewritten, source_catalog_bundle=source_bundle)


def test_bundle_verifier_rejects_plausible_but_forged_derived_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    bundle = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=tmp_path / "diagnostics",
    )
    parquet_path = bundle / "group_shape_diagnostics.parquet"
    frame = pq.read_table(parquet_path).to_pandas()
    frame.loc[0, "late_minus_central_block_eur_mwh"] += 0.01
    diagnostics_module._write_frame(frame, parquet_path, diagnostics_module.GROUP_SCHEMA)
    manifest_path = bundle / "diagnostic-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["name"] == "group_shape_diagnostics.parquet"
    )
    artifact["bytes"] = parquet_path.stat().st_size
    artifact["sha256"] = file_sha256(parquet_path)
    body = dict(manifest)
    body.pop("bundle_id")
    new_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    manifest["bundle_id"] = new_id
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (bundle / "diagnostic-manifest.sha256").write_bytes(
        f"{file_sha256(manifest_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryShapeDiagnosticError, match="deterministic replay mismatch"):
        verify_shape_diagnostic_bundle(rewritten, source_catalog_bundle=source_bundle)


def test_builder_adopts_identical_concurrent_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    original_replace = diagnostics_module._replace_directory_once
    raced = False

    def concurrent_replace(source: Path, target: Path) -> None:
        nonlocal raced
        if not raced and source.name.startswith(".staging-"):
            raced = True
            shutil.copytree(source, target)
            raise PermissionError("simulated identical concurrent publisher")
        original_replace(source, target)

    monkeypatch.setattr(diagnostics_module, "_replace_directory_once", concurrent_replace)
    monkeypatch.setattr(diagnostics_module.time, "sleep", lambda _seconds: None)

    bundle = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=tmp_path / "diagnostics",
    )

    assert raced is True
    assert (
        verify_shape_diagnostic_bundle(bundle, source_catalog_bundle=source_bundle)["status"]
        == DIAGNOSTIC_STATUS
    )


def test_builder_quarantines_poisoned_existing_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = tmp_path / "source" / ("f" * 64)
    source_bundle.parent.mkdir()
    catalog, _, _ = _write_fake_source_catalog(source_bundle)
    monkeypatch.setattr(diagnostics_module, "verify_catalog_bundle", lambda _path: catalog)
    output_root = tmp_path / "diagnostics"
    bundle = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=output_root,
    )
    (bundle / "diagnostic-manifest.sha256").write_bytes(b"poisoned\n")

    rebuilt = build_shape_diagnostic_bundle(
        workspace=Path.cwd(),
        source_catalog_bundle=source_bundle,
        output_root=output_root,
    )

    quarantines = list(output_root.glob(f".quarantine-{bundle.name}-*"))
    assert rebuilt == bundle
    assert len(quarantines) == 1
    assert (
        verify_shape_diagnostic_bundle(rebuilt, source_catalog_bundle=source_bundle)["status"]
        == DIAGNOSTIC_STATUS
    )


def test_contract_forbids_legacy_or_synthetic_eex_entsoe_substitution() -> None:
    path = Path(
        ".planning/phases/14-lt-audit-remediation/"
        "AFRY-CH-2026-Q2-SHAPE-DIAGNOSTIC-CONTRACT-V1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))

    assert contract["external_dependencies"]["eex_databricks"].startswith("PENDING_")
    assert contract["external_dependencies"]["entsoe_databricks"].startswith("PENDING_")
    assert (
        contract["external_dependencies"]["legacy_local_eex_or_entsoe_substitution_authorized"]
        is False
    )
    assert contract["external_dependencies"]["synthetic_substitution_authorized"] is False


def test_future_agents_are_routed_to_current_afry_governance() -> None:
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    handoff = Path(".planning/HANDOFF.md").read_text(encoding="utf-8")
    decision_log = Path(
        ".planning/phases/14-lt-audit-remediation/DECISION-LOG.md"
    ).read_text(encoding="utf-8")
    context_name = "AFRY-CH-2026-Q2-AGENT-DATA-CONTEXT.md"
    session_name = "SESSION-HANDOFF-20260803-AFRY-SHAPE-DIAGNOSTICS-V3.md"
    blocker = "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS"

    assert context_name in agents
    assert blocker in agents
    assert "Legacy local and synthetic EEX/ENTSO-E substitution are forbidden" in agents
    assert "T057" in agents and "remains sealed" in agents
    assert blocker in handoff
    assert session_name in handoff
    assert "D-20260803-207" in decision_log
    assert "No model training, selection, candidate assembly" in decision_log


def test_publish_directory_retries_transient_windows_handle_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    calls = 0

    def transient_replace(observed_source: Path, observed_target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient Windows handle")
        observed_source.replace(observed_target)

    monkeypatch.setattr(diagnostics_module, "_replace_directory_once", transient_replace)
    monkeypatch.setattr(diagnostics_module.time, "sleep", lambda _seconds: None)

    diagnostics_module._publish_directory(source, target)

    assert calls == 3
    assert target.is_dir()
    assert not source.exists()


def test_stable_file_reader_rejects_windows_reparse_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b"{}\n")
    monkeypatch.setattr(
        diagnostics_module,
        "_is_link_or_reparse",
        lambda observed, _path_stat: observed == path,
    )

    with pytest.raises(AfryShapeDiagnosticError, match="not a plain file"):
        diagnostics_module._stable_file_bytes(
            path,
            label="test artifact",
            maximum_bytes=1024,
        )


def test_workspace_ancestry_rejects_reparse_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "child"
    original = diagnostics_module._is_link_or_reparse
    monkeypatch.setattr(
        diagnostics_module,
        "_is_link_or_reparse",
        lambda observed, path_stat: observed == tmp_path or original(observed, path_stat),
    )

    with pytest.raises(AfryShapeDiagnosticError, match="ancestry contains a reparse point"):
        diagnostics_module._require_plain_workspace_ancestry(
            Path.cwd(),
            target,
            label="test ancestry",
        )
