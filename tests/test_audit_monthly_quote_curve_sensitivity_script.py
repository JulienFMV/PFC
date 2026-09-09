from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.audit_monthly_quote_curve_sensitivity as sensitivity_script
from pfc_shaping.pipeline.monthly_curve_authority import (
    DEFAULT_MONTHLY_SOLVER_CONFIG,
    solve_monthly_level_authority_from_history,
)
from scripts.audit_monthly_quote_curve_sensitivity import (
    build_monthly_sensitivity_artifact,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    history_path = tmp_path / "forwards.parquet"
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-24"),
                "market": "CH",
                "load_type": "BASE",
                "product": "2028",
                "price": 80.40,
            },
            {
                "date": pd.Timestamp("2026-07-24"),
                "market": "CH",
                "load_type": "BASE",
                "product": "2028-Q1",
                "price": 109.97,
            },
        ]
    ).to_parquet(history_path, index=False)
    forwards_payload = history_path.read_bytes()
    settings = dict(DEFAULT_MONTHLY_SOLVER_CONFIG)
    settings.update({"enabled": True, "constraint_tolerance": 1e-9})
    authority = solve_monthly_level_authority_from_history(
        forwards_path=history_path,
        market="CH",
        delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
        settings=settings,
        timezone="Europe/Zurich",
        forwards_payload=forwards_payload,
    )
    manifest_path = tmp_path / "monthly_manifest.json"
    manifest_payload = (
        json.dumps(
            authority.manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_payload)
    return (
        manifest_path,
        history_path,
        hashlib.sha256(manifest_payload).hexdigest(),
        hashlib.sha256(forwards_payload).hexdigest(),
    )


def test_artifact_reconstructs_bound_solver_and_is_byte_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    manifest, forwards, manifest_hash, forwards_hash = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    left = build_monthly_sensitivity_artifact(
        monthly_manifest_path=manifest,
        expected_monthly_manifest_sha256=manifest_hash,
        forwards_path=forwards,
        expected_forwards_sha256=forwards_hash,
        output_dir=tmp_path / "left",
    )
    right = build_monthly_sensitivity_artifact(
        monthly_manifest_path=manifest,
        expected_monthly_manifest_sha256=manifest_hash,
        forwards_path=forwards,
        expected_forwards_sha256=forwards_hash,
        output_dir=tmp_path / "right",
    )

    assert left["status"] == "PASS_LOCAL_DETERMINISTIC_SENSITIVITY_NOT_PRODUCTION"
    assert left["production_authorization"] is False
    assert left["promotion_gate"] is False
    assert left["code"]["execution_attestation"] == (
        "NOT_PROVIDED_POST_IMPORT_OBSERVATION_ONLY"
    )
    assert left == right
    for name in ("quote_to_month_sensitivity.csv", "summary.json"):
        assert (tmp_path / "left" / name).read_bytes() == (
            tmp_path / "right" / name
        ).read_bytes()


def test_artifact_rejects_unbound_manifest_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    manifest, forwards, _, forwards_hash = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match="caller-held"):
        build_monthly_sensitivity_artifact(
            monthly_manifest_path=manifest,
            expected_monthly_manifest_sha256="0" * 64,
            forwards_path=forwards,
            expected_forwards_sha256=forwards_hash,
            output_dir=output,
        )

    assert not output.exists()


def test_artifact_crash_before_summary_never_publishes_partial_final_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    manifest, forwards, manifest_hash, forwards_hash = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "retryable"
    original_write = sensitivity_script.write_durable_exact_bytes
    call_count = 0

    def fail_before_summary(*args: object, **kwargs: object) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected crash before summary")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(
        sensitivity_script,
        "write_durable_exact_bytes",
        fail_before_summary,
    )
    with pytest.raises(OSError, match="injected crash"):
        build_monthly_sensitivity_artifact(
            monthly_manifest_path=manifest,
            expected_monthly_manifest_sha256=manifest_hash,
            forwards_path=forwards,
            expected_forwards_sha256=forwards_hash,
            output_dir=output,
        )

    assert not output.exists()
    residues = list(
        tmp_path.glob(".monthly-sensitivity-retryable-staging-*")
    )
    assert len(residues) == 1
    assert sorted(path.name for path in residues[0].iterdir()) == [
        "quote_to_month_sensitivity.csv"
    ]

    monkeypatch.setattr(
        sensitivity_script,
        "write_durable_exact_bytes",
        original_write,
    )
    result = build_monthly_sensitivity_artifact(
        monthly_manifest_path=manifest,
        expected_monthly_manifest_sha256=manifest_hash,
        forwards_path=forwards,
        expected_forwards_sha256=forwards_hash,
        output_dir=output,
    )

    assert result["status"] == "PASS_LOCAL_DETERMINISTIC_SENSITIVITY_NOT_PRODUCTION"
    assert sorted(path.name for path in output.iterdir()) == [
        "quote_to_month_sensitivity.csv",
        "summary.json",
    ]
