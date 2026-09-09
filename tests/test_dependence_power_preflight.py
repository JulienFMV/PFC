from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import pfc_shaping.validation.dependence_power_preflight as preflight_module
from pfc_shaping.validation.dependence_power_preflight import (
    SCHEMA_VERSION,
    STATUS,
    DependencePowerPreflightError,
    audit_dependence_power_pilot,
)
from scripts.audit_ch_lt_dependence_power_preflight import (
    _canonical_output,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "candidate"
BENCHMARK = "benchmark"


def _evidence(
    tmp_path: Path,
    *,
    origin_count: int = 32,
    cadence_days: int = 14,
    production_authorization: object = False,
    promotion_gate: object = False,
    ompex_used: object = False,
    authority_status: object = "COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO",
    delta_pattern: tuple[float, ...] = (0.35, -0.10, 0.20, 0.00, 0.30, -0.05),
) -> tuple[Path, str, Path, str]:
    start = datetime(2025, 9, 1, tzinfo=timezone.utc)
    fieldnames = [
        "origin_utc",
        "test_end_utc",
        "model",
        "n_quarter_hours",
        "mae_eur_mwh",
    ]
    rows: list[dict[str, object]] = []
    candidate_losses: list[float] = []
    benchmark_losses: list[float] = []
    for index in range(origin_count):
        origin = start + timedelta(days=cadence_days * index)
        test_end = origin + timedelta(days=14)
        benchmark_loss = 10.0 + (index % 5) * 0.25
        delta = delta_pattern[index % len(delta_pattern)]
        candidate_loss = benchmark_loss - delta
        benchmark_losses.append(benchmark_loss)
        candidate_losses.append(candidate_loss)
        for model, loss in (
            (CANDIDATE, candidate_loss),
            (BENCHMARK, benchmark_loss),
        ):
            rows.append(
                {
                    "origin_utc": origin.isoformat(),
                    "test_end_utc": test_end.isoformat(),
                    "model": model,
                    "n_quarter_hours": 1344,
                    "mae_eur_mwh": loss,
                }
            )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    fold_payload = buffer.getvalue().encode("utf-8")
    fold_path = tmp_path / "fold_metrics.csv"
    fold_path.write_bytes(fold_payload)
    fold_sha = hashlib.sha256(fold_payload).hexdigest()

    summary = {
        "schema_version": "intraday_shape_estimator_rolling_origin.v2",
        "status": "LOCAL_DIAGNOSTIC_CANDIDATE_REJECTED_NOT_PRODUCTION",
        "production_authorization": production_authorization,
        "promotion_gate": promotion_gate,
        "ompex_used": ompex_used,
        "input": {"authority_status": authority_status},
        "design": {
            "fold_days": 14,
            "target": "quarter_hour_price_given_realized_parent_hour_mean",
        },
        "fold_count": origin_count,
        "metrics": {
            CANDIDATE: {"mae_eur_mwh": sum(candidate_losses) / origin_count},
            BENCHMARK: {"mae_eur_mwh": sum(benchmark_losses) / origin_count},
        },
        "artifacts": {"fold_metrics_csv_sha256": fold_sha},
    }
    summary_payload = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_bytes(summary_payload)
    summary_sha = hashlib.sha256(summary_payload).hexdigest()
    return fold_path, fold_sha, summary_path, summary_sha


def _audit(evidence: tuple[Path, str, Path, str]) -> dict[str, object]:
    fold_path, fold_sha, summary_path, summary_sha = evidence
    return audit_dependence_power_pilot(
        fold_csv_path=fold_path,
        expected_fold_csv_sha256=fold_sha,
        source_summary_path=summary_path,
        expected_source_summary_sha256=summary_sha,
        candidate_model=CANDIDATE,
        benchmark_model=BENCHMARK,
    )


def _rewrite_summary(summary_path: Path, mutate: Callable[[dict[str, object]], None]) -> str:
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    mutate(document)
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_valid_pilot_is_quantified_but_never_authorizes_science(tmp_path: Path) -> None:
    result = _audit(_evidence(tmp_path))

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == STATUS
    assert result["scientific_admission"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False
    assert result["future_holdout_consumed"] is False
    assert result["t057_consumed"] is False
    comparison = result["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["origin_count"] == 32
    dependence = result["dependence"]
    assert isinstance(dependence, dict)
    assert 1.0 <= dependence["observed_plugin_effective_origin_count"] <= 32.0
    assert dependence["is_upper_confidence_bound"] is False
    power = result["power_design"]
    assert isinstance(power, dict)
    assert power["hypothesis_family_sizes"] == [1, 4, 8]
    assert power["interpretation"] == (
        "EXPLORATORY_POST_SELECTION_DE_LU_PLUGIN_SENSITIVITY_NOT_CH_ACQUISITION_PLAN"
    )
    assert (
        power["delete_one_plugin_required_origins_max"]
        >= power["observed_plugin_required_origins_max"]
    )
    assert power["ch_acquisition_requirement_supported"] is False
    assert power["ch_acquisition_requirement_origins"] is None
    assert power["pilot_reuse_for_confirmatory_test_forbidden"] is True
    assert "POST_HOC_SELECTED_CANDIDATE_PILOT" in result["blockers"]
    assert "TARGET_NOT_FULL_CH_LT" in result["blockers"]


def test_overlapping_test_windows_are_rejected(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, cadence_days=7)

    with pytest.raises(DependencePowerPreflightError, match="overlap"):
        _audit(evidence)


def test_window_duration_must_match_bound_source_design(tmp_path: Path) -> None:
    fold_path, _, summary_path, summary_sha = _evidence(tmp_path)
    lines = fold_path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    origin = datetime.fromisoformat(cells[0])
    cells[1] = (origin + timedelta(minutes=15)).isoformat()
    cells[3] = "1"
    lines[1] = ",".join(cells)
    fold_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    fold_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()

    with pytest.raises(DependencePowerPreflightError, match="fold duration"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_observation_count_must_match_full_window(tmp_path: Path) -> None:
    fold_path, _, summary_path, summary_sha = _evidence(tmp_path)
    text = fold_path.read_text(encoding="utf-8").replace("1344,", "1,", 1)
    fold_path.write_text(text, encoding="utf-8", newline="")
    fold_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()

    with pytest.raises(DependencePowerPreflightError, match="observation count"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_source_target_cannot_be_relabelled_as_full_ch_lt(tmp_path: Path) -> None:
    fold_path, fold_sha, summary_path, _ = _evidence(tmp_path)

    def relabel(document: dict[str, object]) -> None:
        design = document["design"]
        assert isinstance(design, dict)
        design["target"] = "DELIVERED_CH_LT_QUARTER_HOURLY_PRICE"

    summary_sha = _rewrite_summary(summary_path, relabel)

    with pytest.raises(DependencePowerPreflightError, match="target is unsupported"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_degenerate_paired_variance_is_invalid_not_required_n_two(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path, delta_pattern=(0.0,))

    with pytest.raises(DependencePowerPreflightError, match="variance is degenerate"):
        _audit(evidence)


def test_less_than_eight_origins_is_rejected(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, origin_count=7)

    with pytest.raises(DependencePowerPreflightError, match="eight distinct"):
        _audit(evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("production_authorization", True, "authorizes production"),
        ("promotion_gate", True, "authorizes promotion"),
        ("ompex_used", True, "OMPEX"),
        ("authority_status", "PRODUCTION_READY", "authority is overstated"),
    ],
)
def test_source_cannot_be_relabelled_as_stronger_authority(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    kwargs = {field: value}
    evidence = _evidence(tmp_path, **kwargs)

    with pytest.raises(DependencePowerPreflightError, match=message):
        _audit(evidence)


def test_bool_cannot_impersonate_integer_fold_count(tmp_path: Path) -> None:
    fold_path, fold_sha, summary_path, _ = _evidence(tmp_path)
    summary_sha = _rewrite_summary(
        summary_path, lambda document: document.__setitem__("fold_count", True)
    )

    with pytest.raises(DependencePowerPreflightError, match="must be an integer"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_unbounded_json_integer_is_normalized_to_invalid_input(tmp_path: Path) -> None:
    fold_path, fold_sha, summary_path, _ = _evidence(tmp_path)

    def overflow_loss(document: dict[str, object]) -> None:
        metrics = document["metrics"]
        assert isinstance(metrics, dict)
        candidate = metrics[CANDIDATE]
        assert isinstance(candidate, dict)
        candidate["mae_eur_mwh"] = 10**400

    summary_sha = _rewrite_summary(summary_path, overflow_loss)

    with pytest.raises(DependencePowerPreflightError, match="invalid"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_summary_must_bind_exact_fold_bytes(tmp_path: Path) -> None:
    fold_path, fold_sha, summary_path, _ = _evidence(tmp_path)
    summary_sha = _rewrite_summary(
        summary_path,
        lambda document: document["artifacts"].__setitem__("fold_metrics_csv_sha256", "0" * 64),
    )

    with pytest.raises(DependencePowerPreflightError, match="does not bind"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_fractional_observation_count_is_rejected(tmp_path: Path) -> None:
    fold_path, _, summary_path, summary_sha = _evidence(tmp_path)
    text = fold_path.read_text(encoding="utf-8").replace("1344,", "1344.0,", 1)
    fold_path.write_text(text, encoding="utf-8", newline="")
    fold_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()

    with pytest.raises(DependencePowerPreflightError, match="exact integer"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_nonfinite_loss_is_rejected(tmp_path: Path) -> None:
    fold_path, _, summary_path, summary_sha = _evidence(tmp_path)
    lines = fold_path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    cells[-1] = "nan"
    lines[1] = ",".join(cells)
    fold_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    fold_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()

    with pytest.raises(DependencePowerPreflightError, match="must be finite"):
        _audit((fold_path, fold_sha, summary_path, summary_sha))


def test_candidate_and_benchmark_must_differ(tmp_path: Path) -> None:
    fold_path, fold_sha, summary_path, summary_sha = _evidence(tmp_path)

    with pytest.raises(DependencePowerPreflightError, match="must differ"):
        audit_dependence_power_pilot(
            fold_csv_path=fold_path,
            expected_fold_csv_sha256=fold_sha,
            source_summary_path=summary_path,
            expected_source_summary_sha256=summary_sha,
            candidate_model=CANDIDATE,
            benchmark_model=CANDIDATE,
        )


def test_each_source_file_is_captured_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _evidence(tmp_path)
    original = preflight_module.read_stable_single_link_file
    calls: list[str] = []

    def counted(path: str | Path, *, label: str, max_bytes: int | None = None) -> bytes:
        calls.append(str(path))
        return original(path, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(preflight_module, "read_stable_single_link_file", counted)

    _audit(evidence)

    assert calls == [str(evidence[0]), str(evidence[2])]


def test_cli_validate_diagnostic_reports_valid_no_go_pilot_as_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fold_path, fold_sha, summary_path, summary_sha = _evidence(tmp_path)

    code = main(
        [
            "--fold-csv",
            str(fold_path),
            "--expected-fold-csv-sha256",
            fold_sha,
            "--source-summary",
            str(summary_path),
            "--expected-source-summary-sha256",
            summary_sha,
            "--candidate-model",
            CANDIDATE,
            "--benchmark-model",
            BENCHMARK,
            "--mode",
            "validate-diagnostic",
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == STATUS
    assert result["execution_authorized"] is False


def test_cli_default_scientific_admission_mode_exits_no_go(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fold_path, fold_sha, summary_path, summary_sha = _evidence(tmp_path)

    code = main(
        [
            "--fold-csv",
            str(fold_path),
            "--expected-fold-csv-sha256",
            fold_sha,
            "--source-summary",
            str(summary_path),
            "--expected-source-summary-sha256",
            summary_sha,
            "--candidate-model",
            CANDIDATE,
            "--benchmark-model",
            BENCHMARK,
        ]
    )

    assert code == 3
    result = json.loads(capsys.readouterr().out)
    assert result["scientific_admission"] is False


def test_cli_hash_failure_is_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fold_path, _, summary_path, summary_sha = _evidence(tmp_path)

    code = main(
        [
            "--fold-csv",
            str(fold_path),
            "--expected-fold-csv-sha256",
            "0" * 64,
            "--source-summary",
            str(summary_path),
            "--expected-source-summary-sha256",
            summary_sha,
            "--candidate-model",
            CANDIDATE,
            "--benchmark-model",
            BENCHMARK,
        ]
    )

    assert code == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    result = json.loads(streams.err)
    assert result["status"] == "INVALID_DEPENDENCE_POWER_PREFLIGHT"
    assert result["execution_authorized"] is False


def test_output_is_restricted_to_canonical_namespace() -> None:
    with pytest.raises(ValueError, match="canonical namespace"):
        _canonical_output(ROOT / "arbitrary-power-preflight.json")
