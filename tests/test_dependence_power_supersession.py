from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.validation.dependence_power_supersession as supersession
import scripts.verify_ch_lt_dependence_power_preflight_supersession as cli_module


def _payload(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | str]:
    root = tmp_path / "repo"
    registry_relative = ".planning/phase14/supersession.json"
    old_relative = "output/diagnostic-v1.json"
    new_relative = "output/diagnostic-v2.json"
    fold_relative = "output/folds.csv"
    summary_relative = "output/source-summary.json"
    for relative in (
        registry_relative,
        old_relative,
        new_relative,
        fold_relative,
        summary_relative,
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)

    old_payload = _payload(
        {
            "schema_version": "ch_lt_dependence_power_preflight.v1",
            "scientific_admission": False,
            "execution_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
        }
    )
    new_payload = _payload(
        {
            "schema_version": "ch_lt_dependence_power_preflight.v2",
            "status": "LOCAL_DE_LU_INTRAHOUR_PLUGIN_SENSITIVITY_NO_GO",
            "scientific_admission": False,
            "execution_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
        }
    )
    fold_payload = b"origin_utc,loss\n2026-01-01T00:00:00+00:00,1.0\n"
    summary_payload = _payload({"schema_version": "intraday_shape_estimator_rolling_origin.v2"})
    old_sha = hashlib.sha256(old_payload).hexdigest()
    new_sha = hashlib.sha256(new_payload).hexdigest()
    fold_sha = hashlib.sha256(fold_payload).hexdigest()
    summary_sha = hashlib.sha256(summary_payload).hexdigest()
    (root / old_relative).write_bytes(old_payload)
    (root / new_relative).write_bytes(new_payload)
    (root / fold_relative).write_bytes(fold_payload)
    (root / summary_relative).write_bytes(summary_payload)

    monkeypatch.setattr(supersession, "REGISTRY_RELATIVE_PATH", registry_relative)
    monkeypatch.setattr(supersession, "SUPERSEDED_RELATIVE_PATH", old_relative)
    monkeypatch.setattr(supersession, "REPLACEMENT_RELATIVE_PATH", new_relative)
    monkeypatch.setattr(supersession, "FOLD_RELATIVE_PATH", fold_relative)
    monkeypatch.setattr(supersession, "SOURCE_SUMMARY_RELATIVE_PATH", summary_relative)
    monkeypatch.setattr(supersession, "SUPERSEDED_SHA256", old_sha)
    monkeypatch.setattr(supersession, "REPLACEMENT_SHA256", new_sha)
    monkeypatch.setattr(supersession, "FOLD_SHA256", fold_sha)
    monkeypatch.setattr(supersession, "SOURCE_SUMMARY_SHA256", summary_sha)

    registry = {
        "schema_version": supersession.REGISTRY_SCHEMA_VERSION,
        "status": supersession.REGISTRY_STATUS,
        "superseded": {
            "path": old_relative,
            "sha256": old_sha,
            "schema_version": "ch_lt_dependence_power_preflight.v1",
            "reason": "test supersession",
        },
        "replacement": {
            "path": new_relative,
            "sha256": new_sha,
            "schema_version": "ch_lt_dependence_power_preflight.v2",
            "status": "LOCAL_DE_LU_INTRAHOUR_PLUGIN_SENSITIVITY_NO_GO",
        },
        "forbidden_confirmatory_reuse": [
            {"role": "paired_fold_csv", "path": fold_relative, "sha256": fold_sha},
            {
                "role": "source_summary",
                "path": summary_relative,
                "sha256": summary_sha,
            },
        ],
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }
    registry_payload = _payload(registry)
    registry_path = root / registry_relative
    registry_path.write_bytes(registry_payload)
    return {
        "root": root,
        "registry": registry_path,
        "registry_sha": hashlib.sha256(registry_payload).hexdigest(),
        "old": root / old_relative,
        "old_sha": old_sha,
        "new": root / new_relative,
        "new_sha": new_sha,
    }


def _verify(evidence: dict[str, Path | str], *, current: bool) -> dict[str, object]:
    return supersession.verify_dependence_power_selection(
        repo_root=evidence["root"],
        registry_path=evidence["registry"],
        expected_registry_sha256=str(evidence["registry_sha"]),
        selected_artifact_path=evidence["new" if current else "old"],
        expected_selected_artifact_sha256=str(evidence["new_sha" if current else "old_sha"]),
    )


def test_exact_v2_is_current_but_never_scientifically_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _verify(_fixture(tmp_path, monkeypatch), current=True)

    assert result["status"] == supersession.CURRENT_STATUS
    assert result["local_diagnostic_selection_current"] is True
    assert result["scientific_admission"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False
    assert result["forbidden_confirmatory_reuse"] == [
        {"role": "paired_fold_csv", "sha256": supersession.FOLD_SHA256},
        {"role": "source_summary", "sha256": supersession.SOURCE_SUMMARY_SHA256},
    ]


def test_exact_v1_is_opposably_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _verify(_fixture(tmp_path, monkeypatch), current=False)

    assert result["status"] == supersession.STALE_STATUS
    assert result["local_diagnostic_selection_current"] is False
    assert result["rejection_reason"] == ("SELECTED_SCHEMA_AND_HASH_ARE_EXPLICITLY_SUPERSEDED")


def test_same_byte_shadow_path_is_not_a_current_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    shadow = Path(evidence["root"]) / "output" / "shadow-v2.json"
    shadow.write_bytes(Path(evidence["new"]).read_bytes())

    with pytest.raises(
        supersession.DependencePowerSupersessionError,
        match="neither canonical v1 nor canonical v2",
    ):
        supersession.verify_dependence_power_selection(
            repo_root=evidence["root"],
            registry_path=evidence["registry"],
            expected_registry_sha256=str(evidence["registry_sha"]),
            selected_artifact_path=shadow,
            expected_selected_artifact_sha256=str(evidence["new_sha"]),
        )


def test_denylist_omission_fails_even_when_registry_is_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    registry_path = Path(evidence["registry"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["forbidden_confirmatory_reuse"].pop()
    payload = _payload(registry)
    registry_path.write_bytes(payload)

    with pytest.raises(supersession.DependencePowerSupersessionError, match="denylist"):
        supersession.verify_dependence_power_selection(
            repo_root=evidence["root"],
            registry_path=registry_path,
            expected_registry_sha256=hashlib.sha256(payload).hexdigest(),
            selected_artifact_path=evidence["new"],
            expected_selected_artifact_sha256=str(evidence["new_sha"]),
        )


def test_each_registry_artifact_and_denylisted_input_is_captured_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    original = supersession.read_stable_single_link_file
    calls: list[str] = []

    def counted(path: str | Path, *, label: str, max_bytes: int | None = None) -> bytes:
        calls.append(str(path))
        return original(path, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(supersession, "read_stable_single_link_file", counted)

    _verify(evidence, current=True)

    assert len(calls) == 5
    assert len(set(calls)) == 5


def test_cli_current_is_zero_and_stale_is_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "__file__",
        str(Path(evidence["root"]) / "scripts" / "verify.py"),
    )
    common = [
        "--registry",
        str(evidence["registry"]),
        "--expected-registry-sha256",
        str(evidence["registry_sha"]),
    ]

    current_code = cli_module.main(
        [
            *common,
            "--selected-artifact",
            str(evidence["new"]),
            "--expected-selected-artifact-sha256",
            str(evidence["new_sha"]),
        ]
    )
    current = json.loads(capsys.readouterr().out)
    stale_code = cli_module.main(
        [
            *common,
            "--selected-artifact",
            str(evidence["old"]),
            "--expected-selected-artifact-sha256",
            str(evidence["old_sha"]),
        ]
    )
    stale = json.loads(capsys.readouterr().out)

    assert current_code == 0
    assert current["local_diagnostic_selection_current"] is True
    assert stale_code == 3
    assert stale["local_diagnostic_selection_current"] is False
