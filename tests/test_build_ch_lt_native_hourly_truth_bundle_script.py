from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation import ch_lt_native_hourly_truth_bundle as builder

ROOT = Path(r"C:\Users\jbattaglia\PFC_LT")


@pytest.fixture
def governed_tmp_path() -> Path:
    selected = (
        ROOT
        / "build"
        / "test-workspaces"
        / "native-hourly-truth-bundle"
        / uuid.uuid4().hex
    )
    selected.mkdir(parents=True, exist_ok=False)
    return selected


def _synthetic_truth_document() -> dict[str, object]:
    return {
        "schema_version": "ch_lt_native_hourly_truth_bundle.v1",
        "status": "COMPLETE_LOCAL_NATIVE_HOURLY_TRUTH_NO_GO",
        "test_fixture": False,
        "market": "CH",
        "target_id": "synthetic-2026-08",
        "delivery_month": "2026-08",
        "delivery_end_utc": "2026-08-31T22:00:00Z",
        "native_hourly_cadence": True,
        "quarter_hour_proxy": False,
        "native_hourly_interval_count": 1,
        "truth_opened_at_utc_untrusted": "2026-09-01T00:00:00Z",
        "source_ledger": {"synthetic": True},
        "prediction_commitment": {
            "sha256": "a" * 64,
            "commitment_id": "b" * 64,
        },
        "future_origin_selection": {
            "path": str(builder.origin_selection_auditor.REGISTRY_RELATIVE).replace(
                "\\", "/"
            ),
            "sha256": builder.origin_selection_auditor.REGISTRY_SHA256,
            "selection_id": (
                "5340bb8c5cd0c2ba65f346c70e75ba7aa6bb17a5553df353f476ef8c50f03984"
            ),
        },
        "observations": [],
        "outcomes_opened": True,
        "truth_publication_receipt_present": True,
        "prospective_truth_consumed": True,
        "future_holdout_consumed": True,
        "t057_consumed": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
        "truth_bundle_id": "c" * 64,
    }


def test_extracts_only_exact_stepwise_native_hours() -> None:
    index = pd.date_range("2026-08-01T00:00:00Z", periods=12, freq="15min")
    frame = pd.DataFrame(
        {"price_eur_mwh": [10.0] * 4 + [-2.0] * 4 + [35.5] * 4},
        index=index,
    )

    native = builder.extract_native_hourly_prices(frame, label="synthetic")

    assert list(native.index) == list(
        pd.date_range("2026-08-01T00:00:00Z", periods=3, freq="h")
    )
    assert native.tolist() == [10.0, -2.0, 35.5]


def test_rejects_non_stepwise_quarter_hour_values() -> None:
    index = pd.date_range("2026-08-01T00:00:00Z", periods=4, freq="15min")
    frame = pd.DataFrame({"price_eur_mwh": [10.0, 10.0, 11.0, 10.0]}, index=index)

    with pytest.raises(
        builder.NativeHourlyTruthBundleError,
        match="not an exact stepwise native-hourly transport proxy",
    ):
        builder.extract_native_hourly_prices(frame, label="synthetic")


def test_current_ledger_fails_closed_before_first_target_truth_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builder,
        "_local_wall_clock_utc",
        lambda: pd.Timestamp("2026-09-01T00:00:01Z"),
    )
    with pytest.raises(
        builder.NativeHourlyTruthBundleError,
        match="does not cover the complete committed target",
    ):
        builder.build_truth_bundle(
            ROOT,
            ledger_path=Path(
                "build/prospective-ledgers/ch-hourly-local-ledger-20260731-v10.json"
            ),
            expected_ledger_sha256=(
                "3805b71f1368a0e742c8627d4995a85c791de1360257f97b78be0b1665723140"
            ),
            commitment_path=Path(
                "build/ch-lt-structural-prediction-commitments/"
                "structural-dry-run-20260731T072027Z-v7-quant-contract/commitment.json"
            ),
            expected_commitment_sha256=(
                "67cc6fe722586cd4a02149252d1af7fbbb35e90a89e8bc7965f8d0bf87eda9b4"
            ),
            delivery_month="2026-08",
            truth_opened_at_utc_untrusted="2026-09-01T00:00:00Z",
        )


def test_future_declared_open_cannot_reach_ledger_before_wall_clock_maturity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builder,
        "_local_wall_clock_utc",
        lambda: pd.Timestamp("2026-07-31T08:30:00Z"),
    )
    labels: list[str] = []
    original = builder._bound_json

    def _recording_bound_json(*args, label: str, **kwargs):
        labels.append(label)
        return original(*args, label=label, **kwargs)

    monkeypatch.setattr(builder, "_bound_json", _recording_bound_json)

    with pytest.raises(
        builder.NativeHourlyTruthBundleError,
        match="local wall clock has not reached target maturity",
    ):
        builder.build_truth_bundle(
            ROOT,
            ledger_path=Path("build/never-read-ledger.json"),
            expected_ledger_sha256="0" * 64,
            commitment_path=Path(
                "build/ch-lt-structural-prediction-commitments/"
                "structural-dry-run-20260731T072027Z-v7-quant-contract/commitment.json"
            ),
            expected_commitment_sha256=(
                "67cc6fe722586cd4a02149252d1af7fbbb35e90a89e8bc7965f8d0bf87eda9b4"
            ),
            delivery_month="2026-08",
            truth_opened_at_utc_untrusted="2026-09-01T00:00:00Z",
        )

    assert "prospective capture ledger" not in labels


def test_publishes_bundle_and_receipt_as_one_atomic_directory(
    governed_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _synthetic_truth_document()
    monkeypatch.setattr(builder, "build_truth_bundle", lambda *args, **kwargs: document)
    output = governed_tmp_path / "published"

    result = builder.publish_truth_bundle(ROOT, output)

    assert result == document
    assert {path.name for path in output.iterdir()} == {
        "truth-bundle.json",
        "truth-publication-receipt.json",
    }
    bundle_payload = (output / "truth-bundle.json").read_bytes()
    assert bundle_payload == builder._canonical(document)
    receipt = json.loads(
        (output / "truth-publication-receipt.json").read_text("ascii")
    )
    assert receipt["truth_bundle"]["sha256"] == hashlib.sha256(
        bundle_payload
    ).hexdigest()
    assert receipt["delivery_end_utc"] == document["delivery_end_utc"]
    assert all(value is False for value in receipt["authority_boundary"].values())


def test_resumes_exact_partial_staging_without_publishing_partial_directory(
    governed_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _synthetic_truth_document()
    monkeypatch.setattr(builder, "build_truth_bundle", lambda *args, **kwargs: document)
    output = governed_tmp_path / "resumed"
    staging = governed_tmp_path / ".resumed.staging"
    staging.mkdir()
    (staging / "truth-bundle.json").write_bytes(builder._canonical(document))

    builder.publish_truth_bundle(ROOT, output)

    assert output.is_dir()
    assert not staging.exists()
    assert {path.name for path in output.iterdir()} == {
        "truth-bundle.json",
        "truth-publication-receipt.json",
    }


def test_rejects_divergent_staging_residue(
    governed_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        builder,
        "build_truth_bundle",
        lambda *args, **kwargs: _synthetic_truth_document(),
    )
    output = governed_tmp_path / "divergent"
    staging = governed_tmp_path / ".divergent.staging"
    staging.mkdir()
    (staging / "truth-bundle.json").write_bytes(b"divergent\n")

    with pytest.raises(
        builder.NativeHourlyTruthBundleError,
        match="differs",
    ):
        builder.publish_truth_bundle(ROOT, output)

    assert not output.exists()
