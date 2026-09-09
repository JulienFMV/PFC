from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.cli import score_ch_lt_structural_prediction_commitment as score_cli
from pfc_shaping.validation import ch_lt_prospective_hourly_scoring as scoring
from pfc_shaping.validation.ch_lt_prospective_hourly_scoring import (
    ProspectiveHourlyScoringError,
    aggregate_exact_quarter_hours,
    canonical_score_bytes,
    score_hourly_prediction,
    score_structural_commitment,
)

ROOT = Path(r"C:\Users\jbattaglia\PFC_LT")
COMMITMENT = Path(
    "build/ch-lt-structural-prediction-commitments/"
    "structural-dry-run-20260731T072027Z-v7-quant-contract/commitment.json"
)
COMMITMENT_SHA256 = "67cc6fe722586cd4a02149252d1af7fbbb35e90a89e8bc7965f8d0bf87eda9b4"
PREDICTIONS = COMMITMENT.with_name("predictions.json")
PREDICTIONS_SHA256 = "859c4622eb5b699f22242af0121141729d6c00472b6b9070bdb87c2ba9132302"
SCORING_CONTRACT = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-FUTURE-HOURLY-SCORING-CONTRACT-V2-20260731.json"
)
SCORING_CONTRACT_SHA256 = (
    "d84d7ebc2f3c6aa4b820fab26f0a5de18940a6d35af8e796323025eab694ea4d"
)


@pytest.fixture
def governed_tmp_path() -> Path:
    selected = (
        ROOT
        / "build"
        / "test-workspaces"
        / "prospective-hourly-scoring"
        / uuid.uuid4().hex
    )
    selected.mkdir(parents=True, exist_ok=False)
    return selected


@pytest.fixture(autouse=True)
def matured_local_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scoring,
        "_local_wall_clock_utc",
        lambda: pd.Timestamp("2026-09-01T00:00:01Z"),
    )


def _august_central_hourly() -> tuple[dict[str, object], pd.Series]:
    inventory = json.loads(
        (
            ROOT
            / "build/ch-lt-origin-target-mask-inventories/"
            "structural-dry-run-20260731T072027Z-schema-v2.json"
        ).read_text(encoding="utf-8")
    )
    target = inventory["targets"][0]
    start = pd.Timestamp(target["delivery_start_utc"])
    end = pd.Timestamp(target["delivery_end_utc"])
    frame = pd.read_parquet(
        ROOT / "build/local-model-runs/run31n1c/local-pfc-smoke_central.parquet"
    )
    quarter_hours = frame.loc[
        (frame.index >= start) & (frame.index < end), "price_shape"
    ]
    hourly = aggregate_exact_quarter_hours(
        quarter_hours,
        start_utc=start,
        end_utc=end,
        expected_hour_count=target["expected_native_hourly_interval_count"],
        label="test central",
    )
    return target, hourly


def _truth_fixture(path: Path, *, omit_last: bool = False) -> str:
    target, hourly = _august_central_hourly()
    observations = [
        {
            "delivery_start_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "price_eur_mwh": format(float(price), ".17g"),
            "available_at_utc": target["delivery_end_utc"],
        }
        for timestamp, price in hourly.items()
    ]
    if omit_last:
        observations.pop()
    document = {
        "schema_version": "ch_lt_native_hourly_truth_bundle.v1",
        "status": "TEST_FIXTURE_COMPLETE_NATIVE_HOURLY_TRUTH_NO_GO",
        "test_fixture": True,
        "market": "CH",
        "target_id": target["target_id"],
        "delivery_month": target["delivery_month"],
        "delivery_end_utc": target["delivery_end_utc"],
        "native_hourly_cadence": True,
        "quarter_hour_proxy": False,
        "native_hourly_interval_count": target["expected_native_hourly_interval_count"],
        "truth_opened_at_utc_untrusted": target["delivery_end_utc"],
        "observations": observations,
        "outcomes_opened": True,
        "truth_publication_receipt_present": True,
        "prospective_truth_consumed": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    payload = canonical_score_bytes(document)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _truth_publication_receipt(path: Path, truth_path: Path, truth_sha256: str) -> str:
    target, _ = _august_central_hourly()
    core = {
        "schema_version": "ch_lt_local_truth_publication_receipt.v1",
        "status": "TEST_FIXTURE_LOCAL_DIAGNOSTIC_TRUTH_PUBLICATION_RECEIPT_NO_GO",
        "test_fixture": True,
        "target_id": target["target_id"],
        "delivery_month": target["delivery_month"],
        "delivery_end_utc": target["delivery_end_utc"],
        "truth_opened_at_utc_untrusted": target["delivery_end_utc"],
        "local_diagnostic_truth_read_occurred": True,
        "truth_bundle": {
            "path": truth_path.relative_to(ROOT).as_posix(),
            "sha256": truth_sha256,
        },
        "prediction_commitment": {
            "sha256": COMMITMENT_SHA256,
            "commitment_id": (
                "799ed71cdeddb9bf81fdc63574f352174bd2ac432c83008a25c63b4cf4094d2a"
            ),
        },
        "hourly_scoring_contract": {
            "sha256": SCORING_CONTRACT_SHA256,
            "contract_id": (
                "ae85de71a3745b4ea8b9704090f9bd742e26a41161c2d0b804f4888fb372653e"
            ),
        },
        "future_origin_selection": {
            "sha256": (
                "11966d1ee85ace46e97006fa74f8aab4789c71f7438de909ed71541aab480df7"
            ),
            "selection_id": (
                "5340bb8c5cd0c2ba65f346c70e75ba7aa6bb17a5553df353f476ef8c50f03984"
            ),
        },
        "authority_boundary": {
            "independent_trusted_time_receipt_present": False,
            "independent_signature_present": False,
            "builder_inaccessible_external_cas_receipt_present": False,
            "independent_registry_receipt_present": False,
            "scientific_truth_open_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
        },
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    receipt_id = hashlib.sha256(canonical_score_bytes(core)).hexdigest()
    payload = canonical_score_bytes({**core, "receipt_id": receipt_id})
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_scores_bound_commitment_and_keeps_level_shape_separate(
    governed_tmp_path: Path,
) -> None:
    truth_path = governed_tmp_path / "synthetic-truth.json"
    truth_sha256 = _truth_fixture(truth_path)
    receipt_path = governed_tmp_path / "truth-publication-receipt.json"
    receipt_sha256 = _truth_publication_receipt(
        receipt_path, truth_path, truth_sha256
    )

    result = score_structural_commitment(
        ROOT,
        commitment_path=COMMITMENT,
        expected_commitment_sha256=COMMITMENT_SHA256,
        predictions_path=PREDICTIONS,
        expected_predictions_sha256=PREDICTIONS_SHA256,
        scoring_contract_path=SCORING_CONTRACT,
        expected_scoring_contract_sha256=SCORING_CONTRACT_SHA256,
        truth_publication_receipt_path=receipt_path,
        expected_truth_publication_receipt_sha256=receipt_sha256,
        delivery_month="2026-08",
        allow_test_fixture=True,
    )

    assert result["status"] == "LOCAL_PROSPECTIVE_SCORING_DIAGNOSTIC_NO_GO"
    assert result["native_hourly_interval_count"] == 744
    assert result["scenario_interpretation"] == (
        "STRUCTURAL_SENSITIVITY_NOT_PROBABILISTIC_COVERAGE"
    )
    assert result["bindings"]["hourly_scoring_contract"]["sha256"] == (
        SCORING_CONTRACT_SHA256
    )
    assert [item["scenario"] for item in result["scores"]] == [
        "slow",
        "central",
        "fast",
    ]
    central = result["scores"][1]
    assert float(central["monthly_mean_error_eur_mwh"]) == pytest.approx(0.0, abs=1e-12)
    assert float(central["zero_mean_hourly_shape"]["mae_eur_mwh"]) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(central["zero_mean_hourly_shape"]["rmse_eur_mwh"]) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(central["zero_mean_hourly_shape"]["pearson_correlation"]) == (
        pytest.approx(1.0, abs=1e-12)
    )
    assert result["future_holdout_consumed"] is False
    assert result["t057_consumed"] is False
    assert result["scientific_admission"] is False
    assert result["production_authorization"] is False


def test_fixture_truth_is_forbidden_by_default(governed_tmp_path: Path) -> None:
    truth_path = governed_tmp_path / "synthetic-truth.json"
    truth_sha256 = _truth_fixture(truth_path)
    receipt_path = governed_tmp_path / "truth-publication-receipt.json"
    receipt_sha256 = _truth_publication_receipt(
        receipt_path, truth_path, truth_sha256
    )

    with pytest.raises(ProspectiveHourlyScoringError, match="fixture is forbidden"):
        score_structural_commitment(
            ROOT,
            commitment_path=COMMITMENT,
            expected_commitment_sha256=COMMITMENT_SHA256,
            predictions_path=PREDICTIONS,
            expected_predictions_sha256=PREDICTIONS_SHA256,
            scoring_contract_path=SCORING_CONTRACT,
            expected_scoring_contract_sha256=SCORING_CONTRACT_SHA256,
            truth_publication_receipt_path=receipt_path,
            expected_truth_publication_receipt_sha256=receipt_sha256,
            delivery_month="2026-08",
        )


def test_incomplete_truth_never_scores_as_zero(governed_tmp_path: Path) -> None:
    truth_path = governed_tmp_path / "incomplete-truth.json"
    truth_sha256 = _truth_fixture(truth_path, omit_last=True)
    receipt_path = governed_tmp_path / "truth-publication-receipt.json"
    receipt_sha256 = _truth_publication_receipt(
        receipt_path, truth_path, truth_sha256
    )

    with pytest.raises(ProspectiveHourlyScoringError, match="incomplete"):
        score_structural_commitment(
            ROOT,
            commitment_path=COMMITMENT,
            expected_commitment_sha256=COMMITMENT_SHA256,
            predictions_path=PREDICTIONS,
            expected_predictions_sha256=PREDICTIONS_SHA256,
            scoring_contract_path=SCORING_CONTRACT,
            expected_scoring_contract_sha256=SCORING_CONTRACT_SHA256,
            truth_publication_receipt_path=receipt_path,
            expected_truth_publication_receipt_sha256=receipt_sha256,
            delivery_month="2026-08",
            allow_test_fixture=True,
        )


def test_zero_variance_correlation_is_explicitly_unsupported() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=24, freq="h")
    constant = pd.Series(50.0, index=index)

    score = score_hourly_prediction(constant, constant, label="constant")

    assert score["zero_mean_hourly_shape"]["pearson_correlation"] is None
    assert score["zero_mean_hourly_shape"]["pearson_correlation_status"] == (
        "UNSUPPORTED_ZERO_VARIANCE"
    )


def test_parallel_level_shift_does_not_create_shape_error() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=24, freq="h")
    truth = pd.Series([float(hour - 12) for hour in range(24)], index=index)
    prediction = truth + 17.5

    score = score_hourly_prediction(prediction, truth, label="parallel shift")

    assert float(score["monthly_mean_error_eur_mwh"]) == pytest.approx(17.5)
    assert float(score["zero_mean_hourly_shape"]["mae_eur_mwh"]) == pytest.approx(0.0)
    assert float(score["functional_errors"]["midday_premium_error_eur_mwh"]) == (
        pytest.approx(0.0)
    )
    assert float(score["functional_errors"]["evening_premium_error_eur_mwh"]) == (
        pytest.approx(0.0)
    )


def test_fixture_status_biconditional_cannot_be_bypassed(
    governed_tmp_path: Path,
) -> None:
    truth_path = governed_tmp_path / "synthetic-truth.json"
    _truth_fixture(truth_path)
    document = json.loads(truth_path.read_text(encoding="utf-8"))
    document["test_fixture"] = False
    target, _ = _august_central_hourly()

    with pytest.raises(ProspectiveHourlyScoringError, match="fixture/status/schema"):
        scoring._truth_series(
            document,
            target_id=target["target_id"],
            delivery_month="2026-08",
            start_utc=pd.Timestamp(target["delivery_start_utc"]),
            end_utc=pd.Timestamp(target["delivery_end_utc"]),
            expected_count=target["expected_native_hourly_interval_count"],
            allow_test_fixture=False,
        )


def test_invalid_pre_outcome_chain_never_reads_truth_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    original = scoring._read_bound_json

    def observed_reader(*args, **kwargs):
        labels.append(str(kwargs.get("label")))
        return original(*args, **kwargs)

    monkeypatch.setattr(scoring, "_read_bound_json", observed_reader)
    with pytest.raises(
        ProspectiveHourlyScoringError,
        match="canonically selected future origin chain",
    ):
        score_structural_commitment(
            ROOT,
            commitment_path=COMMITMENT,
            expected_commitment_sha256="0" * 64,
            predictions_path=PREDICTIONS,
            expected_predictions_sha256=PREDICTIONS_SHA256,
            scoring_contract_path=SCORING_CONTRACT,
            expected_scoring_contract_sha256=SCORING_CONTRACT_SHA256,
            truth_publication_receipt_path=Path("build/never-read-receipt.json"),
            expected_truth_publication_receipt_sha256="0" * 64,
            delivery_month="2026-08",
        )

    assert "native hourly truth bundle" not in labels
    assert "truth publication receipt" not in labels


def test_real_future_dated_receipt_cannot_open_truth_before_wall_clock_maturity(
    governed_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = governed_tmp_path / "synthetic-truth.json"
    truth_sha256 = _truth_fixture(truth_path)
    receipt_path = governed_tmp_path / "truth-publication-receipt.json"
    _truth_publication_receipt(receipt_path, truth_path, truth_sha256)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "LOCAL_DIAGNOSTIC_TRUTH_PUBLICATION_RECEIPT_NO_GO"
    receipt["test_fixture"] = False
    receipt.pop("receipt_id")
    receipt["receipt_id"] = hashlib.sha256(canonical_score_bytes(receipt)).hexdigest()
    receipt_payload = canonical_score_bytes(receipt)
    receipt_path.write_bytes(receipt_payload)
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    labels: list[str] = []
    original = scoring._read_bound_json

    def observed_reader(*args, **kwargs):
        labels.append(str(kwargs.get("label")))
        return original(*args, **kwargs)

    monkeypatch.setattr(scoring, "_read_bound_json", observed_reader)
    monkeypatch.setattr(
        scoring,
        "_local_wall_clock_utc",
        lambda: pd.Timestamp("2026-07-31T12:00:00Z"),
    )
    with pytest.raises(ProspectiveHourlyScoringError, match="not reached target maturity"):
        score_structural_commitment(
            ROOT,
            commitment_path=COMMITMENT,
            expected_commitment_sha256=COMMITMENT_SHA256,
            predictions_path=PREDICTIONS,
            expected_predictions_sha256=PREDICTIONS_SHA256,
            scoring_contract_path=SCORING_CONTRACT,
            expected_scoring_contract_sha256=SCORING_CONTRACT_SHA256,
            truth_publication_receipt_path=receipt_path,
            expected_truth_publication_receipt_sha256=receipt_sha256,
            delivery_month="2026-08",
        )

    assert "truth publication receipt" in labels
    assert "native hourly truth bundle" not in labels


def test_fixture_receipt_cannot_reclassify_real_truth_bundle(
    governed_tmp_path: Path,
) -> None:
    truth_path = governed_tmp_path / "synthetic-truth.json"
    truth_sha256 = _truth_fixture(truth_path)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["test_fixture"] = False
    truth["status"] = "COMPLETE_LOCAL_NATIVE_HOURLY_TRUTH_NO_GO"
    truth["prospective_truth_consumed"] = True
    truth["future_holdout_consumed"] = True
    truth["source_ledger"] = {"path": "build/never-read.json"}
    truth["prediction_commitment"] = {"sha256": COMMITMENT_SHA256}
    truth["future_origin_selection"] = {"sha256": "0" * 64}
    truth_core = dict(truth)
    truth_core.pop("truth_bundle_id", None)
    truth["truth_bundle_id"] = hashlib.sha256(
        canonical_score_bytes(truth_core)
    ).hexdigest()
    truth_payload = canonical_score_bytes(truth)
    truth_path.write_bytes(truth_payload)
    truth_sha256 = hashlib.sha256(truth_payload).hexdigest()
    receipt_path = governed_tmp_path / "truth-publication-receipt.json"
    receipt_sha256 = _truth_publication_receipt(
        receipt_path, truth_path, truth_sha256
    )

    with pytest.raises(
        ProspectiveHourlyScoringError,
        match="fixture classifications differ",
    ):
        score_structural_commitment(
            ROOT,
            commitment_path=COMMITMENT,
            expected_commitment_sha256=COMMITMENT_SHA256,
            predictions_path=PREDICTIONS,
            expected_predictions_sha256=PREDICTIONS_SHA256,
            scoring_contract_path=SCORING_CONTRACT,
            expected_scoring_contract_sha256=SCORING_CONTRACT_SHA256,
            truth_publication_receipt_path=receipt_path,
            expected_truth_publication_receipt_sha256=receipt_sha256,
            delivery_month="2026-08",
            allow_test_fixture=True,
        )


def test_score_publication_resumes_empty_staging(
    governed_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score_core = {
        "schema_version": "synthetic-score.v1",
        "status": "LOCAL_DIAGNOSTIC_NO_GO",
        "production_authorization": False,
    }
    monkeypatch.setattr(
        score_cli,
        "score_structural_commitment",
        lambda *args, **kwargs: score_core,
    )
    output = governed_tmp_path / "score-output"
    (governed_tmp_path / ".score-output.staging").mkdir()

    result = score_cli.build_score(
        ROOT,
        output,
        commitment_path=COMMITMENT,
        expected_commitment_sha256=COMMITMENT_SHA256,
        predictions_path=PREDICTIONS,
        expected_predictions_sha256=PREDICTIONS_SHA256,
        scoring_contract_path=SCORING_CONTRACT,
        expected_scoring_contract_sha256=SCORING_CONTRACT_SHA256,
        truth_publication_receipt_path=Path("build/not-read.json"),
        expected_truth_publication_receipt_sha256="0" * 64,
        delivery_month="2026-08",
    )

    assert result["status"] == "LOCAL_DIAGNOSTIC_NO_GO"
    assert (output / "score.json").is_file()
    assert not (governed_tmp_path / ".score-output.staging").exists()
