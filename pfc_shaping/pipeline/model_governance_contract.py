from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

MODEL_GOVERNANCE_RECEIPT_SCHEMA = "model_governance_artifact_receipt.v1"
MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV = (
    "PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH"
)
SIGNATURE_ALGORITHM = "Ed25519"


class ModelGovernanceAuthenticationError(ValueError):
    """Raised when an independent model-governance input is not authentic."""


def sign_model_governance_artifact_receipt(
    artifact_path: str | Path,
    *,
    artifact_role: str,
    authority_id: str,
    issued_at_utc: str,
    data_cutoff_utc: str,
    private_key_path: str | Path,
) -> dict[str, object]:
    artifact = Path(artifact_path).resolve()
    if not artifact.is_file():
        raise ModelGovernanceAuthenticationError(
            f"governance artifact is unavailable: {artifact}"
        )
    payload: dict[str, object] = {
        "schema_version": MODEL_GOVERNANCE_RECEIPT_SCHEMA,
        "artifact_role": str(artifact_role),
        "artifact_sha256": _sha256_file(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "authority_id": _required_text(authority_id, label="authority_id"),
        "issued_at_utc": _aware_utc(issued_at_utc, label="issued_at_utc").isoformat(),
        "data_cutoff_utc": _aware_utc(
            data_cutoff_utc,
            label="data_cutoff_utc",
        ).isoformat(),
        "approved": True,
    }
    receipt_id = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    signed = {**payload, "receipt_id": receipt_id}
    private_key = _load_private_key(private_key_path)
    signature = private_key.sign(_canonical_json_bytes(signed))
    signed["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(private_key.public_key()),
        "value_base64": base64.b64encode(signature).decode("ascii"),
    }
    return signed


def verify_model_governance_artifact_receipt(
    receipt: Mapping[str, object],
    *,
    artifact_path: str | Path,
    expected_role: str,
    valuation_timestamp: str | pd.Timestamp,
) -> dict[str, object]:
    key_value = os.environ.get(MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV)
    if not key_value:
        raise ModelGovernanceAuthenticationError(
            f"{MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV} is required"
        )
    artifact = Path(artifact_path).resolve()
    if not artifact.is_file():
        raise ModelGovernanceAuthenticationError(
            f"governance artifact is unavailable: {artifact}"
        )
    payload = dict(receipt)
    signature_block = payload.pop("signature", None)
    if payload.get("schema_version") != MODEL_GOVERNANCE_RECEIPT_SCHEMA:
        raise ModelGovernanceAuthenticationError("governance receipt schema mismatch")
    if payload.get("approved") is not True:
        raise ModelGovernanceAuthenticationError("governance artifact is not approved")
    if str(payload.get("artifact_role", "")) != str(expected_role):
        raise ModelGovernanceAuthenticationError("governance artifact role mismatch")
    if str(payload.get("artifact_sha256", "")) != _sha256_file(artifact):
        raise ModelGovernanceAuthenticationError("governance artifact hash mismatch")
    if payload.get("artifact_size_bytes") != artifact.stat().st_size:
        raise ModelGovernanceAuthenticationError("governance artifact size mismatch")
    _required_text(payload.get("authority_id"), label="authority_id")
    issued_at = _aware_utc(payload.get("issued_at_utc"), label="issued_at_utc")
    data_cutoff = _aware_utc(payload.get("data_cutoff_utc"), label="data_cutoff_utc")
    valuation = _aware_utc(valuation_timestamp, label="valuation_timestamp")
    if issued_at > valuation:
        raise ModelGovernanceAuthenticationError(
            "governance receipt was issued after valuation"
        )
    if data_cutoff > issued_at or data_cutoff > valuation:
        raise ModelGovernanceAuthenticationError(
            "governance artifact data cutoff is not point-in-time"
        )
    receipt_id = str(payload.pop("receipt_id", ""))
    if receipt_id != hashlib.sha256(_canonical_json_bytes(payload)).hexdigest():
        raise ModelGovernanceAuthenticationError("governance receipt_id mismatch")
    signed_payload = {**payload, "receipt_id": receipt_id}
    if not isinstance(signature_block, Mapping):
        raise ModelGovernanceAuthenticationError("governance receipt signature is missing")
    if signature_block.get("algorithm") != SIGNATURE_ALGORITHM:
        raise ModelGovernanceAuthenticationError(
            "governance receipt signature algorithm is unsupported"
        )
    public_key = _load_public_key(key_value)
    if str(signature_block.get("key_id", "")) != _public_key_id(public_key):
        raise ModelGovernanceAuthenticationError(
            "governance receipt signing key is not trusted"
        )
    try:
        signature = base64.b64decode(
            str(signature_block.get("value_base64", "")),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise ModelGovernanceAuthenticationError(
            "governance receipt signature is not valid base64"
        ) from exc
    try:
        public_key.verify(signature, _canonical_json_bytes(signed_payload))
    except InvalidSignature as exc:
        raise ModelGovernanceAuthenticationError(
            "governance receipt signature is invalid"
        ) from exc
    if expected_role == "historical_thresholds":
        validate_historical_thresholds_artifact(
            artifact,
            data_cutoff_utc=data_cutoff,
        )
    return signed_payload


def validate_historical_thresholds_artifact(
    artifact_path: str | Path,
    *,
    data_cutoff_utc: str | pd.Timestamp,
) -> pd.DataFrame:
    required_columns = {
        "gate_id",
        "metric",
        "market",
        "delivery_bucket",
        "lookback_start",
        "lookback_end",
        "n_snapshots",
        "min_required_n",
        "p50",
        "p90",
        "p975",
        "max_observed",
        "regime_filter",
        "status",
    }
    try:
        frame = pd.read_csv(artifact_path)
    except (OSError, ValueError) as exc:
        raise ModelGovernanceAuthenticationError(
            "historical thresholds artifact is unreadable"
        ) from exc
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ModelGovernanceAuthenticationError(
            f"historical thresholds columns are missing: {missing}"
        )
    expected_gate_metrics = {
        "same_month_rank_consistency": "same_month_shape_delta_abs_eur_mwh",
        "residual_vs_implied_comparable_block": (
            "comparable_block_shape_delta_abs_eur_mwh"
        ),
    }
    expected_buckets = {"all", *(f"month_{month:02d}" for month in range(1, 13))}
    identities = set(zip(frame["gate_id"].astype(str), frame["delivery_bucket"].astype(str)))
    expected_identities = {
        (gate_id, bucket)
        for gate_id in expected_gate_metrics
        for bucket in expected_buckets
    }
    if identities != expected_identities or len(frame) != len(expected_identities):
        raise ModelGovernanceAuthenticationError(
            "historical thresholds gate/bucket coverage mismatch"
        )
    if set(frame["market"].astype(str).str.upper()) != {"CH"}:
        raise ModelGovernanceAuthenticationError(
            "historical thresholds market must be CH"
        )
    if not set(frame["status"].astype(str)).issubset({"PASS", "UNSUPPORTED"}):
        raise ModelGovernanceAuthenticationError("historical thresholds status is invalid")
    for gate_id, metric in expected_gate_metrics.items():
        actual = set(frame.loc[frame["gate_id"].eq(gate_id), "metric"].astype(str))
        if actual != {metric}:
            raise ModelGovernanceAuthenticationError(
                "historical thresholds gate/metric mismatch"
            )
    cutoff = _aware_utc(data_cutoff_utc, label="data_cutoff_utc")
    lookback_start = pd.to_datetime(frame["lookback_start"], utc=True, errors="coerce")
    lookback_end = pd.to_datetime(frame["lookback_end"], utc=True, errors="coerce")
    populated_start = frame["lookback_start"].fillna("").astype(str).str.strip().ne("")
    populated = frame["lookback_end"].fillna("").astype(str).str.strip().ne("")
    if not populated_start.equals(populated):
        raise ModelGovernanceAuthenticationError(
            "historical thresholds lookback window is incomplete"
        )
    if (
        lookback_start[populated].isna().any()
        or lookback_end[populated].isna().any()
        or bool((lookback_start[populated] > lookback_end[populated]).any())
        or bool((lookback_end[populated] > cutoff).any())
    ):
        raise ModelGovernanceAuthenticationError(
            "historical thresholds lookback exceeds signed data cutoff"
        )
    n_snapshots = pd.to_numeric(frame["n_snapshots"], errors="coerce")
    min_required = pd.to_numeric(frame["min_required_n"], errors="coerce")
    if (
        n_snapshots.isna().any()
        or min_required.isna().any()
        or bool((n_snapshots < 0).any())
        or bool((min_required <= 0).any())
        or bool((n_snapshots % 1 != 0).any())
        or bool((min_required % 1 != 0).any())
    ):
        raise ModelGovernanceAuthenticationError(
            "historical thresholds sample counts are invalid"
        )
    supported = frame["status"].astype(str).eq("PASS")
    if bool((n_snapshots[supported] < min_required[supported]).any()):
        raise ModelGovernanceAuthenticationError(
            "historical thresholds PASS has insufficient samples"
        )
    p50 = pd.to_numeric(frame.loc[supported, "p50"], errors="coerce")
    p90 = pd.to_numeric(frame.loc[supported, "p90"], errors="coerce")
    p975 = pd.to_numeric(frame.loc[supported, "p975"], errors="coerce")
    maximum = pd.to_numeric(frame.loc[supported, "max_observed"], errors="coerce")
    if (
        p50.isna().any()
        or p90.isna().any()
        or p975.isna().any()
        or maximum.isna().any()
        or bool((p50 < 0).any())
        or bool((p50 > p90).any())
        or bool((p90 > p975).any())
        or bool((p975 > maximum).any())
    ):
        raise ModelGovernanceAuthenticationError(
            "historical thresholds supported quantiles are invalid"
        )
    unsupported = ~supported
    unsupported_quantiles = frame.loc[
        unsupported,
        ["p50", "p90", "p975", "max_observed"],
    ].apply(pd.to_numeric, errors="coerce")
    if unsupported_quantiles.notna().any().any():
        raise ModelGovernanceAuthenticationError(
            "historical thresholds UNSUPPORTED rows contain quantiles"
        )
    return frame


def validate_selected_lambda_decision(
    decision: Mapping[str, object],
    *,
    expected_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from pfc_shaping.calibration.monthly_curve_config_identity import config_hash

    if decision.get("schema_version") != "monthly_curve_selected_lambda_decision.v1":
        raise ModelGovernanceAuthenticationError("selected lambda decision schema mismatch")
    if decision.get("production_approved") is not True:
        raise ModelGovernanceAuthenticationError(
            "selected lambda decision is not production approved"
        )
    if decision.get("selection_status") != "PRODUCTION_APPROVED":
        raise ModelGovernanceAuthenticationError(
            "selected lambda decision status is not production approved"
        )
    for key in ("decision_id", "calibration_campaign_id", "selection_reason"):
        _required_text(decision.get(key), label=key)
    forbidden = {
        "candidate_manifest",
        "candidate_manifest_sha256",
        "monthly_solution_hash",
        "active_constraints_hash",
        "run_id",
        "export_manifest",
    }
    present = sorted(forbidden.intersection(decision))
    if present:
        raise ModelGovernanceAuthenticationError(
            f"selected lambda decision contains post-run fields: {present}"
        )
    canonical = decision.get("canonical_config")
    if not isinstance(canonical, Mapping):
        raise ModelGovernanceAuthenticationError(
            "selected lambda decision canonical_config is missing"
        )
    computed = config_hash(canonical)
    if str(decision.get("config_hash", "")) != computed:
        raise ModelGovernanceAuthenticationError(
            "selected lambda decision config hash mismatch"
        )
    if expected_config is not None:
        expected_hash = config_hash(expected_config)
        if computed != expected_hash or dict(canonical) != dict(expected_config):
            raise ModelGovernanceAuthenticationError(
                "selected lambda decision does not match active solver config"
            )
    return dict(decision)


def _aware_utc(value: object, *, label: str) -> pd.Timestamp:
    if value is None or value == "":
        raise ModelGovernanceAuthenticationError(f"{label} is missing")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ModelGovernanceAuthenticationError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ModelGovernanceAuthenticationError(f"{label} is missing")
    return text


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise ModelGovernanceAuthenticationError(
            "cannot load model governance private key"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ModelGovernanceAuthenticationError(
            "model governance private key is not Ed25519"
        )
    return key


def _load_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(Path(path).read_bytes())
    except (OSError, TypeError, ValueError) as exc:
        raise ModelGovernanceAuthenticationError(
            "cannot load trusted model governance public key"
        ) from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ModelGovernanceAuthenticationError(
            "trusted model governance public key is not Ed25519"
        )
    return key


def _public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
