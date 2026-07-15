"""Serialize LT results into an immutable-candidate staging directory."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.data.lt_input_sources import verify_frame_receipt, verify_source_receipt
from pfc_shaping.data.shared_data_root import LT_DATA_VIEW_ID
from pfc_shaping.pipeline.probabilistic_output_governance import (
    deterministic_only_frame,
)
from pfc_shaping.pipeline.production_phases import LoadedInputs, LongTermArtifacts


class CandidateSerializationError(RuntimeError):
    """Raised when candidate evidence cannot be serialized consistently."""


def write_long_term_candidate(
    staging_path: str | Path,
    *,
    run_id: str,
    long_term: LongTermArtifacts,
    inputs: LoadedInputs,
    project_root: str | Path,
    pre_run_governance_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write one LT-only candidate without touching shared production outputs."""

    staging = Path(staging_path).resolve()
    if not staging.is_dir():
        raise CandidateSerializationError(f"candidate staging directory is missing: {staging}")
    if any(staging.iterdir()) and pre_run_governance_manifest is None:
        raise CandidateSerializationError(f"candidate staging directory is not empty: {staging}")
    verified_pre_run: Mapping[str, object] | None = None
    if pre_run_governance_manifest is not None:
        from pfc_shaping.pipeline.candidate_evidence import (
            verify_pre_run_governance_evidence,
        )

        reference = pd.Timestamp(inputs.reference_timestamp)
        verified_pre_run = verify_pre_run_governance_evidence(
            staging,
            expected_run_id=run_id,
            expected_valuation_timestamp=reference.tz_convert("UTC").isoformat(),
            require_only_pre_run_files=True,
        )
        if dict(pre_run_governance_manifest) != verified_pre_run:
            raise CandidateSerializationError(
                "pre-run governance manifest changed before candidate serialization"
            )

    manifests_dir = staging / "manifests"
    evidence_dir = staging / "evidence"
    models_dir = staging / "models"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True)
    source_evidence = _archive_forward_source(
        long_term,
        evidence_dir,
        staging,
        pre_run_governance=verified_pre_run,
    )

    market_outputs: dict[str, object] = {}
    for market, branch in sorted(long_term.markets.items()):
        code = str(market).upper()
        market_dir = staging / "markets" / code
        market_dir.mkdir(parents=True)
        parquet_path = market_dir / "pfc_15min.parquet"
        csv_path = market_dir / "pfc_15min.csv"
        candidate_pfc = _deterministic_pfc_for_candidate(branch.pfc, market=code)
        candidate_pfc.to_parquet(parquet_path)
        candidate_pfc.to_csv(csv_path, sep=";", index_label="timestamp_local")
        projection_path = market_dir / "final_product_projection.json"
        _write_json(projection_path, branch.final_product_projection_report)

        manifest = dict(long_term.monthly_curve_manifests.get(code, {}))
        manifest_path: Path | None = None
        if manifest:
            if code == "CH" and isinstance(manifest.get("forward_snapshot"), Mapping):
                portable_snapshot = dict(manifest["forward_snapshot"])
                portable_snapshot["source_path"] = Path("..", str(source_evidence["forward_source_archive"])).as_posix()
                portable_snapshot["source_description"] = "archived EEX source in LT candidate bundle"
                manifest["forward_snapshot"] = portable_snapshot
            manifest_path = manifests_dir / f"production_monthly_curve_manifest_{code.lower()}.json"
            _write_json(manifest_path, manifest)
        market_outputs[code] = {
            "rows": int(len(candidate_pfc)),
            "columns": list(candidate_pfc.columns),
            "probabilistic_status": "DETERMINISTIC_ONLY",
            "pfc_parquet": parquet_path.relative_to(staging).as_posix(),
            "pfc_parquet_sha256": _sha256_file(parquet_path),
            "pfc_csv": csv_path.relative_to(staging).as_posix(),
            "pfc_csv_sha256": _sha256_file(csv_path),
            "final_product_projection": projection_path.relative_to(staging).as_posix(),
            "final_product_projection_sha256": _sha256_file(projection_path),
            "monthly_curve_manifest": (
                manifest_path.relative_to(staging).as_posix() if manifest_path else None
            ),
            "monthly_curve_manifest_sha256": (
                _sha256_file(manifest_path) if manifest_path else None
            ),
        }
        _save_market_models(branch, models_dir / code.lower())

    _save_shared_models(long_term, models_dir / "shared")
    deterministic_artifacts = _deterministic_artifact_inventory(staging)
    deterministic_artifacts_sha256 = _semantic_json_sha256(
        deterministic_artifacts
    )
    input_sources = _verified_input_sources(inputs)
    if inputs.config_source_receipt is None:
        raise CandidateSerializationError("candidate is missing the consumed config receipt")
    try:
        verify_source_receipt(inputs.config_source_receipt)
    except (OSError, ValueError) as exc:
        raise CandidateSerializationError("config changed after it was consumed") from exc
    config_path = Path(inputs.config_source_receipt.path)
    config_sha256 = inputs.config_source_receipt.sha256
    if inputs.config_semantic_sha256 != _semantic_json_sha256(inputs.config):
        raise CandidateSerializationError("config mapping changed after it was consumed")
    config_payload = config_path.read_bytes()
    if hashlib.sha256(config_payload).hexdigest() != config_sha256:
        raise CandidateSerializationError("config changed before candidate archival")
    config_archive = evidence_dir / "config" / "config.yaml"
    config_archive.parent.mkdir(parents=True)
    _write_bytes_fsync(config_archive, config_payload)
    data_metadata = _archive_data_metadata(inputs, evidence_dir, staging)
    governed_run_spec: Mapping[str, object] | None = None
    governed_run_spec_sha256: str | None = None
    if verified_pre_run is not None:
        raw_run_spec = verified_pre_run.get("governed_run_spec")
        governed_run_spec_sha256 = str(
            verified_pre_run.get("governed_run_spec_sha256", "")
        )
        if not isinstance(raw_run_spec, Mapping):
            raise CandidateSerializationError("candidate is missing the governed run spec")
        governed_run_spec = dict(raw_run_spec)
        snapshot_binding = governed_run_spec.get("input_snapshot")
        if not isinstance(snapshot_binding, Mapping):
            raise CandidateSerializationError("governed run spec input binding is missing")
        actual_pointer_sha256 = (
            inputs.data_pointer_receipt.sha256
            if inputs.data_pointer_receipt is not None
            else None
        )
        actual_contract_sha256 = (
            inputs.data_contract_receipt.sha256
            if inputs.data_contract_receipt is not None
            else None
        )
        if (
            governed_run_spec.get("config", {}).get("sha256") != config_sha256
            or snapshot_binding.get("generation_id") != inputs.data_generation_id
            or snapshot_binding.get("contract_sha256") != actual_contract_sha256
            or snapshot_binding.get("pointer_sha256") != actual_pointer_sha256
        ):
            raise CandidateSerializationError(
                "consumed config or input snapshot differs from governed run spec"
            )
    canonical_data_view = (
        inputs.data_layout == "external_v2"
        and inputs.data_pointer_receipt is not None
        and inputs.data_pointer_receipt.logical_path == "views/pfc_lt/current.json"
    )
    reference = pd.Timestamp(inputs.reference_timestamp)
    if reference.tzinfo is None:
        raise CandidateSerializationError("candidate reference timestamp must be timezone-aware")
    run_manifest: dict[str, object] = {
        "schema_version": "lt_candidate_run.v1",
        "run_id": str(run_id),
        "pipeline_scope": "LT_ONLY",
        "promotion_eligible": False,
        "probabilistic_status": "DETERMINISTIC_ONLY",
        "interval_columns": [],
        "intervals_permitted_for_pricing_or_risk": False,
        "candidate_serialized_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_timestamp": reference.tz_convert("UTC").isoformat(),
        "reference_timestamp_is_explicit": bool(inputs.reference_timestamp_is_explicit),
        "config_path": config_archive.relative_to(staging).as_posix(),
        "config_sha256": config_sha256,
        "markets": market_outputs,
        "source_evidence": source_evidence,
        "input_sources": input_sources,
        "deterministic_artifacts": deterministic_artifacts,
        "deterministic_artifacts_sha256": deterministic_artifacts_sha256,
        "data_layout": inputs.data_layout,
        "data_root_id": (
            LT_DATA_VIEW_ID
            if canonical_data_view
            else "legacy_external_pointer"
            if inputs.data_layout == "external_v2"
            else "legacy_repo"
        ),
        "data_generation_id": inputs.data_generation_id,
        "data_contract": data_metadata.get("contract"),
        "data_pointer": data_metadata.get("pointer"),
        "freshness_reports": {
            key: {
                "dataset": report.dataset,
                "checks_passed": int(report.checks_passed),
                "warnings": list(report.warnings),
                "errors": list(report.errors),
                "metrics": dict(report.metrics),
            }
            for key, report in sorted(inputs.freshness_reports.items())
        },
    }
    if governed_run_spec is not None:
        run_manifest["governed_run_spec"] = governed_run_spec
        run_manifest["governed_run_spec_sha256"] = governed_run_spec_sha256
        run_manifest["candidate_serialized_at_utc"] = governed_run_spec[
            "build_timestamp_utc"
        ]
    if verified_pre_run is not None:
        from pfc_shaping.pipeline.candidate_evidence import (
            PRE_RUN_EVIDENCE_MANIFEST,
        )
        pre_run_path = staging / PRE_RUN_EVIDENCE_MANIFEST
        run_manifest["pre_run_governance_manifest"] = PRE_RUN_EVIDENCE_MANIFEST.as_posix()
        run_manifest["pre_run_governance_manifest_sha256"] = _sha256_file(pre_run_path)
    run_manifest_path = manifests_dir / "candidate_run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    return run_manifest


def _archive_forward_source(
    long_term: LongTermArtifacts,
    evidence_dir: Path,
    staging: Path,
    *,
    pre_run_governance: Mapping[str, object] | None,
) -> dict[str, object]:
    manifest = long_term.monthly_curve_manifests.get("CH", {})
    snapshot = manifest.get("forward_snapshot") if isinstance(manifest, Mapping) else None
    if not isinstance(snapshot, Mapping):
        raise CandidateSerializationError("CH candidate is missing forward_snapshot evidence")
    source_path_value = snapshot.get("source_path")
    source_sha256 = str(snapshot.get("source_sha256", ""))
    if not source_path_value or len(source_sha256) != 64:
        raise CandidateSerializationError("CH forward source path/hash evidence is incomplete")
    source_path = Path(str(source_path_value)).resolve()
    captured_path: Path | None = None
    if pre_run_governance is not None:
        pre_run_artifacts = pre_run_governance.get("artifacts")
        pre_run_eex = (
            pre_run_artifacts.get("eex_forward_source")
            if isinstance(pre_run_artifacts, Mapping)
            else None
        )
        if not isinstance(pre_run_eex, Mapping):
            raise CandidateSerializationError(
                "candidate is missing pre-run authenticated EEX source"
            )
        acquisition = pre_run_eex.get("acquisition_contract")
        if not isinstance(acquisition, Mapping) or pd.Timestamp(
            snapshot.get("available_at")
        ) != pd.Timestamp(acquisition.get("available_at_utc")):
            raise CandidateSerializationError(
                "consumed forward availability does not match the acquisition contract"
            )
        captured_path = staging / str(pre_run_eex.get("path", ""))
        if (
            not captured_path.is_file()
            or str(pre_run_eex.get("sha256", "")) != source_sha256
            or _sha256_file(captured_path) != source_sha256
        ):
            raise CandidateSerializationError(
                "pre-run EEX source does not match the consumed forward snapshot"
            )
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise CandidateSerializationError(f"cannot archive EEX source: {source_path}") from exc
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != source_sha256:
        raise CandidateSerializationError("EEX source changed before candidate serialization")
    archive_dir = evidence_dir / "eex"
    archive_dir.mkdir(parents=True)
    if captured_path is None:
        suffix = source_path.suffix.lower() or ".bin"
        archive_path = archive_dir / f"{actual_hash}{suffix}"
        _write_bytes_fsync(archive_path, source_bytes)
    else:
        archive_path = captured_path
    quotes = snapshot.get("quotes")
    if not isinstance(quotes, list) or not quotes:
        raise CandidateSerializationError("CH forward snapshot quotes are missing")
    quote_rows: list[dict[str, object]] = []
    for quote in quotes:
        if not isinstance(quote, Mapping):
            raise CandidateSerializationError("CH forward snapshot quote is invalid")
        raw_product = str(quote.get("product", ""))
        load_type = str(quote.get("load_type", "")).upper()
        product = raw_product
        if load_type == "PEAK" and raw_product.lower().endswith("-peak"):
            product = raw_product[:-5]
        elif load_type == "OFFPEAK" and raw_product.lower().endswith("-offpeak"):
            product = raw_product[:-8]
        quote_rows.append(
            {
                "date": pd.Timestamp(snapshot.get("snapshot_date")).tz_localize(None).normalize(),
                "product": product,
                "load_type": load_type,
                "product_type": _product_type(product),
                "price": float(quote.get("price_eur_mwh")),
                "market": str(snapshot.get("market", "CH")).upper(),
                "source": "candidate_forward_snapshot",
                "quote_id": quote.get("quote_id"),
            }
        )
    quotes_path = archive_dir / "forward_quote_snapshot.parquet"
    pd.DataFrame(quote_rows).to_parquet(quotes_path, index=False)
    return {
        "forward_source_original_name": source_path.name,
        "forward_source_archive": archive_path.relative_to(staging).as_posix(),
        "forward_source_sha256": actual_hash,
        "forward_quote_snapshot": quotes_path.relative_to(staging).as_posix(),
        "forward_quote_snapshot_sha256": _sha256_file(quotes_path),
        "forward_snapshot_id": snapshot.get("snapshot_id"),
        "forward_observation_id": snapshot.get("observation_id"),
    }


def _product_type(product: str) -> str:
    value = str(product)
    if len(value) == 4 and value.isdigit():
        return "Cal"
    if "-Q" in value:
        return "Quarter"
    if len(value) == 7 and value[4] == "-":
        return "Month"
    return "Unknown"


def _verified_input_sources(inputs: LoadedInputs) -> dict[str, object]:
    required = {"epex_ch", "epex_de", "entso", "hydro"}
    missing = required.difference(inputs.input_source_receipts)
    if missing:
        raise CandidateSerializationError(
            f"candidate is missing consumed input receipts: {sorted(missing)}"
        )
    out: dict[str, object] = {}
    frames = dict(inputs.input_frames)
    if not frames:
        frames = {
            "epex_ch": inputs.epex_ch,
            "epex_de": inputs.epex_de,
            "entso": inputs.entso,
            "hydro": inputs.hydro,
        }
    for role, receipt in sorted(inputs.input_source_receipts.items()):
        try:
            verify_source_receipt(receipt)
        except (OSError, ValueError) as exc:
            raise CandidateSerializationError(
                f"input source changed after it was consumed: {role}"
            ) from exc
        if role in frames:
            try:
                verify_frame_receipt(frames[role], receipt)
            except ValueError as exc:
                raise CandidateSerializationError(
                    f"input frame changed after it was consumed: {role}"
                ) from exc
        out[role] = receipt.to_manifest()
    return out


def _archive_data_metadata(
    inputs: LoadedInputs,
    evidence_dir: Path,
    staging: Path,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, receipt, payload, filename in (
        (
            "contract",
            inputs.data_contract_receipt,
            inputs.data_contract_bytes,
            "lt_input_snapshot.json",
        ),
        ("pointer", inputs.data_pointer_receipt, inputs.data_pointer_bytes, "current.json"),
    ):
        if receipt is None and payload is None:
            continue
        if receipt is None or payload is None:
            raise CandidateSerializationError(f"candidate has incomplete data {name} evidence")
        if hashlib.sha256(payload).hexdigest() != receipt.sha256:
            raise CandidateSerializationError(f"captured data {name} bytes do not match receipt")
        if name == "contract":
            try:
                verify_source_receipt(receipt)
            except (OSError, ValueError) as exc:
                raise CandidateSerializationError(
                    "LT input snapshot contract changed after consumption"
                ) from exc
        destination = evidence_dir / "data" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_fsync(destination, payload)
        item = receipt.to_manifest()
        item["archive_path"] = destination.relative_to(staging).as_posix()
        out[name] = item
    return out


def _semantic_json_sha256(payload: object) -> str:
    raw = json.dumps(
        _json_clean(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _deterministic_artifact_inventory(staging: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for root_name in ("markets", "models"):
        root = staging / root_name
        for path in sorted(root.rglob("*")):
            if path.is_file():
                inventory[path.relative_to(staging).as_posix()] = _sha256_file(path)
    if not inventory:
        raise CandidateSerializationError("candidate has no deterministic model artifacts")
    return inventory


def verify_deterministic_artifact_inventory(
    staging_path: str | Path,
    run_manifest: Mapping[str, object],
) -> str:
    """Recompute and verify the model-output identity recorded by the candidate."""

    staging = Path(staging_path).resolve()
    expected = run_manifest.get("deterministic_artifacts")
    expected_sha256 = str(run_manifest.get("deterministic_artifacts_sha256", ""))
    if not isinstance(expected, Mapping):
        raise CandidateSerializationError(
            "candidate deterministic artifact inventory is missing"
        )
    normalized = {str(path): str(digest) for path, digest in expected.items()}
    actual = _deterministic_artifact_inventory(staging)
    actual_sha256 = _semantic_json_sha256(actual)
    if normalized != actual or expected_sha256 != actual_sha256:
        raise CandidateSerializationError(
            "candidate deterministic artifact inventory/hash mismatch"
        )
    return actual_sha256


def _save_market_models(branch: object, destination: Path) -> None:
    destination.mkdir(parents=True)
    shape = getattr(branch, "sh", None)
    if shape is not None and hasattr(shape, "save"):
        suffix = ".pkl" if shape.__class__.__name__ == "ShapeHourlyMLP" else ".parquet"
        shape.save(destination / f"shape_hourly{suffix}")
    water_value = getattr(branch, "wv", None)
    if water_value is not None and hasattr(water_value, "save"):
        water_value.save(destination / "water_value.parquet")


def _save_shared_models(long_term: LongTermArtifacts, destination: Path) -> None:
    destination.mkdir(parents=True)
    artifact = long_term.shared.si
    if artifact is None or not hasattr(artifact, "save"):
        raise CandidateSerializationError(
            "shared model artifact is not serializable: shape_intraday.parquet"
        )
    artifact.save(destination / "shape_intraday.parquet")


def _deterministic_pfc_for_candidate(
    pfc: pd.DataFrame,
    *,
    market: str,
) -> pd.DataFrame:
    """Reject publishable intervals and return the deterministic candidate frame."""

    try:
        return deterministic_only_frame(pfc, context=f"{market} candidate")
    except ValueError as exc:
        raise CandidateSerializationError(str(exc)) from exc


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_clean(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_clean(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value
