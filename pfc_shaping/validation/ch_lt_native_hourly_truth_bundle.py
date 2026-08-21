"""Build a complete local CH native-hourly truth bundle from an exact ledger."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.durable_artifact import (
    DurableArtifactExistsError,
    write_durable_exact_bytes,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import (
    ch_lt_local_future_origin_selection as origin_selection_auditor,
)
from pfc_shaping.validation.ch_lt_prospective_capture_ledger import (
    build_prospective_capture_ledger,
)

# Capture the governed invocation root once without embedding a workstation path.
ROOT = Path.cwd()
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_PARQUET_BYTES = 64 * 1024 * 1024
_SCORING_CONTRACT_SHA256 = (
    "d84d7ebc2f3c6aa4b820fab26f0a5de18940a6d35af8e796323025eab694ea4d"
)
_SCORING_CONTRACT_ID = (
    "ae85de71a3745b4ea8b9704090f9bd742e26a41161c2d0b804f4888fb372653e"
)


class NativeHourlyTruthBundleError(ValueError):
    """Raised when a complete exact target truth bundle cannot be built."""


def _local_wall_clock_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NativeHourlyTruthBundleError(f"{label} must be an object")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _directory_identity(path: Path) -> tuple[int, int]:
    selected = assert_absolute_path_has_no_links(path)
    metadata = selected.stat(follow_symlinks=False)
    if not selected.is_dir():
        raise NativeHourlyTruthBundleError("output parent is not a directory")
    return int(metadata.st_dev), int(metadata.st_ino)


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeHourlyTruthBundleError(f"{label} is not lowercase SHA-256")
    return value


def _bound_bytes(
    root: Path,
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> tuple[Path, bytes]:
    selected = path if path.is_absolute() else root / path
    selected = Path(os.path.abspath(selected))
    if not _inside(selected, root / "build") and not _inside(
        selected, root / ".planning"
    ):
        raise NativeHourlyTruthBundleError(f"{label} escapes governed namespaces")
    payload = read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)
    if hashlib.sha256(payload).hexdigest() != _sha256(
        expected_sha256, label=f"{label} SHA-256"
    ):
        raise NativeHourlyTruthBundleError(f"{label} SHA-256 mismatch")
    return selected, payload


def _bound_json(
    root: Path, path: Path, expected_sha256: str, *, label: str
) -> tuple[Path, Mapping[str, object]]:
    selected, payload = _bound_bytes(
        root,
        path,
        expected_sha256,
        label=label,
        max_bytes=_MAX_JSON_BYTES,
    )
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise NativeHourlyTruthBundleError(f"{label} is not strict JSON") from exc
    return selected, _mapping(parsed, label=label)


def _utc(value: object, *, label: str) -> pd.Timestamp:
    try:
        selected = pd.Timestamp(str(value))
    except (TypeError, ValueError) as exc:
        raise NativeHourlyTruthBundleError(f"{label} is not a timestamp") from exc
    if selected.tzinfo is None:
        raise NativeHourlyTruthBundleError(f"{label} is not timezone-aware")
    return selected.tz_convert("UTC")


def extract_native_hourly_prices(frame: pd.DataFrame, *, label: str) -> pd.Series:
    """Recover native hours only when each four-row proxy block is exact."""

    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.tz is None
        or "price_eur_mwh" not in frame.columns
    ):
        raise NativeHourlyTruthBundleError(f"{label} frame contract differs")
    selected = frame.loc[:, "price_eur_mwh"].copy()
    selected.index = selected.index.tz_convert("UTC")
    selected = selected.sort_index()
    if selected.empty or not selected.index.is_unique:
        raise NativeHourlyTruthBundleError(f"{label} index is empty or duplicated")
    expected = pd.date_range(
        selected.index[0], periods=len(selected), freq="15min", tz="UTC"
    )
    if not selected.index.equals(expected) or len(selected) % 4:
        raise NativeHourlyTruthBundleError(f"{label} quarter-hour transport grid differs")
    values = pd.to_numeric(selected, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise NativeHourlyTruthBundleError(f"{label} contains non-finite prices")
    hour_key = values.index.floor("h")
    counts = values.groupby(hour_key, sort=True).count()
    minima = values.groupby(hour_key, sort=True).min()
    maxima = values.groupby(hour_key, sort=True).max()
    if not (counts == 4).all() or not np.allclose(
        minima.to_numpy(), maxima.to_numpy(), rtol=0.0, atol=1e-12
    ):
        raise NativeHourlyTruthBundleError(
            f"{label} is not an exact stepwise native-hourly transport proxy"
        )
    return values.groupby(hour_key, sort=True).mean()


def build_truth_bundle(
    root: Path,
    *,
    ledger_path: Path,
    expected_ledger_sha256: str,
    commitment_path: Path,
    expected_commitment_sha256: str,
    delivery_month: str,
    truth_opened_at_utc_untrusted: str,
) -> dict[str, object]:
    """Rehash the ledger recursively and assemble exactly one matured month."""

    root = assert_absolute_path_has_no_links(root)
    if root != ROOT or Path.cwd() != root:
        raise NativeHourlyTruthBundleError(
            "cwd and root must be the canonical repository"
        )
    selection_audit = origin_selection_auditor.audit_selection(
        registry_path=root / origin_selection_auditor.REGISTRY_RELATIVE,
        expected_registry_sha256=origin_selection_auditor.REGISTRY_SHA256,
    )
    if (
        selection_audit.get("status")
        != (
            "VALID_LOCAL_FUTURE_ORIGIN_REHEARSAL_V7_PACKAGED_MODULE_"
            "SOURCES_BOUND_NONCOUNTABLE_NO_GO"
        )
        or selection_audit.get("outcome_blind") is not True
        or selection_audit.get("future_holdout_consumed") is not False
        or selection_audit.get("t057_consumed") is not False
        or selection_audit.get("scientific_admission") is not False
        or selection_audit.get("production_authorization") is not False
    ):
        raise NativeHourlyTruthBundleError("future origin selection audit differs")
    selection_selected, future_selection = _bound_json(
        root,
        origin_selection_auditor.REGISTRY_RELATIVE,
        origin_selection_auditor.REGISTRY_SHA256,
        label="future origin selection",
    )
    selected_bindings = _mapping(
        future_selection.get("bindings"), label="future selection bindings"
    )
    selected_commitment = _mapping(
        selected_bindings.get("prediction_commitment"),
        label="selected prediction commitment",
    )
    supplied_commitment = (
        commitment_path if commitment_path.is_absolute() else root / commitment_path
    )
    if (
        Path(os.path.abspath(supplied_commitment))
        != root / Path(str(selected_commitment.get("path")))
        or expected_commitment_sha256 != selected_commitment.get("sha256")
    ):
        raise NativeHourlyTruthBundleError(
            "caller commitment differs from future origin selection"
        )
    commitment_selected, commitment = _bound_json(
        root,
        commitment_path,
        expected_commitment_sha256,
        label="prediction commitment",
    )
    if (
        commitment.get("schema_version") != "ch_lt_structural_prediction_commitment.v3"
        or commitment.get("status")
        != "SEALED_LOCAL_STRUCTURAL_DRY_RUN_NONCOUNTABLE_NO_GO"
    ):
        raise NativeHourlyTruthBundleError("prediction commitment boundary differs")
    bindings = _mapping(commitment.get("bindings"), label="commitment bindings")
    inventory_binding = _mapping(
        bindings.get("origin_target_mask_inventory"), label="inventory binding"
    )
    _, inventory = _bound_json(
        root,
        Path(str(inventory_binding.get("path"))),
        str(inventory_binding.get("sha256")),
        label="origin target inventory",
    )
    targets = inventory.get("targets")
    if not isinstance(targets, list):
        raise NativeHourlyTruthBundleError("inventory targets are missing")
    matching = [
        _mapping(item, label="inventory target")
        for item in targets
        if isinstance(item, Mapping) and item.get("delivery_month") == delivery_month
    ]
    if len(matching) != 1:
        raise NativeHourlyTruthBundleError("delivery month is not a unique target")
    target = matching[0]
    start = _utc(target.get("delivery_start_utc"), label="target start")
    end = _utc(target.get("delivery_end_utc"), label="target end")
    expected_count = int(target.get("expected_native_hourly_interval_count"))
    opened_at = _utc(truth_opened_at_utc_untrusted, label="truth opened at")
    if opened_at < end:
        raise NativeHourlyTruthBundleError("truth open precedes target maturity")
    wall_clock = _local_wall_clock_utc()
    if wall_clock < end:
        raise NativeHourlyTruthBundleError(
            "local wall clock has not reached target maturity"
        )
    if opened_at > wall_clock:
        raise NativeHourlyTruthBundleError(
            "declared truth open exceeds the current local wall clock"
        )

    ledger_selected, ledger = _bound_json(
        root,
        ledger_path,
        expected_ledger_sha256,
        label="prospective capture ledger",
    )
    construction_request = _mapping(
        ledger.get("construction_request"), label="ledger construction request"
    )
    reconstructed = build_prospective_capture_ledger(
        repo_root=root,
        request_path=root / Path(str(construction_request.get("path"))),
        expected_request_sha256=str(construction_request.get("sha256")),
    )
    if dict(reconstructed) != dict(ledger):
        raise NativeHourlyTruthBundleError("recursive ledger reconstruction differs")
    if (
        ledger.get("schema_version") != "ch_lt_prospective_capture_ledger.v1"
        or ledger.get("market") != "CH"
        or ledger.get("source_role") != "epex_ch"
        or ledger.get("future_holdout_consumed") is not False
        or ledger.get("t057_consumed") is not False
        or ledger.get("scientific_admission") is not False
        or ledger.get("production_authorization") is not False
    ):
        raise NativeHourlyTruthBundleError("ledger authority boundary differs")
    window = _mapping(ledger.get("window"), label="ledger window")
    ledger_start = _utc(window.get("start_utc"), label="ledger start")
    ledger_end = _utc(window.get("end_utc_exclusive"), label="ledger end")
    if ledger_start > start or ledger_end < end:
        raise NativeHourlyTruthBundleError(
            "ledger does not cover the complete committed target"
        )

    prices: dict[pd.Timestamp, tuple[float, pd.Timestamp, str]] = {}
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise NativeHourlyTruthBundleError("ledger entries are missing")
    for position, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, label=f"ledger entry {position}")
        entry_start = _utc(entry.get("start_utc"), label=f"entry {position} start")
        entry_end = _utc(entry.get("end_utc"), label=f"entry {position} end")
        if entry_end <= start or entry_start >= end:
            continue
        artifacts = _mapping(
            entry.get("acquisition_artifacts"), label=f"entry {position} artifacts"
        )
        bronze = _mapping(artifacts.get("bronze.parquet"), label="bronze binding")
        _, payload = _bound_bytes(
            root,
            Path(str(bronze.get("path"))),
            str(bronze.get("sha256")),
            label=f"entry {position} bronze",
            max_bytes=_MAX_PARQUET_BYTES,
        )
        native = extract_native_hourly_prices(
            pd.read_parquet(io.BytesIO(payload)), label=f"entry {position} bronze"
        )
        available_at = _utc(
            entry.get("received_at_utc_untrusted"),
            label=f"entry {position} received at",
        )
        if available_at > opened_at:
            raise NativeHourlyTruthBundleError(
                f"entry {position} became available after truth open"
            )
        for timestamp, price in native.loc[(native.index >= start) & (native.index < end)].items():
            if timestamp in prices:
                raise NativeHourlyTruthBundleError("truth native hours overlap")
            selected_price = float(price)
            if not math.isfinite(selected_price):
                raise NativeHourlyTruthBundleError("truth price is not finite")
            prices[timestamp] = (
                selected_price,
                available_at,
                str(entry.get("acquisition_id")),
            )

    expected_index = pd.date_range(start, end, freq="h", inclusive="left")
    if len(expected_index) != expected_count or list(prices) != list(expected_index):
        raise NativeHourlyTruthBundleError(
            "assembled truth does not exactly cover the committed UTC hours"
        )
    observations = [
        {
            "delivery_start_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "price_eur_mwh": format(prices[timestamp][0], ".17g"),
            "available_at_utc": prices[timestamp][1]
            .isoformat()
            .replace("+00:00", "Z"),
            "acquisition_id": prices[timestamp][2],
        }
        for timestamp in expected_index
    ]
    core: dict[str, object] = {
        "schema_version": "ch_lt_native_hourly_truth_bundle.v1",
        "status": "COMPLETE_LOCAL_NATIVE_HOURLY_TRUTH_NO_GO",
        "test_fixture": False,
        "market": "CH",
        "target_id": target.get("target_id"),
        "delivery_month": delivery_month,
        "delivery_end_utc": end.isoformat().replace("+00:00", "Z"),
        "native_hourly_cadence": True,
        "quarter_hour_proxy": False,
        "native_hourly_interval_count": expected_count,
        "truth_opened_at_utc_untrusted": opened_at.isoformat().replace("+00:00", "Z"),
        "source_ledger": {
            "path": ledger_selected.relative_to(root).as_posix(),
            "sha256": expected_ledger_sha256,
            "ledger_id": ledger.get("ledger_id"),
            "recursive_reconstruction_verified": True,
        },
        "prediction_commitment": {
            "path": commitment_selected.relative_to(root).as_posix(),
            "sha256": expected_commitment_sha256,
            "commitment_id": commitment.get("commitment_id"),
        },
        "future_origin_selection": {
            "path": selection_selected.relative_to(root).as_posix(),
            "sha256": origin_selection_auditor.REGISTRY_SHA256,
            "selection_id": future_selection.get("selection_id"),
        },
        "observations": observations,
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
    }
    bundle_id = hashlib.sha256(_canonical(core)).hexdigest()
    return {**core, "truth_bundle_id": bundle_id}


def publish_truth_bundle(
    root: Path,
    output_directory: Path,
    **kwargs: object,
) -> dict[str, object]:
    root = assert_absolute_path_has_no_links(root)
    output_directory = Path(os.path.abspath(output_directory))
    if not _inside(output_directory, root / "build"):
        raise NativeHourlyTruthBundleError("output must remain below build/")
    if output_directory.exists() or not output_directory.parent.is_dir():
        raise NativeHourlyTruthBundleError(
            "output must be new and its parent must already exist"
        )
    assert_absolute_path_has_no_links(output_directory.parent)
    parent_identity = _directory_identity(output_directory.parent)
    document = build_truth_bundle(root, **kwargs)
    bundle_payload = _canonical(document)
    bundle_sha256 = hashlib.sha256(bundle_payload).hexdigest()
    receipt_core: dict[str, object] = {
        "schema_version": "ch_lt_local_truth_publication_receipt.v1",
        "status": "LOCAL_DIAGNOSTIC_TRUTH_PUBLICATION_RECEIPT_NO_GO",
        "test_fixture": False,
        "target_id": document["target_id"],
        "delivery_month": document["delivery_month"],
        "delivery_end_utc": document["delivery_end_utc"],
        "truth_opened_at_utc_untrusted": document[
            "truth_opened_at_utc_untrusted"
        ],
        "local_diagnostic_truth_read_occurred": True,
        "truth_bundle": {
            "path": (
                output_directory.relative_to(root) / "truth-bundle.json"
            ).as_posix(),
            "sha256": bundle_sha256,
        },
        "prediction_commitment": {
            "sha256": document["prediction_commitment"]["sha256"],
            "commitment_id": document["prediction_commitment"]["commitment_id"],
        },
        "hourly_scoring_contract": {
            "sha256": _SCORING_CONTRACT_SHA256,
            "contract_id": _SCORING_CONTRACT_ID,
        },
        "future_origin_selection": {
            "sha256": document["future_origin_selection"]["sha256"],
            "selection_id": document["future_origin_selection"]["selection_id"],
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
    receipt_id = hashlib.sha256(_canonical(receipt_core)).hexdigest()
    receipt_payload = _canonical({**receipt_core, "receipt_id": receipt_id})
    staging = output_directory.parent / f".{output_directory.name}.staging"
    expected_files = {
        "truth-bundle.json": bundle_payload,
        "truth-publication-receipt.json": receipt_payload,
    }
    if _directory_identity(output_directory.parent) != parent_identity:
        raise NativeHourlyTruthBundleError(
            "output parent identity changed during truth assembly"
        )
    if staging.exists():
        assert_absolute_path_has_no_links(staging)
        if not staging.is_dir() or any(
            child.name not in expected_files for child in staging.iterdir()
        ):
            raise NativeHourlyTruthBundleError("truth staging residue differs")
    else:
        staging.mkdir()
    assert_absolute_path_has_no_links(staging)
    for name, payload in expected_files.items():
        target = staging / name
        try:
            write_durable_exact_bytes(target, payload, label=f"native truth {name}")
        except DurableArtifactExistsError as exc:
            raise NativeHourlyTruthBundleError(
                f"native truth {name} staging residue differs"
            ) from exc
        if target.read_bytes() != payload:
            raise NativeHourlyTruthBundleError(f"native truth {name} differs after write")
    if set(path.name for path in staging.iterdir()) != set(expected_files):
        raise NativeHourlyTruthBundleError("truth staging file set differs")
    if _directory_identity(output_directory.parent) != parent_identity:
        raise NativeHourlyTruthBundleError(
            "output parent identity changed before truth publication"
        )
    os.rename(staging, output_directory)
    assert_absolute_path_has_no_links(output_directory)
    if _directory_identity(output_directory.parent) != parent_identity:
        raise NativeHourlyTruthBundleError(
            "output parent identity changed after truth publication"
        )
    for name, payload in expected_files.items():
        if read_stable_single_link_file(
            output_directory / name,
            label=f"published native truth {name}",
        ) != payload:
            raise NativeHourlyTruthBundleError(
                f"published native truth {name} verification differs"
            )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--expected-commitment-sha256", required=True)
    parser.add_argument("--delivery-month", required=True)
    parser.add_argument("--truth-opened-at-utc-untrusted", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = publish_truth_bundle(
        ROOT,
        args.output_directory,
        ledger_path=args.ledger,
        expected_ledger_sha256=args.expected_ledger_sha256,
        commitment_path=args.commitment,
        expected_commitment_sha256=args.expected_commitment_sha256,
        delivery_month=args.delivery_month,
        truth_opened_at_utc_untrusted=args.truth_opened_at_utc_untrusted,
    )
    print(_canonical(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
