"""Capture one hash-bound, value-blind ENTSO-E temporal diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_day_ahead_prd import (
    PROFILE_RESULT_ROW_LIMIT,
    build_day_ahead_profile_parameters,
)
from pfc_shaping.validation.entsoe_day_ahead_temporal_diagnostic import (
    COLUMNS,
    COUNT_COLUMNS,
    SIGNED_METRIC_COLUMNS,
    SQL_PATH,
    SQL_SHA256,
    assess_temporal_diagnostic,
    verify_sql_binding,
)
from scripts.capture_entsoe_day_ahead_prd_profile import (
    ROOT,
    CaptureError,
    _canonical_sha256,
    _load_credentials,
    _request_json,
    _resolve_output,
    _sha256_bytes,
    _utc_now,
    verify_capture_path,
)

SCHEMA_VERSION = "fmv_entsoe_day_ahead_temporal_diagnostic_capture.v1"
USER_AGENT = "fmv-entsoe-day-ahead-temporal-diagnostic/1"


def _signed_integer(value: object, column: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", str(value)) is None:
        raise CaptureError(f"{column} is not an integer or null")
    return int(str(value))


def _nonnegative_integer(value: object, column: str) -> int:
    result = _signed_integer(value, column)
    if result is None or result < 0:
        raise CaptureError(f"{column} is not a nonnegative integer")
    return result


def _diagnostic_frame(response: dict[str, Any]) -> pd.DataFrame:
    manifest = response.get("manifest")
    result = response.get("result")
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        raise CaptureError("Successful diagnostic response lacks manifest or result")
    if manifest.get("truncated") is not False:
        raise CaptureError("Diagnostic result is truncated or truncation is unknown")
    schema = manifest.get("schema")
    columns = schema.get("columns") if isinstance(schema, dict) else None
    if not isinstance(columns, list):
        raise CaptureError("Diagnostic result schema is missing")
    names = tuple(item.get("name") if isinstance(item, dict) else None for item in columns)
    if names != COLUMNS:
        raise CaptureError("Diagnostic result columns differ from the hash-bound contract")
    rows = result.get("data_array")
    if not isinstance(rows, list) or not rows:
        raise CaptureError("Diagnostic result is empty")
    if len(rows) >= PROFILE_RESULT_ROW_LIMIT:
        raise CaptureError("Diagnostic result reached the 101-row rejection sentinel")
    normalized: list[list[object]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(COLUMNS):
            raise CaptureError(f"Diagnostic row {position} has an invalid shape")
        normalized.append(
            [
                _nonnegative_integer(value, column)
                if column in COUNT_COLUMNS
                else _signed_integer(value, column)
                if column in SIGNED_METRIC_COLUMNS
                else value
                for column, value in zip(COLUMNS, row, strict=True)
            ]
        )
    return pd.DataFrame(normalized, columns=COLUMNS)


def _profile_binding(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    replay = verify_capture_path(path)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    payload = read_stable_single_link_file(
        candidate.resolve(strict=True),
        label="bound day-ahead profile capture",
        max_bytes=4_000_000,
    )
    document = json.loads(payload)
    counts = {
        row["series_key"]: {
            "row_count": row["vintage_row_count"],
            "invalid_availability_order_count": row["invalid_availability_order_count"],
            "invalid_interval_count": row["invalid_interval_count"],
        }
        for row in document["profile_rows"]
    }
    binding = {
        "relative_path": str(candidate.resolve(strict=True).relative_to(ROOT)).replace("\\", "/"),
        "file_sha256": replay["file_sha256"],
        "content_id": replay["content_id"],
        "window_parameters": document["query"]["parameters"],
    }
    return binding, counts


def _atomic_write(output_dir: Path, receipt: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "diagnostic.json"
    temporary = output_dir / "diagnostic.json.tmp"
    temporary.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def capture_diagnostic(
    *,
    start_utc: str,
    end_utc: str,
    profile_capture: str,
    output: str,
) -> dict[str, object]:
    verify_sql_binding()
    parameters = build_day_ahead_profile_parameters(
        window_start_utc=start_utc,
        window_end_utc=end_utc,
    )
    profile_binding, expected_counts = _profile_binding(profile_capture)
    if profile_binding["window_parameters"] != parameters:
        raise CaptureError("Diagnostic window differs from the bound profile capture")
    sql_bytes = SQL_PATH.read_bytes()
    if _sha256_bytes(sql_bytes) != SQL_SHA256:
        raise CaptureError("Temporal diagnostic SQL hash differs")
    sql_text = sql_bytes.decode("utf-8")
    output_dir = _resolve_output(output)
    host, warehouse_id, token = _load_credentials()
    warehouse = _request_json(
        method="GET",
        url=f"{host}/api/2.0/sql/warehouses/{warehouse_id}",
        token=token,
        user_agent=USER_AGENT,
    )
    if warehouse.get("state") != "RUNNING":
        raise CaptureError("Warehouse is not already RUNNING; diagnostic execution refused")
    request_parameters = [
        {
            "name": name,
            "value": str(value),
            "type": "TIMESTAMP" if name.endswith("_utc") else "INT",
        }
        for name, value in parameters.items()
    ]
    started = time.monotonic()
    response = _request_json(
        method="POST",
        url=f"{host}/api/2.0/sql/statements",
        token=token,
        body={
            "warehouse_id": warehouse_id,
            "statement": sql_text,
            "wait_timeout": "50s",
            "on_wait_timeout": "CANCEL",
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
            "parameters": request_parameters,
        },
        user_agent=USER_AGENT,
    )
    elapsed_seconds = round(time.monotonic() - started, 3)
    state = (response.get("status") or {}).get("state")
    if state != "SUCCEEDED":
        raise CaptureError(f"Temporal diagnostic did not succeed: {state or 'UNKNOWN'}")
    frame = _diagnostic_frame(response)
    assessment = assess_temporal_diagnostic(
        frame,
        window_start_utc=start_utc,
        window_end_utc=end_utc,
        query_sha256=SQL_SHA256,
        expected_profile_counts=expected_counts,
    ).as_dict()
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": _utc_now(),
        "profile_capture": profile_binding,
        "query": {
            "relative_path": str(SQL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": SQL_SHA256,
            "parameters": parameters,
            "business_value_columns_opened": 0,
            "result_row_limit": PROFILE_RESULT_ROW_LIMIT,
        },
        "warehouse": {
            "id_sha256": _sha256_bytes(warehouse_id.encode("utf-8")),
            "name": warehouse.get("name"),
            "state_before": warehouse.get("state"),
            "cluster_size": warehouse.get("cluster_size"),
            "warehouse_type": warehouse.get("warehouse_type"),
            "enable_serverless_compute": warehouse.get("enable_serverless_compute"),
            "min_num_clusters": warehouse.get("min_num_clusters"),
            "max_num_clusters": warehouse.get("max_num_clusters"),
            "auto_stop_mins": warehouse.get("auto_stop_mins"),
        },
        "execution": {
            "statement_id": response.get("statement_id"),
            "statement_status": state,
            "control_plane_get_count": 1,
            "databricks_statement_count": 1,
            "warehouse_start_count": 0,
            "databricks_write_count": 0,
            "retry_count": 0,
            "wait_timeout_seconds": 50,
            "on_wait_timeout": "CANCEL",
            "elapsed_seconds": elapsed_seconds,
            "result_row_count": len(frame),
            "result_truncated": False,
        },
        "diagnostic_rows": frame.where(pd.notna(frame), None).to_dict(orient="records"),
        "assessment": assessment,
        "authorities": {
            "bounded_pit_extraction": False,
            "model_input": False,
            "model_selection": False,
            "production": False,
        },
    }
    receipt["content_id"] = _canonical_sha256(receipt)
    output_path = _atomic_write(output_dir, receipt)
    return {
        "status": state,
        "evidence_status": assessment["evidence_status"],
        "source_quality_status": assessment["source_quality_status"],
        "cause_totals": assessment["metrics"]["cause_totals"],
        "result_row_count": len(frame),
        "elapsed_seconds": elapsed_seconds,
        "output": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "content_id": receipt["content_id"],
    }


def verify_diagnostic_capture_path(path: str | Path) -> dict[str, object]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    payload = read_stable_single_link_file(
        candidate.resolve(strict=True),
        label="temporal diagnostic capture",
        max_bytes=4_000_000,
    )
    try:
        candidate.resolve(strict=True).relative_to((ROOT / "build").resolve(strict=True))
        document = json.loads(payload)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError("Temporal diagnostic capture is not governed JSON") from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise CaptureError("Temporal diagnostic capture schema differs")
    content_id = document.get("content_id")
    unsigned = {key: value for key, value in document.items() if key != "content_id"}
    if not isinstance(content_id, str) or content_id != _canonical_sha256(unsigned):
        raise CaptureError("Temporal diagnostic capture content ID differs")
    query = document.get("query")
    if not isinstance(query, dict) or query.get("sha256") != SQL_SHA256:
        raise CaptureError("Temporal diagnostic capture query binding differs")
    profile = document.get("profile_capture")
    if not isinstance(profile, dict):
        raise CaptureError("Temporal diagnostic profile binding is missing")
    profile_path = ROOT / str(profile.get("relative_path", ""))
    replay, expected_counts = _profile_binding(profile_path)
    if replay != profile:
        raise CaptureError("Temporal diagnostic profile binding differs")
    rows = document.get("diagnostic_rows")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != set(COLUMNS) for row in rows
    ):
        raise CaptureError("Temporal diagnostic persisted rows differ")
    frame = pd.DataFrame(rows, columns=COLUMNS)
    parameters = query.get("parameters")
    if not isinstance(parameters, dict):
        raise CaptureError("Temporal diagnostic parameters differ")
    assessment = assess_temporal_diagnostic(
        frame,
        window_start_utc=parameters.get("start_utc"),
        window_end_utc=parameters.get("end_utc"),
        query_sha256=SQL_SHA256,
        expected_profile_counts=expected_counts,
    ).as_dict()
    if assessment != document.get("assessment"):
        raise CaptureError("Temporal diagnostic assessment does not replay exactly")
    execution = document.get("execution")
    if not isinstance(execution, dict) or any(
        execution.get(key) != value
        for key, value in {
            "statement_status": "SUCCEEDED",
            "control_plane_get_count": 1,
            "databricks_statement_count": 1,
            "warehouse_start_count": 0,
            "databricks_write_count": 0,
            "retry_count": 0,
            "result_row_count": len(frame),
            "result_truncated": False,
        }.items()
    ):
        raise CaptureError("Temporal diagnostic execution invariants differ")
    if document.get("authorities") != {
        "bounded_pit_extraction": False,
        "model_input": False,
        "model_selection": False,
        "production": False,
    }:
        raise CaptureError("Temporal diagnostic authorities differ")
    return {
        "status": "PASS_DIAGNOSTIC_CAPTURE_REPLAY",
        "content_id": content_id,
        "file_sha256": _sha256_bytes(payload),
        "source_quality_status": assessment["source_quality_status"],
        "cause_totals": assessment["metrics"]["cause_totals"],
        "result_row_count": len(frame),
        "databricks_statement_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--profile-capture")
    parser.add_argument("--output")
    parser.add_argument("--verify-capture")
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify_capture:
            if any(
                (
                    arguments.start_utc,
                    arguments.end_utc,
                    arguments.profile_capture,
                    arguments.output,
                )
            ):
                parser.error("--verify-capture cannot be combined with capture arguments")
            result = verify_diagnostic_capture_path(arguments.verify_capture)
        else:
            if not all(
                (
                    arguments.start_utc,
                    arguments.end_utc,
                    arguments.profile_capture,
                    arguments.output,
                )
            ):
                parser.error(
                    "capture requires --start-utc, --end-utc, --profile-capture and --output"
                )
            result = capture_diagnostic(
                start_utc=arguments.start_utc,
                end_utc=arguments.end_utc,
                profile_capture=arguments.profile_capture,
                output=arguments.output,
            )
    except (CaptureError, OSError, ValueError) as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
