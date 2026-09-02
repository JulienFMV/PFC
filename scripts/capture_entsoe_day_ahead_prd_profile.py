"""Capture one bounded, value-blind ENTSO-E PRD profile from Databricks.

The command is intentionally single-purpose and single-shot. It refuses to
start a stopped warehouse, executes the hash-bound aggregate profile exactly
once, and writes only non-price metadata below ``build/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import certifi
import pandas as pd

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_day_ahead_prd import (
    PROFILE_COLUMNS,
    PROFILE_RESULT_ROW_LIMIT,
    PROFILE_SQL_PATH,
    PROFILE_SQL_SHA256,
    SERIES_INVENTORY_SPEC,
    assess_day_ahead_prd_profile,
    build_day_ahead_profile_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build"
ENV_PATH = ROOT / ".env"
WAREHOUSE_PATH = re.compile(r"^/sql/1\.0/warehouses/([A-Za-z0-9_-]+)$")
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
INTEGER_COLUMNS = frozenset(
    {
        "series_id",
        "vintage_row_count",
        "distinct_interval_count",
        "null_value_count",
        "dq_failed_count",
        "unknown_availability_count",
        "invalid_availability_order_count",
        "invalid_interval_count",
        "canonical_series_key_mismatch_count",
        "profile_row_count",
        "duplicate_vintage_key_count",
        "orphan_series_key_count",
        "gold_series_key_duplicate_count",
        "latest_grain_duplicate_count",
        "legacy_new_overlap_interval_count",
    }
)


class CaptureError(RuntimeError):
    """Raised when the live capture cannot prove its safety contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _load_credentials() -> tuple[str, str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    host = os.environ.get("DATABRICKS_HOST") or values.get("DATABRICKS_HOST", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH") or values.get("DATABRICKS_HTTP_PATH", "")
    token = os.environ.get("DATABRICKS_TOKEN") or values.get("DATABRICKS_TOKEN", "")
    if not host or not http_path or not token:
        raise CaptureError("Databricks host, HTTP path and token are required")
    parsed = urlparse(host if "://" in host else f"https://{host}")
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise CaptureError("Databricks host must be a credential-free HTTPS hostname")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise CaptureError("Databricks host must not contain a path, query or fragment")
    match = WAREHOUSE_PATH.fullmatch(http_path)
    if match is None:
        raise CaptureError("DATABRICKS_HTTP_PATH is not a SQL warehouse path")
    return f"https://{parsed.hostname}", match.group(1), token


def _request_json(
    *,
    method: str,
    url: str,
    token: str,
    body: dict[str, object] | None = None,
    timeout_seconds: int = 65,
    user_agent: str = "fmv-entsoe-day-ahead-value-blind-profile/1",
) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request,
            timeout=timeout_seconds,
            context=TLS_CONTEXT,
        ) as response:
            raw = response.read(4_000_001)
    except urllib.error.HTTPError as exc:
        raise CaptureError(f"Databricks API returned HTTP {exc.code}") from exc
    except (ssl.SSLError, urllib.error.URLError, TimeoutError) as exc:
        raise CaptureError("Databricks API request failed or timed out") from exc
    if len(raw) > 4_000_000:
        raise CaptureError("Databricks API response exceeds 4 MB")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError("Databricks API response is not valid JSON") from exc
    if not isinstance(document, dict):
        raise CaptureError("Databricks API response must be an object")
    return document


def _coerce_integer(value: object, column: str) -> int:
    if isinstance(value, bool) or value is None:
        raise CaptureError(f"{column} is not an integer")
    text = str(value)
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise CaptureError(f"{column} is not a nonnegative integer")
    return int(text)


def _profile_frame(response: dict[str, Any]) -> pd.DataFrame:
    manifest = response.get("manifest")
    result = response.get("result")
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        raise CaptureError("Successful response lacks manifest or result")
    if manifest.get("truncated") is not False:
        raise CaptureError("Profile result is truncated or truncation is unknown")
    schema = manifest.get("schema")
    columns = schema.get("columns") if isinstance(schema, dict) else None
    if not isinstance(columns, list):
        raise CaptureError("Profile result schema is missing")
    names = tuple(item.get("name") if isinstance(item, dict) else None for item in columns)
    if names != PROFILE_COLUMNS:
        raise CaptureError("Profile result columns differ from the hash-bound contract")
    rows = result.get("data_array")
    if not isinstance(rows, list) or not rows:
        raise CaptureError("Profile result is empty")
    if len(rows) >= PROFILE_RESULT_ROW_LIMIT:
        raise CaptureError("Profile result reached the 101-row rejection sentinel")
    normalized: list[list[object]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(PROFILE_COLUMNS):
            raise CaptureError(f"Profile row {position} has an invalid shape")
        normalized.append(
            [
                _coerce_integer(value, column) if column in INTEGER_COLUMNS else value
                for column, value in zip(PROFILE_COLUMNS, row, strict=True)
            ]
        )
    return pd.DataFrame(normalized, columns=PROFILE_COLUMNS)


def _resolve_output(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    resolved = output.resolve(strict=False)
    try:
        resolved.relative_to(BUILD_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CaptureError("Output must be a new directory below repo-local build") from exc
    if resolved.exists():
        raise CaptureError("Output directory already exists")
    return resolved


def _inventory_binding() -> dict[str, str]:
    return {
        slot: f"day_ahead_prices||{field}"
        + (f"||{classification}" if classification is not None else "")
        for slot, field, classification in SERIES_INVENTORY_SPEC
    }


def verify_capture_path(path: str | Path) -> dict[str, object]:
    """Replay one persisted capture without contacting Databricks."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(BUILD_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise CaptureError("Capture must be below repo-local build") from exc
    try:
        payload = read_stable_single_link_file(
            resolved,
            label="day-ahead live capture",
            max_bytes=4_000_000,
        )
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CaptureError("Capture is not a stable bounded JSON file") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "captured_at_utc",
        "query",
        "warehouse",
        "execution",
        "profile_rows",
        "assessment",
        "authorities",
        "content_id",
    }:
        raise CaptureError("Capture envelope differs")
    if document["schema_version"] != "fmv_entsoe_day_ahead_prd_live_capture.v1":
        raise CaptureError("Capture schema differs")
    content_id = document["content_id"]
    unsigned = {key: value for key, value in document.items() if key != "content_id"}
    if not isinstance(content_id, str) or content_id != _canonical_sha256(unsigned):
        raise CaptureError("Capture content ID differs")

    query = document["query"]
    if not isinstance(query, dict) or set(query) != {
        "relative_path",
        "sha256",
        "parameters",
        "value_columns_opened",
        "result_row_limit",
    }:
        raise CaptureError("Capture query binding differs")
    if query["relative_path"] != "docs/data/sql/databricks_prd_entsoe_day_ahead_profile.sql":
        raise CaptureError("Capture query path differs")
    if query["sha256"] != PROFILE_SQL_SHA256:
        raise CaptureError("Capture query hash differs")
    if query["value_columns_opened"] != 0 or query["result_row_limit"] != (
        PROFILE_RESULT_ROW_LIMIT
    ):
        raise CaptureError("Capture value/row fence differs")
    parameters = query["parameters"]
    if not isinstance(parameters, dict):
        raise CaptureError("Capture query parameters differ")
    expected_parameters = build_day_ahead_profile_parameters(
        window_start_utc=parameters.get("start_utc"),
        window_end_utc=parameters.get("end_utc"),
    )
    if parameters != expected_parameters:
        raise CaptureError("Capture partition parameters differ")

    rows = document["profile_rows"]
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) >= PROFILE_RESULT_ROW_LIMIT
        or any(not isinstance(row, dict) or set(row) != set(PROFILE_COLUMNS) for row in rows)
    ):
        raise CaptureError("Capture profile rows differ")
    frame = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    expected_assessment = assess_day_ahead_prd_profile(
        frame,
        window_start_utc=parameters["start_utc"],
        window_end_utc=parameters["end_utc"],
        assessed_at_utc=document["captured_at_utc"],
        profile_query_sha256=PROFILE_SQL_SHA256,
        series_inventory_binding=_inventory_binding(),
        rebuild_evidence=None,
    ).as_dict()
    if document["assessment"] != expected_assessment:
        raise CaptureError("Capture assessment does not replay exactly")
    execution = document["execution"]
    if not isinstance(execution, dict):
        raise CaptureError("Capture execution receipt differs")
    expected_execution = {
        "statement_status": "SUCCEEDED",
        "control_plane_get_count": 1,
        "databricks_statement_count": 1,
        "warehouse_start_count": 0,
        "databricks_write_count": 0,
        "retry_count": 0,
        "wait_timeout_seconds": 50,
        "on_wait_timeout": "CANCEL",
        "result_row_count": len(frame),
        "result_truncated": False,
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise CaptureError("Capture execution invariants differ")
    authorities = document["authorities"]
    if authorities != {
        "bounded_pit_extraction": expected_assessment["authorities"][
            "bounded_pit_extraction_authorized"
        ],
        "model_input": False,
        "model_selection": False,
        "production": False,
    }:
        raise CaptureError("Capture authorities differ")
    return {
        "status": "PASS_CAPTURE_REPLAY",
        "content_id": content_id,
        "file_sha256": _sha256_bytes(payload),
        "assessment_status": expected_assessment["status"],
        "finding_codes": [item["code"] for item in expected_assessment["findings"]],
        "result_row_count": len(frame),
        "databricks_statement_count": 0,
    }


def capture(*, start_utc: str, end_utc: str, output: str) -> dict[str, object]:
    parameters = build_day_ahead_profile_parameters(
        window_start_utc=start_utc,
        window_end_utc=end_utc,
    )
    sql_bytes = PROFILE_SQL_PATH.read_bytes()
    if _sha256_bytes(sql_bytes) != PROFILE_SQL_SHA256:
        raise CaptureError("Profile SQL hash differs")
    sql_text = sql_bytes.decode("utf-8")
    if "price_eur_per_mwh" in sql_text.lower():
        raise CaptureError("Value-blind profile unexpectedly selects a price")
    output_dir = _resolve_output(output)
    host, warehouse_id, token = _load_credentials()
    warehouse = _request_json(
        method="GET",
        url=f"{host}/api/2.0/sql/warehouses/{warehouse_id}",
        token=token,
    )
    if warehouse.get("state") != "RUNNING":
        raise CaptureError("Warehouse is not already RUNNING; SQL execution refused")

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
    )
    elapsed_seconds = round(time.monotonic() - started, 3)
    state = (response.get("status") or {}).get("state")
    if state != "SUCCEEDED":
        raise CaptureError(f"Profile statement did not succeed: {state or 'UNKNOWN'}")

    frame = _profile_frame(response)
    captured_at = _utc_now()
    assessment = assess_day_ahead_prd_profile(
        frame,
        window_start_utc=start_utc,
        window_end_utc=end_utc,
        assessed_at_utc=captured_at,
        profile_query_sha256=PROFILE_SQL_SHA256,
        series_inventory_binding=_inventory_binding(),
        rebuild_evidence=None,
    ).as_dict()
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    receipt: dict[str, object] = {
        "schema_version": "fmv_entsoe_day_ahead_prd_live_capture.v1",
        "captured_at_utc": captured_at,
        "query": {
            "relative_path": str(PROFILE_SQL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": PROFILE_SQL_SHA256,
            "parameters": parameters,
            "value_columns_opened": 0,
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
        "profile_rows": rows,
        "assessment": assessment,
        "authorities": {
            "bounded_pit_extraction": assessment["authorities"][
                "bounded_pit_extraction_authorized"
            ],
            "model_input": False,
            "model_selection": False,
            "production": False,
        },
    }
    receipt["content_id"] = _canonical_sha256(receipt)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "capture.json"
    temporary = output_dir / "capture.json.tmp"
    temporary.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return {
        "status": state,
        "assessment_status": assessment["status"],
        "finding_codes": [item["code"] for item in assessment["findings"]],
        "result_row_count": len(frame),
        "elapsed_seconds": elapsed_seconds,
        "output": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "content_id": receipt["content_id"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--output")
    parser.add_argument("--verify-capture")
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify_capture:
            if any((arguments.start_utc, arguments.end_utc, arguments.output)):
                parser.error("--verify-capture cannot be combined with capture arguments")
            result = verify_capture_path(arguments.verify_capture)
        else:
            if not all((arguments.start_utc, arguments.end_utc, arguments.output)):
                parser.error("capture requires --start-utc, --end-utc and --output")
            result = capture(
                start_utc=arguments.start_utc,
                end_utc=arguments.end_utc,
                output=arguments.output,
            )
    except CaptureError as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
