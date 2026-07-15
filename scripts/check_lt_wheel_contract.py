#!/usr/bin/env python3
"""Fail closed when an FMV LT wheel contains files outside its build contract."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import stat
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from pfc_shaping.package_contract import (
    ALLOWED_DIST_INFO_FILES,
    ALLOWED_RUNTIME_PYTHON_FILES,
    BUILD_IDENTITY_PLACEHOLDER,
    BUILD_IDENTITY_TEMPLATE,
    REPRODUCIBLE_BUILD_EPOCH,
)
from pfc_shaping.version import __version__

FORBIDDEN_PREFIXES = (
    ".planning/",
    "dashboard/",
    "data/",
    "powerbi/",
    "scripts/",
    "tests/",
    "pfc_shaping/ct/",
    "pfc_shaping/model/",
    "pfc_shaping/storage/",
    "pfc_shaping/tools/",
)
FORBIDDEN_MODULES = frozenset(
    {
        "pfc_shaping/pipeline/autoresearch.py",
        "pfc_shaping/pipeline/export_euler.py",
        "pfc_shaping/pipeline/rolling_update.py",
        "pfc_shaping/pipeline/structural_break.py",
        "pfc_shaping/pipeline/swiss_short_term.py",
        "pfc_shaping/validation/backtest.py",
        "pfc_shaping/validation/christoffersen.py",
        "pfc_shaping/validation/compare_hfc.py",
        "pfc_shaping/validation/dm_test.py",
        "pfc_shaping/validation/perfect_foresight.py",
        "pfc_shaping/validation/scorecard.py",
        "pfc_shaping/validation/structural_tests.py",
    }
)
FORBIDDEN_SUFFIXES = (
    ".csv",
    ".duckdb",
    ".env",
    ".key",
    ".log",
    ".parquet",
    ".pem",
    ".xlsx",
    ".yaml",
    ".yml",
)
REQUIRED_FILES = frozenset(
    {
        "pfc_shaping/version.py",
        "pfc_shaping/build_identity.py",
        "pfc_shaping/cli/governed_release.py",
        "pfc_shaping/pipeline/candidate_build.py",
        "pfc_shaping/pipeline/candidate_finalize.py",
        "pfc_shaping/pipeline/probabilistic_output_governance.py",
        "pfc_shaping/pipeline/release_request_contract.py",
        "pfc_shaping/validation/product_normalization.py",
        "pfc_shaping/calibration/monthly_curve_capstone.py",
    }
)
FORBIDDEN_SOURCE_MARKERS = (
    b"C:\\Users\\",
    b"C:/Users/",
    b"H:\\",
    b"H:/",
    b"\\\\fmvfs1\\",
    b"//fmvfs1/",
    b"dotenv",
)
WINDOWS_DRIVE_PATH = re.compile(rb"(?i)(?<![A-Za-z0-9])[a-z]:[\\/]")
UNC_PATH = re.compile(rb"(?:\\\\|(?<!:)//)[A-Za-z0-9_.-]+[\\/]")
DEVICE_PATH = re.compile(rb"(?i)(?:\\\\[.?]\\|//[.?]/)")
DEVICE_NAMESPACE_GUARD_SOURCE = "pfc_shaping/path_safety.py"
DEVICE_NAMESPACE_GUARD_LITERALS = (b"\\\\.\\", b"\\\\?\\")
EXPECTED_DIST_INFO = f"fmv_pfc_lt-{__version__}.dist-info"
EXPECTED_ZIP_DATETIME = datetime.fromtimestamp(
    REPRODUCIBLE_BUILD_EPOCH,
    tz=timezone.utc,
).timetuple()[:6]
EXPECTED_REQUIRES_DIST = frozenset(
    {
        "cryptography==47.0.0",
        "holidays==0.83",
        "numpy==2.0.2",
        "openpyxl==3.1.5",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
        "PyYAML==6.0.3",
        "scikit-learn==1.6.1",
        "scipy==1.13.1",
        'duckdb==1.5.0; extra == "dashboard"',
        'plotly==6.6.0; extra == "dashboard"',
        'streamlit==1.50.0; extra == "dashboard"',
        'matplotlib==3.8.4; extra == "validation"',
        'statsmodels==0.14.6; extra == "validation"',
    }
)


def audit_wheel(path: str | Path) -> dict[str, object]:
    wheel = Path(path).resolve(strict=True)
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("wheel path must identify one .whl file")
    wheel_bytes = wheel.read_bytes()
    wheel_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        case_collisions = sorted(
            {
                name.casefold()
                for name in names
                if sum(candidate.casefold() == name.casefold() for candidate in names) > 1
            }
        )
        unsafe_members = sorted(
            item.filename
            for item in infos
            if item.is_dir()
            or item.filename != PurePosixPath(item.filename).as_posix()
            or PurePosixPath(item.filename).is_absolute()
            or ".." in PurePosixPath(item.filename).parts
            or "\\" in item.filename
            or stat.S_ISLNK(item.external_attr >> 16)
        )
        timestamp_drift = sorted(
            item.filename
            for item in infos
            if not item.is_dir() and item.date_time != EXPECTED_ZIP_DATETIME
        )
        files = set(names)
        expected_members = set(ALLOWED_RUNTIME_PYTHON_FILES) | {
            f"{EXPECTED_DIST_INFO}/{name}" for name in ALLOWED_DIST_INFO_FILES
        }
        unexpected_members = sorted(files.difference(expected_members))
        missing_members = sorted(expected_members.difference(files))
        runtime_python_files = {
            name
            for name in files
            if name.startswith("pfc_shaping/") and name.endswith(".py")
        }
        forbidden = sorted(
            name
            for name in files
            if name in FORBIDDEN_MODULES
            or name.lower().endswith(FORBIDDEN_SUFFIXES)
            or name.startswith(FORBIDDEN_PREFIXES)
        )
        missing = sorted(REQUIRED_FILES.difference(files))
        unexpected_runtime_sources = sorted(
            runtime_python_files.difference(ALLOWED_RUNTIME_PYTHON_FILES)
        )
        missing_runtime_sources = sorted(
            ALLOWED_RUNTIME_PYTHON_FILES.difference(runtime_python_files)
        )
        safe_runtime_python_files = runtime_python_files.difference(unsafe_members)
        nonportable_sources = sorted(
            name
            for name in safe_runtime_python_files
            if _contains_nonportable_content(archive.read(name), source_name=name)
        )
        embedded_revision = _embedded_source_revision(archive, files)
        computed_revision = _runtime_source_revision(archive, files)
        metadata_errors = _metadata_errors(archive, files)
        wheel_metadata_errors = _wheel_metadata_errors(archive, files)
        entry_point_errors = _entry_point_errors(archive, files)
        record_errors = _record_errors(archive, files)
    errors: list[str] = []
    if duplicate_names:
        errors.append(f"duplicate wheel members: {duplicate_names}")
    if case_collisions:
        errors.append(f"case-colliding wheel members: {case_collisions}")
    if unsafe_members:
        errors.append(f"unsafe wheel members: {unsafe_members}")
    if unexpected_members:
        errors.append(f"unexpected wheel members: {unexpected_members}")
    if timestamp_drift:
        errors.append(f"non-reproducible wheel member timestamps: {timestamp_drift}")
    if missing_members:
        errors.append(f"missing wheel members: {missing_members}")
    if forbidden:
        errors.append(f"forbidden wheel members: {forbidden}")
    if missing:
        errors.append(f"missing runtime members: {missing}")
    if unexpected_runtime_sources:
        errors.append(f"unexpected runtime sources: {unexpected_runtime_sources}")
    if missing_runtime_sources:
        errors.append(f"missing allowlisted runtime sources: {missing_runtime_sources}")
    if nonportable_sources:
        errors.append(f"non-portable runtime paths: {nonportable_sources}")
    if embedded_revision != computed_revision:
        errors.append("embedded source revision does not match wheel runtime sources")
    errors.extend(metadata_errors)
    errors.extend(wheel_metadata_errors)
    errors.extend(entry_point_errors)
    errors.extend(record_errors)
    return {
        "schema_version": "fmv_pfc_lt_wheel_contract.v2",
        "wheel": str(wheel),
        "wheel_sha256": wheel_sha256,
        "member_count": len(files),
        "source_revision": embedded_revision,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "promotion_eligible": False,
    }


def _embedded_source_revision(
    archive: zipfile.ZipFile,
    files: set[str],
) -> str | None:
    identity = "pfc_shaping/build_identity.py"
    if identity not in files:
        return None
    payload = archive.read(identity)
    match = re.fullmatch(
        rb'"""Build-generated identity of the installed governed LT runtime\."""\n\n'
        rb'SOURCE_REVISION = "([0-9a-f]{64})"\n',
        payload,
    )
    if match is None:
        return None
    revision = match.group(1).decode("ascii")
    expected = BUILD_IDENTITY_TEMPLATE.format(source_revision=revision).encode("ascii")
    return revision if payload == expected else None


def _runtime_source_revision(
    archive: zipfile.ZipFile,
    files: set[str],
) -> str:
    digest = hashlib.sha256()
    prefix = "pfc_shaping/"
    for name in sorted(files):
        if not name.startswith(prefix) or not name.endswith(".py"):
            continue
        relative = name.removeprefix(prefix)
        payload = archive.read(name)
        if name == "pfc_shaping/build_identity.py":
            payload = BUILD_IDENTITY_TEMPLATE.format(
                source_revision=BUILD_IDENTITY_PLACEHOLDER
            ).encode("ascii")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _contains_nonportable_content(payload: bytes, *, source_name: str) -> bool:
    scanned = payload
    if source_name == DEVICE_NAMESPACE_GUARD_SOURCE and all(
        payload.count(literal) == 1 for literal in DEVICE_NAMESPACE_GUARD_LITERALS
    ):
        for literal in DEVICE_NAMESPACE_GUARD_LITERALS:
            scanned = scanned.replace(literal, b"")
    lowered = scanned.lower()
    return (
        any(marker.lower() in lowered for marker in FORBIDDEN_SOURCE_MARKERS)
        or WINDOWS_DRIVE_PATH.search(scanned) is not None
        or UNC_PATH.search(scanned) is not None
        or DEVICE_PATH.search(scanned) is not None
    )


def _metadata_errors(archive: zipfile.ZipFile, files: set[str]) -> list[str]:
    name = f"{EXPECTED_DIST_INFO}/METADATA"
    if name not in files:
        return ["wheel METADATA is missing"]
    message = BytesParser().parsebytes(archive.read(name))
    errors: list[str] = []
    expected_single = {
        "Metadata-Version": "2.4",
        "Name": "fmv-pfc-lt",
        "Version": __version__,
        "Summary": "Governed Swiss long-term price forward curve pipeline for FMV",
        "Author": "FMV SA",
        "License-Expression": "LicenseRef-Proprietary",
        "Requires-Python": "<3.12,>=3.11",
        "Description-Content-Type": "text/markdown",
    }
    allowed_headers = set(expected_single) | {"Requires-Dist", "Provides-Extra"}
    unexpected_headers = sorted(set(message.keys()).difference(allowed_headers))
    if unexpected_headers:
        errors.append(f"unexpected METADATA headers: {unexpected_headers}")
    for key, expected in expected_single.items():
        values = message.get_all(key, [])
        if values != [expected]:
            errors.append(f"invalid METADATA {key}")
    if frozenset(message.get_all("Requires-Dist", [])) != EXPECTED_REQUIRES_DIST:
        errors.append("invalid METADATA dependency inventory")
    if message.get_all("Provides-Extra", []) != ["dashboard", "validation"]:
        errors.append("invalid METADATA extras inventory")
    description = str(message.get_payload()).replace("\r\n", "\n")
    if not description.startswith("# FMV PFC LT\n"):
        errors.append("invalid METADATA description")
    return errors


def _wheel_metadata_errors(archive: zipfile.ZipFile, files: set[str]) -> list[str]:
    name = f"{EXPECTED_DIST_INFO}/WHEEL"
    if name not in files:
        return ["wheel WHEEL metadata is missing"]
    message = BytesParser().parsebytes(archive.read(name))
    expected = {
        "Wheel-Version": ["1.0"],
        "Generator": ["setuptools (80.9.0)"],
        "Root-Is-Purelib": ["true"],
        "Tag": ["py3-none-any"],
    }
    if set(message.keys()) != set(expected):
        return ["invalid WHEEL header inventory"]
    return [
        f"invalid WHEEL {key}"
        for key, values in expected.items()
        if message.get_all(key, []) != values
    ]


def _entry_point_errors(archive: zipfile.ZipFile, files: set[str]) -> list[str]:
    entry_name = f"{EXPECTED_DIST_INFO}/entry_points.txt"
    top_level_name = f"{EXPECTED_DIST_INFO}/top_level.txt"
    if entry_name not in files or top_level_name not in files:
        return ["canonical pfc-lt entry point metadata is missing"]
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(archive.read(entry_name).decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error):
        return ["canonical pfc-lt entry point metadata is invalid"]
    if parser.sections() != ["console_scripts"] or dict(parser["console_scripts"]) != {
        "pfc-lt": "pfc_shaping.cli.governed_release:main"
    }:
        return ["canonical pfc-lt entry point is missing or ambiguous"]
    if archive.read(top_level_name) != b"pfc_shaping\n":
        return ["wheel top-level package metadata is invalid"]
    return []


def _record_errors(archive: zipfile.ZipFile, files: set[str]) -> list[str]:
    record_name = f"{EXPECTED_DIST_INFO}/RECORD"
    if record_name not in files:
        return ["wheel RECORD is missing"]
    try:
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error):
        return ["wheel RECORD is invalid"]
    if any(len(row) != 3 for row in rows):
        return ["wheel RECORD row shape is invalid"]
    names = [row[0] for row in rows]
    if len(names) != len(set(names)) or set(names) != files:
        return ["wheel RECORD inventory mismatch"]
    errors: list[str] = []
    for member, digest, size in rows:
        if member == record_name:
            if digest or size:
                errors.append("wheel RECORD self-entry must not be hashed")
            continue
        payload = archive.read(member)
        expected_digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        if digest != f"sha256={expected_digest.decode('ascii')}" or size != str(len(payload)):
            errors.append(f"wheel RECORD hash mismatch: {member}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    result = audit_wheel(args.wheel)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
