"""Compare frozen HPFC and OMPEX forecasts against the same realised truth.

This is a local descriptive diagnostic only.  It never selects a timestamp
alignment, treats OMPEX as truth, or authorizes model selection/promotion.
All inputs must be hash-bound files below the canonical C: workspace and all
outputs must be written below ``build/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from io import BytesIO
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.validation.ompex_benchmark import parse_ompex_workbook
from pfc_shaping.validation.ompex_truth_comparison import (
    compare_hourly_forecasts_against_truth,
)

DEFAULT_HPFC_PRICE = "price_weighted_mean_eur_mwh"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
OFFSET_PATTERN = re.compile(r"^UTC(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})$")


def _workspace_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise ValueError("OMPEX comparison requires the canonical workspace as cwd")
    return root


def _has_reparse_ancestry(path: Path, *, stop: Path) -> bool:
    current = path
    while True:
        info = current.lstat()
        attributes = getattr(info, "st_file_attributes", 0)
        if current.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            return True
        if current == stop:
            return False
        current = current.parent


def _read_bound_local_bytes(
    path: Path, *, expected_sha256: str, label: str, root: Path
) -> tuple[Path, bytes]:
    digest = str(expected_sha256).lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} expected SHA-256 is invalid")
    selected = path.resolve(strict=True)
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be below the canonical workspace") from exc
    if not selected.is_file() or _has_reparse_ancestry(selected, stop=root):
        raise ValueError(f"{label} must be a regular non-reparse local file")
    before = selected.stat()
    payload = selected.read_bytes()
    after = selected.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    return selected, payload


def _series_from_frame(
    frame: pd.DataFrame,
    *,
    price_column: str,
    label: str,
) -> pd.Series:
    if price_column not in frame.columns:
        raise ValueError(f"{label} missing price column {price_column!r}")
    if "delivery_start_utc" in frame.columns:
        timestamps = pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="coerce")
    elif {"timestamp_ch", "utc_offset_ch"}.issubset(frame.columns):
        local = pd.to_datetime(frame["timestamp_ch"], dayfirst=True, errors="coerce")
        offsets: list[pd.Timedelta | pd.NaT] = []
        for raw in frame["utc_offset_ch"]:
            match = OFFSET_PATTERN.fullmatch(str(raw).strip())
            if match is None:
                offsets.append(pd.NaT)
                continue
            minutes = 60 * int(match.group("hours")) + int(match.group("minutes"))
            if match.group("sign") == "-":
                minutes = -minutes
            offsets.append(pd.Timedelta(minutes=minutes))
        utc_naive = local - pd.Series(offsets, index=frame.index)
        timestamps = pd.DatetimeIndex(utc_naive).tz_localize("UTC")
    else:
        raise ValueError(
            f"{label} requires delivery_start_utc or timestamp_ch plus utc_offset_ch"
        )

    prices = pd.to_numeric(frame[price_column], errors="coerce")
    series = pd.Series(
        prices.to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(timestamps, name="delivery_start_utc"),
        name=label,
    )
    if series.index.hasnans or not bool(np.isfinite(series.to_numpy()).all()):
        raise ValueError(f"{label} timestamps and prices must be finite and complete")
    if series.index.has_duplicates:
        raise ValueError(f"{label} timestamps must be unique")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps must be sorted")
    return series


def _csv_series(
    payload: bytes, *, price_column: str, label: str
) -> pd.Series:
    try:
        frame = pd.read_csv(BytesIO(payload), sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"{label} CSV cannot be parsed") from exc
    return _series_from_frame(frame, price_column=price_column, label=label)


def _prepare_output_dir(path: Path, *, root: Path) -> Path:
    build_root = (root / "build").resolve()
    output = path.resolve(strict=False)
    try:
        output.relative_to(build_root)
    except ValueError as exc:
        raise ValueError("OMPEX comparison output must be below build/") from exc
    if output.exists():
        raise ValueError("OMPEX comparison output directory must be new")
    output.mkdir(parents=True, exist_ok=False)
    return output


def compare(
    *,
    hpfc_csv: Path,
    hpfc_sha256: str,
    ompex_xlsx: Path,
    ompex_sha256: str,
    truth_csv: Path,
    truth_sha256: str,
    output_dir: Path,
    origin_utc: pd.Timestamp,
    target_start_utc: pd.Timestamp,
    target_end_utc: pd.Timestamp,
    price_col: str = DEFAULT_HPFC_PRICE,
    truth_price_col: str = "price_eur_mwh",
) -> dict[str, object]:
    """Run a hash-bound local descriptive triad comparison."""

    root = _workspace_root()
    hpfc_path, hpfc_payload = _read_bound_local_bytes(
        hpfc_csv, expected_sha256=hpfc_sha256, label="HPFC candidate", root=root
    )
    ompex_path, ompex_payload = _read_bound_local_bytes(
        ompex_xlsx, expected_sha256=ompex_sha256, label="OMPEX vintage", root=root
    )
    truth_path, truth_payload = _read_bound_local_bytes(
        truth_csv, expected_sha256=truth_sha256, label="realised truth", root=root
    )

    candidate = _csv_series(
        hpfc_payload, price_column=price_col, label="candidate"
    )
    truth = _csv_series(
        truth_payload, price_column=truth_price_col, label="truth"
    )
    parsed_ompex = parse_ompex_workbook(
        ompex_payload,
        source_name=ompex_path.name,
        accept_unconfirmed_hour_end_semantics=True,
    )
    ompex = parsed_ompex.data.set_index("delivery_start_utc")["price_eur_mwh"]

    result = compare_hourly_forecasts_against_truth(
        candidate=candidate,
        ompex=ompex,
        truth=truth,
        origin_utc=origin_utc,
        target_start_utc=target_start_utc,
        target_end_utc=target_end_utc,
        timestamp_semantics_authenticated=bool(
            parsed_ompex.receipt["hour_end_interpretation_vendor_authenticated"]
        ),
        truth_independence_authenticated=False,
    )
    output = _prepare_output_dir(output_dir, root=root)
    input_bindings: Mapping[str, object] = {
        "candidate": {
            "path": hpfc_path.relative_to(root).as_posix(),
            "sha256": hpfc_sha256.lower(),
        },
        "ompex": {
            "path": ompex_path.relative_to(root).as_posix(),
            "sha256": ompex_sha256.lower(),
            "receipt": dict(parsed_ompex.receipt),
        },
        "truth": {
            "path": truth_path.relative_to(root).as_posix(),
            "sha256": truth_sha256.lower(),
            "independence_authenticated": False,
        },
    }
    summary = {
        **dict(result.summary),
        "benchmark_policy": "advisory",
        "read_only_inputs": True,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "input_bindings": input_bindings,
    }
    result.aligned.reset_index().to_csv(
        output / "paired_hourly_against_truth.csv", index=False
    )
    result.subgroup_metrics.to_csv(output / "subgroup_metrics.csv", index=False)
    (output / "benchmark_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpfc-csv", type=Path, required=True)
    parser.add_argument("--hpfc-sha256", required=True)
    parser.add_argument("--ompex-xlsx", type=Path, required=True)
    parser.add_argument("--ompex-sha256", required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--truth-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--origin-utc", required=True)
    parser.add_argument("--target-start-utc", required=True)
    parser.add_argument("--target-end-utc", required=True)
    parser.add_argument("--price-col", default=DEFAULT_HPFC_PRICE)
    parser.add_argument("--truth-price-col", default="price_eur_mwh")
    args = parser.parse_args(argv)

    summary = compare(
        hpfc_csv=args.hpfc_csv,
        hpfc_sha256=args.hpfc_sha256,
        ompex_xlsx=args.ompex_xlsx,
        ompex_sha256=args.ompex_sha256,
        truth_csv=args.truth_csv,
        truth_sha256=args.truth_sha256,
        output_dir=args.output_dir,
        origin_utc=pd.Timestamp(args.origin_utc),
        target_start_utc=pd.Timestamp(args.target_start_utc),
        target_end_utc=pd.Timestamp(args.target_end_utc),
        price_col=args.price_col,
        truth_price_col=args.truth_price_col,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
