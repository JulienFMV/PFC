"""Audit EEX workbook snapshot integrity for suspicious far-horizon zero wipes.

This is a read-only workbook audit. It checks whether the latest dated row for
one market suddenly zeros out far-horizon Cal/Q/M products that were quoted on
the previous dated row, while nearer-horizon products remain quoted. That
pattern is operationally suspicious and usually indicates a partial workbook
update rather than a true market state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.data.ingest_forwards import _normalize_product


SCHEMA_VERSION = "eex_report_snapshot_integrity.v1"


def audit_snapshot_integrity(
    *,
    workbook: Path,
    market: str = "CH",
    far_horizon_start_year: int | None = None,
    min_quoted_products: int = 1,
) -> dict[str, object]:
    raw = pd.read_excel(workbook, sheet_name=market, header=None)
    if raw.shape[0] < 5 or raw.shape[1] < 2:
        raise ValueError(f"unexpected workbook format in {workbook} sheet={market}")

    product_codes = raw.iloc[0, 1:]
    dates = _parse_workbook_dates(raw.iloc[3:, 0])
    valid_dates = dates[dates.notna()]
    if valid_dates.empty:
        raise ValueError(f"no dated rows found in {workbook} sheet={market}")
    distinct_dates = sorted(pd.Timestamp(d).normalize() for d in valid_dates.unique())
    latest_workbook_date = distinct_dates[-1]
    quote_counts = _quote_counts_by_date(raw, dates)
    usable_dates = [
        date
        for date in distinct_dates
        if quote_counts.get(date.date().isoformat(), 0) >= int(min_quoted_products)
    ]
    if not usable_dates:
        return {
            "schema_version": SCHEMA_VERSION,
            "workbook_path": str(workbook),
            "market": str(market).upper(),
            "latest_workbook_date": latest_workbook_date.date().isoformat(),
            "latest_date": latest_workbook_date.date().isoformat(),
            "latest_usable_date": "",
            "previous_date": "",
            "min_quoted_products": int(min_quoted_products),
            "far_horizon_start_year": int(far_horizon_start_year or (int(latest_workbook_date.year) + 3)),
            "status": "LATEST_SNAPSHOT_EMPTY",
            "reason": "workbook has no dated row meeting the minimum non-zero Cal/Q/M coverage",
            "latest_far_horizon_nonzero_count": 0,
            "previous_far_horizon_nonzero_count": 0,
            "latest_near_horizon_nonzero_count": 0,
            "zero_wipe_count": 0,
            "zero_wipe_products": [],
            "far_horizon_products": [],
            "quote_counts_by_date": quote_counts,
        }
    latest_date = usable_dates[-1]
    previous_date = usable_dates[-2] if len(usable_dates) >= 2 else None
    if previous_date is None:
        raise ValueError(f"need at least two dated rows in {workbook} sheet={market}")

    far_horizon_year = int(far_horizon_start_year or (int(latest_date.year) + 3))
    latest_row = raw.iloc[dates[dates.eq(latest_date)].index[-1], 1:]
    previous_row = raw.iloc[dates[dates.eq(previous_date)].index[-1], 1:]

    records: list[dict[str, object]] = []
    for code, latest_raw, previous_raw in zip(product_codes.tolist(), latest_row.tolist(), previous_row.tolist()):
        if pd.isna(code):
            continue
        code_text = str(code).strip().upper()
        parsed = _normalize_product(code_text)
        if parsed is None:
            continue
        delivery_key, load_type, product_type = parsed
        if product_type == "Week":
            continue
        delivery_year = int(delivery_key[:4])
        if delivery_year < far_horizon_year:
            continue
        latest_is_zero = pd.isna(latest_raw) or latest_raw == 0
        previous_is_zero = pd.isna(previous_raw) or previous_raw == 0
        records.append(
            {
                "product_code": code_text,
                "delivery_key": delivery_key,
                "delivery_year": delivery_year,
                "load_type": load_type,
                "product_type": product_type,
                "latest_raw": None if pd.isna(latest_raw) else latest_raw,
                "previous_raw": None if pd.isna(previous_raw) else previous_raw,
                "latest_is_zero": bool(latest_is_zero),
                "previous_is_zero": bool(previous_is_zero),
                "zero_wipe": bool(latest_is_zero and not previous_is_zero),
            }
        )

    frame = pd.DataFrame(records)
    zero_wipes = frame[frame["zero_wipe"]] if not frame.empty else pd.DataFrame()
    latest_nonzero_count = int((~frame["latest_is_zero"]).sum()) if not frame.empty else 0
    previous_nonzero_count = int((~frame["previous_is_zero"]).sum()) if not frame.empty else 0

    near_horizon_nonzero = _near_horizon_nonzero_count(raw, dates, latest_date)
    status = "PASS"
    reason = "no suspicious far-horizon zero wipe detected"
    if latest_nonzero_count == 0 and near_horizon_nonzero == 0:
        status = "LATEST_SNAPSHOT_EMPTY"
        reason = "latest workbook row has no non-zero Cal/Q/M products for this market"
    elif not zero_wipes.empty and near_horizon_nonzero > 0:
        status = "SUSPICIOUS_ZERO_WIPE"
        reason = (
            "latest row zeros far-horizon products that were quoted on the previous row "
            "while nearer-horizon products remain quoted"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "workbook_path": str(workbook),
        "market": str(market).upper(),
        "latest_workbook_date": latest_workbook_date.date().isoformat(),
        "latest_date": latest_date.date().isoformat(),
        "latest_usable_date": latest_date.date().isoformat(),
        "previous_date": previous_date.date().isoformat(),
        "min_quoted_products": int(min_quoted_products),
        "far_horizon_start_year": far_horizon_year,
        "status": status,
        "reason": reason,
        "latest_far_horizon_nonzero_count": latest_nonzero_count,
        "previous_far_horizon_nonzero_count": previous_nonzero_count,
        "latest_near_horizon_nonzero_count": int(near_horizon_nonzero),
        "zero_wipe_count": int(len(zero_wipes)),
        "zero_wipe_products": zero_wipes.to_dict("records") if not zero_wipes.empty else [],
        "far_horizon_products": frame.to_dict("records") if not frame.empty else [],
        "quote_counts_by_date": quote_counts,
    }


def _quote_counts_by_date(raw: pd.DataFrame, dates: pd.Series) -> dict[str, int]:
    product_codes = raw.iloc[0, 1:]
    counts: dict[str, int] = {}
    for row_idx in dates[dates.notna()].index:
        date = pd.Timestamp(dates.loc[row_idx]).normalize().date().isoformat()
        values = raw.iloc[row_idx, 1:]
        count = 0
        for code, raw_value in zip(product_codes.tolist(), values.tolist()):
            if pd.isna(code):
                continue
            parsed = _normalize_product(str(code).strip().upper())
            if parsed is None:
                continue
            _, _, product_type = parsed
            if product_type == "Week":
                continue
            if not (pd.isna(raw_value) or raw_value == 0):
                count += 1
        counts[date] = count
    return counts


def _parse_workbook_dates(values: pd.Series) -> pd.Series:
    parsed: list[pd.Timestamp | pd.NaT] = []
    for value in values.tolist():
        if pd.isna(value):
            parsed.append(pd.NaT)
            continue
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            parsed.append(pd.to_datetime(value.strip(), format="%Y-%m-%d", errors="coerce"))
            continue
        parsed.append(pd.to_datetime(value, dayfirst=True, errors="coerce"))
    return pd.Series(parsed, index=values.index)


def _near_horizon_nonzero_count(raw: pd.DataFrame, dates: pd.Series, latest_date: pd.Timestamp) -> int:
    product_codes = raw.iloc[0, 1:]
    latest_row = raw.iloc[dates[dates.eq(latest_date)].index[-1], 1:]
    count = 0
    for code, raw_value in zip(product_codes.tolist(), latest_row.tolist()):
        if pd.isna(code):
            continue
        code_text = str(code).strip().upper()
        parsed = _normalize_product(code_text)
        if parsed is None:
            continue
        delivery_key, _, product_type = parsed
        if product_type == "Week":
            continue
        delivery_year = int(delivery_key[:4])
        if delivery_year >= int(latest_date.year) + 3:
            continue
        if not (pd.isna(raw_value) or raw_value == 0):
            count += 1
    return count


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--market", default="CH")
    parser.add_argument("--far-horizon-start-year", type=int, default=None)
    parser.add_argument("--min-quoted-products", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = audit_snapshot_integrity(
        workbook=args.workbook,
        market=args.market,
        far_horizon_start_year=args.far_horizon_start_year,
        min_quoted_products=args.min_quoted_products,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
