"""Read-only CH LT product-normalization audit.

The audit compares a delivered hourly CH curve against traceable EEX CH
forwards. It fails closed: missing evidence, partial product windows, broken
direct quotes, timestamp defects, and conflicting duplicate quotes are reported
as gates rather than silently repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets  # noqa: E402
from pfc_shaping.lt.model.shape_constraints import eex_peak_mask  # noqa: E402

LOCAL_TZ = "Europe/Zurich"
UTC = "UTC"
DEFAULT_TIMESTAMP_COLUMN = "timestamp_utc"
DEFAULT_PRICE_COLUMN = "price_weighted_mean_eur_mwh"
REQUIRED_FORWARD_COLUMNS = {"market", "load_type", "date", "product", "price"}
PRODUCT_RE = re.compile(r"^(?P<year>\d{4})(?:-(?:(?P<month>\d{2})|Q(?P<quarter>[1-4])))?$")


@dataclass(frozen=True)
class Product:
    key: str
    year: int
    month: int | None = None
    quarter: int | None = None

    @property
    def granularity(self) -> str:
        if self.month is not None:
            return "MONTH"
        if self.quarter is not None:
            return "QUARTER"
        return "CALENDAR"

    @property
    def start_local(self) -> pd.Timestamp:
        if self.month is not None:
            return pd.Timestamp(year=self.year, month=self.month, day=1, tz=LOCAL_TZ)
        if self.quarter is not None:
            return pd.Timestamp(year=self.year, month=(self.quarter - 1) * 3 + 1, day=1, tz=LOCAL_TZ)
        return pd.Timestamp(year=self.year, month=1, day=1, tz=LOCAL_TZ)

    @property
    def end_local(self) -> pd.Timestamp:
        if self.month is not None:
            return self.start_local + pd.DateOffset(months=1)
        if self.quarter is not None:
            return self.start_local + pd.DateOffset(months=3)
        return self.start_local + pd.DateOffset(years=1)

    @property
    def start_utc(self) -> pd.Timestamp:
        return self.start_local.tz_convert(UTC)

    @property
    def end_utc(self) -> pd.Timestamp:
        return self.end_local.tz_convert(UTC)

    @property
    def expected_utc_index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start_utc, self.end_utc - pd.Timedelta(hours=1), freq="h", tz=UTC)


def parse_product(value: str) -> Product | None:
    text = str(value).strip()
    match = PRODUCT_RE.match(text)
    if match is None:
        return None
    year = int(match.group("year"))
    month = match.group("month")
    quarter = match.group("quarter")
    if month is not None:
        month_i = int(month)
        if not 1 <= month_i <= 12:
            return None
        return Product(key=text, year=year, month=month_i)
    if quarter is not None:
        return Product(key=text, year=year, quarter=int(quarter))
    return Product(key=text, year=year)


def gate(
    status: str,
    gate_name: str,
    *,
    load_type: str | None = None,
    product: str | None = None,
    bucket: str | None = None,
    message: str = "",
    **values: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "status": status,
        "gate": gate_name,
        "load_type": load_type or "",
        "product": product or "",
        "bucket": bucket or "",
        "message": message,
    }
    row.update(values)
    return row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format for {path}; expected .csv or .parquet")


def parse_utc_timestamp_series(values: pd.Series) -> tuple[pd.DatetimeIndex | None, list[dict[str, object]]]:
    parsed: list[pd.Timestamp] = []
    failures: list[str] = []
    non_utc = 0
    naive = 0
    for pos, raw in enumerate(values.astype(str).tolist()):
        try:
            ts = pd.Timestamp(raw)
        except Exception as exc:  # pragma: no cover - exact parser errors vary by pandas
            failures.append(f"row {pos}: {exc}")
            continue
        if ts.tzinfo is None or ts.utcoffset() is None:
            naive += 1
            continue
        if ts.utcoffset() != pd.Timedelta(0):
            non_utc += 1
            continue
        parsed.append(ts.tz_convert(UTC))
    gates: list[dict[str, object]] = []
    if failures:
        gates.append(gate("CRITICAL", "timestamp_parse", message="timestamp parsing failed", failure_count=len(failures), first_failure=failures[0]))
    if naive:
        gates.append(gate("CRITICAL", "timestamp_utc_timezone", message="timestamp column contains timezone-naive values", row_count=naive))
    if non_utc:
        gates.append(gate("CRITICAL", "timestamp_utc_timezone", message="timestamp column contains non-UTC timezone offsets", row_count=non_utc))
    if gates:
        return None, gates
    return pd.DatetimeIndex(parsed, tz=UTC), []


def load_hourly_curve(
    csv_path: Path,
    *,
    timestamp_column: str,
    price_column: str,
) -> tuple[pd.DataFrame | None, list[dict[str, object]]]:
    gates: list[dict[str, object]] = []
    df = pd.read_csv(csv_path)
    missing = [column for column in [timestamp_column, price_column] if column not in df.columns]
    if missing:
        return None, [gate("CRITICAL", "csv_schema", message="CSV missing required columns", missing_columns=",".join(missing))]
    idx, ts_gates = parse_utc_timestamp_series(df[timestamp_column])
    gates.extend(ts_gates)
    prices = pd.to_numeric(df[price_column], errors="coerce")
    if prices.isna().any():
        gates.append(gate("CRITICAL", "price_column", message="price column contains non-numeric or NaN values", row_count=int(prices.isna().sum())))
    if idx is None:
        return None, gates
    if len(idx) != len(df):
        gates.append(gate("CRITICAL", "timestamp_parse", message="parsed timestamp count does not match CSV row count", parsed_rows=len(idx), csv_rows=len(df)))
        return None, gates
    if idx.has_duplicates:
        gates.append(gate("CRITICAL", "timestamp_duplicate", message="CSV contains duplicate UTC timestamps", duplicate_count=int(idx.duplicated().sum())))
    if not idx.is_monotonic_increasing:
        gates.append(gate("CRITICAL", "timestamp_monotone", message="UTC timestamps are not strictly monotone increasing"))
    if len(idx) > 1 and idx.is_monotonic_increasing and not idx.has_duplicates:
        expected = pd.date_range(idx[0], idx[-1], freq="h", tz=UTC)
        if not idx.equals(expected):
            gates.append(
                gate(
                    "CRITICAL",
                    "timestamp_hourly_frequency",
                    message="UTC timestamps are not a complete hourly range",
                    expected_rows=len(expected),
                    actual_rows=len(idx),
                    missing_rows=int(len(expected.difference(idx))),
                )
            )
    elif len(idx) == 0:
        gates.append(gate("CRITICAL", "csv_empty", message="delivered hourly CSV is empty"))
    out = df.copy()
    out["_timestamp_utc"] = idx
    out["_timestamp_ch"] = idx.tz_convert(LOCAL_TZ)
    out["_price"] = prices.astype(float)
    return out, gates


def select_forwards_snapshot(
    forwards_path: Path,
    *,
    market: str,
    as_of: str | None,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.Timestamp | None, list[dict[str, object]]]:
    gates: list[dict[str, object]] = []
    frame = read_table(forwards_path)
    missing = sorted(REQUIRED_FORWARD_COLUMNS - set(frame.columns))
    if missing:
        return pd.DataFrame(), None, [gate("CRITICAL", "forward_schema", message="forwards source missing required columns", missing_columns=",".join(missing))]
    sub = frame[frame["market"].astype(str).str.upper() == market.upper()].copy()
    if sub.empty:
        return pd.DataFrame(), None, [gate("CRITICAL", "forward_evidence_empty", message=f"no forwards for market={market!r}")]
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["load_type"] = sub["load_type"].astype(str).str.upper().str.strip()
    sub["product"] = sub["product"].astype(str).str.strip()
    sub["price"] = pd.to_numeric(sub["price"], errors="coerce")
    bad_rows = sub["date"].isna() | sub["price"].isna()
    if bool(bad_rows.any()):
        gates.append(gate("CRITICAL", "forward_schema", message="forwards source contains invalid date or price values", row_count=int(bad_rows.sum())))
        sub = sub[~bad_rows].copy()
    if sub.empty:
        gates.append(gate("CRITICAL", "forward_evidence_empty", message="no valid forwards after schema validation"))
        return pd.DataFrame(), None, gates
    snapshot = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(sub["date"].max())
    snap = sub[sub["date"] == snapshot].copy()
    if snap.empty:
        gates.append(gate("CRITICAL", "forward_evidence_empty", message="selected forwards snapshot is empty", snapshot=str(snapshot)))
        return pd.DataFrame(), snapshot, gates
    for (load_type, product), group in snap.groupby(["load_type", "product"]):
        prices = group["price"].astype(float).to_numpy()
        if len(prices) <= 1:
            continue
        if bool(np.all(prices == prices[0])):
            gates.append(
                gate(
                    "PASS",
                    "duplicate_exact_quote",
                    load_type=str(load_type),
                    product=str(product),
                    message="duplicate quote rows carry the exact same price",
                    duplicate_rows=int(len(prices)),
                )
            )
        else:
            gates.append(
                gate(
                    "CRITICAL",
                    "duplicate_conflicting_quote",
                    load_type=str(load_type),
                    product=str(product),
                    message="conflicting duplicate quotes for same as-of/product/load_type",
                    min_price_eur_mwh=float(np.min(prices)),
                    max_price_eur_mwh=float(np.max(prices)),
                    duplicate_rows=int(len(prices)),
                )
            )
    deduped = (
        snap.sort_values(["load_type", "product", "price"])
        .drop_duplicates(["load_type", "product"], keep="first")
        .reset_index(drop=True)
    )
    invalid_products = [p for p in deduped["product"].astype(str).unique() if parse_product(p) is None]
    for product in sorted(invalid_products):
        gates.append(gate("UNSUPPORTED", "unsupported_product_code", product=product, message="product code is not YYYY, YYYY-Qn, or YYYY-MM"))
    deduped = deduped[~deduped["product"].isin(invalid_products)].copy()
    if deduped.empty:
        gates.append(gate("CRITICAL", "forward_evidence_empty", message="no supported forwards in selected snapshot"))
    return deduped, snapshot, gates


def product_mask(index_utc: pd.DatetimeIndex, product: Product) -> np.ndarray:
    return (index_utc >= product.start_utc) & (index_utc < product.end_utc)


def product_coverage(index_utc: pd.DatetimeIndex, product: Product) -> tuple[str, int, int]:
    expected = product.expected_utc_index
    actual = index_utc[product_mask(index_utc, product)]
    if len(actual) == 0:
        return "ABSENT", 0, len(expected)
    if actual.equals(expected):
        return "FULL", len(actual), len(expected)
    return "PARTIAL", len(actual), len(expected)


def mean_for_load(hourly: pd.DataFrame, product: Product, load_type: str) -> tuple[float | None, int, int]:
    index = pd.DatetimeIndex(hourly["_timestamp_utc"])
    base_mask = product_mask(index, product)
    expected_index = product.expected_utc_index
    expected_mask = np.ones(len(expected_index), dtype=bool)
    if load_type == "PEAK":
        peak = eex_peak_mask(index, country="CH")
        expected_mask = eex_peak_mask(expected_index, country="CH")
        mask = base_mask & peak
    elif load_type == "OFFPEAK":
        peak = eex_peak_mask(index, country="CH")
        expected_mask = ~eex_peak_mask(expected_index, country="CH")
        mask = base_mask & ~peak
    else:
        mask = base_mask
    rows = int(mask.sum())
    expected_rows = int(expected_mask.sum()) if load_type in {"PEAK", "OFFPEAK"} else len(expected_index)
    if rows <= 0:
        return None, rows, expected_rows
    return float(hourly.loc[mask, "_price"].mean()), rows, expected_rows


def direct_quote_checks(
    hourly: pd.DataFrame,
    forwards: pd.DataFrame,
    *,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = pd.DatetimeIndex(hourly["_timestamp_utc"])
    for _, quote in forwards.sort_values(["load_type", "product"]).iterrows():
        load_type = str(quote["load_type"]).upper()
        if load_type not in {"BASE", "PEAK", "OFFPEAK"}:
            rows.append(gate("UNSUPPORTED", "unsupported_load_type", load_type=load_type, product=str(quote["product"]), message="load type is not supported by P1 audit"))
            continue
        product = parse_product(str(quote["product"]))
        if product is None:
            continue
        coverage, actual_rows, expected_rows = product_coverage(index, product)
        if coverage == "ABSENT":
            rows.append(
                gate(
                    "UNSUPPORTED",
                    "quoted_product_absent",
                    load_type=load_type,
                    product=product.key,
                    message="quoted product has no delivered hourly rows",
                    expected_rows=expected_rows,
                )
            )
            continue
        if coverage == "PARTIAL":
            rows.append(
                gate(
                    "UNSUPPORTED",
                    "partial_product_window",
                    load_type=load_type,
                    product=product.key,
                    message="quoted product is only partially covered by delivered CSV",
                    actual_rows=actual_rows,
                    expected_rows=expected_rows,
                )
            )
            continue
        achieved, rows_count, expected_load_rows = mean_for_load(hourly, product, load_type)
        if achieved is None or rows_count != expected_load_rows:
            rows.append(
                gate(
                    "UNSUPPORTED",
                    "partial_load_window",
                    load_type=load_type,
                    product=product.key,
                    message="load-type window is absent or incomplete",
                    actual_rows=rows_count,
                    expected_rows=expected_load_rows,
                )
            )
            continue
        target = float(quote["price"])
        residual = float(achieved - target)
        rows.append(
            gate(
                "PASS" if abs(residual) <= tolerance else "CRITICAL",
                "direct_quote_mean",
                load_type=load_type,
                product=product.key,
                message="direct quoted product mean checked on exact product window",
                target_eur_mwh=target,
                achieved_eur_mwh=achieved,
                residual_eur_mwh=residual,
                abs_residual_eur_mwh=abs(residual),
                tolerance_eur_mwh=float(tolerance),
                rows=rows_count,
            )
        )
    return rows


def implied_offpeak_quotes(forwards: pd.DataFrame) -> pd.DataFrame:
    prices = {
        (str(row["load_type"]).upper(), str(row["product"])): float(row["price"])
        for _, row in forwards.iterrows()
    }
    products = sorted({product for load, product in prices if load == "BASE"} & {product for load, product in prices if load == "PEAK"})
    rows: list[dict[str, object]] = []
    for key in products:
        product = parse_product(key)
        if product is None:
            continue
        expected = product.expected_utc_index
        peak_hours = int(eex_peak_mask(expected, country="CH").sum())
        base_hours = len(expected)
        offpeak_hours = base_hours - peak_hours
        if offpeak_hours <= 0:
            continue
        target = (prices[("BASE", key)] * base_hours - prices[("PEAK", key)] * peak_hours) / offpeak_hours
        rows.append({"load_type": "OFFPEAK", "product": key, "price": float(target), "date": pd.NaT, "market": "CH"})
    return pd.DataFrame(rows)


def product_load_hours(product: Product, load_type: str) -> int:
    expected = product.expected_utc_index
    if load_type == "PEAK":
        return int(eex_peak_mask(expected, country="CH").sum())
    if load_type == "OFFPEAK":
        return int((~eex_peak_mask(expected, country="CH")).sum())
    return int(len(expected))


def weighted_quote_mean(prices: dict[str, float], product_keys: list[str], load_type: str) -> float | None:
    weighted_sum = 0.0
    total_hours = 0
    for key in product_keys:
        product = parse_product(key)
        if product is None or key not in prices:
            return None
        hours = product_load_hours(product, load_type)
        weighted_sum += float(prices[key]) * hours
        total_hours += hours
    if total_hours <= 0:
        return None
    return weighted_sum / total_hours


def complete_child_products(product: Product, prices: dict[str, float]) -> list[str]:
    if product.granularity == "QUARTER":
        start_month = (int(product.quarter or 1) - 1) * 3 + 1
        children = [f"{product.year}-{month:02d}" for month in range(start_month, start_month + 3)]
        return children if all(child in prices for child in children) else []
    if product.granularity == "CALENDAR":
        quarter_children = [f"{product.year}-Q{quarter}" for quarter in range(1, 5)]
        month_children = [f"{product.year}-{month:02d}" for month in range(1, 13)]
        if all(child in prices for child in quarter_children):
            return quarter_children
        if all(child in prices for child in month_children):
            return month_children
    return []


def active_quote_set(forwards: pd.DataFrame, *, tolerance: float) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Apply the P1 direct-quote hierarchy: Month > Quarter > Calendar.

    A parent with a complete directly quoted child set is removed from direct
    mean checks. The removal is still evidence: consistent parents are logged
    as redundant, while conflicting parents are fail-closed source gates.
    """
    rows: list[dict[str, object]] = []
    drop_keys: set[tuple[str, str]] = set()
    for load_type in ["BASE", "PEAK", "OFFPEAK"]:
        prices = quote_price_dict(forwards, load_type)
        if not prices:
            continue
        products = {key: parse_product(key) for key in prices}
        for key, product in sorted(products.items()):
            if product is None:
                continue
            children = complete_child_products(product, prices)
            if not children:
                continue
            implied = weighted_quote_mean(prices, children, load_type)
            if implied is None:
                continue
            target = float(prices[key])
            residual = float(implied - target)
            status = "PASS" if abs(residual) <= tolerance else "CRITICAL"
            drop_keys.add((load_type, key))
            rows.append(
                gate(
                    status,
                    "active_quote_set_parent_dropped",
                    load_type=load_type,
                    product=key,
                    message=(
                        "parent quote dropped from direct checks because a complete finer quoted child set exists"
                        if status == "PASS"
                        else "conflicting parent quote dropped from direct checks because a complete finer quoted child set exists"
                    ),
                    dropped_reason="redundant_consistent" if status == "PASS" else "parent_child_conflict",
                    child_products=",".join(children),
                    target_eur_mwh=target,
                    achieved_eur_mwh=float(implied),
                    residual_eur_mwh=residual,
                    abs_residual_eur_mwh=abs(residual),
                    tolerance_eur_mwh=float(tolerance),
                )
            )
    if not drop_keys:
        return forwards.copy(), rows
    mask = [
        (str(row["load_type"]).upper(), str(row["product"])) not in drop_keys
        for _, row in forwards.iterrows()
    ]
    return forwards.loc[mask].reset_index(drop=True), rows


def missing_peak_checks(hourly: pd.DataFrame, forwards: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    peak_products = set(forwards.loc[forwards["load_type"] == "PEAK", "product"].astype(str))
    index = pd.DatetimeIndex(hourly["_timestamp_utc"])
    for product_key in sorted(forwards.loc[forwards["load_type"] == "BASE", "product"].astype(str).unique()):
        product = parse_product(product_key)
        if product is None or product_key in peak_products:
            continue
        coverage, actual_rows, expected_rows = product_coverage(index, product)
        if actual_rows > 0:
            rows.append(
                gate(
                    "UNSUPPORTED",
                    "missing_peak_quote",
                    load_type="PEAK",
                    product=product_key,
                    message="BASE product is covered by the delivered curve but no same-product PEAK quote is available",
                    coverage=coverage,
                    actual_rows=actual_rows,
                    expected_rows=expected_rows,
                )
            )
    return rows


def bucket_checks_for_load(
    hourly: pd.DataFrame,
    prices: dict[str, float],
    *,
    load_type: str,
    tolerance: float,
) -> list[dict[str, object]]:
    if not prices:
        return []
    ts = pd.Series(hourly["_timestamp_ch"], index=hourly.index)
    if load_type == "PEAK":
        mask = eex_peak_mask(pd.DatetimeIndex(hourly["_timestamp_utc"]), country="CH")
    elif load_type == "OFFPEAK":
        mask = ~eex_peak_mask(pd.DatetimeIndex(hourly["_timestamp_utc"]), country="CH")
    else:
        mask = np.ones(len(hourly), dtype=bool)
    sub = hourly.loc[mask].copy()
    if sub.empty:
        return [gate("UNSUPPORTED", "bucket_evidence_empty", load_type=load_type, message="no delivered rows for load-type bucket audit")]
    buckets, targets = calibration_buckets(pd.Series(sub["_timestamp_ch"], index=sub.index), prices)
    out: list[dict[str, object]] = []
    for bucket_name, target in sorted(targets.items()):
        idx = buckets[buckets == bucket_name].index
        if len(idx) == 0:
            out.append(gate("UNSUPPORTED", "bucket_empty", load_type=load_type, bucket=str(bucket_name), message="quote-aware bucket has no delivered rows"))
            continue
        product = parse_product(str(bucket_name))
        if product is not None:
            _, rows_count, expected_rows = mean_for_load(hourly, product, load_type)
            if rows_count != expected_rows:
                out.append(
                    gate(
                        "UNSUPPORTED",
                        "partial_bucket_window",
                        load_type=load_type,
                        bucket=str(bucket_name),
                        message="quote-aware direct bucket is only partially covered by delivered CSV",
                        actual_rows=rows_count,
                        expected_rows=expected_rows,
                    )
                )
                continue
        achieved = float(sub.loc[idx, "_price"].mean())
        residual = achieved - float(target)
        out.append(
            gate(
                "PASS" if abs(residual) <= tolerance else "CRITICAL",
                "quote_aware_bucket_mean",
                load_type=load_type,
                bucket=str(bucket_name),
                message="quote-aware non-overlapping bucket mean checked",
                bucket_type="RESIDUAL" if str(bucket_name).endswith("-RESIDUAL") else "DIRECT",
                target_eur_mwh=float(target),
                achieved_eur_mwh=achieved,
                residual_eur_mwh=float(residual),
                abs_residual_eur_mwh=abs(float(residual)),
                tolerance_eur_mwh=float(tolerance),
                rows=int(len(idx)),
            )
        )
    return out


def quote_price_dict(forwards: pd.DataFrame, load_type: str) -> dict[str, float]:
    sub = forwards[forwards["load_type"].astype(str).str.upper() == load_type]
    return {str(row["product"]): float(row["price"]) for _, row in sub.iterrows()}


def scope_forwards_to_csv_window(hourly: pd.DataFrame, forwards: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    index = pd.DatetimeIndex(hourly["_timestamp_utc"])
    if len(index) == 0 or forwards.empty:
        return forwards.copy(), []
    csv_start = index.min()
    csv_end_exclusive = index.max() + pd.Timedelta(hours=1)
    keep: list[bool] = []
    rows: list[dict[str, object]] = []
    for _, quote in forwards.iterrows():
        product_key = str(quote["product"])
        product = parse_product(product_key)
        if product is None:
            keep.append(True)
            continue
        out_of_scope = product.end_utc <= csv_start or product.start_utc >= csv_end_exclusive
        keep.append(not out_of_scope)
        if out_of_scope:
            rows.append(
                gate(
                    "INFO",
                    "out_of_scope_quote",
                    load_type=str(quote["load_type"]).upper(),
                    product=product_key,
                    message="quoted product has no overlap with delivered CSV window and was excluded by explicit scope option",
                    csv_start_utc=str(csv_start),
                    csv_end_exclusive_utc=str(csv_end_exclusive),
                    product_start_utc=str(product.start_utc),
                    product_end_utc=str(product.end_utc),
                )
            )
    return forwards.loc[keep].reset_index(drop=True), rows


def audit(
    csv_path: Path,
    forwards_path: Path,
    *,
    tolerance_eur_mwh: float,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    price_column: str = DEFAULT_PRICE_COLUMN,
    market: str = "CH",
    as_of: str | None = None,
    manifest_path: Path | None = None,
    scope_to_csv_window: bool = False,
) -> dict[str, object]:
    if tolerance_eur_mwh < 0.0 or not np.isfinite(float(tolerance_eur_mwh)):
        raise ValueError("tolerance_eur_mwh must be a finite non-negative value")
    gates: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "csv_path": str(csv_path),
        "forwards_path": str(forwards_path),
        "market": market,
        "timestamp_column": timestamp_column,
        "price_column": price_column,
        "tolerance_eur_mwh": float(tolerance_eur_mwh),
        "csv_sha256": sha256_file(csv_path) if csv_path.exists() else "",
        "forwards_sha256": sha256_file(forwards_path) if forwards_path.exists() else "",
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
        "manifest_sha256": sha256_file(manifest_path) if manifest_path is not None and manifest_path.exists() else "",
        "scope_to_csv_window": bool(scope_to_csv_window),
    }
    if not csv_path.exists():
        gates.append(gate("CRITICAL", "csv_missing", message="explicit delivered hourly CSV path does not exist", path=str(csv_path)))
    if not forwards_path.exists():
        gates.append(gate("CRITICAL", "forwards_missing", message="explicit forwards source path does not exist", path=str(forwards_path)))
    if manifest_path is None:
        gates.append(gate("CRITICAL", "manifest_missing", message="selected manifest/artifact path was not provided"))
    elif not manifest_path.exists():
        gates.append(gate("CRITICAL", "manifest_missing", message="explicit manifest path does not exist", path=str(manifest_path)))
    if any(row.get("gate") in {"csv_missing", "forwards_missing"} for row in gates):
        return {"metadata": metadata, "gates": pd.DataFrame(gates)}
    try:
        hourly, csv_gates = load_hourly_curve(csv_path, timestamp_column=timestamp_column, price_column=price_column)
    except Exception as exc:
        gates.append(gate("CRITICAL", "csv_read_error", message="failed to read delivered hourly CSV", error=str(exc)))
        return {"metadata": metadata, "gates": pd.DataFrame(gates)}
    gates.extend(csv_gates)
    try:
        forwards, snapshot, forward_gates = select_forwards_snapshot(
            forwards_path,
            market=market,
            as_of=as_of,
            tolerance=tolerance_eur_mwh,
        )
    except Exception as exc:
        gates.append(gate("CRITICAL", "forwards_read_error", message="failed to read forwards source", error=str(exc)))
        return {"metadata": metadata, "gates": pd.DataFrame(gates)}
    gates.extend(forward_gates)
    metadata["forward_snapshot_date"] = str(snapshot.date()) if snapshot is not None and not pd.isna(snapshot) else ""
    metadata["forward_quote_rows"] = int(len(forwards))
    blocking_csv_gates = {
        "csv_schema",
        "csv_empty",
        "price_column",
        "timestamp_parse",
        "timestamp_utc_timezone",
        "timestamp_duplicate",
        "timestamp_monotone",
        "timestamp_hourly_frequency",
    }
    has_blocking_csv_gate = any(
        row.get("status") == "CRITICAL" and row.get("gate") in blocking_csv_gates
        for row in gates
    )
    if hourly is None or forwards.empty or has_blocking_csv_gate:
        return {"metadata": metadata, "gates": pd.DataFrame(gates)}
    if scope_to_csv_window:
        forwards, scope_gates = scope_forwards_to_csv_window(hourly, forwards)
        gates.extend(scope_gates)
        metadata["out_of_scope_forward_quote_rows"] = int(len(scope_gates))
        metadata["scoped_forward_quote_rows"] = int(len(forwards))
        if forwards.empty:
            gates.append(gate("CRITICAL", "forward_evidence_empty", message="no forwards remain after explicit CSV-window scoping"))
            return {"metadata": metadata, "gates": pd.DataFrame(gates)}
    active_forwards, active_quote_gates = active_quote_set(forwards, tolerance=tolerance_eur_mwh)
    gates.extend(active_quote_gates)
    metadata["active_forward_quote_rows"] = int(len(active_forwards))
    metadata["dropped_forward_quote_rows"] = int(len(forwards) - len(active_forwards))
    gates.extend(missing_peak_checks(hourly, active_forwards))
    gates.extend(direct_quote_checks(hourly, active_forwards, tolerance=tolerance_eur_mwh))
    offpeak = implied_offpeak_quotes(active_forwards)
    if not offpeak.empty:
        gates.extend(direct_quote_checks(hourly, offpeak, tolerance=tolerance_eur_mwh))
    gates.extend(bucket_checks_for_load(hourly, quote_price_dict(active_forwards, "BASE"), load_type="BASE", tolerance=tolerance_eur_mwh))
    gates.extend(bucket_checks_for_load(hourly, quote_price_dict(active_forwards, "PEAK"), load_type="PEAK", tolerance=tolerance_eur_mwh))
    if not offpeak.empty:
        gates.extend(bucket_checks_for_load(hourly, quote_price_dict(offpeak, "OFFPEAK"), load_type="OFFPEAK", tolerance=tolerance_eur_mwh))
    if not any(row["status"] in {"CRITICAL", "UNSUPPORTED"} for row in gates):
        gates.append(gate("PASS", "audit_complete", message="all P1 product-normalization gates passed"))
    return {"metadata": metadata, "gates": pd.DataFrame(gates)}


def status_counts(gates: pd.DataFrame) -> dict[str, int]:
    if gates.empty or "status" not in gates:
        return {"PASS": 0, "CRITICAL": 0, "UNSUPPORTED": 0, "INFO": 0}
    counts = gates["status"].value_counts().to_dict()
    return {key: int(counts.get(key, 0)) for key in ["PASS", "CRITICAL", "UNSUPPORTED", "INFO"]}


def frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def write_report(path: Path, *, metadata: dict[str, object], gates: pd.DataFrame) -> None:
    counts = status_counts(gates)
    lines = [
        "# CH LT Product Normalization Audit",
        "",
        "## Evidence",
        "",
        f"- CSV: `{metadata.get('csv_path', '')}`",
        f"- CSV sha256: `{metadata.get('csv_sha256', '')}`",
        f"- Forwards: `{metadata.get('forwards_path', '')}`",
        f"- Forwards sha256: `{metadata.get('forwards_sha256', '')}`",
        f"- Forward snapshot date: `{metadata.get('forward_snapshot_date', '')}`",
        f"- Manifest: `{metadata.get('manifest_path', '')}`",
        f"- Manifest sha256: `{metadata.get('manifest_sha256', '')}`",
        f"- Tolerance EUR/MWh: `{metadata.get('tolerance_eur_mwh', '')}`",
        f"- Scope forwards to CSV window: `{metadata.get('scope_to_csv_window', False)}`",
        "",
        "## Gate Counts",
        "",
        f"- PASS: {counts['PASS']}",
        f"- CRITICAL: {counts['CRITICAL']}",
        f"- UNSUPPORTED: {counts['UNSUPPORTED']}",
        f"- INFO: {counts['INFO']}",
        "",
        "## Gates",
        "",
    ]
    if gates.empty:
        lines.append("_No gates produced._")
    else:
        columns = ["status", "gate", "load_type", "product", "bucket", "message", "target_eur_mwh", "achieved_eur_mwh", "residual_eur_mwh", "rows"]
        available = [column for column in columns if column in gates.columns]
        lines.append(markdown_table(gates[available]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.fillna("").astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = ["| " + " | ".join(value.replace("\n", " ") for value in row) + " |" for row in text.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_json(path: Path, *, metadata: dict[str, object], gates: pd.DataFrame) -> None:
    payload = {
        "metadata": metadata,
        "status_counts": status_counts(gates),
        "gates": frame_records(gates),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Delivered final hourly CH LT CSV.")
    parser.add_argument("--forwards", required=True, help="Traceable EEX forwards source (.csv or .parquet).")
    parser.add_argument("--tolerance-eur-mwh", required=True, type=float, help="Explicit absolute tolerance in EUR/MWh.")
    parser.add_argument("--timestamp-column", default=DEFAULT_TIMESTAMP_COLUMN)
    parser.add_argument("--price-column", default=DEFAULT_PRICE_COLUMN)
    parser.add_argument("--market", default="CH")
    parser.add_argument("--as-of", default=None, help="Optional exact forwards snapshot date. Defaults to latest market snapshot.")
    parser.add_argument("--manifest", default=None, help="Selected manifest/artifact path. Omission is reported as CRITICAL.")
    parser.add_argument(
        "--scope-forwards-to-csv-window",
        action="store_true",
        help="Explicitly exclude quoted products with no overlap with the delivered CSV window. Partial windows still fail as UNSUPPORTED.",
    )
    parser.add_argument("--report", required=True, help="Markdown report output path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON report output path.")
    parser.add_argument("--gates-output", default=None, help="Optional gates CSV output path.")
    parser.add_argument("--allow-failed-gates", action="store_true", help="Diagnostic mode: write outputs but return 0 despite CRITICAL/UNSUPPORTED gates.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = audit(
        Path(args.csv),
        Path(args.forwards),
        tolerance_eur_mwh=args.tolerance_eur_mwh,
        timestamp_column=args.timestamp_column,
        price_column=args.price_column,
        market=args.market,
        as_of=args.as_of,
        manifest_path=Path(args.manifest) if args.manifest else None,
        scope_to_csv_window=bool(args.scope_forwards_to_csv_window),
    )
    metadata = result["metadata"]  # type: ignore[assignment]
    gates = result["gates"]  # type: ignore[assignment]
    if not isinstance(gates, pd.DataFrame):
        raise TypeError("audit gates must be a DataFrame")
    write_report(Path(args.report), metadata=metadata, gates=gates)  # type: ignore[arg-type]
    if args.json_output:
        write_json(Path(args.json_output), metadata=metadata, gates=gates)  # type: ignore[arg-type]
    if args.gates_output:
        Path(args.gates_output).parent.mkdir(parents=True, exist_ok=True)
        gates.to_csv(args.gates_output, index=False)
    counts = status_counts(gates)
    print(
        "[product-normalization-audit] "
        f"pass={counts['PASS']} critical={counts['CRITICAL']} unsupported={counts['UNSUPPORTED']}"
    )
    print(f"[product-normalization-audit] report -> {args.report}")
    if (counts["CRITICAL"] or counts["UNSUPPORTED"]) and not args.allow_failed_gates:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
