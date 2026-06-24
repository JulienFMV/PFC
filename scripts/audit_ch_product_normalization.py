"""Read-only audit of delivered CH product normalization.

This script audits the exact exported hourly CSV against EEX product averages.
It does not regenerate or mutate a curve. The hard checks are intentionally
simple: quoted BASE products must reprice on all delivery hours, quoted PEAK
products must reprice on EEX peak hours, and implied OFFPEAK must balance when
BASE and PEAK are both quoted for the same product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.calibration.cascading import count_hours, parse_key  # noqa: E402
from pfc_shaping.calibration.eex_contract_selection import calibration_buckets  # noqa: E402

LOCAL_TZ = "Europe/Zurich"
DEFAULT_PRICE_COLUMN = "price_weighted_mean_eur_mwh"
SCHEMA_VERSION = "ch_product_normalization_audit.v2"
SOURCE_HIERARCHY_POLICY_SCHEMA_VERSION = "ch_quote_conflict_source_hierarchy_policy.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_hierarchy_policy(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raise ValueError(f"unsupported source hierarchy policy type: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"source hierarchy policy must be a mapping: {path}")
    return payload


def evaluate_source_hierarchy_policy(
    policy: Mapping[str, Any] | None,
    *,
    policy_path: Path | None,
    policy_hash: str | None,
    market: str,
    forward_date: pd.Timestamp,
    quote_conflict_count: int,
) -> dict[str, Any]:
    if policy is None:
        return {
            "path": None,
            "sha256": None,
            "status": "NOT_PROVIDED",
            "production_approved": False,
            "accepted_quote_conflict_count": 0,
            "blocking_quote_conflict_count": quote_conflict_count,
            "reason": "no source hierarchy policy artifact provided",
        }

    errors: list[str] = []
    if str(policy.get("schema_version", "")) != SOURCE_HIERARCHY_POLICY_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if str(policy.get("market", "")).upper() != str(market).upper():
        errors.append("market_mismatch")
    policy_forward_date = str(policy.get("forward_snapshot_date", ""))
    if policy_forward_date and policy_forward_date != forward_date.date().isoformat():
        errors.append("forward_snapshot_date_mismatch")
    if str(policy.get("source_hierarchy", "")) != "quote_aware_finer_buckets_over_redundant_parent":
        errors.append("source_hierarchy_mismatch")
    if not bool(policy.get("accept_quote_conflict", False)):
        errors.append("accept_quote_conflict_false")

    production_approved = bool(policy.get("production_approved", False))
    accepted = quote_conflict_count if production_approved and not errors else 0
    status = (
        "ACCEPTED_PRODUCTION_APPROVED"
        if accepted
        else "VALID_NOT_PRODUCTION_APPROVED"
        if not errors
        else "INVALID"
    )
    return {
        "path": str(policy_path) if policy_path is not None else None,
        "sha256": policy_hash,
        "status": status,
        "production_approved": production_approved,
        "accepted_quote_conflict_count": accepted,
        "blocking_quote_conflict_count": quote_conflict_count - accepted,
        "reason": ";".join(errors) if errors else str(policy.get("decision", "")),
    }


def _parse_timestamp_ch(values: pd.Series, offsets: pd.Series | None = None) -> pd.Series:
    text = values.astype(str)
    if offsets is not None:
        offset = offsets.astype(str).str.strip()
        offset = offset.str.replace("UTC", "", regex=False)
        numeric = offset.str.fullmatch(r"\d{1,4}", na=False)
        offset = offset.where(~numeric, "+" + offset.str.zfill(4))
        offset = offset.str.replace(":", "", regex=False)
        combined = text.str.strip() + " " + offset
        return pd.to_datetime(combined, format="%d.%m.%Y %H:%M %z", utc=True).dt.tz_convert(LOCAL_TZ)
    has_offset = text.str.contains(r"[+-]\d{4}$", regex=True, na=False)
    if bool(has_offset.all()):
        return pd.to_datetime(text, utc=True).dt.tz_convert(LOCAL_TZ)
    parsed = pd.to_datetime(text, format="%d.%m.%Y %H:%M", errors="raise")
    return parsed.dt.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")


def _holiday_set(years: list[int], *, country: str) -> set[object]:
    out: set[object] = set()
    country_u = country.upper()
    for year in years:
        if country_u == "CH":
            out |= set(holidays.Switzerland(years=year).keys())
        elif country_u == "DE":
            out |= set(holidays.Germany(years=year).keys())
        elif country_u == "AT":
            out |= set(holidays.Austria(years=year).keys())
        elif country_u == "FR":
            out |= set(holidays.France(years=year).keys())
        elif country_u == "IT":
            out |= set(holidays.Italy(years=year).keys())
        else:
            raise ValueError(f"unsupported EEX peak holiday country: {country!r}")
    return out


def eex_peak_mask(ts_ch: pd.Series, *, country: str = "CH") -> pd.Series:
    ts = pd.Series(ts_ch, copy=False)
    if ts.empty:
        return pd.Series(dtype=bool, index=ts.index)
    years = sorted(int(year) for year in ts.dt.year.dropna().unique())
    holiday_set = _holiday_set(years, country=country)
    return (
        (ts.dt.weekday < 5)
        & (ts.dt.hour >= 8)
        & (ts.dt.hour < 20)
        & (~ts.dt.date.isin(holiday_set))
    )


def load_delivered_hourly_csv(path: Path, *, price_column: str = DEFAULT_PRICE_COLUMN) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted({"timestamp_ch", price_column} - set(frame.columns))
    if missing:
        raise ValueError(f"delivered CSV missing required columns: {missing}")
    frame = frame.copy()
    frame["ts_ch"] = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    frame["ts_utc"] = frame["ts_ch"].dt.tz_convert("UTC")
    if "timestamp_utc" in frame.columns:
        timestamp_utc = pd.to_datetime(frame["timestamp_utc"], format="%d.%m.%Y %H:%M", errors="raise")
        timestamp_utc = timestamp_utc.dt.tz_localize("UTC")
        if not bool((timestamp_utc == frame["ts_utc"]).all()):
            mismatches = int((timestamp_utc != frame["ts_utc"]).sum())
            raise ValueError(f"timestamp_ch/utc_offset_ch does not round-trip to timestamp_utc ({mismatches} rows)")
    if frame["ts_utc"].duplicated().any():
        raise ValueError("delivered CSV contains duplicate UTC timestamps")
    if not frame["ts_utc"].is_monotonic_increasing:
        raise ValueError("delivered CSV timestamps must be sorted by UTC time")
    if len(frame) > 1:
        deltas = frame["ts_utc"].diff().dropna()
        bad = deltas != pd.Timedelta(hours=1)
        if bool(bad.any()):
            raise ValueError(f"delivered CSV is not contiguous hourly data ({int(bad.sum())} bad gaps)")
    frame["year"] = frame["ts_ch"].dt.year.astype(int)
    frame["month"] = frame["ts_ch"].dt.month.astype(int)
    frame["quarter"] = frame["ts_ch"].dt.quarter.astype(int)
    frame[price_column] = pd.to_numeric(frame[price_column], errors="raise").astype(float)
    if not bool(np.isfinite(frame[price_column].to_numpy(dtype=float)).all()):
        raise ValueError(f"delivered CSV price column contains non-finite values: {price_column}")
    return frame.sort_values("ts_ch").reset_index(drop=True)


def load_forward_snapshot(
    path: Path,
    *,
    market: str = "CH",
    required_forward_date: str | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    frame = pd.read_parquet(path)
    sub = frame[frame["market"].astype(str).str.upper() == market.upper()].copy()
    if sub.empty:
        raise ValueError(f"no forwards for market={market!r} in {path}")
    sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None)
    sub["product"] = sub["product"].astype(str)
    sub["load_type"] = sub["load_type"].astype(str).str.upper()
    sub["price"] = pd.to_numeric(sub["price"], errors="raise").astype(float)
    if required_forward_date is None:
        forward_date = pd.Timestamp(sub["date"].max()).normalize()
    else:
        forward_date = pd.Timestamp(required_forward_date).tz_localize(None).normalize()
    snap = sub[sub["date"].dt.normalize() == forward_date].copy()
    if snap.empty:
        raise ValueError(f"no forwards for market={market!r} on {forward_date.date()}")
    duplicate_mask = snap.duplicated(["product", "load_type"], keep=False)
    if bool(duplicate_mask.any()):
        dupes = snap.loc[duplicate_mask, ["product", "load_type", "price"]].sort_values(["load_type", "product"])
        raise ValueError(f"duplicate forward quote rows in snapshot {forward_date.date()}: {dupes.to_dict('records')}")
    return forward_date, snap.sort_values(["load_type", "product"]).reset_index(drop=True)


def product_month_range(product: str) -> tuple[str, int, int, int]:
    product_type, year, sub_index = parse_key(product)
    if product_type == "Cal":
        return product_type, year, 1, 12
    if product_type == "Quarter":
        if sub_index is None:
            raise ValueError(f"quarter product missing quarter index: {product!r}")
        start = (sub_index - 1) * 3 + 1
        return product_type, year, start, start + 2
    if product_type == "Month":
        if sub_index is None:
            raise ValueError(f"month product missing month index: {product!r}")
        return product_type, year, sub_index, sub_index
    raise ValueError(f"unsupported product type for delivered audit: {product!r}")


def product_mask(hourly: pd.DataFrame, product: str) -> pd.Series:
    product_type, year, start_month, end_month = product_month_range(product)
    mask = (hourly["year"] == year) & (hourly["month"] >= start_month) & (hourly["month"] <= end_month)
    if product_type == "Quarter":
        quarter = ((start_month - 1) // 3) + 1
        mask = mask & (hourly["quarter"] == quarter)
    return mask


def _status_from_residual(abs_residual: float, tolerance: float) -> str:
    if not np.isfinite(abs_residual):
        return "UNSUPPORTED"
    return "PASS" if abs_residual <= tolerance else "CRITICAL"


def _product_type_for_label(product: str) -> str:
    if str(product).endswith("-RESIDUAL"):
        return "Residual"
    try:
        product_type, _, _, _ = product_month_range(product)
    except ValueError:
        return "Bucket"
    return product_type


def _severity(status: str) -> str:
    if status == "CRITICAL":
        return "critical"
    if status == "QUOTE_CONFLICT":
        return "quote_conflict"
    if status == "UNSUPPORTED":
        return "unsupported"
    return "info"


def _gate_row(
    *,
    gate_id: str,
    status: str,
    market: str,
    load_type: str,
    product: str,
    product_type: str,
    forward_date: pd.Timestamp,
    target: float | None,
    curve_mean: float | None,
    residual: float | None,
    tolerance: float,
    rows: int,
    expected_rows: int,
    total_hours: int,
    peak_hours: int,
    offpeak_hours: int,
    evidence: str,
    price_column: str,
) -> dict[str, Any]:
    abs_residual = abs(residual) if residual is not None and np.isfinite(residual) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": gate_id,
        "status": status,
        "severity": _severity(status),
        "market": market.upper(),
        "load_type": load_type.upper(),
        "product": product,
        "product_type": product_type,
        "forward_date": forward_date.date().isoformat(),
        "target_eur_mwh": target,
        "curve_mean_eur_mwh": curve_mean,
        "residual_eur_mwh": residual,
        "abs_residual_eur_mwh": abs_residual,
        "tolerance_eur_mwh": tolerance,
        "rows": rows,
        "expected_rows": expected_rows,
        "total_hours": total_hours,
        "peak_hours": peak_hours,
        "offpeak_hours": offpeak_hours,
        "evidence": evidence,
        "remediation_hint": _remediation_hint(gate_id, status),
        "price_column": price_column,
    }


def _remediation_hint(gate_id: str, status: str) -> str:
    if status == "PASS":
        return ""
    if status == "QUOTE_CONFLICT":
        return (
            "Inspect the EEX snapshot hierarchy: finer quote-aware buckets pass, "
            "but the redundant parent quote is inconsistent with them."
        )
    if status == "UNSUPPORTED":
        return "Verify the delivered CSV fully covers this product window and the EEX quote exists."
    if gate_id == "hard_peak_product_repricing":
        return "Inspect PEAK calibration and any post-calibration hourly reshaping on the delivered artifact."
    if gate_id == "hard_base_product_repricing":
        return "Inspect monthly solver authority, BASE calibration, and any downstream level rewrite."
    if gate_id == "implied_offpeak_identity":
        return "Inspect BASE/PEAK energy balance and offpeak compensation after PEAK calibration."
    return "Inspect product normalization logic."


def build_product_normalization_gates(
    hourly: pd.DataFrame,
    forwards: pd.DataFrame,
    *,
    forward_date: pd.Timestamp,
    market: str = "CH",
    price_column: str = DEFAULT_PRICE_COLUMN,
    hard_tolerance: float = 1e-6,
    peak_country: str = "CH",
    required_load_types: tuple[str, ...] = ("BASE", "PEAK"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quotes: dict[tuple[str, str], float] = {}
    product_meta: dict[str, tuple[str, int, int, int, int, int]] = {}
    fully_covered_prices: dict[str, dict[str, float]] = {"BASE": {}, "PEAK": {}, "OFFPEAK": {}}
    required_load_type_set = {str(load_type).upper() for load_type in required_load_types}

    for _, quote in forwards.iterrows():
        product = str(quote["product"])
        load_type = str(quote["load_type"]).upper()
        if load_type not in {"BASE", "PEAK", "OFFPEAK"}:
            continue
        try:
            product_type, year, start_month, end_month = product_month_range(product)
        except ValueError:
            continue
        total_hours, peak_hours, offpeak_hours = count_hours(
            year,
            start_month,
            end_month,
            tz=LOCAL_TZ,
            country=peak_country,
        )
        product_meta[product] = (product_type, year, total_hours, peak_hours, offpeak_hours, start_month)
        quotes[(product, load_type)] = float(quote["price"])

        mask = product_mask(hourly, product)
        expected_rows = total_hours
        gate_id = "hard_base_product_repricing"
        if load_type == "PEAK":
            mask = mask & eex_peak_mask(hourly["ts_ch"], country=peak_country)
            expected_rows = peak_hours
            gate_id = "hard_peak_product_repricing"
        elif load_type == "OFFPEAK":
            mask = mask & ~eex_peak_mask(hourly["ts_ch"], country=peak_country)
            expected_rows = offpeak_hours
            gate_id = "hard_offpeak_product_repricing"

        row_count = int(mask.sum())
        if row_count != expected_rows or expected_rows <= 0:
            rows.append(
                _gate_row(
                    gate_id=gate_id,
                    status="UNSUPPORTED",
                    market=market,
                    load_type=load_type,
                    product=product,
                    product_type=product_type,
                    forward_date=forward_date,
                    target=float(quote["price"]),
                    curve_mean=None,
                    residual=None,
                    tolerance=hard_tolerance,
                    rows=row_count,
                    expected_rows=expected_rows,
                    total_hours=total_hours,
                    peak_hours=peak_hours,
                    offpeak_hours=offpeak_hours,
                    evidence="partial_or_missing_delivered_rows",
                    price_column=price_column,
                )
            )
            continue

        fully_covered_prices[load_type][product] = float(quote["price"])
        curve_mean = float(hourly.loc[mask, price_column].mean())
        target = float(quote["price"])
        residual = curve_mean - target
        status = _status_from_residual(abs(residual), hard_tolerance)
        rows.append(
            _gate_row(
                gate_id=gate_id,
                status=status,
                market=market,
                load_type=load_type,
                product=product,
                product_type=product_type,
                forward_date=forward_date,
                target=target,
                curve_mean=curve_mean,
                residual=residual,
                tolerance=hard_tolerance,
                rows=row_count,
                expected_rows=expected_rows,
                total_hours=total_hours,
                peak_hours=peak_hours,
                offpeak_hours=offpeak_hours,
                evidence="delivered_csv_product_mean",
                price_column=price_column,
            )
        )

    for product in sorted({product for product, _load_type in quotes}):
        if product not in product_meta:
            continue
        product_type, _year, total_hours, peak_hours, offpeak_hours, _start_month = product_meta[product]
        if int(product_mask(hourly, product).sum()) != total_hours:
            continue
        for required_load_type in sorted(required_load_type_set):
            if (product, required_load_type) in quotes:
                continue
            if required_load_type == "PEAK":
                expected_rows = peak_hours
            elif required_load_type == "OFFPEAK":
                expected_rows = offpeak_hours
            else:
                expected_rows = total_hours
            rows.append(
                _gate_row(
                    gate_id="required_forward_quote_present",
                    status="UNSUPPORTED",
                    market=market,
                    load_type=required_load_type,
                    product=product,
                    product_type=product_type,
                    forward_date=forward_date,
                    target=None,
                    curve_mean=None,
                    residual=None,
                    tolerance=hard_tolerance,
                    rows=0,
                    expected_rows=expected_rows,
                    total_hours=total_hours,
                    peak_hours=peak_hours,
                    offpeak_hours=offpeak_hours,
                    evidence="missing_required_forward_quote",
                    price_column=price_column,
                )
            )

    for load_type, forward_prices in fully_covered_prices.items():
        if load_type not in {"BASE", "PEAK"} or not forward_prices:
            continue
        if load_type == "PEAK":
            frame = hourly.loc[eex_peak_mask(hourly["ts_ch"], country=peak_country)].copy()
            gate_id = "quote_aware_peak_bucket_repricing"
        else:
            frame = hourly
            gate_id = "quote_aware_base_bucket_repricing"
        buckets, targets = calibration_buckets(frame["ts_ch"], forward_prices)
        for bucket, idx in buckets.dropna().groupby(buckets.dropna()).groups.items():
            bucket_s = str(bucket)
            if bucket_s not in targets:
                continue
            curve_mean = float(frame.loc[idx, price_column].mean())
            target = float(targets[bucket_s])
            residual = curve_mean - target
            status = _status_from_residual(abs(residual), hard_tolerance)
            product_type = _product_type_for_label(bucket_s)
            rows.append(
                _gate_row(
                    gate_id=gate_id,
                    status=status,
                    market=market,
                    load_type=load_type,
                    product=bucket_s,
                    product_type=product_type,
                    forward_date=forward_date,
                    target=target,
                    curve_mean=curve_mean,
                    residual=residual,
                    tolerance=hard_tolerance,
                    rows=len(idx),
                    expected_rows=len(idx),
                    total_hours=len(idx),
                    peak_hours=len(idx) if load_type == "PEAK" else 0,
                    offpeak_hours=0,
                    evidence="delivered_csv_quote_aware_non_overlapping_bucket",
                    price_column=price_column,
                )
            )

    for product in sorted({product for product, load_type in quotes if load_type in {"BASE", "PEAK"}}):
        if (product, "BASE") not in quotes or (product, "PEAK") not in quotes:
            continue
        product_type, year, total_hours, peak_hours, offpeak_hours, _start_month = product_meta[product]
        if offpeak_hours <= 0:
            continue
        base_target = quotes[(product, "BASE")]
        peak_target = quotes[(product, "PEAK")]
        offpeak_target = (base_target * total_hours - peak_target * peak_hours) / offpeak_hours
        mask = product_mask(hourly, product) & ~eex_peak_mask(hourly["ts_ch"], country=peak_country)
        row_count = int(mask.sum())
        if row_count != offpeak_hours:
            status = "UNSUPPORTED"
            curve_mean = None
            residual = None
            evidence = "partial_or_missing_delivered_rows"
        else:
            curve_mean = float(hourly.loc[mask, price_column].mean())
            residual = curve_mean - offpeak_target
            status = _status_from_residual(abs(residual), hard_tolerance)
            evidence = "delivered_csv_implied_offpeak_energy_balance"
        rows.append(
            _gate_row(
                gate_id="implied_offpeak_identity",
                status=status,
                market=market,
                load_type="OFFPEAK",
                product=product,
                product_type=product_type,
                forward_date=forward_date,
                target=offpeak_target,
                curve_mean=curve_mean,
                residual=residual,
                tolerance=hard_tolerance,
                rows=row_count,
                expected_rows=offpeak_hours,
                total_hours=total_hours,
                peak_hours=peak_hours,
                offpeak_hours=offpeak_hours,
                evidence=evidence,
                price_column=price_column,
            )
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = _reclassify_redundant_quote_conflicts(
            out,
            hourly=hourly,
            fully_covered_prices=fully_covered_prices,
            price_column=price_column,
            hard_tolerance=hard_tolerance,
            peak_country=peak_country,
        )

    if not rows:
        rows.append(
            _gate_row(
                gate_id="audit_evidence_present",
                status="CRITICAL",
                market=market,
                load_type="ALL",
                product="*",
                product_type="Audit",
                forward_date=forward_date,
                target=None,
                curve_mean=None,
                residual=None,
                tolerance=hard_tolerance,
                rows=0,
                expected_rows=1,
                total_hours=0,
                peak_hours=0,
                offpeak_hours=0,
                evidence="no_supported_or_unsupported_product_gates_emitted",
                price_column=price_column,
            )
        )
        out = pd.DataFrame(rows)
    return out.sort_values(["gate_id", "load_type", "product"]).reset_index(drop=True)


def _reclassify_redundant_quote_conflicts(
    gates: pd.DataFrame,
    *,
    hourly: pd.DataFrame,
    fully_covered_prices: dict[str, dict[str, float]],
    price_column: str,
    hard_tolerance: float,
    peak_country: str,
) -> pd.DataFrame:
    out = gates.copy()
    direct_conflict_products: set[tuple[str, str]] = set()
    for load_type in ("BASE", "PEAK"):
        forward_prices = fully_covered_prices.get(load_type, {})
        if not forward_prices:
            continue
        if load_type == "PEAK":
            frame = hourly.loc[eex_peak_mask(hourly["ts_ch"], country=peak_country)].copy()
            gate_id = "hard_peak_product_repricing"
            quote_aware_gate_id = "quote_aware_peak_bucket_repricing"
        else:
            frame = hourly
            gate_id = "hard_base_product_repricing"
            quote_aware_gate_id = "quote_aware_base_bucket_repricing"
        buckets, targets = calibration_buckets(frame["ts_ch"], forward_prices)
        quote_aware_pass = set(
            out.loc[
                (out["gate_id"] == quote_aware_gate_id) & (out["status"] == "PASS"),
                "product",
            ].astype(str)
        )
        direct_mask = (
            (out["gate_id"] == gate_id)
            & (out["load_type"] == load_type)
            & (out["status"] == "CRITICAL")
            & (out["product_type"].isin(["Cal", "Quarter"]))
        )
        for idx, row in out.loc[direct_mask].iterrows():
            product = str(row["product"])
            if not _direct_parent_conflict_is_explained_by_finer_buckets(
                frame=frame,
                product=product,
                buckets=buckets,
                targets=targets,
                quote_aware_pass=quote_aware_pass,
                price_column=price_column,
                hard_tolerance=hard_tolerance,
            ):
                continue
            parent_mask = product_mask(frame, product)
            bucket_names = sorted(str(bucket) for bucket in buckets.loc[parent_mask].dropna().unique())
            out.loc[idx, "status"] = "QUOTE_CONFLICT"
            out.loc[idx, "severity"] = _severity("QUOTE_CONFLICT")
            out.loc[idx, "evidence"] = "redundant_parent_quote_conflict_finer_buckets_pass"
            out.loc[idx, "remediation_hint"] = _remediation_hint(gate_id, "QUOTE_CONFLICT")
            out.loc[idx, "quote_conflict_basis"] = "finer_quote_aware_buckets"
            out.loc[idx, "covered_by_quote_aware_products"] = ",".join(bucket_names)
            direct_conflict_products.add((load_type, product))

    offpeak_mask = (out["gate_id"] == "implied_offpeak_identity") & (out["status"] == "CRITICAL")
    for idx, row in out.loc[offpeak_mask].iterrows():
        product = str(row["product"])
        if ("BASE", product) not in direct_conflict_products and ("PEAK", product) not in direct_conflict_products:
            continue
        out.loc[idx, "status"] = "QUOTE_CONFLICT"
        out.loc[idx, "severity"] = _severity("QUOTE_CONFLICT")
        out.loc[idx, "evidence"] = "implied_offpeak_quote_conflict_from_parent_base_peak"
        out.loc[idx, "remediation_hint"] = _remediation_hint(
            "implied_offpeak_identity",
            "QUOTE_CONFLICT",
        )
        out.loc[idx, "quote_conflict_basis"] = "parent_base_peak_quote_conflict"
        out.loc[idx, "covered_by_quote_aware_products"] = product
    return out


def _direct_parent_conflict_is_explained_by_finer_buckets(
    *,
    frame: pd.DataFrame,
    product: str,
    buckets: pd.Series,
    targets: Mapping[str, float],
    quote_aware_pass: set[str],
    price_column: str,
    hard_tolerance: float,
) -> bool:
    parent_mask = product_mask(frame, product)
    if not bool(parent_mask.any()):
        return False
    parent_buckets = buckets.loc[parent_mask]
    if parent_buckets.isna().any():
        return False
    bucket_names = {str(bucket) for bucket in parent_buckets.dropna().unique()}
    if not bucket_names:
        return False
    if product in bucket_names or f"{product}-RESIDUAL" in bucket_names:
        return False
    if not bucket_names.issubset(quote_aware_pass):
        return False
    implied = 0.0
    total_rows = 0
    for bucket, idx in parent_buckets.groupby(parent_buckets).groups.items():
        bucket_s = str(bucket)
        if bucket_s not in targets:
            return False
        rows = len(idx)
        implied += float(targets[bucket_s]) * rows
        total_rows += rows
    if total_rows <= 0:
        return False
    curve_mean = float(frame.loc[parent_mask, price_column].mean())
    return abs(curve_mean - implied / total_rows) <= hard_tolerance * 10


def build_summary(
    gates: pd.DataFrame,
    *,
    csv_path: Path,
    forwards_path: Path,
    forward_date: pd.Timestamp,
    market: str,
    price_column: str,
    hard_tolerance: float,
    command_argv: list[str] | None = None,
    source_hierarchy_policy: Mapping[str, Any] | None = None,
    source_hierarchy_policy_path: Path | None = None,
) -> dict[str, Any]:
    if gates.empty:
        status_counts: dict[str, int] = {}
        critical_count = 0
        quote_conflict_count = 0
        delivered_curve_drift_count = 0
        hard_max = None
    else:
        status_counts = {str(k): int(v) for k, v in gates["status"].value_counts().sort_index().items()}
        critical_count = int((gates["status"] == "CRITICAL").sum())
        quote_conflict_count = int((gates["status"] == "QUOTE_CONFLICT").sum())
        delivered_curve_drift_count = int(
            (
                gates["status"].eq("CRITICAL")
                & gates["gate_id"].isin(
                    [
                        "hard_base_product_repricing",
                        "hard_peak_product_repricing",
                        "hard_offpeak_product_repricing",
                        "implied_offpeak_identity",
                    ]
                )
            ).sum()
        )
        supported = gates[gates["status"] != "UNSUPPORTED"]
        hard_max = (
            None
            if supported.empty
            else float(pd.to_numeric(supported["abs_residual_eur_mwh"], errors="coerce").max())
        )
    unsupported_count = int(status_counts.get("UNSUPPORTED", 0))
    policy_hash = _sha256_file(source_hierarchy_policy_path) if source_hierarchy_policy_path is not None else None
    policy_eval = evaluate_source_hierarchy_policy(
        source_hierarchy_policy,
        policy_path=source_hierarchy_policy_path,
        policy_hash=policy_hash,
        market=market,
        forward_date=forward_date,
        quote_conflict_count=quote_conflict_count,
    )
    blocking_quote_conflict_count = int(policy_eval["blocking_quote_conflict_count"])
    all_gates_pass = critical_count == 0 and unsupported_count == 0 and blocking_quote_conflict_count == 0
    script_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "read_only": True,
        "market": market.upper(),
        "price_column": price_column,
        "audit_script": str(script_path),
        "audit_script_sha256": _sha256_file(script_path),
        "command_argv": command_argv,
        "forward_snapshot_date": forward_date.date().isoformat(),
        "hard_tolerance_eur_mwh": hard_tolerance,
        "input_csv": str(csv_path),
        "input_csv_sha256": _sha256_file(csv_path),
        "forwards_path": str(forwards_path),
        "forwards_sha256": _sha256_file(forwards_path),
        "status_counts": status_counts,
        "critical_count": critical_count,
        "quote_conflict_count": quote_conflict_count,
        "accepted_quote_conflict_count": int(policy_eval["accepted_quote_conflict_count"]),
        "blocking_quote_conflict_count": blocking_quote_conflict_count,
        "delivered_curve_drift_count": delivered_curve_drift_count,
        "unsupported_count": unsupported_count,
        "source_hierarchy_policy": policy_eval,
        "supported_hard_gate_max_abs_residual_eur_mwh": hard_max,
        "supported_hard_gates_no_critical": critical_count == 0,
        "covered_hard_gates_pass": all_gates_pass,
        "all_gates_pass": all_gates_pass,
        "note": (
            "UNSUPPORTED rows mean the delivered CSV or quote snapshot did not fully cover "
            "that product window; QUOTE_CONFLICT rows mean redundant parent quotes conflict "
            "with finer quote-aware buckets. QUOTE_CONFLICT can be accepted only by an "
            "explicit production-approved source hierarchy policy artifact."
        ),
    }


def run_audit(
    *,
    csv_path: Path,
    forwards_path: Path,
    market: str = "CH",
    price_column: str = DEFAULT_PRICE_COLUMN,
    required_forward_date: str | None = None,
    hard_tolerance: float = 1e-6,
    peak_country: str = "CH",
    command_argv: list[str] | None = None,
    required_load_types: tuple[str, ...] = ("BASE", "PEAK"),
    source_hierarchy_policy_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hourly = load_delivered_hourly_csv(csv_path, price_column=price_column)
    forward_date, forwards = load_forward_snapshot(
        forwards_path,
        market=market,
        required_forward_date=required_forward_date,
    )
    gates = build_product_normalization_gates(
        hourly,
        forwards,
        forward_date=forward_date,
        market=market,
        price_column=price_column,
        hard_tolerance=hard_tolerance,
        peak_country=peak_country,
        required_load_types=required_load_types,
    )
    summary = build_summary(
        gates,
        csv_path=csv_path,
        forwards_path=forwards_path,
        forward_date=forward_date,
        market=market,
        price_column=price_column,
        hard_tolerance=hard_tolerance,
        command_argv=command_argv,
        source_hierarchy_policy=load_source_hierarchy_policy(source_hierarchy_policy_path),
        source_hierarchy_policy_path=source_hierarchy_policy_path,
    )
    return gates, summary


def _parse_load_types(value: str) -> tuple[str, ...]:
    out = tuple(part.strip().upper() for part in str(value).split(",") if part.strip())
    invalid = sorted(set(out) - {"BASE", "PEAK", "OFFPEAK"})
    if invalid:
        raise ValueError(f"unsupported required load types: {invalid}")
    if not out:
        raise ValueError("at least one required load type must be provided")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Delivered hourly CH CSV to audit.")
    parser.add_argument("--forwards", required=True, type=Path, help="EEX forwards parquet.")
    parser.add_argument("--market", default="CH")
    parser.add_argument("--price-column", default=DEFAULT_PRICE_COLUMN)
    parser.add_argument("--required-forward-date", default=None)
    parser.add_argument("--hard-tolerance", type=float, default=1e-6)
    parser.add_argument("--peak-country", default="CH")
    parser.add_argument(
        "--required-load-types",
        default="BASE,PEAK",
        help="Comma-separated load types that must be present for fully covered quoted products.",
    )
    parser.add_argument(
        "--source-hierarchy-policy",
        type=Path,
        default=None,
        help=(
            "Optional manifest-backed policy for accepting redundant parent quote conflicts. "
            "The policy must be production_approved=true before QUOTE_CONFLICT stops blocking."
        ),
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument(
        "--allow-failed-gates",
        action="store_true",
        help="Write outputs and return 0 even when CRITICAL/UNSUPPORTED/QUOTE_CONFLICT gates are present.",
    )
    parser.add_argument("--fail-on-critical", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fail-on-unsupported", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    gates, summary = run_audit(
        csv_path=args.csv,
        forwards_path=args.forwards,
        market=args.market,
        price_column=args.price_column,
        required_forward_date=args.required_forward_date,
        hard_tolerance=args.hard_tolerance,
        peak_country=args.peak_country,
        command_argv=sys.argv if argv is None else ["audit_ch_product_normalization.py", *argv],
        required_load_types=_parse_load_types(args.required_load_types),
        source_hierarchy_policy_path=args.source_hierarchy_policy,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    gates.to_csv(args.output_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_critical and int(summary["critical_count"]) > 0:
        return 1
    if args.fail_on_unsupported and int(summary["unsupported_count"]) > 0:
        return 1
    if not args.allow_failed_gates and not bool(summary["all_gates_pass"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
