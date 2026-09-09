"""
forward_proxy.py
----------------
Derive forward price estimates from spot history when live EEX forwards
are not available (e.g., no XLSX report or Databricks connection).

Method:
    1. Compute seasonal shape from recent 2-year EPEX spot history
       (monthly ratio vs global mean).
    2. Estimate Cal-year forward levels using:
       - Spot-based anchor: trailing 6-month average (more stable than 3m)
       - Term-structure decay: estimated from historical year-over-year
         price changes (mean reversion).
    3. Build full monthly, quarterly, and annual base prices.

This is a PROXY only. In production, base_prices should come from
live EEX forward quotes (XLSX desk report or API).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.path_safety import read_stable_single_link_file

logger = logging.getLogger(__name__)
_VERIFIED_WORKBOOK_TOKEN = object()
PRODUCTION_EEX_MAX_AGE_BUSINESS_DAYS = 1


def _read_preflight_eex_workbook(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    selected = Path(path).resolve(strict=True)
    payload = read_stable_single_link_file(
        selected,
        label=label,
        max_bytes=64 * 1024 * 1024,
    )
    from pfc_shaping.data.eex_forward_vintage_intake import (
        preflight_eex_forward_workbook_bytes,
    )

    preflight_eex_forward_workbook_bytes(payload)
    return selected, payload


class ForwardSourceUnavailableError(RuntimeError):
    """Raised when production requires a market forward source but none is available."""


@dataclass(frozen=True)
class ForwardSnapshot:
    """Point-in-time market-forward evidence consumed by hard constraints."""

    prices: Mapping[str, float]
    market: str
    source_kind: str
    source_description: str
    snapshot_date: pd.Timestamp | None
    available_at: pd.Timestamp | None
    source_path: str | None
    source_sha256: str | None
    quote_lineage: Mapping[str, Mapping[str, object]] | None = None
    source_sheet: str | None = None
    parser_version: str = "ingest_forwards.v1"
    selection_mode: str = "latest_available"
    schema_version: str = "forward_snapshot.v1"
    _verification_token: InitVar[object | None] = None
    hard_quote_eligible: bool = field(init=False)
    quote_set_sha256: str = field(init=False)
    quote_lineage_sha256: str = field(init=False)
    snapshot_id: str = field(init=False)
    observation_id: str = field(init=False)

    def __post_init__(self, _verification_token: object | None) -> None:
        clean_prices: dict[str, float] = {}
        for key, value in dict(self.prices).items():
            price = float(value)
            if not np.isfinite(price):
                raise ValueError(f"non-finite forward price for product {key}")
            clean_prices[str(key)] = price
        object.__setattr__(self, "prices", MappingProxyType(clean_prices))
        object.__setattr__(self, "market", str(self.market).upper())
        object.__setattr__(self, "source_kind", str(self.source_kind).upper())
        object.__setattr__(self, "snapshot_date", _utc_timestamp(self.snapshot_date))
        object.__setattr__(self, "available_at", _aware_utc_timestamp(self.available_at))
        clean_lineage: dict[str, Mapping[str, object]] = {}
        for key, raw_lineage in dict(self.quote_lineage or {}).items():
            lineage = dict(raw_lineage)
            if str(self.source_kind).upper() == "EEX_VINTAGE":
                if not _is_sha256(lineage.get("quote_id")):
                    raise ValueError("EEX vintage lineage requires its authenticated quote_id")
            else:
                lineage["quote_id"] = _sha256_json(
                    {
                        "source_sha256": self.source_sha256,
                        "market": self.market,
                        "source_sheet": self.source_sheet,
                        "snapshot_date": _iso_or_none(self.snapshot_date),
                        "product": str(key),
                        "source_product_code": lineage.get("source_product_code"),
                        "source_row_index": lineage.get("source_row_index"),
                        "source_column_index": lineage.get("source_column_index"),
                        "price_eur_mwh": clean_prices.get(str(key)),
                    }
                )
            clean_lineage[str(key)] = _deep_freeze(lineage)
        object.__setattr__(self, "quote_lineage", MappingProxyType(clean_lineage))
        quote_payload = [[key, clean_prices[key]] for key in sorted(clean_prices)]
        quote_hash = _sha256_json(quote_payload)
        object.__setattr__(self, "quote_set_sha256", quote_hash)
        lineage_payload = {
            key: _deep_thaw(clean_lineage[key])
            for key in sorted(clean_lineage)
        }
        lineage_hash = _sha256_json(lineage_payload)
        object.__setattr__(self, "quote_lineage_sha256", lineage_hash)
        source_kind = str(self.source_kind).upper()
        lineage_complete = set(clean_lineage) == set(clean_prices) and all(
            bool(lineage.get("is_direct_market_quote", False))
            for lineage in clean_lineage.values()
        )
        eligible = (
            source_kind in {"EEX_XLSX", "EEX_UNC", "EEX_VINTAGE"}
            and bool(clean_prices)
            and self.snapshot_date is not None
            and self.available_at is not None
            and bool(self.source_path)
            and _is_sha256(self.source_sha256)
            and lineage_complete
            and _verification_token is _VERIFIED_WORKBOOK_TOKEN
        )
        object.__setattr__(self, "hard_quote_eligible", eligible)
        identity = {
            "schema_version": self.schema_version,
            "market": self.market,
            "source_kind": source_kind,
            "snapshot_date": _iso_or_none(self.snapshot_date),
            "source_sha256": self.source_sha256,
            "source_sheet": self.source_sheet,
            "parser_version": self.parser_version,
            "selection_mode": self.selection_mode,
            "quote_set_sha256": quote_hash,
            "quote_lineage_sha256": lineage_hash,
        }
        snapshot_id = _sha256_json(identity)
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(
            self,
            "observation_id",
            _sha256_json(
                {
                    "snapshot_id": snapshot_id,
                    "available_at": _iso_or_none(self.available_at),
                }
            ),
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "market": self.market,
            "source_kind": self.source_kind,
            "source_description": self.source_description,
            "snapshot_date": _iso_or_none(self.snapshot_date),
            "available_at": _iso_or_none(self.available_at),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_sheet": self.source_sheet,
            "parser_version": self.parser_version,
            "selection_mode": self.selection_mode,
            "hard_quote_eligible": bool(self.hard_quote_eligible),
            "quote_count": len(self.prices),
            "quote_set_sha256": self.quote_set_sha256,
            "quote_lineage_sha256": self.quote_lineage_sha256,
            "snapshot_id": self.snapshot_id,
            "observation_id": self.observation_id,
            "quotes": [
                {
                    "product": key,
                    "load_type": (
                        "PEAK" if key.lower().endswith("-peak") else
                        "OFFPEAK" if key.lower().endswith("-offpeak") else "BASE"
                    ),
                    "price_eur_mwh": float(self.prices[key]),
                    **_deep_thaw(self.quote_lineage[key]),
                }
                for key in sorted(self.prices)
                if key in self.quote_lineage
            ],
        }


@dataclass(frozen=True)
class ForwardEligibility:
    snapshot_id: str
    observation_id: str
    source_sha256: str
    quote_set_sha256: str
    quote_lineage_sha256: str
    reference_timestamp: pd.Timestamp
    available_at: pd.Timestamp
    business_age_days: int
    max_age_business_days: int
    requested_role: str = "HARD_LEVEL"
    schema_version: str = "forward_eligibility.v1"
    status: str = field(init=False, default="PASS")
    policy_sha256: str = field(init=False)
    eligibility_id: str = field(init=False)

    def __post_init__(self) -> None:
        reference = _aware_utc_timestamp(self.reference_timestamp)
        available_at = _aware_utc_timestamp(self.available_at)
        assert reference is not None
        assert available_at is not None
        object.__setattr__(self, "reference_timestamp", reference)
        object.__setattr__(self, "available_at", available_at)
        policy_hash = _sha256_json(self.policy)
        object.__setattr__(self, "policy_sha256", policy_hash)
        object.__setattr__(
            self,
            "eligibility_id",
            _sha256_json(
                {
                    "schema_version": self.schema_version,
                    "status": self.status,
                    "requested_role": self.requested_role,
                    "snapshot_id": self.snapshot_id,
                    "observation_id": self.observation_id,
                    "source_sha256": self.source_sha256,
                    "quote_set_sha256": self.quote_set_sha256,
                    "quote_lineage_sha256": self.quote_lineage_sha256,
                    "reference_timestamp": reference.isoformat(),
                    "available_at": available_at.isoformat(),
                    "business_age_days": int(self.business_age_days),
                    "policy_sha256": policy_hash,
                }
            ),
        )

    @property
    def policy(self) -> dict[str, object]:
        return {
            "policy_version": "eex_freshness.weekday.v1",
            "max_age_business_days": int(self.max_age_business_days),
            "calendar": "WEEKDAY_MON_FRI",
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "requested_role": self.requested_role,
            "snapshot_id": self.snapshot_id,
            "observation_id": self.observation_id,
            "source_sha256": self.source_sha256,
            "quote_set_sha256": self.quote_set_sha256,
            "quote_lineage_sha256": self.quote_lineage_sha256,
            "reference_timestamp": self.reference_timestamp.isoformat(),
            "available_at": self.available_at.isoformat(),
            "business_age_days": int(self.business_age_days),
            "policy": self.policy,
            "policy_sha256": self.policy_sha256,
            "eligibility_id": self.eligibility_id,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_manifest()[key]


def validate_forward_snapshot(
    snapshot: ForwardSnapshot,
    *,
    reference_timestamp: str | pd.Timestamp | None = None,
    max_age_business_days: int = 1,
) -> ForwardEligibility:
    """Fail closed unless a snapshot is admissible as production quote evidence."""

    if not snapshot.hard_quote_eligible:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source {snapshot.source_kind} is not eligible for hard constraints"
        )
    if snapshot.source_kind in {"SPOT_PROXY", "CONFIG_FALLBACK", "UNKNOWN", "TEST_FIXTURE"}:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source kind {snapshot.source_kind} is forbidden in production"
        )
    if not snapshot.prices:
        raise ForwardSourceUnavailableError(f"{snapshot.market} forward snapshot contains no quotes")
    if snapshot.snapshot_date is None:
        raise ForwardSourceUnavailableError(f"{snapshot.market} forward snapshot date is missing")
    if snapshot.available_at is None:
        raise ForwardSourceUnavailableError(f"{snapshot.market} forward availability timestamp is missing")
    if not snapshot.source_path or not snapshot.source_sha256:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source path/hash evidence is incomplete"
        )
    source_path = Path(snapshot.source_path)
    if not source_path.is_file():
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source path does not exist: {source_path}"
        )
    try:
        report_bytes = read_stable_single_link_file(
            source_path,
            label=f"{snapshot.market} forward source",
            max_bytes=64 * 1024 * 1024,
        )
    except (OSError, ValueError) as exc:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source cannot be read: {source_path}"
        ) from exc
    if _sha256_bytes(report_bytes) != snapshot.source_sha256:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source hash does not match current source bytes"
        )
    try:
        from pfc_shaping.data.eex_forward_vintage_intake import (
            preflight_eex_forward_workbook_bytes,
        )
        from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report_bytes

        preflight_eex_forward_workbook_bytes(report_bytes)
        parsed_prices, parsed_date, parsed_metadata = load_base_prices_from_eex_report_bytes(
            report_bytes,
            market=str(snapshot.source_sheet or snapshot.market),
            source_label=str(source_path),
            return_snapshot_metadata=True,
        )
    except Exception as exc:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source cannot be reparsed as the declared workbook snapshot: {exc}"
        ) from exc
    if dict(parsed_prices) != dict(snapshot.prices):
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward prices do not match reparsed source bytes"
        )
    parsed_date_utc = _utc_timestamp(parsed_date)
    if parsed_date_utc != snapshot.snapshot_date:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward snapshot date does not match reparsed source bytes"
        )
    expected_lineage = dict(parsed_metadata["quote_lineage"])
    if str(snapshot.source_kind).upper() == "EEX_VINTAGE":
        required_vintage_fields = {
            "quote_id",
            "revision_id",
            "snapshot_id",
            "acquisition_id",
            "source_document_id",
            "parser_config_sha256",
        }
        for key, parsed_lineage in expected_lineage.items():
            observed = _deep_thaw(snapshot.quote_lineage[key])
            missing = sorted(required_vintage_fields - observed.keys())
            if missing:
                raise ForwardSourceUnavailableError(
                    f"{snapshot.market} EEX vintage lineage is incomplete for {key}: "
                    + ", ".join(missing)
                )
            if any(observed.get(name) != value for name, value in parsed_lineage.items()):
                raise ForwardSourceUnavailableError(
                    f"{snapshot.market} EEX vintage lineage does not match reparsed source bytes"
                )
    else:
        observed_lineage = {
            key: {
                name: value
                for name, value in _deep_thaw(snapshot.quote_lineage[key]).items()
                if name != "quote_id"
            }
            for key in snapshot.quote_lineage
        }
        if expected_lineage != observed_lineage:
            raise ForwardSourceUnavailableError(
                f"{snapshot.market} forward quote lineage does not match reparsed source bytes"
            )

    reference = _utc_timestamp(reference_timestamp or pd.Timestamp.now("UTC"))
    assert reference is not None
    if snapshot.snapshot_date > reference:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward snapshot {snapshot.snapshot_date.isoformat()} is in the future"
        )
    if snapshot.available_at > reference:
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source availability {snapshot.available_at.isoformat()} is in the future"
        )
    if snapshot.available_at.date() < snapshot.snapshot_date.date():
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward source was available before its quoted snapshot date"
        )

    snapshot_day = np.datetime64(snapshot.snapshot_date.date(), "D")
    reference_day = np.datetime64(reference.tz_convert("Europe/Zurich").date(), "D")
    business_age = int(np.busday_count(snapshot_day, reference_day))
    if business_age < 0:
        raise ForwardSourceUnavailableError(f"{snapshot.market} forward snapshot business age is negative")
    if business_age > int(max_age_business_days):
        raise ForwardSourceUnavailableError(
            f"{snapshot.market} forward snapshot is stale "
            f"({business_age} business days > {int(max_age_business_days)})"
        )
    return ForwardEligibility(
        snapshot_id=snapshot.snapshot_id,
        observation_id=snapshot.observation_id,
        source_sha256=str(snapshot.source_sha256),
        quote_set_sha256=snapshot.quote_set_sha256,
        quote_lineage_sha256=snapshot.quote_lineage_sha256,
        reference_timestamp=reference,
        available_at=snapshot.available_at,
        business_age_days=business_age,
        max_age_business_days=int(max_age_business_days),
    )


def verify_forward_eligibility(
    snapshot: ForwardSnapshot,
    eligibility: ForwardEligibility,
    *,
    valuation_timestamp: str | pd.Timestamp,
    expected_max_age_business_days: int,
) -> None:
    valuation = _aware_utc_timestamp(valuation_timestamp)
    assert valuation is not None
    if eligibility.reference_timestamp != valuation:
        raise ValueError("forward eligibility reference_timestamp does not match valuation timestamp")
    if eligibility.max_age_business_days != int(expected_max_age_business_days):
        raise ValueError(
            "forward eligibility freshness policy does not match the production policy"
        )
    expected = validate_forward_snapshot(
        snapshot,
        reference_timestamp=valuation,
        max_age_business_days=int(expected_max_age_business_days),
    )
    if eligibility.to_manifest() != expected.to_manifest():
        raise ValueError("forward eligibility receipt does not match recomputed eligibility")


def verify_forward_manifest_binding(
    snapshot_manifest: Mapping[str, object],
    eligibility_manifest: Mapping[str, object],
    *,
    expected_max_age_business_days: int = PRODUCTION_EEX_MAX_AGE_BUSINESS_DAYS,
    verify_source: bool = True,
    expected_reference_timestamp: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Recompute the complete snapshot/eligibility identity from manifest evidence."""

    expected_age = int(expected_max_age_business_days)
    if expected_age < 0:
        raise ValueError("expected_max_age_business_days must be non-negative")
    if snapshot_manifest.get("schema_version") != "forward_snapshot.v1":
        raise ValueError("forward snapshot manifest schema is not forward_snapshot.v1")
    if eligibility_manifest.get("schema_version") != "forward_eligibility.v1":
        raise ValueError("forward eligibility manifest schema is not forward_eligibility.v1")

    market = str(snapshot_manifest.get("market", "")).upper()
    source_kind = str(snapshot_manifest.get("source_kind", "")).upper()
    source_sha256 = str(snapshot_manifest.get("source_sha256", ""))
    source_sheet = str(snapshot_manifest.get("source_sheet") or market)
    parser_version = str(snapshot_manifest.get("parser_version", ""))
    selection_mode = str(snapshot_manifest.get("selection_mode", ""))
    if source_kind not in {"EEX_XLSX", "EEX_UNC"}:
        raise ValueError(f"forbidden hard-forward source kind: {source_kind!r}")
    if not market or not source_sheet:
        raise ValueError("forward snapshot market/source_sheet is missing")
    if not _is_sha256(source_sha256):
        raise ValueError("forward snapshot source_sha256 is invalid")
    if parser_version != "ingest_forwards.v1" or selection_mode != "latest_available":
        raise ValueError("forward snapshot parser/selection policy is unsupported")
    if snapshot_manifest.get("hard_quote_eligible") is not True:
        raise ValueError("forward snapshot is not declared hard-quote eligible")

    snapshot_date = _utc_timestamp(snapshot_manifest.get("snapshot_date"))
    available_at = _aware_utc_timestamp(snapshot_manifest.get("available_at"))
    if snapshot_date is None or available_at is None:
        raise ValueError("forward snapshot date/available_at is missing")

    quotes = snapshot_manifest.get("quotes")
    if not isinstance(quotes, list) or not quotes:
        raise ValueError("forward snapshot quotes are missing")
    prices: dict[str, float] = {}
    lineage: dict[str, dict[str, object]] = {}
    for raw_quote in quotes:
        if not isinstance(raw_quote, Mapping):
            raise ValueError("forward snapshot quote is not an object")
        product = str(raw_quote.get("product", ""))
        if not product or product in prices:
            raise ValueError(f"forward snapshot quote product is missing/duplicated: {product!r}")
        price = float(raw_quote.get("price_eur_mwh"))
        if not np.isfinite(price):
            raise ValueError(f"non-finite forward manifest price for {product}")
        expected_load_type = (
            "PEAK" if product.lower().endswith("-peak") else
            "OFFPEAK" if product.lower().endswith("-offpeak") else "BASE"
        )
        if str(raw_quote.get("load_type", "")).upper() != expected_load_type:
            raise ValueError(f"forward load_type does not match canonical product key for {product}")
        quote_lineage = {
            str(key): _deep_thaw(value)
            for key, value in raw_quote.items()
            if key not in {"product", "load_type", "price_eur_mwh", "quote_id"}
        }
        expected_quote_id = _sha256_json(
            {
                "source_sha256": source_sha256,
                "market": market,
                "source_sheet": source_sheet,
                "snapshot_date": snapshot_date.isoformat(),
                "product": product,
                "source_product_code": quote_lineage.get("source_product_code"),
                "source_row_index": quote_lineage.get("source_row_index"),
                "source_column_index": quote_lineage.get("source_column_index"),
                "price_eur_mwh": price,
            }
        )
        if str(raw_quote.get("quote_id", "")) != expected_quote_id:
            raise ValueError(f"forward quote_id does not match canonical evidence for {product}")
        quote_lineage["quote_id"] = expected_quote_id
        prices[product] = price
        lineage[product] = quote_lineage

    if int(snapshot_manifest.get("quote_count", -1)) != len(prices):
        raise ValueError("forward snapshot quote_count mismatch")
    quote_set_sha256 = _sha256_json([[key, prices[key]] for key in sorted(prices)])
    quote_lineage_sha256 = _sha256_json({key: lineage[key] for key in sorted(lineage)})
    if str(snapshot_manifest.get("quote_set_sha256", "")) != quote_set_sha256:
        raise ValueError("forward snapshot quote_set_sha256 mismatch")
    if str(snapshot_manifest.get("quote_lineage_sha256", "")) != quote_lineage_sha256:
        raise ValueError("forward snapshot quote_lineage_sha256 mismatch")

    identity = {
        "schema_version": "forward_snapshot.v1",
        "market": market,
        "source_kind": source_kind,
        "snapshot_date": snapshot_date.isoformat(),
        "source_sha256": source_sha256,
        "source_sheet": source_sheet,
        "parser_version": parser_version,
        "selection_mode": selection_mode,
        "quote_set_sha256": quote_set_sha256,
        "quote_lineage_sha256": quote_lineage_sha256,
    }
    snapshot_id = _sha256_json(identity)
    observation_id = _sha256_json(
        {"snapshot_id": snapshot_id, "available_at": available_at.isoformat()}
    )
    if str(snapshot_manifest.get("snapshot_id", "")) != snapshot_id:
        raise ValueError("forward snapshot_id mismatch")
    if str(snapshot_manifest.get("observation_id", "")) != observation_id:
        raise ValueError("forward observation_id mismatch")

    if verify_source:
        source_path_value = snapshot_manifest.get("source_path")
        if not source_path_value:
            raise ValueError("forward snapshot source_path is missing")
        source_path = Path(str(source_path_value))
        try:
            source_path, report_bytes = _read_preflight_eex_workbook(
                source_path,
                label="forward manifest source",
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"forward snapshot source cannot be read: {source_path}") from exc
        if _sha256_bytes(report_bytes) != source_sha256:
            raise ValueError("forward snapshot source bytes do not match source_sha256")
        from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report_bytes

        parsed_prices, parsed_date, parsed_metadata = load_base_prices_from_eex_report_bytes(
            report_bytes,
            market=source_sheet,
            source_label=str(source_path),
            return_snapshot_metadata=True,
        )
        parsed_date_utc = _utc_timestamp(parsed_date)
        parsed_lineage = {
            key: {name: value for name, value in lineage[key].items() if name != "quote_id"}
            for key in lineage
        }
        if dict(parsed_prices) != prices or parsed_date_utc != snapshot_date:
            raise ValueError("forward manifest prices/date do not match source bytes")
        if dict(parsed_metadata["quote_lineage"]) != parsed_lineage:
            raise ValueError("forward manifest lineage does not match source bytes")

    if eligibility_manifest.get("status") != "PASS":
        raise ValueError("forward eligibility status is not PASS")
    if eligibility_manifest.get("requested_role") != "HARD_LEVEL":
        raise ValueError("forward eligibility requested_role is not HARD_LEVEL")
    policy = eligibility_manifest.get("policy")
    expected_policy = {
        "policy_version": "eex_freshness.weekday.v1",
        "max_age_business_days": expected_age,
        "calendar": "WEEKDAY_MON_FRI",
    }
    if not isinstance(policy, Mapping) or dict(policy) != expected_policy:
        raise ValueError("forward eligibility freshness policy mismatch")
    policy_sha256 = _sha256_json(expected_policy)
    if str(eligibility_manifest.get("policy_sha256", "")) != policy_sha256:
        raise ValueError("forward eligibility policy_sha256 mismatch")

    reference = _aware_utc_timestamp(eligibility_manifest.get("reference_timestamp"))
    eligibility_available_at = _aware_utc_timestamp(eligibility_manifest.get("available_at"))
    if reference is None or eligibility_available_at is None:
        raise ValueError("forward eligibility reference/available_at is missing")
    if expected_reference_timestamp is not None:
        expected_reference = _aware_utc_timestamp(expected_reference_timestamp)
        if reference != expected_reference:
            raise ValueError(
                "forward eligibility reference_timestamp does not match promotion timestamp"
            )
    if eligibility_available_at != available_at:
        raise ValueError("forward eligibility available_at does not match snapshot")
    if snapshot_date > reference or available_at > reference:
        raise ValueError("forward snapshot evidence is after the eligibility reference")
    if available_at.date() < snapshot_date.date():
        raise ValueError("forward snapshot available_at predates snapshot_date")
    business_age = int(
        np.busday_count(
            np.datetime64(snapshot_date.date(), "D"),
            np.datetime64(reference.tz_convert("Europe/Zurich").date(), "D"),
        )
    )
    if business_age < 0 or business_age > expected_age:
        raise ValueError("forward eligibility business age violates production policy")
    if int(eligibility_manifest.get("business_age_days", -1)) != business_age:
        raise ValueError("forward eligibility business_age_days mismatch")
    for key, expected_value in {
        "snapshot_id": snapshot_id,
        "observation_id": observation_id,
        "source_sha256": source_sha256,
        "quote_set_sha256": quote_set_sha256,
        "quote_lineage_sha256": quote_lineage_sha256,
    }.items():
        if str(eligibility_manifest.get(key, "")) != expected_value:
            raise ValueError(f"forward eligibility {key} mismatch")
    eligibility_id = _sha256_json(
        {
            "schema_version": "forward_eligibility.v1",
            "status": "PASS",
            "requested_role": "HARD_LEVEL",
            "snapshot_id": snapshot_id,
            "observation_id": observation_id,
            "source_sha256": source_sha256,
            "quote_set_sha256": quote_set_sha256,
            "quote_lineage_sha256": quote_lineage_sha256,
            "reference_timestamp": reference.isoformat(),
            "available_at": available_at.isoformat(),
            "business_age_days": business_age,
            "policy_sha256": policy_sha256,
        }
    )
    if str(eligibility_manifest.get("eligibility_id", "")) != eligibility_id:
        raise ValueError("forward eligibility_id mismatch")
    return {
        "snapshot_id": snapshot_id,
        "observation_id": observation_id,
        "source_sha256": source_sha256,
        "quote_set_sha256": quote_set_sha256,
        "quote_lineage_sha256": quote_lineage_sha256,
        "policy_sha256": policy_sha256,
        "eligibility_id": eligibility_id,
        "business_age_days": business_age,
        "max_age_business_days": expected_age,
    }


def _utc_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _aware_utc_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("available_at must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _iso_or_none(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(payload: object) -> str:
    import json

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_sha256(value: str | None) -> bool:
    if value is None or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def derive_base_prices(
    epex: pd.DataFrame,
    start_year: int | None = None,
    n_years: int = 4,
    anchor_months: int = 6,
) -> dict[str, float]:
    """Derive base prices from EPEX spot history.

    Args:
        epex: DataFrame with 'price_eur_mwh' column, DatetimeIndex UTC.
        start_year: First delivery year. Default: current year.
        n_years: Number of forward years to generate.
        anchor_months: Months of trailing spot to anchor level.

    Returns:
        Dict with keys like '2026', '2026-Q1', '2026-01' etc.
    """
    if start_year is None:
        start_year = pd.Timestamp.now().year

    idx_zh = epex.index.tz_convert("Europe/Zurich")

    # ── 1. Trailing anchor level ──────────────────────────────────────
    n_rows = min(96 * 30 * anchor_months, len(epex))
    anchor = epex["price_eur_mwh"].iloc[-n_rows:].mean()
    logger.info("Forward proxy anchor (trailing %dm): %.1f EUR/MWh", anchor_months, anchor)

    # ── 2. Seasonal shape from last 2+ years ─────────────────────────
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=2)
    recent = epex[epex.index >= cutoff]
    if len(recent) < 96 * 180:  # less than 6 months
        recent = epex  # fallback to full history

    recent_zh = recent.index.tz_convert("Europe/Zurich")
    monthly_avg = recent.groupby(recent_zh.month)["price_eur_mwh"].mean()
    global_mean = monthly_avg.mean()

    if abs(global_mean) < 1.0:
        seasonal_ratio = pd.Series(1.0, index=range(1, 13))
    else:
        seasonal_ratio = monthly_avg / global_mean

    # ── 2b. Peak/Base ratio from history ───────────────────────────
    # Peak = weekday 08:00-20:00 Zurich time
    is_weekday = recent_zh.dayofweek < 5
    is_peak_hour = (recent_zh.hour >= 8) & (recent_zh.hour < 20)
    peak_mask = is_weekday & is_peak_hour

    peak_avg_by_month = recent[peak_mask].groupby(recent_zh[peak_mask].month)["price_eur_mwh"].mean()
    base_avg_by_month = recent.groupby(recent_zh.month)["price_eur_mwh"].mean()

    peak_ratio_by_month = {}
    for m in range(1, 13):
        if m in peak_avg_by_month.index and m in base_avg_by_month.index and base_avg_by_month[m] > 0:
            peak_ratio_by_month[m] = float(peak_avg_by_month[m] / base_avg_by_month[m])
        else:
            peak_ratio_by_month[m] = 1.15  # default Swiss peak/base ratio

    logger.info(
        "Peak/Base ratios: winter=%.2f, summer=%.2f",
        np.mean([peak_ratio_by_month.get(m, 1.15) for m in [12, 1, 2]]),
        np.mean([peak_ratio_by_month.get(m, 1.15) for m in [6, 7, 8]]),
    )

    logger.info(
        "Seasonal ratios: winter=%.2f, summer=%.2f",
        seasonal_ratio[[12, 1, 2]].mean(),
        seasonal_ratio[[6, 7, 8]].mean(),
    )

    # ── 3. Term-structure decay (mean reversion) ─────────────────────
    # Estimate from year-over-year changes in the spot history
    yearly_avg = epex.groupby(idx_zh.year)["price_eur_mwh"].mean()
    if len(yearly_avg) >= 3:
        yoy_ratios = yearly_avg.pct_change().dropna()
        # Dampened mean reversion: converge toward long-term average
        annual_decay = float(np.clip(yoy_ratios.median(), -0.10, 0.0))
    else:
        annual_decay = -0.03  # conservative 3% backwardation per year

    logger.info("Estimated annual decay: %.1f%%", annual_decay * 100)

    # ── 4. Build Cal-year levels ─────────────────────────────────────
    base_prices: dict[str, float] = {}
    for i in range(n_years):
        year = start_year + i
        cal_level = anchor * (1 + annual_decay) ** i
        base_prices[str(year)] = round(cal_level, 1)

        # Monthly prices (base + peak)
        for month in range(1, 13):
            ratio = seasonal_ratio.get(month, 1.0)
            monthly_base = round(cal_level * ratio, 1)
            base_prices[f"{year}-{month:02d}"] = monthly_base
            # Peak price = base × peak/base ratio for that month
            pk_ratio = peak_ratio_by_month.get(month, 1.15)
            base_prices[f"{year}-{month:02d}-Peak"] = round(monthly_base * pk_ratio, 1)

        # Quarterly prices (hour-weighted average of months for energy conservation)
        import calendar
        for q, q_months in {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}.items():
            q_prices = [base_prices[f"{year}-{m:02d}"] for m in q_months]
            q_peak_prices = [base_prices[f"{year}-{m:02d}-Peak"] for m in q_months]
            hours = [calendar.monthrange(year, m)[1] * 24 for m in q_months]
            total_h = sum(hours)
            base_prices[f"{year}-Q{q}"] = round(
                sum(p * h for p, h in zip(q_prices, hours)) / total_h, 1)
            base_prices[f"{year}-Q{q}-Peak"] = round(
                sum(p * h for p, h in zip(q_peak_prices, hours)) / total_h, 1)

    logger.info(
        "Forward proxy: %d keys, Cal range: %s",
        len(base_prices),
        {k: f"{v:.0f}" for k, v in base_prices.items() if len(k) == 4},
    )

    return base_prices


def _observation_timestamp_utc() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def load_forward_snapshot(
    epex: pd.DataFrame,
    eex_report_path: str | None = None,
    config: dict | None = None,
    market: str = "CH",
    allow_spot_proxy: bool = True,
    *,
    exact_source_only: bool = False,
    source_available_at: str | pd.Timestamp | None = None,
) -> ForwardSnapshot:
    """Load forwards and retain the evidence needed for production use.

    Args:
        epex: EPEX spot DataFrame for fallback proxy derivation.
        eex_report_path: Path to EEX XLSX report.
        config: Optional config dict.
        market: 'CH' or 'DE' — EEX sheet to load from.

    Returns:
        Structured point-in-time snapshot. Call ``validate_forward_snapshot``
        before using its prices as production hard constraints.
    """
    if SOURCE_REVISION is not None and not exact_source_only:
        raise ForwardSourceUnavailableError(
            "installed governed runtime requires exact_source_only EEX replay"
        )
    if exact_source_only:
        if not eex_report_path:
            raise ForwardSourceUnavailableError(
                "governed forward loading requires one explicit EEX source"
            )
        available_at = _aware_utc_timestamp(source_available_at)
        if available_at is None:
            raise ForwardSourceUnavailableError(
                "governed forward loading requires authenticated source availability"
            )
        try:
            snapshot = _load_eex_workbook_snapshot(
                eex_report_path,
                market=market,
                available_at=available_at,
                source_kind="EEX_XLSX",
            )
        except (OSError, ValueError) as exc:
            raise ForwardSourceUnavailableError(
                f"authenticated EEX source is invalid for {market}"
            ) from exc
        if len(snapshot.prices) < 3:
            raise ForwardSourceUnavailableError(
                f"authenticated EEX source has insufficient {market} quote coverage"
            )
        return snapshot

    # ── Try 1: EEX XLSX report (desk quotidien) ──────────────────────
    if eex_report_path or (config and config.get("forwards", {}).get("eex_report_path")):
        path = eex_report_path or config["forwards"]["eex_report_path"]
        try:
            from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report_bytes

            # The explicit function argument takes precedence. The config
            # fallback is only used when callers do not pass a market.
            eex_market = market or "CH"
            if config and not market:
                eex_market = config.get("forwards", {}).get("eex_market", eex_market)

            selected, report_bytes = _read_preflight_eex_workbook(
                path,
                label="EEX XLSX forward source",
            )
            resolved = str(selected)
            observed_at = _observation_timestamp_utc()
            source_sha256 = hashlib.sha256(report_bytes).hexdigest()
            prices, snapshot_date, snapshot_metadata = load_base_prices_from_eex_report_bytes(
                report_bytes,
                market=eex_market,
                source_label=resolved,
                return_snapshot_metadata=True,
            )
            if prices and len(prices) >= 3:
                logger.info("Loaded %d EEX forward prices from XLSX (%s): %s", len(prices), eex_market, path)
                return ForwardSnapshot(
                    prices=prices,
                    market=eex_market,
                    source_kind="EEX_XLSX",
                    source_description=f"EEX XLSX {eex_market} ({len(prices)} keys)",
                    snapshot_date=snapshot_date,
                    available_at=observed_at,
                    source_path=resolved,
                    source_sha256=source_sha256,
                    quote_lineage=snapshot_metadata["quote_lineage"],
                    source_sheet=eex_market,
                    _verification_token=_VERIFIED_WORKBOOK_TOKEN,
                )
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("EEX XLSX not available: %s", exc)

    # ── Try 2: EEX report UNC path ───────────────────────────────────
    if config and config.get("forwards", {}).get("eex_report_path_unc"):
        unc_path = config["forwards"]["eex_report_path_unc"]
        try:
            from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report_bytes

            eex_market = market or "CH"
            if not market:
                eex_market = config.get("forwards", {}).get("eex_market", eex_market)
            selected, report_bytes = _read_preflight_eex_workbook(
                unc_path,
                label="EEX UNC forward source",
            )
            resolved = str(selected)
            observed_at = _observation_timestamp_utc()
            source_sha256 = hashlib.sha256(report_bytes).hexdigest()
            prices, snapshot_date, snapshot_metadata = load_base_prices_from_eex_report_bytes(
                report_bytes,
                market=eex_market,
                source_label=resolved,
                return_snapshot_metadata=True,
            )
            if prices and len(prices) >= 3:
                logger.info("Loaded %d EEX forward prices from UNC: %s", len(prices), unc_path)
                return ForwardSnapshot(
                    prices=prices,
                    market=eex_market,
                    source_kind="EEX_UNC",
                    source_description=f"EEX UNC {eex_market} ({len(prices)} keys)",
                    snapshot_date=snapshot_date,
                    available_at=observed_at,
                    source_path=resolved,
                    source_sha256=source_sha256,
                    quote_lineage=snapshot_metadata["quote_lineage"],
                    source_sheet=eex_market,
                    _verification_token=_VERIFIED_WORKBOOK_TOKEN,
                )
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("EEX UNC not available: %s", exc)

    # Databricks is an upstream acquisition concern and is deliberately
    # absent from the governed LT runtime wheel.
    if config and config.get("databricks", {}).get("enabled"):
        logger.warning(
            "Databricks runtime loading is disabled; materialize a governed input snapshot"
        )

    # ── Fallback: Derive from spot ───────────────────────────────────
    logger.info("No EEX forward source available — deriving from spot history")
    if not allow_spot_proxy:
        raise ForwardSourceUnavailableError(
            f"No eligible market forward source available for {market}; "
            "spot-derived proxy is forbidden for production hard constraints"
        )
    prices = derive_base_prices(epex)
    latest = epex.index.max() if isinstance(epex.index, pd.DatetimeIndex) and len(epex) else None
    return ForwardSnapshot(
        prices=prices,
        market=market,
        source_kind="SPOT_PROXY",
        source_description=f"Spot-derived proxy ({len(prices)} keys)",
        snapshot_date=latest,
        available_at=_observation_timestamp_utc(),
        source_path=None,
        source_sha256=None,
    )


def _load_eex_workbook_snapshot(
    path: str | Path,
    *,
    market: str,
    available_at: pd.Timestamp,
    source_kind: str,
) -> ForwardSnapshot:
    from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report_bytes

    eex_market = str(market or "CH").upper()
    selected, report_bytes = _read_preflight_eex_workbook(
        path,
        label="exact EEX workbook source",
    )
    resolved = str(selected)
    source_sha256 = hashlib.sha256(report_bytes).hexdigest()
    prices, snapshot_date, snapshot_metadata = load_base_prices_from_eex_report_bytes(
        report_bytes,
        market=eex_market,
        source_label=resolved,
        return_snapshot_metadata=True,
    )
    return ForwardSnapshot(
        prices=prices,
        market=eex_market,
        source_kind=source_kind,
        source_description=(
            f"{source_kind.replace('_', ' ')} {eex_market} ({len(prices)} keys)"
        ),
        snapshot_date=snapshot_date,
        available_at=available_at,
        source_path=resolved,
        source_sha256=source_sha256,
        quote_lineage=snapshot_metadata["quote_lineage"],
        source_sheet=eex_market,
        _verification_token=_VERIFIED_WORKBOOK_TOKEN,
    )


def forward_snapshot_from_verified_eex_vintage(
    history: pd.DataFrame,
    *,
    evidence: object,
    catalog: Mapping[str, object],
    catalog_path: str | Path,
    market: str = "CH",
) -> ForwardSnapshot:
    """Build hard-level evidence only from a verified PIT vintage selection."""

    from pfc_shaping.data.eex_historical_vintage import (
        VerifiedEexHistoricalVintageCatalog,
    )
    from pfc_shaping.data.shared_data_root import resolve_confined_path
    from pfc_shaping.pipeline.strict_structured_data import load_strict_json

    if (
        not isinstance(evidence, VerifiedEexHistoricalVintageCatalog)
        or not evidence.verified
    ):
        raise ValueError("verified EEX vintage catalog evidence is required")
    catalog_metadata = history.attrs.get("verified_vintage_catalog")
    if (
        not isinstance(catalog_metadata, Mapping)
        or str(catalog_metadata.get("catalog_id", "")) != evidence.catalog_id
        or str(catalog_metadata.get("catalog_sha256", "")) != evidence.catalog_sha256
        or str(catalog_metadata.get("history_sha256", "")) != evidence.history_sha256
    ):
        raise ValueError("EEX vintage frame/catalog evidence binding mismatch")
    selected_market = str(market).upper()
    market_rows = history.loc[
        history["market"].astype(str).str.upper().eq(selected_market)
    ].copy()
    if market_rows.empty:
        raise ValueError(f"verified EEX vintage has no {selected_market} quotes")
    latest_date = pd.to_datetime(market_rows["date"]).max().normalize()
    rows = market_rows.loc[
        pd.to_datetime(market_rows["date"]).dt.normalize().eq(latest_date)
    ].copy()
    if rows.empty:
        raise ValueError("verified EEX vintage latest snapshot is empty")
    single_value_columns = (
        "snapshot_id",
        "source_document_id",
        "source_document_sha256",
        "available_at",
        "parser_version",
    )
    if any(rows[column].nunique(dropna=False) != 1 for column in single_value_columns):
        raise ValueError("verified EEX hard-level snapshot mixes source vintages")
    snapshot_id = str(rows["snapshot_id"].iloc[0])
    documents = catalog.get("source_documents")
    if not isinstance(documents, list):
        raise ValueError("verified EEX catalog source inventory is missing")
    matches = [
        entry
        for entry in documents
        if isinstance(entry, Mapping)
        and snapshot_id in {str(value) for value in entry.get("snapshot_ids", [])}
    ]
    if len(matches) != 1:
        raise ValueError("verified EEX snapshot source document is ambiguous")
    source_entry = matches[0]
    parser_config = source_entry.get("parser_config")
    required_prospective_config = {
        "parser_version",
        "selection_mode",
        "markets",
        "quote_convention",
        "expected_quotes",
        "intake_id",
    }
    if (
        not isinstance(parser_config, Mapping)
        or set(parser_config) != required_prospective_config
        or parser_config.get("quote_convention") != "EEX_SETTLEMENT_EUR_MWH"
    ):
        raise ValueError(
            "verified EEX hard-level source lacks prospective settlement admission"
        )
    source_hash = str(rows["source_document_sha256"].iloc[0])
    if (
        str(source_entry.get("source_document_id", ""))
        != str(rows["source_document_id"].iloc[0])
        or str(source_entry.get("sha256", "")) != source_hash
    ):
        raise ValueError("verified EEX snapshot source binding mismatch")
    catalog_file = Path(catalog_path)
    if not catalog_file.is_absolute():
        raise ValueError("verified EEX catalog path must be absolute")
    catalog_payload = read_stable_single_link_file(
        catalog_file,
        label="verified EEX hard-level catalog",
        max_bytes=4 * 1024 * 1024,
    )
    if hashlib.sha256(catalog_payload).hexdigest() != evidence.catalog_sha256:
        raise ValueError("verified EEX hard-level catalog bytes changed")
    try:
        persisted_catalog = load_strict_json(catalog_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("verified EEX hard-level catalog is unreadable") from exc
    if persisted_catalog != dict(catalog):
        raise ValueError("verified EEX hard-level catalog mapping/path mismatch")
    source_path = resolve_confined_path(
        catalog_file.parent,
        str(source_entry.get("path", "")),
        label="verified EEX hard-level source document",
        require_exists=True,
        require_file=True,
    )
    prices: dict[str, float] = {}
    lineage: dict[str, dict[str, object]] = {}
    for row in rows.to_dict(orient="records"):
        load_type = str(row["load_type"]).upper()
        key = str(row["product"])
        if load_type == "PEAK":
            key = f"{key}-Peak"
        elif load_type == "OFFPEAK":
            key = f"{key}-Offpeak"
        if key in prices:
            raise ValueError("verified EEX hard-level snapshot repeats a product")
        prices[key] = float(row["price"])
        lineage[key] = {
            "quote_id": str(row["quote_id"]),
            "revision_id": str(row["revision_id"]),
            "snapshot_id": str(row["snapshot_id"]),
            "acquisition_id": str(row["acquisition_id"]),
            "source_document_id": str(row["source_document_id"]),
            "source_product_code": str(row["source_product_code"]),
            "source_row_index": int(row["source_row_index"]),
            "source_column_index": int(row["source_column_index"]),
            "parser_config_sha256": str(row["parser_config_sha256"]),
            "is_direct_market_quote": True,
        }
    if not prices or not any(
        not key.lower().endswith(("-peak", "-offpeak")) for key in prices
    ):
        raise ValueError("verified EEX hard-level snapshot has insufficient CH BASE coverage")
    return ForwardSnapshot(
        prices=prices,
        market=selected_market,
        source_kind="EEX_VINTAGE",
        source_description=(
            f"verified catalog-backed EEX vintage {selected_market} ({len(prices)} keys)"
        ),
        snapshot_date=latest_date,
        available_at=pd.Timestamp(rows["available_at"].iloc[0]),
        source_path=str(source_path),
        source_sha256=source_hash,
        quote_lineage=lineage,
        source_sheet=selected_market,
        parser_version=str(rows["parser_version"].iloc[0]),
        selection_mode="catalog_pit_latest_snapshot",
        _verification_token=_VERIFIED_WORKBOOK_TOKEN,
    )


def load_base_prices(
    epex: pd.DataFrame,
    eex_report_path: str | None = None,
    config: dict | None = None,
    market: str = "CH",
    allow_spot_proxy: bool = True,
) -> tuple[dict[str, float], str]:
    """Compatibility wrapper for research and legacy orchestration."""

    snapshot = load_forward_snapshot(
        epex,
        eex_report_path=eex_report_path,
        config=config,
        market=market,
        allow_spot_proxy=allow_spot_proxy,
    )
    return dict(snapshot.prices), snapshot.source_description
