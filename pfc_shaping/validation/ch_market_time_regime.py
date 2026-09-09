"""Fail-closed Swiss market-time-regime contract.

The model's 15-minute valuation grid is deliberately distinct from the native
resolution of observed Swiss auction prices.  A planned market transition is
not an actual go-live and cannot silently upgrade hourly observations to
native quarter-hour truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_market_time_regime_contract.v1"
AUDIT_SCHEMA_VERSION = "ch_market_time_regime_audit.v1"
STATUS = "STRUCTURE_VALID_TRANSITION_NOT_ADMITTED"
CONTRACT_ID = "898d35e37a2df9dc9814039698eb605fdf220ece33d2035c7c7a2ea1b7bc9dba"
DOCUMENT_SHA256 = "71711ae80b64556b8deab88e70581e1c7a0ef7c684d672b2cb550f2058c19c25"
CONTRACT_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-MARKET-TIME-REGIME-CONTRACT-20260730.json"
)

_MAX_CONTRACT_BYTES = 2_000_000
_MAX_EVIDENCE_BYTES = 16_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_name",
    "as_of",
    "scope",
    "regimes",
    "transition_admission_gate",
    "data_and_validation_policy",
    "evidence",
    "supersession",
    "execution_controls",
    "contract_id",
}


class ChMarketTimeRegimeError(ValueError):
    """Raised when the market-time regime is malformed, weakened or unbound."""


@dataclass(frozen=True)
class MarketTimeRegimeAssessment:
    contract_id: str
    document_sha256: str
    status: str
    evidence_files_verified: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "document_sha256": self.document_sha256,
            "status": self.status,
            "native_day_ahead_market_time_unit_minutes": 60,
            "native_intraday_auction_market_time_unit_minutes": 60,
            "planned_first_trading_date": "2026-11-03",
            "confirmed_go_live": False,
            "effective_first_delivery_start_utc": None,
            "transition_admitted": False,
            "evidence_files_verified": self.evidence_files_verified,
            "scientific_admission": False,
            "execution_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
            "blockers": list(self.blockers),
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_contract_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("contract_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def load_market_time_regime_contract(
    path: str | Path,
    *,
    expected_sha256: str,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, object], bytes]:
    _require_sha256(expected_sha256, label="expected contract")
    if expected_sha256 != DOCUMENT_SHA256:
        raise ChMarketTimeRegimeError(
            "caller-held contract SHA-256 does not match frozen canonical identity"
        )
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path(os.path.abspath(selected))
    canonical_root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).expanduser()
    )
    if not canonical_root.is_absolute():
        canonical_root = Path(os.path.abspath(canonical_root))
    canonical = canonical_root.joinpath(
        *CONTRACT_RELATIVE_PATH.split("/")
    )
    if _path_key(selected) != _path_key(canonical):
        raise ChMarketTimeRegimeError("market-time contract path is not canonical")
    payload = read_stable_single_link_file(
        selected,
        label="CH market-time regime contract",
        max_bytes=_MAX_CONTRACT_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != DOCUMENT_SHA256:
        raise ChMarketTimeRegimeError("market-time contract SHA-256 mismatch")
    return dict(_strict_mapping(payload, label="market-time contract")), payload


def assess_market_time_regime(
    document: Mapping[str, object],
    *,
    document_bytes: bytes,
    repo_root: str | Path | None = None,
    verify_evidence_files: bool = False,
) -> MarketTimeRegimeAssessment:
    if hashlib.sha256(document_bytes).hexdigest() != DOCUMENT_SHA256:
        raise ChMarketTimeRegimeError(
            "captured contract bytes do not match frozen canonical identity"
        )
    captured = _strict_mapping(document_bytes, label="captured market-time contract")
    if not _strict_equal(captured, document):
        raise ChMarketTimeRegimeError("contract mapping differs from captured bytes")

    _exact_keys(document, _TOP_LEVEL_KEYS, label="contract")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("contract_name"), "ch_market_time_regime_20260730_v1", "name")
    contract_id = _sha256(document.get("contract_id"), label="contract_id")
    if contract_id != CONTRACT_ID or contract_id != compute_contract_id(document):
        raise ChMarketTimeRegimeError("contract_id does not bind canonical content")

    _verify_as_of(document.get("as_of"))
    _verify_scope(document.get("scope"))
    _verify_regimes(document.get("regimes"))
    blockers = _verify_transition_gate(document.get("transition_admission_gate"))
    _verify_data_policy(document.get("data_and_validation_policy"))
    _verify_evidence_policy(document.get("evidence"))
    _verify_supersession(document.get("supersession"))
    _verify_execution_controls(document.get("execution_controls"))

    evidence_verified = False
    if verify_evidence_files:
        if repo_root is None:
            raise ChMarketTimeRegimeError(
                "repo_root is required when evidence byte verification is requested"
            )
        verify_bound_evidence_files(document, repo_root=repo_root)
        evidence_verified = True

    return MarketTimeRegimeAssessment(
        contract_id=contract_id,
        document_sha256=DOCUMENT_SHA256,
        status=STATUS,
        evidence_files_verified=evidence_verified,
        blockers=tuple(blockers),
    )


def verify_bound_evidence_files(
    document: Mapping[str, object], *, repo_root: str | Path
) -> None:
    """Re-read every locally bound byte object and verify its size and SHA-256."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    evidence = _mapping(document.get("evidence"), label="evidence")
    sources = _sequence(
        evidence.get("official_source_captures"), label="official source captures"
    )
    bindings: list[tuple[str, int, str]] = []
    for raw in sources:
        item = _mapping(raw, label="official source capture")
        bindings.append(
            (
                _text(item.get("path"), label="official source path"),
                _positive_int(item.get("size_bytes"), label="official source size"),
                _sha256(item.get("sha256"), label="official source SHA-256"),
            )
        )

    hourly = _mapping(evidence.get("local_hourly_observation"), label="hourly evidence")
    for prefix in (
        "selection_manifest",
        "selected_ledger",
        "selected_execution_receipt",
    ):
        bindings.append(
            (
                _text(hourly.get(f"{prefix}_path"), label=f"{prefix} path"),
                _positive_int(
                    hourly.get(f"{prefix}_size_bytes"), label=f"{prefix} size"
                ),
                _sha256(
                    hourly.get(f"{prefix}_sha256"), label=f"{prefix} SHA-256"
                ),
            )
        )

    supersession = _mapping(document.get("supersession"), label="supersession")
    superseded = _mapping(
        supersession.get("superseded_document"), label="superseded document"
    )
    bindings.append(
        (
            _text(superseded.get("path"), label="superseded path"),
            _positive_int(superseded.get("size_bytes"), label="superseded size"),
            _sha256(superseded.get("sha256"), label="superseded SHA-256"),
        )
    )

    for relative, expected_size, expected_sha256 in bindings:
        candidate = _safe_repo_relative_path(root, relative)
        payload = read_stable_single_link_file(
            candidate,
            label=f"market-time evidence {relative}",
            max_bytes=_MAX_EVIDENCE_BYTES,
        )
        if len(payload) != expected_size:
            raise ChMarketTimeRegimeError(f"evidence size mismatch: {relative}")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ChMarketTimeRegimeError(f"evidence SHA-256 mismatch: {relative}")


def _verify_as_of(value: object) -> None:
    _equal(
        dict(_mapping(value, label="as_of")),
        {
            "local_workstation_clock": "2026-07-30",
            "timezone": "Europe/Zurich",
            "time_authority": "UNTRUSTED_LOCAL_CLOCK_NOT_PRODUCTION_EVIDENCE",
        },
        "as_of",
    )


def _verify_scope(value: object) -> None:
    scope = _mapping(value, label="scope")
    _equal(scope.get("market"), "CH", "market")
    _equal(scope.get("products"), ["DAY_AHEAD_AUCTION", "INTRADAY_AUCTIONS"], "products")
    _equal(scope.get("delivery_curve_grid_minutes"), 15, "valuation grid")
    _equal(
        scope.get("delivery_curve_grid_role"),
        "MODELLED_VALUATION_GRID_NEVER_NATIVE_MARKET_TRUTH_CLAIM",
        "valuation-grid role",
    )
    _equal(
        scope.get("monthly_level_authority"),
        "SOLVER_WITH_HARD_GOVERNED_CH_EEX_FORWARD_CONSTRAINTS",
        "monthly authority",
    )
    _equal(
        scope.get("ompex_role"),
        "BENCHMARK_ONLY_NEVER_MODEL_INPUT_TRUTH_OR_LEVEL_AUTHORITY",
        "OMPEX role",
    )


def _verify_regimes(value: object) -> None:
    regimes = _mapping(value, label="regimes")
    _exact_keys(
        regimes, {"pre_transition", "planned_transition", "post_transition"}, label="regimes"
    )
    pre = _mapping(regimes.get("pre_transition"), label="pre-transition regime")
    _equal(
        pre.get("state"),
        "ACTIVE_FAIL_CLOSED_DEFAULT_UNTIL_POST_TRANSITION_ADMISSION",
        "pre-transition state",
    )
    _equal(pre.get("native_day_ahead_market_time_unit_minutes"), 60, "pre-transition MTU")
    _equal(pre.get("native_intraday_auction_market_time_unit_minutes"), 60, "pre-transition IDA MTU")
    _equal(pre.get("native_day_ahead_hourly_truth_claim_possible_after_external_admission"), True, "day-ahead hourly eligibility")
    _equal(pre.get("native_intraday_auction_hourly_truth_claim_possible_after_external_admission"), True, "intraday-auction hourly eligibility")
    _equal(pre.get("native_quarter_hour_truth_claim_possible"), False, "quarter-hour eligibility")
    _equal(pre.get("hourly_to_quarter_hour_stepwise_values_role"), "TECHNICAL_PROXY_ONLY", "proxy role")
    _equal(
        pre.get("hourly_duplication_interpolation_or_smoothing_as_native_quarter_hour_truth"),
        "FORBIDDEN",
        "proxy truth prohibition",
    )

    planned = _mapping(regimes.get("planned_transition"), label="planned transition")
    _equal(planned.get("state"), "PLANNED_NOT_CONFIRMED_NOT_ADMITTED", "planned state")
    latest = _mapping(
        planned.get("latest_locally_observed_official_plan"),
        label="latest locally observed official plan",
    )
    _equal(latest.get("operator"), "EPEX_SPOT", "latest operator")
    _equal(latest.get("planned_first_trading_date"), "2026-11-03", "planned trading date")
    _equal(latest.get("planned_first_delivery_start_utc"), None, "planned delivery boundary")
    _equal(latest.get("wording_status"), "PLANNED_NOT_ACTUAL_GO_LIVE", "planning status")
    _equal(planned.get("confirmed_go_live"), False, "go-live confirmation")
    _equal(planned.get("effective_first_delivery_start_utc"), None, "effective boundary")
    _equal(
        planned.get("planning_change_interpretation"),
        "SCHEDULE_EVOLUTION_NOT_GO_LIVE_EVIDENCE",
        "planning interpretation",
    )
    earlier = _sequence(planned.get("earlier_official_plans"), label="earlier plans")
    if len(earlier) != 2:
        raise ChMarketTimeRegimeError("earlier official plan count is invalid")

    post = _mapping(regimes.get("post_transition"), label="post-transition regime")
    _equal(post.get("state"), "NOT_ADMITTED", "post-transition state")
    _equal(post.get("native_day_ahead_market_time_unit_minutes_after_admission"), 15, "post-transition DA MTU")
    _equal(post.get("native_intraday_auction_market_time_unit_minutes_after_admission"), 15, "post-transition IDA MTU")
    _equal(post.get("day_ahead_cross_product_matching_order_durations_minutes"), [15, 30, 60], "day-ahead cross-product durations")
    _equal(post.get("intraday_auction_cross_product_matching"), "NONE", "intraday-auction cross-product matching")
    _equal(post.get("cross_border_capacity_product_granularity_is_energy_price_truth"), False, "capacity-product truth")
    _equal(
        post.get("published_hourly_index_rule_after_admission"),
        "STRICT_ARITHMETIC_AVERAGE_OF_FOUR_CORRESPONDING_15_MIN_CLEARING_PRICES",
        "post-transition hourly index",
    )


def _verify_transition_gate(value: object) -> list[str]:
    gate = _mapping(value, label="transition gate")
    _equal(gate.get("all_required"), True, "all-required gate")
    _equal(gate.get("transition_admitted"), False, "transition admission")
    _equal(
        gate.get("missing_or_conflicting_evidence_policy"),
        "KEEP_PRE_TRANSITION_HOURLY_REGIME_AND_MARK_15_MIN_TRUTH_UNSUPPORTED",
        "missing-evidence policy",
    )
    requirements = _sequence(gate.get("requirements"), label="transition requirements")
    expected = {
        "EPEX_SPOT_ACTUAL_GO_LIVE_NOTICE_CAPTURED_WITH_EXACT_BYTES_AND_TRUSTED_PUBLICATION_TIME",
        "SWISSGRID_ACTUAL_GO_LIVE_NOTICE_CAPTURED_WITH_EXACT_BYTES_AND_TRUSTED_PUBLICATION_TIME",
        "EXACT_FIRST_TRADING_DATE_AND_FIRST_DELIVERY_INTERVAL_UTC_AGREE_ACROSS_OFFICIAL_SOURCES",
        "LICENSED_NATIVE_CH_15_MIN_CLEARING_PRICE_BYTES_WITH_AUCTION_SESSION_AND_PRODUCT_IDENTITIES",
        "PROVIDER_PRODUCT_SEMANTICS_SCHEMA_REVISION_AND_AVAILABILITY_LAG_MANIFESTS",
        "INDEPENDENT_SIGNATURE_TRUSTED_TIME_AND_EXTERNAL_CAS_RECEIPTS",
        "DST_92_96_100_QUARTER_HOUR_INTERVAL_TESTS_AND_OPERATIONAL_REPLAY",
        "NOTICE_PUBLICATION_AND_REVISION_VINTAGE_MANIFESTS_AVAILABLE_AT_EACH_FORECAST_ORIGIN",
    }
    observed: list[str] = []
    for raw in requirements:
        item = _mapping(raw, label="transition requirement")
        _exact_keys(item, {"requirement", "status"}, label="transition requirement")
        requirement = _text(item.get("requirement"), label="requirement")
        _equal(item.get("status"), "MISSING", f"{requirement} status")
        observed.append(requirement)
    if set(observed) != expected or len(observed) != len(expected):
        raise ChMarketTimeRegimeError("transition requirement inventory is invalid")
    return observed


def _verify_data_policy(value: object) -> None:
    policy = _mapping(value, label="data policy")
    _equal(policy.get("timestamp_axis"), "UTC_WITH_EUROPE_ZURICH_LOCAL_DELIVERY_CALENDAR", "timestamp axis")
    _equal(policy.get("negative_and_zero_prices_retained"), True, "negative-price policy")
    _equal(policy.get("pre_and_post_transition_training_scoring_and_diagnostics_stratified"), True, "regime stratification")
    _equal(policy.get("resampling_across_transition_boundary"), "FORBIDDEN", "boundary resampling")
    _equal(policy.get("native_resolution_provenance_required_per_observation"), True, "resolution provenance")
    _equal(policy.get("proxy_rows_may_train_or_score_native_quarter_hour_layer"), False, "proxy training")
    _equal(policy.get("dst_local_day_expected_quarter_hour_counts"), [92, 96, 100], "DST counts")
    _equal(policy.get("market_regime_boundary_must_be_frozen_before_model_selection"), True, "boundary freeze")
    _equal(policy.get("post_boundary_change_requires_new_candidate_and_backtest_design"), True, "candidate reset")
    _equal(policy.get("sixty_minute_index_reconstruction_before_transition"), "NOT_APPLICABLE_NATIVE_HOURLY_PRODUCT", "pre-transition hourly index")
    _equal(policy.get("sixty_minute_index_reconstruction_after_transition"), "FOUR_NATIVE_QUARTER_HOUR_CLEARING_PRICES_ONLY", "post-transition hourly index")
    _equal(policy.get("regime_metadata_vintage_manifest_required_per_origin"), True, "per-origin regime vintage")
    _equal(policy.get("notice_publication_and_revision_inventory_required"), True, "notice revision inventory")
    _equal(policy.get("regime_metadata_available_at_utc_must_be_le_forecast_origin_utc"), True, "regime PIT ordering")
    _equal(policy.get("delivery_interval_regime_assignment_uses_frozen_effective_boundary_utc"), True, "delivery regime assignment")
    _equal(policy.get("post_origin_boundary_revision_role"), "SCORING_METADATA_ONLY_NEVER_MODEL_INPUT", "post-origin revision role")
    _equal(policy.get("boundary_unknown_at_origin_policy"), "PRE_TRANSITION_INFORMATION_SET_AND_UNSUPPORTED_POST_TRANSITION_NATIVE_TRUTH_CLAIM", "unknown-boundary policy")


def _verify_evidence_policy(value: object) -> None:
    evidence = _mapping(value, label="evidence")
    sources = _sequence(evidence.get("official_source_captures"), label="source captures")
    if len(sources) != 3:
        raise ChMarketTimeRegimeError("official source capture count is invalid")
    expected_ids = {
        "EPEX_SPOT_15_MINUTE_PRODUCTS_IN_MARKET_COUPLING",
        "SWISSGRID_BALANCING_ROADMAP_2026_2030",
        "SWISSGRID_BGM_PARTNER_MEETING_2025",
    }
    observed_ids = set()
    for raw in sources:
        item = _mapping(raw, label="official source capture")
        observed_ids.add(_text(item.get("source_id"), label="source_id"))
        _text(item.get("url"), label="source URL")
        _text(item.get("path"), label="source path")
        _positive_int(item.get("size_bytes"), label="source size")
        _sha256(item.get("sha256"), label="source SHA-256")
        if not _sequence(item.get("supports"), label="source supports"):
            raise ChMarketTimeRegimeError("source support inventory must not be empty")
    _equal(observed_ids, expected_ids, "official source identities")

    hourly = _mapping(evidence.get("local_hourly_observation"), label="hourly observation")
    _equal(hourly.get("observed_native_cadence_minutes"), 60, "observed cadence")
    _equal(hourly.get("observed_through_local_date"), "2026-07-29", "observation end")
    _equal(hourly.get("quarter_hour_rows_are_stepwise_proxies"), True, "stepwise proxy label")
    _equal(hourly.get("scientific_or_production_authority"), False, "hourly authority")
    limitations = _mapping(evidence.get("capture_limitations"), label="capture limitations")
    _equal(limitations.get("capture_time_authority"), "UNTRUSTED_LOCAL_WORKSTATION_CLOCK", "capture time")
    _equal(limitations.get("tls_revocation_check"), "BEST_EFFORT_UNAVAILABLE", "TLS revocation")
    for field in ("independent_signature", "external_cas_receipt", "production_authority"):
        _equal(limitations.get(field), False, field)


def _verify_execution_controls(value: object) -> None:
    controls = _mapping(value, label="execution controls")
    expected_keys = {
        "contract_authorizes_data_acquisition",
        "contract_authorizes_model_selection",
        "contract_authorizes_holdout_consumption",
        "contract_authorizes_production",
        "promotion_gate",
        "t057_consumed",
        "future_holdout_consumed",
    }
    _exact_keys(controls, expected_keys, label="execution controls")
    if any(item is not False for item in controls.values()):
        raise ChMarketTimeRegimeError("market-time contract cannot grant authority")


def _verify_supersession(value: object) -> None:
    supersession = _mapping(value, label="supersession")
    _equal(
        supersession.get("supersession_scope"),
        "MARKET_TIME_REGIME_STATUS_PLANNING_DATE_AND_CAPTURE_BINDINGS_ONLY",
        "supersession scope",
    )
    old = _mapping(supersession.get("superseded_document"), label="superseded document")
    _equal(
        old.get("path"),
        ".planning/phases/14-lt-audit-remediation/"
        "CH-LT-PROSPECTIVE-DATA-ACQUISITION-PLAN-20260727.json",
        "superseded path",
    )
    _equal(
        old.get("sha256"),
        "99361b488e5a031f66c64866c7d69f21a7d8d031e30cf37b097a452bb5627c5a",
        "superseded SHA-256",
    )
    _equal(old.get("size_bytes"), 9953, "superseded size")
    _equal(
        old.get("plan_id"),
        "a0faf97bc10b4d51e23b21c200a73cdd95f92ffa9f8ea7976bc4a10859a8688f",
        "superseded plan_id",
    )
    _equal(
        supersession.get("superseded_assertions"),
        [
            "CH_EXPLICIT_AUCTION_15MIN_GO_LIVE_MONITOR.PLANNED_GO_LIVE_WINDOW_Q3_2026",
            "PRIMARY_SOURCE_OBSERVATIONS.CONTENT_BYTES_CAPTURED_FALSE",
        ],
        "superseded assertions",
    )
    _equal(
        supersession.get("preserved_invariants"),
        [
            "PRE_TRANSITION_NATIVE_CH_DAY_AHEAD_TRUTH_IS_HOURLY",
            "NATIVE_CH_QUARTER_HOUR_TRUTH_UNSUPPORTED_UNTIL_EXTERNAL_ADMISSION",
            "HOURLY_DUPLICATION_AS_NATIVE_QUARTER_HOUR_TRUTH_FORBIDDEN",
            "MONTHLY_SOLVER_REMAINS_LEVEL_AUTHORITY",
            "OMPEX_REMAINS_BENCHMARK_ONLY",
            "NO_EXECUTION_SCIENTIFIC_PRODUCTION_OR_PROMOTION_AUTHORITY",
        ],
        "preserved invariants",
    )
    _equal(
        supersession.get("superseded_document_may_override_this_contract"),
        False,
        "superseded override",
    )


def _safe_repo_relative_path(root: Path, relative: str) -> Path:
    lexical = Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts:
        raise ChMarketTimeRegimeError("evidence path must be repository-relative")
    candidate = root.joinpath(*lexical.parts)
    resolved_parent = candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise ChMarketTimeRegimeError("evidence path escapes repository root") from exc
    return candidate


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChMarketTimeRegimeError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChMarketTimeRegimeError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChMarketTimeRegimeError(f"{label} must be an array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChMarketTimeRegimeError(f"{label} must be non-empty text")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ChMarketTimeRegimeError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChMarketTimeRegimeError(f"{label} must be lowercase SHA-256")
    return value


def _require_sha256(value: str, *, label: str) -> None:
    _sha256(value, label=label)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChMarketTimeRegimeError(f"{label} fields are invalid")


def _equal(actual: object, expected: object, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ChMarketTimeRegimeError(f"{label} is invalid")


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))
