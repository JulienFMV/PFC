"""Build a hash-bound local quote-to-month sensitivity audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
from pathlib import Path

import pandas as pd

from pfc_shaping.calibration.monthly_curve_sensitivity import (
    MonthlySensitivityPolicy,
    audit_monthly_quote_curve_sensitivity,
)
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    solve_monthly_level_authority_from_history,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def build_monthly_sensitivity_artifact(
    *,
    monthly_manifest_path: Path,
    expected_monthly_manifest_sha256: str,
    forwards_path: Path,
    expected_forwards_sha256: str,
    output_dir: Path,
    perturbation_eur_mwh: float = 0.01,
) -> dict[str, object]:
    """Reconstruct the exact monthly solve and publish non-promotional evidence."""

    repo_root = _workspace_root()
    manifest_path = _workspace_path(monthly_manifest_path, repo_root=repo_root)
    history_path = _workspace_path(forwards_path, repo_root=repo_root)
    manifest_payload = read_stable_single_link_file(
        manifest_path,
        label="monthly sensitivity source manifest",
    )
    forwards_payload = read_stable_single_link_file(
        history_path,
        label="monthly sensitivity forwards payload",
    )
    manifest_sha256 = _require_hash(
        manifest_payload,
        expected_monthly_manifest_sha256,
        label="monthly manifest",
    )
    forwards_sha256 = _require_hash(
        forwards_payload,
        expected_forwards_sha256,
        label="forwards payload",
    )
    manifest = _strict_mapping(manifest_payload, label="monthly manifest")
    _validate_manifest_binding(manifest, forwards_sha256=forwards_sha256)

    delivery_months = pd.PeriodIndex(manifest["delivery_months"], freq="M")
    settings = manifest.get("solver_config")
    if not isinstance(settings, dict):
        raise ValueError("monthly manifest solver_config is invalid")
    market = str(manifest.get("market", "")).upper()
    if market != "CH":
        raise ValueError("monthly sensitivity artifact is currently restricted to CH")
    timezone = "Europe/Zurich"
    authority = solve_monthly_level_authority_from_history(
        forwards_path=history_path,
        market=market,
        delivery_months=delivery_months,
        settings=settings,
        timezone=timezone,
        as_of_date=str(manifest["run_timestamp"]),
        forwards_payload=forwards_payload,
    )
    for key in ("active_constraints_hash", "monthly_solution_hash", "active_config_hash"):
        if str(authority.manifest.get(key, "")) != str(manifest.get(key, "")):
            raise ValueError(f"reconstructed monthly authority differs on {key}")

    audit = audit_monthly_quote_curve_sensitivity(
        delivery_months=delivery_months,
        own_quotes=authority.inputs.own_quotes,
        config=authority.inputs.config,
        shape_prior=authority.shape_prior,
        market=market,
        timezone=timezone,
        policy=MonthlySensitivityPolicy(
            perturbation_eur_mwh=float(perturbation_eur_mwh)
        ),
    )
    destination, staging = _fresh_output_staging(
        output_dir,
        repo_root=repo_root,
    )
    csv_payload = audit.sensitivities.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    csv_name = "quote_to_month_sensitivity.csv"
    csv_path = staging / csv_name
    write_durable_exact_bytes(
        csv_path,
        csv_payload,
        label="monthly quote-to-curve sensitivity matrix",
    )
    script_path = Path(__file__).resolve()
    module_path = Path(
        audit_monthly_quote_curve_sensitivity.__code__.co_filename
    ).resolve()
    script_payload = read_stable_single_link_file(
        script_path,
        label="monthly sensitivity artifact script",
    )
    module_payload = read_stable_single_link_file(
        module_path,
        label="monthly sensitivity audit module",
    )
    summary: dict[str, object] = {
        "schema_version": "monthly_quote_curve_sensitivity_artifact.v3",
        "status": audit.summary["status"],
        "production_authorization": False,
        "promotion_gate": False,
        "inputs": {
            "monthly_manifest_sha256": manifest_sha256,
            "forwards_payload_sha256": forwards_sha256,
            "monthly_solution_hash": manifest["monthly_solution_hash"],
            "active_constraints_hash": manifest["active_constraints_hash"],
            "active_config_hash": manifest["active_config_hash"],
            "forward_source_kind": manifest.get("forward_source_kind"),
            "hard_quote_eligible": bool(manifest.get("hard_quote_eligible", False)),
            "promotion_eligible": bool(manifest.get("promotion_eligible", False)),
        },
        "code": {
            "observed_script_source_at_report_time_sha256": _sha256(script_payload),
            "observed_module_source_at_report_time_sha256": _sha256(module_payload),
            "execution_attestation": "NOT_PROVIDED_POST_IMPORT_OBSERVATION_ONLY",
        },
        "audit": audit.summary,
        "artifacts": {
            csv_name: {
                "sha256": _sha256(csv_payload),
                "size_bytes": len(csv_payload),
            }
        },
        "authority_boundary": (
            "LOCAL_HASH_BOUND_DIAGNOSTIC_ONLY_REQUIRES_GOVERNED_FRESH_CH_INPUTS"
        ),
    }
    summary_payload = _canonical_json(summary) + b"\n"
    write_durable_exact_bytes(
        staging / "summary.json",
        summary_payload,
        label="monthly quote-to-curve sensitivity summary",
    )
    _publish_staged_directory(staging, destination)
    return summary


def _validate_manifest_binding(
    manifest: dict[str, object],
    *,
    forwards_sha256: str,
) -> None:
    required = {
        "active_config_hash",
        "active_constraints_hash",
        "delivery_months",
        "market",
        "monthly_solution_hash",
        "run_timestamp",
        "solver_config",
        "source_hashes",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"monthly manifest is missing required fields: {missing}")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("monthly manifest source_hashes is invalid")
    if source_hashes.get("forwards_path") != forwards_sha256:
        raise ValueError("monthly manifest is not bound to the supplied forwards bytes")
    if manifest.get("monthly_level_authority") != "solver":
        raise ValueError("monthly manifest level authority is not the solver")
    if not bool(manifest.get("skip_legacy_level_cascade", False)):
        raise ValueError("monthly manifest does not disable the legacy level cascade")
    if not bool(manifest.get("skip_legacy_base_smoothing", False)):
        raise ValueError("monthly manifest does not disable legacy base smoothing")
    settings = manifest.get("solver_config")
    if not isinstance(settings, dict):
        raise ValueError("monthly manifest solver_config is invalid")
    if float(settings.get("constraint_tolerance", float("inf"))) != 1e-9:
        raise ValueError("monthly manifest hard repricing tolerance must be exactly 1e-9")


def _workspace_root() -> Path:
    selected = assert_absolute_path_has_no_links(
        Path(os.path.abspath(Path.cwd()))
    )
    if not (selected / ".git").exists():
        raise ValueError("monthly sensitivity artifact must run from the workspace root")
    return selected


def _workspace_path(path: Path, *, repo_root: Path) -> Path:
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        selected.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("monthly sensitivity path escapes the workspace") from exc
    return assert_absolute_path_has_no_links(selected)


def _fresh_output_staging(
    path: Path,
    *,
    repo_root: Path,
) -> tuple[Path, Path]:
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        selected.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("monthly sensitivity output escapes the workspace") from exc
    if selected.exists():
        raise FileExistsError("monthly sensitivity output directory already exists")
    parent = assert_absolute_path_has_no_links(selected.parent)
    if not parent.is_dir():
        raise ValueError("monthly sensitivity output parent is unavailable")
    staging = parent / (
        f".monthly-sensitivity-{selected.name}-staging-"
        f"{secrets.token_hex(16)}"
    )
    # tempfile.mkdtemp deliberately creates a private directory. On Windows
    # that restrictive ACL survives directory rename and can make the final
    # artifact unreadable to independent consumers. A default-mode mkdir
    # inherits the already-governed parent ACL while retaining atomic,
    # unguessable and fail-if-present staging creation.
    staging.mkdir(mode=0o777, exist_ok=False)
    return selected, assert_absolute_path_has_no_links(staging)


def _publish_staged_directory(staging: Path, destination: Path) -> None:
    selected_staging = assert_absolute_path_has_no_links(staging)
    selected_parent = assert_absolute_path_has_no_links(destination.parent)
    if selected_staging.parent != selected_parent:
        raise ValueError("monthly sensitivity staging directory changed parent")
    if destination.exists():
        raise FileExistsError("monthly sensitivity output directory already exists")
    before = selected_staging.stat(follow_symlinks=False)
    os.rename(selected_staging, destination)
    published = assert_absolute_path_has_no_links(destination)
    after = published.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError("monthly sensitivity published directory identity changed")
    _fsync_directory(selected_parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Python exposes no portable Windows directory flush. Exact-volume
        # power-loss qualification remains an external production gate.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_hash(payload: bytes, expected: str, *, label: str) -> str:
    normalized = str(expected).lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"expected {label} SHA-256 is invalid")
    actual = _sha256(payload)
    if actual != normalized:
        raise ValueError(f"{label} differs from the caller-held SHA-256")
    return actual


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant in {label}: {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-manifest", type=Path, required=True)
    parser.add_argument("--expected-monthly-manifest-sha256", required=True)
    parser.add_argument("--forwards", type=Path, required=True)
    parser.add_argument("--expected-forwards-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturbation-eur-mwh", type=float, default=0.01)
    args = parser.parse_args(argv)
    summary = build_monthly_sensitivity_artifact(
        monthly_manifest_path=args.monthly_manifest,
        expected_monthly_manifest_sha256=args.expected_monthly_manifest_sha256,
        forwards_path=args.forwards,
        expected_forwards_sha256=args.expected_forwards_sha256,
        output_dir=args.output_dir,
        perturbation_eur_mwh=args.perturbation_eur_mwh,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
