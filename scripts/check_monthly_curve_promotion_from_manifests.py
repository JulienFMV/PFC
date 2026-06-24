"""Check monthly curve promotion using real prod/export manifests.

This is the promotion capstone for the monthly BASE solver.  Unlike
``check_monthly_curve_promotion.py``, it does not accept raw parity hashes on
the command line.  It reads them from the production/export monthly solver
manifests and reads the selected lambda hash from the calibration artifact,
then appends the required governance gates before evaluating promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.calibration.monthly_curve_audit import build_monthly_curve_governance_gates
from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.calibration.monthly_curve_promotion import evaluate_monthly_curve_promotion
from pfc_shaping.pipeline.monthly_curve_authority import monthly_curve_config_from_settings


REQUIRED_GOVERNANCE_GATES = {
    "lambda_calibration_artifact_present",
    "production_export_path_parity",
    "selected_config_manifest_parity",
    "selected_config_production_approval",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    audit_gates = pd.read_csv(args.audit_gates)
    historical_thresholds = pd.read_csv(args.historical_thresholds)
    production_manifest = _load_mapping(args.production_manifest)
    export_manifest = _load_mapping(args.export_manifest)
    selected_config = _load_mapping(args.selected_config_artifact)
    selected_hash = _selected_config_hash(selected_config, args.selected_config_artifact)
    active_hash = _active_config_hash(production_manifest)

    governance = build_manifest_governance_gates(
        run_timestamp=pd.Timestamp(args.run_timestamp) if args.run_timestamp else None,
        production_manifest=production_manifest,
        export_manifest=export_manifest,
        selected_config=selected_config,
        active_config_hash=active_hash,
        selected_config_hash=selected_hash,
    )
    augmented = _replace_gate_rows(audit_gates, governance)
    manifest = _promotion_manifest(
        audit_manifest=_load_mapping(args.manifest) if args.manifest is not None and args.manifest.exists() else {},
        production_manifest=production_manifest,
        export_manifest=export_manifest,
        selected_config=selected_config,
        governance_gates=governance,
    )

    decision = evaluate_monthly_curve_promotion(
        augmented,
        historical_thresholds,
        run_timestamp=pd.Timestamp(args.run_timestamp) if args.run_timestamp else None,
        far_horizon_min_years=int(args.far_horizon_min_years),
        required_governance_gates=REQUIRED_GOVERNANCE_GATES,
        manifest=manifest,
    )

    if args.augmented_audit_gates is not None:
        args.augmented_audit_gates.parent.mkdir(parents=True, exist_ok=True)
        augmented.to_csv(args.augmented_audit_gates, index=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(decision.summary, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        decision.details.to_csv(args.details_output, index=False)

    print(json.dumps(decision.summary, sort_keys=True, default=str))
    return 0 if decision.approved else 1


def build_manifest_governance_gates(
    *,
    run_timestamp: pd.Timestamp | None,
    production_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    selected_config: Mapping[str, object],
    active_config_hash: str,
    selected_config_hash: str,
) -> pd.DataFrame:
    """Return strict governance gates backed by manifest-derived hashes."""

    base = build_monthly_curve_governance_gates(
        run_timestamp=pd.Timestamp(run_timestamp) if run_timestamp is not None else pd.Timestamp.utcnow(),
        active_config_hash=active_config_hash,
        selected_config_hash=selected_config_hash,
        production_monthly_solution_hash=_required_manifest_hash(
            production_manifest,
            "monthly_solution_hash",
            role="production",
        ),
        export_monthly_solution_hash=_required_manifest_hash(
            export_manifest,
            "monthly_solution_hash",
            role="export",
        ),
        production_active_constraints_hash=_required_manifest_hash(
            production_manifest,
            "active_constraints_hash",
            role="production",
        ),
        export_active_constraints_hash=_required_manifest_hash(
            export_manifest,
            "active_constraints_hash",
            role="export",
        ),
        require_lambda_artifact=True,
        require_path_parity=True,
    ).loc[lambda frame: frame["gate_id"].isin(REQUIRED_GOVERNANCE_GATES)].reset_index(drop=True)
    extra = pd.DataFrame(
        [
            _selected_config_production_approval_row(selected_config),
            _selected_config_manifest_parity_row(
                selected_config=selected_config,
                production_manifest=production_manifest,
                export_manifest=export_manifest,
                active_config_hash=active_config_hash,
                selected_config_hash=selected_config_hash,
            ),
        ]
    )
    return pd.concat([base, extra], ignore_index=True)


def _replace_gate_rows(audit_gates: pd.DataFrame, governance: pd.DataFrame) -> pd.DataFrame:
    base = audit_gates[~audit_gates["gate_id"].astype(str).isin(REQUIRED_GOVERNANCE_GATES)]
    return pd.concat([base, governance], ignore_index=True)


def _load_mapping(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raise ValueError(f"unsupported structured file type for {path}")


def _selected_config_hash(config: Mapping[str, object], path: Path) -> str:
    value = config.get("config_hash") or config.get("selected_config_hash")
    if not value:
        raise ValueError(f"selected config artifact missing config_hash or selected_config_hash: {path}")
    return str(value)


def _selected_config_production_approval_row(selected_config: Mapping[str, object]) -> dict[str, object]:
    approved = selected_config.get("production_approved") is True
    promotion_approval = selected_config.get("production_promotion_approved")
    promotion_approved = promotion_approval is True if "production_promotion_approved" in selected_config else True
    selection_status = str(selected_config.get("selection_status", ""))
    status_is_prod = selection_status == "PRODUCTION_APPROVED"
    if approved and promotion_approved and status_is_prod:
        status = "PASS"
        severity = "INFO"
        metric_value = 0.0
        evidence = (
            "selected config is production approved; "
            f"production_approved={selected_config.get('production_approved')!r}, "
            f"production_promotion_approved={promotion_approval!r}, "
            f"selection_status={selection_status}"
        )
    else:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = (
            "selected config is not production-approved; "
            f"production_approved={selected_config.get('production_approved')!r}, "
            f"production_promotion_approved={promotion_approval!r}, "
            f"selection_status={selection_status!r}"
        )
    return _governance_gate_row(
        gate_id="selected_config_production_approval",
        status=status,
        severity=severity,
        product="monthly_curve_selected_config",
        parent_block_id="selected_config_approval",
        metric_name="selected_config_not_production_approved",
        metric_value=metric_value,
        threshold_source="selected_config_artifact",
        evidence=evidence,
        remediation_hint="Use a production-approved selected config artifact before promotion.",
    )


def _selected_config_manifest_parity_row(
    *,
    selected_config: Mapping[str, object],
    production_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    active_config_hash: str,
    selected_config_hash: str,
) -> dict[str, object]:
    missing: list[str] = []
    mismatches: list[str] = []

    expected = {
        "config_hash": active_config_hash,
        "active_config_hash_from_candidate_manifest": active_config_hash,
        "monthly_solution_hash": _required_manifest_hash(
            production_manifest,
            "monthly_solution_hash",
            role="production",
        ),
        "active_constraints_hash": _required_manifest_hash(
            production_manifest,
            "active_constraints_hash",
            role="production",
        ),
    }
    for key, expected_value in expected.items():
        value = selected_config.get(key)
        if not value:
            missing.append(key)
        elif str(value) != str(expected_value):
            mismatches.append(f"{key}: selected={value} expected={expected_value}")

    if str(export_manifest.get("monthly_solution_hash", "")) != str(expected["monthly_solution_hash"]):
        mismatches.append("export monthly_solution_hash does not match production")
    if str(export_manifest.get("active_constraints_hash", "")) != str(expected["active_constraints_hash"]):
        mismatches.append("export active_constraints_hash does not match production")
    if selected_config_hash != active_config_hash:
        mismatches.append(
            f"selected_config_hash={selected_config_hash} active_config_hash={active_config_hash}"
        )
    if str(selected_config.get("schema_version", "")) != "monthly_curve_selected_config.v1":
        mismatches.append("schema_version is not monthly_curve_selected_config.v1")
    if not selected_config.get("candidate_manifest"):
        missing.append("candidate_manifest")

    if missing or mismatches:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = f"missing={missing}; mismatches={mismatches}"
    else:
        status = "PASS"
        severity = "INFO"
        metric_value = 0.0
        evidence = (
            "selected config hashes match production/export active config, "
            "monthly solution and active constraints"
        )
    return _governance_gate_row(
        gate_id="selected_config_manifest_parity",
        status=status,
        severity=severity,
        product="monthly_curve_selected_config",
        parent_block_id="selected_prod_export_triad",
        metric_name="selected_manifest_hash_mismatch",
        metric_value=metric_value,
        threshold_source="selected_config_prod_export_manifest_triad",
        evidence=evidence,
        remediation_hint="Regenerate or approve the selected config artifact from the same production/export manifest triad.",
    )


def _governance_gate_row(
    *,
    gate_id: str,
    status: str,
    severity: str,
    product: str,
    parent_block_id: str,
    metric_name: str,
    metric_value: float,
    threshold_source: str,
    evidence: str,
    remediation_hint: str,
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "status": status,
        "severity": severity,
        "year": 0,
        "month": None,
        "product": product,
        "parent_block_id": parent_block_id,
        "parent_block_type": "governance",
        "parent_hours": float("nan"),
        "parent_mean": float("nan"),
        "month_price": float("nan"),
        "month_deviation": float("nan"),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold_warning": 0.0,
        "threshold_critical": 0.0,
        "threshold_source": threshold_source,
        "n_history": float("nan"),
        "n_neighbors": float("nan"),
        "evidence": evidence,
        "remediation_hint": remediation_hint,
    }


def _active_config_hash(manifest: Mapping[str, object]) -> str:
    explicit = manifest.get("active_config_hash")
    if explicit:
        return str(explicit)
    solver_config = manifest.get("solver_config") or manifest.get("config")
    if not isinstance(solver_config, Mapping):
        raise ValueError("production manifest missing active_config_hash or solver_config/config")
    monthly_config = monthly_curve_config_from_settings(solver_config)
    payload = dict(monthly_config.__dict__)
    payload.update(
        {
            "markets": sorted(str(m).upper() for m in solver_config.get("markets", [])),
            "history_lookback_years": solver_config.get("history_lookback_years"),
            "min_structural_snapshots": solver_config.get("min_structural_snapshots"),
            "allow_template_structural_fallback": bool(
                solver_config.get("allow_template_structural_fallback", False)
            ),
            "structural_amplitude_eur_mwh": float(
                solver_config.get("structural_amplitude_eur_mwh", 110.0)
            ),
            "panel_weight": float(solver_config.get("panel_weight", 1.0)),
            "history_weight": float(solver_config.get("history_weight", 0.5)),
            "structural_weight": float(solver_config.get("structural_weight", 1.0)),
        }
    )
    return config_hash(payload)


def _required_manifest_hash(manifest: Mapping[str, object], key: str, *, role: str) -> str:
    value = manifest.get(key)
    if not value:
        raise ValueError(f"{role} manifest missing required {key}")
    return str(value)


def _promotion_manifest(
    *,
    audit_manifest: Mapping[str, object],
    production_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    selected_config: Mapping[str, object],
    governance_gates: pd.DataFrame,
) -> dict[str, object]:
    manifest = dict(audit_manifest)
    manifest["promotion_evidence_source"] = "prod_export_manifests"
    manifest["production_monthly_solution_hash"] = production_manifest.get("monthly_solution_hash", "")
    manifest["export_monthly_solution_hash"] = export_manifest.get("monthly_solution_hash", "")
    manifest["production_active_constraints_hash"] = production_manifest.get("active_constraints_hash", "")
    manifest["export_active_constraints_hash"] = export_manifest.get("active_constraints_hash", "")
    manifest["active_config_hash"] = _active_config_hash(production_manifest)
    manifest["selected_config_hash"] = _selected_config_hash(selected_config, Path("<selected_config_artifact>"))
    manifest["selected_config_production_approved"] = selected_config.get("production_approved", False)
    manifest["selected_config_selection_status"] = selected_config.get("selection_status", "")
    manifest["selected_config_monthly_solution_hash"] = selected_config.get("monthly_solution_hash", "")
    manifest["selected_config_active_constraints_hash"] = selected_config.get("active_constraints_hash", "")
    manifest["governance_gate_summary"] = governance_gates["status"].value_counts().to_dict()
    return manifest


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-gates", type=Path, required=True)
    parser.add_argument("--historical-thresholds", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--selected-config-artifact", type=Path, required=True)
    parser.add_argument("--run-timestamp", default=None)
    parser.add_argument("--far-horizon-min-years", type=int, default=2)
    parser.add_argument("--augmented-audit-gates", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--details-output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
