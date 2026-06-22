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
    active_config_hash: str,
    selected_config_hash: str,
) -> pd.DataFrame:
    """Return strict governance gates backed by manifest-derived hashes."""

    return build_monthly_curve_governance_gates(
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


def _active_config_hash(manifest: Mapping[str, object]) -> str:
    explicit = manifest.get("active_config_hash")
    if explicit:
        return str(explicit)
    solver_config = manifest.get("solver_config") or manifest.get("config")
    if not isinstance(solver_config, Mapping):
        raise ValueError("production manifest missing active_config_hash or solver_config/config")
    monthly_config = monthly_curve_config_from_settings(solver_config)
    payload = dict(monthly_config.__dict__)
    payload["history_lookback_years"] = solver_config.get("history_lookback_years")
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
