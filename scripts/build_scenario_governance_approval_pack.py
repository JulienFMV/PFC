"""Build a scenario-governance approval pack for Phase 13 LT.

The pack is an audit artifact, not an approval mechanism. It consolidates the
manifest, governance validation, data gap register and optional expert reviews
so a risk/quant committee can make a traceable decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_MANIFEST = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml")
DEFAULT_GOVERNANCE_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-VALIDATION.md")
DEFAULT_GAP_REGISTER = Path("data/lt_scenario_governance_gap_register.csv")
DEFAULT_EXPERT_REVIEWS = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-EXPERT-REVIEWS.md")
DEFAULT_OUTPUT = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-APPROVAL-PACK.md")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_optional(path: Path) -> str:
    if not path.exists():
        return "_not available_"
    text = path.read_text(encoding="utf-8").strip()
    return text if text else "_empty_"


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _decision_status(manifest: dict[str, Any], gaps: pd.DataFrame) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if manifest.get("status") != "approved":
        blockers.append("manifest status is not approved")
    if manifest.get("approved_for_production") is not True:
        blockers.append("approved_for_production is not true")
    if not manifest.get("approval_date"):
        blockers.append("approval_date is missing")
    if not manifest.get("approved_by"):
        blockers.append("approved_by is empty")
    if len(gaps):
        blockers.append(f"{len(gaps)} data/governance blockers remain in the gap register")
    return ("GO" if not blockers else "NO-GO", blockers)


def build_pack(
    *,
    manifest_path: Path,
    governance_report_path: Path,
    gap_register_path: Path,
    expert_reviews_path: Path,
    output_path: Path,
) -> str:
    manifest = _load_yaml(manifest_path)
    gaps = pd.read_csv(gap_register_path) if gap_register_path.exists() else pd.DataFrame()
    status, blockers = _decision_status(manifest, gaps)

    weights = manifest.get("scenario_weights") or {}
    weights_df = pd.DataFrame(
        [{"scenario": scenario, "weight": weight} for scenario, weight in weights.items()],
        columns=["scenario", "weight"],
    )
    source_df = pd.DataFrame(manifest.get("source_components") or [])
    if not gaps.empty:
        summary = (
            gaps.groupby(["priority", "family"], dropna=False)
            .size()
            .reset_index(name="gap_count")
            .sort_values(["priority", "family"])
        )
        field_actions = (
            gaps.loc[gaps["field"].fillna("").astype(str) != ""]
            .groupby(["priority", "field", "owner", "source_candidate", "required_action"], dropna=False)
            .agg(missing_rows=("missing_rows", "max"))
            .reset_index()
            .sort_values(["priority", "field"])
        )
    else:
        summary = pd.DataFrame(columns=["priority", "family", "gap_count"])
        field_actions = pd.DataFrame(
            columns=["priority", "field", "owner", "source_candidate", "required_action", "missing_rows"]
        )

    lines = [
        "# Scenario Governance Approval Pack",
        "",
        "## Decision Status",
        "",
        f"* decision_id: `{manifest.get('decision_id', '')}`",
        f"* manifest_status: `{manifest.get('status', '')}`",
        f"* approved_for_production: `{manifest.get('approved_for_production', '')}`",
        f"* approval_date: `{manifest.get('approval_date', '')}`",
        f"* approved_by: `{manifest.get('approved_by', [])}`",
        f"* pack_recommendation: `{status}`",
        "",
        "### Blocking Conditions",
        "",
        _md_table(pd.DataFrame({"blocker": blockers}) if blockers else pd.DataFrame(columns=["blocker"])),
        "",
        "## Scenario Scope",
        "",
        "```yaml",
        yaml.safe_dump(manifest.get("scope") or {}, sort_keys=False).strip(),
        "```",
        "",
        "## Scenario Weights",
        "",
        _md_table(weights_df),
        "",
        "## Source Components",
        "",
        _md_table(source_df),
        "",
        "## Gap Summary",
        "",
        _md_table(summary),
        "",
        "## Field-Level Required Actions",
        "",
        _md_table(field_actions),
        "",
        "## Expert Reviews",
        "",
        _read_optional(expert_reviews_path),
        "",
        "## Governance Validation Report",
        "",
        _read_optional(governance_report_path),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--governance-report", default=str(DEFAULT_GOVERNANCE_REPORT))
    parser.add_argument("--gap-register", default=str(DEFAULT_GAP_REGISTER))
    parser.add_argument("--expert-reviews", default=str(DEFAULT_EXPERT_REVIEWS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    status = build_pack(
        manifest_path=Path(args.manifest),
        governance_report_path=Path(args.governance_report),
        gap_register_path=Path(args.gap_register),
        expert_reviews_path=Path(args.expert_reviews),
        output_path=Path(args.output),
    )
    print(f"[approval-pack] recommendation={status}")
    print(f"[approval-pack] output -> {args.output}")
    return 0 if status == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
