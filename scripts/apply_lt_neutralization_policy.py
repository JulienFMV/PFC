"""Apply explicit governed neutralisations to non-core LT scenario fields.

This script is intentionally conservative: it only fills selected missing P1
fields with zero when the output row also carries a field-level zero
justification and an auditable source component. It never overwrites non-null
source values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    load_electrification_scenarios,
    validate_electrification_scenarios,
    write_hpfc_scenario_features,
)

DEFAULT_INPUT = Path("data/electrification_scenarios_composed_p0_public_sources_2030.parquet")
DEFAULT_OUTPUT = Path("data/electrification_scenarios_prod_candidate_neutralized_2030.parquet")
DEFAULT_FEATURE_OUTPUT = Path("data/hpfc_scenario_features_prod_candidate_neutralized_2030.parquet")
DEFAULT_AUDIT = Path("data/electrification_scenarios_neutralization_audit_2030.csv")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/LT-NEUTRALIZATION-POLICY.md")

SOURCE_ID = "lt_field_neutralization_policy_20260612"
PUBLICATION_DATE = pd.Timestamp("2026-06-12", tz="UTC")

NEUTRALIZED_FIELDS: dict[str, str] = {
    "coal_gw": (
        "Explicit zero neutralisation where no governed 2030 coal capacity source "
        "is wired for this country/scenario. Existing non-null coal source values "
        "are preserved."
    ),
    "dsm_gw": (
        "DSM structural flex layer inactive until a governed DSM scenario source is wired."
    ),
    "electrolysis_twh": (
        "Electrolysis structural demand layer inactive until a governed hydrogen scenario source is wired."
    ),
    "p2x_gw": (
        "P2X structural flex layer inactive until a governed hydrogen/P2X capacity source is wired."
    ),
    "managed_charging_share": (
        "Conservative no-managed-charging assumption until a governed smart-charging share is approved."
    ),
}


def _append_token(value: object, token: str) -> str:
    parts = [part for part in str(value or "").split(",") if part]
    if token not in parts:
        parts.append(token)
    return ",".join(parts)


def _append_quality_suffix(value: object, suffix: str) -> str:
    text = str(value or "")
    return text if suffix in text else f"{text}_{suffix}" if text else suffix


def apply_neutralization_policy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = validate_electrification_scenarios(frame, require_recommended=True).copy()
    if "component_ids" not in df.columns:
        df["component_ids"] = ""
    if "quality_flag" not in df.columns:
        df["quality_flag"] = "neutralized_explicit"
    if "ingested_at_utc" not in df.columns:
        df["ingested_at_utc"] = PUBLICATION_DATE

    audit_rows: list[dict[str, object]] = []
    for field, justification in NEUTRALIZED_FIELDS.items():
        if field not in df.columns:
            df[field] = pd.NA
        justification_col = f"{field}_zero_justification"
        if justification_col not in df.columns:
            df[justification_col] = pd.NA
        missing = df[field].isna()
        if not bool(missing.any()):
            continue
        df.loc[missing, field] = 0.0
        df.loc[missing, justification_col] = justification
        df.loc[missing, "component_ids"] = df.loc[missing, "component_ids"].map(
            lambda value: _append_token(value, SOURCE_ID)
        )
        df.loc[missing, "quality_flag"] = df.loc[missing, "quality_flag"].map(
            lambda value: _append_quality_suffix(value, "neutralized_explicit")
        )
        for _, row in df.loc[missing].iterrows():
            audit_rows.append(
                {
                    "source_id": SOURCE_ID,
                    "publication_date": PUBLICATION_DATE,
                    "country": row.get("country", ""),
                    "scenario": row.get("scenario", ""),
                    "delivery_year": row.get("delivery_year", ""),
                    "field": field,
                    "value": 0.0,
                    "justification": justification,
                }
            )

    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "source_id",
            "publication_date",
            "country",
            "scenario",
            "delivery_year",
            "field",
            "value",
            "justification",
        ],
    )
    return validate_electrification_scenarios(df, require_recommended=True), audit


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


def write_report(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    feature_output: Path,
    audit_path: Path,
    audit: pd.DataFrame,
) -> None:
    summary = (
        audit.groupby("field", dropna=False)
        .size()
        .reset_index(name="neutralized_rows")
        .sort_values("field")
        if not audit.empty
        else pd.DataFrame(columns=["field", "neutralized_rows"])
    )
    lines = [
        "# LT Neutralization Policy",
        "",
        f"* source_id: `{SOURCE_ID}`",
        f"* publication_date: `{PUBLICATION_DATE}`",
        f"* input: `{input_path}`",
        f"* output: `{output_path}`",
        f"* hpfc features: `{feature_output}`",
        f"* audit: `{audit_path}`",
        "",
        "## Scope",
        "",
        (
            "This artefact closes numeric P1 gaps through explicit, auditable zero "
            "neutralisations. It does not approve the scenario inventory for "
            "production and does not remove proxy/partial quality flags."
        ),
        "",
        "## Neutralized Fields",
        "",
        _md_table(summary),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--feature-output", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    feature_output = Path(args.feature_output)
    audit_path = Path(args.audit_output)

    frame = load_electrification_scenarios(input_path, require_recommended=True)
    out, audit = apply_neutralization_policy(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    write_hpfc_scenario_features(out, feature_output)
    write_report(
        Path(args.report),
        input_path=input_path,
        output_path=output_path,
        feature_output=feature_output,
        audit_path=audit_path,
        audit=audit,
    )

    print(f"[neutralization] rows={len(out)} neutralized={len(audit)}")
    print(f"[neutralization] output -> {output_path}")
    print(f"[neutralization] features -> {feature_output}")
    print(f"[neutralization] audit -> {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
