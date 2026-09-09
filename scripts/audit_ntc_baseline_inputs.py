"""Audit whether local ENTSO-E NTC columns can support a proxy baseline.

The script intentionally does not impute NTC values. If the local NTC columns
are empty, it writes a blocking report and exits successfully so the readiness
workflow can document the gap without creating false data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_ENTSO = Path("pfc_shaping/data/entso_15min.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/NTC-BASELINE-AUDIT.md")

BORDERS = ["de", "fr", "it", "at"]
KINDS = ["import", "export", "net", "total"]


def audit_ntc_inputs(entso: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for border in BORDERS:
        for kind in KINDS:
            col = f"ntc_{kind}_ch_{border}_mw"
            if col not in entso.columns:
                rows.append(
                    {
                        "border": border.upper(),
                        "kind": kind,
                        "column": col,
                        "available": False,
                        "non_null_rows": 0,
                        "first_ts": "",
                        "last_ts": "",
                        "median_mw": "",
                    }
                )
                continue
            series = pd.to_numeric(entso[col], errors="coerce").dropna()
            rows.append(
                {
                    "border": border.upper(),
                    "kind": kind,
                    "column": col,
                    "available": bool(len(series)),
                    "non_null_rows": int(len(series)),
                    "first_ts": str(series.index.min()) if len(series) else "",
                    "last_ts": str(series.index.max()) if len(series) else "",
                    "median_mw": float(series.median()) if len(series) else "",
                }
            )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(path: Path, audit: pd.DataFrame, *, entso_path: Path) -> None:
    usable = bool(audit["available"].all()) if len(audit) else False
    lines = [
        "# NTC Baseline Input Audit",
        "",
        f"* source: `{entso_path}`",
        f"* usable for proxy baseline: `{'YES' if usable else 'NO'}`",
        "* rule: no NTC baseline is produced unless local observations exist",
        "",
        "## Column Coverage",
        "",
        _md_table(audit),
        "",
        "## Decision",
        "",
        "The local ENTSO-E cache cannot currently support an NTC proxy baseline if any required CH border NTC column has zero observations. Final production still requires governed Swissgrid/JAO/TYNDP NTC assumptions.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entso", default=str(DEFAULT_ENTSO))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)
    entso_path = Path(args.entso)
    entso = pd.read_parquet(entso_path)
    audit = audit_ntc_inputs(entso)
    write_report(Path(args.report), audit, entso_path=entso_path)
    usable = bool(audit["available"].all()) if len(audit) else False
    print(f"[ntc-audit] usable={'YES' if usable else 'NO'} -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
