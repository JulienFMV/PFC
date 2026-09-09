"""
phase0_sniff_forwards.py
------------------------
Phase 0 du roadmap PFC long-terme — cadrage des sources EEX.

Sniff des deux fichiers maîtres EEX maintenus par le desk FMV :
  - Price_Report_EEX_Yearly.xlsx              (DE, CH, AT, IT, FR — snapshot courant)
  - Price_Report_EEX_CH_DE_Historique2019.xlsx (CH, DE — depuis 2019)

Pour chaque feuille, dump :
  - les 3 premières lignes × 10 premières colonnes (en clair)
  - le format détecté ("yearly" via ISIN ligne 1 + libellé FR ligne 2,
                       "legacy" via codes Y01_/Q0N_/M0N_ ligne 1)
  - la couverture temporelle (date min, date max, nombre de lignes valides)
  - le nombre de produits par type (Cal / Quarter / Month) × (BASE / PEAK)

Sortie :
  docs/research/phase0_forwards_snapshot.jsonl  (1 ligne JSON par feuille)
  docs/research/phase0_forwards_summary.md       (résumé lisible)

Usage (PowerShell, env ppa_env) :
    $env:PATH='C:\\Users\\jbattaglia\\.conda\\ppa_env;C:\\Users\\jbattaglia\\.conda\\ppa_env\\Library\\bin;C:\\Users\\jbattaglia\\.conda\\ppa_env\\Scripts;' + $env:PATH
    C:\\Users\\jbattaglia\\.conda\\ppa_env\\python.exe scripts/phase0_sniff_forwards.py

Aucun écrit ailleurs que dans docs/research/. Ce script ne modifie pas les
fichiers Excel sources.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("phase0_sniff_forwards")


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_YEARLY_PATHS = [
    Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx"),
    Path(r"\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx"),
]
DEFAULT_HISTORY_PATHS = [
    Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Historique2019.xlsx"),
    Path(r"\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Historique2019.xlsx"),
]

OUT_JSONL = REPO_ROOT / "docs" / "research" / "phase0_forwards_snapshot.jsonl"
OUT_MD = REPO_ROOT / "docs" / "research" / "phase0_forwards_summary.md"


# ISIN EEX format: 2-letter country + 10 alphanum (ex. DE000EEX0D8L1)
_ISIN_RE = re.compile(r"^[A-Z]{2}[0-9A-Z]{10}$")
# Legacy product code (existing parser)
_LEGACY_RE = re.compile(r"^(Y01|Q\d{2}|M\d{2})_(\d{4})_(BASE|PEAK)$", re.IGNORECASE)
# French label heuristic: month abbreviation FR + 2-digit year + BASE/PEAK
_FR_MONTH_RE = re.compile(
    r"^(Jan|F[ée]v|Mar|Avr|Mai|Juin|Juil|Ao[uû]|Sep|Oct|Nov|D[ée]c)\s+\d{2}\s+(BASE|PEAK)\b",
    re.IGNORECASE,
)
_FR_QUARTER_RE = re.compile(r"^[QT]\s*[1-4]\s+\d{2}\s+(BASE|PEAK)\b", re.IGNORECASE)
_FR_CAL_RE = re.compile(r"^Cal\s+\d{2}\s+(BASE|PEAK)\b", re.IGNORECASE)


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


def _classify_cell(cell: object) -> str:
    """Return one of: 'isin', 'legacy', 'fr_label', 'empty', 'other'."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return "empty"
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none"):
        return "empty"
    if _ISIN_RE.match(s):
        return "isin"
    if _LEGACY_RE.match(s):
        return "legacy"
    if _FR_MONTH_RE.match(s) or _FR_QUARTER_RE.match(s) or _FR_CAL_RE.match(s):
        return "fr_label"
    return "other"


def _detect_format(row1: list[object], row2: list[object]) -> str:
    """Return 'yearly', 'legacy', or 'unknown'."""
    classes_1 = [_classify_cell(c) for c in row1]
    classes_2 = [_classify_cell(c) for c in row2]
    n_isin_1 = classes_1.count("isin")
    n_legacy_1 = classes_1.count("legacy")
    n_fr_2 = classes_2.count("fr_label")

    non_empty_1 = max(sum(1 for c in classes_1 if c != "empty"), 1)
    non_empty_2 = max(sum(1 for c in classes_2 if c != "empty"), 1)

    if n_isin_1 / non_empty_1 > 0.5 and n_fr_2 / non_empty_2 > 0.3:
        return "yearly"
    if n_legacy_1 / non_empty_1 > 0.5:
        return "legacy"
    # Yearly without ISIN (just FR labels in row 1 or row 2)
    if classes_1.count("fr_label") / non_empty_1 > 0.3:
        return "yearly_no_isin"
    return "unknown"


def _safe_cell(cell: object) -> object:
    """Convert to JSON-serializable scalar."""
    if cell is None:
        return None
    if isinstance(cell, float) and pd.isna(cell):
        return None
    if isinstance(cell, (int, float, str, bool)):
        return cell
    if isinstance(cell, pd.Timestamp):
        return cell.isoformat()
    return str(cell)


def sniff_sheet(file_path: Path, sheet_name: str) -> dict:
    """Inspect one sheet and return a dict ready for JSON Lines."""
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    n_rows, n_cols = raw.shape

    # First 3 rows × first 10 columns (clear text dump)
    head_block: list[list[object]] = []
    for r in range(min(3, n_rows)):
        row = []
        for c in range(min(10, n_cols)):
            row.append(_safe_cell(raw.iat[r, c]))
        head_block.append(row)

    row1 = list(raw.iloc[0, 1 : min(n_cols, 200)]) if n_rows >= 1 else []
    row2 = list(raw.iloc[1, 1 : min(n_cols, 200)]) if n_rows >= 2 else []
    fmt = _detect_format(row1, row2)

    # Coverage : column A from row index 3 (=4th row in Excel)
    date_series = (
        pd.to_datetime(raw.iloc[3:, 0], dayfirst=True, errors="coerce")
        if n_rows > 3
        else pd.Series(dtype="datetime64[ns]")
    )
    valid_dates = date_series.dropna()
    date_min = valid_dates.min().isoformat() if not valid_dates.empty else None
    date_max = valid_dates.max().isoformat() if not valid_dates.empty else None
    n_valid_rows = int(valid_dates.notna().sum())

    # Product type counts (heuristic)
    label_row = row2 if fmt == "yearly" else row1
    counts = {"Cal_BASE": 0, "Cal_PEAK": 0, "Q_BASE": 0, "Q_PEAK": 0,
              "M_BASE": 0, "M_PEAK": 0, "other": 0, "empty": 0}
    for cell in label_row:
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            counts["empty"] += 1
            continue
        s = str(cell).strip()
        if not s:
            counts["empty"] += 1
            continue
        is_peak = s.upper().endswith("PEAK")
        suffix = "_PEAK" if is_peak else "_BASE"
        if _FR_CAL_RE.match(s) or (_LEGACY_RE.match(s) and s.upper().startswith("Y01")):
            counts[f"Cal{suffix}"] += 1
        elif _FR_QUARTER_RE.match(s) or (_LEGACY_RE.match(s) and s.upper().startswith("Q")):
            counts[f"Q{suffix}"] += 1
        elif _FR_MONTH_RE.match(s) or (_LEGACY_RE.match(s) and s.upper().startswith("M")):
            counts[f"M{suffix}"] += 1
        else:
            counts["other"] += 1

    return {
        "file": file_path.name,
        "sheet": sheet_name,
        "shape": [int(n_rows), int(n_cols)],
        "format_detected": fmt,
        "head_3x10": head_block,
        "row1_first_5_cols_classified": [
            {"value": _safe_cell(row1[i]) if i < len(row1) else None,
             "class": _classify_cell(row1[i] if i < len(row1) else None)}
            for i in range(min(5, len(row1)))
        ],
        "row2_first_5_cols_classified": [
            {"value": _safe_cell(row2[i]) if i < len(row2) else None,
             "class": _classify_cell(row2[i] if i < len(row2) else None)}
            for i in range(min(5, len(row2)))
        ],
        "date_min": date_min,
        "date_max": date_max,
        "n_valid_date_rows": n_valid_rows,
        "product_type_counts": counts,
    }


def sniff_file(file_path: Path) -> list[dict]:
    """Sniff every sheet of a workbook."""
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return []
    logger.info("Sniffing %s", file_path)
    xl = pd.ExcelFile(file_path)
    logger.info("  Sheets: %s", xl.sheet_names)
    results: list[dict] = []
    for sheet in xl.sheet_names:
        try:
            r = sniff_sheet(file_path, sheet)
            results.append(r)
            logger.info(
                "  %-12s | shape=%s | fmt=%-15s | dates=%s..%s | products=%s",
                sheet,
                r["shape"],
                r["format_detected"],
                r["date_min"],
                r["date_max"],
                {k: v for k, v in r["product_type_counts"].items() if v},
            )
        except Exception as exc:
            logger.warning("  Failed to sniff sheet %s: %s", sheet, exc)
            results.append({
                "file": file_path.name,
                "sheet": sheet,
                "error": str(exc),
            })
    return results


def render_summary_md(records: list[dict]) -> str:
    """Render a human-readable Markdown summary."""
    lines: list[str] = [
        "# Phase 0 — snapshot des sources EEX (auto-généré)",
        "",
        f"_Généré par `scripts/phase0_sniff_forwards.py`_",
        "",
    ]
    by_file: dict[str, list[dict]] = {}
    for r in records:
        by_file.setdefault(r.get("file", "?"), []).append(r)

    for fname, recs in by_file.items():
        lines.append(f"## {fname}")
        lines.append("")
        lines.append("| Feuille | Format | Date min | Date max | Lignes valides | Produits |")
        lines.append("|---|---|---|---|---|---|")
        for r in recs:
            if "error" in r:
                lines.append(f"| {r['sheet']} | ERREUR | — | — | — | {r['error']} |")
                continue
            counts = r["product_type_counts"]
            non_zero = {k: v for k, v in counts.items() if v}
            lines.append(
                f"| {r['sheet']} | {r['format_detected']} | {r['date_min']} | "
                f"{r['date_max']} | {r['n_valid_date_rows']} | {non_zero} |"
            )
        lines.append("")

        # Show row1 / row2 of first sheet for parser contract
        first = recs[0] if recs else None
        if first and "head_3x10" in first:
            lines.append(f"### Échantillon 3×10 — feuille `{first['sheet']}`")
            lines.append("```")
            for row in first["head_3x10"]:
                lines.append(" | ".join(str(c) if c is not None else "" for c in row))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yearly",
        type=Path,
        default=None,
        help="Path to Price_Report_EEX_Yearly.xlsx (default: probes H: and UNC)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Path to Price_Report_EEX_CH_DE_Historique2019.xlsx (default: probes H: and UNC)",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=OUT_JSONL,
        help=f"Output JSONL path (default: {OUT_JSONL})",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=OUT_MD,
        help=f"Output Markdown summary path (default: {OUT_MD})",
    )
    args = parser.parse_args()

    yearly = args.yearly or _first_existing(DEFAULT_YEARLY_PATHS)
    history = args.history or _first_existing(DEFAULT_HISTORY_PATHS)

    if yearly is None and history is None:
        logger.error(
            "No source file found. Tried H:\\ and UNC paths. "
            "Pass --yearly and/or --history explicitly."
        )
        return 2

    all_records: list[dict] = []
    for path in [p for p in [yearly, history] if p is not None]:
        all_records.extend(sniff_file(path))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("JSONL written: %s (%d records)", args.out_jsonl, len(all_records))

    md = render_summary_md(all_records)
    args.out_md.write_text(md, encoding="utf-8")
    logger.info("Markdown summary written: %s", args.out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
