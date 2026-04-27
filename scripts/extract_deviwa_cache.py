"""Extract Deviwa, EEX yearly and HFC OMPEX sources to a fast parquet cache.

Production-grade pipeline for the FMV demo_deviwa dashboard:
- Resolves sources platform-aware (Windows H:\ → local data/ → legacy parquet)
- mtime + content-hash double validation (anti-fragile cache invalidation)
- Schema validation per file kind, fail-loud on drift
- Business-invariant data quality checks (e.g. EDSH supplier sum = programme)
- Atomic writes (.tmp + rename)
- Structured JSON logging to data/_cache/extract.log
- Idempotent: running twice on unchanged sources is a no-op

CLI:
    python scripts/extract_deviwa_cache.py            # incremental, mtime-aware
    python scripts/extract_deviwa_cache.py --force    # bypass cache
    python scripts/extract_deviwa_cache.py --quiet    # log warnings/errors only

Layout produced:
    data/_cache/
    ├── deviwa_deals.parquet           (all actors, scope flipped to client view)
    ├── deviwa_programme.parquet       (hourly programme + real per actor)
    ├── deviwa_edsh_suppliers.parquet  (8 cols incl. supplier breakdown)
    ├── eex_yearly_ch.parquet          (long-format CH forwards history)
    ├── pfc_ompex.parquet              (15min PFC reference, when available)
    ├── _meta.json                     (per-file fingerprints + audit trail)
    └── extract.log                    (append-only structured log)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Importer le parseur Deviwa existant pour réutiliser le scope flip et la
# normalisation des colonnes (pas de duplication de logique).
sys.path.insert(0, str(REPO_ROOT / "demo_deviwa"))
from deviwa_parser import (  # noqa: E402
    _invert_to_client_perspective,
    _standardize_columns,
    _assign_actor,
    HPFC_SHEET_KEYS,
)

# ---------------------------------------------------------------------------
# Source priority (Windows FMV → Mac local → legacy parquet)
# ---------------------------------------------------------------------------

WINDOWS_EEX_YEARLY = Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx")
WINDOWS_HFC_DIR = Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min")

LOCAL_EEX_YEARLY = DATA_DIR / "Price_Report_EEX_Yearly.xlsx"
LOCAL_PFC_OMPEX = DATA_DIR / "PFC_OMPEX.xlsx"
LOCAL_DEVIWA = DATA_DIR / "Deviwa.xlsx"

LEGACY_EEX_PARQUET = DATA_DIR / "eex_forwards_history.parquet"


# ---------------------------------------------------------------------------
# Structured logging (JSON-line format, append to file + stderr)
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        # Inclure les attributs `extra=` passés au log
        skip = {"name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "taskName"}
        for k, v in record.__dict__.items():
            if k in skip or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logger(quiet: bool = False) -> logging.Logger:
    log = logging.getLogger("deviwa_extract")
    log.handlers.clear()
    log.setLevel(logging.WARNING if quiet else logging.INFO)

    # Stderr handler (compact, lisible)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(sh)

    # File handler (JSON, append-only audit trail)
    fh = logging.FileHandler(CACHE_DIR / "extract.log", mode="a", encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    log.addHandler(fh)

    return log


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceCandidate:
    path: Path
    kind: str  # "windows_h_drive" | "local_data" | "legacy_parquet"


def resolve_eex_yearly() -> SourceCandidate | None:
    candidates = [
        SourceCandidate(WINDOWS_EEX_YEARLY, "windows_h_drive"),
        SourceCandidate(LOCAL_EEX_YEARLY, "local_data"),
        SourceCandidate(LEGACY_EEX_PARQUET, "legacy_parquet"),
    ]
    for c in candidates:
        try:
            if c.path.exists() and c.path.stat().st_size > 0:
                return c
        except OSError:
            continue
    return None


def resolve_hfc_ompex() -> SourceCandidate | None:
    # H:\ : on prend le plus récent HFC_Ompex_*.xlsx
    try:
        if WINDOWS_HFC_DIR.exists():
            candidates = sorted(
                WINDOWS_HFC_DIR.glob("HFC_Ompex_*.xlsx"),
                key=lambda p: p.name, reverse=True,
            )
            if not candidates:
                candidates = sorted(
                    WINDOWS_HFC_DIR.glob("HFC*.xlsx"),
                    key=lambda p: p.name, reverse=True,
                )
            if candidates:
                return SourceCandidate(candidates[0], "windows_h_drive")
    except OSError:
        pass

    if LOCAL_PFC_OMPEX.exists() and LOCAL_PFC_OMPEX.stat().st_size > 100:
        return SourceCandidate(LOCAL_PFC_OMPEX, "local_data")

    return None


def resolve_deviwa() -> SourceCandidate | None:
    if LOCAL_DEVIWA.exists() and LOCAL_DEVIWA.stat().st_size > 0:
        return SourceCandidate(LOCAL_DEVIWA, "local_data")
    return None


# ---------------------------------------------------------------------------
# Fingerprint : path + (mtime_ns, size) fast-path + SHA-256 full slow-path
# ---------------------------------------------------------------------------
#
# Stratégie deux niveaux :
#   1. Fast path (no I/O au-delà de stat) : si (path, mtime_ns, size) match
#      le précédent fingerprint, on assume que le fichier n'a pas changé.
#      Couvre le cas le plus fréquent (la source n'a pas bougé) à coût ~0.
#   2. Slow path (lecture I/O complète) : si mtime_ns ou size change,
#      on calcule un SHA-256 du fichier ENTIER pour décider si re-extraire.
#      Coût : ~10ms pour un xlsx 4 MB. Évite de re-extraire après un simple
#      "touch" qui n'a pas vraiment changé le contenu.
#
# Refus du sha256 partiel (head 64KB) parce que pour un xlsx (zip),
# une édition au-delà des 64 premiers KB sans changement de taille passerait
# inaperçue et servirait du parquet stale.

CHUNK_SIZE = 1024 * 1024  # 1 MB


def _sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path, *, full_hash: bool = True) -> dict:
    """Compute fingerprint. full_hash=False skips SHA-256 (used for fast probe)."""
    st = path.stat()
    fp = {
        "path": str(path),
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
    }
    if full_hash:
        fp["sha256_full"] = _sha256_full(path)
    return fp


def fingerprints_match(prior: dict | None, current_path: Path) -> bool:
    """Decide if cache is still valid for `current_path` given prior fingerprint.

    Two-level check :
    - Fast : path + mtime_ns + size — if all match, accept (no hash I/O).
    - Slow : if mtime_ns OR size differs, fall back to SHA-256 full check.
      If hash matches → still accept (just an mtime touch, no real change).
    """
    if not prior:
        return False
    try:
        st = current_path.stat()
    except OSError:
        return False

    if prior.get("path") != str(current_path):
        return False

    fast_match = (
        prior.get("mtime_ns") == int(st.st_mtime_ns)
        and prior.get("size") == int(st.st_size)
    )
    if fast_match:
        return True

    # Size or mtime changed — recompute hash to detect false positives
    prior_hash = prior.get("sha256_full")
    if not prior_hash:
        return False  # legacy fingerprint without full hash → re-extract
    current_hash = _sha256_full(current_path)
    return current_hash == prior_hash


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def write_parquet_atomic(df: pd.DataFrame, dest: Path) -> int:
    """Write DataFrame to parquet atomically. Returns file size in bytes.

    Use a unique tmp name (PID + nanosecond) so concurrent writers don't
    collide on the same temp file. The final rename is atomic on POSIX
    (rename(2)) and on NTFS (MoveFileEx with REPLACE_EXISTING).
    """
    import os as _os
    dest.parent.mkdir(parents=True, exist_ok=True)
    unique_suffix = f".{_os.getpid()}.{time.time_ns()}.tmp"
    tmp = dest.with_suffix(dest.suffix + unique_suffix)
    try:
        df.to_parquet(tmp, compression="snappy", index=isinstance(df.index, pd.DatetimeIndex))
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    tmp.replace(dest)
    return dest.stat().st_size


# ---------------------------------------------------------------------------
# Meta / fingerprint store
# ---------------------------------------------------------------------------

META_FILE = CACHE_DIR / "_meta.json"


def load_meta() -> dict:
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_meta(meta: dict) -> None:
    tmp = META_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(META_FILE)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    pass


REQUIRED_DEALS_RAW_COLS = {
    "Counterparty", "Asset Type", "Deal", "Deal Trade Date",
    "Deal Delivery Start", "Deal Delivery End", "Product", "Scope",
    "Date",
    # Volume — gross + net mean/sum (net colonnes utilisées par le scope flip)
    "Volume (Sum)", "Volume (Mean)", "Volume (Net) (Sum)", "Volume (Net) (Mean)",
    # Market value — utilisé par le scope flip (sign flip)
    "Market Value (Sum)", "Market Value (Mean)",
    # Notional + PnL
    "Notional (Sum)", "Notional (Mean)", "PNL (Sum)", "PNL (Mean)",
}


def validate_deals_raw_schema(df: pd.DataFrame, sheet: str) -> None:
    missing = REQUIRED_DEALS_RAW_COLS - set(df.columns)
    if missing:
        raise SchemaError(
            f"Sheet '{sheet}': colonnes attendues manquantes: {sorted(missing)}.\n"
            f"Colonnes trouvées: {sorted(df.columns)}"
        )


def validate_programme_raw_schema(df: pd.DataFrame, sheet: str) -> None:
    cols_lower = [str(c).lower() for c in df.columns]
    has_date = any(c in ("date", "datum", "zeit", "timestamp") for c in cols_lower)
    if not has_date:
        raise SchemaError(f"Sheet '{sheet}': aucune colonne date/datum/timestamp trouvée.")
    # Au moins une colonne Programme/Real attendue
    has_prog = any(any(k in c for k in ("programm", "_program", "plan", "soll")) for c in cols_lower)
    has_real = any(any(k in c for k in ("real", "_ist", "actual")) for c in cols_lower)
    if not (has_prog or has_real):
        raise SchemaError(
            f"Sheet '{sheet}': aucune colonne Programme/Real détectée. "
            f"Colonnes : {sorted(df.columns)}"
        )


# Colonnes obligatoires si on détecte un sheet EDSH avec décomposition fournisseurs.
# Si une de ces colonnes disparaît, on fail-loud plutôt que d'écrire silencieusement
# 0.0 partout (ce qui passerait les checks d'invariant si tous = 0).
EDSH_SUPPLIER_REQUIRED_SUFFIXES = {"_bkw", "_enalpin", "_ewz", "_spot"}


def validate_edsh_supplier_schema(df: pd.DataFrame, sheet: str) -> None:
    cols_lower = [str(c).strip().lower() for c in df.columns]
    missing: list[str] = []
    for suffix in EDSH_SUPPLIER_REQUIRED_SUFFIXES:
        if not any(c.endswith(suffix) for c in cols_lower):
            missing.append(suffix)
    # FMV : tolère le typo "ESDH_FMV" (présent dans la source) ou EDSH_FMV
    has_fmv = any("fmv" in c for c in cols_lower)
    if not has_fmv:
        missing.append("_fmv (or fmv variant)")
    if missing:
        raise SchemaError(
            f"EDSH sheet '{sheet}': colonnes fournisseur manquantes : {missing}.\n"
            f"Colonnes trouvées : {sorted(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Quality checks (business invariants)
# ---------------------------------------------------------------------------

def quality_check_deals(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if df.empty:
        issues.append("deals: dataframe vide")
        return issues

    # Volumes négatifs sur volume_sum (devrait être absolu après extract)
    if "volume_sum" in df.columns:
        n_neg = int((pd.to_numeric(df["volume_sum"], errors="coerce") < 0).sum())
        if n_neg > 0:
            issues.append(f"deals: {n_neg} lignes avec volume_sum < 0 (anomalie)")

    # Dates plausibles
    if "delivery_from" in df.columns:
        dlv = pd.to_datetime(df["delivery_from"], errors="coerce")
        if dlv.dt.year.min() < 2020 or dlv.dt.year.max() > 2032:
            issues.append(f"deals: plage delivery suspecte {dlv.min()} → {dlv.max()}")

    # Cohérence scope ↔ valeurs (après flip côté client)
    if "scope" in df.columns:
        unknown = df.loc[~df["scope"].astype(str).str.lower().str.contains(
            "intake|withdrawal", na=True), "scope"].dropna().unique()
        if len(unknown) > 0:
            issues.append(f"deals: scopes inattendus après flip: {list(unknown)[:5]}")

    return issues


def quality_check_edsh_suppliers(df: pd.DataFrame, tol_mw: float = 0.001) -> list[str]:
    """Invariant métier: BKW + EnAlpin + EWZ + FMV + Spot ≈ PROGRAMME.

    Note métier : fmv_mw et spot_mw peuvent être négatifs (= EDSH vend
    le surplus). Seuls bkw/enalpin/ewz (contrats fermes en achat) doivent
    rester >= 0. Plages excessives (> 50 MW) flaggées comme suspectes.
    """
    issues: list[str] = []
    if df.empty:
        return ["edsh_suppliers: dataframe vide"]

    supplier_cols = ["bkw_mw", "enalpin_mw", "ewz_mw", "fmv_mw", "spot_mw"]
    missing = [c for c in supplier_cols if c not in df.columns]
    if missing:
        issues.append(f"edsh_suppliers: colonnes manquantes {missing}")
        return issues

    # Invariant principal : programme = sum
    sum_supply = df[supplier_cols].sum(axis=1)
    diff = (sum_supply - df["programme_mw"]).abs()
    bad = int((diff > tol_mw).sum())
    n = len(df)
    if bad > 0:
        ratio = 100.0 * bad / n
        issues.append(
            f"edsh_suppliers: {bad}/{n} ({ratio:.2f}%) heures avec sum(suppliers+spot) "
            f"≠ programme (>{tol_mw:.4f} MW d'écart)"
        )

    # Contrats fermes positifs uniquement (BKW/EnAlpin/EWZ)
    for col in ("bkw_mw", "enalpin_mw", "ewz_mw"):
        n_neg = int((df[col] < -tol_mw).sum())
        if n_neg > 0:
            issues.append(f"edsh_suppliers: {col} a {n_neg} valeurs < 0 (suspect — contrat ferme)")

    # Plages plausibles (50 MW max pour un GRD du Haut-Valais)
    for col in supplier_cols:
        if df[col].abs().max() > 50:
            issues.append(f"edsh_suppliers: |{col}| max = {df[col].abs().max():.1f} MW (suspect > 50)")

    return issues


def quality_check_programme(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if df.empty:
        return ["programme: dataframe vide"]

    # Pas de duplicates (timestamp, actor)
    if {"timestamp", "actor"}.issubset(df.columns):
        dup = int(df.duplicated(subset=["timestamp", "actor"]).sum())
        if dup > 0:
            issues.append(f"programme: {dup} duplicates (timestamp, actor)")

    # Programme net devrait être borné (typique GRD < 50 MW)
    if "programme_mw" in df.columns:
        p = pd.to_numeric(df["programme_mw"], errors="coerce")
        if p.abs().max() > 100:
            issues.append(f"programme: |programme_mw| max = {p.abs().max():.1f} MW (suspect > 100)")

    return issues


def quality_check_eex_yearly(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if df.empty:
        return ["eex_yearly: dataframe vide"]

    if "price" in df.columns:
        p = pd.to_numeric(df["price"], errors="coerce")
        if (p < 0).any():
            issues.append(f"eex_yearly: {(p < 0).sum()} prix négatifs")
        if (p > 1000).any():
            issues.append(f"eex_yearly: {(p > 1000).sum()} prix > 1000 EUR/MWh (suspects)")

    return issues


# ---------------------------------------------------------------------------
# Readers (extract logic per file kind)
# ---------------------------------------------------------------------------

def _read_deviwa_xlsx(path: Path) -> dict[str, pd.DataFrame]:
    """Charge tout Deviwa.xlsx en 3 frames :
       deals (long, scope flippé), programme (long), edsh_suppliers (wide).
    """
    xls = pd.ExcelFile(path)

    deals_frames: list[pd.DataFrame] = []
    programme_frames: list[pd.DataFrame] = []
    edsh_suppliers: pd.DataFrame | None = None

    for sheet in xls.sheet_names:
        sn = sheet.lower()
        if any(k in sn for k in HPFC_SHEET_KEYS):
            continue  # HPFC géré ailleurs si besoin

        # ── DEALS ──────────────────────────────────────────────────────
        if "deal" in sn:
            actor = _assign_actor(sn)
            if actor is None:
                continue
            raw = pd.read_excel(xls, sheet_name=sheet)
            validate_deals_raw_schema(raw, sheet)
            df = _standardize_columns(raw)
            df = _invert_to_client_perspective(df)
            df["actor"] = actor
            df["source_sheet"] = sheet
            deals_frames.append(df)
            continue

        # ── PRO_REAL ────────────────────────────────────────────────────
        if "pro_real" in sn or "programme" in sn:
            raw = pd.read_excel(xls, sheet_name=sheet)
            validate_programme_raw_schema(raw, sheet)
            actor = _assign_actor(sn)

            # Cas 1 : EDSH avec ses 8 colonnes — extraction spéciale
            cols_norm = {c: str(c).strip() for c in raw.columns}
            cols_lower = {k: v.lower() for k, v in cols_norm.items()}
            edsh_supplier_signature = {"edsh_bkw", "edsh_enalpin", "edsh_ewz"}
            present = {v for v in cols_lower.values()}
            is_edsh_full = any(s in v for v in present for s in edsh_supplier_signature)

            if is_edsh_full:
                # Fail-loud si une colonne fournisseur disparaît silencieusement
                validate_edsh_supplier_schema(raw, sheet)
                edsh_suppliers = _extract_edsh_suppliers(raw)
                # On extrait aussi programme/real standard pour le frame programme global
                std = _extract_standard_programme(raw, actor or "EDSH")
                if std is not None and not std.empty:
                    programme_frames.append(std)
            else:
                std = _extract_standard_programme(raw, actor or sheet.strip())
                if std is not None and not std.empty:
                    programme_frames.append(std)
            continue

    deals = pd.concat(deals_frames, ignore_index=True) if deals_frames else pd.DataFrame()
    programme = pd.concat(programme_frames, ignore_index=True) if programme_frames else pd.DataFrame()
    return {
        "deals": deals,
        "programme": programme,
        "edsh_suppliers": edsh_suppliers if edsh_suppliers is not None else pd.DataFrame(),
    }


def _detect_date_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() in ("date", "datum", "zeit", "timestamp"):
            return c
    return None


def _resolve_dst_ambiguous(s: pd.Series) -> np.ndarray:
    """Return the ``ambiguous=`` array for a naive Zurich-local timestamp series.

    Strategy : detect duplicate naive timestamps (= the autumn DST hour
    written twice). For each duplicate, the **first** occurrence is the
    pre-transition (CEST, ``True``) reading, the **second** is the post-
    transition (CET, ``False``) reading. For non-duplicated timestamps the
    default is CEST (``True``) — pandas only reads the array on ambiguous
    moments anyway, so the value at non-DST hours is irrelevant.

    Using this array with ``tz_localize(..., ambiguous=arr)`` preserves
    the 25 physical hours of the autumn DST day instead of silently
    dropping the duplicates with ``ambiguous="NaT"``.
    """
    counts = s.value_counts()
    dup_set = set(counts[counts >= 2].index)
    amb_arr = np.ones(len(s), dtype=bool)
    if not dup_set:
        return amb_arr
    seen: dict[pd.Timestamp, int] = {}
    for i, t in enumerate(s):
        if pd.isna(t):
            continue
        if t in dup_set:
            c = seen.get(t, 0)
            seen[t] = c + 1
            amb_arr[i] = (c == 0)
    return amb_arr


def _localize_zurich_no_shift(ts):
    """Localize naive Zurich timestamps without the end-of-interval shift.

    Same DST-safe logic as :func:`_localize_zurich` (resolves the autumn
    DST duplicate via per-element ``ambiguous`` array, no ``NaT`` data
    loss) but does **not** subtract one hour. Use this for sources that
    already publish ``timestamp = start of interval`` — typically PFC /
    HFC OMPEX 15-min curves and EEX settlements.

    Accepts a :class:`pandas.Series` or :class:`pandas.DatetimeIndex` and
    returns the same shape, tz-aware ``Europe/Zurich``.
    """
    is_index = isinstance(ts, pd.DatetimeIndex)
    if is_index:
        if ts.tz is not None:
            return ts.tz_convert("Europe/Zurich")
        s = pd.Series(ts)
    else:
        if getattr(ts.dt, "tz", None) is not None:
            return ts.dt.tz_convert("Europe/Zurich")
        s = ts

    amb_arr = _resolve_dst_ambiguous(s)
    localized = s.dt.tz_localize(
        "Europe/Zurich",
        nonexistent="shift_forward",
        ambiguous=amb_arr,
    )
    return pd.DatetimeIndex(localized) if is_index else localized


def _localize_zurich(ts: pd.Series) -> pd.Series:
    """Localize naive Deviwa timestamps to Europe/Zurich without DST data loss.

    Convention source Deviwa : ``timestamp = HEURE DE FIN d'intervalle``.
    Un jour commence donc à 01:00 (= fin de l'intervalle 00:00→01:00) et
    se termine à 00:00 du jour suivant (= fin de 23:00→00:00). Sur la
    transition automne (CEST→CET), la source écrit deux fois "03:00"
    pour les deux heures physiques (fin CEST puis fin CET).

    On normalise en convention ``timestamp = DÉBUT d'intervalle`` en
    soustrayant 1 heure à toutes les valeurs naïves. Après ce shift :
      - la journée commence à 00:00 (= début de 00:00→01:00)
      - les doublons "03:00" deviennent doublons "02:00" — pile l'heure
        ambiguë de la transition DST automne en Europe/Zurich
      - on peut alors utiliser ``ambiguous=array`` proprement :
        1ère occurrence = True (CEST = avant transition)
        2nde occurrence = False (CET = après transition)
    Ce qui préserve les 25 heures réelles de la journée automne sans NaT.
    """
    if ts.dt.tz is not None:
        return ts.dt.tz_convert("Europe/Zurich")

    # Conversion fin-d'intervalle → début-d'intervalle
    ts_start = ts - pd.Timedelta(hours=1)
    amb_arr = _resolve_dst_ambiguous(ts_start)
    return ts_start.dt.tz_localize(
        "Europe/Zurich",
        nonexistent="shift_forward",
        ambiguous=amb_arr,
    )


def _extract_standard_programme(raw: pd.DataFrame, actor: str) -> pd.DataFrame | None:
    """Extrait programme/real/budget en format long (timestamp, actor, ...)."""
    date_col = _detect_date_column(raw)
    if date_col is None:
        return None

    ts = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    cols_lower = {c: str(c).strip().lower() for c in raw.columns}

    programme_col = None
    real_col = None
    budget_col = None

    for c, lc in cols_lower.items():
        if c == date_col:
            continue
        # Suffixe explicite
        if lc.endswith("_programm") or lc.endswith("_program"):
            programme_col = programme_col or c
        elif lc.endswith("_real") or lc.endswith("_ist") or lc.endswith("_actual"):
            real_col = real_col or c
        elif lc.endswith("_budget"):
            budget_col = budget_col or c
        # Heuristique fallback
        elif any(k in lc for k in ("programm", "plan", "geplant", "soll")) and programme_col is None:
            programme_col = c
        elif any(k in lc for k in ("real", "ist", "actual", "spot")) and real_col is None:
            # éviter les colonnes _spot supplier (EDSH)
            if not lc.startswith("edsh_") and "spot" not in lc:
                real_col = c
        elif "budget" in lc and budget_col is None:
            budget_col = c

    if programme_col is None and real_col is None:
        return None

    out = pd.DataFrame({"timestamp": ts})
    out["programme_mw"] = pd.to_numeric(raw[programme_col], errors="coerce") if programme_col else np.nan
    out["actual_mw"] = pd.to_numeric(raw[real_col], errors="coerce") if real_col else np.nan
    out["budget_mw"] = pd.to_numeric(raw[budget_col], errors="coerce") if budget_col else np.nan
    out = out.dropna(subset=["timestamp"]).copy()
    if out.empty:
        return None

    out["timestamp"] = _localize_zurich(out["timestamp"])
    out = out[out["timestamp"].notna()].sort_values("timestamp")
    # Dédupliquer les transitions DST automne (CEST→CET produit 2× la même heure
    # locale ; on garde la première, les valeurs sont identiques en pratique).
    out = out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    out["actor"] = actor
    out["delta_mw"] = out["actual_mw"] - out["programme_mw"]
    return out


def _extract_edsh_suppliers(raw: pd.DataFrame) -> pd.DataFrame:
    """Extrait les 8 colonnes EDSH en schéma normalisé."""
    date_col = _detect_date_column(raw)
    if date_col is None:
        return pd.DataFrame()

    ts = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    cols_lower = {c: str(c).strip().lower() for c in raw.columns}

    def find_col(suffix: str) -> str | None:
        for c, lc in cols_lower.items():
            if lc.endswith(suffix.lower()):
                return c
        return None

    out = pd.DataFrame({"timestamp": ts})
    out["programme_mw"] = pd.to_numeric(raw[find_col("programme")], errors="coerce") if find_col("programme") else np.nan
    out["actual_mw"] = pd.to_numeric(raw[find_col("real")], errors="coerce") if find_col("real") else np.nan
    out["bkw_mw"] = pd.to_numeric(raw[find_col("_bkw")], errors="coerce") if find_col("_bkw") else 0.0
    out["enalpin_mw"] = pd.to_numeric(raw[find_col("_enalpin")], errors="coerce") if find_col("_enalpin") else 0.0
    out["ewz_mw"] = pd.to_numeric(raw[find_col("_ewz")], errors="coerce") if find_col("_ewz") else 0.0

    # FMV : tolère le typo "ESDH_FMV" présent dans la source
    fmv_col = find_col("_fmv")
    if fmv_col is None:
        for c, lc in cols_lower.items():
            if "fmv" in lc:
                fmv_col = c
                break
    out["fmv_mw"] = pd.to_numeric(raw[fmv_col], errors="coerce") if fmv_col else 0.0

    out["spot_mw"] = pd.to_numeric(raw[find_col("_spot")], errors="coerce") if find_col("_spot") else 0.0

    out = out.dropna(subset=["timestamp"]).copy()
    if out.empty:
        return out

    out["timestamp"] = _localize_zurich(out["timestamp"])
    out = out[out["timestamp"].notna()].sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    out["actor"] = "EDSH"
    return out


def _read_eex_yearly_xlsx(path: Path) -> pd.DataFrame:
    """Parse Price_Report_EEX_Yearly.xlsx → long-format CH forwards."""
    raw = pd.read_excel(path, sheet_name="CH", header=None)
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        return pd.DataFrame()

    codes = raw.iloc[0].astype(str)
    labels = raw.iloc[1].astype(str)
    dates = pd.to_datetime(raw.iloc[2:, 0], errors="coerce", dayfirst=True)

    rows: list[pd.DataFrame] = []
    for col_idx in range(1, raw.shape[1]):
        code = str(codes.iloc[col_idx]).strip()
        label = str(labels.iloc[col_idx]).strip()
        if not label or label.lower() == "nan" or label.startswith("Unnamed"):
            label = code
        is_peak = "PEAK" in code.upper() or "PEAK" in label.upper()
        load_type = "peak" if is_peak else "base"
        prices = pd.to_numeric(raw.iloc[2:, col_idx], errors="coerce")
        mask = dates.notna() & prices.notna() & (prices > 0)
        if not mask.any():
            continue
        rows.append(pd.DataFrame({
            "date": dates[mask].values,
            "market": "CH",
            "product": f"{label} {code}".strip(),
            "load_type": load_type,
            "price": prices[mask].values,
        }))

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _read_eex_yearly_legacy_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _read_pfc_ompex_xlsx(path: Path) -> pd.DataFrame:
    """Parse HFC OMPEX 15-min curve (sheet HFC ou similaire)."""
    try:
        df = pd.read_excel(path, sheet_name="HFC")
    except (ValueError, KeyError):
        # Premier sheet par défaut
        df = pd.read_excel(path)

    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()

    cols_lower = {c: str(c).strip().lower() for c in df.columns}
    date_col = next((c for c, lc in cols_lower.items() if lc in ("date", "timestamp", "datum")), None)
    price_col = next((c for c, lc in cols_lower.items() if "eur" in lc and "mwh" in lc), None)
    if date_col is None or price_col is None:
        # Fallback : 2 premières colonnes
        date_col = df.columns[0]
        price_col = df.columns[1]

    ts = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    price = pd.to_numeric(df[price_col], errors="coerce")
    out = pd.DataFrame({"price_shape": price.values}, index=ts)
    out = out.dropna(subset=["price_shape"])
    out = out[out.index.notna()].sort_index()
    if out.empty:
        return out
    out.index = pd.DatetimeIndex(out.index)
    out.index = _localize_zurich_no_shift(out.index)
    return out


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ExtractResult:
    file_kind: str
    action: str  # "extracted" | "skipped_up_to_date" | "no_source" | "failed"
    source_path: str | None = None
    source_kind: str | None = None
    cache_path: str | None = None
    cache_size_kb: int = 0
    n_rows: int = 0
    elapsed_ms: float = 0.0
    quality_issues: list[str] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def run_extract_step(
    file_kind: str,
    resolver: Callable[[], SourceCandidate | None],
    extractor: Callable[[Path], dict | pd.DataFrame],
    quality_checker: Callable[[pd.DataFrame], list[str]] | None,
    output_specs: list[tuple[str, str]],  # [(meta_key, cache_filename), ...]
    meta: dict,
    log: logging.Logger,
    force: bool,
) -> ExtractResult:
    """Run one extract step. output_specs maps logical keys to cache filenames.
    For multi-output extractors (e.g. Deviwa → 3 frames), the extractor returns
    a dict {key: DataFrame}. For single-output, it returns a DataFrame and
    output_specs has 1 entry.
    """
    t0 = time.perf_counter()
    src = resolver()
    if src is None:
        log.warning("source_unresolved", extra={"file_kind": file_kind})
        return ExtractResult(file_kind=file_kind, action="no_source")

    prior_meta = meta.get(file_kind, {})
    prior = prior_meta.get("source_fingerprint")
    cache_paths_exist = all((CACHE_DIR / fname).exists() for _, fname in output_specs)
    # Cas spécial : source connue comme vide dans le passé pour ce fingerprint.
    # On ne re-extrait pas — le fichier source n'a juste pas de contenu utile.
    prior_was_empty = (
        prior_meta.get("n_rows_primary", -1) == 0
        and prior_meta.get("source_empty_sentinel") is True
    )

    # Two-level fingerprint check (no I/O if mtime/size match prior)
    is_match = fingerprints_match(prior, src.path)

    if not force and is_match and (cache_paths_exist or prior_was_empty):
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("cache_hit", extra={
            "file_kind": file_kind, "source_kind": src.kind,
            "elapsed_ms": round(elapsed, 1),
            "empty_sentinel": prior_was_empty,
        })
        return ExtractResult(
            file_kind=file_kind,
            action="skipped_up_to_date",
            source_path=str(src.path), source_kind=src.kind,
            cache_path=str(CACHE_DIR / output_specs[0][1]),
            elapsed_ms=elapsed,
        )

    log.info("extract_start", extra={
        "file_kind": file_kind, "source_kind": src.kind,
        "source_path": str(src.path),
    })

    try:
        result = extractor(src.path)
    except SchemaError as e:
        log.error("schema_error", extra={"file_kind": file_kind, "error": str(e)})
        return ExtractResult(file_kind=file_kind, action="failed", error=f"schema: {e}")
    except Exception as e:
        log.error("extract_error", extra={
            "file_kind": file_kind,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        })
        return ExtractResult(file_kind=file_kind, action="failed",
                             error=f"{type(e).__name__}: {e}")

    # Normaliser : dict ou DataFrame
    frames = result if isinstance(result, dict) else {output_specs[0][0]: result}

    quality_all: list[str] = []
    primary_rows = 0
    primary_size = 0

    for sub_key, cache_name in output_specs:
        df = frames.get(sub_key, pd.DataFrame())
        if df is None or df.empty:
            log.warning("empty_frame", extra={"file_kind": file_kind, "sub_key": sub_key})
            continue

        # Quality checks par sub_key
        if quality_checker is not None and sub_key == output_specs[0][0]:
            issues = quality_checker(df)
            if issues:
                for it in issues:
                    log.warning("quality_issue", extra={"file_kind": file_kind, "issue": it})
                quality_all.extend(issues)

        cache_path = CACHE_DIR / cache_name
        size = write_parquet_atomic(df, cache_path)
        if sub_key == output_specs[0][0]:
            primary_rows = len(df)
            primary_size = size
        log.info("wrote_parquet", extra={
            "file_kind": file_kind, "sub_key": sub_key,
            "cache_path": str(cache_path), "rows": len(df), "size_kb": size // 1024,
        })

    elapsed = (time.perf_counter() - t0) * 1000
    log.info("extract_done", extra={
        "file_kind": file_kind, "elapsed_ms": round(elapsed, 1),
        "n_quality_issues": len(quality_all),
    })

    # Update meta — y compris pour les sources vides (sentinel pour éviter de
    # re-extraire à chaque run quand le source xlsx est vide localement).
    primary_frame = frames.get(output_specs[0][0], pd.DataFrame())
    source_empty_sentinel = primary_frame is None or primary_frame.empty

    # Compute full-hash fingerprint to record (1 MB chunked SHA-256, ~10ms for xlsx)
    fp = fingerprint(src.path, full_hash=True)

    meta[file_kind] = {
        "source_path": str(src.path),
        "source_kind": src.kind,
        "source_fingerprint": fp,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "cache_files": [name for _, name in output_specs],
        "n_rows_primary": primary_rows,
        "size_kb_primary": primary_size // 1024,
        "quality_issues": quality_all,
        "source_empty_sentinel": source_empty_sentinel,
    }

    return ExtractResult(
        file_kind=file_kind, action="extracted",
        source_path=str(src.path), source_kind=src.kind,
        cache_path=str(CACHE_DIR / output_specs[0][1]),
        cache_size_kb=primary_size // 1024,
        n_rows=primary_rows,
        elapsed_ms=elapsed,
        quality_issues=quality_all,
    )


def run_all(force: bool = False, quiet: bool = False) -> list[ExtractResult]:
    log = setup_logger(quiet=quiet)
    meta = load_meta()
    results: list[ExtractResult] = []

    log.info("run_start", extra={"force": force, "cache_dir": str(CACHE_DIR)})

    # ── Deviwa : 1 source, 3 outputs ───────────────────────────────────
    def deviwa_extractor(p: Path) -> dict:
        return _read_deviwa_xlsx(p)

    def quality_deviwa_deals(df: pd.DataFrame) -> list[str]:
        return quality_check_deals(df)

    # On exécute l'extract Deviwa une seule fois mais on enregistre 3 cache files
    # via une astuce : on fait 3 appels qui partagent le même fingerprint mais
    # extraient un sous-key différent. Plus simple : un seul appel multi-output.
    results.append(run_extract_step(
        file_kind="deviwa_deals",
        resolver=resolve_deviwa,
        extractor=deviwa_extractor,
        quality_checker=quality_deviwa_deals,
        output_specs=[
            ("deals", "deviwa_deals.parquet"),
            ("programme", "deviwa_programme.parquet"),
            ("edsh_suppliers", "deviwa_edsh_suppliers.parquet"),
        ],
        meta=meta, log=log, force=force,
    ))

    # Vérifications additionnelles spécifiques aux sous-frames Deviwa.
    # Important : on REMPLACE quality_issues, on n'append pas — sinon la liste
    # grandit indéfiniment à chaque cache-hit run (re-lecture du parquet).
    if "deviwa_deals" in meta:
        deals_primary_issues = list(meta["deviwa_deals"].get("quality_issues", []) or [])
        # On garde uniquement les issues primaires (deals), on recalcule les
        # sous-frames issues à frais sans accumulation.
        # Note : deals_primary_issues a été rempli à l'extract initial.
        sub_issues: list[str] = []

        edsh_path = CACHE_DIR / "deviwa_edsh_suppliers.parquet"
        if edsh_path.exists():
            try:
                df_edsh = pd.read_parquet(edsh_path)
                issues_edsh = quality_check_edsh_suppliers(df_edsh)
                for it in issues_edsh:
                    log.warning("quality_issue", extra={"file_kind": "deviwa_edsh_suppliers", "issue": it})
                sub_issues.extend(issues_edsh)
            except Exception as e:
                log.warning("post_quality_check_failed", extra={"file_kind": "deviwa_edsh_suppliers", "error": str(e)})

        prog_path = CACHE_DIR / "deviwa_programme.parquet"
        if prog_path.exists():
            try:
                df_prog = pd.read_parquet(prog_path)
                issues_prog = quality_check_programme(df_prog)
                for it in issues_prog:
                    log.warning("quality_issue", extra={"file_kind": "deviwa_programme", "issue": it})
                sub_issues.extend(issues_prog)
            except Exception as e:
                log.warning("post_quality_check_failed", extra={"file_kind": "deviwa_programme", "error": str(e)})

        # Filtrer les issues "deals primaires" en gardant uniquement les non-EDSH
        # et non-programme (i.e. celles de quality_check_deals), puis appendre
        # les sub_issues fraîchement calculées. Idempotent.
        primary_only = [
            it for it in deals_primary_issues
            if not it.startswith("edsh_suppliers:") and not it.startswith("programme:")
        ]
        meta["deviwa_deals"]["quality_issues"] = primary_only + sub_issues

    # ── EEX yearly ─────────────────────────────────────────────────────
    def eex_extractor(p: Path) -> pd.DataFrame:
        if p.suffix.lower() == ".parquet":
            return _read_eex_yearly_legacy_parquet(p)
        return _read_eex_yearly_xlsx(p)

    results.append(run_extract_step(
        file_kind="eex_yearly_ch",
        resolver=resolve_eex_yearly,
        extractor=eex_extractor,
        quality_checker=quality_check_eex_yearly,
        output_specs=[("primary", "eex_yearly_ch.parquet")],
        meta=meta, log=log, force=force,
    ))

    # ── PFC OMPEX ──────────────────────────────────────────────────────
    results.append(run_extract_step(
        file_kind="pfc_ompex",
        resolver=resolve_hfc_ompex,
        extractor=_read_pfc_ompex_xlsx,
        quality_checker=None,
        output_specs=[("primary", "pfc_ompex.parquet")],
        meta=meta, log=log, force=force,
    ))

    save_meta(meta)
    log.info("run_done", extra={
        "n_extracted": sum(1 for r in results if r.action == "extracted"),
        "n_skipped": sum(1 for r in results if r.action == "skipped_up_to_date"),
        "n_failed": sum(1 for r in results if r.action == "failed"),
        "n_no_source": sum(1 for r in results if r.action == "no_source"),
    })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Deviwa/EEX/HFC sources to parquet cache.")
    parser.add_argument("--force", action="store_true", help="Bypass mtime/hash cache, force re-extract.")
    parser.add_argument("--quiet", action="store_true", help="Log warnings/errors only.")
    args = parser.parse_args()

    results = run_all(force=args.force, quiet=args.quiet)

    print("\n=== Extract summary ===")
    for r in results:
        marker = {
            "extracted": "✅",
            "skipped_up_to_date": "⏭️ ",
            "no_source": "⚠️ ",
            "failed": "❌",
        }.get(r.action, "?")
        rows_str = f" ({r.n_rows:,} rows, {r.cache_size_kb} KB)" if r.action == "extracted" else ""
        src_str = f" [{r.source_kind}]" if r.source_kind else ""
        print(f"  {marker} {r.file_kind:<22} {r.action}{src_str}{rows_str} ({r.elapsed_ms:.0f}ms)")
        if r.quality_issues:
            for it in r.quality_issues:
                print(f"     ⚠️  {it}")
        if r.error:
            print(f"     ❌ {r.error}")

    n_failed = sum(1 for r in results if r.action == "failed")
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
