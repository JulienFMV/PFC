"""Fail-closed governance for uncalibrated LT probabilistic outputs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

INTERVAL_COLUMNS = ("p10", "p90")
DASHBOARD_PFC_EXPORT_COLUMNS = ("price_shape", "profile_type", "confidence")
DETERMINISTIC_PFC_COLUMNS = frozenset(
    {
        "price_shape",
        "B",
        "f_S",
        "f_W",
        "f_H",
        "f_Q",
        "f_WV",
        "delta_wv",
        "f_bridge",
        "profile_type",
        "confidence",
        "calibrated",
    }
)
DETERMINISTIC_ONLY_STATUS = "DETERMINISTIC_ONLY"
LEGACY_INTERVAL_STATUS = "QUARANTINED_UNCALIBRATED"
GOVERNED_INTERVAL_STATUSES = frozenset({"CALIBRATED_ROLLING_ORIGIN"})
_BOUND_TOKENS = frozenset({"lower", "upper", "low", "high", "lo", "hi"})


def _normalized_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def is_interval_column(column: object) -> bool:
    """Recognize legacy quantile/bound aliases without relying on exact casing."""

    normalized = _normalized_column_name(column)
    compact = normalized.replace("_", "")
    tokens = tuple(token for token in normalized.split("_") if token)
    if any(token in _BOUND_TOKENS for token in tokens):
        return True
    if any(re.fullmatch(r"[pq](?:\d{1,3})", token) for token in tokens):
        return True
    if re.search(r"(?:p|q)\d{1,3}", normalized):
        return True
    if re.search(r"(?:quantile|percentile)\d{1,3}", normalized):
        return True
    if "lowerbound" in compact or "upperbound" in compact:
        return True
    if re.search(r"(?:^|\d)(?:lower|upper|low|high)(?:\d{0,3})$", compact):
        return True
    if re.search(r"^ci\d{1,3}(?:lower|upper|low|high)$", compact):
        return True
    if re.search(r"(?:pctl|pctile)\d{1,3}", compact):
        return True
    if re.search(r"\d{1,3}(?:st|nd|rd|th)?(?:quantile|percentile)", compact):
        return True
    if any(
        token in {"p", "q"}
        and index + 1 < len(tokens)
        and re.fullmatch(r"\d{1,3}", tokens[index + 1])
        for index, token in enumerate(tokens)
    ):
        return True
    return any(
        token in {"quantile", "percentile"}
        and index + 1 < len(tokens)
        and re.fullmatch(r"\d{1,3}", tokens[index + 1])
        for index, token in enumerate(tokens)
    )


def deterministic_only_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return an LT frame without intervals, rejecting any populated bounds."""

    if not frame.columns.is_unique:
        raise ValueError(f"{context} contains duplicate column labels")
    unexpected = sorted(set(frame.columns).difference(DETERMINISTIC_PFC_COLUMNS).difference(INTERVAL_COLUMNS))
    if unexpected:
        raise ValueError(
            f"{context} contains columns outside the deterministic LT schema: {unexpected}"
        )
    if "price_shape" not in frame.columns:
        raise ValueError(f"{context} is missing required deterministic column: price_shape")
    interval_columns = [column for column in INTERVAL_COLUMNS if column in frame.columns]
    populated = [column for column in interval_columns if frame[column].notna().any()]
    if populated:
        raise ValueError(
            f"{context} contains populated uncalibrated interval columns: {populated}"
        )
    deterministic = frame.drop(columns=interval_columns).copy()
    deterministic.attrs.update(frame.attrs)
    deterministic.attrs["probabilistic_status"] = DETERMINISTIC_ONLY_STATUS
    return deterministic


def quarantine_legacy_intervals(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Hide legacy LT bounds while preserving an explicit quarantine classification."""

    interval_columns = [column for column in frame.columns if is_interval_column(column)]
    deterministic = frame.drop(columns=interval_columns).copy()
    deterministic.attrs.update(frame.attrs)
    deterministic.attrs["probabilistic_status"] = (
        LEGACY_INTERVAL_STATUS if interval_columns else DETERMINISTIC_ONLY_STATUS
    )
    deterministic.attrs["quarantined_interval_columns"] = interval_columns
    deterministic.attrs["probabilistic_context"] = str(context)
    return deterministic


def deterministic_pfc_export_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Project a dashboard PFC export onto its deterministic positive schema."""

    if not frame.columns.is_unique:
        raise ValueError(f"{context} contains duplicate column labels")
    quarantined = quarantine_legacy_intervals(frame, context=context)
    if "price_shape" not in quarantined.columns:
        raise ValueError(f"{context} is missing required deterministic column: price_shape")
    columns = [
        column for column in DASHBOARD_PFC_EXPORT_COLUMNS if column in quarantined.columns
    ]
    exported = quarantined.loc[:, columns].copy()
    exported.attrs.update(quarantined.attrs)
    return exported


def governed_intervals_available(frame: pd.DataFrame) -> bool:
    """Return true only for finite bounds carrying a governed probabilistic status."""

    if not frame.columns.is_unique:
        return False
    if not {"p10", "p90"}.issubset(frame.columns):
        return False
    if frame.attrs.get("probabilistic_status") not in GOVERNED_INTERVAL_STATUSES:
        return False
    if frame.empty:
        return False
    raw = frame[["p10", "p90"]]
    if raw.map(lambda value: isinstance(value, (bool, np.bool_))).to_numpy().any():
        return False
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and (values[:, 0] <= values[:, 1]).all())
