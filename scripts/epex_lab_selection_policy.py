"""Shared no-OMPEX selection policy checks for EPEX lab promotion artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_policy_pass(
    selection: dict[str, Any] | None,
    *,
    adjusted_csv: Path | None = None,
    adjusted_csv_sha256: str | None = None,
) -> bool:
    return selection_policy_value(
        selection,
        adjusted_csv=adjusted_csv,
        adjusted_csv_sha256=adjusted_csv_sha256,
    )["pass"] is True


def selection_policy_value(
    selection: dict[str, Any] | None,
    *,
    adjusted_csv: Path | None = None,
    adjusted_csv_sha256: str | None = None,
) -> dict[str, Any]:
    selection = selection or {}
    adjusted_sha = adjusted_csv_sha256
    if adjusted_csv is not None:
        if not adjusted_csv.exists():
            adjusted_sha = None
        else:
            adjusted_sha = sha256(adjusted_csv)
    verdict = selection.get("replacement_verdict") or {}
    selected_trial = selection.get("selected_trial") if isinstance(selection.get("selected_trial"), dict) else {}
    selected_shas = [
        selection.get("selected_adjusted_csv_sha256"),
        selected_trial.get("adjusted_csv_sha256"),
    ]
    value = {
        "selection_summary_present": bool(selection),
        "adjusted_csv_sha256": adjusted_sha,
        "replace_incumbent": verdict.get("replace_incumbent"),
        "ompex_used_in_model": selection.get("ompex_used_in_model"),
        "ompex_used_in_selection": selection.get("ompex_used_in_selection"),
        "ompex_used_in_backtest": selection.get("ompex_used_in_backtest"),
        "selected_adjusted_csv_sha256": selection.get("selected_adjusted_csv_sha256"),
        "selected_trial_adjusted_csv_sha256": selected_trial.get("adjusted_csv_sha256"),
    }
    value["pass"] = bool(
        selection
        and adjusted_sha
        and verdict.get("replace_incumbent") is True
        and selection.get("ompex_used_in_model") is False
        and selection.get("ompex_used_in_selection") is False
        and selection.get("ompex_used_in_backtest") is False
        and adjusted_sha in {str(sha) for sha in selected_shas if sha}
    )
    return value


def selection_policy_manifest_value(
    production_manifest: dict[str, Any],
    *,
    adjusted_csv: Path,
) -> dict[str, Any]:
    selection_path_text = str(production_manifest.get("selection_summary") or "").strip()
    value: dict[str, Any] = {
        "self_attested": production_manifest.get("selection_policy_pass"),
        "selection_summary": production_manifest.get("selection_summary"),
        "selection_summary_sha256": production_manifest.get("selection_summary_sha256"),
        "validated": False,
    }
    if not selection_path_text:
        value["error"] = "selection_summary"
        return value
    selection_path = Path(selection_path_text)
    if not selection_path.exists():
        value["error"] = "selection_summary_missing"
        return value
    actual_sha = sha256(selection_path)
    value["selection_summary_actual_sha256"] = actual_sha
    if production_manifest.get("selection_summary_sha256") != actual_sha:
        value["error"] = "selection_summary_sha256"
        return value
    try:
        selection = load_json(selection_path)
    except (OSError, json.JSONDecodeError):
        value["error"] = "selection_summary_json"
        return value
    policy_value = selection_policy_value(selection, adjusted_csv=adjusted_csv)
    value.update(policy_value)
    value["validated"] = production_manifest.get("selection_policy_pass") is True and policy_value["pass"] is True
    if value["validated"] is not True:
        value["error"] = "selection_policy_pass"
    return value
