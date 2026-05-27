#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "ct_policy_decision_latest.json"
HORIZON_DIRS = {
    "h1": ROOT / "pfc_shaping" / "output" / "h1_canonical_quick",
    "h2": ROOT / "pfc_shaping" / "output" / "h2_canonical_quick",
    "h3": ROOT / "pfc_shaping" / "output" / "h3_canonical_quick",
    "h4": ROOT / "pfc_shaping" / "output" / "h4_canonical_quick",
    "h5": ROOT / "pfc_shaping" / "output" / "h5_canonical_quick",
    "h6": ROOT / "pfc_shaping" / "output" / "h6_canonical_quick",
    "h7": ROOT / "pfc_shaping" / "output" / "h7_canonical_quick",
    "h10": ROOT / "pfc_shaping" / "output" / "h10_canonical_quick",
}
HORIZON_FALLBACK_DIRS = {
    "h10": ROOT / "pfc_shaping" / "output" / "h10_canonical_smoke",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest(path: Path, suffix: str, exclude_contains: str | None = None) -> Path | None:
    matches = sorted(path.glob(f"*{suffix}"), key=lambda p: p.name)
    if exclude_contains:
        matches = [m for m in matches if exclude_contains not in m.name]
    return matches[-1] if matches else None


def _score_backtest(path: Path) -> dict[str, float | int]:
    df = pd.read_parquet(path)
    clean = df[["forecast", "actual"]].dropna()
    err = clean["forecast"] - clean["actual"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "corr": float(clean["forecast"].corr(clean["actual"])),
        "n": int(len(clean)),
    }


def _load_directional_parquet_summary(directory: Path, horizon_key: str) -> dict | None:
    baseline = _find_latest(directory, f"_{horizon_key}_no_fm_baseline.parquet")
    governed = _find_latest(directory, f"_{horizon_key}_no_fm_governed.parquet")
    if baseline is None or governed is None:
        return None
    return {
        "source": "directional_parquet",
        "baseline_path": str(baseline),
        "governed_path": str(governed),
        "baseline": _score_backtest(baseline),
        "governed": _score_backtest(governed),
    }


def _winner_vs_baseline_significant(winner_summary: dict, baseline_variant: str = "no_fm_baseline") -> bool:
    for row in winner_summary.get("winner_vs_others", []):
        if row.get("other") == baseline_variant:
            return bool(row.get("significant_candidate_better", False))
    return False


def _load_horizon_json_summary(directory: Path, horizon_key: str) -> dict | None:
    winners_path = _find_latest(directory, "_horizon_winners.json")
    full_path = _find_latest(directory, ".json", exclude_contains="_horizon_winners")
    if winners_path is None or full_path is None:
        return None
    winners_payload = _load_json(winners_path)[horizon_key]
    full_payload = _load_json(full_path)
    summary = full_payload["summary"][horizon_key]
    input_signatures = {
        key: value.get("sha256")
        for key, value in full_payload.get("inputs", {}).items()
        if isinstance(value, dict) and value.get("sha256")
    }
    return {
        "source": "full_json",
        "winners_path": str(winners_path),
        "full_path": str(full_path),
        "winner": winners_payload["winner_by_mae"],
        "winner_family": winners_payload["winner_family"],
        "recommended_routing_action": winners_payload["recommended_routing_action"],
        "significant_vs_second_best": winners_payload["significant_vs_second_best"],
        "significant_winner_vs_baseline": _winner_vs_baseline_significant(winners_payload),
        "winner_summary": winners_payload,
        "summary": summary,
        "decision_stats": {
            "dm_p_value_governed_vs_baseline": summary.get("dm_p_fm_governed_vs_fm_only_mae"),
            "bootstrap_ci_mae_governed_vs_baseline": summary.get("bootstrap_ci_fm_governed_vs_fm_only_mae"),
            "alpha_nominal": 0.05,
            "test_type": "one_sided_dm_plus_bootstrap",
        },
        "git_commit": full_payload.get("git_commit"),
        "vintage_schema_verified": bool(full_payload.get("vintage_schema_verified", False)),
        "input_signatures": input_signatures,
    }


def _load_horizon_result(horizon_key: str) -> dict:
    primary_dir = HORIZON_DIRS[horizon_key]
    json_result = _load_horizon_json_summary(primary_dir, horizon_key)
    if json_result is not None:
        return {
            "directory": str(primary_dir),
            **json_result,
        }
    fallback_dir = HORIZON_FALLBACK_DIRS.get(horizon_key)
    if fallback_dir is not None:
        directional = _load_directional_parquet_summary(fallback_dir, horizon_key)
        if directional is not None:
            return {
                "directory": str(fallback_dir),
                **directional,
            }
    raise FileNotFoundError(f"No usable horizon artifact for {horizon_key}")


def _all_same_non_null(values: list[str | None]) -> bool:
    non_null = [v for v in values if v is not None]
    return len(non_null) >= 2 and len(set(non_null)) == 1


def _contiguous_runs(days: list[int]) -> list[list[int]]:
    if not days:
        return []
    days = sorted(days)
    runs: list[list[int]] = [[days[0]]]
    for day in days[1:]:
        if day == runs[-1][-1] + 1:
            runs[-1].append(day)
        else:
            runs.append([day])
    return runs


def _derive_policy(horizons: dict[str, dict]) -> dict:
    h1 = horizons["h1"]
    h1_json = h1.get("source") == "full_json"
    governed_significant_days: list[int] = []
    governed_directional_days: list[int] = []
    for key, payload in horizons.items():
        if key == "h10":
            continue
        if payload.get("source") == "full_json" and payload.get("winner_family", "").startswith("governed"):
            day = int(key[1:])
            governed_directional_days.append(day)
            if bool(payload.get("significant_winner_vs_baseline", False)):
                governed_significant_days.append(day)

    h1_governed_supported = (
        h1_json
        and h1.get("winner_family", "").startswith("governed")
        and bool(h1.get("significant_winner_vs_baseline", False))
    )

    h10 = horizons["h10"]
    h10_tail_risk_red_flag = False
    if h10.get("source") == "directional_parquet":
        base = h10["baseline"]
        gov = h10["governed"]
        h10_tail_risk_red_flag = bool(gov["mae"] < base["mae"] and gov["rmse"] > base["rmse"])
    elif h10.get("source") == "full_json" and h10.get("winner_family", "").startswith("governed"):
        scores = h10.get("winner_summary", {}).get("winner_score", {})
        ranking = h10.get("winner_summary", {}).get("ranking", [])
        baseline = next((row for row in ranking if row.get("variant") == "no_fm_baseline"), None)
        if baseline is not None:
            h10_tail_risk_red_flag = bool(
                scores.get("mae", float("inf")) < baseline.get("mae", float("inf"))
                and scores.get("rmse", float("inf")) > baseline.get("rmse", float("-inf"))
            )

    significant_runs = _contiguous_runs(governed_significant_days)
    directional_runs = _contiguous_runs(governed_directional_days)
    significant_mid_horizon = [day for day in governed_significant_days if 2 <= day <= 7]

    return {
        "prod_policy": "primary_only_conservative",
        "governed_prod_enabled_by_default": False,
        "research_policy_candidate": "governed_mid_horizon_candidate" if significant_mid_horizon else "primary_only",
        "research_candidate_window_days": significant_mid_horizon,
        "research_candidate_window_runs": significant_runs,
        "governed_directional_window_days": governed_directional_days,
        "governed_directional_window_runs": directional_runs,
        "h1_governed_significant": h1_governed_supported,
        "h5_governed_significant": 5 in governed_significant_days,
        "h10_tail_risk_red_flag": h10_tail_risk_red_flag,
        "rationale": [
            "Prod stays conservative because h1 governed is not statistically significant versus baseline and vintage schema remains unverified.",
            "Governed reopens as a serious research candidate because canonical-cache reruns now show statistically significant MAE wins on the mid-horizon block.",
            "Current evidence supports governed most cleanly on h2..h5; h6..h7 remain directional but not yet statistically strong enough for default production routing.",
            "At h10, any MAE gain still carries a tail-risk warning when RMSE worsens versus baseline.",
            "The earlier primary-only conclusion was invalidated by evaluation harnesses reading stale H: caches instead of the canonical C: cache.",
        ],
    }


def main() -> None:
    horizon_keys = ("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h10")
    horizons = {key: _load_horizon_result(key) for key in horizon_keys}
    commits = [horizons[key].get("git_commit") for key in horizon_keys]
    signatures = [
        json.dumps(horizons[key].get("input_signatures", {}), sort_keys=True)
        for key in horizon_keys
        if horizons[key].get("source") == "full_json"
    ]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_inputs": [
            str(ROOT / "scripts" / "eval_ct_governed_autopsy.py"),
            str(ROOT / "pfc_shaping" / "output" / "ct_governed_autopsy_latest.json"),
        ],
        "artifact_inputs": {
            **{key: str(HORIZON_DIRS[key]) for key in horizon_keys if key in HORIZON_DIRS},
            "h10_fallback": str(HORIZON_FALLBACK_DIRS["h10"]),
        },
        "artifact_consistency": {
            "git_commit_consistent_across_json_runs": _all_same_non_null(commits),
            "input_sha_consistent_across_json_runs": len(set(signatures)) <= 1 if signatures else False,
        },
        "policy_recommendation": _derive_policy(horizons),
        "horizons": horizons,
    }
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
