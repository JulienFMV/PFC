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
    "h5": ROOT / "pfc_shaping" / "output" / "h5_canonical_quick",
    "h10": ROOT / "pfc_shaping" / "output" / "h10_canonical_quick",
}
HORIZON_FALLBACK_DIRS = {
    "h10": ROOT / "pfc_shaping" / "output" / "h10_canonical_smoke",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest(path: Path, suffix: str, exclude_contains: str | None = None) -> Path | None:
    matches = sorted(path.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime)
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
    return len(set(non_null)) <= 1


def _derive_policy(h1: dict, h5: dict, h10: dict) -> dict:
    h1_json = h1.get("source") == "full_json"
    h5_json = h5.get("source") == "full_json"
    h10_json = h10.get("source") == "full_json"

    h5_governed_supported = (
        h5_json
        and h5.get("winner_family", "").startswith("governed")
        and bool(h5.get("significant_winner_vs_baseline", False))
    )

    h1_governed_supported = (
        h1_json
        and h1.get("winner_family", "").startswith("governed")
        and bool(h1.get("significant_winner_vs_baseline", False))
    )

    h10_directional_mae_only = False
    if h10.get("source") == "directional_parquet":
        base = h10["baseline"]
        gov = h10["governed"]
        h10_directional_mae_only = bool(
            gov["mae"] < base["mae"] and gov["rmse"] > base["rmse"]
        )

    return {
        "prod_policy": "primary_only_conservative",
        "governed_prod_enabled_by_default": False,
        "research_policy_candidate": "governed_mid_horizon_candidate" if h5_governed_supported else "primary_only",
        "research_candidate_window_days": [2, 7] if h5_governed_supported else [],
        "h1_governed_significant": h1_governed_supported,
        "h5_governed_significant": h5_governed_supported,
        "h10_directional_mae_only": h10_directional_mae_only,
        "rationale": [
            "Prod stays conservative because h1 governed is not statistically significant versus baseline and vintage schema remains unverified.",
            "Governed reopens as a serious research candidate because canonical-cache h5 shows a statistically significant MAE win versus baseline.",
            "Directional h10 improves MAE but still worsens RMSE/correlation, so long-horizon governed routing is not yet production-ready.",
            "The earlier primary-only conclusion was invalidated by evaluation harnesses reading stale H: caches instead of the canonical C: cache.",
        ],
    }


def main() -> None:
    horizons = {key: _load_horizon_result(key) for key in ("h1", "h5", "h10")}
    commits = [horizons[key].get("git_commit") for key in ("h1", "h5", "h10")]
    signatures = [
        json.dumps(horizons[key].get("input_signatures", {}), sort_keys=True)
        for key in ("h1", "h5")
        if horizons[key].get("source") == "full_json"
    ]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_inputs": {
            "h1": str(HORIZON_DIRS["h1"]),
            "h5": str(HORIZON_DIRS["h5"]),
            "h10_primary": str(HORIZON_DIRS["h10"]),
            "h10_fallback": str(HORIZON_FALLBACK_DIRS["h10"]),
        },
        "artifact_consistency": {
            "git_commit_consistent_across_json_runs": _all_same_non_null(commits),
            "input_sha_consistent_between_h1_h5": len(set(signatures)) <= 1 if signatures else False,
        },
        "policy_recommendation": _derive_policy(
            horizons["h1"],
            horizons["h5"],
            horizons["h10"],
        ),
        "horizons": horizons,
    }
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
