#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = ROOT / "pfc_shaping" / "output" / "horizon_winners_ref_fixed"
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "ct_policy_decision_latest.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_single(path: Path, suffix: str, exclude_contains: str | None = None) -> Path:
    matches = sorted(path.glob(f"*{suffix}"))
    if exclude_contains:
        matches = [m for m in matches if exclude_contains not in m.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {suffix} in {path}, found {len(matches)}")
    return matches[0]


def main() -> None:
    input_root = DEFAULT_INPUT_ROOT
    horizons = {}
    for horizon in ("h1", "h5", "h10"):
        directory = input_root / horizon
        winners_path = _find_single(directory, "_horizon_winners.json")
        full_path = _find_single(directory, ".json", exclude_contains="_horizon_winners")
        winners_payload = _load_json(winners_path)[horizon]
        full_payload = _load_json(full_path)
        horizons[horizon] = {
            "winners_path": str(winners_path),
            "full_path": str(full_path),
            "winner": winners_payload["winner_by_mae"],
            "winner_family": winners_payload["winner_family"],
            "recommended_routing_action": winners_payload["recommended_routing_action"],
            "significant_vs_second_best": winners_payload["significant_vs_second_best"],
            "summary": full_payload["summary"][horizon],
        }

    h1 = horizons["h1"]
    h5 = horizons["h5"]
    h10 = horizons["h10"]
    primary_only_recommended = (
        h1["recommended_routing_action"] == "hold"
        and h5["winner_family"].startswith("primary")
        and h10["winner_family"].startswith("primary")
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "policy_recommendation": {
            "prod_policy": "primary_only",
            "primary_only_recommended": primary_only_recommended,
            "governed_prod_enabled_by_default": False,
            "foundation_short_horizon_only": True,
            "rationale": [
                "h1 winner is fm_only but not significant versus fm_governed",
                "h5 winner is primary baseline and governed underperforms",
                "h10 winner is primary baseline and governed materially underperforms",
                "vintage schema remains unverified for governed datasets",
            ],
        },
        "horizons": horizons,
    }
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
