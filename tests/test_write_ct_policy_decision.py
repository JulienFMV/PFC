from pathlib import Path

from scripts.write_ct_policy_decision import _all_same_non_null, _derive_policy, _find_latest


def test_policy_reopens_governed_mid_horizon_when_h5_is_significant() -> None:
    horizons = {
        "h1": {
            "source": "full_json",
            "winner_family": "governed_no_fm",
            "significant_winner_vs_baseline": False,
        },
        "h2": {
            "source": "full_json",
            "winner_family": "governed_no_fm",
            "significant_winner_vs_baseline": True,
        },
        "h3": {
            "source": "full_json",
            "winner_family": "governed_no_fm",
            "significant_winner_vs_baseline": True,
        },
        "h4": {
            "source": "full_json",
            "winner_family": "governed_no_fm",
            "significant_winner_vs_baseline": True,
        },
        "h5": {
            "source": "full_json",
            "winner_family": "governed_no_fm",
            "significant_winner_vs_baseline": True,
        },
        "h6": {
            "source": "full_json",
            "winner_family": "governed_no_fm",
            "significant_winner_vs_baseline": False,
        },
        "h7": {
            "source": "full_json",
            "winner_family": "governed_no_fm",
            "significant_winner_vs_baseline": False,
        },
        "h10": {
            "source": "directional_parquet",
            "baseline": {"mae": 24.0, "rmse": 29.8},
            "governed": {"mae": 18.0, "rmse": 33.0},
        },
    }

    policy = _derive_policy(horizons)

    assert policy["prod_policy"] == "primary_only_conservative"
    assert policy["governed_prod_enabled_by_default"] is False
    assert policy["research_policy_candidate"] == "governed_mid_horizon_candidate"
    assert policy["research_candidate_window_days"] == [2, 3, 4, 5]
    assert policy["research_candidate_window_runs"] == [[2, 3, 4, 5]]
    assert policy["governed_directional_window_days"] == [1, 2, 3, 4, 5, 6, 7]
    assert policy["h1_governed_significant"] is False
    assert policy["h5_governed_significant"] is True
    assert policy["h10_tail_risk_red_flag"] is True


def test_all_same_non_null_requires_at_least_two_non_null_values() -> None:
    assert _all_same_non_null([None, None, None]) is False
    assert _all_same_non_null(["abc", None, None]) is False
    assert _all_same_non_null(["abc", "abc", None]) is True
    assert _all_same_non_null(["abc", "def", None]) is False


def test_find_latest_prefers_lexicographically_latest_run_id(tmp_path: Path) -> None:
    older = tmp_path / "lear_governed_forecast_ab_20260527_080309.json"
    newer = tmp_path / "lear_governed_forecast_ab_20260527_123133.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    assert _find_latest(tmp_path, ".json") == newer
