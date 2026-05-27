from scripts.write_ct_policy_decision import _derive_policy


def test_policy_reopens_governed_mid_horizon_when_h5_is_significant() -> None:
    h1 = {
        "source": "full_json",
        "winner_family": "governed_no_fm",
        "significant_winner_vs_baseline": False,
    }
    h5 = {
        "source": "full_json",
        "winner_family": "governed_no_fm",
        "significant_winner_vs_baseline": True,
    }
    h10 = {
        "source": "directional_parquet",
        "baseline": {"mae": 24.0, "rmse": 29.8},
        "governed": {"mae": 18.0, "rmse": 33.0},
    }

    policy = _derive_policy(h1, h5, h10)

    assert policy["prod_policy"] == "primary_only_conservative"
    assert policy["governed_prod_enabled_by_default"] is False
    assert policy["research_policy_candidate"] == "governed_mid_horizon_candidate"
    assert policy["research_candidate_window_days"] == [2, 7]
    assert policy["h1_governed_significant"] is False
    assert policy["h5_governed_significant"] is True
    assert policy["h10_directional_mae_only"] is True
