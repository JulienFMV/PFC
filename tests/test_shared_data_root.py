from __future__ import annotations

from pathlib import Path

import pytest

from pfc_shaping.data.shared_data_root import (
    FMV_DATA_ROOT_ENV,
    PFC_ENTSOE_DATA_ROOT_ENV,
    PFC_LT_DATA_ROOT_ENV,
    PFC_SHARED_DATA_ROOT_ENV,
    configured_lt_data_root,
    resolve_data_domain_root,
    resolve_fmv_data_root,
)


def test_canonical_fmv_root_is_consumer_neutral(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()

    actual = resolve_fmv_data_root(
        environ={FMV_DATA_ROOT_ENV: str(root)},
        require_exists=True,
    )

    assert actual == root.resolve()


def test_matching_legacy_alias_is_accepted_during_transition(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()

    actual = resolve_fmv_data_root(
        compatibility_envs=(PFC_SHARED_DATA_ROOT_ENV, PFC_LT_DATA_ROOT_ENV),
        environ={
            FMV_DATA_ROOT_ENV: str(root),
            PFC_SHARED_DATA_ROOT_ENV: str(root),
            PFC_LT_DATA_ROOT_ENV: str(root),
        },
    )

    assert actual == root.resolve()


def test_divergent_legacy_alias_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    with pytest.raises(ValueError, match="configured FMV data roots diverge"):
        resolve_fmv_data_root(
            compatibility_envs=(PFC_SHARED_DATA_ROOT_ENV,),
            environ={
                FMV_DATA_ROOT_ENV: str(first),
                PFC_SHARED_DATA_ROOT_ENV: str(second),
            },
        )


def test_explicit_root_does_not_depend_on_process_aliases(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"

    actual = resolve_fmv_data_root(
        explicit,
        environ={FMV_DATA_ROOT_ENV: str(tmp_path / "other")},
    )

    assert actual == explicit.resolve()


def test_domain_root_is_confined_under_generic_root(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    entsoe = root / "entsoe"
    entsoe.mkdir(parents=True)

    actual = resolve_data_domain_root(
        "entsoe",
        environ={FMV_DATA_ROOT_ENV: str(root)},
    )

    assert actual == entsoe.resolve()


def test_direct_domain_alias_must_name_the_domain(tmp_path: Path) -> None:
    wrong = tmp_path / "shared" / "not-entsoe"
    wrong.mkdir(parents=True)

    with pytest.raises(ValueError, match="must end with 'entsoe'"):
        resolve_data_domain_root(
            "entsoe",
            domain_env=PFC_ENTSOE_DATA_ROOT_ENV,
            environ={PFC_ENTSOE_DATA_ROOT_ENV: str(wrong)},
        )


def test_parent_and_direct_domain_roots_must_match(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    (root / "entsoe").mkdir(parents=True)
    other = tmp_path / "other" / "entsoe"
    other.mkdir(parents=True)

    with pytest.raises(ValueError, match="do not identify the same store"):
        resolve_data_domain_root(
            "entsoe",
            data_root=root,
            domain_root=other,
        )


def test_lt_cli_accepts_historical_shared_root_alias(tmp_path: Path) -> None:
    root = tmp_path / "shared"

    actual = configured_lt_data_root(
        environ={PFC_SHARED_DATA_ROOT_ENV: str(root)},
    )

    assert actual == root
