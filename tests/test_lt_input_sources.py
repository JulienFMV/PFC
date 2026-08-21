from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

import pfc_shaping.data.lt_input_sources as lt_input_sources_module
from pfc_shaping.data.lt_input_sources import (
    PFC_LT_DATA_ROOT_ENV,
    LTInputPaths,
    consumed_lt_input_roles,
    read_json_snapshot,
    read_parquet_snapshot,
    read_yaml_snapshot,
    resolve_lt_input_paths,
    validate_lt_input_consumption,
    verify_source_receipt,
)
from pfc_shaping.data.shared_data_root import (
    FMV_DATA_ROOT_ENV,
    PFC_SHARED_DATA_ROOT_ENV,
)
from scripts.create_lt_input_snapshot import create_snapshot


@pytest.fixture(autouse=True)
def _isolate_data_root_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (FMV_DATA_ROOT_ENV, PFC_LT_DATA_ROOT_ENV, PFC_SHARED_DATA_ROOT_ENV):
        monkeypatch.delenv(name, raising=False)


def _external_snapshot(
    tmp_path: Path,
    *,
    include_forward_history: bool = False,
) -> tuple[Path, Path]:
    source = tmp_path / "source"
    root = tmp_path / "shared"
    root.mkdir()
    for relative in (
        "market/epex/epex_15min.parquet",
        "market/epex/epex_de_15min.parquet",
        "entsoe/curated/entso_15min.parquet",
        "hydro/curated/hydro_reservoir.parquet",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("ascii"))
    if include_forward_history:
        history_path = source / "market" / "eex" / "eex_forwards_history.parquet"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-10")],
                "product": ["2027"],
                "load_type": ["BASE"],
                "product_type": ["Cal"],
                "price": [80.0],
                "market": ["CH"],
                "source": ["TEST"],
            }
        ).to_parquet(history_path, index=False)
    create_snapshot(
        source_root=source,
        data_root=root,
        generation_id="generation-001",
        available_at="2026-07-13T10:00:00Z",
        provenance="TEST",
    )
    return source, root


def test_explicit_external_root_uses_reusable_layout(tmp_path: Path) -> None:
    _, root = _external_snapshot(tmp_path)
    paths = resolve_lt_input_paths(tmp_path / "project", data_root=root)

    assert paths.layout == "external_v2"
    assert paths.generation_id == "generation-001"
    assert paths.calibration_eligible is False
    assert paths.entso == root / "snapshots" / "generation-001" / "entsoe" / "curated" / "entso_15min.parquet"
    assert paths.epex_ch == root / "snapshots" / "generation-001" / "market" / "epex" / "epex_15min.parquet"


def test_explicit_snapshot_binding_accepts_only_exact_pointer_contract_and_generation(
    tmp_path: Path,
) -> None:
    _, root = _external_snapshot(tmp_path)
    pointer = root / "views" / "pfc_lt" / "current.json"
    contract = root / "snapshots" / "generation-001" / "lt_input_snapshot.json"

    paths = resolve_lt_input_paths(
        tmp_path / "project",
        data_root=root,
        expected_snapshot_contract=contract,
        expected_pointer_contract=pointer,
        expected_snapshot_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        expected_pointer_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(),
        expected_generation_id="generation-001",
    )

    assert paths.generation_id == "generation-001"
    assert paths.contract_receipt is not None
    assert paths.contract_receipt.path == str(contract.resolve())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected_snapshot_sha256": "a" * 64}, "snapshot SHA-256"),
        ({"expected_pointer_sha256": "a" * 64}, "pointer SHA-256"),
        ({"expected_generation_id": "generation-other"}, "generation_id"),
    ],
)
def test_explicit_snapshot_binding_rejects_identity_mismatch(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    _, root = _external_snapshot(tmp_path)
    pointer = root / "views" / "pfc_lt" / "current.json"
    contract = root / "snapshots" / "generation-001" / "lt_input_snapshot.json"
    binding: dict[str, object] = {
        "expected_snapshot_contract": contract,
        "expected_pointer_contract": pointer,
        "expected_snapshot_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "expected_pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
        "expected_generation_id": "generation-001",
    }
    binding.update(override)

    with pytest.raises(ValueError, match=message):
        resolve_lt_input_paths(tmp_path / "project", data_root=root, **binding)


def test_explicit_snapshot_binding_requires_complete_identity(tmp_path: Path) -> None:
    _, root = _external_snapshot(tmp_path)
    contract = root / "snapshots" / "generation-001" / "lt_input_snapshot.json"

    with pytest.raises(ValueError, match="requires contract, hashes and generation"):
        resolve_lt_input_paths(
            tmp_path / "project",
            data_root=root,
            expected_snapshot_contract=contract,
        )


def test_snapshot_pointer_rejects_duplicate_root_key(tmp_path: Path) -> None:
    _, root = _external_snapshot(tmp_path)
    pointer = root / "views" / "pfc_lt" / "current.json"
    payload = pointer.read_text(encoding="utf-8")
    pointer.write_text(
        '{"generation_id":"forged",' + payload.lstrip()[1:],
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: generation_id"):
        resolve_lt_input_paths(tmp_path / "project", data_root=root)


def test_snapshot_contract_rejects_duplicate_nested_key(tmp_path: Path) -> None:
    _, root = _external_snapshot(tmp_path)
    contract = root / "snapshots" / "generation-001" / "lt_input_snapshot.json"
    payload = contract.read_text(encoding="utf-8")
    contract.write_text(
        payload.replace('"path":', '"path":"forged","path":', 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: path"):
        resolve_lt_input_paths(tmp_path / "project", data_root=root)


def test_json_snapshot_rejects_non_finite_number(tmp_path: Path) -> None:
    artifact = tmp_path / "contract.json"
    artifact.write_text('{"available_at_utc":NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON number"):
        read_json_snapshot(artifact, role="contract")


def test_yaml_snapshot_rejects_duplicate_key(tmp_path: Path) -> None:
    artifact = tmp_path / "config.yaml"
    artifact.write_text("model:\n  enabled: false\n  enabled: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate YAML key: enabled"):
        read_yaml_snapshot(artifact)


def test_archived_pointer_replays_snapshot_after_current_rotation(tmp_path: Path) -> None:
    source, root = _external_snapshot(tmp_path)
    current = root / "views" / "pfc_lt" / "current.json"
    archived_pointer = tmp_path / "archived" / "current.json"
    archived_pointer.parent.mkdir()
    archived_pointer.write_bytes(current.read_bytes())
    contract = root / "snapshots" / "generation-001" / "lt_input_snapshot.json"
    create_snapshot(
        source_root=source,
        data_root=root,
        generation_id="generation-002",
        available_at="2026-07-14T10:00:00Z",
        provenance="TEST",
    )

    paths = resolve_lt_input_paths(
        tmp_path / "project",
        data_root=root,
        expected_snapshot_contract=contract,
        expected_pointer_contract=archived_pointer,
        expected_snapshot_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        expected_pointer_sha256=hashlib.sha256(
            archived_pointer.read_bytes()
        ).hexdigest(),
        expected_generation_id="generation-001",
    )

    assert paths.generation_id == "generation-001"
    assert paths.pointer_receipt is not None
    assert paths.pointer_receipt.path == str(archived_pointer.resolve())


def test_environment_root_has_priority_over_user_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, shared = _external_snapshot(tmp_path)
    monkeypatch.setenv(PFC_LT_DATA_ROOT_ENV, str(shared))

    paths = resolve_lt_input_paths(tmp_path / "project")

    assert paths.root == shared.resolve()
    assert paths.layout == "external_v2"


def test_generic_environment_root_selects_lt_consumer_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, shared = _external_snapshot(tmp_path)
    monkeypatch.delenv(PFC_LT_DATA_ROOT_ENV, raising=False)
    monkeypatch.setenv(FMV_DATA_ROOT_ENV, str(shared))

    paths = resolve_lt_input_paths(tmp_path / "project")

    assert paths.pointer_receipt is not None
    assert paths.pointer_receipt.logical_path == "views/pfc_lt/current.json"


def test_historical_shared_environment_alias_selects_lt_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, shared = _external_snapshot(tmp_path)
    monkeypatch.delenv(FMV_DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(PFC_LT_DATA_ROOT_ENV, raising=False)
    monkeypatch.setenv(PFC_SHARED_DATA_ROOT_ENV, str(shared))

    paths = resolve_lt_input_paths(tmp_path / "project")

    assert paths.root == shared.resolve()


def test_legacy_root_pointer_remains_readable_during_transition(tmp_path: Path) -> None:
    _, shared = _external_snapshot(tmp_path)
    canonical = shared / "views" / "pfc_lt" / "current.json"
    (shared / "current.json").write_bytes(canonical.read_bytes())
    canonical.unlink()

    paths = resolve_lt_input_paths(tmp_path / "project", data_root=shared)

    assert paths.pointer_receipt is not None
    assert paths.pointer_receipt.logical_path == "current.json"


def test_consumer_view_is_authoritative_over_legacy_pointer(tmp_path: Path) -> None:
    _, shared = _external_snapshot(tmp_path)
    (shared / "current.json").write_text("{}", encoding="utf-8")

    paths = resolve_lt_input_paths(tmp_path / "project", data_root=shared)

    assert paths.generation_id == "generation-001"
    assert paths.pointer_receipt is not None
    assert paths.pointer_receipt.logical_path == "views/pfc_lt/current.json"


def test_lt_reader_rejects_linked_canonical_pointer(tmp_path: Path) -> None:
    _, shared = _external_snapshot(tmp_path)
    canonical = shared / "views" / "pfc_lt" / "current.json"
    outside = tmp_path / "outside-current.json"
    outside.write_bytes(canonical.read_bytes())
    canonical.unlink()
    try:
        os.symlink(outside, canonical)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="link or junction"):
        resolve_lt_input_paths(tmp_path / "project", data_root=shared)


def test_snapshot_writer_rejects_linked_view_before_publishing(tmp_path: Path) -> None:
    source, shared = _external_snapshot(tmp_path)
    canonical = shared / "views" / "pfc_lt" / "current.json"
    canonical.unlink()
    canonical.parent.rmdir()
    (shared / "views").rmdir()
    outside = tmp_path / "outside-views"
    outside.mkdir()
    try:
        os.symlink(outside, shared / "views", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="link or junction"):
        create_snapshot(
            source_root=source,
            data_root=shared,
            generation_id="generation-002",
            available_at="2026-07-14T10:00:00Z",
            provenance="TEST",
        )

    assert not (shared / "snapshots" / "generation-002").exists()


def test_external_snapshot_exposes_optional_governed_forward_history(
    tmp_path: Path,
) -> None:
    _, root = _external_snapshot(tmp_path, include_forward_history=True)

    paths = resolve_lt_input_paths(tmp_path / "project", data_root=root)

    assert paths.eex_forwards_history == (
        root
        / "snapshots"
        / "generation-001"
        / "market"
        / "eex"
        / "eex_forwards_history.parquet"
    )
    contract = json.loads(
        (root / "snapshots" / "generation-001" / "lt_input_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["files"]["eex_forwards_history"]["expected_cadence_seconds"] is None


def test_missing_external_root_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PFC_LT_DATA_ROOT_ENV, raising=False)

    with pytest.raises(ValueError, match="explicit FMV_DATA_ROOT"):
        resolve_lt_input_paths(tmp_path / "project")


def test_legacy_repo_layout_requires_explicit_opt_in_and_is_never_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PFC_LT_DATA_ROOT_ENV, raising=False)

    paths = resolve_lt_input_paths(
        tmp_path / "project",
        allow_legacy_repo=True,
    )

    assert paths.layout == "legacy_repo"
    assert paths.calibration_eligible is False


@pytest.mark.parametrize("value", ["", "relative/path"])
def test_environment_root_must_be_nonempty_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(PFC_LT_DATA_ROOT_ENV, value)
    with pytest.raises(ValueError):
        resolve_lt_input_paths(tmp_path / "project")


def test_parquet_receipt_represents_consumed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    expected = pd.DataFrame({"value": [1.0, 2.0]})
    expected.to_parquet(path)

    actual, receipt = read_parquet_snapshot(path, role="fixture")
    pd.testing.assert_frame_equal(actual, expected)
    verify_source_receipt(receipt)

    pd.DataFrame({"value": [9.0]}).to_parquet(path)
    with pytest.raises(ValueError, match="changed after consumption"):
        verify_source_receipt(receipt)


def _eligible_paths(
    *,
    role_overrides: dict[str, object] | None = None,
    schema_version: str | None = None,
) -> LTInputPaths:
    available_at = "2026-07-13T10:00:00+00:00"
    roles = {
        role: {
            "path": f"{role}.parquet",
            "calibration_eligible": True,
            "available_at_utc": available_at,
        }
        for role in ("epex_ch", "epex_de", "entso", "hydro", "epex_fr")
    }
    if role_overrides:
        roles["epex_fr"].update(role_overrides)
    root = Path("C:/governed-test-root")
    return LTInputPaths(
        root=root,
        snapshot_root=root / "snapshot",
        layout="external_v2",
        generation_id="generation-001",
        calibration_eligible=True,
        available_at_utc=available_at,
        files={role: root / f"{role}.parquet" for role in roles},
        expected_files=roles,
        schema_version=schema_version or "lt_input_snapshot.v3",
    )


@pytest.mark.parametrize("role", ["outages", "commodities"])
def test_v3_consumption_rejects_roles_without_replay_governance(role: str) -> None:
    paths = _eligible_paths()
    paths.expected_files[role] = {
        "path": f"{role}.parquet",
        "calibration_eligible": True,
        "available_at_utc": paths.available_at_utc,
    }

    with pytest.raises(ValueError, match="lack deterministic replay governance"):
        validate_lt_input_consumption(
            paths,
            roles={"epex_ch", "epex_de", "entso", "hydro", role},
            reference_timestamp="2026-07-13T11:00:00Z",
        )


def test_v2_consumption_is_archive_only_without_provider_raw_evidence() -> None:
    paths = _eligible_paths(schema_version="lt_input_snapshot.v2")

    with pytest.raises(ValueError, match="requires lt_input_snapshot.v3"):
        validate_lt_input_consumption(
            paths,
            roles={"epex_ch", "epex_de", "entso", "hydro"},
            reference_timestamp="2026-07-13T11:00:00Z",
        )


def test_consumed_optional_role_must_be_calibration_eligible_before_io() -> None:
    paths = _eligible_paths(role_overrides={"calibration_eligible": False})

    with pytest.raises(ValueError, match="not calibration eligible: epex_fr"):
        validate_lt_input_consumption(
            paths,
            roles={"epex_ch", "epex_de", "entso", "hydro", "epex_fr"},
            reference_timestamp="2026-07-13T11:00:00Z",
        )


def test_consumed_role_cannot_be_newer_than_snapshot_cutoff() -> None:
    paths = _eligible_paths(
        role_overrides={"available_at_utc": "2026-07-13T10:30:00Z"}
    )

    with pytest.raises(ValueError, match="newer than snapshot cutoff: epex_fr"):
        validate_lt_input_consumption(
            paths,
            roles={"epex_ch", "epex_de", "entso", "hydro", "epex_fr"},
            reference_timestamp="2026-07-13T11:00:00Z",
        )


def test_snapshot_must_be_available_before_valuation() -> None:
    paths = _eligible_paths()

    with pytest.raises(ValueError, match="not available at valuation"):
        validate_lt_input_consumption(
            paths,
            roles={"epex_ch", "epex_de", "entso", "hydro"},
            reference_timestamp="2026-07-13T09:59:59Z",
        )


@pytest.mark.parametrize(
    "config",
    [
        {"quality": {"freshness": {"outages_enabled": 1}}},
        {"forwards": {"monthly_curve_solver": {"enabled": 1}}},
    ],
)
def test_consumed_roles_reject_truthy_non_boolean_flags(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        consumed_lt_input_roles(config, available_roles=set())


@pytest.mark.parametrize(
    ("config", "label"),
    [
        ({"forwards": "disabled"}, "forwards"),
        ({"forwards": {"monthly_curve_solver": []}}, "forwards.monthly_curve_solver"),
        ({"quality": False}, "quality"),
        ({"quality": {"freshness": "default"}}, "quality.freshness"),
    ],
)
def test_consumed_roles_reject_malformed_config_sections(
    config: dict[str, object],
    label: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{label} must be a mapping"):
        consumed_lt_input_roles(config, available_roles=set())


def test_quality_reader_rejects_hydro_row_bomb_before_dataframe_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"fill_pct": 50.0},
        index=pd.date_range("2000-01-03", periods=5_001, freq="W-MON", tz="UTC"),
    )
    buffer = BytesIO()
    frame.to_parquet(buffer, index=True)
    monkeypatch.setattr(
        lt_input_sources_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DataFrame allocation must not occur")
        ),
    )

    with pytest.raises(ValueError, match="parquet row budget is exceeded"):
        lt_input_sources_module._quality_frame_metrics(
            buffer.getvalue(),
            role="hydro",
            label="derived",
        )


def test_quality_reader_rejects_repeated_numeric_parquet_before_dataframe_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"nested": [[value for value in range(10_000)]]})
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    monkeypatch.setattr(
        lt_input_sources_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DataFrame allocation must not occur")
        ),
    )

    with pytest.raises(ValueError, match="flat and non-repeated"):
        lt_input_sources_module._quality_frame_metrics(
            buffer.getvalue(),
            role="hydro",
            label="derived",
        )
