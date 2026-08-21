from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pfc_shaping.data import forward_proxy


def _spot() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=96 * 10, freq="15min", tz="UTC")
    return pd.DataFrame({"price_eur_mwh": 80.0}, index=index)


def test_strict_loader_never_derives_spot_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("derive_base_prices must not run in strict mode")

    monkeypatch.setattr(forward_proxy, "derive_base_prices", forbidden)

    with pytest.raises(forward_proxy.ForwardSourceUnavailableError, match="proxy is forbidden"):
        forward_proxy.load_base_prices(_spot(), config={}, allow_spot_proxy=False)


def test_research_loader_keeps_explicit_spot_proxy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forward_proxy, "derive_base_prices", lambda *args, **kwargs: {"2027": 81.0})

    prices, source = forward_proxy.load_base_prices(_spot(), config={}, allow_spot_proxy=True)

    assert prices == {"2027": 81.0}
    assert source == "Spot-derived proxy (1 keys)"


def _write_workbook(tmp_path: Path, *, snapshot_date: str = "2026-07-10") -> Path:
    report = tmp_path / f"eex-{snapshot_date}.xlsx"
    rows = [
        [None, "Y01_2027_BASE", "Y01_2028_BASE", "Y01_2029_BASE"],
        [None, "ISIN-1", "ISIN-2", "ISIN-3"],
        ["Date", None, None, None],
        [pd.Timestamp(snapshot_date).strftime("%d.%m.%Y"), 81.0, 79.0, 77.0],
    ]
    with pd.ExcelWriter(report, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="CH", index=False, header=False)
    return report


def _snapshot(tmp_path: Path, *, snapshot_date: str = "2026-07-10") -> forward_proxy.ForwardSnapshot:
    report = _write_workbook(tmp_path, snapshot_date=snapshot_date)
    observed_at = pd.Timestamp(snapshot_date, tz="UTC") + pd.Timedelta(hours=12)
    with patch(
        "pfc_shaping.data.forward_proxy._observation_timestamp_utc",
        return_value=observed_at,
    ):
        return forward_proxy.load_forward_snapshot(
            _spot(),
            eex_report_path=str(report),
            config={},
            market="CH",
            allow_spot_proxy=False,
        )


def test_strict_loader_accepts_eex_xlsx(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, snapshot_date="2026-07-13")

    assert snapshot.prices["2027"] == 81.0
    assert snapshot.source_description.startswith("EEX XLSX CH")
    assert snapshot.source_sha256 == hashlib.sha256(Path(snapshot.source_path).read_bytes()).hexdigest()
    assert snapshot.hard_quote_eligible is True


def test_forward_snapshot_is_immutable_and_manifested(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    with pytest.raises(TypeError):
        snapshot.prices["2028"] = 79.0
    with pytest.raises(TypeError):
        snapshot.quote_lineage["2027"]["source_product_code"] = "tampered"

    manifest = snapshot.to_manifest()
    assert manifest["schema_version"] == "forward_snapshot.v1"
    assert manifest["snapshot_date"] == "2026-07-10T00:00:00+00:00"
    assert manifest["source_sha256"] == hashlib.sha256(Path(snapshot.source_path).read_bytes()).hexdigest()
    assert manifest["hard_quote_eligible"] is True
    assert len(manifest["quote_set_sha256"]) == 64
    assert len(manifest["snapshot_id"]) == 64


def test_snapshot_identity_is_stable_across_observation_times(tmp_path: Path) -> None:
    report = _write_workbook(tmp_path)
    first = forward_proxy.load_forward_snapshot(
        _spot(), eex_report_path=str(report), config={}, market="CH", allow_spot_proxy=False
    )
    replay = forward_proxy.load_forward_snapshot(
        _spot(), eex_report_path=str(report), config={}, market="CH", allow_spot_proxy=False
    )

    assert first.snapshot_id == replay.snapshot_id
    assert first.observation_id != replay.observation_id


def test_governed_snapshot_replay_uses_authenticated_availability(tmp_path: Path) -> None:
    report = _write_workbook(tmp_path)
    available_at = "2026-07-10T12:00:00+00:00"
    with patch(
        "pfc_shaping.data.forward_proxy._observation_timestamp_utc",
        return_value=pd.Timestamp("2026-07-11T00:00:00Z"),
    ):
        first = forward_proxy.load_forward_snapshot(
            _spot(),
            eex_report_path=str(report),
            config={"databricks": {"enabled": True}},
            market="CH",
            allow_spot_proxy=False,
            exact_source_only=True,
            source_available_at=available_at,
        )
    with patch(
        "pfc_shaping.data.forward_proxy._observation_timestamp_utc",
        return_value=pd.Timestamp("2027-01-01T00:00:00Z"),
    ):
        replay = forward_proxy.load_forward_snapshot(
            _spot(),
            eex_report_path=str(report),
            config={"databricks": {"enabled": True}},
            market="CH",
            allow_spot_proxy=False,
            exact_source_only=True,
            source_available_at=available_at,
        )

    assert first.to_manifest() == replay.to_manifest()
    assert first.available_at == pd.Timestamp(available_at)


def test_governed_snapshot_requires_authenticated_availability(tmp_path: Path) -> None:
    report = _write_workbook(tmp_path)

    with pytest.raises(
        forward_proxy.ForwardSourceUnavailableError,
        match="authenticated source availability",
    ):
        forward_proxy.load_forward_snapshot(
            _spot(),
            eex_report_path=str(report),
            config={"databricks": {"enabled": True}},
            market="CH",
            allow_spot_proxy=False,
            exact_source_only=True,
        )


def test_governed_invalid_source_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _write_workbook(tmp_path)

    def invalid_source(*args: object, **kwargs: object) -> object:
        raise ValueError("invalid governed workbook")

    def forbidden_fallback(*args: object, **kwargs: object) -> object:
        raise AssertionError("a governed source must be terminal")

    monkeypatch.setattr(forward_proxy, "_load_eex_workbook_snapshot", invalid_source)
    monkeypatch.setattr(forward_proxy, "derive_base_prices", forbidden_fallback)

    with pytest.raises(
        forward_proxy.ForwardSourceUnavailableError,
        match="authenticated EEX source is invalid",
    ):
        forward_proxy.load_forward_snapshot(
            _spot(),
            eex_report_path=str(report),
            config={"databricks": {"enabled": True}},
            market="CH",
            allow_spot_proxy=True,
            exact_source_only=True,
            source_available_at="2026-07-10T12:00:00+00:00",
        )


def test_installed_runtime_rejects_legacy_forward_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forward_proxy, "SOURCE_REVISION", "a" * 64)

    with pytest.raises(
        forward_proxy.ForwardSourceUnavailableError,
        match="exact_source_only",
    ):
        forward_proxy.load_forward_snapshot(
            _spot(),
            config={"databricks": {"enabled": True}},
            market="CH",
            allow_spot_proxy=True,
        )


def test_hard_quote_eligibility_cannot_be_forged() -> None:
    with pytest.raises(TypeError, match="hard_quote_eligible"):
        forward_proxy.ForwardSnapshot(
            prices={"2027": 81.0},
            market="CH",
            source_kind="SPOT_PROXY",
            source_description="proxy",
            snapshot_date="2026-07-10",
            available_at="2026-07-10",
            source_path="C:/fake",
            source_sha256="a" * 64,
            quote_lineage={},
            hard_quote_eligible=True,
        )


def test_manual_price_substitution_cannot_become_hard_eligible(tmp_path: Path) -> None:
    verified = _snapshot(tmp_path)
    lineage = {
        key: {
            name: value
            for name, value in dict(verified.quote_lineage[key]).items()
            if name != "quote_id"
        }
        for key in verified.quote_lineage
    }
    prices = dict(verified.prices)
    prices["2027"] += 10.0
    forged = forward_proxy.ForwardSnapshot(
        prices=prices,
        market=verified.market,
        source_kind=verified.source_kind,
        source_description=verified.source_description,
        snapshot_date=verified.snapshot_date,
        available_at=verified.available_at,
        source_path=verified.source_path,
        source_sha256=verified.source_sha256,
        quote_lineage=lineage,
        source_sheet=verified.source_sheet,
    )

    assert forged.hard_quote_eligible is False
    with pytest.raises(forward_proxy.ForwardSourceUnavailableError, match="not eligible"):
        forward_proxy.validate_forward_snapshot(
            forged,
            reference_timestamp="2026-07-13T20:00:00Z",
        )


def test_forward_snapshot_rejects_naive_available_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        forward_proxy.ForwardSnapshot(
            prices={"2027": 81.0}, market="CH", source_kind="EEX_XLSX",
            source_description="manual", snapshot_date="2026-07-10",
            available_at="2026-07-10 08:00:00", source_path="C:/fake",
            source_sha256="a" * 64, quote_lineage={}, source_sheet="CH",
        )


def test_forward_snapshot_gate_accepts_friday_on_monday(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    report = forward_proxy.validate_forward_snapshot(
        snapshot,
        reference_timestamp=pd.Timestamp("2026-07-13 23:59", tz="Europe/Zurich"),
        max_age_business_days=1,
    )
    assert report["business_age_days"] == 1
    assert report["schema_version"] == "forward_eligibility.v1"
    assert len(report["eligibility_id"]) == 64


def test_forward_eligibility_cannot_be_replayed_at_another_valuation(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    reference = snapshot.available_at + pd.Timedelta(hours=1)
    receipt = forward_proxy.validate_forward_snapshot(
        snapshot,
        reference_timestamp=reference,
    )

    with pytest.raises(ValueError, match="reference_timestamp"):
        forward_proxy.verify_forward_eligibility(
            snapshot,
            receipt,
            valuation_timestamp=reference + pd.Timedelta(hours=1),
            expected_max_age_business_days=1,
        )


def test_forward_eligibility_cannot_relax_production_freshness_policy(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    reference = snapshot.available_at + pd.Timedelta(hours=1)
    receipt = forward_proxy.validate_forward_snapshot(
        snapshot,
        reference_timestamp=reference,
        max_age_business_days=100,
    )

    with pytest.raises(ValueError, match="freshness policy"):
        forward_proxy.verify_forward_eligibility(
            snapshot,
            receipt,
            valuation_timestamp=reference,
            expected_max_age_business_days=1,
        )


def test_forward_snapshot_validation_hashes_and_parses_one_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    target = Path(snapshot.source_path).resolve()
    original_stable_read = forward_proxy.read_stable_single_link_file
    reads = 0

    def counted_stable_read(path: Path, **kwargs) -> bytes:
        nonlocal reads
        if path.resolve() == target:
            reads += 1
        return original_stable_read(path, **kwargs)

    monkeypatch.setattr(
        forward_proxy,
        "read_stable_single_link_file",
        counted_stable_read,
    )

    forward_proxy.validate_forward_snapshot(
        snapshot,
        reference_timestamp="2026-07-13T20:00:00Z",
    )

    assert reads == 1


def test_forward_manifest_binding_recomputes_all_identity_fields(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    receipt = forward_proxy.validate_forward_snapshot(
        snapshot,
        reference_timestamp="2026-07-13T20:00:00Z",
    )

    binding = forward_proxy.verify_forward_manifest_binding(
        snapshot.to_manifest(),
        receipt.to_manifest(),
        expected_max_age_business_days=1,
    )

    assert binding["snapshot_id"] == snapshot.snapshot_id
    assert binding["eligibility_id"] == receipt.eligibility_id


def test_forward_manifest_binding_rejects_coherent_string_substitution(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    receipt = forward_proxy.validate_forward_snapshot(
        snapshot,
        reference_timestamp="2026-07-13T20:00:00Z",
    )
    manifest = snapshot.to_manifest()
    manifest["observation_id"] = "0" * 64

    with pytest.raises(ValueError, match="observation_id mismatch"):
        forward_proxy.verify_forward_manifest_binding(
            manifest,
            receipt.to_manifest(),
            expected_max_age_business_days=1,
        )


def test_forward_manifest_binding_rejects_load_type_mutation(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    receipt = forward_proxy.validate_forward_snapshot(
        snapshot,
        reference_timestamp="2026-07-13T20:00:00Z",
    )
    manifest = snapshot.to_manifest()
    manifest["quotes"][0]["load_type"] = "PEAK"

    with pytest.raises(ValueError, match="load_type"):
        forward_proxy.verify_forward_manifest_binding(
            manifest,
            receipt.to_manifest(),
            expected_max_age_business_days=1,
        )


def test_forward_snapshot_gate_rehashes_source_path(tmp_path: Path) -> None:
    missing = _snapshot(tmp_path)
    Path(missing.source_path).unlink()

    with pytest.raises(forward_proxy.ForwardSourceUnavailableError, match="does not exist"):
        forward_proxy.validate_forward_snapshot(
            missing,
            reference_timestamp="2026-07-13T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("snapshot_date", "reference", "reason"),
    [
        ("2026-07-14", "2026-07-13T23:59:00Z", "future"),
        ("2026-07-08", "2026-07-13T23:59:00Z", "stale"),
    ],
)
def test_forward_snapshot_gate_fails_closed(tmp_path, snapshot_date, reference, reason) -> None:
    snapshot = _snapshot(tmp_path, snapshot_date=snapshot_date)
    with pytest.raises(forward_proxy.ForwardSourceUnavailableError, match=reason):
        forward_proxy.validate_forward_snapshot(
            snapshot,
            reference_timestamp=reference,
            max_age_business_days=1,
        )
