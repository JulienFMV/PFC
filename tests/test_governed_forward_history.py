from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.data.lt_input_sources import InputSourceReceipt, dataframe_sha256
from pfc_shaping.pipeline.production_phases import _prepare_governed_forward_history


def _fixture(tmp_path: Path) -> tuple[pd.DataFrame, InputSourceReceipt, Path]:
    path = tmp_path / "eex_forwards_history.parquet"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-09"]),
            "product": ["2027", "2027"],
            "load_type": ["BASE", "BASE"],
            "product_type": ["Cal", "Cal"],
            "price": [80.0, 81.0],
            "market": ["CH", "CH"],
            "source": ["EEX", "EEX"],
        }
    )
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = InputSourceReceipt(
        role="eex_forwards_history",
        path=str(path.resolve()),
        logical_path="market/eex/eex_forwards_history.parquet",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        rows=len(frame),
        frame_sha256=dataframe_sha256(frame),
    )
    return frame, receipt, path


def test_governed_forward_history_binds_consumed_bytes_and_frame(tmp_path: Path) -> None:
    frame, receipt, _ = _fixture(tmp_path)

    actual, source_hash = _prepare_governed_forward_history(
        frame,
        receipt,
        reference_timestamp=pd.Timestamp("2026-07-13T00:00:00Z"),
    )

    assert source_hash == receipt.sha256
    assert actual["date"].max() == pd.Timestamp("2026-07-09")
    assert actual.attrs["eex_source_summary"] == [
        {
            "source": "EEX",
            "rows": 2,
            "date_min": "2026-07-08",
            "date_max": "2026-07-09",
            "markets": ["CH"],
        }
    ]


def test_governed_forward_history_rejects_future_observation(tmp_path: Path) -> None:
    frame, receipt, path = _fixture(tmp_path)
    frame.loc[1, "date"] = pd.Timestamp("2026-07-14")
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = replace(
        receipt,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_sha256=dataframe_sha256(frame),
    )

    with pytest.raises(ValueError, match="observations after valuation"):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T00:00:00Z"),
        )


def test_governed_forward_history_rejects_future_intraday_observation(
    tmp_path: Path,
) -> None:
    frame, receipt, path = _fixture(tmp_path)
    frame["date"] = frame["date"].astype("object")
    frame.loc[1, "date"] = pd.Timestamp("2026-07-13T18:00:00Z")
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = replace(
        receipt,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_sha256=dataframe_sha256(frame),
    )

    with pytest.raises(ValueError, match="observations after valuation"):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T10:00:00Z"),
        )


@pytest.mark.parametrize("source", ["OMPEX", "BENCHMARK"])
def test_governed_forward_history_rejects_non_eex_source(
    tmp_path: Path,
    source: str,
) -> None:
    frame, receipt, path = _fixture(tmp_path)
    frame.loc[1, "source"] = source
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = replace(
        receipt,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_sha256=dataframe_sha256(frame),
    )

    with pytest.raises(ValueError, match="non-EEX sources"):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T23:00:00Z"),
        )


def test_governed_forward_history_requires_source_column(tmp_path: Path) -> None:
    frame, receipt, path = _fixture(tmp_path)
    frame = frame.drop(columns="source")
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = replace(
        receipt,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_sha256=dataframe_sha256(frame),
    )

    with pytest.raises(ValueError, match="missing columns.*source"):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T23:00:00Z"),
        )


def test_governed_forward_history_canonicalizes_before_duplicate_check(
    tmp_path: Path,
) -> None:
    frame, receipt, path = _fixture(tmp_path)
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame.loc[1, "market"] = " ch "
    frame.loc[1, "load_type"] = "base"
    frame.loc[1, "product"] = " 2027 "
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = replace(
        receipt,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_sha256=dataframe_sha256(frame),
    )

    with pytest.raises(ValueError, match="duplicate quote identities"):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T23:00:00Z"),
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("market", "XX", "unsupported market"),
        ("load_type", "INTRADAY", "unsupported load_type"),
        ("product_type", "BANANA", "unsupported product_type"),
        ("product", "NOT_A_PRODUCT", "invalid product"),
        ("product_type", "Month", "product/product_type mismatch"),
    ],
)
def test_governed_forward_history_rejects_invalid_product_taxonomy(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    frame, receipt, path = _fixture(tmp_path)
    frame.loc[1, column] = value
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = replace(
        receipt,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_sha256=dataframe_sha256(frame),
    )

    with pytest.raises(ValueError, match=message):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T23:00:00Z"),
        )


def test_governed_forward_history_rejects_duplicate_quote_identity(tmp_path: Path) -> None:
    frame, receipt, path = _fixture(tmp_path)
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame.to_parquet(path, index=False)
    payload = path.read_bytes()
    receipt = replace(
        receipt,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_sha256=dataframe_sha256(frame),
    )

    with pytest.raises(ValueError, match="duplicate quote identities"):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T00:00:00Z"),
        )


def test_governed_forward_history_rejects_source_mutation(tmp_path: Path) -> None:
    frame, receipt, path = _fixture(tmp_path)
    path.write_bytes(b"mutated")

    with pytest.raises(ValueError, match="changed after consumption"):
        _prepare_governed_forward_history(
            frame,
            receipt,
            reference_timestamp=pd.Timestamp("2026-07-13T00:00:00Z"),
        )
