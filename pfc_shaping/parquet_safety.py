"""Bounded Parquet metadata admission before DataFrame allocation."""

from __future__ import annotations

from io import BytesIO
from typing import Collection

import pyarrow as pa
import pyarrow.parquet as pq


def validate_parquet_allocation_budget(
    payload: bytes,
    *,
    label: str,
    max_rows: int,
    max_columns: int,
    max_cells: int,
    max_row_groups: int,
    allowed_physical_types: Collection[str],
) -> None:
    """Reject allocation bombs using footer metadata before decoding columns."""

    try:
        metadata = pq.ParquetFile(BytesIO(payload)).metadata
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise ValueError(f"{label} is not parquet") from exc
    rows = int(metadata.num_rows)
    columns = int(metadata.num_columns)
    row_groups = int(metadata.num_row_groups)
    if rows < 1 or rows > max_rows:
        raise ValueError(f"{label} parquet row budget is exceeded")
    if (
        columns < 1
        or columns > max_columns
        or rows * columns > max_cells
        or row_groups < 1
        or row_groups > max_row_groups
    ):
        raise ValueError(f"{label} parquet allocation budget is exceeded")
    leaves = [metadata.schema.column(position) for position in range(columns)]
    if any(
        int(leaf.max_repetition_level) != 0 or "." in str(leaf.path)
        for leaf in leaves
    ):
        raise ValueError(f"{label} parquet schema must be flat and non-repeated")
    physical_types = {str(leaf.physical_type) for leaf in leaves}
    unsupported = physical_types.difference(allowed_physical_types)
    if unsupported:
        raise ValueError(f"{label} parquet physical types are unsupported")
