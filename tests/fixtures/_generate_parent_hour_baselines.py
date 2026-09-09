"""Generate frozen baselines after the parent-hour f_H invariant correction.

The historical Phase 5bis fixtures remain untouched.  This generator creates
an explicitly versioned successor pair for the P1 correction that freezes
horizon-dependent ``f_H`` at the UTC parent-hour boundary.  It must be run
from the canonical repository with the repo-local Python runtime.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tests.fixtures._generate_baseline import build_pfc

_FIXTURE_ROOT = Path(__file__).resolve().parent
_HISTORICAL = {
    False: _FIXTURE_ROOT / "baseline_pfc_seed42.parquet",
    True: _FIXTURE_ROOT / "baseline_pfc_seed42_bowl.parquet",
}
_OUTPUTS = {
    False: _FIXTURE_ROOT / "baseline_pfc_seed42_parent_hour_v1.parquet",
    True: _FIXTURE_ROOT / "baseline_pfc_seed42_bowl_parent_hour_v1.parquet",
}
_INTENDED_CHANGED_COLUMNS = {"price_shape", "f_H", "f_Q"}
_EXPECTED_ARTIFACTS = {
    False: ("852a64aa3d9b278c0e949ba276b58690f5fd05d201664f0b7bc1f9b866967afa", 99_992),
    True: ("3dc1894d6de5d26e4e87a7be2df174ae008e8ac22d9f8c2a63c458df6b260d7f", 99_992),
}


def _assert_transition_scope(current: pd.DataFrame, historical: pd.DataFrame) -> None:
    if list(current.index) != list(historical.index):
        raise RuntimeError("parent-hour baseline transition changed the timestamp index")
    if not set(historical.columns).issubset(current.columns):
        raise RuntimeError("parent-hour baseline transition removed historical columns")

    for column in historical.columns:
        if column in _INTENDED_CHANGED_COLUMNS:
            continue
        left = current[column].reset_index(drop=True)
        right = historical[column].reset_index(drop=True)
        pd.testing.assert_series_equal(left, right, check_names=False, check_exact=True)

    f_h_range = current["f_H"].groupby(current.index.floor("h")).agg(
        lambda values: float(values.max() - values.min())
    )
    if float(f_h_range.max()) != 0.0:
        raise RuntimeError("generated f_H is not constant within every UTC parent hour")

    f_q_delta = np.abs(
        current["f_Q"].to_numpy(dtype=float)
        - historical["f_Q"].to_numpy(dtype=float)
    )
    if float(np.nanmax(f_q_delta)) > np.finfo(float).eps * 2:
        raise RuntimeError("parent-hour transition changed f_Q beyond floating-point ULP noise")


def main() -> None:
    existing = [path for path in _OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to replace frozen parent-hour baselines: "
            + ", ".join(path.name for path in existing)
        )
    for flag in (False, True):
        current = build_pfc(seed=42, flag=flag)
        historical = pd.read_parquet(_HISTORICAL[flag])
        _assert_transition_scope(current, historical)
        output = _OUTPUTS[flag]
        payload = current[historical.columns].to_parquet(index=True)
        expected_sha256, expected_size = _EXPECTED_ARTIFACTS[flag]
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_size or observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"generated fixture identity drifted for flag={flag}: "
                f"sha256={observed_sha256} size_bytes={len(payload)}"
            )
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        print(
            f"flag={flag} path={output.name} rows={len(current)} "
            f"sha256={observed_sha256} size_bytes={len(payload)}"
        )


if __name__ == "__main__":
    main()
