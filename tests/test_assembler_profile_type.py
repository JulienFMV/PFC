from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.assembler import _profile_type_labels


def test_profile_type_labels_do_not_mislabel_horizons_beyond_y3():
    index = pd.date_range("2030-01-01", periods=4, freq="h", tz="UTC")

    labels = _profile_type_labels(
        np.asarray([6, 12, 36, 37], dtype=int),
        index=index,
    )

    assert labels.tolist() == ["M+1..M+6", "M+7..M+12", "Y+2/Y+3", "Y+4+"]
