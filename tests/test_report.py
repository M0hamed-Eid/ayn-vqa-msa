"""Unit tests for report.py's aggregation helpers.

`_to_native` gets a dedicated test because it's the one thing standing
between pandas/numpy output and `json.dumps` -- numpy int64 dict keys/values
and NaN are exactly the kind of thing that parses fine in dev and then
throws `TypeError` the first time a column happens to contain them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ayn_vqa.audit.report import _to_native


def test_to_native_converts_numpy_scalars() -> None:
    payload = {"count": np.int64(5), "mean": np.float64(1.5), "flag": np.bool_(True)}
    native = _to_native(payload)

    assert native == {"count": 5, "mean": 1.5, "flag": True}
    json.dumps(native)  # must not raise


def test_to_native_converts_nan_to_none() -> None:
    native = _to_native({"value": float("nan")})
    assert native["value"] is None


def test_to_native_stringifies_dict_keys() -> None:
    series = pd.Series([0, 1, 1, None])
    counts = series.value_counts(dropna=False).to_dict()  # keys include numpy scalars / NaN

    native = _to_native(counts)

    assert all(isinstance(k, str) for k in native)
    json.dumps(native)  # must not raise


def test_to_native_recurses_into_lists() -> None:
    native = _to_native([{"n": np.int64(1)}, {"n": np.int64(2)}])
    assert native == [{"n": 1}, {"n": 2}]
