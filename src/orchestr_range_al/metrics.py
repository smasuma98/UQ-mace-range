"""Small numerical metrics used by validation/calibration."""

from __future__ import annotations

import numpy as np


def mae(error) -> float:
    arr = np.asarray(error, dtype=float)
    return float(np.nanmean(np.abs(arr)))


def rmse(error) -> float:
    arr = np.asarray(error, dtype=float)
    return float(np.sqrt(np.nanmean(arr**2)))
