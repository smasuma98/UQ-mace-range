import numpy as np

from orchestr_range_al.calibration import (
    CalibrationBundle,
    calibrate_energy_total,
    calibrate_force_components,
)


def test_raw_calibration_passthrough():
    bundle = CalibrationBundle(
        mode="raw",
        energy_model=None,
        force_model=None,
        energy_accepted=False,
        force_accepted=False,
        metrics={},
    )
    e = np.array([1.0, 2.0])
    counts = np.array([10, 20])
    f = np.ones((2, 3, 3))
    np.testing.assert_allclose(calibrate_energy_total(e, counts, bundle), e)
    np.testing.assert_allclose(calibrate_force_components(f, bundle), f)
