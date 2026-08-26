import numpy as np

from orchestr_range_al.ensemble import compute_ensemble_statistics


def test_ensemble_shapes_and_ddof():
    energies = np.array([[1.0, 2.0], [3.0, 4.0]])
    forces = np.zeros((2, 2, 1, 3), dtype=float)
    forces[1, :, 0, 0] = 2.0

    stats = compute_ensemble_statistics(energies, forces)

    assert stats["mean_energy"].shape == (2,)
    assert stats["sigma_force_components"].shape == (2, 1, 3)
    np.testing.assert_allclose(stats["sigma_energy"], [np.sqrt(2), np.sqrt(2)])
    np.testing.assert_allclose(
        stats["sigma_force_components"][:, 0, 0], [np.sqrt(2), np.sqrt(2)]
    )
