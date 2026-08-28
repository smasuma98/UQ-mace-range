import numpy as np

from orchestr_range_al.validation import validate_predictions


def test_validation_accepts_mixed_atom_counts():
    stats = {
        "mean_energy": np.array([2.0, 6.0]),
        "mean_forces": [np.ones((1, 3)), np.ones((2, 3))],
        "sigma_energy": np.array([0.2, 0.4]),
        "sigma_energy_per_atom": np.array([0.2, 0.2]),
        "sigma_force_mean": np.array([0.1, 0.2]),
        "sigma_force_max": np.array([0.1, 0.2]),
    }
    result = validate_predictions(
        np.array([1.0, 4.0]),
        [np.zeros((1, 3)), np.zeros((2, 3))],
        stats,
    )
    np.testing.assert_allclose(result["energy_error_per_atom"], [1.0, 1.0])
    assert result["per_frame"]["n_atoms"].tolist() == [1, 2]
    assert isinstance(result["force_error"], list)