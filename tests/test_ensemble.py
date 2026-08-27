import numpy as np
<<<<<<< HEAD
import pytest

from orchestr_range_al.ensemble import compute_ensemble_statistics
from orchestr_range_al.io import write_predictions_npz
from orchestr_range_al.prediction import load_ensemble_predictions
=======

from orchestr_range_al.ensemble import compute_ensemble_statistics
>>>>>>> 22f040ea05dc46910f1f953ae719730865691ffe


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
<<<<<<< HEAD


def test_load_ensemble_predictions_validates_and_reuses_cache(tmp_path):
    frames = [np.zeros((2, 3)), np.zeros((2, 3))]
    model_paths = [tmp_path / "one.model", tmp_path / "two.model"]
    source = np.array([3, 7])
    payload = {
        "all_energies": np.zeros((2, 2)),
        "all_forces": np.zeros((2, 2, 2, 3)),
        "model_labels": ["one", "two"],
        "model_paths": [str(path) for path in model_paths],
        "source_frame_indices": source,
    }
    cache_path = tmp_path / "predictions.npz"
    write_predictions_npz(cache_path, payload)

    loaded = load_ensemble_predictions(cache_path, frames, model_paths, source)

    assert loaded["model_labels"] == ["one", "two"]
    np.testing.assert_array_equal(loaded["source_frame_indices"], source)

    with pytest.raises(ValueError, match="source-frame indices"):
        load_ensemble_predictions(cache_path, frames, model_paths, np.array([3, 8]))
=======
>>>>>>> 22f040ea05dc46910f1f953ae719730865691ffe
