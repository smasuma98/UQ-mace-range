import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.io import write

from orchestr_range_al.io import read_frames, read_reference_dataset


def test_read_frames_and_reference(tmp_path):
    frames = []
    for i in range(3):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7 + 0.01 * i]])
        atoms.info["energy_ref"] = float(i)
        atoms.arrays["forces_ref"] = np.zeros((2, 3))
        frames.append(atoms)

    path = tmp_path / "data.xyz"
    write(path, frames, format="extxyz")

    sampled, idx = read_frames(path, start=0, stop=None, stride=2)
    assert len(sampled) == 2
    assert idx.tolist() == [0, 2]

    ref_frames, E, F, _ = read_reference_dataset(
        path, energy_key="energy_ref", forces_key="forces_ref"
    )
    assert len(ref_frames) == 3
    assert E.tolist() == [0.0, 1.0, 2.0]
    assert F.shape == (3, 2, 3)

def test_read_reference_mixed_atom_counts(tmp_path):
    frames = []
    for formula, count in [("H", 1), ("H2", 2)]:
        atoms = Atoms(formula, positions=np.zeros((count, 3)))
        atoms.info["energy_ref"] = float(count)
        atoms.arrays["forces_ref"] = np.zeros((count, 3))
        frames.append(atoms)
    path = tmp_path / "mixed.xyz"
    write(path, frames, format="extxyz")
    _, energies, forces, _ = read_reference_dataset(
        path, energy_key="energy_ref", forces_key="forces_ref"
    )
    assert energies.tolist() == [1.0, 2.0]
    assert isinstance(forces, list)
    assert [force.shape for force in forces] == [(1, 3), (2, 3)]


def test_write_predictions_npz_ragged(tmp_path):
    ragged = [np.zeros((2, 1, 3)), np.zeros((2, 2, 3))]
    path = tmp_path / "ragged.npz"
    write_predictions_npz(path, {"all_forces": ragged})
    with np.load(path, allow_pickle=True) as data:
        assert data["all_forces"].dtype == object
        assert data["all_forces"].shape == (2,)
        assert data["all_forces"][0].shape == (2, 1, 3)
        assert data["all_forces"][1].shape == (2, 2, 3)
