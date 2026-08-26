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
