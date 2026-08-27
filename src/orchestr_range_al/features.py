"""SOAP/PCA feature-space construction for Orchestr.AI pool selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _compute_soap_with_orchestr(frames, *, train_frames, species, r_cut, n_max, l_max):
    try:
        from orchestr_ai.postprocessing.active_learning import compute_soap_features
    except ImportError as exc:
        raise ImportError(
            "Orchestr.AI is required for the version-1 SOAP feature adapter. "
            "Install the ivan2 branch so orchestr_ai.postprocessing.active_learning "
            "is importable."
        ) from exc

    values, used_species = compute_soap_features(
        frames,
        train_frames=train_frames,
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
    )
    if values is None:
        raise RuntimeError("Orchestr.AI compute_soap_features returned no features.")
    return np.asarray(values, dtype=float), used_species


def build_soap_pca_features(
    train_frames,
    pool_frames,
    cfg: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit SOAP scaler/PCA on training frames, transform training and pool."""
    if bool(cfg.get("periodic", False)):
        raise ValueError(
            "The Orchestr.AI SOAP helper used by this adapter is non-periodic. "
            "Set features.periodic: false for version 1."
        )
    if str(cfg.get("average", "outer")).lower() != "outer":
        raise ValueError(
            "The Orchestr.AI SOAP helper uses average='outer'. "
            "Set features.average: outer for version 1."
        )

    species = cfg.get("species") or None
    r_cut = float(cfg.get("r_cut", 4.0))
    n_max = int(cfg.get("n_max", 4))
    l_max = int(cfg.get("l_max", 4))

    train_raw, species = _compute_soap_with_orchestr(
        train_frames,
        train_frames=train_frames,
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
    )
    pool_raw, _ = _compute_soap_with_orchestr(
        pool_frames,
        train_frames=train_frames,
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    pool_scaled = scaler.transform(pool_raw)

    pca_n_components = cfg.get("pca_n_components", 0.99)
    pca = PCA(n_components=pca_n_components, svd_solver="auto")
    F_train = pca.fit_transform(train_scaled)
    F_pool = pca.transform(pool_scaled)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = output_dir / "soap_scaler.joblib"
    pca_path = output_dir / "soap_pca.joblib"
    metadata_path = output_dir / "feature_metadata.json"
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)

    metadata = {
        "type": "soap_pca",
        "species": list(species),
        "r_cut": r_cut,
        "n_max": n_max,
        "l_max": l_max,
        "periodic": False,
        "average": "outer",
        "raw_dimension": int(train_raw.shape[1]),
        "pca_dimension": int(F_train.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "scaler": scaler_path.name,
        "pca": pca_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "F_train": F_train,
        "F_pool": F_pool,
        "metadata": metadata,
        "scaler": scaler,
        "pca": pca,
    }
