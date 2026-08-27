from pathlib import Path

import numpy as np
import pandas as pd

from orchestr_range_al.plotting import (
    parse_orchestr_diagnostics_file,
    write_pool_energy_trace_npz,
)


def test_parse_orchestr_diagnostics_and_energy_trace(tmp_path: Path):
    diag = tmp_path / "al_pool_per_frame_diagnostics.txt"
    diag.write_text(
        """# Pool Active Learning Diagnostics
# thr_sigma_E_low        = 0.000356
# thr_sigma_E_hi_eff     = 0.007500
# hard_sigma_E_atom_min  = 0.003000
idx window geom_ok caps_ok force_inf γ_gate gamma0 dM Dgain raw_score σE_atom σF_max σF_mean Eabs_exp Fabs_mean cal_ok ood Fmax selected shortlist
0 0-100 1 1 1 1 1.0 2.0 3.0 4.0 0.004 0.20 0.06 0.003 0.04 0 1 1.5 1 0
1 0-100 0 0 0 1 1.0 2.0 3.0 4.0 0.005 0.30 0.07 0.004 0.05 0 1 1.6 0 1
# TRAIN DATASET UNCERTAINTIES
# idx sigma_E_atom sigma_F_max sigma_F_mean Fmax
0 0.0002 0.04 0.02 1.1
1 0.0003 0.05 0.03 1.2
""",
        encoding="utf-8",
    )

    parsed = parse_orchestr_diagnostics_file(diag)
    assert parsed["pool_idx"].tolist() == [0, 1]
    assert parsed["pool_geom_ok"].tolist() == [True, False]
    assert parsed["pool_shortlist"].tolist() == [False, True]
    assert parsed["train_idx"].tolist() == [0, 1]
    assert parsed["thresholds"]["hard_sigma_E_atom_min"] == 0.003

    pool_df = pd.DataFrame(
        {
            "sampled_frame": [0, 1],
            "energy_mean_eV": [-10.0, -9.0],
            "sigma_E_eV": [0.2, 0.4],
        }
    )
    out = write_pool_energy_trace_npz(
        pool_df, parsed, tmp_path / "pool_energy_trace.npz", smoothing_window=2
    )
    data = np.load(out)
    assert data["steps"].tolist() == [0, 1]
    assert data["bad"].tolist() == [False, True]
    assert int(data["smoothing_window"]) == 2
