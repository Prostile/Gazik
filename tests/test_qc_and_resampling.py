import numpy as np
import pandas as pd

from synthetic_well_logs.datasets.curve_aliases import CanonicalWell
from synthetic_well_logs.datasets.qc import QCMaskBuilder
from synthetic_well_logs.datasets.resampling import DepthResampler


def test_resampling_preserves_large_gaps_and_qc_flags_ranges() -> None:
    depth = np.r_[np.arange(0.0, 1.1, 0.1), np.arange(2.0, 3.1, 0.1)]
    gr = 50 + depth
    gr[2] = 400
    well = CanonicalWell(
        well_id="GAP",
        source_file="fixture",
        curves=pd.DataFrame({"DEPT": depth, "GR": gr}),
        units={"DEPT": "m", "GR": "API"},
        original_mnemonics={"DEPT": "DEPT", "GR": "GR"},
    )
    output = DepthResampler(0.1, 0.5).resample(well)
    gap = output.curves["DEPT"].between(1.1, 1.9)
    assert output.curves.loc[gap, "GR"].isna().all()
    assert output.curves["DEPT"].is_monotonic_increasing
    qc = QCMaskBuilder().build(output.curves, "GAP")
    assert qc.masks["GR"].range_violation_mask.any()
    assert qc.masks["GR"].missing_mask.any()
