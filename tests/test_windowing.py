import numpy as np
import pandas as pd

from synthetic_well_logs.datasets.qc import QCMaskBuilder
from synthetic_well_logs.datasets.windowing import WindowSegmenter


def test_window_shape_metadata_and_nan_rejection() -> None:
    size = 96
    frame = pd.DataFrame(
        {
            "DEPT": 1000 + np.arange(size) * 0.1,
            "GR": np.linspace(50, 100, size),
            "RHOB": np.full(size, 2.4),
            "NPHI": np.full(size, 0.2),
            "DT": np.full(size, 85.0),
            "RT": np.full(size, 10.0),
        }
    )
    frame.loc[:20, "NPHI"] = np.nan
    qc = QCMaskBuilder().build(frame, "WINDOW")
    windows = WindowSegmenter(32, 16, 0.9).segment(frame, qc, "WINDOW", "fixture.las")
    assert len(windows) == 4
    assert windows[0].curves.shape == (5, 32)
    assert windows[0].valid_mask.shape == (5, 32)
    assert windows[0].top < windows[0].base
    assert windows[0].source_file == "fixture.las"
