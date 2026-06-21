import lasio
import numpy as np

from synthetic_well_logs.datasets import CurveAliasesConfig, CurveCanonicalizer, UnitNormalizer


def test_aliases_and_units_are_canonicalized(mini_las) -> None:
    las = lasio.read(mini_las)
    well = CurveCanonicalizer(CurveAliasesConfig.from_file()).canonicalize(las, str(mini_las))
    assert list(well.curves) == ["DEPT", "GR", "RHOB", "NPHI", "DT", "RT"]
    assert well.original_mnemonics["GR"] == "GRC"

    normalized = UnitNormalizer().normalize(well)
    assert normalized.units["DEPT"] == "m"
    assert np.isclose(normalized.curves["DEPT"].iloc[0], 1000.0, atol=1e-3)
    assert normalized.curves["RHOB"].between(1.5, 3.1).all()
    assert normalized.curves["NPHI"].between(-0.15, 0.8).all()
    assert normalized.curves["DT"].between(40, 200).all()
