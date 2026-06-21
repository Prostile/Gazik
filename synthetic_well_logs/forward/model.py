from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import GroundTruth

MATRIX_DENSITY = {"sandstone": 2.65, "shale": 2.72, "limestone": 2.71, "dolomite": 2.87}
MATRIX_DT = {"sandstone": 55.5, "shale": 82.0, "limestone": 47.5, "dolomite": 43.5}
FLUID_DENSITY = {"water": 1.0, "oil": 0.82, "gas": 0.24, "mixed": 0.65}
FLUID_DT = {"water": 189.0, "oil": 220.0, "gas": 650.0, "mixed": 320.0}


class PhysicsForwardLogModel:
    def generate(
        self,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        del rng  # Physics response is deterministic for a fixed hidden truth.
        rho_matrix = np.array([MATRIX_DENSITY[item] for item in truth.lithology])
        rho_fluid = np.array([FLUID_DENSITY[item] for item in truth.fluid])
        dt_matrix = np.array([MATRIX_DT[item] for item in truth.lithology])
        dt_fluid = np.array([FLUID_DT[item] for item in truth.fluid])

        gr = 18.0 + 132.0 * truth.vsh
        gr += np.where(truth.lithology == "limestone", -7.0, 0.0)
        rhob = (1 - truth.phi) * rho_matrix + truth.phi * rho_fluid
        nphi = truth.phi + 0.17 * truth.vsh
        gas_mask = truth.fluid == "gas"
        nphi[gas_mask] -= 0.12 + 0.10 * truth.phi[gas_mask]
        dt = (1 - truth.phi) * dt_matrix + truth.phi * dt_fluid + 18.0 * truth.vsh
        rt = 0.08 / (np.maximum(truth.phi, 0.025) ** 2 * np.maximum(truth.sw, 0.05) ** 2)
        rt *= 1 + 1.5 * np.maximum(truth.vsh - 0.35, 0)
        cali = np.full(truth.depth.size, 8.5)

        all_curves = {
            "DEPT": truth.depth,
            "GR": gr,
            "CALI": cali,
            "RHOB": rhob,
            "NPHI": nphi,
            "DT": dt,
            "RT": rt,
        }
        return pd.DataFrame({curve: all_curves[curve] for curve in scenario.curves})

