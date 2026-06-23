from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ResistivityModelConfig, ScenarioConfig
from synthetic_well_logs.domain import GroundTruth
from synthetic_well_logs.rocks import MATRIX_DENSITY, MATRIX_DT

FLUID_DENSITY = {"water": 1.0, "oil": 0.82, "gas": 0.24, "mixed": 0.65}
FLUID_DT = {"water": 189.0, "oil": 220.0, "gas": 650.0, "mixed": 320.0}


def calculate_resistivity(
    phi: np.ndarray,
    sw: np.ndarray,
    vsh: np.ndarray,
    lithology: np.ndarray,
    fluid: np.ndarray,
    config: ResistivityModelConfig,
) -> np.ndarray:
    """Calculate educational Archie or shale-corrected formation resistivity."""
    del fluid
    phi_safe = np.maximum(np.asarray(phi, dtype=float), 0.025)
    sw_safe = np.maximum(np.asarray(sw, dtype=float), 0.05)
    formation_factor = config.a / phi_safe**config.m
    water_component = config.rw / sw_safe**config.n
    rt = formation_factor * water_component
    if config.model == "shaly_sand":
        shale_sensitive = np.isin(lithology, ["sandstone", "siltstone", "marl", "shale"])
        conductivity = 1.0 + config.shale_conductivity_factor * np.asarray(vsh, dtype=float)
        rt = np.where(shale_sensitive, rt / conductivity, rt)
    return rt


def calculate_expected_curves_from_arrays(
    *,
    vsh: np.ndarray,
    phi: np.ndarray,
    sw: np.ndarray,
    lithology: np.ndarray,
    fluid: np.ndarray,
    resistivity_config: ResistivityModelConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return deterministic expected curves from hidden petrophysical arrays."""
    config = resistivity_config or ResistivityModelConfig()
    vsh = np.asarray(vsh, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)
    lithology = np.asarray(lithology, dtype=str)
    fluid = np.asarray(fluid, dtype=str)
    rho_matrix = np.array([MATRIX_DENSITY[item] for item in lithology])
    rho_fluid = np.array([FLUID_DENSITY[item] for item in fluid])
    dt_matrix = np.array([MATRIX_DT[item] for item in lithology])
    dt_fluid = np.array([FLUID_DT[item] for item in fluid])

    gr = 18.0 + 132.0 * vsh
    gr += np.where(lithology == "limestone", -7.0, 0.0)
    gr += np.where(lithology == "siltstone", 10.0, 0.0)
    gr += np.where(lithology == "marl", 15.0, 0.0)
    rhob = (1 - phi) * rho_matrix + phi * rho_fluid
    nphi = phi + 0.17 * vsh
    gas_mask = fluid == "gas"
    nphi[gas_mask] -= 0.12 + 0.10 * phi[gas_mask]
    dt = (1 - phi) * dt_matrix + phi * dt_fluid + 18.0 * vsh
    nphi += np.where(lithology == "marl", 0.05, 0.0)
    dt += np.where(lithology == "marl", 8.0, 0.0)
    rt = calculate_resistivity(
        phi,
        sw,
        vsh,
        lithology,
        fluid,
        config,
    )

    coal = lithology == "coal"
    gr[coal] = np.minimum(gr[coal], 75.0)
    rhob[coal] = np.clip(rhob[coal], 1.15, 1.75)
    nphi[coal] = np.maximum(nphi[coal], 0.30)
    rt[coal] *= 2.0

    anhydrite = lithology == "anhydrite"
    gr[anhydrite] = np.minimum(gr[anhydrite], 35.0)
    rhob[anhydrite] = np.clip(rhob[anhydrite], 2.85, 3.05)
    nphi[anhydrite] = np.clip(nphi[anhydrite], -0.05, 0.08)
    rt[anhydrite] *= 3.0
    return {"GR": gr, "RHOB": rhob, "NPHI": nphi, "DT": dt, "RT": rt}


def calculate_expected_curves(
    truth: GroundTruth,
    resistivity_config: ResistivityModelConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return the deterministic tool response before smoothing and realism."""
    return calculate_expected_curves_from_arrays(
        vsh=truth.vsh,
        phi=truth.phi,
        sw=truth.sw,
        lithology=truth.lithology,
        fluid=truth.fluid,
        resistivity_config=resistivity_config,
    )


class PhysicsForwardLogModel:
    def generate(
        self,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        del rng  # Physics response is deterministic for a fixed hidden truth.
        all_curves = calculate_expected_curves(
            truth,
            scenario.petrophysics.resistivity_model,
        )
        all_curves["DEPT"] = truth.depth
        all_curves["CALI"] = np.full(truth.depth.size, 8.5)
        return pd.DataFrame({curve: all_curves[curve] for curve in scenario.curves})
