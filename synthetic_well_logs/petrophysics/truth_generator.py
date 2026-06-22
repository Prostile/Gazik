from __future__ import annotations

import numpy as np

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import FaciesInterval, GroundTruth
from synthetic_well_logs.rocks import FACIES_DISPLAY_NAMES_RU, LITHOLOGY, PRIORS


def _smooth_texture(size: int, rng: np.random.Generator, correlation: float = 0.94) -> np.ndarray:
    innovations = rng.normal(size=size)
    result = np.empty(size, dtype=float)
    result[0] = innovations[0]
    scale = np.sqrt(1 - correlation**2)
    for index in range(1, size):
        result[index] = correlation * result[index - 1] + scale * innovations[index]
    return np.clip(result / max(np.std(result), 1e-8), -2.5, 2.5)


class PetrophysicalTruthGenerator:
    def generate(
        self,
        depth: np.ndarray,
        facies: np.ndarray,
        facies_intervals: list[FaciesInterval],
        target_mask: np.ndarray,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> GroundTruth:
        size = depth.size
        vsh = np.empty(size)
        phi = np.empty(size)
        sw = np.empty(size)
        lithology = np.empty(size, dtype="U16")

        common_texture = _smooth_texture(size, rng)
        secondary_texture = _smooth_texture(size, rng, 0.88)
        for name in np.unique(facies):
            mask = facies == name
            prior = PRIORS[str(name)]
            lithology[mask] = LITHOLOGY[str(name)]
            v_mid = sum(prior["vsh"]) / 2
            v_half = (prior["vsh"][1] - prior["vsh"][0]) / 2
            p_mid = sum(prior["phi"]) / 2
            p_half = (prior["phi"][1] - prior["phi"][0]) / 2
            s_mid = sum(prior["sw"]) / 2
            s_half = (prior["sw"][1] - prior["sw"][0]) / 2
            vsh[mask] = v_mid + v_half * 0.62 * common_texture[mask]
            phi[mask] = p_mid - p_half * (
                0.38 * common_texture[mask] - 0.35 * secondary_texture[mask]
            )
            sw[mask] = s_mid + s_half * (
                0.28 * common_texture[mask] + 0.32 * secondary_texture[mask]
            )
            vsh[mask] = np.clip(vsh[mask], *prior["vsh"])
            phi[mask] = np.clip(phi[mask], *prior["phi"])
            sw[mask] = np.clip(sw[mask], *prior["sw"])

        vsh[target_mask] = np.minimum(vsh[target_mask], 0.45)
        phi[target_mask] = np.clip(
            phi[target_mask],
            scenario.target.porosity_range[0],
            scenario.target.porosity_range[1],
        )
        sw[target_mask] = np.clip(
            sw[target_mask],
            scenario.target.water_saturation_range[0],
            scenario.target.water_saturation_range[1],
        )

        fluid = np.full(size, "water", dtype="U8")
        if scenario.target.hydrocarbon != "water":
            fluid[target_mask] = scenario.target.hydrocarbon
        standard_reservoir = (
            np.isin(facies, ["clean_sandstone", "shaly_sandstone", "limestone", "dolomite"])
            & (vsh < 0.5)
            & (phi >= 0.10)
        )
        siltstone_reservoir = (facies == "siltstone") & (vsh < 0.45) & (phi >= 0.12)
        explicit_target = target_mask & ~np.isin(facies, ["coal", "anhydrite", "siltstone"])
        is_reservoir = standard_reservoir | siltstone_reservoir | explicit_target
        is_pay = is_reservoir & target_mask & (fluid != "water") & (sw < 0.68)
        is_pay &= ~np.isin(facies, ["coal", "anhydrite"])

        truth = GroundTruth(
            well_id=scenario.well.name,
            depth_unit=scenario.depth.unit,
            depth=depth.copy(),
            facies=facies.copy(),
            lithology=lithology,
            vsh=vsh,
            phi=phi,
            sw=sw,
            fluid=fluid,
            is_reservoir=is_reservoir,
            is_pay=is_pay,
            bad_hole_mask=np.zeros(size, dtype=bool),
            contacts=self._contacts(depth, target_mask, scenario),
        )
        truth.intervals = self._summarize_intervals(truth, facies_intervals)
        return truth

    @staticmethod
    def _contacts(
        depth: np.ndarray, target_mask: np.ndarray, scenario: ScenarioConfig
    ) -> dict[str, float | None]:
        indices = np.flatnonzero(target_mask)
        if not indices.size or scenario.target.hydrocarbon == "water":
            return {"owc": None, "goc": None}
        top = float(depth[indices[0]])
        if scenario.target.hydrocarbon == "gas":
            return {"owc": None, "goc": top}
        if scenario.target.hydrocarbon == "oil":
            return {"owc": float(depth[indices[-1]]), "goc": None}
        return {"owc": float(depth[indices[-1]]), "goc": top}

    @staticmethod
    def _summarize_intervals(
        truth: GroundTruth, facies_intervals: list[FaciesInterval]
    ) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for number, interval in enumerate(facies_intervals):
            if number == len(facies_intervals) - 1:
                mask = (truth.depth >= interval.top) & (truth.depth <= interval.base)
            else:
                mask = (truth.depth >= interval.top) & (truth.depth < interval.base)
            fluids, fluid_counts = np.unique(truth.fluid[mask], return_counts=True)
            dominant_fluid = str(fluids[np.argmax(fluid_counts)]) if mask.any() else "water"
            output.append(
                {
                    "top": interval.top,
                    "base": interval.base,
                    "facies": interval.facies,
                    "facies_display_name_ru": FACIES_DISPLAY_NAMES_RU[interval.facies],
                    "lithology": interval.lithology,
                    "fluid": dominant_fluid,
                    "vsh_mean": round(float(np.mean(truth.vsh[mask])), 5),
                    "phi_mean": round(float(np.mean(truth.phi[mask])), 5),
                    "sw_mean": round(float(np.mean(truth.sw[mask])), 5),
                    "is_reservoir": bool(np.any(truth.is_reservoir[mask])),
                    "is_pay": bool(np.any(truth.is_pay[mask])),
                }
            )
        return output
