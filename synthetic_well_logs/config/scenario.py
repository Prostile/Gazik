from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SUPPORTED_CURVES = {"DEPT", "GR", "CALI", "RHOB", "NPHI", "DT", "RT"}
SUPPORTED_FACIES = {
    "shale",
    "clean_sandstone",
    "shaly_sandstone",
    "tight_sandstone",
    "limestone",
    "dolomite",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WellConfig(StrictModel):
    name: str = Field(min_length=1)
    field: str = "Synthetic Training Field"
    country: str = "Synthetic"


class DepthConfig(StrictModel):
    start: float
    stop: float
    step: float = Field(gt=0)
    unit: Literal["m", "ft"] = "m"

    @model_validator(mode="after")
    def validate_range(self) -> DepthConfig:
        if self.stop <= self.start:
            raise ValueError("depth.stop must be greater than depth.start")
        samples = (self.stop - self.start) / self.step
        if samples > 2_000_000:
            raise ValueError("depth grid exceeds the 2,000,001 sample safety limit")
        return self


class GeologyConfig(StrictModel):
    depositional_environment: str
    stacking_pattern: str
    facies_set: list[str] = Field(
        default_factory=lambda: [
            "shale",
            "shaly_sandstone",
            "clean_sandstone",
            "tight_sandstone",
        ],
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_facies(self) -> GeologyConfig:
        unknown = set(self.facies_set) - SUPPORTED_FACIES
        if unknown:
            raise ValueError(f"unsupported facies: {sorted(unknown)}")
        if len(set(self.facies_set)) != len(self.facies_set):
            raise ValueError("geology.facies_set must not contain duplicates")
        return self


class TargetConfig(StrictModel):
    reservoir_type: str
    hydrocarbon: Literal["water", "oil", "gas", "mixed"]
    net_pay_thickness_m: tuple[float, float] = (6.0, 14.0)
    porosity_range: tuple[float, float] = (0.15, 0.28)
    water_saturation_range: tuple[float, float] = (0.2, 0.55)

    @model_validator(mode="after")
    def validate_ranges(self) -> TargetConfig:
        for name in ("net_pay_thickness_m", "porosity_range", "water_saturation_range"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"target.{name} lower bound must not exceed upper bound")
        if not 0 <= self.porosity_range[0] <= self.porosity_range[1] <= 0.5:
            raise ValueError("target.porosity_range must be within [0, 0.5]")
        if not 0 <= self.water_saturation_range[0] <= self.water_saturation_range[1] <= 1:
            raise ValueError("target.water_saturation_range must be within [0, 1]")
        return self


class ArtifactConfig(StrictModel):
    noise: bool = True
    washout: bool = False
    spikes: bool = False
    missing_intervals: bool = False
    depth_shift: bool = False


class RealismConfig(StrictModel):
    mode: Literal["none", "statistical", "autoencoder_mcmc", "diffusion_residual"] = (
        "statistical"
    )
    strength: float = Field(default=0.35, ge=0, le=1)


class ScenarioConfig(StrictModel):
    well: WellConfig
    depth: DepthConfig
    geology: GeologyConfig
    target: TargetConfig
    curves: list[str] = Field(min_length=2)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    realism: RealismConfig = Field(default_factory=RealismConfig)
    difficulty: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    seed: int = 42

    @model_validator(mode="after")
    def validate_scenario(self) -> ScenarioConfig:
        self.curves = [curve.upper() for curve in self.curves]
        unknown = set(self.curves) - SUPPORTED_CURVES
        if unknown:
            raise ValueError(f"unsupported curves: {sorted(unknown)}")
        if "DEPT" not in self.curves:
            raise ValueError("curves must include DEPT")
        if len(set(self.curves)) != len(self.curves):
            raise ValueError("curves must not contain duplicates")
        if self.target.reservoir_type not in self.geology.facies_set:
            raise ValueError("target.reservoir_type must be present in geology.facies_set")
        if self.realism.mode in {"autoencoder_mcmc", "diffusion_residual"}:
            raise ValueError(
                f"realism mode '{self.realism.mode}' is reserved for an optional future extra"
            )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> ScenarioConfig:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        elif source.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            raise ValueError("scenario must be a YAML or JSON file")
        if not isinstance(payload, dict):
            raise ValueError("scenario root must be an object")
        return cls.model_validate(payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioConfig:
        return cls.from_file(path)

