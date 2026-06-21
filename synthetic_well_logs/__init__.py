"""Public API for the synthetic well-log generator."""

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain.models import GeneratedWell
from synthetic_well_logs.generator import generate_well

__all__ = ["GeneratedWell", "ScenarioConfig", "generate_well"]
__version__ = "0.1.0"

