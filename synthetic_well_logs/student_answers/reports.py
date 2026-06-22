from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StudentCheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    well_id: str
    total_score: float
    reservoir_score: dict[str, Any]
    pay_score: dict[str, Any]
    lithology_score: dict[str, Any]
    quality_score: dict[str, Any]
    feedback: list[str]

