from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StudentInterval(AnswerModel):
    top: float
    base: float
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_depths(self) -> StudentInterval:
        if self.base <= self.top:
            raise ValueError("interval base must be greater than top")
        return self


class PayInterval(StudentInterval):
    fluid: Literal["water", "oil", "gas", "mixed"] | None = None


class LithologyInterval(StudentInterval):
    lithology: str = Field(min_length=1)


class FaciesInterval(StudentInterval):
    facies: str = Field(min_length=1)


class BadHoleInterval(StudentInterval):
    reason: Literal["washout", "missing_interval", "spikes", "depth_shift", "other"]


class StudentAnswer(AnswerModel):
    well_id: str = Field(min_length=1)
    reservoir_intervals: list[StudentInterval] = Field(default_factory=list)
    pay_intervals: list[PayInterval] = Field(default_factory=list)
    lithology_intervals: list[LithologyInterval] = Field(default_factory=list)
    facies_intervals: list[FaciesInterval] = Field(default_factory=list)
    bad_hole_intervals: list[BadHoleInterval] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> StudentAnswer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("student answer root must be a JSON object")
        return cls.model_validate(payload)
