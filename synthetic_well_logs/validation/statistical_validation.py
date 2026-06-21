from __future__ import annotations

from typing import Any


def statistical_validation_status(realism_report: dict[str, Any] | None) -> dict[str, Any]:
    if not realism_report:
        return {"status": "not_calibrated", "valid": True}
    return {"status": "model_applied", "valid": True, **realism_report}
