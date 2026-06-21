from __future__ import annotations

from typing import Any


def summarize_physical_validation(constraint_report: dict[str, Any]) -> dict[str, Any]:
    rate = float(constraint_report.get("constraint_violation_rate", 1.0))
    return {
        "valid": bool(
            rate <= 0.05
            and constraint_report.get("pay_interval_preserved", False)
            and constraint_report.get("gas_effect_preserved", False)
        ),
        **constraint_report,
    }
