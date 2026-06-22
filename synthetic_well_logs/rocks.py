"""Shared rock properties used by truth, physics, constraints and reports."""

FACIES_DISPLAY_NAMES_RU = {
    "shale": "глина",
    "shaly_sandstone": "глинистый песчаник",
    "clean_sandstone": "чистый песчаник",
    "tight_sandstone": "плотный песчаник",
    "limestone": "известняк",
    "dolomite": "доломит",
    "siltstone": "алевролит",
    "marl": "мергель",
    "coal": "уголь",
    "anhydrite": "ангидрит",
}

LITHOLOGY = {
    "shale": "shale",
    "clean_sandstone": "sandstone",
    "shaly_sandstone": "sandstone",
    "tight_sandstone": "sandstone",
    "limestone": "limestone",
    "dolomite": "dolomite",
    "siltstone": "siltstone",
    "marl": "marl",
    "coal": "coal",
    "anhydrite": "anhydrite",
}

PRIORS = {
    "shale": {"vsh": (0.65, 1.00), "phi": (0.03, 0.15), "sw": (0.80, 1.00)},
    "shaly_sandstone": {
        "vsh": (0.25, 0.60),
        "phi": (0.08, 0.22),
        "sw": (0.35, 1.00),
    },
    "clean_sandstone": {
        "vsh": (0.00, 0.20),
        "phi": (0.15, 0.32),
        "sw": (0.20, 1.00),
    },
    "tight_sandstone": {
        "vsh": (0.00, 0.25),
        "phi": (0.03, 0.12),
        "sw": (0.60, 1.00),
    },
    "limestone": {"vsh": (0.00, 0.15), "phi": (0.02, 0.25), "sw": (0.25, 1.00)},
    "dolomite": {"vsh": (0.00, 0.15), "phi": (0.02, 0.20), "sw": (0.25, 1.00)},
    "siltstone": {"vsh": (0.35, 0.70), "phi": (0.06, 0.18), "sw": (0.55, 1.00)},
    "marl": {"vsh": (0.30, 0.65), "phi": (0.03, 0.14), "sw": (0.65, 1.00)},
    "coal": {"vsh": (0.05, 0.35), "phi": (0.02, 0.12), "sw": (0.20, 1.00)},
    "anhydrite": {"vsh": (0.00, 0.10), "phi": (0.00, 0.04), "sw": (0.80, 1.00)},
}

MATRIX_DENSITY = {
    "sandstone": 2.65,
    "shale": 2.72,
    "limestone": 2.71,
    "dolomite": 2.87,
    "siltstone": 2.68,
    "marl": 2.70,
    "coal": 1.35,
    "anhydrite": 2.98,
}

MATRIX_DT = {
    "sandstone": 55.5,
    "shale": 82.0,
    "limestone": 47.5,
    "dolomite": 43.5,
    "siltstone": 70.0,
    "marl": 65.0,
    "coal": 110.0,
    "anhydrite": 50.0,
}

FACIES_CURVE_RANGES = {
    "shale": {
        "GR": (80.0, 180.0),
        "RHOB": (2.15, 2.85),
        "NPHI": (0.15, 0.65),
        "RT": (0.2, 100.0),
    },
    "clean_sandstone": {
        "GR": (10.0, 80.0),
        "RHOB": (1.75, 2.65),
        "NPHI": (-0.05, 0.38),
        "RT": (0.2, 10_000.0),
    },
    "shaly_sandstone": {
        "GR": (35.0, 130.0),
        "RHOB": (1.85, 2.75),
        "NPHI": (0.02, 0.50),
        "RT": (0.2, 2_000.0),
    },
    "tight_sandstone": {
        "GR": (10.0, 90.0),
        "RHOB": (2.35, 2.75),
        "NPHI": (-0.05, 0.25),
        "DT": (45.0, 90.0),
        "RT": (0.5, 1_000.0),
    },
    "limestone": {
        "GR": (0.0, 65.0),
        "RHOB": (2.15, 2.80),
        "NPHI": (-0.05, 0.35),
        "DT": (45.0, 100.0),
        "RT": (0.5, 10_000.0),
    },
    "dolomite": {
        "GR": (0.0, 65.0),
        "RHOB": (2.30, 2.95),
        "NPHI": (-0.05, 0.30),
        "DT": (40.0, 95.0),
        "RT": (0.5, 10_000.0),
    },
    "siltstone": {
        "GR": (55.0, 130.0),
        "RHOB": (2.35, 2.75),
        "NPHI": (0.10, 0.35),
        "DT": (65.0, 115.0),
        "RT": (0.5, 120.0),
    },
    "marl": {
        "GR": (45.0, 140.0),
        "RHOB": (2.45, 2.85),
        "NPHI": (0.08, 0.32),
        "DT": (55.0, 105.0),
        "RT": (0.5, 150.0),
    },
    "coal": {
        "GR": (10.0, 90.0),
        "RHOB": (1.10, 1.80),
        "NPHI": (0.25, 0.75),
        "DT": (85.0, 170.0),
        "RT": (5.0, 10_000.0),
    },
    "anhydrite": {
        "GR": (0.0, 45.0),
        "RHOB": (2.80, 3.10),
        "NPHI": (-0.10, 0.12),
        "DT": (40.0, 75.0),
        "RT": (10.0, 10_000.0),
    },
}
