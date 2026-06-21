from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires the 'hybrid2' extra: pip install -e '.[hybrid2]'"
        ) from exc


def read_parquet(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires the 'hybrid2' extra: pip install -e '.[hybrid2]'"
        ) from exc
