from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class TimeSeriesDataset:
    X: np.ndarray  # shape (T, N), values in {0,1}
    genes: list[str]

    @property
    def T(self) -> int:
        return int(self.X.shape[0])

    @property
    def N(self) -> int:
        return int(self.X.shape[1])


def load_csv_timeseries(path: str, *, drop_cols: Optional[list[str]] = None) -> TimeSeriesDataset:
    """
    Load a time-series CSV.
    Expected: rows=timepoints, columns=genes, values are 0/1 (or numeric that can be cast to int).

    The loader is robust:
      - drops common non-gene columns: 'time', 'Time', 'phase', 'Phase'
      - drops any non-numeric columns automatically
    """
    df = pd.read_csv(path)

    # Drop known non-gene columns if present
    default_drop = {"time", "Time", "phase", "Phase"}
    if drop_cols:
        default_drop |= set(drop_cols)
    cols_to_drop = [c for c in df.columns if c in default_drop]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Keep only numeric columns (auto-drops Phase-like columns)
    df = df.select_dtypes(include=["number"])

    if df.shape[1] == 0:
        raise ValueError(
            f"No numeric gene columns found in {path}. "
            "Ensure the CSV has numeric 0/1 columns for genes."
        )

    # Convert to int8
    X = df.to_numpy(dtype=np.int8)

    # Validate binary (allow only 0/1)
    uniq = np.unique(X)
    if not set(int(v) for v in uniq).issubset({0, 1}):
        raise ValueError(
            f"Expected binary 0/1 values after loading {path}, but got values: {uniq}."
        )

    genes = list(df.columns)
    return TimeSeriesDataset(X=X, genes=genes)


def kmeans_binarize(df: pd.DataFrame, random_state: int = 0) -> TimeSeriesDataset:
    """
    Per-gene k-means binarization (k=2), as used in the paper for continuous inputs.
    Drops 'time'/'phase' columns if present, then binarizes each remaining column.
    """
    # Drop non-gene columns
    drop = [c for c in df.columns if c.lower() in ("time", "phase")]
    if drop:
        df = df.drop(columns=drop)

    # Keep numeric only
    df = df.select_dtypes(include=["number"])
    if df.shape[1] == 0:
        raise ValueError("No numeric columns available for binarization.")

    Xb = np.zeros(df.shape, dtype=np.int8)
    for j, col in enumerate(df.columns):
        x = df[col].to_numpy(dtype=float).reshape(-1, 1)
        km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        labels = km.fit_predict(x)

        # Map cluster with higher mean to 1, lower mean to 0
        means = [x[labels == k].mean() for k in (0, 1)]
        one_cluster = int(np.argmax(means))
        Xb[:, j] = (labels == one_cluster).astype(np.int8)

    return TimeSeriesDataset(X=Xb, genes=list(df.columns))