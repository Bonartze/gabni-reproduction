from __future__ import annotations

import numpy as np
import pandas as pd

from .mi import mutual_information_binary
from .data import TimeSeriesDataset


def cap_edges_by_mi(ds: TimeSeriesDataset, signed_edges: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Keep top-k directed edges by MI(u(t), v(t+1)).
    This is a postprocessing step that does NOT use gold edges.
    """
    if k <= 0:
        return signed_edges.iloc[0:0].copy()

    X = ds.X.astype(int)
    genes = ds.genes
    idx = {g: i for i, g in enumerate(genes)}

    # compute MI for each edge
    scores = []
    for r in signed_edges.itertuples(index=False):
        s, d = r.src, r.dst
        if s not in idx or d not in idx:
            continue
        i, j = idx[s], idx[d]
        u = X[:-1, i]       # u(t)
        v = X[1:, j]        # v(t+1)
        mi = float(mutual_information_binary(u, v))
        scores.append(mi)

    out = signed_edges.copy()
    if len(scores) != len(out):
        # fallback: if something got skipped, drop those rows
        mask = []
        for r in out.itertuples(index=False):
            mask.append((r.src in idx) and (r.dst in idx))
        out = out.loc[mask].reset_index(drop=True)
        # recompute scores consistently
        scores = []
        for r in out.itertuples(index=False):
            i, j = idx[r.src], idx[r.dst]
            mi = float(mutual_information_binary(X[:-1, i], X[1:, j]))
            scores.append(mi)

    out["_mi"] = scores
    out = out.sort_values("_mi", ascending=False).head(k).drop(columns=["_mi"]).reset_index(drop=True)
    return out