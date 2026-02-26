from __future__ import annotations
import numpy as np

def mutual_information_binary(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information I(x;y) for binary variables, in bits.

    x,y are 1D arrays of 0/1 of same length.
    """
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    n = x.size
    if n == 0:
        return 0.0
    # joint counts
    c00 = np.sum((x == 0) & (y == 0))
    c01 = np.sum((x == 0) & (y == 1))
    c10 = np.sum((x == 1) & (y == 0))
    c11 = np.sum((x == 1) & (y == 1))
    joint = np.array([[c00, c01], [c10, c11]], dtype=float) / n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in (0, 1):
        for j in (0, 1):
            pxy = joint[i, j]
            if pxy <= 0:
                continue
            mi += pxy * np.log2(pxy / (px[i, 0] * py[0, j]))
    return float(mi)

def approx_multivariate_mi(target_next: np.ndarray, candidates: list[np.ndarray]) -> float:
    """A simple approximation: sum of pairwise MI with target_next.

    The original MIBNI uses an approximation used in MIFS; we expose this helper
    for debugging and unit tests. Not used directly in the GA stage.
    """
    return float(sum(mutual_information_binary(c, target_next) for c in candidates))
