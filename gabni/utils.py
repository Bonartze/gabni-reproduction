from __future__ import annotations
import numpy as np

def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_bool01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if not np.isin(x, [0, 1]).all():
        raise ValueError("Expected binary array with values {0,1}.")
    return x.astype(np.int8)

def powerset_bitstrings(k: int):
    # yields tuples of length k in lexicographic order (0..2^k-1)
    for i in range(1 << k):
        yield tuple((i >> (k - 1 - b)) & 1 for b in range(k))
