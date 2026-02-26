from __future__ import annotations
import itertools
import numpy as np

from .mi import mutual_information_binary

def _predict_rule(X_prev: np.ndarray, op: str, neg_mask: np.ndarray) -> np.ndarray:
    """X_prev: (T-1,k) binary int8; neg_mask: (k,) bool where True means negate."""
    Xl = X_prev.astype(bool)
    if neg_mask.any():
        Xl = np.logical_xor(Xl, neg_mask)  # flip bits where negate
    if op == "AND":
        return np.all(Xl, axis=1).astype(np.int8)
    elif op == "OR":
        return np.any(Xl, axis=1).astype(np.int8)
    else:
        raise ValueError(op)

def minimize_regulators_exact(
    target: np.ndarray,
    X: np.ndarray,
    candidates: list[int],
    mi_min: float,
    max_k: int,
) -> tuple[list[int], np.ndarray, float]:
    """
    Find a minimal-size regulator set and a simple rule (AND/OR of possibly negated literals)
    that yields perfect one-step prediction on the observed transitions.
    Tie-break: higher mean MI among selected regulators.
    Returns (regs_idx, pred_full, consistency)
    """
    target = target.astype(np.int8)
    X = X.astype(np.int8)
    T, N = X.shape
    y_true = target[1:]
    X_prev = X[:-1, :]

    # compute MI for candidate ranking/filtering
    mi = []
    for i in candidates:
        mi.append((i, float(mutual_information_binary(X_prev[:, i], y_true))))
    # filter by mi_min first (as in paper)
    filt = [i for i, m in mi if m >= mi_min]
    # if too few remain, fall back to top MI candidates
    if len(filt) == 0:
        mi_sorted = sorted(mi, key=lambda x: x[1], reverse=True)
        filt = [i for i, _ in mi_sorted[:min(len(mi_sorted), max_k)]]
    else:
        # keep at most max_k best MI (limits search)
        mi_sorted = sorted([(i, dict(mi)[i]) for i in filt], key=lambda x: x[1], reverse=True)
        filt = [i for i, _ in mi_sorted[:min(len(mi_sorted), max_k)]]

    # allow constant rules (k=0)
    for const in (0, 1):
        if np.all(y_true == const):
            pred = target.copy()
            pred[1:] = const
            return [], pred, 1.0

    mi_map = {i: m for i, m in mi}

    best = None  # (k, -mean_mi, regs, op, neg_mask)
    # brute force from k=1..max_k
    for k in range(1, min(max_k, len(filt)) + 1):
        for regs in itertools.combinations(filt, k):
            regs = list(regs)
            Xk = X_prev[:, regs]
            # iterate over operator and negations
            for op in ("AND", "OR"):
                for neg_bits in itertools.product([False, True], repeat=k):
                    neg_mask = np.array(neg_bits, dtype=bool)
                    y_pred = _predict_rule(Xk, op, neg_mask)
                    if np.array_equal(y_pred, y_true):
                        mean_mi = sum(mi_map.get(i, 0.0) for i in regs) / k
                        cand = (k, -mean_mi, regs, op, neg_mask)
                        if best is None or cand < best:
                            best = cand
        if best is not None:
            break

    if best is None:
        # if no perfect rule found in this class, return empty (caller can fall back)
        return [], target.copy(), float(np.mean(y_true == y_true))  # 1.0 placeholder

    _, _, regs, op, neg_mask = best
    pred = target.copy()
    pred[1:] = _predict_rule(X_prev[:, regs], op, neg_mask)
    return regs, pred, 1.0
