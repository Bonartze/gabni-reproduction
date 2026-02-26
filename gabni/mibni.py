from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .mi import mutual_information_binary
from .metrics import gene_dynamics_consistency


@dataclass
class MIBNIResult:
    regulators: list[int]          # indices of regulators
    negated: dict[int, bool]       # regulator index -> whether to negate it
    op: str                        # 'AND' or 'OR'
    consistency: float
    pred: np.ndarray               # length T


def _apply_literals(X: np.ndarray, regs: list[int], neg: dict[int, bool]) -> np.ndarray:
    """Return matrix of regulator literals u or ¬u for t=0..T-2."""
    mat = X[:-1, regs].astype(np.int8)  # shape (T-1, k)
    for j, r in enumerate(regs):
        if neg.get(r, False):
            mat[:, j] = 1 - mat[:, j]
    return mat


def _predict_and_or_lag1(X: np.ndarray, target: np.ndarray, regs: list[int], neg: dict[int, bool], op: str) -> np.ndarray:
    T = X.shape[0]
    pred = np.zeros(T, dtype=np.int8)
    pred[0] = int(target[0])  # unused in consistency
    if len(regs) == 0:
        pred[1:] = 0
        return pred

    mat = _apply_literals(X, regs, neg)  # u(t) literals
    if op == "AND":
        out = (mat.sum(axis=1) == mat.shape[1]).astype(np.int8)
    elif op == "OR":
        out = (mat.sum(axis=1) >= 1).astype(np.int8)
    else:
        raise ValueError(op)

    pred[1:] = out
    return pred


def _best_and_or_with_literals(target: np.ndarray, X: np.ndarray, regs: list[int]) -> tuple[dict[int, bool], str, float, np.ndarray]:
    """
    For a fixed regulator set, choose the best (negations, op) among:
      - all 2^k negation patterns
      - AND/OR
    """
    if len(regs) == 0:
        neg = {}
        p = _predict_and_or_lag1(X, target, regs, neg, "AND")
        c = gene_dynamics_consistency(target, p)
        return neg, "AND", float(c), p

    best_neg: dict[int, bool] = {}
    best_op = "AND"
    best_c = -1.0
    best_pred = None

    k = len(regs)
    for mask in range(1 << k):
        neg = {regs[i]: bool((mask >> i) & 1) for i in range(k)}
        for op in ("AND", "OR"):
            p = _predict_and_or_lag1(X, target, regs, neg, op)
            c = gene_dynamics_consistency(target, p)
            if c > best_c + 1e-12:
                best_c = float(c)
                best_op = op
                best_neg = neg
                best_pred = p

    return best_neg, best_op, best_c, best_pred


def mifs(target_next: np.ndarray, X_t: np.ndarray, mi_min: float, max_k: int, candidates: list[int] | None = None) -> list[int]:
    N = X_t.shape[1]
    base = candidates if candidates is not None else list(range(N))

    # MI filter computed on raw u(t) (not negated); matches standard MIFS usage
    cand = [i for i in base if mutual_information_binary(X_t[:, i], target_next) >= mi_min]
    if not cand:
        return []

    selected: list[int] = []
    best0 = max(cand, key=lambda i: mutual_information_binary(X_t[:, i], target_next))
    selected.append(best0)
    remaining = [i for i in cand if i != best0]

    while remaining and len(selected) < max_k:
        def score(i: int) -> float:
            mi_tv = mutual_information_binary(X_t[:, i], target_next)
            red = sum(mutual_information_binary(X_t[:, i], X_t[:, s]) for s in selected)
            return mi_tv - red
        nxt = max(remaining, key=score)
        selected.append(nxt)
        remaining.remove(nxt)

    return selected


def swap_improve(target: np.ndarray, X: np.ndarray, regs: list[int], mi_min: float, max_k: int, candidates: list[int] | None = None) -> MIBNIResult:
    N = X.shape[1]
    target_next = target[1:]
    X_t = X[:-1, :]
    base = candidates if candidates is not None else list(range(N))
    allowed = [i for i in base if mutual_information_binary(X_t[:, i], target_next) >= mi_min]
    allowed_set = set(allowed)

    regs = [i for i in regs if i in allowed_set][:max_k]
    neg, op, c, pred = _best_and_or_with_literals(target, X, regs)

    improved = True
    while improved:
        improved = False
        best_regs, best_neg, best_op, best_c, best_pred = regs, neg, op, c, pred
        unselected = [i for i in allowed if i not in regs]

        for i_out in list(regs):
            for i_in in unselected:
                trial = [r for r in regs if r != i_out] + [i_in]
                trial = trial[:max_k]
                t_neg, t_op, t_c, t_pred = _best_and_or_with_literals(target, X, trial)
                if t_c > best_c + 1e-12:
                    best_regs, best_neg, best_op, best_c, best_pred = trial, t_neg, t_op, t_c, t_pred

        if best_c > c + 1e-12:
            regs, neg, op, c, pred = best_regs, best_neg, best_op, best_c, best_pred
            improved = True

    return MIBNIResult(regulators=sorted(regs), negated=neg, op=op, consistency=float(c), pred=pred)


def run_mibni(target: np.ndarray, X: np.ndarray, mi_min: float, max_k: int, candidates: list[int] | None = None) -> MIBNIResult:
    target_next = target[1:]
    X_t = X[:-1, :]

    regs0 = mifs(target_next=target_next, X_t=X_t, mi_min=mi_min, max_k=max_k, candidates=candidates)
    return swap_improve(target=target, X=X, regs=regs0, mi_min=mi_min, max_k=max_k, candidates=candidates)