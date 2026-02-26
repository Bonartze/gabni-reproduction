from __future__ import annotations
import numpy as np
import pandas as pd

DONT_CARE = -1
TIE = -2

def gene_dynamics_consistency(target: np.ndarray, pred: np.ndarray) -> float:
    """
    Backward-compatible C(v,v') for reporting: treat DONT_CARE/TIE as mismatches.
    """
    target = np.asarray(target, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if target.shape != pred.shape:
        raise ValueError(f"target and pred must have same shape, got {target.shape} vs {pred.shape}")
    if target.size <= 1:
        return 1.0
    t = target[1:]
    p = pred[1:]
    return float((t == p).mean())

def gene_consistency_and_coverage(target: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """
    Returns:
      match_consistency: accuracy on defined positions only (pred in {0,1})
      coverage: fraction of positions that are defined (pred in {0,1})
    Both computed on t=1..T-1.
    """
    target = np.asarray(target, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if target.shape != pred.shape:
        raise ValueError(f"target and pred must have same shape, got {target.shape} vs {pred.shape}")
    if target.size <= 1:
        return 1.0, 1.0

    t = target[1:]
    p = pred[1:]
    defined = (p == 0) | (p == 1)
    cov = float(defined.mean())
    if cov == 0.0:
        return 0.0, 0.0
    match = float((t[defined] == p[defined]).mean())
    return match, cov

def dynamics_accuracy(obs_mat: np.ndarray, pred_mat: np.ndarray) -> float:
    N = obs_mat.shape[1]
    return float(np.mean([gene_dynamics_consistency(obs_mat[:, i], pred_mat[:, i]) for i in range(N)]))

def edge_metrics_unsigned(pred_edges: pd.DataFrame, gold_edges: pd.DataFrame, genes: list[str]) -> dict:
    gold = {(r.src, r.dst) for r in gold_edges.itertuples(index=False)}
    pred = {(r.src, r.dst) for r in pred_edges.itertuples(index=False)}

    possible = [(s, d) for s in genes for d in genes if s != d]
    TP = FP = FN = TN = 0
    for s, d in possible:
        g = (s, d) in gold
        p = (s, d) in pred
        if g and p:
            TP += 1
        elif (not g) and (not p):
            TN += 1
        elif (not g) and p:
            FP += 1
        elif g and (not p):
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=precision, recall=recall, structural_accuracy=acc)

def edge_metrics_signed(pred_edges: pd.DataFrame, gold_edges: pd.DataFrame, genes: list[str]) -> dict:
    idx = {(r.src, r.dst): int(r.sign) for r in gold_edges.itertuples(index=False)}
    pdx = {(r.src, r.dst): int(r.sign) for r in pred_edges.itertuples(index=False)}

    possible = [(s, d) for s in genes for d in genes if s != d]
    TP = FP = FN = TN = 0
    for s, d in possible:
        g = idx.get((s, d), 0)
        p = pdx.get((s, d), 0)
        if g != 0 and p != 0:
            if g == p:
                TP += 1
            else:
                FP += 1
                FN += 1
        elif g == 0 and p == 0:
            TN += 1
        elif g == 0 and p != 0:
            FP += 1
        elif g != 0 and p == 0:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=precision, recall=recall, structural_accuracy=acc)