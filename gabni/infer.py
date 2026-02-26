from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TimeSeriesDataset
from .mibni import run_mibni
from .ga import GAParams, run_ga_for_target, prune_regulators
from .postprocess import minimize_regulators_exact
from .metrics import dynamics_accuracy


@dataclass
class InferenceParams:
    mi_min: float = 0.05
    max_reg_frac: float = 0.6
    ga_iterations: int = 1000
    mutation_prob: float = 0.01
    H: int = 3
    c: int = 28
    seed: int = 0
    exact_minimize: bool = True
    force_ga: bool = False
    prune: bool = False  # optional; not in paper


@dataclass
class InferenceResult:
    genes: list[str]
    regulators: dict[str, list[str]]
    pred_matrix: np.ndarray
    gene_consistency: dict[str, float]
    signed_edges: pd.DataFrame  # src,dst,sign


def _choose_better(
    a_regs: list[int], a_pred: np.ndarray, a_c: float,
    b_regs: list[int], b_pred: np.ndarray, b_c: float,
) -> tuple[list[int], np.ndarray, float]:
    """Pick better solution by consistency, then by smaller regulator count."""
    if b_c > a_c + 1e-12:
        return b_regs, b_pred, b_c
    if a_c > b_c + 1e-12:
        return a_regs, a_pred, a_c
    # tie: choose smaller k
    if len(b_regs) < len(a_regs):
        return b_regs, b_pred, b_c
    return a_regs, a_pred, a_c


def infer_gabni(ds: TimeSeriesDataset, params: InferenceParams) -> InferenceResult:
    X = ds.X.astype(np.int8)
    T, N = X.shape

    max_k = max(1, int(round(params.max_reg_frac * N)))
    pop_size = 200

    np.random.seed(params.seed)

    regulators: dict[str, list[str]] = {}
    gene_consistency: dict[str, float] = {}
    pred_mat = np.zeros_like(X, dtype=int)

    for j, gene in enumerate(ds.genes):
        target = X[:, j].astype(int)
        candidates = [i for i in range(N) if i != j]

        # Always compute MIBNI baseline
        mib = run_mibni(target=target, X=X, mi_min=params.mi_min, max_k=max_k, candidates=candidates)
        regs_idx, pred, c_val = mib.regulators, mib.pred, float(mib.consistency)

        # Optionally run GA, but keep the better of (MIBNI, GA)
        if params.force_ga or (mib.consistency < 1.0 - 1e-12):
            ga_params = GAParams(
                population_size=pop_size,
                max_regulators=max_k,
                mi_min=params.mi_min,
                c=params.c,
                H=params.H,
                iterations=5000,
                mutation_prob=params.mutation_prob,
                seed=params.seed,
            )
            ga = run_ga_for_target(target=target, X=X, params=ga_params, candidates=candidates)
            regs_idx, pred, c_val = _choose_better(
                regs_idx, pred, c_val,
                ga.regulators, ga.pred, float(ga.consistency)
            )

        # Exact minimize: ACCEPT empty regulator set if it achieves perfect consistency
        if params.exact_minimize:
            regs2, pred2, c2 = minimize_regulators_exact(
                target=target, X=X, candidates=candidates, mi_min=params.mi_min, max_k=max_k
            )
            if c2 >= 1.0 - 1e-12:
                regs_idx, pred, c_val = _choose_better(
                    regs_idx, pred, c_val,
                    regs2, pred2, float(c2)
                )

        # Optional pruning (not in paper)
        if params.prune and c_val >= 1.0 - 1e-12:
            regs_idx, c_val, pred = prune_regulators(target=target, X=X, regs=regs_idx, c=params.c)

        regulators[gene] = [ds.genes[i] for i in regs_idx]
        gene_consistency[gene] = float(c_val)
        pred_mat[:, j] = pred

    # Sign typing (u(t), v(t+1))
    edges: list[tuple[str, str, int]] = []
    for dst, regs in regulators.items():
        j = ds.genes.index(dst)
        v = X[:, j]
        for src in regs:
            i = ds.genes.index(src)
            u = X[:, i]
            c00 = c01 = c10 = c11 = 0
            for t in range(T - 1):
                vt1 = int(v[t + 1])
                ut = int(u[t])
                if vt1 == 0 and ut == 0:
                    c00 += 1
                elif vt1 == 0 and ut == 1:
                    c01 += 1
                elif vt1 == 1 and ut == 0:
                    c10 += 1
                else:
                    c11 += 1
            sign = 1 if (c00 + c11) >= (c01 + c10) else -1
            edges.append((src, dst, sign))

    signed_edges = pd.DataFrame(edges, columns=["src", "dst", "sign"])

    return InferenceResult(
        genes=ds.genes,
        regulators=regulators,
        pred_matrix=pred_mat,
        gene_consistency=gene_consistency,
        signed_edges=signed_edges,
    )


def summarize_result(ds: TimeSeriesDataset, res: InferenceResult) -> dict:
    return {
        "N": ds.N,
        "T": ds.T,
        "mean_gene_consistency": float(np.mean(list(res.gene_consistency.values()))),
        "dynamics_accuracy": dynamics_accuracy(ds.X, res.pred_matrix),
        "num_edges": int(res.signed_edges.shape[0]),
    }