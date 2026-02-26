from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .mi import mutual_information_binary
from .metrics import gene_consistency_and_coverage
from .utils import powerset_bitstrings

DONT_CARE = -1
TIE = -2


@dataclass
class GAParams:
    population_size: int
    max_regulators: int
    mi_min: float = 0.05
    c: int = 28
    H: int = 3
    iterations: int = 1000
    mutation_prob: float = 0.01
    seed: int = 0

@dataclass
class GAResult:
    regulators: list[int]
    consistency: float
    pred: np.ndarray  # length T, includes DONT_CARE/TIE markers for t>=1


def _literal_value(X: np.ndarray, t: int, r: int, neg: bool) -> int:
    v = int(X[t, r])
    return 1 - v if neg else v


def _build_best_predictor(target: np.ndarray, X: np.ndarray, regs: list[int], negated: dict[int, bool]) -> np.ndarray:
    """
    Rule specification (LUT) as in GABNI description:
      For each input pattern b of selected (possibly negated) regulators, set v'(t+1)
      to the most frequent v(t+1) among occurrences of that b.
    Unseen b -> DONT_CARE, tie -> TIE.
    """
    T, _N = X.shape

    if len(regs) == 0:
        # No regulators: unconditional most-likely value of v(t+1)
        c0 = int((target[1:] == 0).sum())
        c1 = int((target[1:] == 1).sum())
        if c0 == c1:
            fill = TIE
        else:
            fill = 1 if c1 > c0 else 0
        pred = np.full(T, fill, dtype=int)
        pred[0] = int(target[0])
        return pred

    k = len(regs)
    # counts[b] = [count0, count1] for target(t+1)
    counts = {b: [0, 0] for b in powerset_bitstrings(k)}

    for t in range(T - 1):
        b = tuple(_literal_value(X, t, r, negated.get(r, False)) for r in regs)
        counts[b][int(target[t + 1])] += 1

    decision: dict[tuple[int, ...], int] = {}
    for b, (c0, c1) in counts.items():
        if c0 + c1 == 0:
            decision[b] = DONT_CARE
        elif c0 == c1:
            decision[b] = TIE
        else:
            decision[b] = 1 if c1 > c0 else 0

    pred = np.full(T, DONT_CARE, dtype=int)
    pred[0] = int(target[0])
    for t in range(T - 1):
        b = tuple(_literal_value(X, t, r, negated.get(r, False)) for r in regs)
        pred[t + 1] = decision[b]
    return pred


def _fitness_from_assignment(
    target: np.ndarray,
    X: np.ndarray,
    regs: list[int],
    negated: dict[int, bool],
    c: int,
) -> tuple[float, float, np.ndarray]:
    pred = _build_best_predictor(target, X, regs, negated)
    match_C, cov = gene_consistency_and_coverage(target, pred)
    C_eff = match_C * cov
    k = len(regs)
    denom = (1.0 - C_eff) * float(c) + float(k)
    fit = 1.0 / max(denom, 1e-12)
    return float(fit), float(C_eff), pred


def _fitness_best_negation(
    target: np.ndarray,
    X: np.ndarray,
    regs: list[int],
    c: int,
) -> tuple[float, float, np.ndarray, dict[int, bool]]:
    """
    For pruning/backstop, evaluate the *best* negation pattern for a fixed regulator set.
    (Used to avoid losing performance when we later only keep regulator indices.)
    """
    if len(regs) == 0:
        fit, C_eff, pred = _fitness_from_assignment(target, X, regs, {}, c)
        return fit, C_eff, pred, {}

    k = len(regs)
    best = (-1.0, -1.0, None, None)  # fit, Ceff, pred, neg
    for mask in range(1 << k):
        neg = {regs[i]: bool((mask >> i) & 1) for i in range(k)}
        fit, C_eff, pred = _fitness_from_assignment(target, X, regs, neg, c)
        if fit > best[0] + 1e-12:
            best = (fit, C_eff, pred, neg)
    return float(best[0]), float(best[1]), np.asarray(best[2]), dict(best[3])


def _repair(chrom: np.ndarray, max_k: int, allowed_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    chrom length = 2N:
      bit 2*i   => use gene i as positive literal
      bit 2*i+1 => use gene i as negated literal
    Constraints:
      - only allowed bits can be 1
      - at most one of (pos,neg) per gene
      - total selected genes <= max_k
    """
    chrom = chrom.copy().astype(np.int8)
    chrom &= allowed_mask.astype(np.int8)

    N2 = chrom.size
    if N2 % 2 != 0:
        raise ValueError("Chromosome length must be even (2N).")
    N = N2 // 2

    # Enforce "not both" per gene
    for i in range(N):
        p = 2 * i
        n = 2 * i + 1
        if chrom[p] == 1 and chrom[n] == 1:
            # randomly keep one
            if rng.random() < 0.5:
                chrom[n] = 0
            else:
                chrom[p] = 0

    # Enforce max_k genes selected
    selected_genes = [i for i in range(N) if chrom[2 * i] == 1 or chrom[2 * i + 1] == 1]
    if len(selected_genes) > max_k:
        drop = rng.choice(np.array(selected_genes, dtype=int), size=len(selected_genes) - max_k, replace=False)
        for i in drop:
            chrom[2 * i] = 0
            chrom[2 * i + 1] = 0

    return chrom


def run_ga_for_target(
    target: np.ndarray,
    X: np.ndarray,
    params: GAParams,
    candidates: list[int] | None = None,
) -> GAResult:
    """
    GA over chromosomes of length 2N (pos/neg literal selection per gene).
    Returns only regulator indices (negations are used internally to build pred).
    """
    T, N = X.shape

    # MI filter computed for candidate genes using lag-1 association u(t) vs v(t+1)
    target_next = target[1:]
    X_t = X[:-1, :]

    base = candidates if candidates is not None else list(range(N))
    mi = np.array([mutual_information_binary(X_t[:, i], target_next) for i in range(N)], dtype=float)

    allowed_genes = np.array([i for i in base if mi[i] >= params.mi_min], dtype=int)
    if allowed_genes.size == 0:
        # No allowed regulators => fallback to empty set with best-neg fitness
        _, C_eff, pred, _neg = _fitness_best_negation(target, X, [], params.c)
        return GAResult(regulators=[], consistency=float(C_eff), pred=np.asarray(pred))

    # allowed_mask over 2N bits
    allowed_mask = np.zeros(2 * N, dtype=np.int8)
    for i in allowed_genes:
        allowed_mask[2 * i] = 1
        allowed_mask[2 * i + 1] = 1

    rng = np.random.default_rng(params.seed)

    pop_size = params.population_size
    max_k = params.max_regulators
    L = 2 * N

    # Initialize population
    pop = np.zeros((pop_size, L), dtype=np.int8)
    for p in range(pop_size):
        k = int(rng.integers(1, max_k + 1))
        chosen = rng.choice(allowed_genes, size=min(k, allowed_genes.size), replace=False)
        for g in chosen:
            # random literal polarity
            if rng.random() < 0.5:
                pop[p, 2 * g] = 1
            else:
                pop[p, 2 * g + 1] = 1
        pop[p] = _repair(pop[p], max_k, allowed_mask, rng)

    def decode(ch: np.ndarray) -> tuple[list[int], dict[int, bool]]:
        regs = []
        neg = {}
        for i in range(N):
            if ch[2 * i] == 1:
                regs.append(i)
                neg[i] = False
            elif ch[2 * i + 1] == 1:
                regs.append(i)
                neg[i] = True
        regs.sort()
        return regs, neg

    fits = np.zeros(pop_size, dtype=float)
    cons = np.zeros(pop_size, dtype=float)
    preds: list[np.ndarray] = [None] * pop_size  # type: ignore

    for i in range(pop_size):
        regs, neg = decode(pop[i])
        f, C_eff, pr = _fitness_from_assignment(target, X, regs, neg, params.c)
        fits[i], cons[i], preds[i] = f, C_eff, pr

    def adjusted(fits_arr: np.ndarray) -> np.ndarray:
        fmax, fmin = float(fits_arr.max()), float(fits_arr.min())
        if abs(fmax - fmin) < 1e-12:
            return np.ones_like(fits_arr)
        a = params.H / (fmax - fmin)
        b = 1.0 - a * fmin
        return a * fits_arr + b

    def select_one() -> int:
        af = adjusted(fits)
        p = af / af.sum()
        return int(rng.choice(np.arange(pop_size), p=p))

    def crossover(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eq = (p1 == p2)
        o1 = np.empty_like(p1)
        o2 = np.empty_like(p2)
        o1[eq] = p1[eq]
        o2[eq] = p1[eq]
        diff = ~eq
        chooser1 = rng.random(L) < 0.5
        chooser2 = rng.random(L) < 0.5
        o1[diff] = np.where(chooser1[diff], p1[diff], p2[diff])
        o2[diff] = np.where(chooser2[diff], p1[diff], p2[diff])
        return o1, o2

    def mutate(ch: np.ndarray) -> np.ndarray:
        flip = rng.random(L) < params.mutation_prob
        out = ch.copy()
        out[flip] = 1 - out[flip]
        return out

    best_idx = int(np.argmax(cons))
    best_c = float(cons[best_idx])
    best_ch = pop[best_idx].copy()
    best_pred = preds[best_idx]

    for _it in range(params.iterations):
        i1, i2 = select_one(), select_one()
        p1, p2 = pop[i1], pop[i2]
        o1, o2 = crossover(p1, p2)
        o1, o2 = mutate(o1), mutate(o2)
        o1 = _repair(o1, max_k, allowed_mask, rng)
        o2 = _repair(o2, max_k, allowed_mask, rng)

        for parent_idx, child in [(i1, o1), (i2, o2)]:
            regs, neg = decode(child)
            f_child, c_child, pr_child = _fitness_from_assignment(target, X, regs, neg, params.c)
            if f_child > fits[parent_idx] + 1e-12:
                pop[parent_idx] = child
                fits[parent_idx] = f_child
                cons[parent_idx] = c_child
                preds[parent_idx] = pr_child

        cur_best = int(np.argmax(cons))
        if cons[cur_best] > best_c + 1e-12:
            best_c = float(cons[cur_best])
            best_ch = pop[cur_best].copy()
            best_pred = preds[cur_best]

    regs_best, _neg_best = decode(best_ch)
    return GAResult(regulators=regs_best, consistency=float(best_c), pred=np.asarray(best_pred))


def prune_regulators(target: np.ndarray, X: np.ndarray, regs: list[int], c: int = 28) -> tuple[list[int], float, np.ndarray]:
    """
    Greedy backward elimination to minimize regulator set without reducing effective consistency.
    Uses best negation pattern for each trial regulator set (so pruning doesn't "forget" negations).
    """
    regs = list(sorted(set(regs)))

    # Evaluate baseline with best-neg for this regulator set
    _fit0, best_Ceff, best_pred, _neg0 = _fitness_best_negation(target, X, regs, c)

    improved = True
    while improved and len(regs) > 1:
        improved = False
        for r in regs.copy():
            trial = [x for x in regs if x != r]
            _fit, Ceff, pred, _neg = _fitness_best_negation(target, X, trial, c)
            if Ceff >= best_Ceff - 1e-12:
                regs, best_Ceff, best_pred = trial, float(Ceff), pred
                improved = True
                break

    return regs, float(best_Ceff), np.asarray(best_pred)