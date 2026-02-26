from __future__ import annotations

import argparse
import json

import pandas as pd

from .data import load_csv_timeseries, kmeans_binarize
from .infer import infer_gabni, InferenceParams, summarize_result
from .metrics import edge_metrics_unsigned, edge_metrics_signed
from .utils import set_seed


def main():
    ap = argparse.ArgumentParser(
        prog="gabni",
        description="GABNI reproduction (MIBNI + GA) for Boolean network inference.",
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--mi-min", type=float, default=0.05)
    ap.add_argument("--max-reg-frac", type=float, default=0.6)

    ap.add_argument("--force-ga", action="store_true")
    ap.add_argument("--no-exact-minimize", action="store_true")

    # NOT in paper: optional pruning step
    ap.add_argument("--prune", action="store_true", help="Enable greedy prune_regulators post-process (not in paper).")

    ap.add_argument("--out_edges", default="pred_edges.csv")
    ap.add_argument("--out_summary", default="summary.json")
    ap.add_argument("--gold_edges", default=None)
    ap.add_argument("--cap-edges", type=int, default=None,
                    help="If set, keep only top-K edges by MI(u(t), v(t+1)) after inference (for Table S3-style edge budget).")
    args = ap.parse_args()

    set_seed(args.seed)

    if args.binarize:
        df = pd.read_csv(args.input)
        ds = kmeans_binarize(df, random_state=args.seed)
    else:
        ds = load_csv_timeseries(args.input)

    params = InferenceParams(
        mi_min=args.mi_min,
        max_reg_frac=args.max_reg_frac,
        seed=args.seed,
        exact_minimize=(not args.no_exact_minimize),
        force_ga=args.force_ga,
        prune=args.prune,
    )

    res = infer_gabni(ds, params=params)
    from .edge_postprocess import cap_edges_by_mi
    if args.cap_edges is not None:
        res.signed_edges = cap_edges_by_mi(ds, res.signed_edges, int(args.cap_edges))
    res.signed_edges.to_csv(args.out_edges, index=False)

    summary = summarize_result(ds, res)

    if args.gold_edges:
        gold = pd.read_csv(args.gold_edges)
        summary["structure"] = edge_metrics_unsigned(res.signed_edges, gold, ds.genes)
        summary["structure_signed"] = edge_metrics_signed(res.signed_edges, gold, ds.genes)

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()