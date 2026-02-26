# GABNI (reproduction)

Clean-room, from-paper reimplementation of **GABNI** (Genetic algorithm-based Boolean network inference) from:

- Shohag Barman, Yung-Keun Kwon (2018) *A Boolean network inference from time-series gene expression data using a genetic algorithm*. **Bioinformatics** 34(17):i927–i934.  
  DOI: 10.1093/bioinformatics/bty584

This repository provides:
- a reproducible CLI (`gabni`)
- the yeast cell-cycle benchmark dataset (Supplementary Table S2)
- the yeast gold network (29 directed interactions, no self-loops)
- commands to reproduce the best results obtained in this reimplementation

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Data

Included datasets:

- `datasets/yeast_cell_cycle_timeseries_s2.csv`  
  Boolean time-series for **11 genes** across **13 time points** (from **Supplementary Table S2**).

- `datasets/yeast_cell_cycle_gold_edges_no_self.csv`  
  Gold standard **signed directed** edges (**29 interactions**) excluding self-loops (matching the yeast evaluation protocol used in the paper).

> The CSV loader ignores non-gene columns such as `Time` and `Phase` if present.

---

## Best reproducible result (reference run)

This is the **best configuration found** during parameter/seed sweeps for this reimplementation, balancing:
- **perfect dynamics accuracy** (1.0)
- **very low FP** (2)
- highest TP achieved under these constraints

### Command

```bash
mkdir -p results

gabni --seed 0 --force-ga --no-exact-minimize --prune \
  --max-reg-frac 0.4 \
  --input datasets/yeast_cell_cycle_timeseries_s2.csv \
  --gold_edges datasets/yeast_cell_cycle_gold_edges_no_self.csv \
  --out_edges results/pred_edges_seed0.csv \
  --out_summary results/summary_seed0.json
```

### Expected output (summary)

From `results/summary_seed0.json`:

- `dynamics_accuracy = 1.0`
- `structure` (directed unsigned): **TP=15, FP=2, FN=14**
- `num_edges = 17`

---

## Running on your own data

### Boolean inputs
Provide a CSV where rows are time points and columns are genes (0/1 values):

```bash
gabni --seed 0 \
  --input path/to/your_timeseries.csv \
  --out_edges pred_edges.csv \
  --out_summary summary.json
```

### Continuous inputs (with discretization)
Enable KMeans binarization per gene:

```bash
gabni --seed 0 --binarize \
  --input path/to/your_expression_timeseries.csv \
  --out_edges pred_edges.csv \
  --out_summary summary.json
```

---

## Metrics (when gold network is provided)

If you pass `--gold_edges`, the JSON summary reports:

- `structure`: directed **unsigned** metrics (presence/absence only), intended to align with Table S3 protocol
- `structure_signed`: directed **signed** metrics (wrong sign counts as both FP and FN)

Example:

```bash
gabni --seed 0 \
  --input datasets/yeast_cell_cycle_timeseries_s2.csv \
  --gold_edges datasets/yeast_cell_cycle_gold_edges_no_self.csv \
  --out_edges pred_edges.csv \
  --out_summary summary.json
```

---

## Comparison to the paper (yeast, noise = 0)

The paper reports (Supplementary Table S3) for yeast cell-cycle at 0% noise:

- `dynamics_accuracy = 1.0`
- approximately **TP=18, FP=3, FN=11**

This reimplementation reproduces **perfect dynamics** on yeast (`dynamics_accuracy=1.0`) and yields a **sparser inferred network** with high precision, but lower recall in structure recovery.

---

## Notes on discrepancy (TP/FN vs Table S3)

Exact structural recovery can differ from Table S3 due to:

- **Short time-series (T=13) under-determination:** many networks reproduce identical dynamics.
- **Parsimony pressure:** this implementation explicitly encourages compact regulator sets (fitness includes regulator-count penalty; optional pruning), which can increase precision but reduce recall.
- **Missing low-level details:** the original implementation is unavailable; some GA/selection/tie-breaking details are not fully specified in the paper.

---

## Seed sweep (sanity check)

To reproduce the seed sweep used during development:

```bash
mkdir -p results

for s in 0 1 2 3 4 5 6 7 8 9; do
  gabni --seed "$s" --force-ga --no-exact-minimize --prune \
    --max-reg-frac 0.4 \
    --input datasets/yeast_cell_cycle_timeseries_s2.csv \
    --gold_edges datasets/yeast_cell_cycle_gold_edges_no_self.csv \
    --out_summary "results/sweep_${s}.json" > /dev/null
  python -c "import json; s=$s; d=json.load(open(f'results/sweep_{s}.json')); print(s, d['num_edges'], d['dynamics_accuracy'], d['structure']['TP'], d['structure']['FP'], d['structure']['FN'])"
done
```

---
