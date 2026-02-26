import pandas as pd
from gabni.data import load_csv_timeseries
from gabni.infer import infer_gabni, InferenceParams

def test_smoke_yeast():
    ds = load_csv_timeseries("datasets/yeast_cell_cycle_timeseries.csv")
    res = infer_gabni(ds, InferenceParams(seed=0, ga_iterations=50))  # fast
    assert set(res.genes) == set(ds.genes)
    assert res.signed_edges.shape[0] >= 0
