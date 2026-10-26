import json

import numpy as np
import pandas as pd
import pytest

from src import clustering as cl


@pytest.fixture
def features_csv(tmp_path):
    rng = np.random.default_rng(0)
    groups = [(10, 8, 3000), (15, 2, 400), (60, 3, 1200), (120, 1, 150)]
    rows = []
    for g, (r, f, m) in enumerate(groups):
        for i in range(40):
            rows.append({
                "CustomerID": g * 1000 + i,
                "recency_days": max(0, r + rng.normal(0, 3)),
                "frequency": max(1, f + rng.normal(0, 0.3)),
                "monetary": max(1, m * rng.lognormal(0, 0.1)),
                "distinct_products": 5,
            })
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_clustering_writes_artifacts(features_csv, tmp_path):
    out = tmp_path / "results"
    model = tmp_path / "kmeans.joblib"
    meta = cl.run_clustering(features_csv, model, out, k=4)
    assert model.is_file()
    for name in ["k_selection.csv", "cluster_assignments.csv", "segment_summary.csv",
                 "clustering_metadata.json", "elbow_plot.png", "silhouette_plot.png"]:
        assert (out / name).is_file()
    a = pd.read_csv(out / "cluster_assignments.csv")
    assert len(a) == 160 and a["segment"].notna().all()
    assert set(meta["segment_names"].values()) >= {"Loyal Champions", "Lapsed / At-Risk"}
    assert json.loads((out / "clustering_metadata.json").read_text())["k"] == 4

    pipe, names = cl.load_segmenter(model, out)
    assert cl.predict_segment(10, 8, 3000, pipe, names)[1] == "Loyal Champions"
    assert cl.predict_segment(120, 1, 150, pipe, names)[1] == "Lapsed / At-Risk"


def test_predict_segment_rejects_bad_input(features_csv, tmp_path):
    cl.run_clustering(features_csv, tmp_path / "m.joblib", tmp_path / "r", k=4)
    pipe, names = cl.load_segmenter(tmp_path / "m.joblib", tmp_path / "r")
    with pytest.raises(ValueError):
        cl.predict_segment(-1, 2, 100, pipe, names)


def test_load_features_validates(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"CustomerID": [1], "recency_days": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        cl.load_features(p)
    with pytest.raises(FileNotFoundError):
        cl.load_features(tmp_path / "missing.csv")
