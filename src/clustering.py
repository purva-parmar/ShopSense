"""K-means customer segmentation on RFM features (Member 3).

Input:  data/processed/customer_features.csv (from src.features, Member 1).
Outputs:
- models/kmeans_pipeline.joblib            fitted log1p -> StandardScaler -> KMeans pipeline
- results/clustering/k_selection.csv       inertia + silhouette for each k tried
- results/clustering/cluster_assignments.csv  CustomerID, RFM, cluster, segment
- results/clustering/segment_summary.csv   per-segment profile table
- results/clustering/clustering_metadata.json  chosen k, features, segment names, paths
- results/clustering/*.png                 elbow, silhouette, segment plots

Run from the repository root:
    python -m src.clustering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

RANDOM_SEED = 42
IDENTIFIER_COLUMN = "CustomerID"
# Classic RFM: how recently, how often, how much. All three are heavily
# right-skewed, so they are log1p-transformed before scaling.
CLUSTER_FEATURES = ("recency_days", "frequency", "monetary")
K_RANGE = range(2, 11)
DEFAULT_K = 4

FEATURES_PATH = Path("data/processed/customer_features.csv")
MODEL_PATH = Path("models/kmeans_pipeline.joblib")
RESULTS_DIR = Path("results/clustering")


def load_features(path: str | Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the customer feature table and check the RFM columns are usable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Customer feature table not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in (IDENTIFIER_COLUMN, *CLUSTER_FEATURES) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError("CustomerID values must be unique")
    values = df[list(CLUSTER_FEATURES)]
    if values.isna().any().any() or (values < 0).any().any():
        raise ValueError("RFM features must be non-missing and non-negative")
    return df


def build_pipeline(k: int, random_seed: int = RANDOM_SEED) -> Pipeline:
    """log1p -> standard scaling -> K-means."""
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=random_seed)),
        ]
    )


def evaluate_k(X: pd.DataFrame, k_values=K_RANGE, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fit K-means for each k and record inertia (elbow) and silhouette score."""
    rows = []
    for k in k_values:
        pipe = build_pipeline(k, random_seed).fit(X)
        Z = pipe[:-1].transform(X)
        labels = pipe.named_steps["kmeans"].labels_
        rows.append(
            {
                "k": k,
                "inertia": pipe.named_steps["kmeans"].inertia_,
                "silhouette": silhouette_score(Z, labels),
            }
        )
    return pd.DataFrame(rows)


def name_segments(centroids: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a business name from its centroid (original units).

    Clusters are ranked by an RFM score (recent, frequent, high spend = high).
    Best -> "Loyal Champions", worst -> "Lapsed / At-Risk". Middle clusters are
    split by recency: recent ones are "Occasional Shoppers", older ones are
    "Cooling Off". This answers the project question: loyal vs occasional vs
    customers who may have stopped buying.
    """
    ranks = (
        centroids["recency_days"].rank(ascending=False)
        + centroids["frequency"].rank()
        + centroids["monetary"].rank()
    )
    order = list(ranks.sort_values(ascending=False).index)
    names = {order[0]: "Loyal Champions", order[-1]: "Lapsed / At-Risk"}
    middle = order[1:-1]
    median_recency = centroids["recency_days"].median()
    used: dict[str, int] = {}
    for cid in middle:
        base = (
            "Occasional Shoppers"
            if centroids.loc[cid, "recency_days"] <= median_recency
            else "Cooling Off"
        )
        used[base] = used.get(base, 0) + 1
        names[cid] = base if used[base] == 1 else f"{base} {used[base]}"
    return {int(k): v for k, v in names.items()}


def profile_segments(assignments: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Per-segment size, share and mean/median of each feature."""
    g = assignments.groupby(["cluster", "segment"])
    summary = g[list(feature_cols)].mean().add_prefix("mean_")
    summary.insert(0, "customers", g.size())
    summary.insert(1, "share_pct", (summary["customers"] / len(assignments) * 100).round(1))
    med = g[list(CLUSTER_FEATURES)].median().add_prefix("median_")
    summary = summary.join(med).reset_index()
    return summary.sort_values("mean_monetary", ascending=False).round(2)


def _plot_k_selection(scores: pd.DataFrame, chosen_k: int, out: Path) -> list[Path]:
    paths = []
    for col, title, fname in (
        ("inertia", "Elbow method (inertia vs k)", "elbow_plot.png"),
        ("silhouette", "Silhouette score vs k", "silhouette_plot.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(scores["k"], scores[col], marker="o")
        ax.axvline(chosen_k, color="red", linestyle="--", label=f"chosen k = {chosen_k}")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(col.capitalize())
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)
        paths.append(out / fname)
    return paths


def _plot_segments(assignments: pd.DataFrame, pipe: Pipeline, out: Path) -> list[Path]:
    paths = []
    X = assignments[list(CLUSTER_FEATURES)]
    Z = pipe[:-1].transform(X)
    segments = sorted(assignments["segment"].unique())
    colors = dict(zip(segments, plt.cm.tab10.colors))

    # 2-D PCA projection of the scaled RFM space.
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(Z)
    P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    for seg in segments:
        m = (assignments["segment"] == seg).to_numpy()
        ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.6, color=colors[seg], label=seg)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("Customer segments (PCA of scaled log-RFM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "segments_pca.png", dpi=120)
    plt.close(fig)
    paths.append(out / "segments_pca.png")

    # Recency vs monetary in original units (log y axis).
    fig, ax = plt.subplots(figsize=(8, 6))
    for seg in segments:
        d = assignments[assignments["segment"] == seg]
        ax.scatter(d["recency_days"], d["monetary"].clip(lower=1), s=12, alpha=0.6,
                   color=colors[seg], label=seg)
    ax.set_yscale("log")
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Monetary value (log scale)")
    ax.set_title("Recency vs spend by segment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "segments_recency_monetary.png", dpi=120)
    plt.close(fig)
    paths.append(out / "segments_recency_monetary.png")

    # Segment sizes.
    counts = assignments["segment"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=[colors[s] for s in counts.index])
    ax.set_ylabel("Customers")
    ax.set_title("Customers per segment")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out / "segment_sizes.png", dpi=120)
    plt.close(fig)
    paths.append(out / "segment_sizes.png")

    # Box plots of each RFM feature by segment.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, CLUSTER_FEATURES):
        data = [assignments.loc[assignments["segment"] == s, col] for s in segments]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(segments) + 1), segments, rotation=20, ha="right")
        ax.set_title(col)
    fig.suptitle("RFM distribution by segment (outliers hidden)")
    fig.tight_layout()
    fig.savefig(out / "segment_rfm_boxplots.png", dpi=120)
    plt.close(fig)
    paths.append(out / "segment_rfm_boxplots.png")
    return paths


def run_clustering(
    features_path: str | Path = FEATURES_PATH,
    model_path: str | Path = MODEL_PATH,
    results_dir: str | Path = RESULTS_DIR,
    k: int | None = DEFAULT_K,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Full workflow: evaluate k, fit final model, profile, export artifacts.

    k=None picks the k with the highest silhouette score. The default k=4 is
    used because it gives interpretable business segments; the silhouette and
    elbow plots are exported so this choice can be justified in the report.
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_features(features_path)
    X = df[list(CLUSTER_FEATURES)]

    scores = evaluate_k(X, random_seed=random_seed)
    scores.to_csv(out / "k_selection.csv", index=False)
    if k is None:
        k = int(scores.loc[scores["silhouette"].idxmax(), "k"])

    pipe = build_pipeline(k, random_seed).fit(X)
    labels = pipe.named_steps["kmeans"].labels_

    # Centroids back in original units for naming and the UI.
    centers_scaled = pipe.named_steps["kmeans"].cluster_centers_
    centers = np.expm1(pipe.named_steps["scale"].inverse_transform(centers_scaled))
    centroids = pd.DataFrame(centers, columns=CLUSTER_FEATURES)
    segment_names = name_segments(centroids)

    assignments = df.copy()
    assignments["cluster"] = labels
    assignments["segment"] = assignments["cluster"].map(segment_names)
    assignments.to_csv(out / "cluster_assignments.csv", index=False)

    feature_cols = [c for c in df.columns if c != IDENTIFIER_COLUMN]
    summary = profile_segments(assignments, feature_cols)
    summary.to_csv(out / "segment_summary.csv", index=False)

    figures = _plot_k_selection(scores, k, out) + _plot_segments(assignments, pipe, out)
    joblib.dump(pipe, model_path)

    chosen = scores.set_index("k").loc[k]
    metadata = {
        "features": list(CLUSTER_FEATURES),
        "preprocessing": "log1p then StandardScaler",
        "algorithm": "KMeans(n_init=20)",
        "random_seed": random_seed,
        "n_customers": int(len(df)),
        "k": int(k),
        "k_range_tested": [int(v) for v in scores["k"]],
        "silhouette_at_k": round(float(chosen["silhouette"]), 4),
        "best_silhouette_k": int(scores.loc[scores["silhouette"].idxmax(), "k"]),
        "segment_names": {str(c): n for c, n in segment_names.items()},
        "centroids_original_units": centroids.round(2).to_dict(orient="index"),
        "artifacts": {
            "model": str(model_path.as_posix()),
            "assignments": str((out / "cluster_assignments.csv").as_posix()),
            "summary": str((out / "segment_summary.csv").as_posix()),
            "k_selection": str((out / "k_selection.csv").as_posix()),
            "figures": [p.as_posix() for p in figures],
        },
    }
    metadata["centroids_original_units"] = {
        str(c): v for c, v in metadata["centroids_original_units"].items()
    }
    (out / "clustering_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_segmenter(model_path: str | Path = MODEL_PATH, results_dir: str | Path = RESULTS_DIR):
    """For the UI: return (pipeline, {cluster_id: segment_name})."""
    pipe = joblib.load(model_path)
    meta = json.loads((Path(results_dir) / "clustering_metadata.json").read_text())
    return pipe, {int(c): n for c, n in meta["segment_names"].items()}


def predict_segment(
    recency_days: float, frequency: float, monetary: float, pipe=None, names=None
) -> tuple[int, str]:
    """For the UI: assign a single customer to a segment from RFM values."""
    for name, v in (("recency_days", recency_days), ("frequency", frequency), ("monetary", monetary)):
        if v is None or not np.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if pipe is None or names is None:
        pipe, names = load_segmenter()
    row = pd.DataFrame([[recency_days, frequency, monetary]], columns=CLUSTER_FEATURES)
    cluster = int(pipe.predict(row)[0])
    return cluster, names[cluster]


if __name__ == "__main__":
    meta = run_clustering()
    print(f"k = {meta['k']}  silhouette = {meta['silhouette_at_k']}")
    for c, n in meta["segment_names"].items():
        print(f"  cluster {c}: {n}  centroid {meta['centroids_original_units'][c]}")
