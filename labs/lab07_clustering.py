"""Lab 07 — clustering and hierarchical clustering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans


POINTS = np.array(
    [
        [4, 21],
        [5, 19],
        [10, 24],
        [4, 17],
        [3, 16],
        [11, 25],
        [14, 24],
        [6, 22],
        [10, 21],
        [12, 21],
    ]
)


def main() -> None:
    output_dir = Path("artifacts/lab07")
    output_dir.mkdir(parents=True, exist_ok=True)

    linkage_matrix = linkage(POINTS, method="ward", metric="euclidean")

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, distance_sort="ascending", show_leaf_counts=True)
    plt.title("Hierarchical Clustering Dendrogram — Ward Linkage")
    plt.xlabel("Data point index")
    plt.ylabel("Euclidean distance")
    plt.tight_layout()
    plt.savefig(output_dir / "dendrogram.png", dpi=160)
    plt.close()

    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = model.fit_predict(POINTS)

    plt.figure(figsize=(8, 5))
    plt.scatter(POINTS[:, 0], POINTS[:, 1], c=labels)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="x", s=150)
    plt.title("K-means Clusters")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(output_dir / "kmeans_clusters.png", dpi=160)
    plt.close()

    print(f"Saved clustering figures to {output_dir}")


if __name__ == "__main__":
    main()
