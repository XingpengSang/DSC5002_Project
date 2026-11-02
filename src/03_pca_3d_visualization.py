"""DSC5002: Static 3D PCA visualization."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Project paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "annual_11features_raw.csv"
OUTPUT_DIR = ROOT / "outputs" / "pca_visualization"
OUTPUT_PNG = OUTPUT_DIR / "cluster_map_PC1_PC2_PC3_3D.png"
OUTPUT_CSV = ROOT / "data" / "processed" / "pca_3d_scores_with_clusters.csv"

# Features
FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3_8h",
    "temperature", "humidity", "precipitation",
    "wind_speed", "pressure"
]

# Cluster memberships from the original report
CLUSTER_1 = [
    "乌鲁木齐", "兰州", "呼和浩特", "哈尔滨", "天津",
    "太原", "沈阳", "西宁", "西安", "银川", "长春"
]

CLUSTER_2 = [
    "拉萨", "昆明", "海口", "福州", "贵阳"
]

CLUSTER_3 = [
    "上海", "北京", "南京", "南宁", "南昌", "合肥",
    "广州", "成都", "杭州", "武汉", "济南", "石家庄",
    "郑州", "重庆", "长沙"
]

CLUSTER_MAP = {
    **{city: 1 for city in CLUSTER_1},
    **{city: 2 for city in CLUSTER_2},
    **{city: 3 for city in CLUSTER_3}
}

CLUSTER_NAMES = {
    1: "Cluster 1 (11 cities)",
    2: "Cluster 2 (5 cities)",
    3: "Cluster 3 (15 cities)"
}

CLUSTER_COLORS = {
    1: "#2878B5",
    2: "#E07A2D",
    3: "#399B68"
}

CLUSTER_MARKERS = {
    1: "o",
    2: "s",
    3: "^"
}

# PCA sign alignment
SIGN_ANCHORS = ["NO2", "pressure", "O3_8h"]


def main():

    # 1. Load data
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required = ["city"] + FEATURES
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(data) != 31:
        raise ValueError(f"Expected 31 cities, found {len(data)}.")

    data["city"] = data["city"].astype(str).str.strip()

    if data["city"].duplicated().any():
        raise ValueError("Duplicate city names found.")

    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    if data[FEATURES].isna().any().any():
        raise ValueError("Missing or non-numeric feature values found.")

    if not np.isfinite(data[FEATURES].to_numpy()).all():
        raise ValueError("Infinite values found.")

    # 2. Restore cluster labels
    data["Cluster"] = data["city"].map(CLUSTER_MAP)

    if data["Cluster"].isna().any():
        unknown = data.loc[
            data["Cluster"].isna(), "city"
        ].tolist()
        raise ValueError(f"Unknown cities: {unknown}")

    if set(data["city"]) != set(CLUSTER_MAP):
        raise ValueError("City membership does not match the report.")

    # 3. Standardization and PCA
    X_scaled = StandardScaler().fit_transform(data[FEATURES])

    pca = PCA(n_components=3)
    scores = pca.fit_transform(X_scaled)

    # Align PCA component signs with the report
    for i, anchor in enumerate(SIGN_ANCHORS):
        anchor_idx = FEATURES.index(anchor)

        if pca.components_[i, anchor_idx] > 0:
            pca.components_[i, :] *= -1
            scores[:, i] *= -1

    score_df = pd.DataFrame(
        scores,
        columns=["PC1", "PC2", "PC3"]
    )

    result = pd.concat(
        [
            data[["city", "Cluster"]].reset_index(drop=True),
            score_df
        ],
        axis=1
    )

    # 4. Save PCA scores
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # 5. Create 3D scatter plot
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot clusters WITHOUT city labels
    for cluster in [1, 2, 3]:

        subset = result[result["Cluster"] == cluster]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            subset["PC3"],
            c=CLUSTER_COLORS[cluster],
            marker=CLUSTER_MARKERS[cluster],
            s=75,
            alpha=0.88,
            edgecolors="black",
            linewidths=0.55,
            depthshade=True,
            label=CLUSTER_NAMES[cluster]
        )

    # Plot cluster centroids
    for cluster in [1, 2, 3]:

        subset = result[result["Cluster"] == cluster]

        center = subset[
            ["PC1", "PC2", "PC3"]
        ].mean()

        ax.scatter(
            center["PC1"],
            center["PC2"],
            center["PC3"],
            marker="*",
            s=230,
            c=CLUSTER_COLORS[cluster],
            edgecolors="black",
            linewidths=1.0,
            depthshade=False
        )

    # Axis labels
    variance = pca.explained_variance_ratio_ * 100

    ax.set_xlabel(
        f"PC1 ({variance[0]:.2f}%)",
        labelpad=12,
        fontsize=11
    )

    ax.set_ylabel(
        f"PC2 ({variance[1]:.2f}%)",
        labelpad=12,
        fontsize=11
    )

    ax.set_zlabel(
        f"PC3 ({variance[2]:.2f}%)",
        labelpad=10,
        fontsize=11
    )

    ax.set_title(
        "Three-dimensional PCA Visualization of 31 Chinese Cities",
        pad=22,
        fontsize=14
    )

    ax.view_init(elev=24, azim=-55)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True
    )

    ax.grid(True, alpha=0.25)

    fig.text(
        0.5,
        0.025,
        "Cluster labels are taken from the original report; K-means was not rerun.",
        ha="center",
        fontsize=9,
        color="#555555"
    )

    # Save figure
    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
        format="png"
    )

    plt.close(fig)

    print("3D PCA visualization completed.")
    print(f"Figure saved to: {OUTPUT_PNG}")
    print(f"PCA scores saved to: {OUTPUT_CSV}")

    print("\nExplained variance:")

    for i, value in enumerate(variance, start=1):
        print(f"PC{i}: {value:.2f}%")

    print(f"Cumulative: {variance.sum():.2f}%")

    print("\nCluster sizes:")
    print(
        result["Cluster"].value_counts().sort_index()
    )


if __name__ == "__main__":
    main()