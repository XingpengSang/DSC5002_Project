"""
DSC5002 - Interactive 3D PCA visualization using Plotly.

The scatter points do not display city labels.
City names and PCA scores remain available through hover information.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "annual_11features_raw.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pca_visualization"

OUTPUT_HTML = (
    OUTPUT_DIR / "cluster_map_PC1_PC2_PC3_3D.html"
)

OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed"
    / "pca_3d_scores_with_clusters.csv"
)


# ---------------------------------------------------------------------
# 2. Dataset settings
# ---------------------------------------------------------------------

FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3_8h",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
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

CLUSTER_LABELS = {
    1: "Cluster 1 (11 cities)",
    2: "Cluster 2 (5 cities)",
    3: "Cluster 3 (15 cities)"
}

CLUSTER_COLORS = {
    1: "#2878B5",
    2: "#E07A2D",
    3: "#399B68"
}

CLUSTER_SYMBOLS = {
    1: "circle",
    2: "square",
    3: "diamond"
}

SIGN_ANCHORS = ["NO2", "pressure", "O3_8h"]

INITIAL_CAMERA = {
    "eye": {
        "x": 1.55,
        "y": 1.55,
        "z": 1.25
    }
}


# ---------------------------------------------------------------------
# 3. Load and validate data
# ---------------------------------------------------------------------

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    required_columns = ["city"] + FEATURES

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if len(data) != 31:
        raise ValueError(
            f"Expected 31 cities, found {len(data)}."
        )

    data["city"] = (
        data["city"].astype(str).str.strip()
    )

    if data["city"].duplicated().any():
        raise ValueError("Duplicate city names found.")

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data[FEATURES].isna().any().any():
        raise ValueError(
            "Missing or non-numeric feature values found."
        )

    if not np.isfinite(
        data[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Infinite values found.")

    # Restore cluster labels from the report
    data["Cluster"] = data["city"].map(CLUSTER_MAP)

    if data["Cluster"].isna().any():
        unknown_cities = data.loc[
            data["Cluster"].isna(),
            "city"
        ].tolist()

        raise ValueError(
            f"Unknown cities in cluster lists: {unknown_cities}"
        )

    if set(data["city"]) != set(CLUSTER_MAP.keys()):
        raise ValueError(
            "City membership does not match the original report."
        )

    return data


# ---------------------------------------------------------------------
# 4. Standardization and PCA
# ---------------------------------------------------------------------

def calculate_pca(data):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        data[FEATURES]
    )

    pca = PCA(n_components=3)

    scores = pca.fit_transform(X_scaled)

    # Align component signs with the report
    for i, anchor in enumerate(SIGN_ANCHORS):

        anchor_index = FEATURES.index(anchor)

        if pca.components_[i, anchor_index] > 0:

            pca.components_[i, :] *= -1
            scores[:, i] *= -1

    score_data = pd.DataFrame(
        scores,
        columns=["PC1", "PC2", "PC3"],
        index=data.index
    )

    result = pd.concat(
        [
            data[["city", "Cluster"]],
            score_data
        ],
        axis=1
    )

    return result, pca


# ---------------------------------------------------------------------
# 5. Create interactive 3D figure
# ---------------------------------------------------------------------

def create_3d_figure(result, pca):

    variance = pca.explained_variance_ratio_ * 100

    fig = go.Figure()

    # Plot each cluster
    for cluster_id in [1, 2, 3]:

        subset = result[
            result["Cluster"] == cluster_id
        ]

        custom_data = np.column_stack(
            [
                subset["city"].to_numpy(),
                subset["Cluster"].astype(str).to_numpy()
            ]
        )

        fig.add_trace(
            go.Scatter3d(

                x=subset["PC1"],
                y=subset["PC2"],
                z=subset["PC3"],

                # Only markers: no city names displayed on the plot
                mode="markers",

                name=CLUSTER_LABELS[cluster_id],

                marker=dict(
                    size=7,
                    color=CLUSTER_COLORS[cluster_id],
                    symbol=CLUSTER_SYMBOLS[cluster_id],
                    opacity=0.88,
                    line=dict(
                        color="black",
                        width=0.7
                    )
                ),

                customdata=custom_data,

                # City names appear only when hovering
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + CLUSTER_LABELS[cluster_id]
                    + "<br>PC1: %{x:.3f}"
                    + "<br>PC2: %{y:.3f}"
                    + "<br>PC3: %{z:.3f}"
                    + "<extra></extra>"
                )
            )
        )

    # Add cluster centroids
    for cluster_id in [1, 2, 3]:

        subset = result[
            result["Cluster"] == cluster_id
        ]

        centroid = subset[
            ["PC1", "PC2", "PC3"]
        ].mean()

        fig.add_trace(
            go.Scatter3d(

                x=[centroid["PC1"]],
                y=[centroid["PC2"]],
                z=[centroid["PC3"]],

                mode="markers",

                name=f"Cluster {cluster_id} centroid",

                marker=dict(
                    size=12,
                    color=CLUSTER_COLORS[cluster_id],
                    symbol="diamond",
                    line=dict(
                        color="black",
                        width=1.5
                    )
                ),

                hovertemplate=(
                    f"<b>Cluster {cluster_id} centroid</b><br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "PC3: %{z:.3f}"
                    "<extra></extra>"
                )
            )
        )

    # Layout
    fig.update_layout(

        title=dict(
            text=(
                "Three-dimensional PCA Visualization of 31 Chinese Cities"
                "<br><sup>Points are colored by the cluster memberships "
                "reported in the group analysis</sup>"
            ),
            x=0.5,
            xanchor="center"
        ),

        scene=dict(

            xaxis_title=(
                f"PC1 ({variance[0]:.2f}% variance)"
            ),

            yaxis_title=(
                f"PC2 ({variance[1]:.2f}% variance)"
            ),

            zaxis_title=(
                f"PC3 ({variance[2]:.2f}% variance)"
            ),

            camera=INITIAL_CAMERA,

            aspectmode="cube",

            xaxis=dict(
                showbackground=True,
                backgroundcolor="rgb(245,245,245)"
            ),

            yaxis=dict(
                showbackground=True,
                backgroundcolor="rgb(245,245,245)"
            ),

            zaxis=dict(
                showbackground=True,
                backgroundcolor="rgb(245,245,245)"
            )
        ),

        legend=dict(
            title="Cluster membership",
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.85)"
        ),

        template="plotly_white",

        width=1100,
        height=850,

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=100
        ),

        hoverlabel=dict(font_size=13)
    )

    return fig, variance


# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------

def main():

    print("=" * 65)
    print("DSC5002 - Interactive 3D PCA Visualization")
    print("=" * 65)

    data = load_data()

    print(
        f"Loaded data: {len(data)} cities, "
        f"{len(FEATURES)} variables."
    )

    result, pca = calculate_pca(data)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    fig, variance = create_3d_figure(
        result,
        pca
    )

    # Save a self-contained HTML file
    fig.write_html(

        str(OUTPUT_HTML),

        include_plotlyjs=True,

        full_html=True,

        config={

            "responsive": True,

            "scrollZoom": True,

            "displaylogo": False,

            "toImageButtonOptions": {
                "format": "png",
                "filename": "PCA_3D_Cluster_Visualization",
                "height": 850,
                "width": 1100,
                "scale": 2
            }
        }
    )

    print("\nCompleted successfully.")

    print(f"Interactive 3D HTML: {OUTPUT_HTML}")
    print(f"PCA scores CSV: {OUTPUT_CSV}")

    print("\nExplained variance:")

    for i, value in enumerate(variance, start=1):
        print(f"PC{i}: {value:.2f}%")

    print(
        f"Cumulative: {variance.sum():.2f}%"
    )

    print("\nCluster sizes:")

    print(
        result["Cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nHow to use the interactive figure:")

    print("1. Open the HTML file in Chrome or Edge.")
    print("2. Drag with the left mouse button to rotate.")
    print("3. Use the mouse wheel to zoom.")
    print("4. Hover over a point to inspect its city and PCA scores.")
    print("5. Click a legend item to hide or show a cluster.")


if __name__ == "__main__":
    main()