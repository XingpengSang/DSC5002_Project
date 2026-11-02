
"""
DSC5002 - Interactive 3D PCA Visualization with Variable-based Coloring.

Features:
1. 31 cities displayed in the PC1-PC2-PC3 coordinate system.
2. A dropdown menu switches between 11 original indicators.
3. Point colors represent standardized Z-scores.
4. Red = below average, White = near average, Blue = above average.
5. City names, cluster membership, PCA scores and Z-scores
   are available through hover information.
6. The 3D coordinates remain fixed when switching indicators.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "annual_11features_raw.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pca_visualization"
)

OUTPUT_HTML = (
    OUTPUT_DIR
    / "Interactive_3D_PCA_Variable_Coloring.html"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca_3d_scores_with_clusters.csv"
)


# ============================================================
# 2. Dataset settings
# ============================================================

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


# Original cluster memberships from the group report

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
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3"
}


# PCA component sign anchors
SIGN_ANCHORS = [
    "NO2",
    "pressure",
    "O3_8h"
]


# Fixed initial camera
INITIAL_CAMERA = {
    "eye": {
        "x": 1.55,
        "y": 1.55,
        "z": 1.25
    }
}


# Consistent red-white-blue color scale
COLOR_SCALE = [
    [0.0, "#C94C4C"],
    [0.5, "#F5F5F5"],
    [1.0, "#3976C6"]
]

COLOR_MIN = -2
COLOR_MAX = 2


# ============================================================
# 3. Load and validate data
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
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

    # Restore original cluster labels
    data["Cluster"] = data["city"].map(CLUSTER_MAP)

    if data["Cluster"].isna().any():
        unknown = data.loc[
            data["Cluster"].isna(),
            "city"
        ].tolist()

        raise ValueError(
            f"Unknown cities: {unknown}"
        )

    if set(data["city"]) != set(CLUSTER_MAP.keys()):
        raise ValueError(
            "City membership does not match the original report."
        )

    return data


# ============================================================
# 4. Standardization and PCA
# ============================================================

def calculate_pca(data):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        data[FEATURES]
    )

    # Store standardized original variables
    standardized_data = pd.DataFrame(
        X_scaled,
        columns=FEATURES,
        index=data.index
    )

    pca = PCA(n_components=3)

    scores = pca.fit_transform(X_scaled)

    # Align component signs with the original report
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

    return result, standardized_data, pca


# ============================================================
# 5. Create interactive 3D variable-coloring figure
# ============================================================

def create_3d_figure(result, standardized_data, pca):

    variance = (
        pca.explained_variance_ratio_ * 100
    )

    fig = go.Figure()

    # One trace for each original indicator.
    # All traces use exactly the same PCA coordinates.

    for feature_index, feature in enumerate(FEATURES):

        z_values = standardized_data[feature].to_numpy()

        custom_data = np.column_stack(
            [
                result["city"].to_numpy(),
                result["Cluster"].to_numpy(),
                result["PC1"].to_numpy(),
                result["PC2"].to_numpy(),
                result["PC3"].to_numpy()
            ]
        )

        hover_template = (
            "<b>%{customdata[0]}</b><br>"
            "Cluster: %{customdata[1]}<br>"
            "<br>"
            "<b>PCA Scores</b><br>"
            "PC1: %{customdata[2]:.3f}<br>"
            "PC2: %{customdata[3]:.3f}<br>"
            "PC3: %{customdata[4]:.3f}<br>"
            "<br>"
            f"<b>{feature}</b>: "
            "%{marker.color:.3f} (Z-score)"
            "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter3d(

                x=result["PC1"],
                y=result["PC2"],
                z=result["PC3"],

                mode="markers",

                name=feature,

                visible=(feature_index == 0),

                marker=dict(
                    size=7,

                    color=z_values,

                    colorscale=COLOR_SCALE,

                    cmin=COLOR_MIN,
                    cmax=COLOR_MAX,

                    showscale=True,

                    colorbar=dict(
                        title=dict(
                            text="Z-score",
                            side="right"
                        ),
                        thickness=18,
                        len=0.72,
                        x=1.02,
                        tickvals=[-2, -1, 0, 1, 2]
                    ),

                    opacity=0.9,

                    line=dict(
                        color="black",
                        width=0.5
                    )
                ),

                customdata=custom_data,

                hovertemplate=hover_template
            )
        )

    # Dropdown menu: switch the coloring indicator

    buttons = []

    for feature_index, feature in enumerate(FEATURES):

        visibility = [
            i == feature_index
            for i in range(len(FEATURES))
        ]

        buttons.append(
            dict(
                label=feature,

                method="update",

                args=[
                    {
                        "visible": visibility
                    },
                    {
                        "title.text": (
                            "Interactive 3D PCA Visualization"
                            f"<br><sup>Color represents "
                            f"{feature} (standardized Z-score)</sup>"
                        )
                    }
                ]
            )
        )

    # Layout

    fig.update_layout(

        title=dict(
            text=(
                "Interactive 3D PCA Visualization"
                "<br><sup>Color represents PM2.5 "
                "(standardized Z-score)</sup>"
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

        updatemenus=[
            dict(
                type="dropdown",

                direction="down",

                x=0.02,
                y=1.12,

                xanchor="left",
                yanchor="top",

                buttons=buttons,

                showactive=True,

                bgcolor="white",

                bordercolor="gray",

                borderwidth=1
            )
        ],

        annotations=[
            dict(
                text="<b>Select indicator:</b>",
                x=0.02,
                y=1.17,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(size=13)
            )
        ],

        template="plotly_white",

        width=1150,
        height=900,

        margin=dict(
            l=0,
            r=80,
            b=0,
            t=160
        ),

        hoverlabel=dict(
            font_size=13
        )
    )

    return fig, variance


# ============================================================
# 6. Main
# ============================================================

def main():

    print("=" * 65)
    print("DSC5002 - Interactive 3D PCA Variable Coloring")
    print("=" * 65)

    data = load_data()

    print(
        f"Loaded data: {len(data)} cities, "
        f"{len(FEATURES)} indicators."
    )

    result, standardized_data, pca = calculate_pca(
        data
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save PCA scores and original cluster labels
    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    fig, variance = create_3d_figure(
        result,
        standardized_data,
        pca
    )

    # Export self-contained HTML
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
                "filename": "Interactive_3D_PCA_Variable_Coloring",
                "height": 900,
                "width": 1150,
                "scale": 2
            }
        }
    )

    print("\nCompleted successfully.")

    print(f"Interactive HTML: {OUTPUT_HTML}")
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

    print("\nInteractive functions:")
    print("1. Select one of the 11 indicators from the dropdown.")
    print("2. Red = below average.")
    print("3. White = near average.")
    print("4. Blue = above average.")
    print("5. Drag to rotate the 3D plot.")
    print("6. Scroll to zoom in or out.")
    print("7. Hover to inspect city names, cluster and PCA scores.")
    print("8. The PCA coordinates remain unchanged when switching indicators.")


if __name__ == "__main__":
    main()
