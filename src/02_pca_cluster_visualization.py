"""
DSC5002: PCA score reconstruction and cluster visualization.

This script adds a visualization step without changing the original
preprocessing workflow in src/01_preprocessing.py.

Input:
    data/processed/annual_11features_raw.csv

Outputs:
    data/processed/pca_scores_with_clusters.csv
    data/processed/pca_explained_variance.csv
    data/processed/pca_loadings.csv
    outputs/pca_visualization/cluster_map_PC1_PC2.png
    outputs/pca_visualization/variable_colored/*.png
    outputs/pca_visualization/11_variables_panel.png

Important:
    PCA scores are recalculated from the 11 annual features.
    Cluster labels are reconstructed from the city membership lists in the
    teammate's report; K-Means is NOT rerun in this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# 1. Project paths and feature definitions
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "annual_11features_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "pca_visualization"
INDIVIDUAL_DIR = OUTPUT_DIR / "variable_colored"

FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3_8h",
    "temperature", "humidity", "precipitation", "wind_speed", "pressure"
]

FEATURE_LABELS = {
    "PM2.5": "PM2.5 (μg/m³)",
    "PM10": "PM10 (μg/m³)",
    "SO2": "SO2 (μg/m³)",
    "NO2": "NO2 (μg/m³)",
    "CO": "CO (mg/m³)",
    "O3_8h": "O3_8h (μg/m³)",
    "temperature": "Temperature (°C)",
    "humidity": "Relative humidity (%)",
    "precipitation": "Annual precipitation (mm)",
    "wind_speed": "Wind speed (m/s)",
    "pressure": "Atmospheric pressure (hPa)",
}

# Cluster memberships transcribed from the teammate's report.
CLUSTER_1 = [
    "乌鲁木齐", "兰州", "呼和浩特", "哈尔滨", "天津", "太原",
    "沈阳", "西宁", "西安", "银川", "长春"
]
CLUSTER_2 = ["拉萨", "昆明", "海口", "福州", "贵阳"]
CLUSTER_3 = [
    "上海", "北京", "南京", "南宁", "南昌", "合肥", "广州", "成都",
    "杭州", "武汉", "济南", "石家庄", "郑州", "重庆", "长沙"
]

CLUSTER_MAP = {
    **{city: 1 for city in CLUSTER_1},
    **{city: 2 for city in CLUSTER_2},
    **{city: 3 for city in CLUSTER_3},
}

CLUSTER_MARKERS = {1: "o", 2: "s", 3: "^"}
CLUSTER_NAMES = {
    1: "Cluster 1 (11 cities)",
    2: "Cluster 2 (5 cities)",
    3: "Cluster 3 (15 cities)",
}

# Align the arbitrary PCA component signs with the directions shown in
# the teammate's loading table: these anchor loadings should be negative.
SIGN_ANCHORS = ["NO2", "pressure", "O3_8h"]


# ---------------------------------------------------------------------
# 2. Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

# Consistent high-contrast color scale for every feature.
FEATURE_CMAP = LinearSegmentedColormap.from_list(
    "red_white_blue_strong",
    ["#B2182B", "#EF8A62", "#F7F7F7", "#67A9CF", "#2166AC"]
)
FEATURE_NORM = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)


def adjust_annotation_labels(ax, annotations, max_iterations=100):
    """Reduce overlapping city labels by shifting their text offsets."""
    if len(annotations) < 2:
        return

    fig = ax.figure
    for _ in range(max_iterations):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [
            ann.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
            for ann in annotations
        ]
        moved = False

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                a, b = boxes[i], boxes[j]
                overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
                overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
                if overlap_x > 0 and overlap_y > 0:
                    direction = 1 if j % 2 == 0 else -1
                    shift_points = (overlap_y + 2) * 72.0 / fig.dpi
                    x_offset, y_offset = annotations[j].get_position()
                    annotations[j].set_position(
                        (x_offset, y_offset + direction * shift_points)
                    )
                    moved = True

        if not moved:
            break
    fig.canvas.draw()


# ---------------------------------------------------------------------
# 3. Read and validate the annual dataset
# ---------------------------------------------------------------------
def load_annual_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run src/01_preprocessing.py first."
        )

    data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required_columns = ["city"] + FEATURES
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data["city"].duplicated().any():
        duplicates = data.loc[data["city"].duplicated(), "city"].tolist()
        raise ValueError(f"Duplicate city names found: {duplicates}")

    if len(data) != 31:
        raise ValueError(
            f"Expected 31 cities, but found {len(data)}. "
            "Please check the annual preprocessing output."
        )

    unknown_cities = sorted(set(data["city"]) - set(CLUSTER_MAP))
    missing_cities = sorted(set(CLUSTER_MAP) - set(data["city"]))
    if unknown_cities or missing_cities:
        raise ValueError(
            "City names do not match the report's cluster lists.\n"
            f"Unmapped data cities: {unknown_cities}\n"
            f"Cities missing from data: {missing_cities}"
        )

    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    if data[FEATURES].isna().any().any():
        bad = data.loc[data[FEATURES].isna().any(axis=1), ["city"] + FEATURES]
        raise ValueError(f"Missing/non-numeric feature values detected:\n{bad}")

    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Infinite values found in the 11 features.")

    data["Cluster"] = data["city"].map(CLUSTER_MAP).astype(int)
    return data


# ---------------------------------------------------------------------
# 4. Recalculate PCA scores using Z-score standardization
# ---------------------------------------------------------------------
def calculate_pca(data):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data[FEATURES])

    pca = PCA(n_components=3)
    scores = pca.fit_transform(standardized)

    # PCA component signs are mathematically arbitrary. Flip each component
    # if needed so the selected anchor loading is negative, matching the
    # direction convention in the teammate's report.
    for component_index, anchor in enumerate(SIGN_ANCHORS):
        anchor_index = FEATURES.index(anchor)
        if pca.components_[component_index, anchor_index] > 0:
            pca.components_[component_index, :] *= -1
            scores[:, component_index] *= -1

    score_df = pd.DataFrame(
        scores, columns=["PC1", "PC2", "PC3"], index=data.index
    )

    result = pd.concat(
        [data[["city"] + FEATURES + ["Cluster"]], score_df], axis=1
    )

    explained = pd.DataFrame({
        "Component": ["PC1", "PC2", "PC3"],
        "Explained variance ratio": pca.explained_variance_ratio_,
        "Cumulative explained variance": np.cumsum(pca.explained_variance_ratio_),
    })

    # Correlation-style PCA loadings, comparable to the loading values
    # commonly reported as component coefficients * sqrt(eigenvalue).
    loading_matrix = (
        pca.components_.T * np.sqrt(pca.explained_variance_)
    )
    loadings = pd.DataFrame(
        loading_matrix,
        index=FEATURES,
        columns=["PC1", "PC2", "PC3"]
    ).rename_axis("Feature").reset_index()

    return result, explained, loadings


# ---------------------------------------------------------------------
# 5. Plot PC1-PC2 cluster map
# ---------------------------------------------------------------------
def plot_cluster_map(data):
    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster in [1, 2, 3]:
        subset = data[data["Cluster"] == cluster]
        ax.scatter(
            subset["PC1"], subset["PC2"],
            marker=CLUSTER_MARKERS[cluster],
            s=72, alpha=0.85,
            edgecolor="black", linewidth=0.5,
            label=CLUSTER_NAMES[cluster]
        )

    annotations = []
    for _, row in data.iterrows():
        annotations.append(ax.annotate(
            row["city"], (row["PC1"], row["PC2"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, alpha=0.9
        ))
    adjust_annotation_labels(ax, annotations)

    ax.set_title("PCA Projection of 31 Cities (Cluster Labels from Report)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, color="grey", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_map_PC1_PC2.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 6. Plot each original feature as the point color
# ---------------------------------------------------------------------
def plot_feature_map(data, feature):
    values = data[feature].astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0:
        raise ValueError(f"{feature} has zero variance; cannot create a color scale.")

    # Standardize each feature for color mapping only. PCA scores are
    # unchanged. Red = below city mean; white = around mean; blue = above.
    z_values = (values - mean) / std
    norm = FEATURE_NORM

    fig, ax = plt.subplots(figsize=(9, 7))

    for cluster in [1, 2, 3]:
        subset = data[data["Cluster"] == cluster]
        positions = subset.index
        ax.scatter(
            subset["PC1"], subset["PC2"],
            c=z_values.loc[positions],
            cmap=FEATURE_CMAP, norm=norm,
            marker=CLUSTER_MARKERS[cluster],
            s=78, alpha=0.9,
            edgecolor="black", linewidth=0.45
        )

    annotations = []
    for idx, row in data.iterrows():
        annotations.append(ax.annotate(
            row["city"], (row["PC1"], row["PC2"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, alpha=0.9
        ))
    adjust_annotation_labels(ax, annotations)

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=FEATURE_CMAP)
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=ax, pad=0.02)
    cbar.set_label(f"{FEATURE_LABELS[feature]} (z-score for color)")

    cluster_handles = [
        Line2D(
            [0], [0], marker=CLUSTER_MARKERS[cluster], color="white",
            markerfacecolor="lightgrey", markeredgecolor="black",
            markersize=8, linestyle="None", label=CLUSTER_NAMES[cluster]
        )
        for cluster in [1, 2, 3]
    ]
    ax.legend(handles=cluster_handles, loc="best", frameon=True)

    ax.set_title(f"PC1-PC2 Projection Colored by {FEATURE_LABELS[feature]}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, color="grey", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.7, alpha=0.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    safe_name = feature.replace(".", "_")
    fig.savefig(INDIVIDUAL_DIR / f"PC1_PC2_by_{safe_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 7. Create a compact 11-panel overview
# ---------------------------------------------------------------------
def plot_all_features_panel(data):
    # A 3x4 layout gives the 11 variables more room and uses the last cell
    # to explain the cluster symbols and shared color scale.
    fig, axes = plt.subplots(3, 4, figsize=(21, 15))
    axes = axes.flatten()

    for ax, feature in zip(axes, FEATURES):
        values = data[feature].astype(float)
        z_values = (values - values.mean()) / values.std(ddof=0)

        for cluster in [1, 2, 3]:
            subset = data[data["Cluster"] == cluster]
            ax.scatter(
                subset["PC1"], subset["PC2"],
                c=z_values.loc[subset.index],
                cmap=FEATURE_CMAP, norm=FEATURE_NORM,
                marker=CLUSTER_MARKERS[cluster],
                s=48, alpha=0.95,
                edgecolor="black", linewidth=0.35
            )

        ax.set_title(FEATURE_LABELS[feature], fontsize=11, pad=8)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.axhline(0, color="grey", linewidth=0.65, alpha=0.45)
        ax.axvline(0, color="grey", linewidth=0.65, alpha=0.45)
        ax.grid(alpha=0.16)

    # Use the 12th cell for a concise legend and color interpretation.
    axes[-1].axis("off")
    cluster_handles = [
        Line2D(
            [0], [0], marker=CLUSTER_MARKERS[cluster], color="white",
            markerfacecolor="lightgrey", markeredgecolor="black",
            markersize=8, linestyle="None", label=CLUSTER_NAMES[cluster]
        )
        for cluster in [1, 2, 3]
    ]
    axes[-1].legend(
        handles=cluster_handles, loc="upper left",
        frameon=False, fontsize=10, title="Cluster groups",
        title_fontsize=11
    )
    axes[-1].text(
        0.02, 0.56,
        "Color: standardized feature value (Z-score)\\n"
        "Red: below the 31-city mean\\n"
        "White: near the 31-city mean\\n"
        "Blue: above the 31-city mean\\n\\n"
        "Shared color scale: -2 to +2.\\n"
        "Values outside the range use end colors.",
        transform=axes[-1].transAxes,
        va="top", ha="left", fontsize=9, linespacing=1.55
    )

    fig.suptitle(
        "PCA City Projection Colored by 11 Original Features",
        fontsize=16, y=0.985
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, top=0.93, bottom=0.065,
        wspace=0.28, hspace=0.34
    )
    fig.savefig(OUTPUT_DIR / "11_variables_panel.png", bbox_inches="tight")
    plt.close(fig)


# 8. Main
# ---------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_annual_data()
    result, explained, loadings = calculate_pca(data)

    # Save numeric outputs for later interpretation and reporting.
    result.to_csv(
        PROCESSED_DIR / "pca_scores_with_clusters.csv",
        index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    explained.to_csv(
        PROCESSED_DIR / "pca_explained_variance.csv",
        index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    loadings.to_csv(
        PROCESSED_DIR / "pca_loadings.csv",
        index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    plot_cluster_map(result)
    for feature in FEATURES:
        plot_feature_map(result, feature)
    plot_all_features_panel(result)

    print("PCA visualization completed.")
    print(f"Cities processed: {len(result)}")
    print("\nExplained variance:")
    print(explained.to_string(index=False))
    print("\nCluster sizes:")
    print(result["Cluster"].value_counts().sort_index().to_string())
    print("\nCluster mean PCA scores:")
    print(
        result.groupby("Cluster")[["PC1", "PC2", "PC3"]]
        .mean().round(3).to_string()
    )
    print(f"\nTables saved to: {PROCESSED_DIR}")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
