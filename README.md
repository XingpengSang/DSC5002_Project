# DSC5002 Group Project — Air Quality Patterns and AQI Prediction

This repository contains our group project on **air quality patterns and AQI prediction across 31 provincial-capital-level cities in mainland China using 2025 data**.

## Project workflow

1. **Annual city-level analysis:** prepare 11 annual features (6 pollutant indicators and 5 meteorological indicators); perform PCA and K-Means clustering; interpret the characteristics of the resulting city clusters.
2. **Daily AQI prediction:** use daily meteorological variables (temperature, humidity, precipitation, wind speed and pressure) to predict AQI. Compare models trained on pooled data within each cluster against models trained on a representative city from each cluster. Same-day pollutant concentrations are excluded from prediction features because AQI is derived from pollutant measurements.
3. **Cluster visualization:** plot cities in PCA space and color the same city points by each of the 11 original annual variables to explore their patterns.

## Repository structure

```text
DSC5002_Project/
├── src/
│   └── 01_preprocessing.py      # Monthly Excel -> annual city-level features
├── data/
│   ├── raw_monthly/            # Original 2025 monthly city Excel files
│   ├── raw_daily/              # Daily data for the prediction stage (when available)
│   └── processed/              # Generated cleaned CSV files
├── results/                    # Future tables and figures
├── requirements.txt            # Python dependencies (if present)
└── README.md
```

## Annual data preprocessing (available now)

**Input:** one AQI Study monthly Excel file per city, named like `aqistudy-北京-月数据-2025.xlsx`, inside `data/raw_monthly/`. The script expects a worksheet named `月数据`, with all 12 months of 2025 and the source column labels matching the Beijing example. Files with unexpected names, missing fields or invalid records are excluded and logged.

**Requirements:** Python 3; `pandas`, `numpy`, `openpyxl`.

Open the repository's root folder in VS Code. In its terminal, run:

```powershell
py -m pip install pandas numpy openpyxl
py src/01_preprocessing.py
```

Or use `python` instead of `py` if that is how Python is configured on your computer.

### Aggregation rules

| Annual feature | Calculation |
| --- | --- |
| PM2.5, PM10, SO2, NO2, CO, O3_8h | Weighted mean of the 12 monthly values, using each month's `有效日数` as its weight |
| Temperature, humidity, wind speed, pressure | Weighted mean using `有效日数` (assuming weather coverage matches that field) |
| Precipitation | Sum of 12 monthly precipitation totals (mm) |

The script checks file names, required columns, 12 unique 2025 months, numeric values, valid-day counts and non-negative precipitation. It does **not** automatically impute missing values. Check the source's metric definitions and coverage before final reporting.

### Output files (`data/processed/`)

| File | Purpose |
| --- | --- |
| `annual_11features_raw.csv` | **Main handoff to the PCA/K-Means teammate:** one row per eligible city; `city` plus 11 unstandardized annual features |
| `annual_with_QC.csv` | Annual features and total valid-day count |
| `monthly_clean.csv` | Accepted, cleaned monthly records |
| `data_quality_issues.csv` | Excluded files/cities and reasons; empty if no issues |

Expected main table: **31 rows × 12 columns** (`city` + 11 features) *only if all 31 input city files pass the checks*. Confirm the terminal's passed-city count and inspect `data_quality_issues.csv` before using the results.

**Important:** `annual_11features_raw.csv` is not standardized. The PCA/K-Means teammate should handle feature scaling, PCA component selection, cluster-number selection and clustering in their own analysis. This preprocessing script does not perform PCA, K-Means or AQI prediction.

## Team handoff

- **Preprocessing → clustering:** use `data/processed/annual_11features_raw.csv`; join later outputs by `city` rather than by row order.
- **Clustering → visualization:** share a table with `city`, `PC1`, `PC2`, `cluster`. This can be joined with the raw annual feature table to create 11 color-coded PCA scatterplots while keeping city positions fixed.
- **Prediction:** requires a separate **daily** dataset. Monthly Excel files in this repository are not sufficient for daily AQI modeling.

## Data source and sharing

Monthly files were downloaded from AQI Study: https://www.aqistudy.cn/. Verify the original dataset's terms and any redistribution restrictions before making raw data or the repository public. The repository's data coverage and derived results should be checked against the source documentation.
