# DSC5002 Group Project — Annual Air Quality Data Preprocessing

English | [简体中文](README_CN.md)

## 1. Project Overview

This repository contains the annual data preprocessing component of our DSC5002 group project.

The objective is to construct a city-level dataset for **31 provincial-capital-level cities in mainland China using 2025 data**, providing a consistent input for subsequent Principal Component Analysis (PCA), K-Means clustering and cluster visualization.

The final dataset contains 11 features from two dimensions:

| Dimension | Variables |
|---|---|
| Air pollutants (6) | PM2.5, PM10, SO2, NO2, CO, O3_8h |
| Meteorological variables (5) | Temperature, relative humidity, precipitation, wind speed, atmospheric pressure |

## 2. Project Structure

```text
DSC5002_Project/
│
├── src/
│   └── 01_preprocessing.py
│
├── data/
│   ├── raw_monthly/
│   └── processed/
│
├── README.md
├── README_CN.md
├── requirements.txt
└── .gitignore
```

## 3. Data Source

The original monthly datasets were downloaded from the China Air Quality Online Monitoring and Analysis Platform (AQI Study).

Website: https://www.aqistudy.cn/

Each Excel file contains monthly observations for one city in 2025.

Example filename:

`aqistudy-北京-月数据-2025.xlsx`

The original data should be retained without manual modification to ensure reproducibility.

## 4. Data Preprocessing

The preprocessing script `src/01_preprocessing.py` automatically reads the monthly Excel files and generates annual city-level observations.

### 4.1 Data Cleaning and Quality Control

The following operations are performed:

1. Read the monthly Excel files and extract the required variables.
2. Check whether the required columns are present.
3. Convert the relevant fields into numeric values.
4. Check missing values and invalid observations.
5. Identify duplicated or missing months.
6. Verify whether each city contains a complete set of 12 monthly observations.
7. Record data quality issues separately.

Records that fail the required completeness checks are not silently imputed or included in the final annual dataset.

### 4.2 Annual Aggregation

Different annual aggregation methods are applied according to the characteristics of each variable.

| Variables | Aggregation method |
|---|---|
| PM2.5, PM10, SO2, NO2, CO, O3_8h | Weighted annual mean |
| Temperature, humidity, wind speed, pressure | Weighted annual mean |
| Precipitation | Sum of monthly precipitation |

For variables calculated using weighted averages:

\[
\bar{x}_{annual}=\frac{\sum_{m=1}^{12}x_m n_m}{\sum_{m=1}^{12}n_m}
\]

where:

- \(x_m\) represents the monthly value.
- \(n_m\) represents the number of valid days in month \(m\).

Annual precipitation is calculated as:

\[
P_{annual}=\sum_{m=1}^{12}P_m
\]

The weighted aggregation assumes that the reported valid-day counts are appropriate for the corresponding monthly measurements. This should be checked against the source's data definitions.

### 4.3 Data Standardization

**No standardization is performed during this preprocessing stage.**

The output retains the original numerical values and units. Feature standardization will be conducted separately before PCA and K-Means clustering.

## 5. Preprocessing Results

### 5.1 Beijing Sample

The preprocessing script has been tested using the Beijing monthly dataset for 2025.

The resulting annual features are:

| Variable | Annual value | Unit |
|---|---:|---|
| PM2.5 | 27.537 | μg/m³ |
| PM10 | 51.205 | μg/m³ |
| SO2 | 3.710 | μg/m³ |
| NO2 | 22.189 | μg/m³ |
| CO | 0.520 | mg/m³ |
| O3_8h | 94.619 | μg/m³ |
| Temperature | 14.302 | °C |
| Humidity | 52.544 | % |
| Precipitation | 377.7 | mm |
| Wind speed | 2.643 | m/s |
| Pressure | 1012.310 | hPa |

These values are calculated from the uploaded monthly records rather than retrieved independently as official annual statistics.

The Beijing sample has passed the initial preprocessing test. Complete processing of all 31 cities should be confirmed using the generated quality-control report.

### 5.2 Expected Final Dataset

Once all 31 city files pass the quality checks, the final dataset should contain:

- 31 rows, representing 31 cities.
- 12 columns, including the city identifier and 11 original features.
- One annual observation per city.
- No feature standardization or PCA transformation.

## 6. Output Files

All processed results are saved in:

`data/processed/`

| Filename | Description |
|---|---|
| `annual_11features_raw.csv` | Main annual dataset for PCA and clustering |
| `annual_with_QC.csv` | Annual features and additional quality-control information |
| `monthly_clean.csv` | Cleaned monthly observations |
| `data_quality_issues.csv` | Identified data-quality issues |

**The main file to be shared with the PCA and clustering member is `annual_11features_raw.csv`.**

## 7. How to Run

Open the entire project folder in VS Code.

Install the required Python libraries:

```bash
py -m pip install pandas numpy openpyxl
```

Run the preprocessing script from the project root directory:

```bash
py src/01_preprocessing.py
```

Check the generated CSV files in `data/processed/`.

Before using the annual dataset for PCA, confirm that all 31 cities have passed the quality checks.

## 8. Next Steps and Team Collaboration

The annual preprocessing output will be used for the following tasks:

1. Standardize the 11 annual features.
2. Apply PCA for dimensionality reduction.
3. Use K-Means to group cities based on their PCA scores.
4. Interpret the characteristics of each cluster.
5. Generate PCA scatterplots colored by the original 11 variables.

For visualization, please share the clustering results with at least the following columns:

`city`, `PC1`, `PC2`, `cluster`

These results will be merged with the original annual dataset using the city identifier.

---

**Data source:** AQI Study — https://www.aqistudy.cn/

Please verify the data provider's usage and redistribution terms before making the original datasets publicly available.