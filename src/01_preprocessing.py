"""DSC 5002: 2025 monthly city sheets -> UNSTANDARDIZED annual 11-feature table.
Run from any directory: python src/01_preprocessing.py
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw_monthly'
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

COLUMN_MAP = {
    'PM₂.₅ (μg/m³)': 'PM2.5', 'PM₁₀ (μg/m³)': 'PM10',
    'SO₂ (μg/m³)': 'SO2', 'NO₂ (μg/m³)': 'NO2',
    'CO (mg/m³)': 'CO', 'O₃ · 8h (μg/m³)': 'O3_8h',
    '温度 (°C)': 'temperature', '湿度 (%)': 'humidity',
    '降水 (mm)': 'precipitation', '风速 (m/s)': 'wind_speed',
    '气压 (hPa)': 'pressure', '有效日数': 'valid_days', '月份': 'month'
}
FEATURES = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3_8h',
            'temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
MEAN_FEATURES = [name for name in FEATURES if name != 'precipitation']
MONTHS = {f'2025-{i:02d}' for i in range(1, 13)}
issues = []
records = []
monthly_clean = []
seen_cities = set()
files = sorted(f for f in RAW.glob('*.xlsx') if not f.name.startswith('~$'))
if not files:
    raise SystemExit(f'No XLSX files found. Put city files in: {RAW}')

for file in files:
    match = re.fullmatch(r'aqistudy-(.+?)-月数据-2025', file.stem)
    if not match:
        issues.append({'file': file.name, 'city': '', 'issue': 'Unexpected filename; skipped'})
        continue
    city = match.group(1).strip()
    if city in seen_cities:
        issues.append({'file': file.name, 'city': city, 'issue': 'Duplicate city file; skipped'})
        continue
    seen_cities.add(city)
    try:
        df = pd.read_excel(file, sheet_name='月数据')
    except Exception as exc:
        issues.append({'file': file.name, 'city': city, 'issue': f'Cannot read sheet 月数据: {exc}'})
        continue
    df.columns = df.columns.astype(str).str.strip()
    missing_columns = set(COLUMN_MAP) - set(df.columns)
    if missing_columns:
        issues.append({'file': file.name, 'city': city, 'issue': f'Missing columns: {sorted(missing_columns)}'})
        continue
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    df['month'] = df['month'].astype(str).str.strip()
    # Reject out-of-year or malformed dates, rather than silently ignoring observations.
    if len(df) != 12 or set(df['month']) != MONTHS or df['month'].duplicated().any():
        missing = sorted(MONTHS - set(df['month']))
        unexpected = sorted(set(df['month']) - MONTHS)
        issues.append({'file': file.name, 'city': city,
                       'issue': f'Month check failed (rows={len(df)}; missing={missing}; unexpected={unexpected})'})
        continue
    for name in FEATURES + ['valid_days']:
        df[name] = pd.to_numeric(df[name], errors='coerce')
    missing = df[FEATURES + ['valid_days']].isna().sum()
    missing = missing[missing > 0].to_dict()
    if missing:
        issues.append({'file': file.name, 'city': city, 'issue': f'Missing/non-numeric: {missing}'})
        continue
    days_in_month = pd.to_datetime(df['month'] + '-01').dt.days_in_month
    if ((df['valid_days'] <= 0) | (df['valid_days'] > days_in_month) |
            (df['valid_days'] % 1 != 0)).any():
        issues.append({'file': file.name, 'city': city, 'issue': 'Invalid valid_days'})
        continue
    if (df['precipitation'] < 0).any():
        issues.append({'file': file.name, 'city': city, 'issue': 'Negative monthly precipitation'})
        continue
    df = df.sort_values('month').reset_index(drop=True)
    df.insert(0, 'city', city)
    monthly_clean.append(df)
    weights = df['valid_days'].to_numpy(dtype=float)
    row = {'city': city}
    for name in MEAN_FEATURES:
        row[name] = float(np.average(df[name], weights=weights))
    # Source monthly precipitation must represent MONTHLY TOTAL (mm).
    row['precipitation'] = float(df['precipitation'].sum())
    row['total_valid_days'] = int(weights.sum())
    records.append(row)

issue_df = pd.DataFrame(issues, columns=['file', 'city', 'issue'])
issue_df.to_csv(OUT / 'data_quality_issues.csv', index=False, encoding='utf-8-sig')
if not records:
    raise SystemExit(f'No city passed quality checks. See {OUT / "data_quality_issues.csv"}')
annual = pd.DataFrame(records).sort_values('city').reset_index(drop=True)
annual[['city'] + FEATURES].to_csv(OUT / 'annual_11features_raw.csv', index=False,
                                    encoding='utf-8-sig', float_format='%.6f')
annual[['city', 'total_valid_days'] + FEATURES].to_csv(OUT / 'annual_with_QC.csv', index=False,
                                                       encoding='utf-8-sig', float_format='%.6f')
pd.concat(monthly_clean, ignore_index=True).to_csv(OUT / 'monthly_clean.csv',
                                                    index=False, encoding='utf-8-sig')
print(f'Files found: {len(files)}; cities passed: {len(annual)}; QC issues: {len(issues)}')
print('Output:', OUT / 'annual_11features_raw.csv')
if len(annual) != 31:
    print('WARNING: Expected 31 cities, but obtained', len(annual), '— review the issue report and input list.')
print('No standardization, PCA or clustering was performed.')
