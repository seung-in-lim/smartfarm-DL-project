import os
import pandas as pd


# =========================
# 1. 파일 경로 설정
# =========================

environment_path = 'outputs/environment_cleaned.csv'
growth_path = 'outputs/growth_cleaned.csv'

output_path = 'outputs/integrated_growth_environment_dataset.csv'


# =========================
# 2. 출력 폴더 생성
# =========================

os.makedirs('outputs', exist_ok=True)


# =========================
# 3. 데이터 불러오기
# =========================

environment_df = pd.read_csv(environment_path)
growth_df = pd.read_csv(growth_path)

environment_df['Time'] = pd.to_datetime(environment_df['Time'])
growth_df['Time'] = pd.to_datetime(growth_df['Time'])

print("환경 데이터 크기:", environment_df.shape)
print("생육 데이터 크기:", growth_df.shape)


# =========================
# 4. 환경 변수 컬럼 설정
# =========================

environment_features = [
    'ExTemperature',
    'OutInsolation',
    'InTemperature',
    'InHumidity',
    'RemainCO2',
    'SoilTemperature'
]


# =========================
# 5. 환경 데이터를 일별 단위로 요약
# =========================

group_cols = [
    'State',
    'Item',
    'Cropping',
    'FarmNum',
    'Time'
]

daily_environment_df = environment_df.groupby(
    group_cols,
    as_index=False
)[environment_features].mean()

print("일별 환경 데이터 크기:", daily_environment_df.shape)


# =========================
# 6. 생육 측정일 이전 환경 데이터 요약 함수
# =========================

def add_past_environment_features(growth_df, daily_environment_df, window_days=7):
    result_rows = []

    for _, growth_row in growth_df.iterrows():
        state = growth_row['State']
        item = growth_row['Item']
        cropping = growth_row['Cropping']
        farm_num = growth_row['FarmNum']
        growth_date = growth_row['Time']

        start_date = growth_date - pd.Timedelta(days=window_days)

        env_window = daily_environment_df[
            (daily_environment_df['State'] == state) &
            (daily_environment_df['Item'] == item) &
            (daily_environment_df['Cropping'] == cropping) &
            (daily_environment_df['FarmNum'] == farm_num) &
            (daily_environment_df['Time'] >= start_date) &
            (daily_environment_df['Time'] < growth_date)
        ]

        new_row = growth_row.to_dict()
        new_row[f'EnvDataCount_{window_days}d'] = len(env_window)

        for col in environment_features:
            new_row[f'{col}_{window_days}d_mean'] = env_window[col].mean()
            new_row[f'{col}_{window_days}d_min'] = env_window[col].min()
            new_row[f'{col}_{window_days}d_max'] = env_window[col].max()

        new_row[f'OutInsolation_{window_days}d_sum'] = env_window['OutInsolation'].sum()

        result_rows.append(new_row)

    return pd.DataFrame(result_rows)


# =========================
# 7. 과거 7일 환경 feature 생성
# =========================

window_days = 7

integrated_df = add_past_environment_features(
    growth_df,
    daily_environment_df,
    window_days=window_days
)

print("통합 전 데이터 크기:", integrated_df.shape)


# =========================
# 8. 환경 데이터가 붙지 않은 행 제거
# =========================

before_rows = len(integrated_df)

integrated_df = integrated_df[
    integrated_df[f'EnvDataCount_{window_days}d'] > 0
]

after_rows = len(integrated_df)

print("환경 데이터가 없는 생육 행 제거 개수:", before_rows - after_rows)
print("통합 후 데이터 크기:", integrated_df.shape)


# =========================
# 9. 남은 결측치 확인 및 제거
# =========================

print("\n통합 데이터 결측치 개수:")
print(integrated_df.isnull().sum())

before_dropna = len(integrated_df)
integrated_df = integrated_df.dropna()
after_dropna = len(integrated_df)

print("결측치 포함 행 제거 개수:", before_dropna - after_dropna)
print("최종 통합 데이터 크기:", integrated_df.shape)


# =========================
# 10. 저장
# =========================

integrated_df.to_csv(
    output_path,
    index=False,
    encoding='utf-8-sig'
)

print("\n통합 데이터셋 생성 완료")
print("저장 파일:", output_path)