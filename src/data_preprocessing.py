import os
import pandas as pd
import numpy as np


# =========================
# 1. 파일 경로 설정
# =========================

environment_path = 'ml_data_sample/environment_data.csv'
growth_path = 'ml_data_sample/growth_data.csv'

environment_output_path = 'outputs/environment_cleaned.csv'
growth_output_path = 'outputs/growth_cleaned.csv'


# =========================
# 2. 출력 폴더 생성
# =========================

os.makedirs('outputs', exist_ok=True)


# =========================
# 3. 데이터 불러오기
# =========================

environment_df = pd.read_csv(environment_path)
growth_df = pd.read_csv(growth_path)


# =========================
# 4. 날짜 컬럼 변환
# =========================

if 'Time' in environment_df.columns:
    environment_df['Time'] = pd.to_datetime(environment_df['Time'], errors='coerce')

if 'Time' in growth_df.columns:
    growth_df['Time'] = pd.to_datetime(growth_df['Time'], errors='coerce')


# =========================
# 5. 예측 대상 컬럼의 이상 대체값 제거
# =========================

target_col = 'NumberOfFruits'
abnormal_target_value = 3.676165803108810

if target_col in growth_df.columns:
    growth_df.loc[
        np.isclose(growth_df[target_col], abnormal_target_value),
        target_col
    ] = np.nan


# =========================
# 6. 결측치 처리 함수
# =========================

def fill_missing_values(df, name, target_col=None):
    print(f"\n===== {name} 결측치 처리 전 =====")
    print(df.isnull().sum())

    if 'Time' in df.columns:
        before = len(df)
        df = df.dropna(subset=['Time'])
        after = len(df)
        print(f"Time 결측 행 제거: {before - after}")

    # 예측 대상 컬럼은 중앙값이나 최빈값으로 채우지 않는다.
    if target_col is not None and target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        after = len(df)
        print(f"{target_col} 결측 행 제거: {before - after}")

    # 수치형 컬럼의 결측치는 중앙값으로 대체
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        if col == target_col:
            continue

        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"{col}: 중앙값 {median_value}로 대체")

    # 문자형 컬럼의 결측치는 최빈값으로 대체
    object_cols = df.select_dtypes(include=['object']).columns

    for col in object_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            print(f"{col}: 최빈값 {mode_value}로 대체")

    print(f"\n===== {name} 결측치 처리 후 =====")
    print(df.isnull().sum())

    return df


# =========================
# 7. 결측치 처리 적용
# =========================

environment_df = fill_missing_values(
    environment_df,
    name='environment_data'
)

growth_df = fill_missing_values(
    growth_df,
    name='growth_data',
    target_col='NumberOfFruits'
)

# =========================
# 8. 환경 데이터 완전 중복 처리
# =========================

print("\n===== 환경 데이터 중복 처리 =====")
print("중복 처리 전 환경 데이터 크기:", environment_df.shape)

environment_duplicate_count = environment_df.duplicated().sum()
print("환경 데이터 완전 중복 행 개수:", environment_duplicate_count)

environment_df = environment_df.drop_duplicates()

print("완전 중복 제거 후 환경 데이터 크기:", environment_df.shape)



# =========================
# 8. 생육 데이터 중복 처리
# =========================

print("\n===== 생육 데이터 중복 처리 =====")
print("중복 처리 전 데이터 크기:", growth_df.shape)

complete_duplicate_count = growth_df.duplicated().sum()
print("완전 중복 행 개수:", complete_duplicate_count)

key_cols = [
    'State',
    'Item',
    'Cropping',
    'FarmNum',
    'Time',
    'CropNum',
    'PlantHeight',
    'NumberOfLeaves',
    'LeafLength',
    'LeafWidth',
    'PetioleLength',
    'StemDiameter'
]

duplicated_rows = growth_df[growth_df.duplicated(subset=key_cols, keep=False)]
print("생육 정보가 같은 중복 행 개수:", len(duplicated_rows))

growth_df = growth_df.drop_duplicates()
print("완전 중복 제거 후 데이터 크기:", growth_df.shape)

growth_df = growth_df.groupby(
    key_cols,
    as_index=False
)['NumberOfFruits'].mean()

print("생육 정보 중복 집계 후 데이터 크기:", growth_df.shape)


# =========================
# 9. 최종 데이터 확인
# =========================

print("\n===== 최종 데이터 확인 =====")

print("\n환경 데이터 크기:", environment_df.shape)
print("환경 데이터 결측치:")
print(environment_df.isnull().sum())

print("\n생육 데이터 크기:", growth_df.shape)
print("생육 데이터 결측치:")
print(growth_df.isnull().sum())

print("\nNumberOfFruits 값 분포:")
print(growth_df['NumberOfFruits'].value_counts().sort_index().head(30))


# =========================
# 10. 전처리 결과 저장
# =========================

environment_df.to_csv(
    environment_output_path,
    index=False,
    encoding='utf-8-sig'
)

growth_df.to_csv(
    growth_output_path,
    index=False,
    encoding='utf-8-sig'
)

print("\n전처리 완료")
print("저장 파일:", environment_output_path)
print("저장 파일:", growth_output_path)