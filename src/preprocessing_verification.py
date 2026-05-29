import pandas as pd

environment_df = pd.read_csv('outputs/environment_cleaned.csv')
growth_df = pd.read_csv('outputs/growth_cleaned.csv')

print("===== 환경 데이터 확인 =====")
print("데이터 크기:", environment_df.shape)
print("결측치 개수:")
print(environment_df.isnull().sum())
print("완전 중복 행 개수:", environment_df.duplicated().sum())

print("\n===== 생육 데이터 확인 =====")
print("데이터 크기:", growth_df.shape)
print("결측치 개수:")
print(growth_df.isnull().sum())
print("완전 중복 행 개수:", growth_df.duplicated().sum())

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

print("생육 정보 기준 중복 행 개수:")
print(growth_df.duplicated(subset=key_cols).sum())

print("\nNumberOfFruits 값 예시:")
print(growth_df['NumberOfFruits'].value_counts().sort_index().head(30))