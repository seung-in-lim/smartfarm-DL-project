import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# =========================
# 1. 파일 경로 설정
# =========================

input_path = 'outputs/growth_cleaned.csv'
model_output_path = 'models/growth_random_forest_model.pkl'
importance_output_path = 'outputs/growth_feature_importance.csv'


# =========================
# 2. 출력 폴더 생성
# =========================

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


# =========================
# 3. 데이터 불러오기
# =========================

growth_df = pd.read_csv(input_path)

print("데이터 크기:", growth_df.shape)
print("컬럼 목록:")
print(growth_df.columns)


# =========================
# 4. 입력 변수와 예측 대상 설정
# =========================

target_col = 'NumberOfFruits'

feature_cols = [
    'Cropping',
    'FarmNum',
    'CropNum',
    'PlantHeight',
    'NumberOfLeaves',
    'LeafLength',
    'LeafWidth',
    'PetioleLength',
    'StemDiameter'
]

X = growth_df[feature_cols]
y = growth_df[target_col]


# =========================
# 5. 학습 데이터와 검증 데이터 분리
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("학습 데이터 크기:", X_train.shape)
print("검증 데이터 크기:", X_valid.shape)


# =========================
# 6. 모델 생성 및 학습
# =========================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 7. 모델 예측 및 평가
# =========================

y_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_valid, y_pred)

print("\n===== 모델 평가 결과 =====")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =========================
# 8. 변수 중요도 확인
# =========================

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n===== 변수 중요도 =====")
print(importance_df)

importance_df.to_csv(
    importance_output_path,
    index=False,
    encoding='utf-8-sig'
)


# =========================
# 9. 모델 저장
# =========================

joblib.dump(model, model_output_path)

print("\n모델 학습 완료")
print("저장된 모델:", model_output_path)
print("변수 중요도 저장:", importance_output_path)