import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# =========================
# 1. 파일 경로 설정
# =========================

input_path = 'outputs/integrated_growth_environment_dataset.csv'

model_output_path = 'models/integrated_random_forest_model.pkl'
importance_output_path = 'outputs/integrated_feature_importance.csv'
result_output_path = 'outputs/integrated_model_result.csv'


# =========================
# 2. 출력 폴더 생성
# =========================

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


# =========================
# 3. 통합 데이터 불러오기
# =========================

data = pd.read_csv(input_path)

print("통합 데이터 크기:", data.shape)
print("컬럼 목록:")
print(data.columns.tolist())


# =========================
# 4. 예측 대상 설정
# =========================

target_col = 'NumberOfFruits'


# =========================
# 5. 입력 변수 설정
# =========================

drop_cols = [
    'State',
    'Item',
    'Time',
    target_col
]

X = data.drop(columns=drop_cols)
y = data[target_col]

print("\n입력 변수 개수:", X.shape[1])
print("입력 변수 목록:")
print(X.columns.tolist())


# =========================
# 6. 학습 데이터와 검증 데이터 분리
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n학습 데이터 크기:", X_train.shape)
print("검증 데이터 크기:", X_valid.shape)


# =========================
# 7. 모델 생성 및 학습
# =========================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 8. 모델 예측 및 평가
# =========================

y_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_valid, y_pred)

print("\n===== 환경 + 생육 기반 모델 평가 결과 =====")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =========================
# 9. 평가 결과 저장
# =========================

result_df = pd.DataFrame({
    'model': ['integrated_random_forest'],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2]
})

result_df.to_csv(
    result_output_path,
    index=False,
    encoding='utf-8-sig'
)


# =========================
# 10. 변수 중요도 확인 및 저장
# =========================

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n===== 변수 중요도 상위 20개 =====")
print(importance_df.head(20))

importance_df.to_csv(
    importance_output_path,
    index=False,
    encoding='utf-8-sig'
)


# =========================
# 11. 모델 저장
# =========================

joblib.dump(model, model_output_path)

print("\n환경 + 생육 기반 모델 학습 완료")
print("저장된 모델:", model_output_path)
print("변수 중요도 저장:", importance_output_path)
print("모델 평가 결과 저장:", result_output_path)