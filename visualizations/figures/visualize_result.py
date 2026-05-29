import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 파일 경로 설정
# =========================

growth_importance_path = 'outputs/growth_feature_importance.csv'
integrated_importance_path = 'outputs/integrated_feature_importance.csv'

figure_dir = 'visualizations/figures'

os.makedirs(figure_dir, exist_ok=True)


# =========================
# 2. 모델 성능 비교 데이터
# =========================

performance_df = pd.DataFrame({
    'Model': [
        'Growth-only Model',
        'Integrated Model'
    ],
    'MAE': [
        1.9011864025154204,
        1.5312308362915659
    ],
    'RMSE': [
        2.468213219230803,
        2.0336918584633192
    ],
    'R2 Score': [
        0.3401392740135192,
        0.6212448690096205
    ]
})

# =========================
# 3. 모델 성능 비교 시각화
# =========================

metrics = ['MAE', 'RMSE', 'R2 Score']

for metric in metrics:
    plt.figure(figsize=(7, 5))
    plt.bar(performance_df['Model'], performance_df[metric])
    plt.title(f'{metric} Comparison')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()

    save_path = os.path.join(
        figure_dir,
        f'{metric.lower().replace(" ", "_")}_comparison.png'
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f'저장 완료: {save_path}')


# =========================
# 4. 성능 비교 통합 그래프
# =========================

performance_plot_df = performance_df.set_index('Model')[['MAE', 'RMSE', 'R2 Score']]

plt.figure(figsize=(9, 6))
performance_plot_df.plot(kind='bar')
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.tight_layout()

save_path = os.path.join(figure_dir, 'model_performance_comparison.png')
plt.savefig(save_path, dpi=300)
plt.close()

print(f'저장 완료: {save_path}')


# =========================
# 5. 생육 기반 모델 변수 중요도 시각화
# =========================

growth_importance_df = pd.read_csv(growth_importance_path)
growth_importance_df = growth_importance_df.sort_values(
    by='importance',
    ascending=True
)

plt.figure(figsize=(8, 6))
plt.barh(
    growth_importance_df['feature'],
    growth_importance_df['importance']
)
plt.title('Growth-only Model Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

save_path = os.path.join(figure_dir, 'growth_feature_importance.png')
plt.savefig(save_path, dpi=300)
plt.close()

print(f'저장 완료: {save_path}')


# =========================
# 6. 통합 모델 변수 중요도 상위 20개 시각화
# =========================

integrated_importance_df = pd.read_csv(integrated_importance_path)

top20_importance_df = integrated_importance_df.sort_values(
    by='importance',
    ascending=False
).head(20)

top20_importance_df = top20_importance_df.sort_values(
    by='importance',
    ascending=True
)

plt.figure(figsize=(10, 8))
plt.barh(
    top20_importance_df['feature'],
    top20_importance_df['importance']
)
plt.title('Integrated Model Feature Importance Top 20')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

save_path = os.path.join(figure_dir, 'integrated_feature_importance_top20.png')
plt.savefig(save_path, dpi=300)
plt.close()

print(f'저장 완료: {save_path}')


# =========================
# 7. 성능 비교 CSV 저장
# =========================

performance_output_path = 'outputs/model_performance_comparison.csv'

performance_df.to_csv(
    performance_output_path,
    index=False,
    encoding='utf-8-sig'
)

print(f'성능 비교 CSV 저장 완료: {performance_output_path}')
print('\n데이터 시각화 완료')