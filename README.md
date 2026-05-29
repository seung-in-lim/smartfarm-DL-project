# 스마트팜 딥러닝 프로젝트 (Smart Farm Deep Learning Project)

> 테이블형 데이터 회귀와 객체 탐지(Object Detection)를 함께 다룬 머신러닝/딥러닝 학습 프로젝트입니다.

---

## 프로젝트 개요

- 머신러닝과 딥러닝을 직접 손으로 익혀보고 싶어 시작한 학습 프로젝트입니다.
- 두 가지 다른 문제 유형을 한 프로젝트에서 경험하는 것을 목표로 했습니다.
  - **회귀(regression)** — 테이블형 데이터로 연속값(과실 수)을 예측합니다.
  - **객체 탐지(object detection)** — 이미지에서 병해 영역을 탐지합니다.
- 데이터 전처리부터 모델 학습·평가까지 전 과정을 스스로 설계하며 ML 파이프라인 전체를 이해하는 데 중점을 두었습니다.

---

## 주요 결과

### Track A — 과실 수 회귀

| 모델 | 사용 특징 | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---:|---:|---:|
| 생육 단독 | 형태학적 특징 9개 | 1.90 | 2.47 | 0.34 |
| **통합** | + 7일 환경 윈도우 | **1.53** | **2.03** | **0.62** |

- 환경 윈도우를 추가하여 R²가 약 2배 상승했습니다.
- 누적 일사량과 7일 평균 내부 온도가 가장 기여도가 높았으며, 이는 과실 형성 시점과 생물학적으로 연결됩니다.

### Track B — 딸기 병해 탐지

YOLOv8n을 COCO 사전학습 가중치에서 7개 클래스(4,918장)에 대해 파인튜닝했습니다.

| 실행 | Epoch | 하드웨어 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---|---:|---:|---:|---:|
| v1 (초기 검증) | 3 | M2 CPU | 0.677 | 0.685 | 0.722 | 0.513 |
| **v2 (본 학습)** | **50** | **Kaggle T4** | **0.934** | **0.897** | **0.952** | **0.814** |

- v1은 로컬환경에서 파이프라인 동작 확인용 짧은 실행이며, v2가 본 학습입니다.
- 추론 속도는 2.1ms/image(약 476 FPS)입니다.

**클래스별 결과 (v2)**

| 클래스 | 샘플 수 | mAP50 | mAP50-95 |
|---|---:|---:|---:|
| Angular Leafspot | 232 | 0.988 | 0.864 |
| Anthracnose Fruit Rot | 71 | 0.892 | 0.681 |
| Blossom Blight | 117 | 0.994 | 0.870 |
| Gray Mold | 215 | 0.925 | 0.699 |
| Leaf Spot | 808 | 0.980 | 0.926 |
| Powdery Mildew Fruit | 134 | 0.931 | 0.781 |
| Powdery Mildew Leaf | 698 | 0.952 | 0.875 |

- 샘플이 가장 적은 Anthracnose Fruit Rot(71개)이 가장 낮은 성능을 보였으며, 클래스 불균형이 성능에 직접적인 영향을 주는 것을 확인했습니다.

**시각화 자료** (`visualizations/figures/yolo/`)

| 파일 | 내용 |
|---|---|
| `results.png` | epoch별 loss/metric 학습 곡선 |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | 클래스 혼동 행렬 |
| `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` | Precision/Recall/F1-Confidence 곡선 |
| `labels.jpg` | 클래스 분포 (불균형 확인) |
| `val_batch*_pred.jpg` | 검증 세트 실제 탐지 결과 |

---

## 아키텍처

```
                       원본 CSV (환경 71,683행 + 생육 1,411행)
                              │
                              ▼
                  data_preprocessing.py
                  (결측치 / 이상값 / 중복 처리)
                              │
                              ▼
                  make_integrated_dataset.py
                  (7일 환경 윈도우: mean/min/max/sum)
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
          생육 단독 RF                  통합 RF
          (베이스라인)                 (생육 + 환경)
                └─────────────┬─────────────┘
                              ▼
                  비교 평가 (MAE / RMSE / R²)

       ══════════════ Track B (병렬) ══════════════

         원본 이미지 (4,918장, YOLO 포맷)
                              │
                              ▼
                  yl_train_model.py
                  (YOLOv8n, MPS/CUDA 자동 감지)
                              │
                              ▼
                  best.pt + Precision/Recall/mAP
```

---

## 방법론

### Track A — 테이블형 ML

**전처리** (`data_preprocessing.py`)

- `NumberOfFruits`에서 반복되는 `3.676165803108810` 값을 이전 결측치 대체 산물로 판단하여 제거했습니다.
- 수치형은 중앙값, 범주형은 최빈값으로 결측치를 대체했습니다.
- 같은 식물·같은 날짜의 중복 측정은 평균으로 집계했습니다.

**특징 공학** (`make_integrated_dataset.py`)

- 생육 측정일 기준 과거 7일 환경 데이터를 mean/min/max로 요약했습니다(일사량은 누적합 추가).
- 환경 조건이 생장에 미치는 누적 효과를 모델에 반영하기 위한 설계입니다.

**모델**

- Random Forest Regressor를 사용했습니다(300 estimators, max_depth=10).
- 소규모 테이블 데이터에 강건하고, 특징 중요도 분석이 가능하며, 별도의 스케일링이 필요 없습니다.

### Track B — 객체 탐지

| 하이퍼파라미터 | 값 |
|---|---|
| Epochs | 50 |
| 이미지 크기 | 640 |
| Batch | 16 |
| Optimizer | AdamW |
| LR 스케줄 | Cosine (lr0=0.001) |
| Patience | 15 (early stopping) |
| Seed | 42 |

---

## 저장소 구조

```
smartfarm-DL-project/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_preprocessing.py
│   ├── preprocessing_verification.py
│   ├── make_integrated_dataset.py
│   ├── train_growth_model.py
│   ├── train_integrated_model.py
│   └── yl_train_model.py
├── ml_data_sample/        # 원본 테이블 데이터 (커밋 제외)
├── yl_data_sample/        # YOLO 이미지 데이터 (커밋 제외)
│   └── data.yaml          # 환경 독립적 상대경로 설정
├── outputs/               # 정제된 CSV, 특징 중요도 테이블
├── visualizations/
│   └── figures/
│       ├── ...            # Track A (테이블 ML) 시각화
│       └── yolo/          # Track B (YOLO) 학습 곡선·혼동 행렬·탐지 결과
├── models/                # 학습된 가중치 (best.pt는 용량 문제로 커밋 제외)
├── runs/                  # YOLO 학습 산출물 (커밋 제외)
└── docs/
```

---

## 재현 방법

### 환경

- macOS 14+ (Apple Silicon M2, MPS) / Kaggle Notebook (CUDA, T4) / Python 3.10+

### 설치

```bash
git clone https://github.com/seung-in-lim/smartfarm-DL-project.git
cd smartfarm-DL-project
pip install -r requirements.txt
```

### Track A

```bash
python src/data_preprocessing.py
python src/preprocessing_verification.py
python src/make_integrated_dataset.py
python src/train_growth_model.py
python src/train_integrated_model.py
```

### Track B

```bash
# 로컬 (M2, MPS)
python src/yl_train_model.py --epochs 50 --name local_run

# 클라우드 (Kaggle T4, CUDA)
python src/yl_train_model.py --epochs 50 --name kaggle_run
```

- 디바이스를 자동으로 감지합니다 (CUDA → MPS → CPU). 하이퍼파라미터는 CLI로 변경할 수 있습니다.
- 학습된 가중치(`best.pt`)는 용량 문제로 저장소에 포함하지 않았습니다. 위 명령으로 재현할 수 있습니다 (seed=42 고정, 약 35분 / Kaggle T4 기준).

### 데이터셋

- 대용량 데이터는 gitignore로 처리했습니다.
- **테이블 데이터**: 한국 스마트팜 공개 데이터셋입니다. `environment_data.csv`, `growth_data.csv`를 `ml_data_sample/`에 배치하시면 됩니다.
- **이미지 데이터**: [Roboflow strawberry-disease-detection-dataset v4](https://universe.roboflow.com/strawberry-disease/strawberry-disease-detection-dataset/dataset/4) (CC BY 4.0)입니다. YOLOv8 포맷으로 받아 `yl_data_sample/`에 압축 해제하시면 됩니다.

---

## 기여

프로젝트로 전 단계에 걸쳐 직접 설계·구현했습니다.

- 데이터 엔지니어링: CSV 프로파일링, 결측치 대체 산물 식별, 중복 처리 전략을 수립했습니다.
- 특징 공학: 7일 윈도우 연산을 정의하고 요약 통계량을 선택했습니다.
- 모델 설계: 베이스라인과 통합 모델을 동일한 조건으로 비교했습니다.
- 객체 탐지: YOLOv8n 학습을 구성하고, 재현성 설정(seed, deterministic), 디바이스 자동 감지, CLI 인자 지원을 구현했습니다.
- 분석: 특징 중요도 비교, 성능 시각화, 클래스별 결과 분석을 수행했습니다.

---

## 한계 및 향후 개선 방향

**한계**

- Track A: `FarmNum`, `CropNum`이 식별자이므로 데이터 누수(leakage) 가능성이 있음. 농가 단위 hold-out 평가가 필요.
- Track B: Anthracnose Fruit Rot 등 소수 클래스의 성능이 낮으며, 클래스 불균형을 아직 처리하지 못함.
- Track B: 현재 검증 세트 성능만 보고했으며, 테스트 세트에 대한 별도 평가가 필요.

**향후 개선 방향**

- 통합 모델의 농가 단위 그룹 교차 검증을 진행.
- 클래스 불균형을 완화 (가중 샘플링 / focal loss).
- 윈도우 크기 ablation을 수행 (1 / 3 / 7 / 14 / 30일).
- YOLOv8 상위 변형(s/m)과 비교하여 정확도와 속도 차이를 분석.

## 📮 Contact

- **이름**: 임승인 
- **소속**: 호서대학교 자동차ICT학과
- **E-MAIL**: rhrhtmddls@naver.com
- **프로젝트 기간**: 2024.06.24 ~ 2024.09.19