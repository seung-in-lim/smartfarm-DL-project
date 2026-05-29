import argparse
import time
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO


# =========================
# 1. 디바이스 자동 감지
# =========================

def detect_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================
# 2. CLI 인자 파싱
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Strawberry Disease Detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--name", type=str, default="train_default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    return parser.parse_args()


# =========================
# 3. data.yaml의 path를 절대경로로 변환
# =========================

def resolve_data_yaml(data_path, root):
    with open(data_path) as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(data_path.parent)  

    resolved = root / "runs" / "_data_resolved.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    return resolved


# =========================
# 4. 모델 학습 
# =========================

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    data_path = Path(args.data) if args.data else root / "yl_data_sample" / "data.yaml"
    data_yaml = resolve_data_yaml(data_path, root)
    device = detect_device()

    print(f"[INFO] device={device}, epochs={args.epochs}, data={data_yaml}")

    model = YOLO(args.model)
    start = time.time()

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=True,
        patience=args.patience,        
        save=True,
        save_period=10,
        seed=args.seed,
        deterministic=True,            
        cache=True,
        amp=True,
        workers=4,
        project=str(root / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # =========================
    # 5. 결과 출력
    # =========================

    elapsed = (time.time() - start) / 60
    print(f"[INFO] done in {elapsed:.1f} min")
    print(f"[INFO] best weights: {root / 'runs' / 'detect' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
