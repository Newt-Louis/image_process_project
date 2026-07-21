"""Cấu hình dùng chung cho toàn bộ demo + thực nghiệm 3 model detector.

Mọi hằng số ảnh hưởng tới việc so sánh công bằng giữa 3 model đều nằm ở đây
(Mục 2 - "Chuẩn hóa điều kiện so sánh" trong HUONG_DAN_THUC_NGHIEM_3_MODEL_DETECTOR.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Đường dẫn ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_DIR = APP_DIR / "data" / "gold_dataset"
TEST_IMAGES_DIR = DATA_DIR / "images" / "test"
TEST_ANN_FILE = DATA_DIR / "coco_annotations" / "instances_test.json"

RESULTS_DIR = APP_DIR / "results"
UPLOADS_DIR = APP_DIR / "data" / "uploads"

WEIGHTS = {
    "yolov11": PROJECT_DIR / "yolov11" / "yolov11_product" / "weights" / "best.pt",
    "fasterrcnn": PROJECT_DIR / "fasterrcnn" / "best.pth",
    "retinanet": PROJECT_DIR / "retinanet" / "best.pth",
}

# Thứ tự hiển thị/ghi bảng thống nhất trong mọi output
MODEL_ORDER = ["yolov11", "fasterrcnn", "retinanet"]

DISPLAY_NAMES = {
    "yolov11": "YOLOv11",
    "fasterrcnn": "Faster R-CNN",
    "retinanet": "RetinaNet",
}

# Màu vẽ bbox (BGR) cho từng model - dùng thống nhất ở demo lẫn ảnh định tính
MODEL_COLORS_BGR = {
    "yolov11": (60, 220, 60),      # xanh lá
    "fasterrcnn": (240, 160, 40),  # xanh dương
    "retinanet": (80, 80, 240),    # đỏ
    "gt": (200, 200, 200),         # xám - ground truth
}

# --- Điều kiện so sánh chuẩn hóa (Mục 2) ----------------------------------
SEED = 42

# Ngưỡng confidence "vận hành": dùng cho demo, visualize, và tính P/R/F1/TP/FP/FN.
CONF_THRESHOLD = 0.25

# Ngưỡng confidence "thô": dùng khi sinh prediction để pycocotools quét toàn dải.
# KHÔNG lọc theo CONF_THRESHOLD trước khi tính mAP.
CONF_RAW = 0.001

# Ngưỡng IoU cho NMS - ép giống nhau cho cả 3 model.
NMS_IOU = 0.5

# Số detection tối đa/ảnh. Ảnh RPC nhiều nhất 20 vật thể nên 100 là dư.
MAX_DET = 100

# IoU dùng khi ghép TP/FP/FN thủ công (đồng nhất với mAP@0.50)
MATCH_IOU = 0.5

# --- Độ phân giải input của từng model ------------------------------------
# YOLO: letterbox về 640x640 (đúng như lúc train, xem yolov11_product/args.yaml)
YOLO_IMGSZ = 640
# torchvision GeneralizedRCNNTransform: resize giữ tỉ lệ, cạnh ngắn 800, cạnh dài <= 1333
TV_MIN_SIZE = 800
TV_MAX_SIZE = 1333

# --- Benchmark ------------------------------------------------------------
WARMUP_IMAGES = 10        # Mục 6.6: warm-up >= 10 ảnh trước khi bấm giờ
LATENCY_SAMPLES = 300     # số ảnh dùng để đo p50/p95/p99

# --- Lát cắt dữ liệu (Mục 3.3) -------------------------------------------
# Quy ước clutter level của RPC theo số instance/ảnh.
# Dùng khi file COCO không còn field `level` (gold_dataset đã bị lược bỏ field này).
LEVEL_BINS = [("easy", 3, 10), ("medium", 11, 15), ("hard", 16, 20)]

# Lát cắt theo mật độ sản phẩm/ảnh
DENSITY_BINS = [("<=8", 0, 8), ("9-15", 9, 15), (">=16", 16, 10_000)]


def resolve_device(device: str | None = None) -> str:
    """'auto' -> cuda nếu có, ngược lại cpu."""
    import torch

    if device is None:
        device = os.environ.get("DETECTOR_DEVICE", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)
