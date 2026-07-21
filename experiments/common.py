"""Hàm dùng chung cho mọi kịch bản thực nghiệm (A -> F).

Gom ở đây: seed, ghi môi trường, nạp COCO test set, gán lát cắt (level/density),
tính mAP bằng pycocotools, tính TP/FP/FN/P/R/F1, đo latency + VRAM + RAM, vẽ bbox.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402


# ==========================================================================
# Seed & môi trường (Mục 1)
# ==========================================================================
def set_seed(seed: int = cfg.SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return "n/a"


def collect_environment(device: str) -> dict:
    """Thu thập đúng các thông số Mục 1 để báo cáo tái lập được."""
    import psutil
    import torch

    cpu_name = platform.processor() or "n/a"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu_name,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {
            "torch": torch.__version__,
            "torchvision": _pkg_version("torchvision"),
            "ultralytics": _pkg_version("ultralytics"),
            "pycocotools": _pkg_version("pycocotools"),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "device_used": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version_torch": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        # Chế độ đo (Mục 1) - bắt buộc ghi vào báo cáo
        "measurement": {
            "batch_size": 1,
            "warmup_images": cfg.WARMUP_IMAGES,
            "latency_samples": cfg.LATENCY_SAMPLES,
            "seed": cfg.SEED,
            "conf_threshold_operational": cfg.CONF_THRESHOLD,
            "conf_threshold_raw_for_mAP": cfg.CONF_RAW,
            "nms_iou": cfg.NMS_IOU,
            "max_det": cfg.MAX_DET,
            "map_backend": "pycocotools (COCOeval bbox) cho cả 3 model",
            "input_resolution": {
                "yolov11": f"{cfg.YOLO_IMGSZ}x{cfg.YOLO_IMGSZ}",
                "fasterrcnn": f"min={cfg.TV_MIN_SIZE}/max={cfg.TV_MAX_SIZE}",
                "retinanet": f"min={cfg.TV_MIN_SIZE}/max={cfg.TV_MAX_SIZE}",
            },
        },
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = {
            "name": props.name,
            "vram_total_gb": round(props.total_memory / 1024 ** 3, 2),
            "capability": f"{props.major}.{props.minor}",
        }
        try:
            drv = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10,
            ).strip()
            env["gpu"]["driver"] = drv
        except Exception:
            env["gpu"]["driver"] = "n/a"
    else:
        env["gpu"] = None

    return env


# ==========================================================================
# Dataset
# ==========================================================================
class TestSet:
    """Bọc COCO ground-truth của gold_dataset/test + các lát cắt phân tích."""

    def __init__(self, ann_file: Path = cfg.TEST_ANN_FILE,
                 images_dir: Path = cfg.TEST_IMAGES_DIR,
                 level_source: Path | None = None):
        from pycocotools.coco import COCO

        self.ann_file = Path(ann_file)
        self.images_dir = Path(images_dir)
        if not self.ann_file.exists():
            raise FileNotFoundError(
                f"Thiếu file annotation: {self.ann_file}\n"
                f"-> Giải nén gold_dataset_detector_3models.zip vào {cfg.DATA_DIR}"
            )
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(str(self.ann_file))
        self.img_ids = sorted(self.coco.getImgIds())

        # số instance/ảnh
        self.n_objs: dict[int, int] = defaultdict(int)
        for ann in self.coco.dataset["annotations"]:
            self.n_objs[ann["image_id"]] += 1

        self.level_is_proxy = True
        self.levels = self._resolve_levels(level_source)
        self.densities = {i: self._bin(self.n_objs[i], cfg.DENSITY_BINS) for i in self.img_ids}

    # --- level ---
    @staticmethod
    def _bin(n: int, bins) -> str:
        for name, lo, hi in bins:
            if lo <= n <= hi:
                return name
        return "other"

    def _resolve_levels(self, level_source: Path | None) -> dict[int, str]:
        """Ưu tiên field `level` thật; nếu không có thì suy ra từ số instance/ảnh.

        gold_dataset/coco_annotations/instances_test.json đã bị lược bỏ field
        `level` của RPC gốc. Nếu bạn có file RPC gốc (instances_test2019.json),
        truyền vào qua --level-source để lấy nhãn chuẩn.
        """
        # (1) có sẵn trong chính file annotation?
        imgs = self.coco.dataset["images"]
        if all("level" in im for im in imgs):
            self.level_is_proxy = False
            return {im["id"]: im["level"] for im in imgs}

        # (2) join theo file_name từ file RPC gốc
        if level_source is not None:
            src = json.load(open(level_source))
            by_name = {im["file_name"]: im.get("level") for im in src["images"]}
            mapped = {}
            for im in imgs:
                lv = by_name.get(im["file_name"])
                if lv is None:
                    mapped = {}
                    break
                mapped[im["id"]] = lv
            if mapped:
                self.level_is_proxy = False
                return mapped
            print("[cảnh báo] --level-source không khớp file_name, quay về proxy theo mật độ.")

        # (3) proxy theo quy ước clutter của RPC
        self.level_is_proxy = True
        return {i: self._bin(self.n_objs[i], cfg.LEVEL_BINS) for i in self.img_ids}

    # --- truy cập ---
    def image_path(self, img_id: int) -> Path:
        return self.images_dir / self.coco.loadImgs(img_id)[0]["file_name"]

    def load_image(self, img_id: int) -> np.ndarray:
        p = self.image_path(img_id)
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {p}")
        return img

    def gt_boxes(self, img_id: int) -> np.ndarray:
        """(M,4) xyxy."""
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id], iscrowd=False))
        if not anns:
            return np.zeros((0, 4), dtype=np.float32)
        b = np.array([a["bbox"] for a in anns], dtype=np.float32)
        b[:, 2] += b[:, 0]
        b[:, 3] += b[:, 1]
        return b

    def subset(self, limit: int | None = None, seed: int = cfg.SEED,
               shuffle: bool = False) -> list[int]:
        ids = list(self.img_ids)
        if shuffle:
            random.Random(seed).shuffle(ids)
        return ids[:limit] if limit else ids

    def group(self, kind: str) -> dict[str, list[int]]:
        mapping = self.levels if kind == "level" else self.densities
        out: dict[str, list[int]] = defaultdict(list)
        for i in self.img_ids:
            out[mapping[i]].append(i)
        return dict(out)

    def size_stats(self) -> dict:
        """Đếm object theo phân nhóm size của COCO -> dùng để giải thích AP_medium = 0."""
        areas = np.array([a["area"] for a in self.coco.dataset["annotations"]])
        return {
            "total": int(len(areas)),
            "small (area < 32^2)": int((areas < 32 ** 2).sum()),
            "medium (32^2 <= area < 96^2)": int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum()),
            "large (area >= 96^2)": int((areas >= 96 ** 2).sum()),
        }


# ==========================================================================
# Metrics
# ==========================================================================
COCO_STAT_KEYS = [
    "mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]


def coco_evaluate(coco_gt, detections: list[dict], img_ids: list[int],
                  return_curve: bool = False) -> dict:
    """Chạy COCOeval (bbox) - dùng CHUNG cho cả 3 model để đồng nhất cách tính mAP."""
    from pycocotools.cocoeval import COCOeval

    if not detections:
        res = {k: 0.0 for k in COCO_STAT_KEYS}
        if return_curve:
            res["pr_curve"] = {"recall": [], "precision": []}
        return res

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes([d for d in detections if d["image_id"] in set(img_ids)])
        E = COCOeval(coco_gt, coco_dt, iouType="bbox")
        E.params.imgIds = list(img_ids)
        E.params.maxDets = [1, 10, cfg.MAX_DET]
        E.evaluate()
        E.accumulate()
        E.summarize()

    out = {k: float(v) for k, v in zip(COCO_STAT_KEYS, E.stats)}
    if return_curve:
        # precision shape [T(iou), R(recall), K(cat), A(area), M(maxDet)]
        prec = E.eval["precision"][0, :, 0, 0, 2]  # IoU=0.50, cat 0, area=all, maxDet=100
        out["pr_curve"] = {
            "recall": E.params.recThrs.tolist(),
            "precision": [float(p) for p in prec],
        }
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa (N,4) và (M,4), toạ độ xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def match_greedy(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray,
                 iou_thr: float = cfg.MATCH_IOU):
    """Ghép greedy theo score giảm dần -> (tp_mask, fp_mask, matched_gt_idx).

    Bài class-agnostic 1 lớp nên không có confusion matrix đa lớp; TP/FP/FN theo
    IoU chính là "confusion" của bài toán này (Mục 3.1).
    """
    n_p, n_g = len(pred_xyxy), len(gt_xyxy)
    tp = np.zeros(n_p, dtype=bool)
    gt_matched = -np.ones(n_p, dtype=int)
    used = np.zeros(n_g, dtype=bool)
    if n_p and n_g:
        ious = iou_matrix(pred_xyxy, gt_xyxy)
        for i in range(n_p):  # pred đã sắp xếp score giảm dần
            cand = np.where(~used)[0]
            if len(cand) == 0:
                break
            j = cand[np.argmax(ious[i, cand])]
            if ious[i, j] >= iou_thr:
                tp[i] = True
                used[j] = True
                gt_matched[i] = j
    return tp, ~tp, gt_matched, used


def prf_from_counts(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def counts_at_conf(per_image: list[dict], conf: float, iou_thr: float = cfg.MATCH_IOU) -> dict:
    """Tính TP/FP/FN/P/R/F1 toàn tập tại một ngưỡng conf."""
    tp = fp = fn = 0
    for rec in per_image:
        pred = rec["pred"]
        pred = pred[pred[:, 4] >= conf] if len(pred) else pred
        t, f, _, used = match_greedy(pred[:, :4] if len(pred) else pred.reshape(0, 5)[:, :4],
                                     rec["gt"], iou_thr)
        tp += int(t.sum())
        fp += int(f.sum())
        fn += int((~used).sum())
    return prf_from_counts(tp, fp, fn)


def f1_confidence_curve(per_image: list[dict], thresholds=None) -> dict:
    """Quét ngưỡng conf -> tìm ngưỡng F1 tối ưu (Mục 3.1, khuyến nghị)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    conf_list, f1_list, p_list, r_list = [], [], [], []
    for c in thresholds:
        m = counts_at_conf(per_image, float(c))
        conf_list.append(float(c))
        f1_list.append(m["f1"])
        p_list.append(m["precision"])
        r_list.append(m["recall"])
    best = int(np.argmax(f1_list))
    return {"conf": conf_list, "f1": f1_list, "precision": p_list, "recall": r_list,
            "best_conf": conf_list[best], "best_f1": f1_list[best]}


def recall_by_bbox_size(per_image: list[dict], conf: float = cfg.CONF_THRESHOLD) -> dict:
    """Recall theo nhóm kích thước bbox GT (Mục 3.3)."""
    bins = [("small (<32^2)", 0, 32 ** 2), ("medium (32^2-96^2)", 32 ** 2, 96 ** 2),
            ("large (>=96^2)", 96 ** 2, float("inf"))]
    hit = defaultdict(int)
    tot = defaultdict(int)
    for rec in per_image:
        pred = rec["pred"]
        pred = pred[pred[:, 4] >= conf] if len(pred) else pred
        _, _, _, used = match_greedy(pred[:, :4] if len(pred) else np.zeros((0, 4)), rec["gt"])
        gt = rec["gt"]
        areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) if len(gt) else np.array([])
        for k, a in enumerate(areas):
            for name, lo, hi in bins:
                if lo <= a < hi:
                    tot[name] += 1
                    hit[name] += int(used[k])
                    break
    return {name: {"n_gt": tot[name], "recall": round(hit[name] / tot[name], 4) if tot[name] else None}
            for name, _, _ in bins}


# ==========================================================================
# Đo tài nguyên (Mục 3.2)
# ==========================================================================
def latency_stats(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms, dtype=np.float64)
    if a.size == 0:
        return {}
    mean = float(a.mean())
    return {
        "n_samples": int(a.size),
        "mean_ms": round(mean, 2),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
        "min_ms": round(float(a.min()), 2),
        "max_ms": round(float(a.max()), 2),
        "fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
    }


def rss_gb() -> float:
    import psutil

    return round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, 3)


def reset_vram_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def vram_peak_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)


# ==========================================================================
# Vẽ
# ==========================================================================
def draw_boxes(img: np.ndarray, boxes: np.ndarray, color=(0, 255, 0),
               labels: list[str] | None = None, thickness: int = 3,
               font_scale: float = 1.0) -> np.ndarray:
    out = img.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels) and labels[i]:
            txt = labels[i]
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, txt, (x1 + 2, max(th, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def label_panel(img: np.ndarray, title: str, bar_h: int = 70) -> np.ndarray:
    """Thêm thanh tiêu đề phía trên ảnh (ghép panel so sánh)."""
    h, w = img.shape[:2]
    bar = np.full((bar_h, w, 3), 30, dtype=np.uint8)
    scale = max(0.8, w / 1200.0)
    cv2.putText(bar, title, (12, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


# ==========================================================================
# I/O
# ==========================================================================
def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)
    print(f"  -> đã ghi {path}")


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def md_table(headers: list[str], rows: list[list]) -> str:
    line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)
