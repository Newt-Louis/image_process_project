"""Nạp 3 model detector và chuẩn hóa output về CÙNG MỘT format.

Mục 6 của bản hướng dẫn: 3 hàm nạp model + chuẩn hóa output trước khi đưa vào
cùng hàm tính mAP -> đảm bảo so sánh công bằng.

Format chuẩn hóa: np.ndarray shape (N, 5) = [x1, y1, x2, y2, score], toạ độ pixel
trên ảnh GỐC, đã sắp xếp score giảm dần. Hàm `to_xywh()` đổi sang format COCO.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config as cfg


# --------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------
def to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> [x,y,w,h] (format COCO)."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy.reshape(0, 4)
    out = boxes_xyxy.astype(np.float64).copy()
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def _disk_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


# --------------------------------------------------------------------------
# Lớp cơ sở
# --------------------------------------------------------------------------
@dataclass
class DetectorInfo:
    """Metadata phục vụ Bảng 2 - System (Mục 8)."""

    name: str
    display_name: str
    weights_path: str
    num_params: int = 0
    disk_size_mb: float = 0.0
    load_time_s: float = 0.0
    input_resolution: str = ""
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["num_params_m"] = round(self.num_params / 1e6, 3)
        return d


class BaseDetector:
    """Giao diện chung của cả 3 model."""

    name = "base"

    def __init__(self, weights_path: Path, device: str = "cpu",
                 nms_iou: float = cfg.NMS_IOU, max_det: int = cfg.MAX_DET,
                 score_thresh: float = cfg.CONF_RAW):
        self.weights_path = Path(weights_path)
        self.device = device
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.score_thresh = score_thresh
        self.info = DetectorInfo(
            name=self.name,
            display_name=cfg.DISPLAY_NAMES.get(self.name, self.name),
            weights_path=str(self.weights_path),
            disk_size_mb=round(_disk_size_mb(self.weights_path), 2),
            device=device,
        )
        t0 = time.perf_counter()
        self._load()
        self.info.load_time_s = round(time.perf_counter() - t0, 3)

    # -- phải override --
    def _load(self) -> None:
        raise NotImplementedError

    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- API dùng chung --
    def predict(self, img_bgr: np.ndarray, conf: float | None = None) -> np.ndarray:
        """Trả về (N,5) [x1,y1,x2,y2,score] trên toạ độ ảnh gốc, score giảm dần."""
        dets = self._infer(img_bgr)
        if conf is not None and len(dets):
            dets = dets[dets[:, 4] >= conf]
        if len(dets):
            dets = dets[np.argsort(-dets[:, 4])]
        return dets.astype(np.float32)

    def predict_timed(self, img_bgr: np.ndarray, conf: float | None = None):
        """Như predict() nhưng trả kèm latency end-to-end (ms), đã sync GPU."""
        self.synchronize()
        t0 = time.perf_counter()
        dets = self.predict(img_bgr, conf=conf)
        self.synchronize()
        return dets, (time.perf_counter() - t0) * 1000.0

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()

    def warmup(self, images: list[np.ndarray]) -> None:
        for img in images:
            self.predict(img, conf=cfg.CONF_THRESHOLD)
        self.synchronize()


# --------------------------------------------------------------------------
# Model 1 - YOLOv11 (Ultralytics)
# --------------------------------------------------------------------------
class YoloV11Detector(BaseDetector):
    name = "yolov11"

    def _load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights_path))
        self.model.to(self.device)
        self.model.model.eval()
        self.info.num_params = sum(p.numel() for p in self.model.model.parameters())
        self.info.input_resolution = f"{cfg.YOLO_IMGSZ}x{cfg.YOLO_IMGSZ} (letterbox)"
        self.info.extra["framework"] = "ultralytics"

    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        res = self.model.predict(
            source=img_bgr,
            imgsz=cfg.YOLO_IMGSZ,
            conf=self.score_thresh,
            iou=self.nms_iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, scores])


# --------------------------------------------------------------------------
# Model 2 & 3 - torchvision (Faster R-CNN, RetinaNet)
# --------------------------------------------------------------------------
class TorchvisionDetector(BaseDetector):
    """Base cho 2 model torchvision.

    Checkpoint được train với backbone dùng FrozenBatchNorm2d (chuẩn của
    torchvision detection), num_classes=2 (background + product).
    """

    def _build_model(self):
        raise NotImplementedError

    def _load(self) -> None:
        import torch

        model = self._build_model()
        ckpt = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        # `num_batches_tracked` không tồn tại trong FrozenBatchNorm2d -> bỏ qua an toàn
        hard_missing = [k for k in missing if "num_batches_tracked" not in k]
        if hard_missing or unexpected:
            raise RuntimeError(
                f"[{self.name}] state_dict không khớp kiến trúc. "
                f"missing={hard_missing[:5]} unexpected={list(unexpected)[:5]}"
            )
        model.eval().to(self.device)
        self.model = model
        self.info.num_params = sum(p.numel() for p in model.parameters())
        self.info.input_resolution = (
            f"min_size={cfg.TV_MIN_SIZE}, max_size={cfg.TV_MAX_SIZE} (giữ tỉ lệ)"
        )
        self.info.extra["framework"] = "torchvision"
        self.info.extra["trained_epoch"] = (
            ckpt.get("epoch") if isinstance(ckpt, dict) else None
        )
        self.info.extra["ckpt_val_mAP50"] = (
            ckpt.get("val_mAP50") if isinstance(ckpt, dict) else None
        )

    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        import torch

        rgb = img_bgr[:, :, ::-1].copy()  # BGR -> RGB
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            out = self.model([tensor])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        keep = labels == 1  # class-agnostic: chỉ có lớp 'product' = 1
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.hstack([boxes, scores.reshape(-1, 1)])


class FasterRCNNDetector(TorchvisionDetector):
    name = "fasterrcnn"

    def _build_model(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.ops import misc as misc_nn_ops

        return fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            min_size=cfg.TV_MIN_SIZE,
            max_size=cfg.TV_MAX_SIZE,
            box_score_thresh=self.score_thresh,
            box_nms_thresh=self.nms_iou,
            box_detections_per_img=self.max_det,
        )


class RetinaNetDetector(TorchvisionDetector):
    name = "retinanet"

    def _build_model(self):
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.ops import misc as misc_nn_ops

        return retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            min_size=cfg.TV_MIN_SIZE,
            max_size=cfg.TV_MAX_SIZE,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_iou,
            detections_per_img=self.max_det,
        )


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
DETECTOR_CLASSES = {
    "yolov11": YoloV11Detector,
    "fasterrcnn": FasterRCNNDetector,
    "retinanet": RetinaNetDetector,
}

_CACHE: dict[tuple, BaseDetector] = {}


def build_detector(name: str, device: str | None = None, **kw) -> BaseDetector:
    if name not in DETECTOR_CLASSES:
        raise ValueError(f"Model không hợp lệ: {name}. Chọn 1 trong {list(DETECTOR_CLASSES)}")
    device = cfg.resolve_device(device)
    weights = cfg.WEIGHTS[name]
    if not Path(weights).exists():
        raise FileNotFoundError(f"Không tìm thấy weights của {name}: {weights}")
    return DETECTOR_CLASSES[name](weights, device=device, **kw)


def get_detector(name: str, device: str | None = None, warm: bool = True, **kw) -> BaseDetector:
    """Như build_detector nhưng có cache - dùng cho web demo (tránh load lại).

    Warm-up ngay sau khi load: lần inference đầu tiên phải khởi tạo CUDA context và
    cuDNN autotune nên chậm gấp ~100 lần. Không warm-up thì latency hiển thị trên
    UI ở request đầu sẽ sai hoàn toàn.
    """
    device = cfg.resolve_device(device)
    key = (name, device, kw.get("nms_iou", cfg.NMS_IOU), kw.get("score_thresh", cfg.CONF_RAW))
    if key not in _CACHE:
        det = build_detector(name, device=device, **kw)
        if warm:
            rng = np.random.default_rng(cfg.SEED)
            dummy = rng.integers(0, 255, (1200, 1200, 3), dtype=np.uint8)
            det.warmup([dummy] * 3)
        _CACHE[key] = det
    return _CACHE[key]


def loaded_detectors() -> list[str]:
    return sorted({k[0] for k in _CACHE})


def unload_all() -> None:
    import torch

    _CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
