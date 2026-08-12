"""Web demo: nạp 3 model detector thật (YOLOv11 / Faster R-CNN / RetinaNet).

Chạy:
    cd demo_app
    uvicorn main:app --host 0.0.0.0 --port 8000

Ảnh đầu vào: ảnh chụp thật từ camera / ảnh tải từ internet (upload qua UI), hoặc
chọn 1 ảnh trong tập test gold_dataset để đối chiếu trực tiếp với ground-truth.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config as cfg
import detectors
from experiments import common as C

app = FastAPI(title="RPC Checkout — Demo 3 Model Detector")

cfg.ensure_dirs()
app.mount("/static", StaticFiles(directory=str(cfg.APP_DIR / "static")), name="static")
if cfg.RESULTS_DIR.exists():
    app.mount("/results", StaticFiles(directory=str(cfg.RESULTS_DIR)), name="results")

_TESTSET: C.TestSet | None = None


def get_testset() -> C.TestSet | None:
    """Nạp COCO test set 1 lần (lazy) — chỉ dùng cho chế độ đối chiếu ground-truth."""
    global _TESTSET
    if _TESTSET is None and cfg.TEST_ANN_FILE.exists():
        _TESTSET = C.TestSet()
    return _TESTSET


# --------------------------------------------------------------------------
def _b64(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _read_upload(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không đọc được ảnh — hãy dùng JPG/PNG/WebP.")
    return img


def _crops(img: np.ndarray, boxes: np.ndarray, limit: int = 30) -> list[dict]:
    out = []
    h, w = img.shape[:2]
    for i, b in enumerate(boxes[:limit]):
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        out.append({"index": i + 1, "confidence": round(float(b[4]), 4),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "image": _b64(img[y1:y2, x1:x2], 80)})
    return out


def _run_models(img: np.ndarray, model_names: list[str], conf: float, nms_iou: float,
                gt: np.ndarray | None = None) -> list[dict]:
    results = []
    for name in model_names:
        det = detectors.get_detector(name, nms_iou=nms_iou)
        boxes, latency = det.predict_timed(img, conf=conf)

        entry = {
            "model": name,
            "display_name": cfg.DISPLAY_NAMES[name],
            "device": det.device,
            "num_params_m": round(det.info.num_params / 1e6, 2),
            "model_size_mb": det.info.disk_size_mb,
            "input_resolution": det.info.input_resolution,
            "latency_ms": round(latency, 2),
            "fps": round(1000.0 / latency, 1) if latency > 0 else None,
            "num_detections": int(len(boxes)),
            "avg_confidence": round(float(boxes[:, 4].mean()), 4) if len(boxes) else None,
            "boxes": [[round(float(v), 1) for v in b[:4]] + [round(float(b[4]), 4)]
                      for b in boxes],
        }

        color = cfg.MODEL_COLORS_BGR[name]
        thick = max(2, int(round(img.shape[1] / 500)))
        fs = max(0.5, img.shape[1] / 1400)

        if gt is not None:
            tp_mask, fp_mask, _, used = C.match_greedy(
                boxes[:, :4] if len(boxes) else np.zeros((0, 4)), gt)
            entry.update(C.prf_from_counts(int(tp_mask.sum()), int(fp_mask.sum()),
                                           int((~used).sum())))
            vis = C.draw_boxes(img, gt[~used] if len(gt) else gt, (40, 200, 235),
                               thickness=thick + 2)
            vis = C.draw_boxes(vis, boxes[tp_mask, :4], (60, 200, 60),
                               [f"{s:.2f}" for s in boxes[tp_mask, 4]], thick, fs)
            vis = C.draw_boxes(vis, boxes[fp_mask, :4], (60, 60, 235),
                               [f"FP {s:.2f}" for s in boxes[fp_mask, 4]], thick, fs)
        else:
            vis = C.draw_boxes(img, boxes[:, :4] if len(boxes) else np.zeros((0, 4)), color,
                               [f"product {s:.2f}" for s in boxes[:, 4]], thick, fs)

        entry["annotated_image"] = _b64(vis)
        entry["crops"] = _crops(img, boxes)
        results.append(entry)
    return results


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return (cfg.APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
async def api_models():
    device = cfg.resolve_device()
    out = []
    for name in cfg.MODEL_ORDER:
        p = cfg.WEIGHTS[name]
        out.append({
            "name": name,
            "display_name": cfg.DISPLAY_NAMES[name],
            "weights": str(p),
            "available": p.exists(),
            "size_mb": round(p.stat().st_size / 1024 ** 2, 1) if p.exists() else None,
            "loaded": name in detectors.loaded_detectors(),
            "color": "#%02x%02x%02x" % cfg.MODEL_COLORS_BGR[name][::-1],
        })
    ts = get_testset()
    return {
        "device": device,
        "models": out,
        "defaults": {"conf": cfg.CONF_THRESHOLD, "nms_iou": cfg.NMS_IOU},
        "gold_dataset_available": ts is not None,
        "gold_dataset_size": len(ts.img_ids) if ts else 0,
    }


@app.post("/api/detect")
async def api_detect(
    file: UploadFile = File(...),
    models: str = Form("yolov11"),
    conf: float = Form(cfg.CONF_THRESHOLD),
    nms_iou: float = Form(cfg.NMS_IOU),
):
    """Detect trên ảnh upload (ảnh chụp thật / ảnh lấy từ internet)."""
    t0 = time.perf_counter()
    img = _read_upload(await file.read())
    names = [m.strip() for m in models.split(",") if m.strip() in detectors.DETECTOR_CLASSES]
    if not names:
        raise HTTPException(400, f"models phải nằm trong {list(detectors.DETECTOR_CLASSES)}")

    results = _run_models(img, names, conf, nms_iou)
    return JSONResponse({
        "source": "upload",
        "file_name": file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "settings": {"conf": conf, "nms_iou": nms_iou, "device": cfg.resolve_device()},
        "results": results,
        "total_time_ms": round((time.perf_counter() - t0) * 1000, 2),
    })


@app.get("/api/gold/list")
async def api_gold_list(limit: int = 24, offset: int = 0, level: str | None = None):
    """Danh sách ảnh trong tập test để chọn nhanh (có kèm ground-truth)."""
    ts = get_testset()
    if ts is None:
        raise HTTPException(404, "Chưa có gold_dataset — giải nén vào demo_app/data/gold_dataset")
    ids = ts.img_ids if not level else [i for i in ts.img_ids if ts.levels[i] == level]
    page = ids[offset:offset + limit]
    return {
        "total": len(ids),
        "level_is_proxy": ts.level_is_proxy,
        "items": [{"image_id": i, "file_name": ts.coco.loadImgs(i)[0]["file_name"],
                   "n_gt": ts.n_objs[i], "level": ts.levels[i],
                   "density": ts.densities[i],
                   "thumb": f"/api/gold/image/{i}?w=220"} for i in page],
    }


@app.get("/api/gold/image/{image_id}")
async def api_gold_image(image_id: int, w: int | None = None):
    ts = get_testset()
    if ts is None:
        raise HTTPException(404, "Chưa có gold_dataset")
    path = ts.image_path(image_id)
    if not path.exists():
        raise HTTPException(404, f"Không có file {path.name}")
    if w is None:
        return FileResponse(path)
    img = cv2.imread(str(path))
    img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.post("/api/detect_gold")
async def api_detect_gold(
    image_id: int = Form(...),
    models: str = Form("yolov11,fasterrcnn,retinanet"),
    conf: float = Form(cfg.CONF_THRESHOLD),
    nms_iou: float = Form(cfg.NMS_IOU),
):
    """Detect trên 1 ảnh của test set và đối chiếu trực tiếp với ground-truth."""
    ts = get_testset()
    if ts is None:
        raise HTTPException(404, "Chưa có gold_dataset")
    if image_id not in set(ts.img_ids):
        raise HTTPException(404, f"image_id {image_id} không có trong test set")

    t0 = time.perf_counter()
    img = ts.load_image(image_id)
    gt = ts.gt_boxes(image_id)
    names = [m.strip() for m in models.split(",") if m.strip() in detectors.DETECTOR_CLASSES]
    results = _run_models(img, names, conf, nms_iou, gt=gt)

    return JSONResponse({
        "source": "gold_dataset",
        "image_id": image_id,
        "file_name": ts.coco.loadImgs(image_id)[0]["file_name"],
        "level": ts.levels[image_id],
        "level_is_proxy": ts.level_is_proxy,
        "n_gt": int(len(gt)),
        "gt_image": _b64(C.draw_boxes(img, gt, (200, 200, 200),
                                      thickness=max(2, img.shape[1] // 400))),
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "settings": {"conf": conf, "nms_iou": nms_iou, "device": cfg.resolve_device()},
        "results": results,
        "total_time_ms": round((time.perf_counter() - t0) * 1000, 2),
    })


@app.get("/api/report")
async def api_report(tag: str = "main"):
    """Trả bảng số liệu đã chạy (nếu có) để hiển thị tab 'Kết quả thực nghiệm'."""
    d = cfg.RESULTS_DIR / tag
    if not (d / "environment.json").exists():
        return {"available": False,
                "hint": "Chạy `python -m experiments.run_eval` rồi `python -m experiments.make_report`."}
    out = {"available": True, "tag": tag, "environment": C.load_json(d / "environment.json"),
           "models": {}, "charts": [], "qualitative": [], "slices": None}
    for m in cfg.MODEL_ORDER:
        f = d / f"eval_{m}.json"
        if f.exists():
            e = C.load_json(f)
            out["models"][m] = {"accuracy": e["accuracy"], "operational": e["operational"],
                                "system": e["system"], "display_name": e["display_name"]}
    if (d / "slices.json").exists():
        out["slices"] = C.load_json(d / "slices.json")
    charts = d / "charts"
    if charts.exists():
        out["charts"] = [f"/results/{tag}/charts/{p.name}" for p in sorted(charts.glob("*.png"))]
    if (d / "qualitative.json").exists():
        q = C.load_json(d / "qualitative.json")
        out["qualitative"] = [{**it, "url": f"/results/{tag}/qualitative/"
                                            f"{Path(it['output']).name}"} for it in q["items"]]
    if (d / "BAO_CAO_THUC_NGHIEM.md").exists():
        out["report_md"] = f"/results/{tag}/BAO_CAO_THUC_NGHIEM.md"
    return out


@app.post("/api/unload")
async def api_unload():
    detectors.unload_all()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
