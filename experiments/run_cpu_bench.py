"""KỊCH BẢN F [->TN] - Benchmark CPU + quantize INT8, CHỈ trên model đã chốt.

Phần 1 (luôn chạy): latency/RAM FP32 trên CPU của model được chọn.
Phần 2 (--onnx):    export ONNX FP32 -> quantize INT8 động (onnxruntime), đo
                    latency CPU và mAP trước/sau quantize để biết mất bao nhiêu
                    độ chính xác.

    python -m experiments.run_cpu_bench --model yolov11 --limit 200
    python -m experiments.run_cpu_bench --model yolov11 --limit 200 --onnx

Yêu cầu cho --onnx:  pip install onnx onnxruntime onnxslim
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from experiments import common as C
import config as cfg
import detectors


# --------------------------------------------------------------------------
# Tiền/hậu xử lý cho YOLO chạy bằng ONNX Runtime (không qua ultralytics)
# --------------------------------------------------------------------------
def letterbox(img, size=cfg.YOLO_IMGSZ):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def onnx_postprocess(raw, r, dx, dy, conf=cfg.CONF_RAW, iou=cfg.NMS_IOU, max_det=cfg.MAX_DET):
    """raw: (1, 4+nc, N) của YOLOv11 -> (M,5) xyxy+score trên ảnh gốc."""
    pred = np.squeeze(raw[0], 0).T                    # (N, 4+nc)
    scores = pred[:, 4:].max(axis=1)
    keep = scores >= conf
    pred, scores = pred[keep], scores[keep]
    if len(pred) == 0:
        return np.zeros((0, 5), np.float32)
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    idx = cv2.dnn.NMSBoxes(
        np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0],
                  boxes[:, 3] - boxes[:, 1]], 1).tolist(),
        scores.tolist(), conf, iou)
    idx = np.array(idx).flatten()[:max_det]
    if idx.size == 0:
        return np.zeros((0, 5), np.float32)
    boxes, scores = boxes[idx], scores[idx]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / r
    return np.hstack([boxes, scores.reshape(-1, 1)]).astype(np.float32)


class OnnxYolo:
    def __init__(self, path, threads=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def predict(self, img_bgr, conf=cfg.CONF_RAW):
        lb, r, dx, dy = letterbox(img_bgr)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self.sess.run(None, {self.iname: x})
        return onnx_postprocess(raw, r, dx, dy, conf=conf)


# --------------------------------------------------------------------------
def bench(predict_fn, ts, img_ids, label) -> tuple[dict, list[dict]]:
    for i in img_ids[:cfg.WARMUP_IMAGES]:
        predict_fn(ts.load_image(i))
    lat, per_image, dets = [], [], []
    for img_id in img_ids:
        img = ts.load_image(img_id)
        t0 = time.perf_counter()
        p = predict_fn(img)
        lat.append((time.perf_counter() - t0) * 1000)
        per_image.append({"image_id": img_id, "pred": p, "gt": ts.gt_boxes(img_id)})
        if len(p):
            for (x, y, w, h), s in zip(detectors.to_xywh(p[:, :4]), p[:, 4]):
                dets.append({"image_id": int(img_id), "category_id": 1,
                             "bbox": [float(x), float(y), float(w), float(h)],
                             "score": float(s)})
    st = C.latency_stats(lat)
    print(f"  {label:22s} p50={st['p50_ms']:7.1f}ms p95={st['p95_ms']:7.1f} "
          f"FPS={st['fps']:6.2f} RAM={C.rss_gb()}GB")
    return {"latency": st, "ram_rss_gb": C.rss_gb()}, (dets, per_image)


def main():
    ap = argparse.ArgumentParser(description="Kịch bản F [->TN] - CPU + INT8")
    ap.add_argument("--model", default="yolov11", help="model đã chốt")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--onnx", action="store_true", help="kèm export ONNX + quantize INT8")
    ap.add_argument("--threads", type=int, default=None, help="intra_op_num_threads cho ORT")
    args = ap.parse_args()

    C.set_seed()
    out_dir = cfg.RESULTS_DIR / args.tag
    ts = C.TestSet()
    img_ids = ts.subset(limit=args.limit, shuffle=True)
    results = {"model": args.model, "n_images": len(img_ids),
               "environment": C.collect_environment("cpu"), "variants": {}}

    print(f"\n=== CPU benchmark: {cfg.DISPLAY_NAMES[args.model]} trên {len(img_ids)} ảnh ===")

    # --- 1. PyTorch FP32 trên CPU ---
    det = detectors.build_detector(args.model, device="cpu")
    sysm, (dets, per_img) = bench(lambda im: det.predict(im, conf=cfg.CONF_RAW),
                                  ts, img_ids, "PyTorch FP32 (CPU)")
    acc = C.coco_evaluate(ts.coco, dets, img_ids)
    results["variants"]["pytorch_fp32_cpu"] = {
        **sysm, "accuracy": acc,
        "operational": C.counts_at_conf(per_img, cfg.CONF_THRESHOLD),
        "model_size_mb": det.info.disk_size_mb}
    print(f"    mAP50={acc['mAP50']:.4f}  mAP50-95={acc['mAP50_95']:.4f}")
    detectors.unload_all()

    # --- 2. ONNX FP32 + INT8 ---
    if args.onnx:
        if args.model != "yolov11":
            raise SystemExit("Đường ONNX hiện chỉ hiện thực cho YOLOv11 (model đã chốt).")
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from ultralytics import YOLO

        onnx_dir = out_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        fp32 = onnx_dir / "yolov11_fp32.onnx"
        int8 = onnx_dir / "yolov11_int8.onnx"

        if not fp32.exists():
            print("  đang export ONNX FP32 ...")
            p = YOLO(str(cfg.WEIGHTS["yolov11"])).export(
                format="onnx", imgsz=cfg.YOLO_IMGSZ, opset=17, simplify=True, dynamic=False)
            Path(p).replace(fp32)
        if not int8.exists():
            print("  đang quantize INT8 (dynamic) ...")
            quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

        for label, path in [("onnx_fp32_cpu", fp32), ("onnx_int8_cpu", int8)]:
            sess = OnnxYolo(path, args.threads)
            sysm, (dets, per_img) = bench(sess.predict, ts, img_ids, label)
            acc = C.coco_evaluate(ts.coco, dets, img_ids)
            results["variants"][label] = {
                **sysm, "accuracy": acc,
                "operational": C.counts_at_conf(per_img, cfg.CONF_THRESHOLD),
                "model_size_mb": round(path.stat().st_size / 1024 ** 2, 2)}
            print(f"    mAP50={acc['mAP50']:.4f}  mAP50-95={acc['mAP50_95']:.4f}")

        b = results["variants"]["onnx_fp32_cpu"]
        q = results["variants"]["onnx_int8_cpu"]
        results["quantization_impact"] = {
            "mAP50_95_before": b["accuracy"]["mAP50_95"],
            "mAP50_95_after": q["accuracy"]["mAP50_95"],
            "mAP50_95_drop_pct": round(100 * (b["accuracy"]["mAP50_95"] - q["accuracy"]["mAP50_95"])
                                       / max(b["accuracy"]["mAP50_95"], 1e-9), 2),
            "speedup_x": round(b["latency"]["mean_ms"] / max(q["latency"]["mean_ms"], 1e-9), 2),
            "size_reduction_x": round(b["model_size_mb"] / max(q["model_size_mb"], 1e-9), 2),
        }
        print("\n  Tác động quantize INT8:", results["quantization_impact"])

    C.save_json(results, out_dir / f"cpu_bench_{args.model}.json")

    print("\n" + "=" * 78 + "\nBẢNG - CPU / quantize\n" + "=" * 78)
    print(C.md_table(["Variant", "p50 (ms)", "p95", "FPS", "mAP50", "mAP50-95",
                      "Size (MB)", "RAM RSS (GB)"],
                     [[k, v["latency"]["p50_ms"], v["latency"]["p95_ms"], v["latency"]["fps"],
                       C.fmt(v["accuracy"]["mAP50"]), C.fmt(v["accuracy"]["mAP50_95"]),
                       v["model_size_mb"], v["ram_rss_gb"]]
                      for k, v in results["variants"].items()]))


if __name__ == "__main__":
    main()
