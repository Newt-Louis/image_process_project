"""KỊCH BẢN A - Định lượng toàn tập (bắt buộc).

Chạy cả 3 model trên cùng tập test gold_dataset/test, cùng NMS IoU, cùng cách
tính mAP (pycocotools) -> xuất đầy đủ Bảng 1 (accuracy) + Bảng 2 (system).

    python -m experiments.run_eval                      # cả 3 model, toàn bộ 19.200 ảnh
    python -m experiments.run_eval --limit 500          # chạy thử nhanh
    python -m experiments.run_eval --models yolov11
    python -m experiments.run_eval --device cpu --limit 100 --tag cpu   # [->TN] Mục 3.2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments import common as C
import config as cfg
import detectors


def _progress(iterable, total, desc):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, ncols=90)
    except ImportError:
        return iterable


def evaluate_model(name: str, ts: C.TestSet, img_ids: list[int], device: str,
                   nms_iou: float, tag: str) -> dict:
    print(f"\n{'=' * 78}\n[{cfg.DISPLAY_NAMES[name]}] device={device} | {len(img_ids)} ảnh\n{'=' * 78}")

    C.reset_vram_peak()
    rss_before = C.rss_gb()

    det = detectors.build_detector(name, device=device, nms_iou=nms_iou,
                                   score_thresh=cfg.CONF_RAW)
    rss_after_load = C.rss_gb()
    vram_after_load = C.vram_peak_gb()
    print(f"  load {det.info.load_time_s}s | {det.info.num_params / 1e6:.2f}M params "
          f"| {det.info.disk_size_mb:.1f} MB đĩa")

    # --- Warm-up (Mục 6.6): >= 10 ảnh, KHÔNG tính vào latency -------------
    warm = [ts.load_image(i) for i in img_ids[:cfg.WARMUP_IMAGES]]
    det.warmup(warm)
    del warm

    # --- Pass 1: sinh prediction toàn tập ở conf thô (cho pycocotools) ----
    C.reset_vram_peak()
    detections: list[dict] = []
    per_image: list[dict] = []
    eval_lat: list[float] = []
    t_wall = time.perf_counter()

    for img_id in _progress(img_ids, len(img_ids), f"  {name} infer"):
        img = ts.load_image(img_id)
        preds, ms = det.predict_timed(img)
        eval_lat.append(ms)
        per_image.append({"image_id": img_id, "pred": preds, "gt": ts.gt_boxes(img_id)})
        if len(preds):
            xywh = detectors.to_xywh(preds[:, :4])
            for (x, y, w, h), s in zip(xywh, preds[:, 4]):
                detections.append({
                    "image_id": int(img_id), "category_id": 1,
                    "bbox": [round(float(x), 2), round(float(y), 2),
                             round(float(w), 2), round(float(h), 2)],
                    "score": round(float(s), 5),
                })
    wall_s = time.perf_counter() - t_wall
    vram_eval = C.vram_peak_gb()
    rss_running = C.rss_gb()
    print(f"  xong pass inference trong {wall_s / 60:.1f} phút, "
          f"{len(detections)} detection thô")

    # --- Pass 2: benchmark latency riêng ở conf vận hành ------------------
    bench_ids = img_ids[:min(cfg.LATENCY_SAMPLES, len(img_ids))]
    C.reset_vram_peak()
    bench_lat = []
    for img_id in _progress(bench_ids, len(bench_ids), f"  {name} latency"):
        img = ts.load_image(img_id)
        _, ms = det.predict_timed(img, conf=cfg.CONF_THRESHOLD)
        bench_lat.append(ms)
    vram_bench = C.vram_peak_gb()

    # --- Accuracy (Mục 3.1) ----------------------------------------------
    print("  đang tính mAP bằng pycocotools ...")
    acc = C.coco_evaluate(ts.coco, detections, img_ids, return_curve=True)
    pr_curve = acc.pop("pr_curve")
    op = C.counts_at_conf(per_image, cfg.CONF_THRESHOLD)
    f1_curve = C.f1_confidence_curve(per_image)
    rec_by_size = C.recall_by_bbox_size(per_image)

    result = {
        "model": name,
        "display_name": cfg.DISPLAY_NAMES[name],
        "tag": tag,
        "device": device,
        "n_images": len(img_ids),
        "n_gt_boxes": int(sum(len(r["gt"]) for r in per_image)),
        "n_raw_detections": len(detections),
        "settings": {
            "conf_raw_for_mAP": cfg.CONF_RAW,
            "conf_operational": cfg.CONF_THRESHOLD,
            "nms_iou": nms_iou,
            "max_det": cfg.MAX_DET,
            "match_iou_for_TP_FP_FN": cfg.MATCH_IOU,
            "input_resolution": det.info.input_resolution,
        },
        # Bảng 1
        "accuracy": acc,
        "operational": op,               # TP/FP/FN/P/R/F1 @ conf=0.25, IoU=0.5
        "recall_by_bbox_size": rec_by_size,
        "pr_curve": pr_curve,            # IoU=0.50, 101 điểm recall
        "f1_confidence_curve": f1_curve,
        # Bảng 2
        "system": {
            "latency": C.latency_stats(bench_lat),
            "latency_during_full_eval": C.latency_stats(eval_lat),
            "model_size_mb": det.info.disk_size_mb,
            "num_params": det.info.num_params,
            "num_params_m": round(det.info.num_params / 1e6, 3),
            "load_time_s": det.info.load_time_s,
            "vram_peak_after_load_gb": vram_after_load,
            "vram_peak_infer_gb": vram_bench,
            "vram_peak_full_eval_gb": vram_eval,
            "ram_rss_before_load_gb": rss_before,
            "ram_rss_after_load_gb": rss_after_load,
            "ram_rss_running_gb": rss_running,
            "wall_time_full_pass_s": round(wall_s, 1),
        },
        "checkpoint_info": det.info.extra,
    }

    out = cfg.RESULTS_DIR / tag
    C.save_json(detections, out / f"predictions_{name}.json")
    C.save_json(result, out / f"eval_{name}.json")

    del det
    detectors.unload_all()
    return result


def main():
    ap = argparse.ArgumentParser(description="Kịch bản A - eval định lượng 3 model")
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="all | yolov11 fasterrcnn retinanet")
    ap.add_argument("--limit", type=int, default=None, help="giới hạn số ảnh (chạy thử)")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="với --limit: lấy N ảnh ĐẦU thay vì lấy mẫu ngẫu nhiên. Mặc định lấy "
                         "ngẫu nhiên (seed cố định) vì test set xếp theo độ khó — N ảnh đầu "
                         "sẽ toàn ảnh easy.")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--nms-iou", type=float, default=cfg.NMS_IOU)
    ap.add_argument("--tag", default="main", help="thư mục con trong results/")
    ap.add_argument("--level-source", default=None,
                    help="instances_test2019.json gốc của RPC (để lấy field `level` thật)")
    args = ap.parse_args()

    C.set_seed()
    cfg.ensure_dirs()
    device = cfg.resolve_device(args.device)
    models = cfg.MODEL_ORDER if args.models == ["all"] else args.models

    ts = C.TestSet(level_source=Path(args.level_source) if args.level_source else None)
    img_ids = ts.subset(limit=args.limit,
                        shuffle=bool(args.limit) and not args.no_shuffle)

    out = cfg.RESULTS_DIR / args.tag
    env = C.collect_environment(device)
    env["dataset"] = {
        "ann_file": str(ts.ann_file),
        "images_dir": str(ts.images_dir),
        "n_images_total": len(ts.img_ids),
        "n_images_evaluated": len(img_ids),
        "n_gt_boxes_total": len(ts.coco.dataset["annotations"]),
        # Dùng để giải thích AP_medium = 0 (Mục 2.6)
        "object_size_distribution_coco": ts.size_stats(),
        "level_is_proxy": ts.level_is_proxy,
    }
    C.save_json(env, out / "environment.json")
    C.save_json(img_ids, out / "eval_ids.json")  # để các kịch bản sau dùng đúng tập ảnh
    print("\nMôi trường:", env["cpu"], "|", env["gpu"], "| device =", device)
    print("Phân bố kích thước object (chuẩn COCO):", env["dataset"]["object_size_distribution_coco"])

    summary = {}
    for name in models:
        summary[name] = evaluate_model(name, ts, img_ids, device, args.nms_iou, args.tag)

    C.save_json({k: {"accuracy": v["accuracy"], "operational": v["operational"],
                     "system": v["system"]} for k, v in summary.items()},
                out / "summary_all_models.json")

    # In nhanh bảng ra console
    print("\n" + "=" * 78 + "\nBẢNG 1 - ACCURACY\n" + "=" * 78)
    print(C.md_table(
        ["Model", "mAP50", "mAP50-95", "mAP75", "P", "R", "F1", "AP_L", "AR100", "TP", "FP", "FN"],
        [[cfg.DISPLAY_NAMES[m], C.fmt(r["accuracy"]["mAP50"]), C.fmt(r["accuracy"]["mAP50_95"]),
          C.fmt(r["accuracy"]["mAP75"]), C.fmt(r["operational"]["precision"]),
          C.fmt(r["operational"]["recall"]), C.fmt(r["operational"]["f1"]),
          C.fmt(r["accuracy"]["AP_large"]), C.fmt(r["accuracy"]["AR100"]),
          r["operational"]["TP"], r["operational"]["FP"], r["operational"]["FN"]]
         for m, r in summary.items()]))

    print("\n" + "=" * 78 + "\nBẢNG 2 - SYSTEM\n" + "=" * 78)
    print(C.md_table(
        ["Model", "p50 (ms)", "p95", "p99", "FPS", "Size (MB)", "Params (M)",
         "VRAM infer (GB)", "RAM RSS (GB)"],
        [[cfg.DISPLAY_NAMES[m], r["system"]["latency"]["p50_ms"], r["system"]["latency"]["p95_ms"],
          r["system"]["latency"]["p99_ms"], r["system"]["latency"]["fps"],
          r["system"]["model_size_mb"], r["system"]["num_params_m"],
          r["system"]["vram_peak_infer_gb"], r["system"]["ram_rss_running_gb"]]
         for m, r in summary.items()]))

    print(f"\nXong. Kết quả ở: {out}")
    print("Bước tiếp theo: python -m experiments.run_slices --tag", args.tag)


if __name__ == "__main__":
    main()
