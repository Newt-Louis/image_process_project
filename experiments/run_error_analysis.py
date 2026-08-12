"""KỊCH BẢN D - Phân tích lỗi (Mục 4D).

Phân loại 4 nhóm lỗi điển hình cho từng model, đếm số lượng toàn tập và xuất
2-3 ảnh minh họa mỗi nhóm:

  1. missed      - Missed detection: GT không có box nào ghép được (bị che khuất/chồng lấp)
  2. false_pos   - False positive: box thừa, IoU < 0.3 với mọi GT (báo nhầm nền/bóng)
  3. double      - Double detection: 1 vật bị 2 box (box thừa nhưng IoU>=0.5 với GT đã ghép) -> NMS chưa gộp
  4. loc_error   - Box lệch: IoU với GT tốt nhất nằm trong [0.3, 0.5) -> nguyên nhân mAP75 tụt

    python -m experiments.run_error_analysis --tag main
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import cv2
import numpy as np

from experiments import common as C
from experiments.run_qualitative import load_preds
import config as cfg

ERROR_TYPES = {
    "missed": "Missed detection — sản phẩm bị bỏ sót (che khuất/chồng lấp)",
    "false_pos": "False positive — báo nhầm nền/bóng đổ thành sản phẩm",
    "double": "Double detection — 1 vật 2 box, NMS chưa gộp",
    "loc_error": "Box lệch — IoU 0.3-0.5, kéo tụt mAP@0.75",
}

COLORS = {"missed": (40, 200, 235), "false_pos": (60, 60, 235),
          "double": (235, 120, 60), "loc_error": (200, 60, 200)}
LOC_LOW, LOC_HIGH = 0.3, cfg.MATCH_IOU


def classify_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Trả về index của từng loại lỗi trong ảnh."""
    tp_mask, fp_mask, matched_gt, used = C.match_greedy(
        pred[:, :4] if len(pred) else np.zeros((0, 4)), gt)

    out = {k: [] for k in ERROR_TYPES}
    out["missed"] = np.where(~used)[0].tolist() if len(gt) else []

    if len(pred) and len(gt):
        ious = C.iou_matrix(pred[:, :4], gt)
        best = ious.max(axis=1)
        for i in np.where(fp_mask)[0]:
            if best[i] >= cfg.MATCH_IOU:
                out["double"].append(int(i))       # trùng lên GT đã có box khác ghép
            elif best[i] >= LOC_LOW:
                out["loc_error"].append(int(i))    # box lệch
            else:
                out["false_pos"].append(int(i))    # nền/bóng
    elif len(pred):
        out["false_pos"] = list(range(len(pred)))

    return {"idx": out, "tp_mask": tp_mask, "matched_gt": matched_gt, "gt_used": used}


def render_error(ts: C.TestSet, img_id: int, pred: np.ndarray, gt: np.ndarray,
                 err: dict, etype: str, model: str, width: int = 1100) -> np.ndarray:
    img = ts.load_image(img_id)
    vis = C.draw_boxes(img, gt, (170, 170, 170), thickness=3)   # toàn bộ GT: xám
    idx = err["idx"][etype]
    if etype == "missed":
        boxes = gt[idx] if len(idx) else np.zeros((0, 4))
        labels = ["MISSED"] * len(idx)
    else:
        boxes = pred[idx, :4] if len(idx) else np.zeros((0, 4))
        labels = [f"{etype} {pred[i, 4]:.2f}" for i in idx]
    vis = C.draw_boxes(vis, boxes, COLORS[etype], labels, thickness=7, font_scale=1.6)
    vis = cv2.resize(vis, (width, int(vis.shape[0] * width / vis.shape[1])))
    return C.label_panel(
        vis, f"{cfg.DISPLAY_NAMES[model]} | {etype.upper()} x{len(idx)} | "
             f"{ts.coco.loadImgs(img_id)[0]['file_name']}")


def main():
    ap = argparse.ArgumentParser(description="Kịch bản D - phân tích lỗi")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--examples", type=int, default=3, help="số ảnh minh họa mỗi loại lỗi")
    args = ap.parse_args()

    C.set_seed()
    out_dir = cfg.RESULTS_DIR / args.tag
    img_out = out_dir / "error_analysis"
    img_out.mkdir(parents=True, exist_ok=True)

    ts = C.TestSet()
    eval_ids = C.load_json(out_dir / "eval_ids.json")
    models = cfg.MODEL_ORDER if args.models == ["all"] else args.models

    report = {"definitions": ERROR_TYPES,
              "conf_threshold": cfg.CONF_THRESHOLD,
              "match_iou": cfg.MATCH_IOU,
              "loc_error_iou_range": [LOC_LOW, LOC_HIGH],
              "models": {}}

    for model in models:
        preds = load_preds(out_dir, model)
        totals = defaultdict(int)
        n_images_with = defaultdict(int)
        ranked = defaultdict(list)      # etype -> [(count, img_id)]
        n_gt_total = n_pred_total = 0

        print(f"\n--- {cfg.DISPLAY_NAMES[model]} ---")
        for img_id in eval_ids:
            gt = ts.gt_boxes(img_id)
            pred = preds.get(img_id, np.zeros((0, 5), np.float32))
            n_gt_total += len(gt)
            n_pred_total += len(pred)
            err = classify_errors(pred, gt)
            for etype, idx in err["idx"].items():
                if idx:
                    totals[etype] += len(idx)
                    n_images_with[etype] += 1
                    ranked[etype].append((len(idx), img_id))

        stats = {}
        for etype in ERROR_TYPES:
            stats[etype] = {
                "count": totals[etype],
                "n_images_affected": n_images_with[etype],
                "rate_per_image": round(totals[etype] / max(1, len(eval_ids)), 4),
                "pct_of_gt" if etype == "missed" else "pct_of_pred": round(
                    100 * totals[etype] / max(1, n_gt_total if etype == "missed" else n_pred_total), 3),
            }
            print(f"  {etype:10s} {totals[etype]:7d}  ({n_images_with[etype]} ảnh bị dính)")

        # xuất ảnh minh họa: lấy ảnh có nhiều lỗi loại đó nhất
        examples = defaultdict(list)
        for etype in ERROR_TYPES:
            for _, img_id in sorted(ranked[etype], reverse=True)[:args.examples]:
                gt = ts.gt_boxes(img_id)
                pred = preds.get(img_id, np.zeros((0, 5), np.float32))
                err = classify_errors(pred, gt)
                canvas = render_error(ts, img_id, pred, gt, err, etype, model)
                fn = ts.coco.loadImgs(img_id)[0]["file_name"].replace(".jpg", "")
                path = img_out / f"{model}__{etype}__{fn}.jpg"
                cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
                examples[etype].append({"image_id": img_id, "file_name": fn + ".jpg",
                                        "n_errors": len(err["idx"][etype]),
                                        "level": ts.levels[img_id],
                                        "output": str(path.relative_to(cfg.APP_DIR))})

        report["models"][model] = {"n_images": len(eval_ids), "n_gt": n_gt_total,
                                   "n_pred": n_pred_total, "stats": stats,
                                   "examples": dict(examples)}

    C.save_json(report, out_dir / "error_analysis.json")

    print("\n" + "=" * 78 + "\nBẢNG - Phân bố lỗi (conf=%.2f, IoU=%.2f)\n" % (
        cfg.CONF_THRESHOLD, cfg.MATCH_IOU) + "=" * 78)
    print(C.md_table(["Model", *ERROR_TYPES],
                     [[cfg.DISPLAY_NAMES[m], *[report["models"][m]["stats"][e]["count"]
                                               for e in ERROR_TYPES]]
                      for m in report["models"]]))
    print(f"\nẢnh minh họa -> {img_out}")


if __name__ == "__main__":
    main()
