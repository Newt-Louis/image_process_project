"""KỊCH BẢN C - Định tính / trực quan (Mục 4C + Mục 5).

Chọn ảnh CÓ CHỦ ĐÍCH (không random) theo 6 tiêu chí ở Mục 5, rồi ghép 1 hình
gồm 4 panel: Ground-truth | YOLOv11 | Faster R-CNN | RetinaNet.
Box được tô màu theo TP (xanh lá) / FP (đỏ) / FN (vàng, vẽ trên panel model).

    python -m experiments.run_qualitative --tag main
    python -m experiments.run_qualitative --tag main --per-criterion 2
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import cv2
import numpy as np

from experiments import common as C
import config as cfg

TP_COLOR = (60, 200, 60)     # xanh lá
FP_COLOR = (60, 60, 235)     # đỏ
FN_COLOR = (40, 200, 235)    # vàng
GT_COLOR = (210, 210, 210)   # xám

CRITERIA_DESC = {
    "1_easy_sparse": "Ảnh dễ: ít vật, tách rời — baseline, cả 3 model nên detect tốt",
    "2_hard_dense": "Ảnh khó: nhiều vật + chồng lấp nhiều — phân hóa model rõ nhất",
    "3_high_density": "Mật độ cao (>=16 sản phẩm) — kiểm tra NMS và bỏ sót",
    "4_occlusion": "Sản phẩm chồng/che khuất nhau (IoU giữa các GT cao) — test recall",
    "5_false_positive": "Ảnh sinh nhiều FP nhất — nghi do bóng đổ/phản chiếu trên mặt bàn",
    "6_disagreement": "YOLO và torchvision cho kết quả lệch nhau rõ nhất — đối chiếu trực tiếp",
}


def gt_overlap_score(gt: np.ndarray) -> float:
    """Trung bình IoU lớn nhất của mỗi GT với GT khác -> mức chồng lấp của ảnh."""
    if len(gt) < 2:
        return 0.0
    m = C.iou_matrix(gt, gt)
    np.fill_diagonal(m, 0.0)
    return float(m.max(axis=1).mean())


def load_preds(out_dir, name) -> dict[int, np.ndarray]:
    """predictions COCO json -> {image_id: (N,5) xyxy+score} đã lọc conf vận hành."""
    by_img = defaultdict(list)
    for d in C.load_json(out_dir / f"predictions_{name}.json"):
        if d["score"] >= cfg.CONF_THRESHOLD:
            x, y, w, h = d["bbox"]
            by_img[d["image_id"]].append([x, y, x + w, y + h, d["score"]])
    return {k: np.array(sorted(v, key=lambda r: -r[4]), dtype=np.float32)
            for k, v in by_img.items()}


def image_stats(preds: np.ndarray, gt: np.ndarray) -> dict:
    tp, fp, _, used = C.match_greedy(preds[:, :4] if len(preds) else np.zeros((0, 4)), gt)
    return {"tp_mask": tp, "fp_mask": fp, "gt_used": used,
            "TP": int(tp.sum()), "FP": int(fp.sum()), "FN": int((~used).sum())}


def select_images(ts: C.TestSet, eval_ids: list[int], preds_all: dict, per_criterion: int) -> dict:
    """Trả về {criterion: [image_id, ...]}"""
    info = {}
    for i in eval_ids:
        gt = ts.gt_boxes(i)
        info[i] = {"n": len(gt), "overlap": gt_overlap_score(gt), "level": ts.levels[i]}

    # Lấy dư ứng viên để sau khi khử trùng lặp giữa các tiêu chí vẫn đủ ảnh
    depth = max(per_criterion * 10, 20)

    def top(key, reverse=True, filt=None, k=depth):
        pool = [i for i in eval_ids if (filt is None or filt(info[i]))]
        pool.sort(key=lambda i: info[i][key], reverse=reverse)
        return pool[:k]

    sel = {
        "1_easy_sparse": top("overlap", reverse=False,
                             filt=lambda d: d["level"] == "easy" and d["n"] <= 6),
        "2_hard_dense": top("overlap", reverse=True, filt=lambda d: d["level"] == "hard"),
        "3_high_density": top("n", reverse=True),
        "4_occlusion": top("overlap", reverse=True),
    }

    # (5) ảnh sinh nhiều FP nhất (tổng trên cả 3 model)
    fp_total = defaultdict(int)
    dis = defaultdict(float)
    for i in eval_ids:
        gt = ts.gt_boxes(i)
        f1s = {}
        for m in cfg.MODEL_ORDER:
            st = image_stats(preds_all[m].get(i, np.zeros((0, 5), np.float32)), gt)
            fp_total[i] += st["FP"]
            f1s[m] = C.prf_from_counts(st["TP"], st["FP"], st["FN"])["f1"]
        # (6) độ lệch giữa YOLO và 2 model torchvision
        dis[i] = max(abs(f1s["yolov11"] - f1s["fasterrcnn"]),
                     abs(f1s["yolov11"] - f1s["retinanet"]))

    sel["5_false_positive"] = sorted(eval_ids, key=lambda i: -fp_total[i])[:depth]
    sel["6_disagreement"] = sorted(eval_ids, key=lambda i: -dis[i])[:depth]

    # Khử trùng lặp: mỗi tiêu chí lấy `per_criterion` ứng viên tốt nhất CHƯA bị
    # tiêu chí trước lấy mất, thay vì bỏ trống tiêu chí đó.
    seen, dedup = set(), {}
    for k, ids in sel.items():
        keep = [i for i in ids if i not in seen][:per_criterion]
        seen.update(keep)
        dedup[k] = keep
    return dedup


def render_comparison(ts: C.TestSet, img_id: int, preds_all: dict, panel_w: int = 700) -> tuple:
    img = ts.load_image(img_id)
    gt = ts.gt_boxes(img_id)
    panels, stats = [], {}

    gt_panel = C.draw_boxes(img, gt, GT_COLOR, thickness=4)
    panels.append(C.label_panel(gt_panel, f"GROUND TRUTH ({len(gt)} objects)"))

    for m in cfg.MODEL_ORDER:
        p = preds_all[m].get(img_id, np.zeros((0, 5), np.float32))
        st = image_stats(p, gt)
        stats[m] = {k: st[k] for k in ("TP", "FP", "FN")}
        vis = img.copy()
        # FN: GT chưa được ghép -> vàng
        vis = C.draw_boxes(vis, gt[~st["gt_used"]] if len(gt) else gt, FN_COLOR, thickness=6)
        if len(p):
            vis = C.draw_boxes(vis, p[st["tp_mask"], :4], TP_COLOR,
                               [f"{s:.2f}" for s in p[st["tp_mask"], 4]], thickness=4, font_scale=1.4)
            vis = C.draw_boxes(vis, p[st["fp_mask"], :4], FP_COLOR,
                               [f"FP {s:.2f}" for s in p[st["fp_mask"], 4]], thickness=4, font_scale=1.4)
        panels.append(C.label_panel(
            vis, f"{cfg.DISPLAY_NAMES[m]}  TP={st['TP']} FP={st['FP']} FN={st['FN']}"))

    resized = [cv2.resize(p, (panel_w, int(p.shape[0] * panel_w / p.shape[1]))) for p in panels]
    h = max(p.shape[0] for p in resized)
    resized = [np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 30, np.uint8)])
               for p in resized]
    return np.hstack(resized), stats, len(gt)


def main():
    ap = argparse.ArgumentParser(description="Kịch bản C - ảnh định tính so sánh 3 model")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--per-criterion", type=int, default=1,
                    help="số ảnh mỗi tiêu chí (6 tiêu chí -> 4-8 ảnh là đủ cho báo cáo)")
    ap.add_argument("--panel-width", type=int, default=700)
    args = ap.parse_args()

    C.set_seed()
    out_dir = cfg.RESULTS_DIR / args.tag
    img_out = out_dir / "qualitative"
    img_out.mkdir(parents=True, exist_ok=True)

    ts = C.TestSet()
    eval_ids = C.load_json(out_dir / "eval_ids.json")
    preds_all = {m: load_preds(out_dir, m) for m in cfg.MODEL_ORDER}

    print(f"Đang chọn ảnh có chủ đích trong {len(eval_ids)} ảnh test ...")
    selected = select_images(ts, eval_ids, preds_all, args.per_criterion)

    manifest = []
    for crit, ids in selected.items():
        for img_id in ids:
            canvas, stats, n_gt = render_comparison(ts, img_id, preds_all, args.panel_width)
            fname = ts.coco.loadImgs(img_id)[0]["file_name"].replace(".jpg", "")
            path = img_out / f"{crit}__{fname}.jpg"
            cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append({"criterion": crit, "description": CRITERIA_DESC[crit],
                             "image_id": img_id, "file_name": fname + ".jpg",
                             "n_gt": n_gt, "level": ts.levels[img_id],
                             "output": str(path.relative_to(cfg.APP_DIR)), "per_model": stats})
            print(f"  [{crit}] {fname}.jpg  n_gt={n_gt} level={ts.levels[img_id]}  "
                  + "  ".join(f"{m}:TP{stats[m]['TP']}/FP{stats[m]['FP']}/FN{stats[m]['FN']}"
                              for m in cfg.MODEL_ORDER))

    C.save_json({"legend": {"xanh lá": "TP", "đỏ": "FP", "vàng": "FN (GT bị bỏ sót)",
                            "xám": "Ground truth"},
                 "conf_threshold": cfg.CONF_THRESHOLD, "match_iou": cfg.MATCH_IOU,
                 "items": manifest}, out_dir / "qualitative.json")
    print(f"\nXong: {len(manifest)} ảnh -> {img_out}")


if __name__ == "__main__":
    main()
