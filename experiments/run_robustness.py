"""KỊCH BẢN E [->TN] - Robustness: đo lại mAP khi ảnh bị nhiễu nhân tạo.

Áp từng loại nhiễu lên một tập con test rồi chạy lại inference + COCOeval, so
với baseline (ảnh sạch) trên đúng tập con đó -> % mAP suy giảm.

    python -m experiments.run_robustness --tag main --limit 500
    python -m experiments.run_robustness --tag main --limit 500 --corruptions blur_gauss jpeg_10
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from experiments import common as C
import config as cfg
import detectors


# --------------------------------------------------------------------------
# Bộ nhiễu. Mỗi hàm: (img_bgr, gt_xyxy) -> (img_bgr', gt_xyxy')
# --------------------------------------------------------------------------
def _brightness(img, gt, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8), gt


def _blur_gauss(img, gt, k):
    return cv2.GaussianBlur(img, (k, k), 0), gt


def _blur_motion(img, gt, k):
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel), gt


def _jpeg(img, gt, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), gt


def _rotate(img, gt, deg):
    """Xoay nhẹ; GT được xoay theo rồi lấy axis-aligned bbox bao ngoài (xấp xỉ)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if len(gt) == 0:
        return out, gt
    corners = np.stack([
        np.stack([gt[:, 0], gt[:, 1]], 1), np.stack([gt[:, 2], gt[:, 1]], 1),
        np.stack([gt[:, 2], gt[:, 3]], 1), np.stack([gt[:, 0], gt[:, 3]], 1)], 1)  # (N,4,2)
    ones = np.ones((*corners.shape[:2], 1), np.float32)
    rot = (np.concatenate([corners, ones], -1) @ M.T)  # (N,4,2)
    new = np.stack([rot[:, :, 0].min(1), rot[:, :, 1].min(1),
                    rot[:, :, 0].max(1), rot[:, :, 1].max(1)], 1)
    new[:, [0, 2]] = np.clip(new[:, [0, 2]], 0, w - 1)
    new[:, [1, 3]] = np.clip(new[:, [1, 3]], 0, h - 1)
    return out, new.astype(np.float32)


def _occlude(img, gt, frac, rng):
    """Che một phần: dán 3 mảng xám kích thước frac*cạnh ảnh lên vị trí ngẫu nhiên."""
    out = img.copy()
    h, w = img.shape[:2]
    bw, bh = int(w * frac), int(h * frac)
    for _ in range(3):
        x = rng.integers(0, max(1, w - bw))
        y = rng.integers(0, max(1, h - bh))
        out[y:y + bh, x:x + bw] = 120
    return out, gt


def build_corruptions(rng):
    return {
        "clean": lambda i, g: (i, g),
        "bright_up_1.5": lambda i, g: _brightness(i, g, 1.5),
        "bright_down_0.5": lambda i, g: _brightness(i, g, 0.5),
        "blur_gauss_9": lambda i, g: _blur_gauss(i, g, 9),
        "blur_motion_15": lambda i, g: _blur_motion(i, g, 15),
        "jpeg_30": lambda i, g: _jpeg(i, g, 30),
        "jpeg_10": lambda i, g: _jpeg(i, g, 10),
        "rotate_5deg": lambda i, g: _rotate(i, g, 5),
        "occlude_15pct": lambda i, g: _occlude(i, g, 0.15, rng),
    }


# --------------------------------------------------------------------------
def eval_with_local_gt(per_image: list[dict], iou_thrs=None) -> dict:
    """mAP tính tay trên GT đã biến đổi (không dùng COCO gốc vì GT có thể đổi khi xoay).

    Cách tính: 101-point interpolated AP, quét IoU 0.50:0.05:0.95 như COCO.
    """
    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 1.0, 0.05)
    n_gt = sum(len(r["gt"]) for r in per_image)
    if n_gt == 0:
        return {}

    aps = {}
    for thr in iou_thrs:
        rows = []
        for r in per_image:
            p = r["pred"]
            tp, fp, _, _ = C.match_greedy(p[:, :4] if len(p) else np.zeros((0, 4)), r["gt"], thr)
            for s, t in zip(p[:, 4] if len(p) else [], tp):
                rows.append((float(s), bool(t)))
        rows.sort(key=lambda x: -x[0])
        if not rows:
            aps[thr] = 0.0
            continue
        tps = np.cumsum([r[1] for r in rows])
        fps = np.cumsum([not r[1] for r in rows])
        rec = tps / n_gt
        prec = tps / np.maximum(tps + fps, 1e-9)
        # bao lồi precision
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        rec_thrs = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, rec_thrs, side="left")
        q = np.where(idx < len(prec), prec[np.clip(idx, 0, len(prec) - 1)], 0.0)
        aps[thr] = float(q.mean())

    return {
        "mAP50": round(aps[iou_thrs[0]], 4),
        "mAP75": round(aps[min(iou_thrs, key=lambda t: abs(t - 0.75))], 4),
        "mAP50_95": round(float(np.mean(list(aps.values()))), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Kịch bản E [->TN] - robustness")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--limit", type=int, default=500, help="số ảnh trong tập con")
    ap.add_argument("--corruptions", nargs="+", default=["all"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    C.set_seed()
    rng = np.random.default_rng(cfg.SEED)
    device = cfg.resolve_device(args.device)
    out_dir = cfg.RESULTS_DIR / args.tag
    ts = C.TestSet()
    img_ids = ts.subset(limit=args.limit, shuffle=True)

    all_corr = build_corruptions(rng)
    names = list(all_corr) if args.corruptions == ["all"] else ["clean", *args.corruptions]
    models = cfg.MODEL_ORDER if args.models == ["all"] else args.models

    print(f"Robustness: {len(models)} model x {len(names)} biến thể x {len(img_ids)} ảnh")
    results = {"n_images": len(img_ids), "corruptions": names, "models": {}}

    for model in models:
        det = detectors.build_detector(model, device=device)
        det.warmup([ts.load_image(i) for i in img_ids[:cfg.WARMUP_IMAGES]])
        per_corr = {}
        for cname in names:
            fn = all_corr[cname]
            per_image = []
            for img_id in img_ids:
                img, gt = fn(ts.load_image(img_id), ts.gt_boxes(img_id))
                per_image.append({"pred": det.predict(img, conf=cfg.CONF_RAW), "gt": gt})
            m = eval_with_local_gt(per_image)
            m.update(C.counts_at_conf(per_image, cfg.CONF_THRESHOLD))
            per_corr[cname] = m
            base = per_corr["clean"]["mAP50_95"]
            drop = 100 * (base - m["mAP50_95"]) / base if base else 0.0
            per_corr[cname]["mAP50_95_drop_pct"] = round(drop, 2)
            print(f"  {model:12s} {cname:16s} mAP50={m['mAP50']:.4f} "
                  f"mAP50-95={m['mAP50_95']:.4f} (giảm {drop:5.2f}%) F1={m['f1']:.4f}")
        results["models"][model] = per_corr
        detectors.unload_all()

    C.save_json(results, out_dir / "robustness.json")

    print("\n" + "=" * 78 + "\nBẢNG - Robustness (mAP50-95, % suy giảm so với ảnh sạch)\n" + "=" * 78)
    print(C.md_table(["Corruption", *[cfg.DISPLAY_NAMES[m] for m in results["models"]]],
                     [[c, *[f"{results['models'][m][c]['mAP50_95']:.4f}"
                            + ("" if c == "clean" else
                               f" (giảm {results['models'][m][c]['mAP50_95_drop_pct']:.1f}%)")
                            for m in results["models"]]] for c in names]))


if __name__ == "__main__":
    main()
