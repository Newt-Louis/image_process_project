"""Sinh 2 hình cho báo cáo mục 4.5/4.6:
  - fig_4_10_pipeline.png : sơ đồ khối pipeline demo (vẽ bằng PIL)
  - fig_4_11_example.png  : ví dụ pipeline thật (input -> bbox+conf -> lưới crop) chạy qua YOLOv11
  - fig_4_6_cropgrid.png  : lưới crop sản phẩm của ảnh khó nhất
Chạy: cd demo_app && ../.venv/bin/python gen_report_figures.py
"""
from __future__ import annotations
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as cfg
import detectors

OUT = cfg.RESULTS_DIR / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


# --------------------------------------------------------------------------
# 1) SƠ ĐỒ KHỐI PIPELINE
# --------------------------------------------------------------------------
def block_diagram():
    W, H = 1180, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fb, fs = _font(24), _font(18)

    steps = [
        ("Ảnh đầu vào", "upload từ camera / internet\nhoặc chọn ảnh tập test", (70, 130, 210)),
        ("Tiền xử lý theo từng model", "YOLOv11: letterbox 640×640\nFaster R-CNN / RetinaNet: cạnh ngắn 800, cạnh dài ≤ 1333", (100, 100, 180)),
        ("Suy luận (chọn 1–3 model)", "YOLOv11 · Faster R-CNN · RetinaNet\nchạy song song trên cùng ảnh", (150, 90, 160)),
        ("Lọc confidence ≥ 0.25  +  NMS IoU 0.5", "chuẩn hóa output → (x1,y1,x2,y2,score), tối đa 100 box", (170, 110, 90)),
        ("Vẽ bbox + nhãn confidence", "màu riêng từng model; nếu có ground-truth: tô TP/FP/FN", (110, 150, 90)),
        ("Cắt (crop) từng sản phẩm", "cắt vùng mỗi bbox thành ảnh riêng", (90, 150, 150)),
        ("Trả kết quả + ghi log JSON", "ảnh đã vẽ, lưới crop, latency, số box, confidence, TP/FP/FN", (70, 120, 120)),
    ]

    bx, bw, bh, gap = 190, 800, 78, 34
    y = 30
    centers = []
    for title, sub, col in steps:
        d.rounded_rectangle([bx, y, bx + bw, y + bh], radius=14, fill=col)
        d.text((bx + bw / 2, y + 22), title, font=fb, fill="white", anchor="mm")
        for i, line in enumerate(sub.split("\n")):
            d.text((bx + bw / 2, y + 44 + i * 18), line, font=fs, fill=(240, 240, 240), anchor="mm")
        centers.append((bx + bw / 2, y, y + bh))
        y += bh + gap

    # mũi tên nối
    for i in range(len(centers) - 1):
        cx, _, y0 = centers[i]
        _, y1, _ = centers[i + 1]
        d.line([cx, y0, cx, y1], fill=(80, 80, 80), width=3)
        d.polygon([(cx - 8, y1 - 12), (cx + 8, y1 - 12), (cx, y1)], fill=(80, 80, 80))

    img.save(OUT / "fig_4_10_pipeline.png")
    print("saved fig_4_10_pipeline.png")


# --------------------------------------------------------------------------
# 2) VÍ DỤ PIPELINE THẬT + LƯỚI CROP (chạy YOLOv11)
# --------------------------------------------------------------------------
def _draw_boxes_conf(img_bgr, boxes):
    vis = img_bgr.copy()
    thick = max(2, int(round(vis.shape[1] / 500)))
    fscale = max(0.5, vis.shape[1] / 1400)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 220, 60), thick)
        cv2.putText(vis, f"{b[4]:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, (60, 220, 60), thick)
    return vis


def _crop_grid(img_bgr, boxes, cols=5, tile=150, pad=8):
    crops = []
    h, w = img_bgr.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        c = img_bgr[y1:y2, x1:x2]
        ch, cw = c.shape[:2]
        s = min(tile / cw, tile / ch)
        c = cv2.resize(c, (max(1, int(cw * s)), max(1, int(ch * s))))
        canvas = np.full((tile, tile, 3), 255, np.uint8)
        oy, ox = (tile - c.shape[0]) // 2, (tile - c.shape[1]) // 2
        canvas[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
        cv2.putText(canvas, f"{b[4]:.2f}", (4, tile - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        crops.append(canvas)
    if not crops:
        return None
    rows = (len(crops) + cols - 1) // cols
    grid = np.full((rows * tile + (rows + 1) * pad, cols * tile + (cols + 1) * pad, 3), 245, np.uint8)
    for i, c in enumerate(crops):
        r, cc = divmod(i, cols)
        y = pad + r * (tile + pad)
        x = pad + cc * (tile + pad)
        grid[y:y + tile, x:x + tile] = c
    return grid


def _label_bar(width, text, h=42):
    bar = np.full((h, width, 3), 40, np.uint8)
    cv2.putText(bar, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return bar


def pipeline_example(fname="test_05_hard_hard_19obj__20180911-15-36-08-135.jpg"):
    det = detectors.get_detector("yolov11")
    path = cfg.UPLOADS_DIR / fname
    img = cv2.imread(str(path))
    boxes, latency = det.predict_timed(img, conf=cfg.CONF_THRESHOLD)
    print(f"{fname}: {len(boxes)} box, {latency:.1f} ms")

    annotated = _draw_boxes_conf(img, boxes)
    grid = _crop_grid(img, boxes)

    # 3 panel: input | annotated | crop grid (chuẩn hóa cùng chiều cao)
    Ht = 720

    def fit_h(im):
        s = Ht / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), Ht))

    p1, p2, p3 = fit_h(img), fit_h(annotated), fit_h(grid)
    panels = [(_label_bar(p1.shape[1], "1. Anh dau vao"), p1),
              (_label_bar(p2.shape[1], f"2. Bbox + confidence ({len(boxes)} san pham, {latency:.0f} ms)"), p2),
              (_label_bar(p3.shape[1], "3. Luoi crop tung san pham"), p3)]
    cols = [np.vstack([bar, pan]) for bar, pan in panels]
    Hmax = max(c.shape[0] for c in cols)
    cols = [np.vstack([c, np.full((Hmax - c.shape[0], c.shape[1], 3), 255, np.uint8)]) for c in cols]
    sep = np.full((Hmax, 12, 3), 255, np.uint8)
    combo = np.hstack([cols[0], sep, cols[1], sep, cols[2]])
    cv2.imwrite(str(OUT / "fig_4_11_example.png"), combo)
    print("saved fig_4_11_example.png")

    if grid is not None:
        cv2.imwrite(str(OUT / "fig_4_6_cropgrid.png"), grid)
        print("saved fig_4_6_cropgrid.png")


if __name__ == "__main__":
    block_diagram()
    pipeline_example()
    print("DONE ->", OUT)
