"""Tổng hợp mọi kết quả -> Bảng 1/2/3 (Mục 8) + biểu đồ (Mục 10) + báo cáo markdown.

    python -m experiments.make_report --tag main

Sinh ra trong results/<tag>/:
    BAO_CAO_THUC_NGHIEM.md   - toàn bộ bảng số đã điền + kết luận
    charts/pr_curve.png              - PR-curve chung cho cả 3 (Mục 3.1)
    charts/f1_confidence.png         - F1 theo ngưỡng conf
    charts/accuracy_bars.png         - mAP50 / mAP50-95 / mAP75
    charts/speed_bars.png            - FPS + latency p50/p95/p99
    charts/footprint_bars.png        - params / size đĩa / VRAM
    charts/map_by_level.png          - mAP theo độ khó
    charts/map_by_density.png        - mAP theo mật độ
    charts/robustness.png            - [->TN] nếu đã chạy run_robustness
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments import common as C  # noqa: E402
import config as cfg  # noqa: E402

# --- Bảng màu đã validate (categorical slot 1-3, light surface) -----------
SERIES = {"yolov11": "#2a78d6", "fasterrcnn": "#008300", "retinanet": "#e87ba4"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 130,
})


def _style(ax, title=None, xlabel=None, ylabel=None, ygrid=True):
    if title:
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y" if ygrid else "x", alpha=0.7)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, values, fmt="{:.3f}"):
    """Direct label trên từng cột — đồng thời là 'relief' cho slot màu contrast thấp."""
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK_2,
                    xytext=(0, 3), textcoords="offset points")


# ==========================================================================
# Biểu đồ
# ==========================================================================
def chart_pr_curve(evals, out):
    """Panel trái: toàn dải. Panel phải: phóng to góc trên-phải, nơi 3 đường mới tách nhau."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (xlim, ylim, title) in zip(axes, [
            ((0, 1.02), (0, 1.02), "Precision–Recall @ IoU=0.50"),
            ((0.85, 1.005), (0.85, 1.005), "Phóng to vùng recall ≥ 0.85")]):
        for m, r in evals.items():
            c = r["pr_curve"]
            if not c["recall"]:
                continue
            ax.plot(c["recall"], c["precision"], lw=2, color=SERIES[m],
                    label=f"{cfg.DISPLAY_NAMES[m]} (AP50={r['accuracy']['mAP50']:.3f})")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _style(ax, title, "Recall", "Precision")
    axes[0].legend(frameon=False, loc="lower left", fontsize=10)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png")
    plt.close(fig)


def chart_f1_conf(evals, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m, r in evals.items():
        c = r["f1_confidence_curve"]
        ax.plot(c["conf"], c["f1"], lw=2, color=SERIES[m], label=cfg.DISPLAY_NAMES[m])
        ax.plot([c["best_conf"]], [c["best_f1"]], "o", ms=8, color=SERIES[m])
        ax.annotate(f"{c['best_f1']:.3f} @ {c['best_conf']:.2f}",
                    (c["best_conf"], c["best_f1"]), xytext=(6, -12),
                    textcoords="offset points", fontsize=9, color=INK_2)
    ax.axvline(cfg.CONF_THRESHOLD, color=INK_2, ls="--", lw=1, alpha=0.5)
    ax.annotate(f"conf vận hành = {cfg.CONF_THRESHOLD}", (cfg.CONF_THRESHOLD, 0.02),
                xytext=(5, 0), textcoords="offset points", fontsize=9, color=INK_2)
    _style(ax, "F1 theo ngưỡng confidence (IoU ghép = 0.50)", "Confidence threshold", "F1-score")
    ax.legend(frameon=False, loc="lower center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "f1_confidence.png")
    plt.close(fig)


def _grouped(ax, groups, models, getter, fmt="{:.3f}"):
    x = np.arange(len(groups))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [getter(m, g) for g in groups]
        b = ax.bar(x + k * w - 0.4 + w / 2, vals, w * 0.88, color=SERIES[m],
                   label=cfg.DISPLAY_NAMES[m])
        _bar_labels(ax, b, vals, fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def chart_accuracy_bars(evals, out):
    metrics = [("mAP50", "mAP@0.50"), ("mAP50_95", "mAP@0.50:0.95"), ("mAP75", "mAP@0.75"),
               ("AP_large", "AP_large"), ("AR100", "AR@100")]
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped(ax, [lbl for _, lbl in metrics], list(evals),
             lambda m, g: evals[m]["accuracy"][dict((l, k) for k, l in metrics)[g]])
    ax.set_ylim(0, 1.12)
    _style(ax, "Độ chính xác — cùng test set, cùng NMS IoU, cùng pycocotools", ylabel="Giá trị")
    ax.legend(frameon=False, ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "accuracy_bars.png")
    plt.close(fig)


def chart_speed(evals, out):
    """FPS và latency là 2 đơn vị khác nhau -> 2 trục riêng, KHÔNG dual-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    models = list(evals)

    vals = [evals[m]["system"]["latency"]["fps"] for m in models]
    b = axes[0].bar([cfg.DISPLAY_NAMES[m] for m in models], vals,
                    color=[SERIES[m] for m in models], width=0.55)
    _bar_labels(axes[0], b, vals, "{:.1f}")
    axes[0].set_ylim(0, max(vals) * 1.2)
    _style(axes[0], "Throughput (batch=1, sau warm-up)", ylabel="FPS")

    groups = ["p50", "p95", "p99"]
    _grouped(axes[1], groups, models,
             lambda m, g: evals[m]["system"]["latency"][f"{g}_ms"], "{:.1f}")
    _style(axes[1], "Latency mỗi ảnh", ylabel="ms")
    lat_max = max(evals[m]["system"]["latency"]["p99_ms"] for m in models)
    axes[1].set_ylim(0, lat_max * 1.35)   # chừa chỗ cho legend, không đè lên cột
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "speed_bars.png")
    plt.close(fig)


def chart_footprint(evals, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    models = list(evals)
    specs = [("Số tham số", lambda m: evals[m]["system"]["num_params_m"], "M params", "{:.2f}"),
             ("Dung lượng trên đĩa", lambda m: evals[m]["system"]["model_size_mb"], "MB", "{:.1f}"),
             ("VRAM peak khi inference", lambda m: evals[m]["system"]["vram_peak_infer_gb"] or 0,
              "GB", "{:.2f}")]
    for ax, (title, fn, unit, fmt) in zip(axes, specs):
        vals = [fn(m) for m in models]
        b = ax.bar([cfg.DISPLAY_NAMES[m] for m in models], vals,
                   color=[SERIES[m] for m in models], width=0.55)
        _bar_labels(ax, b, vals, fmt)
        ax.set_ylim(0, max(max(vals) * 1.2, 1e-6))
        _style(ax, title, ylabel=unit)
        ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(out / "footprint_bars.png")
    plt.close(fig)


def chart_slices(slices, out, kind, metric="mAP50_95"):
    key = "by_level" if kind == "level" else "by_density"
    models = list(slices["models"])
    groups = [g for g in ([n for n, _, _ in cfg.LEVEL_BINS] if kind == "level"
                          else [n for n, _, _ in cfg.DENSITY_BINS])
              if g in slices["models"][models[0]][key]]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _grouped(ax, groups, models, lambda m, g: slices["models"][m][key][g][metric])
    ax.set_ylim(0, 1.1)
    title = ("mAP@0.50:0.95 theo độ khó" + (" (level suy ra từ số instance/ảnh)"
                                            if slices.get("level_is_proxy") else "")
             if kind == "level" else "mAP@0.50:0.95 theo mật độ sản phẩm/ảnh")
    _style(ax, title, xlabel="easy → hard" if kind == "level" else "số sản phẩm/ảnh",
           ylabel="mAP@0.50:0.95")
    ax.legend(frameon=False, ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"map_by_{kind}.png")
    plt.close(fig)


def chart_robustness(rob, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    corr = rob["corruptions"]
    for m in rob["models"]:
        vals = [rob["models"][m][c]["mAP50_95"] for c in corr]
        ax.plot(range(len(corr)), vals, "-o", lw=2, ms=7, color=SERIES[m],
                label=cfg.DISPLAY_NAMES[m])
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr, rotation=25, ha="right")
    _style(ax, f"[→TN] Robustness — mAP@0.50:0.95 dưới nhiễu nhân tạo "
               f"({rob['n_images']} ảnh)", ylabel="mAP@0.50:0.95")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "robustness.png")
    plt.close(fig)


# ==========================================================================
# Báo cáo markdown
# ==========================================================================
def build_markdown(tag, env, evals, slices, errors, rob, cpu) -> str:
    ms = list(evals)
    L = []
    A = L.append
    A(f"# Kết quả thực nghiệm 3 model detector (class-agnostic) — tag `{tag}`\n")
    A(f"*Sinh tự động bởi `experiments/make_report.py` lúc {env['timestamp']}.*\n")

    # --- Mục 1 ---
    A("## 1. Môi trường thực nghiệm (local)\n")
    g = env.get("gpu")
    A(C.md_table(["Hạng mục", "Giá trị"], [
        ["CPU", f"{env['cpu']} ({env['cpu_cores_physical']} core / {env['cpu_threads']} thread)"],
        ["RAM", f"{env['ram_total_gb']} GB"],
        ["GPU", f"{g['name']} — {g['vram_total_gb']} GB VRAM, driver {g['driver']}" if g else "không có"],
        ["CUDA (torch build)", env["cuda_version_torch"]],
        ["cuDNN", env["cudnn_version"]],
        ["OS", env["platform"]],
        ["Python", env["python"]],
        ["torch / torchvision", f"{env['packages']['torch']} / {env['packages']['torchvision']}"],
        ["ultralytics", env["packages"]["ultralytics"]],
        ["pycocotools / opencv", f"{env['packages']['pycocotools']} / {env['packages']['opencv']}"],
        ["Device dùng để đo", env["device_used"]],
        ["Seed", env["measurement"]["seed"]],
    ]))
    A("\n> **Lưu ý bắt buộc nêu trong báo cáo:** số FPS/latency dưới đây đo trên máy local "
      "này, KHÁC môi trường train (Colab A100). Không được trộn lẫn hai bộ số. "
      "Tương tự, VRAM *train peak* (4.18 / 32.36 / 27.79 GB) là số lúc train, khác hoàn "
      "toàn với VRAM *inference* đo ở Bảng 2.\n")

    # --- Mục 2 ---
    m0 = env["measurement"]
    A("## 2. Điều kiện so sánh đã chuẩn hóa\n")
    A(C.md_table(["Điều kiện", "Giá trị (áp dụng chung cho cả 3 model)"], [
        ["Tập test", f"{env['dataset']['n_images_evaluated']} / "
                     f"{env['dataset']['n_images_total']} ảnh của `gold_dataset/test`"],
        ["Ground-truth", f"`{Path(env['dataset']['ann_file']).name}` "
                         f"({env['dataset']['n_gt_boxes_total']} bbox)"],
        ["Batch size", m0["batch_size"]],
        ["Warm-up", f"{m0['warmup_images']} ảnh trước khi bấm giờ"],
        ["Conf khi tính mAP", f"{m0['conf_threshold_raw_for_mAP']} (không lọc — để pycocotools quét toàn dải)"],
        ["Conf vận hành (P/R/F1, visualize)", m0["conf_threshold_operational"]],
        ["IoU cho NMS", m0["nms_iou"]],
        ["max detections / ảnh", m0["max_det"]],
        ["Cách tính mAP", m0["map_backend"]],
        ["Độ phân giải input", ", ".join(f"{k}: {v}" for k, v in m0["input_resolution"].items())],
    ]))

    sz = env["dataset"]["object_size_distribution_coco"]
    rpc = sz.get("rpc_original_test2019")
    A(f"\n### 2.1. Giải thích `AP_medium = 0` (Mục 2.6 của bản hướng dẫn)\n")
    if rpc:
        A(C.md_table(["Nhóm kích thước (chuẩn COCO)", "gold_dataset/test", "Tỉ lệ",
                      "RPC gốc (test2019)", "Tỉ lệ"],
                     [[k, sz[k], f"{100 * sz[k] / sz['total']:.4f}%",
                       rpc[k], f"{100 * rpc[k] / rpc['total']:.4f}%"]
                      for k in ("small (area < 32^2)", "medium (32^2 <= area < 96^2)",
                                "large (area >= 96^2)")]))
    else:
        A(C.md_table(["Nhóm kích thước (chuẩn COCO)", "Số object", "Tỉ lệ"],
                     [[k, v, f"{100 * v / sz['total']:.4f}%"] for k, v in sz.items()
                      if k not in ("total", "rpc_original_test2019")]))
    n_med = sz["medium (32^2 <= area < 96^2)"]
    A(f"\nTest set gần như **toàn large-object**: chỉ {n_med} object thuộc nhóm *medium* và "
      f"{sz['small (area < 32^2)']} object *small* trên tổng {sz['total']}. Với cỡ mẫu nhỏ như vậy, "
      "chỉ cần model miss vài object là `AP_medium` tụt về 0 và `AP_small` là `-1` (COCO trả -1 khi "
      "không có mẫu). **Đây không phải lỗi của model.**")
    if rpc:
        A(f"\nQuan trọng: đây cũng **không phải do quá trình cắt gold_dataset** làm mất vật nhỏ. "
          f"Kiểm chứng trên chính bộ RPC **gốc** `instances_test2019.json` (24.000 ảnh, "
          f"{rpc['total']:,} object): vẫn chỉ có **{rpc['small (area < 32^2)']}** vật *small* và "
          f"**{rpc['medium (32^2 <= area < 96^2)']}** vật *medium*. Ảnh RPC chụp từ trên xuống, sản "
          "phẩm chiếm ~1/10 khung hình 1850px nên gần như không tồn tại vật small/medium theo "
          "định nghĩa COCO — đây là **đặc điểm cố hữu của dataset**, không có file nào 'chữa' được. ")
    A("Chỉ số size đáng tin duy nhất trên bài toán này là `AP_large`; `AP_small`/`AP_medium` "
      "phải được ghi chú rõ khi đưa vào báo cáo.\n")

    A("### 2.2. Chênh lệch số epoch (95 vs 9)\n")
    A("YOLOv11 train 95 epoch (best@80); Faster R-CNN & RetinaNet chỉ 9 epoch (best@3). "
      "Phương án đang áp dụng là **(b) — giữ nguyên và biện luận bằng learning curve**: "
      "xem `fasterrcnn/learning_curves.png`, `retinanet/learning_curves.png` và "
      "`fasterrcnn/history.json`, `retinanet/history.json` — val mAP của cả hai model "
      "torchvision bão hòa ngay sau epoch 3 nên early-stop tại đó là hợp lý, không phải "
      "do train thiếu. Cần chèn 2 biểu đồ này vào báo cáo làm bằng chứng.\n")

    # --- Bảng 1 ---
    A("## 3. Bảng 1 — Accuracy (local, cùng điều kiện)\n")
    A(C.md_table(["Model", "mAP50", "mAP50-95", "mAP75", "P", "R", "F1", "AP_L",
                  "AR100", "TP", "FP", "FN"],
                 [[cfg.DISPLAY_NAMES[m], C.fmt(evals[m]["accuracy"]["mAP50"]),
                   C.fmt(evals[m]["accuracy"]["mAP50_95"]), C.fmt(evals[m]["accuracy"]["mAP75"]),
                   C.fmt(evals[m]["operational"]["precision"]),
                   C.fmt(evals[m]["operational"]["recall"]), C.fmt(evals[m]["operational"]["f1"]),
                   C.fmt(evals[m]["accuracy"]["AP_large"]), C.fmt(evals[m]["accuracy"]["AR100"]),
                   evals[m]["operational"]["TP"], evals[m]["operational"]["FP"],
                   evals[m]["operational"]["FN"]] for m in ms]))
    A(f"\n*P/R/F1/TP/FP/FN đo tại conf={m0['conf_threshold_operational']}, ghép greedy theo IoU="
      f"{cfg.MATCH_IOU}. Bài class-agnostic 1 lớp nên không có confusion matrix đa lớp — "
      "TP/FP/FN theo IoU chính là 'confusion' của bài toán này.*\n")

    A("### 3.1. AP/AR đầy đủ theo COCO\n")
    A(C.md_table(["Chỉ số", *[cfg.DISPLAY_NAMES[m] for m in ms]],
                 [[k, *[C.fmt(evals[m]["accuracy"][k]) for m in ms]] for k in C.COCO_STAT_KEYS]))
    A("\n*Giá trị `-1.0000` = COCO không có mẫu ở nhóm đó (xem lại mục 2.1).*\n")

    A("### 3.2. Recall theo kích thước bbox\n")
    sizes = list(evals[ms[0]]["recall_by_bbox_size"])
    A(C.md_table(["Nhóm bbox", "Số GT", *[cfg.DISPLAY_NAMES[m] for m in ms]],
                 [[s, evals[ms[0]]["recall_by_bbox_size"][s]["n_gt"],
                   *[C.fmt(evals[m]["recall_by_bbox_size"][s]["recall"]) for m in ms]]
                  for s in sizes]))

    A("\n### 3.3. Ngưỡng confidence tối ưu (F1–confidence curve)\n")
    A(C.md_table(["Model", "conf tối ưu", "F1 tại đó", f"F1 tại conf={cfg.CONF_THRESHOLD}"],
                 [[cfg.DISPLAY_NAMES[m], evals[m]["f1_confidence_curve"]["best_conf"],
                   C.fmt(evals[m]["f1_confidence_curve"]["best_f1"]),
                   C.fmt(evals[m]["operational"]["f1"])] for m in ms]))

    # --- Bảng 2 ---
    A("\n## 4. Bảng 2 — System (local, inference)\n")
    A(C.md_table(["Model", "Latency p50 (ms)", "p95", "p99", "mean", "FPS", "Size đĩa (MB)",
                  "Params (M)", "VRAM infer (GB)", "RAM RSS (GB)", "Load model (s)"],
                 [[cfg.DISPLAY_NAMES[m], evals[m]["system"]["latency"]["p50_ms"],
                   evals[m]["system"]["latency"]["p95_ms"], evals[m]["system"]["latency"]["p99_ms"],
                   evals[m]["system"]["latency"]["mean_ms"], evals[m]["system"]["latency"]["fps"],
                   evals[m]["system"]["model_size_mb"], evals[m]["system"]["num_params_m"],
                   evals[m]["system"]["vram_peak_infer_gb"],
                   evals[m]["system"]["ram_rss_running_gb"],
                   evals[m]["system"]["load_time_s"]] for m in ms]))
    A(f"\n*batch=1, đã warm-up {m0['warmup_images']} ảnh, đo trên "
      f"{evals[ms[0]]['system']['latency']['n_samples']} ảnh, device = `{env['device_used']}`. "
      "Latency là end-to-end (tiền xử lý + inference + hậu xử lý/NMS), đã `cuda.synchronize()`.*\n")

    if cpu:
        A("### 4.1. [→TN] CPU benchmark / quantize\n")
        A(C.md_table(["Variant", "p50 (ms)", "p95", "FPS", "mAP50", "mAP50-95", "Size (MB)", "RAM (GB)"],
                     [[k, v["latency"]["p50_ms"], v["latency"]["p95_ms"], v["latency"]["fps"],
                       C.fmt(v["accuracy"]["mAP50"]), C.fmt(v["accuracy"]["mAP50_95"]),
                       v["model_size_mb"], v["ram_rss_gb"]] for k, v in cpu["variants"].items()]))
        if "quantization_impact" in cpu:
            q = cpu["quantization_impact"]
            A(f"\nQuantize INT8: mAP50-95 {q['mAP50_95_before']:.4f} → {q['mAP50_95_after']:.4f} "
              f"(mất {q['mAP50_95_drop_pct']}%), nhanh hơn {q['speedup_x']}×, "
              f"nhẹ hơn {q['size_reduction_x']}×.\n")

    # --- Bảng 3 ---
    if slices:
        A("\n## 5. Bảng 3 — mAP theo lát cắt dữ liệu\n")
        if slices.get("level_is_proxy"):
            A("> ⚠️ `instances_test.json` của gold_dataset đã bị lược bỏ field `level` của RPC gốc. "
              "Level dưới đây được **suy ra từ số instance/ảnh** theo quy ước clutter của RPC ("
              + ", ".join(f"{n}={lo}–{hi}" for n, lo, hi in cfg.LEVEL_BINS)
              + "). Phải ghi rõ điều này trong báo cáo. Đặt `instances_test2019.json` gốc vào "
                "`demo_app/data/` để tự động dùng nhãn chuẩn.\n")
        else:
            A("> Nhãn độ khó `level` (easy/medium/hard) lấy **trực tiếp từ annotation RPC gốc** "
              "`instances_test2019.json` (join theo `file_name`, khớp 100% với gold_dataset/test) — "
              "**không phải số suy đoán**. Phân bố: "
              + ", ".join(f"{g}={slices['models'][ms[0]]['by_level'].get(g, {}).get('n_images', 0)}"
                          for g in ("easy", "medium", "hard")) + " ảnh.\n")
        for kind, key, title in [("level", "by_level", "5.1. Theo độ khó (`level`)"),
                                 ("density", "by_density", "5.2. Theo mật độ sản phẩm/ảnh")]:
            groups = list(slices["models"][ms[0]][key])
            A(f"\n### {title}\n")
            A(C.md_table(["Model", *groups],
                         [[cfg.DISPLAY_NAMES[m],
                           *[f"{slices['models'][m][key][g]['mAP50_95']:.4f}" for g in groups]]
                          for m in slices["models"]]))
            A(f"\n*Số ảnh mỗi nhóm: "
              + ", ".join(f"{g}={slices['models'][ms[0]][key][g]['n_images']}" for g in groups) + "*\n")

    # --- Error analysis ---
    if errors:
        A("\n## 6. Phân tích lỗi (Kịch bản D)\n")
        et = list(errors["definitions"])
        A(C.md_table(["Model", *et],
                     [[cfg.DISPLAY_NAMES[m], *[errors["models"][m]["stats"][e]["count"] for e in et]]
                      for m in errors["models"]]))
        A("\nĐịnh nghĩa:\n")
        for k, v in errors["definitions"].items():
            A(f"- `{k}`: {v}")
        A(f"\nẢnh minh họa mỗi loại: `results/{tag}/error_analysis/`.\n")

    # --- Robustness ---
    if rob:
        A("\n## 7. [→TN] Robustness dưới nhiễu\n")
        A(C.md_table(["Nhiễu", *[cfg.DISPLAY_NAMES[m] for m in rob["models"]]],
                     [[c, *[f"{rob['models'][m][c]['mAP50_95']:.4f}"
                            + ("" if c == "clean" else
                               f" (giảm {rob['models'][m][c]['mAP50_95_drop_pct']:.1f}%)")
                            for m in rob["models"]]] for c in rob["corruptions"]]))
        A(f"\n*Đo trên {rob['n_images']} ảnh lấy ngẫu nhiên (seed={cfg.SEED}) từ test set. "
          "`rotate_5deg` xoay cả GT rồi lấy bbox axis-aligned bao ngoài nên là xấp xỉ.*\n")

    # --- Kết luận ---
    best = max(ms, key=lambda m: evals[m]["accuracy"]["mAP50_95"])
    fastest = max(ms, key=lambda m: evals[m]["system"]["latency"]["fps"])
    lightest = min(ms, key=lambda m: evals[m]["system"]["num_params"])
    A("\n## 8. Kết luận chọn model\n")
    A(C.md_table(["Tiêu chí (thứ tự ưu tiên, Mục 9)", "Model thắng", "Số liệu"], [
        ["1. Độ chính xác (mAP50-95)", cfg.DISPLAY_NAMES[best],
         " · ".join(f"{cfg.DISPLAY_NAMES[m]} {evals[m]['accuracy']['mAP50_95']:.4f}" for m in ms)],
        ["2. Tốc độ (FPS)", cfg.DISPLAY_NAMES[fastest],
         " · ".join(f"{cfg.DISPLAY_NAMES[m]} {evals[m]['system']['latency']['fps']:.1f}" for m in ms)],
        ["3. Độ nhẹ (params)", cfg.DISPLAY_NAMES[lightest],
         " · ".join(f"{cfg.DISPLAY_NAMES[m]} {evals[m]['system']['num_params_m']:.2f}M" for m in ms)],
    ]))
    if best == fastest == lightest:
        speed_x = (evals[best]["system"]["latency"]["fps"]
                   / max(evals[m]["system"]["latency"]["fps"] for m in ms if m != best))
        param_x = (max(evals[m]["system"]["num_params"] for m in ms)
                   / evals[best]["system"]["num_params"])
        A(f"\n**{cfg.DISPLAY_NAMES[best]} vượt trội đồng thời cả độ chính xác lẫn tốc độ, lại nhẹ "
          f"nhất** (nhanh hơn ~{speed_x:.1f}×, ít tham số hơn ~{param_x:.1f}× so với model nặng "
          "nhất). Đây **không phải một đánh đổi** mà là **ưu thế toàn diện** → chốt "
          f"**{cfg.DISPLAY_NAMES[best]}** làm detector cho hệ thống. Với đồ án tốt nghiệp, đặc "
          "tính nhẹ này càng phù hợp mục tiêu chạy trên thiết bị tài nguyên hạn chế.\n")
    else:
        A(f"\nModel chính xác nhất: **{cfg.DISPLAY_NAMES[best]}**; nhanh nhất: "
          f"**{cfg.DISPLAY_NAMES[fastest]}**; nhẹ nhất: **{cfg.DISPLAY_NAMES[lightest]}**. "
          "Kết quả local không trùng khớp hoàn toàn — cần biện luận lại theo thứ tự ưu tiên ở Mục 9.\n")

    A("\n## 9. Đối chiếu với số liệu train (Colab A100)\n")
    A(C.md_table(["Model", "mAP50 (train)", "mAP50 (local)", "mAP50-95 (train)", "mAP50-95 (local)",
                  "FPS (Colab)", "FPS (local)"],
                 [[cfg.DISPLAY_NAMES[m], t50, C.fmt(evals[m]["accuracy"]["mAP50"]),
                   t5095, C.fmt(evals[m]["accuracy"]["mAP50_95"]), tfps,
                   evals[m]["system"]["latency"]["fps"]]
                  for m, t50, t5095, tfps in
                  [("yolov11", 0.990, 0.883, 100.5), ("fasterrcnn", 0.989, 0.799, 37.6),
                   ("retinanet", 0.981, 0.736, 39.3)] if m in evals]))
    A("\n*Cột 'train' lấy từ `comparison_results.json` để đối chiếu, KHÔNG dùng làm số local.*\n")

    A("\n## 10. Sản phẩm đầu ra\n")
    for f, d in [("charts/pr_curve.png", "PR-curve chung cho cả 3 model"),
                 ("charts/f1_confidence.png", "F1 theo ngưỡng confidence"),
                 ("charts/accuracy_bars.png", "Bar chart mAP/AP/AR"),
                 ("charts/speed_bars.png", "FPS + latency p50/p95/p99"),
                 ("charts/footprint_bars.png", "Params / size đĩa / VRAM"),
                 ("charts/map_by_level.png", "mAP theo độ khó"),
                 ("charts/map_by_density.png", "mAP theo mật độ"),
                 ("charts/robustness.png", "[→TN] mAP dưới nhiễu"),
                 ("qualitative/", "4–8 ảnh định tính: GT | 3 model, chú thích TP/FP/FN"),
                 ("error_analysis/", "Ảnh minh họa 4 loại lỗi"),
                 ("summary_all_models.json", "Toàn bộ số liệu thô")]:
        p = cfg.RESULTS_DIR / tag / f
        A(f"- {'✅' if p.exists() else '⬜'} `results/{tag}/{f}` — {d}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Tổng hợp bảng + biểu đồ + báo cáo")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    out_dir = cfg.RESULTS_DIR / args.tag
    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    env = C.load_json(out_dir / "environment.json")
    evals = {}
    for m in cfg.MODEL_ORDER:
        f = out_dir / f"eval_{m}.json"
        if f.exists():
            evals[m] = C.load_json(f)
    if not evals:
        raise SystemExit(f"Chưa có kết quả trong {out_dir} — chạy `run_eval.py` trước.")

    opt = lambda n: C.load_json(out_dir / n) if (out_dir / n).exists() else None  # noqa: E731
    slices, errors, rob = opt("slices.json"), opt("error_analysis.json"), opt("robustness.json")
    cpu = opt("cpu_bench_yolov11.json")

    print("Đang vẽ biểu đồ ...")
    chart_pr_curve(evals, charts)
    chart_f1_conf(evals, charts)
    chart_accuracy_bars(evals, charts)
    chart_speed(evals, charts)
    chart_footprint(evals, charts)
    if slices:
        chart_slices(slices, charts, "level")
        chart_slices(slices, charts, "density")
    if rob:
        chart_robustness(rob, charts)
    print(f"  -> {charts}")

    md = build_markdown(args.tag, env, evals, slices, errors, rob, cpu)
    path = out_dir / "BAO_CAO_THUC_NGHIEM.md"
    path.write_text(md, encoding="utf-8")
    print(f"  -> đã ghi {path}")
    print("\nMở file này để copy thẳng các bảng vào báo cáo.")


if __name__ == "__main__":
    main()
