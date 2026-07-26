# Kết quả thực nghiệm 3 model detector (class-agnostic) — tag `main`

*Sinh tự động bởi `experiments/make_report.py` lúc 2026-07-21 22:56:35.*

## 1. Môi trường thực nghiệm (local)

| Hạng mục | Giá trị |
|---|---|
| CPU | Intel(R) Core(TM) i7-14700F (2 core / 4 thread) |
| RAM | 7.76 GB |
| GPU | NVIDIA GeForce RTX 4060 — 8.0 GB VRAM, driver 576.88 |
| CUDA (torch build) | 12.8 |
| cuDNN | 91002 |
| OS | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| Python | 3.13.13 |
| torch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| ultralytics | 8.4.94 |
| pycocotools / opencv | 2.0.11 / 5.0.0 |
| Device dùng để đo | cuda |
| Seed | 42 |

> **Lưu ý bắt buộc nêu trong báo cáo:** số FPS/latency dưới đây đo trên máy local này, KHÁC môi trường train (Colab A100). Không được trộn lẫn hai bộ số. Tương tự, VRAM *train peak* (4.18 / 32.36 / 27.79 GB) là số lúc train, khác hoàn toàn với VRAM *inference* đo ở Bảng 2.

## 2. Điều kiện so sánh đã chuẩn hóa

| Điều kiện | Giá trị (áp dụng chung cho cả 3 model) |
|---|---|
| Tập test | 19200 / 19200 ảnh của `gold_dataset/test` |
| Ground-truth | `instances_test.json` (235748 bbox) |
| Batch size | 1 |
| Warm-up | 10 ảnh trước khi bấm giờ |
| Conf khi tính mAP | 0.001 (không lọc — để pycocotools quét toàn dải) |
| Conf vận hành (P/R/F1, visualize) | 0.25 |
| IoU cho NMS | 0.5 |
| max detections / ảnh | 100 |
| Cách tính mAP | pycocotools (COCOeval bbox) cho cả 3 model |
| Độ phân giải input | yolov11: 640x640, fasterrcnn: min=800/max=1333, retinanet: min=800/max=1333 |

### 2.1. Giải thích `AP_medium = 0` (Mục 2.6 của bản hướng dẫn)

| Nhóm kích thước (chuẩn COCO) | Số object | Tỉ lệ |
|---|---|---|
| small (area < 32^2) | 0 | 0.0000% |
| medium (32^2 <= area < 96^2) | 7 | 0.0030% |
| large (area >= 96^2) | 235741 | 99.9970% |

Test set gần như **toàn large-object**: chỉ 7 object thuộc nhóm *medium* và 0 object *small* trên tổng 235748. Với cỡ mẫu nhỏ như vậy, chỉ cần model miss vài object là `AP_medium` tụt về 0 và `AP_small` là `-1` (COCO trả -1 khi không có mẫu). **Đây không phải lỗi của model** — chỉ số này không có ý nghĩa thống kê trên dataset RPC checkout và phải được ghi chú rõ khi đưa vào báo cáo. Chỉ số size đáng tin duy nhất ở đây là `AP_large`.

### 2.2. Chênh lệch số epoch (95 vs 9)

YOLOv11 train 95 epoch (best@80); Faster R-CNN & RetinaNet chỉ 9 epoch (best@3). Phương án đang áp dụng là **(b) — giữ nguyên và biện luận bằng learning curve**: xem `fasterrcnn/learning_curves.png`, `retinanet/learning_curves.png` và `fasterrcnn/history.json`, `retinanet/history.json` — val mAP của cả hai model torchvision bão hòa ngay sau epoch 3 nên early-stop tại đó là hợp lý, không phải do train thiếu. Cần chèn 2 biểu đồ này vào báo cáo làm bằng chứng.

## 3. Bảng 1 — Accuracy (local, cùng điều kiện)

| Model | mAP50 | mAP50-95 | mAP75 | P | R | F1 | AP_L | AR100 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv11 | 0.9900 | 0.8826 | 0.9762 | 0.9975 | 0.9975 | 0.9975 | 0.8826 | 0.9124 | 235153 | 580 | 595 |
| Faster R-CNN | 0.9891 | 0.8003 | 0.9305 | 0.8926 | 0.9932 | 0.9402 | 0.8003 | 0.8392 | 234145 | 28168 | 1603 |
| RetinaNet | 0.9808 | 0.7358 | 0.8678 | 0.6563 | 0.9880 | 0.7887 | 0.7358 | 0.7889 | 232908 | 121948 | 2840 |

*P/R/F1/TP/FP/FN đo tại conf=0.25, ghép greedy theo IoU=0.5. Bài class-agnostic 1 lớp nên không có confusion matrix đa lớp — TP/FP/FN theo IoU chính là 'confusion' của bài toán này.*

### 3.1. AP/AR đầy đủ theo COCO

| Chỉ số | YOLOv11 | Faster R-CNN | RetinaNet |
|---|---|---|---|
| mAP50_95 | 0.8826 | 0.8003 | 0.7358 |
| mAP50 | 0.9900 | 0.9891 | 0.9808 |
| mAP75 | 0.9762 | 0.9305 | 0.8678 |
| AP_small | -1.0000 | -1.0000 | -1.0000 |
| AP_medium | 0.0695 | 0.0000 | 0.0000 |
| AP_large | 0.8826 | 0.8003 | 0.7358 |
| AR1 | 0.0789 | 0.0739 | 0.0706 |
| AR10 | 0.6887 | 0.6382 | 0.6003 |
| AR100 | 0.9124 | 0.8392 | 0.7889 |
| AR_small | -1.0000 | -1.0000 | -1.0000 |
| AR_medium | 0.1143 | 0.0571 | 0.0000 |
| AR_large | 0.9125 | 0.8393 | 0.7889 |

*Giá trị `-1.0000` = COCO không có mẫu ở nhóm đó (xem lại mục 2.1).*

### 3.2. Recall theo kích thước bbox

| Nhóm bbox | Số GT | YOLOv11 | Faster R-CNN | RetinaNet |
|---|---|---|---|---|
| small (<32^2) | 0 | n/a | n/a | n/a |
| medium (32^2-96^2) | 7 | 0.2857 | 0.0000 | 0.0000 |
| large (>=96^2) | 235741 | 0.9975 | 0.9932 | 0.9880 |

### 3.3. Ngưỡng confidence tối ưu (F1–confidence curve)

| Model | conf tối ưu | F1 tại đó | F1 tại conf=0.25 |
|---|---|---|---|
| YOLOv11 | 0.35 | 0.9977 | 0.9975 |
| Faster R-CNN | 0.95 | 0.9855 | 0.9402 |
| RetinaNet | 0.8 | 0.9577 | 0.7887 |

## 4. Bảng 2 — System (local, inference)

| Model | Latency p50 (ms) | p95 | p99 | mean | FPS | Size đĩa (MB) | Params (M) | VRAM infer (GB) | RAM RSS (GB) | Load model (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv11 | 10.26 | 15.03 | 15.48 | 11.21 | 89.2 | 18.29 | 9.428 | 0.108 | 1.894 | 0.505 |
| Faster R-CNN | 71.2 | 75.41 | 76.44 | 71.32 | 14.02 | 158.03 | 41.352 | 0.521 | 2.267 | 0.472 |
| RetinaNet | 61.01 | 64.13 | 67.61 | 61.27 | 16.32 | 123.2 | 32.222 | 0.329 | 2.621 | 0.357 |

*batch=1, đã warm-up 10 ảnh, đo trên 300 ảnh, device = `cuda`. Latency là end-to-end (tiền xử lý + inference + hậu xử lý/NMS), đã `cuda.synchronize()`.*


## 5. Bảng 3 — mAP theo lát cắt dữ liệu

> ⚠️ `instances_test.json` của gold_dataset đã bị lược bỏ field `level` của RPC gốc. Level dưới đây được **suy ra từ số instance/ảnh** theo quy ước clutter của RPC (easy=3–10, medium=11–15, hard=16–20). Phải ghi rõ điều này trong báo cáo. Nếu có `instances_test2019.json` gốc, chạy lại với `--level-source` để dùng nhãn chuẩn.


### 5.1. Theo độ khó (`level`)

| Model | easy | medium | hard |
|---|---|---|---|
| YOLOv11 | 0.9251 | 0.8743 | 0.8591 |
| Faster R-CNN | 0.8341 | 0.8016 | 0.7766 |
| RetinaNet | 0.7752 | 0.7383 | 0.7093 |

*Số ảnh mỗi nhóm: easy=7471, medium=6383, hard=5346*


### 5.2. Theo mật độ sản phẩm/ảnh

| Model | <=8 | 9-15 | >=16 |
|---|---|---|---|
| YOLOv11 | 0.9345 | 0.8832 | 0.8591 |
| Faster R-CNN | 0.8434 | 0.8071 | 0.7766 |
| RetinaNet | 0.7856 | 0.7439 | 0.7093 |

*Số ảnh mỗi nhóm: <=8=4755, 9-15=9099, >=16=5346*


## 6. Phân tích lỗi (Kịch bản D)

| Model | missed | false_pos | double | loc_error |
|---|---|---|---|---|
| YOLOv11 | 595 | 197 | 35 | 348 |
| Faster R-CNN | 1603 | 5164 | 5813 | 17193 |
| RetinaNet | 2841 | 1734 | 25254 | 94964 |

Định nghĩa:

- `missed`: Missed detection — sản phẩm bị bỏ sót (che khuất/chồng lấp)
- `false_pos`: False positive — báo nhầm nền/bóng đổ thành sản phẩm
- `double`: Double detection — 1 vật 2 box, NMS chưa gộp
- `loc_error`: Box lệch — IoU 0.3-0.5, kéo tụt mAP@0.75

Ảnh minh họa mỗi loại: `results/main/error_analysis/`.


## 8. Kết luận chọn model

| Tiêu chí (thứ tự ưu tiên, Mục 9) | Model thắng | Số liệu |
|---|---|---|
| 1. Độ chính xác (mAP50-95) | YOLOv11 | YOLOv11 0.8826 · Faster R-CNN 0.8003 · RetinaNet 0.7358 |
| 2. Tốc độ (FPS) | YOLOv11 | YOLOv11 89.2 · Faster R-CNN 14.0 · RetinaNet 16.3 |
| 3. Độ nhẹ (params) | YOLOv11 | YOLOv11 9.43M · Faster R-CNN 41.35M · RetinaNet 32.22M |

**YOLOv11 vượt trội đồng thời cả độ chính xác lẫn tốc độ, lại nhẹ nhất** (nhanh hơn ~5.5×, ít tham số hơn ~4.4× so với model nặng nhất). Đây **không phải một đánh đổi** mà là **ưu thế toàn diện** → chốt **YOLOv11** làm detector cho hệ thống. Với đồ án tốt nghiệp, đặc tính nhẹ này càng phù hợp mục tiêu chạy trên thiết bị tài nguyên hạn chế.


## 9. Đối chiếu với số liệu train (Colab A100)

| Model | mAP50 (train) | mAP50 (local) | mAP50-95 (train) | mAP50-95 (local) | FPS (Colab) | FPS (local) |
|---|---|---|---|---|---|---|
| YOLOv11 | 0.99 | 0.9900 | 0.883 | 0.8826 | 100.5 | 89.2 |
| Faster R-CNN | 0.989 | 0.9891 | 0.799 | 0.8003 | 37.6 | 14.02 |
| RetinaNet | 0.981 | 0.9808 | 0.736 | 0.7358 | 39.3 | 16.32 |

*Cột 'train' lấy từ `comparison_results.json` để đối chiếu, KHÔNG dùng làm số local.*


## 10. Sản phẩm đầu ra

- ✅ `results/main/charts/pr_curve.png` — PR-curve chung cho cả 3 model
- ✅ `results/main/charts/f1_confidence.png` — F1 theo ngưỡng confidence
- ✅ `results/main/charts/accuracy_bars.png` — Bar chart mAP/AP/AR
- ✅ `results/main/charts/speed_bars.png` — FPS + latency p50/p95/p99
- ✅ `results/main/charts/footprint_bars.png` — Params / size đĩa / VRAM
- ✅ `results/main/charts/map_by_level.png` — mAP theo độ khó
- ✅ `results/main/charts/map_by_density.png` — mAP theo mật độ
- ⬜ `results/main/charts/robustness.png` — [→TN] mAP dưới nhiễu
- ✅ `results/main/qualitative/` — 4–8 ảnh định tính: GT | 3 model, chú thích TP/FP/FN
- ✅ `results/main/error_analysis/` — Ảnh minh họa 4 loại lỗi
- ✅ `results/main/summary_all_models.json` — Toàn bộ số liệu thô
