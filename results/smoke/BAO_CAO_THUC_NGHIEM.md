# Kết quả thực nghiệm 3 model detector (class-agnostic) — tag `smoke`

*Sinh tự động bởi `experiments/make_report.py` lúc 2026-07-26 23:12:33.*

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
| Tập test | 80 / 19200 ảnh của `gold_dataset/test` |
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

| Nhóm kích thước (chuẩn COCO) | gold_dataset/test | Tỉ lệ | RPC gốc (test2019) | Tỉ lệ |
|---|---|---|---|---|
| small (area < 32^2) | 0 | 0.0000% | 0 | 0.0000% |
| medium (32^2 <= area < 96^2) | 7 | 0.0030% | 10 | 0.0034% |
| large (area >= 96^2) | 235741 | 99.9970% | 294323 | 99.9966% |

Test set gần như **toàn large-object**: chỉ 7 object thuộc nhóm *medium* và 0 object *small* trên tổng 235748. Với cỡ mẫu nhỏ như vậy, chỉ cần model miss vài object là `AP_medium` tụt về 0 và `AP_small` là `-1` (COCO trả -1 khi không có mẫu). **Đây không phải lỗi của model.**

Quan trọng: đây cũng **không phải do quá trình cắt gold_dataset** làm mất vật nhỏ. Kiểm chứng trên chính bộ RPC **gốc** `instances_test2019.json` (24.000 ảnh, 294,333 object): vẫn chỉ có **0** vật *small* và **10** vật *medium*. Ảnh RPC chụp từ trên xuống, sản phẩm chiếm ~1/10 khung hình 1850px nên gần như không tồn tại vật small/medium theo định nghĩa COCO — đây là **đặc điểm cố hữu của dataset**, không có file nào 'chữa' được. 
Chỉ số size đáng tin duy nhất trên bài toán này là `AP_large`; `AP_small`/`AP_medium` phải được ghi chú rõ khi đưa vào báo cáo.

### 2.2. Chênh lệch số epoch (95 vs 9)

YOLOv11 train 95 epoch (best@80); Faster R-CNN & RetinaNet chỉ 9 epoch (best@3). Phương án đang áp dụng là **(b) — giữ nguyên và biện luận bằng learning curve**: xem `fasterrcnn/learning_curves.png`, `retinanet/learning_curves.png` và `fasterrcnn/history.json`, `retinanet/history.json` — val mAP của cả hai model torchvision bão hòa ngay sau epoch 3 nên early-stop tại đó là hợp lý, không phải do train thiếu. Cần chèn 2 biểu đồ này vào báo cáo làm bằng chứng.

## 3. Bảng 1 — Accuracy (local, cùng điều kiện)

| Model | mAP50 | mAP50-95 | mAP75 | P | R | F1 | AP_L | AR100 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv11 | 0.9901 | 0.8799 | 0.9773 | 0.9970 | 0.9970 | 0.9970 | 0.8810 | 0.9109 | 982 | 3 | 3 |
| Faster R-CNN | 0.9851 | 0.7912 | 0.9323 | 0.9008 | 0.9868 | 0.9419 | 0.7919 | 0.8342 | 972 | 107 | 13 |
| RetinaNet | 0.9801 | 0.7303 | 0.8727 | 0.6528 | 0.9848 | 0.7851 | 0.7318 | 0.7861 | 970 | 516 | 15 |

*P/R/F1/TP/FP/FN đo tại conf=0.25, ghép greedy theo IoU=0.5. Bài class-agnostic 1 lớp nên không có confusion matrix đa lớp — TP/FP/FN theo IoU chính là 'confusion' của bài toán này.*

### 3.1. AP/AR đầy đủ theo COCO

| Chỉ số | YOLOv11 | Faster R-CNN | RetinaNet |
|---|---|---|---|
| mAP50_95 | 0.8799 | 0.7912 | 0.7303 |
| mAP50 | 0.9901 | 0.9851 | 0.9801 |
| mAP75 | 0.9773 | 0.9323 | 0.8727 |
| AP_small | -1.0000 | -1.0000 | -1.0000 |
| AP_medium | 0.2020 | 0.0009 | 0.0000 |
| AP_large | 0.8810 | 0.7919 | 0.7318 |
| AR1 | 0.0780 | 0.0724 | 0.0691 |
| AR10 | 0.6790 | 0.6320 | 0.5942 |
| AR100 | 0.9109 | 0.8342 | 0.7861 |
| AR_small | -1.0000 | -1.0000 | -1.0000 |
| AR_medium | 0.2000 | 0.2000 | 0.0000 |
| AR_large | 0.9123 | 0.8355 | 0.7877 |

*Giá trị `-1.0000` = COCO không có mẫu ở nhóm đó (xem lại mục 2.1).*

### 3.2. Recall theo kích thước bbox

| Nhóm bbox | Số GT | YOLOv11 | Faster R-CNN | RetinaNet |
|---|---|---|---|---|
| small (<32^2) | 0 | n/a | n/a | n/a |
| medium (32^2-96^2) | 2 | 0.5000 | 0.0000 | 0.0000 |
| large (>=96^2) | 983 | 0.9980 | 0.9888 | 0.9868 |

### 3.3. Ngưỡng confidence tối ưu (F1–confidence curve)

| Model | conf tối ưu | F1 tại đó | F1 tại conf=0.25 |
|---|---|---|---|
| YOLOv11 | 0.55 | 0.9985 | 0.9970 |
| Faster R-CNN | 0.95 | 0.9796 | 0.9419 |
| RetinaNet | 0.8 | 0.9600 | 0.7851 |

## 4. Bảng 2 — System (local, inference)

| Model | Latency p50 (ms) | p95 | p99 | mean | FPS | Size đĩa (MB) | Params (M) | VRAM infer (GB) | RAM RSS (GB) | Load model (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv11 | 8.3 | 9.66 | 11.76 | 8.47 | 118.03 | 18.29 | 9.428 | 0.108 | 1.655 | 0.613 |
| Faster R-CNN | 67.32 | 72.29 | 73.91 | 67.7 | 14.77 | 158.03 | 41.352 | 0.554 | 1.815 | 0.445 |
| RetinaNet | 61.86 | 65.98 | 67.35 | 62.19 | 16.08 | 123.2 | 32.222 | 0.362 | 1.954 | 0.363 |

*batch=1, đã warm-up 10 ảnh, đo trên 80 ảnh, device = `cuda`. Latency là end-to-end (tiền xử lý + inference + hậu xử lý/NMS), đã `cuda.synchronize()`.*


## 5. Bảng 3 — mAP theo lát cắt dữ liệu

> Nhãn độ khó `level` (easy/medium/hard) lấy **trực tiếp từ annotation RPC gốc** `instances_test2019.json` (join theo `file_name`, khớp 100% với gold_dataset/test) — **không phải số suy đoán**. Phân bố: easy=24, medium=29, hard=27 ảnh.


### 5.1. Theo độ khó (`level`)

| Model | easy | medium | hard |
|---|---|---|---|
| YOLOv11 | 0.9205 | 0.8673 | 0.8781 |
| Faster R-CNN | 0.8154 | 0.8048 | 0.7773 |
| RetinaNet | 0.7817 | 0.7550 | 0.6991 |

*Số ảnh mỗi nhóm: easy=24, medium=29, hard=27*


### 5.2. Theo mật độ sản phẩm/ảnh

| Model | <=8 | 9-15 | >=16 |
|---|---|---|---|
| YOLOv11 | 0.9200 | 0.8820 | 0.8632 |
| Faster R-CNN | 0.8179 | 0.8105 | 0.7612 |
| RetinaNet | 0.7842 | 0.7546 | 0.6871 |

*Số ảnh mỗi nhóm: <=8=20, 9-15=39, >=16=21*


## 6. Phân tích lỗi (Kịch bản D)

| Model | missed | false_pos | double | loc_error |
|---|---|---|---|---|
| YOLOv11 | 3 | 1 | 0 | 2 |
| Faster R-CNN | 13 | 18 | 21 | 68 |
| RetinaNet | 15 | 8 | 102 | 406 |

Định nghĩa:

- `missed`: Missed detection — sản phẩm bị bỏ sót (che khuất/chồng lấp)
- `false_pos`: False positive — báo nhầm nền/bóng đổ thành sản phẩm
- `double`: Double detection — 1 vật 2 box, NMS chưa gộp
- `loc_error`: Box lệch — IoU 0.3-0.5, kéo tụt mAP@0.75

Ảnh minh họa mỗi loại: `results/smoke/error_analysis/`.


## 8. Kết luận chọn model

| Tiêu chí (thứ tự ưu tiên, Mục 9) | Model thắng | Số liệu |
|---|---|---|
| 1. Độ chính xác (mAP50-95) | YOLOv11 | YOLOv11 0.8799 · Faster R-CNN 0.7912 · RetinaNet 0.7303 |
| 2. Tốc độ (FPS) | YOLOv11 | YOLOv11 118.0 · Faster R-CNN 14.8 · RetinaNet 16.1 |
| 3. Độ nhẹ (params) | YOLOv11 | YOLOv11 9.43M · Faster R-CNN 41.35M · RetinaNet 32.22M |

**YOLOv11 vượt trội đồng thời cả độ chính xác lẫn tốc độ, lại nhẹ nhất** (nhanh hơn ~7.3×, ít tham số hơn ~4.4× so với model nặng nhất). Đây **không phải một đánh đổi** mà là **ưu thế toàn diện** → chốt **YOLOv11** làm detector cho hệ thống. Với đồ án tốt nghiệp, đặc tính nhẹ này càng phù hợp mục tiêu chạy trên thiết bị tài nguyên hạn chế.


## 9. Đối chiếu với số liệu train (Colab A100)

| Model | mAP50 (train) | mAP50 (local) | mAP50-95 (train) | mAP50-95 (local) | FPS (Colab) | FPS (local) |
|---|---|---|---|---|---|---|
| YOLOv11 | 0.99 | 0.9901 | 0.883 | 0.8799 | 100.5 | 118.03 |
| Faster R-CNN | 0.989 | 0.9851 | 0.799 | 0.7912 | 37.6 | 14.77 |
| RetinaNet | 0.981 | 0.9801 | 0.736 | 0.7303 | 39.3 | 16.08 |

*Cột 'train' lấy từ `comparison_results.json` để đối chiếu, KHÔNG dùng làm số local.*


## 10. Sản phẩm đầu ra

- ✅ `results/smoke/charts/pr_curve.png` — PR-curve chung cho cả 3 model
- ✅ `results/smoke/charts/f1_confidence.png` — F1 theo ngưỡng confidence
- ✅ `results/smoke/charts/accuracy_bars.png` — Bar chart mAP/AP/AR
- ✅ `results/smoke/charts/speed_bars.png` — FPS + latency p50/p95/p99
- ✅ `results/smoke/charts/footprint_bars.png` — Params / size đĩa / VRAM
- ✅ `results/smoke/charts/map_by_level.png` — mAP theo độ khó
- ✅ `results/smoke/charts/map_by_density.png` — mAP theo mật độ
- ⬜ `results/smoke/charts/robustness.png` — [→TN] mAP dưới nhiễu
- ✅ `results/smoke/qualitative/` — 4–8 ảnh định tính: GT | 3 model, chú thích TP/FP/FN
- ✅ `results/smoke/error_analysis/` — Ảnh minh họa 4 loại lỗi
- ✅ `results/smoke/summary_all_models.json` — Toàn bộ số liệu thô
