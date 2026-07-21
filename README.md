# Demo + Thực nghiệm 3 model detector — RPC Checkout

Hệ thống nạp **3 model detector đã fine-tune thật** (không còn code giả lập) và chạy đầy đủ
bộ thực nghiệm mô tả trong `../HUONG_DAN_THUC_NGHIEM_3_MODEL_DETECTOR.md`.

| Model | Checkpoint | Params | Size |
|---|---|---|---|
| YOLOv11 (Ultralytics) | `../yolov11/yolov11_product/weights/best.pt` | 9.43 M | 18.3 MB |
| Faster R-CNN (torchvision `fasterrcnn_resnet50_fpn`) | `../fasterrcnn/best.pth` | 41.35 M | 158.0 MB |
| RetinaNet (torchvision `retinanet_resnet50_fpn`) | `../retinanet/best.pth` | 32.22 M | 123.2 MB |

Cả 3 được chuẩn hóa về **cùng một format output** `[x1, y1, x2, y2, score]` trước khi đưa vào
cùng hàm tính mAP (Mục 6.5 của bản hướng dẫn) → so sánh công bằng.

---

## 1. Cấu trúc

```
demo_app/
├── config.py                 # MỌI hằng số ảnh hưởng tới so sánh công bằng (conf, NMS IoU, seed, …)
├── detectors.py              # 3 loader + chuẩn hóa output về cùng format
├── main.py                   # FastAPI server cho web demo
├── run_all_experiments.sh    # chạy tuần tự kịch bản A→D (+E,F)
├── requirements.txt          # khóa version để tái lập
├── static/                   # UI (HTML/CSS/JS thuần, không framework)
├── data/gold_dataset/        # test set giải nén từ gold_dataset_detector_3models.zip
│   ├── images/test/          #   19.200 ảnh
│   └── coco_annotations/instances_test.json
├── experiments/
│   ├── common.py             # seed, env, dataset, metrics, đo tài nguyên, vẽ
│   ├── run_eval.py           # KỊCH BẢN A — định lượng toàn tập (Bảng 1 + Bảng 2)
│   ├── run_slices.py         # KỊCH BẢN B — mAP theo độ khó & mật độ (Bảng 3)
│   ├── run_qualitative.py    # KỊCH BẢN C — ảnh định tính GT|3 model, chú thích TP/FP/FN
│   ├── run_error_analysis.py # KỊCH BẢN D — 4 loại lỗi + ảnh minh họa
│   ├── run_robustness.py     # KỊCH BẢN E [→TN] — mAP dưới nhiễu nhân tạo
│   ├── run_cpu_bench.py      # KỊCH BẢN F [→TN] — CPU + quantize INT8
│   └── make_report.py        # gộp tất cả → BAO_CAO_THUC_NGHIEM.md + biểu đồ
└── results/<tag>/            # toàn bộ output
```

---

## 2. Cài đặt

```bash
cd /home/newt-louis/newt_workspaces/my_air_projects/image_process_project
source .venv/bin/activate

# torch phải khớp driver NVIDIA. Máy này (driver 576.88 = CUDA 12.9) dùng bản cu128:
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r demo_app/requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # phải in True
```

> ⚠️ Nếu in `False`, bản torch không khớp driver → mọi thứ chạy CPU và **không có số VRAM/FPS GPU**
> cho Bảng 2. Xem chú thích trong `requirements.txt`.

**Giải nén dataset** (đã làm sẵn, chỉ cần lặp lại nếu mất):

```bash
cd demo_app/data
unzip -q ../../gold_dataset_detector_3models.zip \
  "coco_annotations/*" "data.yaml" "images/test/*" "labels/test/*" -d gold_dataset
```

---

## 3. Chạy web demo

```bash
cd demo_app
uvicorn main:app --host 0.0.0.0 --port 8000
```

Mở `http://localhost:8000`. Giao diện có 3 tab:

**Tab "Ảnh thật (upload)"** — kéo thả **ảnh chụp thật từ camera/điện thoại hoặc ảnh tải từ
internet**. Chọn 1 hoặc cả 3 model → chạy song song, hiển thị cạnh nhau: ảnh đã vẽ bbox,
số box, latency, FPS, confidence trung bình, và **ảnh từng sản phẩm đã cắt ra**.

**Tab "Tập test gold_dataset"** — duyệt/lọc 19.200 ảnh test theo độ khó, bấm 1 ảnh để chạy.
Vì có ground-truth nên box được tô: 🟢 TP · 🔴 FP · 🟡 FN (bỏ sót), kèm P/R/F1 từng model.

**Tab "Kết quả thực nghiệm"** — hiển thị Bảng 1/2/3, biểu đồ và ảnh định tính đã sinh ra
(sau khi chạy phần thực nghiệm ở mục 4).

Hai thanh trượt **Confidence** và **NMS IoU** áp dụng chung cho cả 3 model.

> Latency hiển thị trên UI chỉ mang tính tham khảo (có overhead HTTP + encode ảnh).
> Số chính thức cho báo cáo lấy từ `run_eval.py`.

---

## 4. Chạy thực nghiệm — lấy đủ số liệu cho báo cáo

### Cách nhanh nhất

```bash
cd demo_app
source ../.venv/bin/activate

./run_all_experiments.sh              # toàn bộ 19.200 ảnh → results/main/
./run_all_experiments.sh 1000 thu     # chạy thử 1000 ảnh → results/thu/
WITH_TN=1 ./run_all_experiments.sh    # kèm kịch bản E + F (đồ án tốt nghiệp)
```

Thời gian ước tính trên máy này (RTX 4060, WSL2 4 thread):
YOLOv11 ~20 phút · Faster R-CNN ~40 phút · RetinaNet ~38 phút → **tổng ~1,5–2 giờ**
cho toàn bộ 19.200 ảnh. Phần lớn thời gian là **decode ảnh 1850px trên CPU**, không phải GPU
— nếu muốn nhanh hơn, tăng số CPU cấp cho WSL trong `C:\Users\<user>\.wslconfig`:

```ini
[wsl2]
processors=12
memory=24GB
```

### Hoặc chạy từng bước

```bash
python -m experiments.run_eval             --tag main   # A · Bảng 1 + Bảng 2 (BẮT BUỘC chạy trước)
python -m experiments.run_slices           --tag main   # B · Bảng 3 (không chạy lại inference)
python -m experiments.run_qualitative      --tag main   # C · 6 ảnh định tính
python -m experiments.run_error_analysis   --tag main   # D · 4 loại lỗi
python -m experiments.run_robustness       --tag main --limit 500             # E [→TN]
python -m experiments.run_cpu_bench        --tag main --model yolov11 --onnx  # F [→TN]
python -m experiments.make_report          --tag main   # gộp → báo cáo + biểu đồ
```

Tuỳ chọn hữu ích:

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--limit N` | chỉ chạy N ảnh (lấy mẫu **ngẫu nhiên** seed=42 — vì test set xếp theo độ khó) |
| `--no-shuffle` | với `--limit`: lấy N ảnh đầu thay vì lấy mẫu ngẫu nhiên |
| `--models yolov11 fasterrcnn` | chỉ chạy một số model |
| `--device cpu` | ép chạy CPU (dùng cho cột `[→TN] CPU ms/ảnh`) |
| `--nms-iou 0.7` | đổi ngưỡng NMS (ghi rõ giá trị dùng trong báo cáo) |
| `--tag <tên>` | tách kết quả sang thư mục khác, không đè lên lần chạy trước |
| `--level-source <path>` | dùng field `level` gốc của RPC (xem mục 6b) |

### Kết quả sinh ra

```
results/main/
├── BAO_CAO_THUC_NGHIEM.md    ← FILE CHÍNH: Bảng 1/2/3 đã điền + kết luận
├── environment.json           # thông số môi trường (Mục 1 — bắt buộc nêu trong báo cáo)
├── eval_<model>.json          # accuracy + system + PR curve + F1 curve của từng model
├── predictions_<model>.json   # prediction thô format COCO
├── slices.json                # mAP theo level & mật độ
├── error_analysis.json        # thống kê 4 loại lỗi
├── robustness.json            # [→TN]
├── charts/                    # 8 biểu đồ PNG (PR-curve chung, bar chart, mAP theo độ khó, …)
├── qualitative/               # 6 ảnh so sánh GT | YOLOv11 | Faster R-CNN | RetinaNet
└── error_analysis/            # ảnh minh họa từng loại lỗi cho từng model
```

Mở `BAO_CAO_THUC_NGHIEM.md` → copy thẳng bảng markdown vào báo cáo.

---

## 5. Điều kiện so sánh đã chuẩn hóa (Mục 2 bản hướng dẫn)

Tất cả nằm trong `config.py`, áp dụng **giống hệt** cho cả 3 model:

| Điều kiện | Giá trị |
|---|---|
| Tập test | 19.200 ảnh `gold_dataset/test`, cùng `instances_test.json` |
| Conf khi tính mAP | `0.001` — **không lọc**, để pycocotools quét toàn dải |
| Conf vận hành (P/R/F1, visualize) | `0.25` |
| IoU cho NMS | `0.5` |
| max detections/ảnh | `100` |
| Cách tính mAP | **pycocotools COCOeval** cho cả 3 (không dùng mAP nội bộ của Ultralytics) |
| Batch size | 1 |
| Warm-up | ≥ 10 ảnh trước khi bấm giờ |
| Seed | 42 |
| Latency | end-to-end (tiền xử lý + infer + NMS), đã `cuda.synchronize()`, có p50/p95/p99 |

---

## 6. Ba điểm cần ghi rõ trong báo cáo

**(a) `AP_medium = 0` không phải lỗi model.** Test set có 235.748 object thì
**235.741 là large**, chỉ 7 object medium và 0 object small (chuẩn COCO). Cỡ mẫu quá nhỏ nên
`AP_medium` không có ý nghĩa thống kê và `AP_small` trả `-1`. Script đã tự tính và giải thích
số này ở mục 2.1 của báo cáo sinh ra.

**(b) Nhãn `level` là proxy.** File `instances_test.json` của gold_dataset **đã bị lược bỏ
field `level`** (easy/medium/hard) có trong annotation RPC gốc. Script suy ra level từ số
instance/ảnh theo quy ước clutter của RPC (easy 3–10, medium 11–15, hard 16–20), cho ra
7.471 / 6.383 / 5.346 ảnh — khớp tỉ lệ ~1/3 của RPC. Nếu bạn tải được
`instances_test2019.json` gốc về, chạy với nhãn chuẩn:

```bash
python -m experiments.run_eval   --tag main --level-source /đường/dẫn/instances_test2019.json
python -m experiments.run_slices --tag main --level-source /đường/dẫn/instances_test2019.json
```

**(c) Chênh lệch epoch (95 vs 9).** Đang dùng phương án (b) của bản hướng dẫn — giữ nguyên và
biện luận bằng learning curve. Bằng chứng có sẵn: `../fasterrcnn/learning_curves.png`,
`../retinanet/learning_curves.png` + `history.json` của cả hai (val mAP bão hòa sau epoch 3).
Nhớ chèn 2 biểu đồ này vào báo cáo.

---

## 7. Checklist bản hướng dẫn ↔ script nào lo

| Mục hướng dẫn | Lo bởi |
|---|---|
| 1. Môi trường thực nghiệm | `common.collect_environment()` → `environment.json` |
| 2. Chuẩn hóa điều kiện so sánh | `config.py` + mục 2 của báo cáo |
| 3.1. Accuracy (mAP/P/R/F1/AP theo size/AR/TP-FP-FN/PR-curve/F1-conf) | `run_eval.py` + `make_report.py` |
| 3.2. System (latency p50/95/99, FPS, size, params, VRAM, RAM, load time) | `run_eval.py` |
| 3.3. Lát cắt (độ khó, mật độ, recall theo size bbox) | `run_slices.py` + `run_eval.py` |
| 4A. Định lượng toàn tập | `run_eval.py` |
| 4B. Phân tích lát cắt | `run_slices.py` |
| 4C. Định tính | `run_qualitative.py` |
| 4D. Phân tích lỗi | `run_error_analysis.py` |
| 4E. [→TN] Robustness | `run_robustness.py` |
| 4F. [→TN] CPU + INT8 | `run_cpu_bench.py` |
| 5. Ảnh test có chủ đích (6 tiêu chí) | `run_qualitative.py` |
| 8. Bảng 1/2/3 | `make_report.py` → `BAO_CAO_THUC_NGHIEM.md` |
| 9. Tiêu chí chốt model | mục 8 của báo cáo (tự sinh, dùng khung "ưu thế toàn diện") |
| 10. Sản phẩm đầu ra | `results/<tag>/` |
