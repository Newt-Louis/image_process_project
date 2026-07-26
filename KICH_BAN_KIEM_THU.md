# Kịch bản kiểm thử thủ công — Demo 3 model detector

> Dùng để chạy tay trên web demo và **điền số vào bảng** khi làm phần định tính của báo cáo
> (Kịch bản C — Mục 4C & 5 của `../HUONG_DAN_THUC_NGHIEM_3_MODEL_DETECTOR.md`).
> Số liệu định lượng toàn tập (Bảng 1/2/3) lấy từ `run_eval.py`, **không** điền ở đây.

---

## 0. Chuẩn bị

```bash
cd demo_app
source ../.venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Mở `http://localhost:8000`. Kiểm tra thanh badge phía trên: **Device = CUDA**,
**gold_dataset = 19.200 ảnh test**. Nếu Device = CPU thì latency sẽ sai lệch — xem README mục 2.

10 ảnh test đã đặt sẵn trong `data/uploads/`. Đây là ảnh **thật lấy từ tập test** (sản phẩm đặt
trên mặt bàn thanh toán RPC), dùng thay cho ảnh tự chụp.

---

## 1. Hai tham số suy luận — và chỉ hai

Sau khi model đã train xong, chỉ còn **2 tham số** làm đổi kết quả detection. Mọi thứ khác
(kích thước input, chuẩn hóa, kiến trúc) đã khóa cứng trong `config.py` để **so sánh 3 model
công bằng** — không được đổi giữa các model.

| Tham số | Ý nghĩa | Tăng lên thì… |
|---|---|---|
| **Confidence** (0.05–0.95) | ngưỡng điểm tin cậy tối thiểu để giữ 1 box | ít box hơn → **Precision ↑, Recall ↓** (bỏ sót nhiều hơn) |
| **NMS IoU** (0.30–0.90) | 2 box đè nhau bao nhiêu thì bị gộp làm 1 | giữ nhiều box chồng nhau → dễ **double-detection / FP** |

**Mặc định dùng cho báo cáo: Confidence = 0.25, NMS IoU = 0.50** (giống hệt điều kiện của
`run_eval.py`). Chỉ đổi khi làm thí nghiệm "độ nhạy tham số" ở mục 4.

---

## 2. Quy trình chạy mỗi ảnh

1. Vào tab **"Ảnh thật (upload)"**.
2. Bật **cả 3 chip model** (YOLOv11 + Faster R-CNN + RetinaNet).
3. Để **Confidence = 0.25**, **NMS IoU = 0.50**.
4. Kéo–thả 1 ảnh từ `data/uploads/` (hoặc bấm để chọn) → **"Chạy nhận diện"**.
5. Đọc 3 card kết quả cạnh nhau, ghi vào **Bảng A** bên dưới: số box + latency mỗi model,
   và nhận xét bằng mắt (model nào vẽ thừa box, model nào gọn nhất).

> ⚠️ Ảnh upload **không có ground-truth**, nên chỉ đếm box + nhìn bbox được, **không có
> TP/FP/FN**. Muốn có TP/FP/FN/Precision/Recall chính xác cho đúng ảnh đó → xem mục 3.

---

## 3. Cách lấy TP / FP / FN cho từng ảnh (tab "Tập test")

Mỗi ảnh trong `data/uploads/` chính là một ảnh của tập test, có `image_id` tra sẵn ở cột cuối
Bảng A. Để lấy số có đối chiếu ground-truth:

1. Vào tab **"Tập test gold_dataset"**.
2. Gõ `image_id` vào ô **"Hoặc nhập image_id"** → **"Chạy"** (đã bật cả 3 model, conf 0.25).
3. Ảnh trên cùng là ground-truth (xám). Mỗi card model hiển thị **TP / FP / FN / Precision /
   Recall / F1** — box tô 🟢 TP · 🔴 FP · 🟡 FN. Ghi vào **Bảng B**.

---

## Bảng A — Định tính trên ảnh upload (conf=0.25, NMS IoU=0.50)

Số sản phẩm thật (cột "GT") đã ghi sẵn để đối chiếu nhanh với số box từng model.

| # | Ảnh | Độ khó | GT | YOLOv11 (số box / ms) | Faster R-CNN (số box / ms) | RetinaNet (số box / ms) | Nhận xét (model nào FP/bỏ sót) | image_id |
|---|---|---|---|---|---|---|---|---|
| 1 | test_01_easy…212 | easy | 6 |  /  |  /  |  /  |  | 2989 |
| 2 | test_02_easy…1457 | easy | 6 |  /  |  /  |  /  |  | 2377 |
| 3 | test_03_medium…2830 | medium | 14 |  /  |  /  |  /  |  | 18010 |
| 4 | test_04_medium…2988 | medium | 12 |  /  |  /  |  /  |  | 14112 |
| 5 | test_05_hard…135 | hard | 19 |  /  |  /  |  /  |  | 28885 |
| 6 | test_06_hard…2088 | hard | 17 |  /  |  /  |  /  |  | 24332 |
| 7 | test_07_hard…1584 | hard | 18 |  /  |  /  |  /  |  | 26049 |
| 8 | test_08_dense…117 | hard | 19 |  /  |  /  |  /  |  | 28138 |
| 9 | test_09_dense…1549 | hard | 18 |  /  |  /  |  /  |  | 26732 |
| 10 | test_10_dense…619 | hard | 20 |  /  |  /  |  /  |  | 29656 |

---

## Bảng B — Định lượng có ground-truth (tab "Tập test", conf=0.25, IoU ghép=0.50)

Điền cho **3–6 ảnh chọn lọc** là đủ cho phần định tính (không cần cả 10).
Gợi ý chọn: 1 easy + 1 medium + 2–3 hard/dense để phân hóa model rõ nhất.

| image_id | Độ khó | GT | YOLOv11 TP/FP/FN · F1 | Faster R-CNN TP/FP/FN · F1 | RetinaNet TP/FP/FN · F1 |
|---|---|---|---|---|---|
| 2989 | easy | 6 |  /  /  ·  |  /  /  ·  |  /  /  ·  |
| 18010 | medium | 14 |  /  /  ·  |  /  /  ·  |  /  /  ·  |
| 28885 | hard | 19 |  /  /  ·  |  /  /  ·  |  /  /  ·  |
| 26049 | hard | 18 |  /  /  ·  |  /  /  ·  |  /  /  ·  |
| 29656 | hard | 20 |  /  /  ·  |  /  /  ·  |  /  /  ·  |
|  |  |  |  /  /  ·  |  /  /  ·  |  /  /  ·  |

*Kỳ vọng (từ lần chạy thử): YOLOv11 gần như TP đủ, FP≈0; Faster R-CNN có vài FP; RetinaNet
FP nhiều nhất, nhất là ảnh hard. Đây là bằng chứng trực quan cho kết luận chọn YOLOv11.*

---

## 4. (Tuỳ chọn) Thí nghiệm độ nhạy tham số

Chọn **1 ảnh hard** (vd image_id 29656) rồi chạy lại với các mức khác nhau, quan sát box thừa
của RetinaNet biến mất khi tăng confidence → minh họa vai trò của ngưỡng:

| Confidence | NMS IoU | YOLOv11 (box) | Faster R-CNN (box) | RetinaNet (box) | Ghi chú |
|---|---|---|---|---|---|
| 0.25 | 0.50 |  |  |  | mặc định |
| 0.50 | 0.50 |  |  |  | siết conf → bớt FP |
| 0.25 | 0.70 |  |  |  | nới NMS → dễ double-detection |

---

## 5. Kết luận rút ra (điền sau khi chạy xong)

- Model gọn nhất, ít FP nhất: ………………………………………………
- Model FP nhiều nhất trên ảnh hard: ……………………………………
- Ảnh 3 model lệch nhau rõ nhất (dùng cho báo cáo): ……………………
- Nhận xét chung khớp với Bảng 1/2 của `run_eval.py` không? ……………

> Ảnh so sánh 4-panel (GT | 3 model, có sẵn chú thích TP/FP/FN) do `run_qualitative.py` tự sinh
> nằm ở `results/main/qualitative/` — dùng luôn cho báo cáo, khỏi chụp màn hình thủ công.
