# Hệ Thống Demo Nhận Diện Vật Thể (Auto Checkout Demo)

Đây là mã nguồn hệ thống demo nhận diện vật thể dựa trên FastAPI và giao diện HTML tĩnh thuần túy (không sử dụng framework frontend), phục vụ cho đồ án môn học "Xử lý ảnh và ứng dụng", cũng như tiền đề cho đồ án tốt nghiệp.

## Cấu Trúc Thư Mục

```
image_process_project/
├── main.py              # File chứa API server FastAPI
├── requirements.txt     # Các thư viện cần cài đặt
└── static/              # Chứa các file giao diện tĩnh
    ├── index.html       # Cấu trúc giao diện
    ├── style.css        # Style (giao diện dark-mode, glassmorphism hiện đại)
    └── script.js        # Logic xử lý upload và gọi API
```

## Cài Đặt Môi Trường

Khuyến nghị sử dụng môi trường ảo (virtual environment) của Python:

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## Chạy Ứng Dụng

Sau khi cài đặt thư viện, bạn có thể chạy server với lệnh:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sau đó, truy cập vào đường dẫn: `http://localhost:8000` trên trình duyệt.

## Hướng Dẫn Tích Hợp Model (Fine-tuned)

Hiện tại, hệ thống đang dùng code mô phỏng (giả lập việc cắt vật thể ở giữa màn hình) để bạn có thể chạy thử UI ngay lập tức.
Khi bạn đã hoàn thiện fine-tuning model (ví dụ YOLO) trên Google Colab, hãy tải file weights (ví dụ: `best.pt`) về máy, bỏ vào thư mục `demo_app/models/` (bạn tự tạo thư mục models này) và mở file `main.py` để chỉnh sửa.

Tìm khối code sau trong `main.py`:

```python
# ==========================================
# KHU VỰC LOAD MODEL
# ==========================================
# model = YOLO("models/best.pt")
```
Bỏ comment ở dòng tải model.

Sau đó tìm đến:
```python
# ==========================================
# KHU VỰC INFERENCE MODEL
# ==========================================
```
Bỏ comment đoạn inference với YOLO, và thay thế đoạn code giả lập bên dưới nó bằng logic lấy bounding boxes thực từ model.
